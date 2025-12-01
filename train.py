import gc
from time import time
from typing import Literal

import torch
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

import config
from data_processing import SRDataset
from models import Discriminator, Generator, TruncatedVGG19
from utils import (
    EMAModel,
    Metrics,
    convert_img,
    create_hyperparameters_str,
    format_time,
    load_checkpoint,
    plot_training_metrics,
    save_checkpoint,
    upscale_img_tiled,
    worker_init_fn,
)

logger = config.create_logger("INFO", __file__)


def train_step(
    data_loader: DataLoader,
    generator: nn.Module,
    discriminator: nn.Module,
    truncated_vgg19: nn.Module,
    ema_handler: EMAModel,
    perceptual_loss_fn: nn.Module,
    pixel_loss_fn: nn.Module,
    adversarial_loss_fn: nn.Module,
    generator_optimizer: optim.Optimizer,
    discriminator_optimizer: optim.Optimizer,
    generator_scaler: GradScaler | None = None,
    discriminator_scaler: GradScaler | None = None,
    device: Literal["cuda", "cpu"] = "cpu",
) -> tuple[float, float]:
    total_generator_loss = 0.0
    total_discriminator_loss = 0.0

    generator.train()
    discriminator.train()

    for i, (hr_img_tensor, lr_img_tensor) in enumerate(data_loader):
        hr_img_tensor = hr_img_tensor.to(device, non_blocking=True)
        lr_img_tensor = lr_img_tensor.to(device, non_blocking=True)

        with autocast(device, dtype=torch.bfloat16, enabled=True):
            sr_img_tensor = generator(lr_img_tensor)

            sr_img_tensor = torch.clamp(sr_img_tensor, -1.0, 1.0)

            sr_img_tensor_imagenet = convert_img(sr_img_tensor, "[-1, 1]", "imagenet")
            hr_img_tensor_imagenet = convert_img(hr_img_tensor, "[-1, 1]", "imagenet")

            sr_img_features_tensor = truncated_vgg19(sr_img_tensor_imagenet)
            with torch.no_grad():
                hr_img_features_tensor = truncated_vgg19(hr_img_tensor_imagenet)

            perceptual_loss = 0.0

            for layer_name, weight in config.PERCEPTUAL_LOSS_LAYER_WEIGHTS.items():
                sr_img_feature_tensor = sr_img_features_tensor[layer_name]
                hr_img_feature_tensor = hr_img_features_tensor[layer_name]

                layer_loss = perceptual_loss_fn(
                    sr_img_feature_tensor, hr_img_feature_tensor
                )
                perceptual_loss += weight * layer_loss

            with torch.no_grad():
                hr_discriminated = discriminator(hr_img_tensor)
            sr_discriminated = discriminator(sr_img_tensor)

            hr_discriminated_avg = hr_discriminated.mean(dim=0, keepdim=True)
            sr_discriminated_avg = sr_discriminated.mean(dim=0, keepdim=True)

            hr_discriminated_relative = hr_discriminated - sr_discriminated_avg
            sr_discriminated_relative = sr_discriminated - hr_discriminated_avg

            adversarial_loss = (
                adversarial_loss_fn(
                    sr_discriminated_relative,
                    torch.ones_like(sr_discriminated_relative),
                )
                + adversarial_loss_fn(
                    hr_discriminated_relative,
                    torch.zeros_like(hr_discriminated_relative),
                )
            ) / 2

            pixel_loss = pixel_loss_fn(sr_img_tensor, hr_img_tensor)

            generator_loss = (
                config.PERCEPTUAL_LOSS_SCALING_VALUE * perceptual_loss
                + config.ADVERSARIAL_LOSS_SCALING_VALUE * adversarial_loss
                + config.PIXEL_LOSS_SCALING_VALUE * pixel_loss
            )

            generator_loss_w_graph = generator_loss / config.GRADIENT_ACCUMULATION_STEPS

        total_generator_loss += generator_loss.item()

        if generator_scaler:
            generator_scaler.scale(generator_loss_w_graph).backward()
        else:
            generator_loss_w_graph.backward()

        # with autocast(device, dtype=torch.bfloat16, enabled=False):
        hr_discriminated = discriminator(hr_img_tensor.float())
        sr_discriminated = discriminator(sr_img_tensor.float().detach())

        hr_discriminated_avg = hr_discriminated.mean(dim=0, keepdim=True)
        sr_discriminated_avg = sr_discriminated.mean(dim=0, keepdim=True)

        hr_discriminated_relative = hr_discriminated - sr_discriminated_avg.detach()
        sr_discriminated_relative = sr_discriminated - hr_discriminated_avg.detach()

        adversarial_loss = (
            adversarial_loss_fn(
                sr_discriminated_relative,
                torch.zeros_like(sr_discriminated_relative),
            )
            + adversarial_loss_fn(
                hr_discriminated_relative,
                torch.ones_like(hr_discriminated_relative),
            )
        ) / 2

        adversarial_loss_w_graph = adversarial_loss / config.GRADIENT_ACCUMULATION_STEPS

        total_discriminator_loss += adversarial_loss.item()

        if discriminator_scaler:
            discriminator_scaler.scale(adversarial_loss_w_graph).backward()
        else:
            adversarial_loss_w_graph.backward()

        if (i + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0 or (i + 1) == len(
            data_loader
        ):
            if generator_scaler:
                generator_scaler.unscale_(generator_optimizer)
                clip_grad_norm_(
                    generator.parameters(), max_norm=config.GRADIENT_CLIPPING_VALUE
                )
                generator_scaler.step(generator_optimizer)
                generator_scaler.update()
            else:
                clip_grad_norm_(
                    generator.parameters(), max_norm=config.GRADIENT_CLIPPING_VALUE
                )
                generator_optimizer.step()

            if discriminator_scaler:
                discriminator_scaler.unscale_(discriminator_optimizer)
                clip_grad_norm_(
                    discriminator.parameters(), max_norm=config.GRADIENT_CLIPPING_VALUE
                )
                discriminator_scaler.step(discriminator_optimizer)
                discriminator_scaler.update()
            else:
                clip_grad_norm_(
                    discriminator.parameters(), max_norm=config.GRADIENT_CLIPPING_VALUE
                )
                discriminator_optimizer.step()

            ema_handler.update()

            generator_optimizer.zero_grad()
            discriminator_optimizer.zero_grad()

        if i % config.PRINT_FREQUENCY == 0:
            logger.debug(f"Processing batch {i}/{len(data_loader)}...")

    total_generator_loss /= len(data_loader)
    total_discriminator_loss /= len(data_loader)

    return total_generator_loss, total_discriminator_loss


def validation_step(
    data_loader: DataLoader,
    generator: nn.Module,
    truncated_vgg19: nn.Module,
    perceptual_loss_fn: nn.Module,
    psnr_metric: PeakSignalNoiseRatio,
    ssim_metric: StructuralSimilarityIndexMeasure,
    device: Literal["cpu", "cuda"] = "cpu",
) -> tuple[float, float, float]:
    total_perceptual_loss = 0.0

    generator.eval()

    with torch.inference_mode():
        for hr_img_tensor, lr_img_tensor in data_loader:
            sr_img_full = upscale_img_tiled(
                model=generator,
                lr_img_tensor=lr_img_tensor,
                scale_factor=config.SCALING_FACTOR,
                tile_size=config.TILE_SIZE,
                tile_overlap=config.TILE_OVERLAP,
                device=device,
            )

            hr_img_cpu = hr_img_tensor.cpu()

            y_hr_tensor = convert_img(hr_img_cpu, "[-1, 1]", "y-channel")
            y_sr_tensor = convert_img(sr_img_full, "[-1, 1]", "y-channel")

            sf = config.SCALING_FACTOR
            y_hr_tensor = y_hr_tensor[:, :, sf:-sf, sf:-sf]
            y_sr_tensor = y_sr_tensor[:, :, sf:-sf, sf:-sf]

            psnr_metric.update(y_sr_tensor, y_hr_tensor)  # type: ignore
            ssim_metric.update(y_sr_tensor, y_hr_tensor)  # type: ignore

            _, _, h_sr, w_sr = sr_img_full.shape

            if h_sr > config.CROP_SIZE and w_sr > config.CROP_SIZE:
                top = (h_sr - config.CROP_SIZE) // 2
                left = (w_sr - config.CROP_SIZE) // 2

                sr_crop = sr_img_full[
                    :, :, top : top + config.CROP_SIZE, left : left + config.CROP_SIZE
                ]

                hr_crop = hr_img_cpu[
                    :, :, top : top + config.CROP_SIZE, left : left + config.CROP_SIZE
                ]

                sr_crop = sr_crop.to(device)
                hr_crop = hr_crop.to(device)
            else:
                sr_crop = sr_img_full.to(device)
                hr_crop = hr_img_cpu.to(device)

            sr_crop_imagenet = convert_img(sr_crop, "[-1, 1]", "imagenet")
            hr_crop_imagenet = convert_img(hr_crop, "[-1, 1]", "imagenet")

            with torch.no_grad():
                sr_features = truncated_vgg19(sr_crop_imagenet)
                hr_features = truncated_vgg19(hr_crop_imagenet)

            perceptual_loss = torch.tensor(0.0, device=device)
            for layer_name, weight in config.PERCEPTUAL_LOSS_LAYER_WEIGHTS.items():
                layer_loss = perceptual_loss_fn(
                    sr_features[layer_name], hr_features[layer_name]
                )
                perceptual_loss += weight * layer_loss

            total_perceptual_loss += perceptual_loss.item()

        total_perceptual_loss /= len(data_loader)
        total_psnr = psnr_metric.compute().item()  # type: ignore
        total_ssim = ssim_metric.compute().item()  # type: ignore

        psnr_metric.reset()
        ssim_metric.reset()

    return total_perceptual_loss, total_psnr, total_ssim


def train(
    train_data_loader: DataLoader,
    val_data_loader: DataLoader,
    generator: nn.Module,
    discriminator: nn.Module,
    truncated_vgg19: nn.Module,
    ema_handler: EMAModel,
    perceptual_loss_fn: nn.Module,
    pixel_loss_fn: nn.Module,
    adversarial_loss_fn: nn.Module,
    generator_optimizer: optim.Optimizer,
    discriminator_optimizer: optim.Optimizer,
    start_epoch: int,
    epochs: int,
    metrics: Metrics,
    psnr_metric: PeakSignalNoiseRatio,
    ssim_metric: StructuralSimilarityIndexMeasure,
    generator_scaler: GradScaler | None = None,
    discriminator_scaler: GradScaler | None = None,
    generator_scheduler: MultiStepLR | None = None,
    discriminator_scheduler: MultiStepLR | None = None,
    device: Literal["cuda", "cpu"] = "cpu",
) -> None:
    if not metrics.epochs:
        metrics.epochs = epochs - start_epoch + 1

    if start_epoch > 1 and metrics.generator_val_losses:
        best_val_loss = min(metrics.generator_val_losses)
    else:
        best_val_loss = float("inf")

    dashes_count = 54

    logger.info("-" * dashes_count)
    logger.info("Model parameters:")
    logger.info(f"Scaling factor: {config.SCALING_FACTOR}")
    logger.info(f"Crop size: {config.CROP_SIZE}")
    logger.info(f"Batch size: {config.TRAIN_BATCH_SIZE}")
    logger.info(f"Epochs: {config.EPOCHS}")
    logger.info(f"Number of workers: {config.NUM_WORKERS}")
    logger.info("-" * dashes_count)
    logger.info("Real-ESRGAN Generator:")
    logger.info(f"Count of channels: {config.GENERATOR_CHANNELS_COUNT}")
    logger.info(f"Count of growths channels: {config.GENERATOR_GROWTHS_CHANNELS_COUNT}")
    logger.info(
        f"Count of residual dense blocks: {config.GENERATOR_RES_DENSE_BLOCKS_COUNT}"
    )
    logger.info(
        f"Count of residual in residual dense blocks: {config.GENERATOR_RRDB_COUNT}"
    )
    logger.info(f"Large kernel size: {config.GENERATOR_LARGE_KERNEL_SIZE}")
    logger.info(f"Small kernel size: {config.GENERATOR_SMALL_KERNEL_SIZE}")
    logger.info(f"Generator initial learning rate: {config.GENERATOR_LEARNING_RATE}")
    logger.info("-" * dashes_count)
    logger.info("U-Net Discriminator:")
    logger.info(f"Count of channels: {config.DISCRIMINATOR_CHANNELS_COUNT}")
    logger.info(
        f"Discriminator initial learning rate: {config.DISCRIMINATOR_LEARNING_RATE}"
    )
    logger.info("-" * dashes_count)
    logger.info("Starting model training...")

    try:
        training_start_time = time()
        for epoch in range(start_epoch, epochs + 1):
            epoch_start_time = time()

            generator_train_loss, discriminator_train_loss = train_step(
                data_loader=train_data_loader,
                generator=generator,
                discriminator=discriminator,
                truncated_vgg19=truncated_vgg19,
                ema_handler=ema_handler,
                perceptual_loss_fn=perceptual_loss_fn,
                pixel_loss_fn=pixel_loss_fn,
                adversarial_loss_fn=adversarial_loss_fn,
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                generator_scaler=generator_scaler,
                discriminator_scaler=discriminator_scaler,
                device=device,
            )

            gc.collect()
            torch.cuda.empty_cache()

            generator_val_loss, generator_val_psnr, generator_val_ssim = (
                validation_step(
                    data_loader=val_data_loader,
                    generator=ema_handler.target_model,
                    truncated_vgg19=truncated_vgg19,
                    perceptual_loss_fn=perceptual_loss_fn,
                    psnr_metric=psnr_metric,
                    ssim_metric=ssim_metric,
                    device=device,
                )
            )

            gc.collect()
            torch.cuda.empty_cache()

            if generator_scheduler:
                generator_scheduler.step()

            if discriminator_scheduler:
                discriminator_scheduler.step()

            end_time = time() - epoch_start_time
            epoch_time = format_time(end_time)
            elapsed_time = format_time(time() - training_start_time)
            remaining_time = format_time(end_time * (epochs - epoch))

            generator_optimizer_lr = generator_optimizer.param_groups[0]["lr"]
            discriminator_optimizer_lr = discriminator_optimizer.param_groups[0]["lr"]

            metrics.generator_learning_rates.append(generator_optimizer_lr)
            metrics.discriminator_learning_rates.append(discriminator_optimizer_lr)
            metrics.generator_train_losses.append(generator_train_loss)
            metrics.discriminator_train_losses.append(discriminator_train_loss)
            metrics.generator_val_losses.append(generator_val_loss)
            metrics.generator_val_psnrs.append(generator_val_psnr)
            metrics.generator_val_ssims.append(generator_val_ssim)

            logger.info(
                f"Epoch: {epoch}/{epochs} ({epoch_time} | {elapsed_time}/{remaining_time}) | Generator LR: {generator_optimizer_lr} | Discriminator LR: {discriminator_optimizer_lr}"
            )
            logger.info(
                f"Generator Train Loss: {generator_train_loss:.4f} | Discriminator Train Loss: {discriminator_train_loss:.4f} | Generator Val Loss: {generator_val_loss:.4f} | Generator Val PSNR: {generator_val_psnr:.4f} | Generator Val SSIM: {generator_val_ssim:.4f}"
            )

            if generator_val_loss < best_val_loss:
                best_val_loss = generator_val_loss
                logger.debug(
                    f"New best model found with val loss: {best_val_loss:.4f} at epoch {epoch}"
                )
                save_checkpoint(
                    checkpoint_dir_path=config.BEST_REAL_ESRGAN_CHECKPOINT_DIR_PATH,
                    epoch=epoch,
                    generator=ema_handler.target_model,
                    discriminator=discriminator,
                    generator_optimizer=generator_optimizer,
                    discriminator_optimizer=discriminator_optimizer,
                    metrics=metrics,
                    generator_scaler=generator_scaler,
                    discriminator_scaler=discriminator_scaler,
                    generator_scheduler=generator_scheduler,
                    discriminator_scheduler=discriminator_scheduler,
                )

            save_checkpoint(
                checkpoint_dir_path=config.REAL_ESRGAN_CHECKPOINT_DIR_PATH,
                epoch=epoch,
                generator=generator,
                discriminator=discriminator,
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                metrics=metrics,
                generator_scaler=generator_scaler,
                discriminator_scaler=discriminator_scaler,
                generator_scheduler=generator_scheduler,
                discriminator_scheduler=discriminator_scheduler,
            )

        plot_training_metrics(
            metrics, create_hyperparameters_str(), model_type="real-esrgan"
        )

    except KeyboardInterrupt:
        logger.info("Saving model's weights and finish training...")
        save_checkpoint(
            checkpoint_dir_path=config.REAL_ESRGAN_CHECKPOINT_DIR_PATH,
            epoch=epoch,
            generator=generator,
            discriminator=discriminator,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            metrics=metrics,
            generator_scaler=generator_scaler,
            discriminator_scaler=discriminator_scaler,
            generator_scheduler=generator_scheduler,
            discriminator_scheduler=discriminator_scheduler,
        )

        plot_training_metrics(
            metrics, create_hyperparameters_str(), model_type="real-esrgan"
        )


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = SRDataset(
        data_path=config.TRAIN_DATASET_PATH,
        scaling_factor=config.SCALING_FACTOR,
        crop_size=config.CROP_SIZE,
        dev_mode=config.DEV_MODE,
    )

    val_dataset = SRDataset(
        data_path=config.VAL_DATASET_PATH,
        scaling_factor=config.SCALING_FACTOR,
        crop_size=config.CROP_SIZE,
        test_mode=True,
        dev_mode=config.DEV_MODE,
    )

    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.REAL_TRAIN_BATCH_SIZE,
        shuffle=True,
        pin_memory=True if device == "cuda" else False,
        num_workers=config.NUM_WORKERS,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=worker_init_fn,
    )

    val_data_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST_BATCH_SIZE,
        shuffle=False,
        pin_memory=True if device == "cuda" else False,
        num_workers=config.NUM_WORKERS,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=worker_init_fn,
    )

    generator = Generator(
        channels_count=config.GENERATOR_CHANNELS_COUNT,
        growth_channels_count=config.GENERATOR_GROWTHS_CHANNELS_COUNT,
        large_kernel_size=config.GENERATOR_LARGE_KERNEL_SIZE,
        small_kernel_size=config.GENERATOR_SMALL_KERNEL_SIZE,
        res_dense_blocks_count=config.GENERATOR_RES_DENSE_BLOCKS_COUNT,
        rrdb_count=config.GENERATOR_RRDB_COUNT,
        scaling_factor=config.SCALING_FACTOR,
    ).to(device)

    ema_generator = Generator(
        channels_count=config.GENERATOR_CHANNELS_COUNT,
        growth_channels_count=config.GENERATOR_GROWTHS_CHANNELS_COUNT,
        large_kernel_size=config.GENERATOR_LARGE_KERNEL_SIZE,
        small_kernel_size=config.GENERATOR_SMALL_KERNEL_SIZE,
        res_dense_blocks_count=config.GENERATOR_RES_DENSE_BLOCKS_COUNT,
        rrdb_count=config.GENERATOR_RRDB_COUNT,
        scaling_factor=config.SCALING_FACTOR,
    ).to(device)

    discriminator = Discriminator(
        in_channels=3, channels_count=config.DISCRIMINATOR_CHANNELS_COUNT
    ).to(device)

    truncated_vgg19 = TruncatedVGG19().to(device)

    perceptual_loss_fn = nn.L1Loss()
    pixel_loss_fn = nn.L1Loss()
    adversarial_loss_fn = nn.BCEWithLogitsLoss()

    metrics = Metrics()
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

    generator_optimizer = optim.Adam(
        generator.parameters(), lr=config.GENERATOR_LEARNING_RATE
    )
    discriminator_optimizer = optim.Adam(
        discriminator.parameters(), lr=config.DISCRIMINATOR_LEARNING_RATE
    )

    generator_scheduler = MultiStepLR(
        optimizer=generator_optimizer,
        milestones=config.SCHEDULER_MILESTONES,
        gamma=config.SCHEDULER_SCALING_VALUE,
    )
    discriminator_scheduler = MultiStepLR(
        optimizer=discriminator_optimizer,
        milestones=config.SCHEDULER_MILESTONES,
        gamma=config.SCHEDULER_SCALING_VALUE,
    )

    start_epoch = 1

    if config.INITIALIZE_WITH_REAL_ESRNET_CHECKPOINT:
        if config.BEST_REAL_ESRNET_CHECKPOINT_DIR_PATH.exists():
            checkpoint_dir_path_to_load = config.BEST_REAL_ESRNET_CHECKPOINT_DIR_PATH

            logger.info(
                f'Initializing Generator with the best Real-ESRNET weights from "{checkpoint_dir_path_to_load}"...'
            )

            _ = load_checkpoint(
                checkpoint_dir_path=checkpoint_dir_path_to_load,
                generator=generator,
                test_mode=True,
                device=device,
            )
        else:
            logger.warning(
                "Real-ESRNET checkpoint not found, start training from the beginning..."
            )

    if config.LOAD_REAL_ESRGAN_CHECKPOINT:
        if (
            config.BEST_REAL_ESRGAN_CHECKPOINT_DIR_PATH.exists()
            or config.REAL_ESRGAN_CHECKPOINT_DIR_PATH.exists()
        ):
            if (
                config.LOAD_BEST_REAL_ESRGAN_CHECKPOINT
                and config.BEST_REAL_ESRGAN_CHECKPOINT_DIR_PATH.exists()
            ):
                checkpoint_dir_path_to_load = (
                    config.BEST_REAL_ESRGAN_CHECKPOINT_DIR_PATH
                )
                logger.info(
                    f'Loading best ESRGAN checkpoint from "{checkpoint_dir_path_to_load}"...'
                )
            elif config.REAL_ESRGAN_CHECKPOINT_DIR_PATH.exists():
                checkpoint_dir_path_to_load = config.REAL_ESRGAN_CHECKPOINT_DIR_PATH
                logger.info(
                    f'Loading ESRGAN checkpoint from "{checkpoint_dir_path_to_load}"...'
                )

            start_epoch = load_checkpoint(
                checkpoint_dir_path=checkpoint_dir_path_to_load,
                generator=generator,
                discriminator=discriminator,
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                metrics=metrics,
                generator_scheduler=generator_scheduler,
                discriminator_scheduler=discriminator_scheduler,
                device=device,
            )

            for param_group in generator_optimizer.param_groups:
                param_group["lr"] = config.GENERATOR_LEARNING_RATE
            if generator_scheduler:
                generator_scheduler.base_lrs = [config.GENERATOR_LEARNING_RATE] * len(
                    generator_optimizer.param_groups
                )

            for param_group in discriminator_optimizer.param_groups:
                param_group["lr"] = config.DISCRIMINATOR_LEARNING_RATE
            if discriminator_scheduler:
                discriminator_scheduler.base_lrs = [
                    config.DISCRIMINATOR_LEARNING_RATE
                ] * len(discriminator_optimizer.param_groups)

            metrics.epochs = config.EPOCHS

            # if generator_scheduler and start_epoch > 1:
            #     logger.info(f"Fast-forwarding schedulers to epoch {start_epoch}...")
            #     for _ in range(start_epoch - 1):
            #         generator_scheduler.step()
            #         if discriminator_scheduler:
            #             discriminator_scheduler.step()
        else:
            logger.warning(
                "ESRGAN checkpoints not found, start training from the beginning..."
            )

    ema_handler = EMAModel(
        source_model=generator, target_model=ema_generator, decay=0.999
    )

    logger.info("Compiling models...")
    generator.compile()

    train(
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        generator=generator,
        discriminator=discriminator,
        truncated_vgg19=truncated_vgg19,
        ema_handler=ema_handler,
        perceptual_loss_fn=perceptual_loss_fn,
        pixel_loss_fn=pixel_loss_fn,
        adversarial_loss_fn=adversarial_loss_fn,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        start_epoch=start_epoch,
        epochs=config.EPOCHS,
        metrics=metrics,
        psnr_metric=psnr_metric,
        ssim_metric=ssim_metric,
        generator_scheduler=generator_scheduler,
        discriminator_scheduler=discriminator_scheduler,
        device=device,
    )


if __name__ == "__main__":
    main()
