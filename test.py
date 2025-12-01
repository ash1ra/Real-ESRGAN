from typing import Literal

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

import config
from data_processing import SRDataset
from models import Generator, TruncatedVGG19
from utils import convert_img, load_checkpoint

logger = config.create_logger("INFO", __file__)


def test_step(
    data_loader: DataLoader,
    generator: nn.Module,
    truncated_vgg19: nn.Module,
    perceptual_loss_fn: nn.Module,
    psnr_metric: PeakSignalNoiseRatio,
    ssim_metric: StructuralSimilarityIndexMeasure,
    device: Literal["cuda", "cpu"] = "cpu",
) -> tuple[float, float, float]:
    total_perceptual_loss = 0.0

    generator.eval()

    with torch.inference_mode():
        for hr_img_tensor, lr_img_tensor in data_loader:
            hr_img_tensor = hr_img_tensor.to(device, non_blocking=True)
            lr_img_tensor = lr_img_tensor.to(device, non_blocking=True)

            sr_img_tensor = generator(lr_img_tensor)
            sr_img_tensor_normalized = convert_img(sr_img_tensor, "[-1, 1]", "imagenet")
            hr_img_tensor_normalized = convert_img(hr_img_tensor, "[-1, 1]", "imagenet")

            with torch.no_grad():
                sr_img_tensor_in_vgg_space = truncated_vgg19(sr_img_tensor_normalized)
                hr_img_tensor_in_vgg_space = truncated_vgg19(hr_img_tensor_normalized)

            perceptual_loss = torch.tensor(0.0, device=device)
            for layer_name, weight in config.PERCEPTUAL_LOSS_LAYER_WEIGHTS.items():
                layer_loss = perceptual_loss_fn(
                    sr_img_tensor_in_vgg_space[layer_name],
                    hr_img_tensor_in_vgg_space[layer_name],
                )
                perceptual_loss += weight * layer_loss

            y_hr_tensor = convert_img(hr_img_tensor, "[-1, 1]", "y-channel")
            y_sr_tensor = convert_img(sr_img_tensor, "[-1, 1]", "y-channel")

            sf = config.SCALING_FACTOR
            y_hr_tensor = y_hr_tensor[:, :, sf:-sf, sf:-sf]
            y_sr_tensor = y_sr_tensor[:, :, sf:-sf, sf:-sf]

            psnr_metric.update(y_sr_tensor, y_hr_tensor)  # type: ignore
            ssim_metric.update(y_sr_tensor, y_hr_tensor)  # type: ignore

            total_perceptual_loss += perceptual_loss.item()

        total_perceptual_loss /= len(data_loader)
        total_psnr = psnr_metric.compute().item()  # type: ignore
        total_ssim = ssim_metric.compute().item()  # type: ignore

        psnr_metric.reset()
        ssim_metric.reset()

    return total_perceptual_loss, total_psnr, total_ssim


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = Generator(
        channels_count=config.GENERATOR_CHANNELS_COUNT,
        growth_channels_count=config.GENERATOR_GROWTHS_CHANNELS_COUNT,
        large_kernel_size=config.GENERATOR_LARGE_KERNEL_SIZE,
        small_kernel_size=config.GENERATOR_SMALL_KERNEL_SIZE,
        res_dense_blocks_count=config.GENERATOR_RES_DENSE_BLOCKS_COUNT,
        rrdb_count=config.GENERATOR_RRDB_COUNT,
        scaling_factor=config.SCALING_FACTOR,
    ).to(device)

    truncated_vgg19 = TruncatedVGG19().to(device)

    perceptual_loss_fn = nn.L1Loss()

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    _ = load_checkpoint(
        checkpoint_dir_path=config.BEST_REAL_ESRGAN_CHECKPOINT_DIR_PATH,
        generator=generator,
        test_mode=True,
    )

    for dataset_path in config.TEST_DATASETS_PATHS:
        dataset = SRDataset(
            data_path=dataset_path,
            scaling_factor=config.SCALING_FACTOR,
            crop_size=config.CROP_SIZE,
            test_mode=True,
        )

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=config.TEST_BATCH_SIZE,
            shuffle=False,
            pin_memory=True if device == "cuda" else False,
            num_workers=config.NUM_WORKERS,
        )

        test_loss, test_psnr, test_ssim = test_step(
            data_loader=data_loader,
            generator=generator,
            truncated_vgg19=truncated_vgg19,
            perceptual_loss_fn=perceptual_loss_fn,
            psnr_metric=psnr_metric,
            ssim_metric=ssim_metric,
            device=device,
        )

        logger.info(
            f"{dataset_path.name} Dataset | Test loss: {test_loss:.4f} | PSNR: {test_psnr:.4f} | SSIM: {test_ssim:.4f}"
        )


if __name__ == "__main__":
    main()
