import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, overload

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file, save_file
from scipy import special
from torch import Tensor, nn, optim
from torch.amp import GradScaler
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.io import decode_image
from torchvision.transforms import v2 as transforms

import config

logger = config.create_logger("INFO", __file__)


@dataclass
class Metrics:
    epochs: int = field(default=0)
    generator_learning_rates: list[float] = field(default_factory=list)
    discriminator_learning_rates: list[float] = field(default_factory=list)
    generator_train_losses: list[float] = field(default_factory=list)
    discriminator_train_losses: list[float] = field(default_factory=list)
    generator_val_losses: list[float] = field(default_factory=list)
    generator_val_psnrs: list[float] = field(default_factory=list)
    generator_val_ssims: list[float] = field(default_factory=list)


class EMAModel:
    def __init__(
        self, source_model: nn.Module, target_model: nn.Module, decay: float = 0.999
    ):
        self.source_model = source_model
        self.target_model = target_model
        self.decay = decay

        self.target_model.eval()
        for param in self.target_model.parameters():
            param.requires_grad = False

        self.target_model.load_state_dict(self.source_model.state_dict())

    def update(self):
        with torch.no_grad():
            for s_param, t_param in zip(
                self.source_model.parameters(), self.target_model.parameters()
            ):
                t_param.data.lerp_(s_param.data, 1.0 - self.decay)

            for s_buffer, t_buffer in zip(
                self.source_model.buffers(), self.target_model.buffers()
            ):
                t_buffer.data.copy_(s_buffer.data)


class ImgDegradationPipeline:
    def __init__(self, scaling_factor: int):
        self.scaling_factor = scaling_factor

        self.blur_kernel_size_list = config.BLUR_KERNEL_SIZE_LIST
        self.blur_kernel_prob = config.BLUR_KERNEL_PROBABILITY
        self.betag_range = config.BETAG_RANGE
        self.betap_range = config.BETAP_RANGE
        self.sinc_prob = config.SINC_PROBABILITY
        self.sinc_kernel_size = config.SINC_KERNEL_SIZE
        self.omega_range = config.OMEGA_RANGE
        self.second_blur_prob = config.SECOND_BLUR_PROBABILITY

        self.blur_sigma_range = config.BLUR_SIGMA_RANGE
        self.resize_prob = config.RESIZE_PROBABILITY
        self.resize_range = config.RESIZE_RANGE
        self.gaussian_noise_prob = config.GAUSSIAN_NOISE_PROBABILITY
        self.noise_range = config.NOISE_RANGE
        self.poisson_scale_range = config.POISSON_SCALE_RANGE
        self.gray_noise_prob = config.GRAY_NOISE_PROBABILITY
        self.jpeg_range = config.JPEG_RANGE

        self.blur_sigma_range_2 = config.BLUR_SIGMA_RANGE_2
        self.resize_prob_2 = config.RESIZE_PROBABILITY_2
        self.resize_range_2 = config.RESIZE_RANGE_2
        self.gaussian_noise_prob_2 = config.GAUSSIAN_NOISE_PROBABILITY_2
        self.noise_range_2 = config.NOISE_RANGE_2
        self.poisson_scale_range_2 = config.POISSON_SCALE_RANGE_2
        self.gray_noise_prob_2 = config.GRAY_NOISE_PROBABILITY_2
        self.jpeg_range_2 = config.JPEG_RANGE_2

    def _calc_rotated_sigma_matrix(self, sigma_x, sigma_y, angle):
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)

        R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

        Sigma = np.array([[sigma_x**2, 0], [0, sigma_y**2]])

        Cov = R @ Sigma @ R.T

        inv_Cov = np.linalg.inv(Cov)
        return inv_Cov

    def _mesh_grid(self, kernel_size):
        ax = np.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = np.meshgrid(ax, ax)
        xy = np.hstack(
            (
                xx.reshape((kernel_size * kernel_size, 1)),
                yy.reshape((kernel_size * kernel_size, 1)),
            )
        ).reshape(kernel_size, kernel_size, 2)
        return xy, xx, yy

    def _get_gaussian_kernel(self, kernel_size, sigma_x, sigma_y, angle, grid=None):
        if grid is None:
            _, xx, yy = self._mesh_grid(kernel_size)
        else:
            xx, yy = grid

        inv_cov = self._calc_rotated_sigma_matrix(sigma_x, sigma_y, angle)
        a = inv_cov[0, 0]
        b = inv_cov[0, 1]
        c = inv_cov[1, 1]

        exponent = -0.5 * (a * xx**2 + 2 * b * xx * yy + c * yy**2)
        kernel = np.exp(exponent)

        return kernel / np.sum(kernel)

    def _get_generalized_gaussian_kernel(self, kernel_size, sigma, beta, grid=None):
        if grid is None:
            _, xx, yy = self._mesh_grid(kernel_size)
        else:
            xx, yy = grid
        kernel = np.exp(
            -0.5 * (np.power(np.abs(xx) ** 2 + np.abs(yy) ** 2, beta / 2)) / sigma**2
        )
        return kernel / np.sum(kernel)

    def _get_plateau_kernel(self, kernel_size, sigma, beta, grid=None):
        if grid is None:
            _, xx, yy = self._mesh_grid(kernel_size)
        else:
            xx, yy = grid
        r = np.sqrt(xx**2 + yy**2)
        kernel = np.reciprocal(np.power(r + 1e-5, beta))
        return kernel / np.sum(kernel)

    def _get_sinc_kernel(self, kernel_size, omega_c):
        _, xx, yy = self._mesh_grid(kernel_size)
        dist = np.sqrt(xx**2 + yy**2)

        with np.errstate(divide="ignore", invalid="ignore"):
            kernel = 2 * special.j1(omega_c * dist) / (omega_c * dist)
            kernel[dist == 0] = 1.0

        window_1d = np.kaiser(kernel_size, 14)
        window_2d = np.outer(window_1d, window_1d)
        kernel = kernel * window_2d

        return kernel / np.sum(kernel)

    def _generate_random_blur_kernel(self, sigma_range):
        kernel_size = random.choice(self.blur_kernel_size_list)
        blur_type = random.choices(
            ["gaussian", "generalized", "plateau"], self.blur_kernel_prob
        )[0]

        _, xx, yy = self._mesh_grid(kernel_size)
        grid = (xx, yy)

        sigma_x = random.uniform(sigma_range[0], sigma_range[1])

        if random.random() < 0.5:
            sigma_y = sigma_x
            angle = 0
        else:
            sigma_y = random.uniform(sigma_range[0], sigma_range[1])
            angle = random.uniform(0, 2 * np.pi)

        if blur_type == "gaussian":
            kernel = self._get_gaussian_kernel(
                kernel_size, sigma_x, sigma_y, angle, grid
            )

        elif blur_type == "generalized":
            beta = random.uniform(self.betag_range[0], self.betag_range[1])

            kernel = self._get_generalized_gaussian_kernel(
                kernel_size,
                sigma_x,
                beta,
                grid,
            )

        else:
            beta = random.uniform(self.betap_range[0], self.betap_range[1])
            kernel = self._get_plateau_kernel(kernel_size, sigma_x, beta, grid)

        return kernel

    def _generate_sinc_kernel(self):
        kernel_size = self.sinc_kernel_size
        omega_c = random.uniform(self.omega_range[0], self.omega_range[1])
        kernel = self._get_sinc_kernel(kernel_size, omega_c)
        return kernel

    def _apply_resize(self, img, resize_prob, resize_range):
        h, w = img.shape[:2]
        resize_type = random.choices(["up", "down", "keep"], resize_prob)[0]

        if resize_type == "up":
            scale = random.uniform(1.0, resize_range[1])
        elif resize_type == "down":
            scale = random.uniform(resize_range[0], 1.0)
        else:
            scale = 1.0

        if scale == 1.0:
            return img

        interpolation = random.choice(
            [
                cv2.INTER_LINEAR,
                cv2.INTER_CUBIC,
                cv2.INTER_AREA,
            ]
        )

        return cv2.resize(
            img, (int(w * scale), int(h * scale)), interpolation=interpolation
        )

    def _apply_noise(
        self, img, gaussian_prob, noise_range, poisson_scale_range, gray_noise_prob
    ):
        h, w, c = img.shape

        use_gray_noise = random.random() < gray_noise_prob

        if random.random() < gaussian_prob:
            sigma = random.uniform(noise_range[0], noise_range[1])
            if use_gray_noise:
                noise = np.random.normal(0, sigma, (h, w, 1))
                noise = np.repeat(noise, c, axis=2)
            else:
                noise = np.random.normal(0, sigma, (h, w, c))

            img = img.astype(np.float32) + noise
        else:
            scale = random.uniform(poisson_scale_range[0], poisson_scale_range[1])
            img = img.astype(np.float32)

            if use_gray_noise:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                vals = len(np.unique(gray))
                vals = 2 ** np.ceil(np.log2(vals))

                noise = np.random.poisson(np.maximum(gray * scale, 0)) / scale - gray
                noise = noise[:, :, np.newaxis]
                noise = np.repeat(noise, c, axis=2)
                img = img + noise
            else:
                noise = np.random.poisson(np.maximum(img * scale, 0)) / scale - img
                img = img + noise

        return np.clip(img, 0, 255).astype(np.uint8)

    def _apply_jpeg(self, img, jpeg_range):
        quality = random.randint(jpeg_range[0], jpeg_range[1])
        _, enc_img = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        img = cv2.imdecode(enc_img, 1)
        return img

    def process(self, img_rgb: np.ndarray) -> np.ndarray:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).astype(np.float32)

        kernel1 = self._generate_random_blur_kernel(sigma_range=self.blur_sigma_range)
        img_bgr = cv2.filter2D(img_bgr, -1, kernel1, borderType=cv2.BORDER_REFLECT)

        img_bgr = self._apply_resize(
            img_bgr, resize_prob=self.resize_prob, resize_range=self.resize_range
        )

        img_bgr = self._apply_noise(
            img_bgr,
            gaussian_prob=self.gaussian_noise_prob,
            noise_range=self.noise_range,
            poisson_scale_range=self.poisson_scale_range,
            gray_noise_prob=self.gray_noise_prob,
        )

        img_bgr = self._apply_jpeg(img_bgr, jpeg_range=self.jpeg_range)

        if random.random() < self.second_blur_prob:
            kernel2 = self._generate_random_blur_kernel(
                sigma_range=self.blur_sigma_range_2
            )
            img_bgr = cv2.filter2D(img_bgr, -1, kernel2, borderType=cv2.BORDER_REFLECT)

            img_bgr = self._apply_resize(
                img_bgr,
                resize_prob=self.resize_prob_2,
                resize_range=self.resize_range_2,
            )

            img_bgr = self._apply_noise(
                img_bgr,
                gaussian_prob=self.gaussian_noise_prob_2,
                noise_range=self.noise_range_2,
                poisson_scale_range=self.poisson_scale_range_2,
                gray_noise_prob=self.gray_noise_prob_2,
            )

            if random.random() < 0.5:
                if random.random() < self.sinc_prob:
                    sinc_kernel = self._generate_sinc_kernel()
                    img_bgr = cv2.filter2D(
                        img_bgr, -1, sinc_kernel, borderType=cv2.BORDER_REFLECT
                    )

                img_bgr = self._apply_jpeg(img_bgr, jpeg_range=self.jpeg_range_2)
            else:
                img_bgr = self._apply_jpeg(img_bgr, jpeg_range=self.jpeg_range_2)

                if random.random() < self.sinc_prob:
                    sinc_kernel = self._generate_sinc_kernel()
                    img_bgr = cv2.filter2D(
                        img_bgr, -1, sinc_kernel, borderType=cv2.BORDER_REFLECT
                    )

        orig_h, orig_w = img_rgb.shape[:2]
        target_h = orig_h // self.scaling_factor
        target_w = orig_w // self.scaling_factor

        out_bgr = cv2.resize(
            img_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4
        )

        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        out_rgb = np.clip(out_rgb, 0, 255).astype(np.uint8)

        return out_rgb


def worker_init_fn(worker_id: int) -> None:
    import cv2

    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)


def apply_usm_sharpening(img: np.ndarray, amount: float, radius: int, threshold: int):
    if radius % 2 == 0:
        radius += 1

    img_float = img.astype(np.float32)

    blurred = cv2.GaussianBlur(img_float, (radius, radius), 0)

    diff = img_float - blurred

    if threshold > 0:
        mask = np.abs(diff) >= threshold
        diff *= mask

    sharpened = img_float + amount * diff

    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return sharpened


def create_hr_and_lr_imgs(
    img_path: str | Path,
    scaling_factor: Literal[2, 4, 8],
    crop_size: int | None = None,
    test_mode: bool = False,
    img_degradation_pipeline: ImgDegradationPipeline | None = None,
) -> tuple[Tensor, Tensor]:
    img_tensor = decode_image(Path(img_path).__fspath__())

    if test_mode:
        _, height, width = img_tensor.shape

        height_remainder = height % scaling_factor
        width_remainder = width % scaling_factor

        top_bound = height_remainder // 2
        left_bound = width_remainder // 2

        bottom_bound = top_bound + (height - height_remainder)
        right_bound = left_bound + (width - width_remainder)

        hr_img_tensor = img_tensor[:, top_bound:bottom_bound, left_bound:right_bound]
    elif crop_size:
        augmentation_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomChoice(
                    [
                        transforms.RandomRotation(degrees=[0, 0]),
                        transforms.RandomRotation(degrees=[90, 90]),
                        transforms.RandomRotation(degrees=[180, 180]),
                        transforms.RandomRotation(degrees=[270, 270]),
                    ]
                ),
                transforms.RandomCrop(size=(crop_size, crop_size)),
            ]
        )

        hr_img_tensor = augmentation_transforms(img_tensor)

    if test_mode:
        lr_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size=(
                        hr_img_tensor.shape[1] // scaling_factor,
                        hr_img_tensor.shape[2] // scaling_factor,
                    ),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                )
            ]
        )
        lr_img_tensor = lr_transforms(hr_img_tensor)
    elif img_degradation_pipeline:
        hr_img_np = hr_img_tensor.permute(1, 2, 0).numpy()

        if config.USE_USM_SHARPENING:
            hr_img_np = apply_usm_sharpening(
                img=hr_img_np,
                amount=config.USM_AMOUNT,
                radius=config.USM_RADIUS,
                threshold=config.USM_THRESHOLD,
            )
            hr_img_tensor = torch.from_numpy(hr_img_np).permute(2, 0, 1)

        lr_img_np = img_degradation_pipeline.process(hr_img_np)
        lr_img_tensor = torch.from_numpy(lr_img_np).permute(2, 0, 1)
    else:
        logger.error("Use either test_mode=True or pass the img_degradation_pipeline")
        raise NotImplementedError

    normalize_transforms = transforms.Compose(
        [
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    hr_img_tensor = normalize_transforms(hr_img_tensor)
    lr_img_tensor = normalize_transforms(lr_img_tensor)

    return hr_img_tensor, lr_img_tensor


@overload
def convert_img(
    img: Tensor,
    source: Literal["[-1, 1]", "[0, 1]", "imagenet", "uint8"],
    target: Literal["pil"],
) -> Image.Image: ...


@overload
def convert_img(
    img: Tensor,
    source: Literal["[-1, 1]", "[0, 1]", "imagenet", "uint8"],
    target: Literal["[-1, 1]", "[0, 1]", "imagenet", "uint8", "y-channel"],
) -> Tensor: ...


def convert_img(
    img: Tensor,
    source: Literal["[-1, 1]", "[0, 1]", "imagenet", "uint8"],
    target: Literal["[-1, 1]", "[0, 1]", "imagenet", "uint8", "pil", "y-channel"],
) -> Tensor | Image.Image:
    if single_tensor := img.dim() == 3:
        img.unsqueeze_(0)

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    ycbcr_weights = [0.299, 0.587, 0.114]

    imagenet_mean_tensor = torch.tensor(imagenet_mean, device=img.device).view(
        1, 3, 1, 1
    )
    imagenet_std_tensor = torch.tensor(imagenet_std, device=img.device).view(1, 3, 1, 1)
    ycbcr_weights_tensor = torch.tensor(ycbcr_weights, device=img.device).view(
        1, 3, 1, 1
    )

    imagenet_norm_transform = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    to_pil_img_transform = transforms.ToPILImage()

    match source:
        case "[0, 1]":
            pass
        case "[-1, 1]":
            img = (img + 1.0) / 2.0
        case "imagenet":
            img = img * imagenet_std_tensor + imagenet_mean_tensor
        case "uint8":
            img = img.to(torch.float32) / 255.0
        case _:
            raise ValueError(f"Unknown source format: {source}")

    match target:
        case "[0, 1]":
            pass
        case "[-1, 1]":
            img = img * 2.0 - 1.0
        case "imagenet":
            img = imagenet_norm_transform(img)
        case "uint8":
            img = (img.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        case "pil":
            img = to_pil_img_transform(img[0])
        case "y-channel":
            img = torch.sum(img * ycbcr_weights_tensor, dim=1, keepdim=True)
        case _:
            raise ValueError(f"Unknown target format: {target}")

    if single_tensor and target != "pil":
        img.squeeze_(0)

    return img


def compare_imgs(
    lr_img_tensor: Tensor,
    sr_img_tensor: Tensor,
    output_path: str | Path,
    hr_img_tensor: Tensor | None = None,
    scaling_factor: Literal[2, 4, 8] = 4,
    orientation: Literal["horizontal", "vertical"] = "vertical",
) -> None:
    bicubic_label = "Bicubic"
    sr_label = "Real-ESRGAN"
    hr_label = "Original"

    lr_img = convert_img(lr_img_tensor, "[-1, 1]", "pil")
    sr_img = convert_img(sr_img_tensor, "[-1, 1]", "pil")

    bicubic_img = transforms.Resize(
        size=(sr_img_tensor.shape[2], sr_img_tensor.shape[3]),
        interpolation=transforms.InterpolationMode.BICUBIC,
    )(lr_img)

    width, height = sr_img.size

    if orientation == "horizontal" and isinstance(hr_img_tensor, Tensor):
        hr_img = convert_img(hr_img_tensor, "[-1, 1]", "pil")

        total_width = width * 3 + 50
        total_height = height

        comparison_img = Image.new("RGB", (total_width, total_height), color="white")

        comparison_img.paste(bicubic_img, (0, 50))
        comparison_img.paste(sr_img, (width + 25, 50))
        comparison_img.paste(hr_img, (width * 2 + 50, 50))
    else:
        total_width = width
        total_height = height * 2 + 100

        comparison_img = Image.new("RGB", (total_width, total_height), color="white")

        comparison_img.paste(bicubic_img, (0, 50))
        comparison_img.paste(sr_img, (0, height + 100))

    draw = ImageDraw.Draw((comparison_img))

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Regular.ttf", size=36
        )
    except OSError:
        font = ImageFont.load_default()

    bicubic_text_width = draw.textlength(bicubic_label, font=font)
    sr_text_width = draw.textlength(sr_label, font=font)
    hr_text_width = draw.textlength(hr_label, font=font)

    if orientation == "horizontal":
        draw.text(
            ((width - bicubic_text_width) / 2, 5),
            bicubic_label,
            fill="black",
            font=font,
        )

        draw.text(
            ((width - sr_text_width) / 2 + width + 25, 5),
            sr_label,
            fill="black",
            font=font,
        )

        draw.text(
            ((width - hr_text_width) / 2 + width * 2 + 50, 5),
            hr_label,
            fill="black",
            font=font,
        )
    else:
        draw.text(
            ((width - bicubic_text_width) / 2, 5),
            bicubic_label,
            fill="black",
            font=font,
        )

        draw.text(
            ((width - sr_text_width) / 2, height + 55),
            sr_label,
            fill="black",
            font=font,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_img.save(output_path, format="PNG")


def _save_optimizer_state(
    optimizer: optim.Optimizer,
    checkpoint_dir_path: str | Path,
    prefix: str,
) -> dict:
    checkpoint_dir_path = Path(checkpoint_dir_path)

    optimizer_state = optimizer.state_dict()
    optimizer_tensors = {}
    optimizer_metadata = {"param_groups": optimizer_state["param_groups"]}

    optimizer_state_buffers = optimizer_state["state"]
    optimizer_metadata["state"] = {}

    for param_id, buffers in optimizer_state_buffers.items():
        param_id_str = str(param_id)
        optimizer_metadata["state"][param_id_str] = {}

        for buffer_name, value in buffers.items():
            if isinstance(value, torch.Tensor):
                tensor_key = f"state_{param_id_str}_{buffer_name}"
                optimizer_tensors[tensor_key] = value
            else:
                optimizer_metadata["state"][param_id_str][buffer_name] = value

    if optimizer_tensors:
        save_file(
            optimizer_tensors, checkpoint_dir_path / f"{prefix}_optimizer.safetensors"
        )

    return optimizer_metadata


def _load_optimizer_state(
    optimizer: optim.Optimizer,
    checkpoint_dir_path: str | Path,
    prefix: str,
    full_metadata: dict,
    device: str,
) -> None:
    checkpoint_dir_path = Path(checkpoint_dir_path)

    optimizer_metadata_key = f"{prefix}_optimizer_metadata"
    if optimizer_metadata_key not in full_metadata:
        logger.warning(
            f"Metadata for '{optimizer_metadata_key}' not found in training_state.json"
        )
        return

    optimizer_metadata = full_metadata[optimizer_metadata_key]

    optimizer_tensors_path = checkpoint_dir_path / f"{prefix}_optimizer.safetensors"
    if optimizer_tensors_path.exists():
        optimizer_tensors = load_file(filename=optimizer_tensors_path, device=device)
    else:
        optimizer_tensors = {}
        logger.warning(f"Optimizer tensor file not found: {optimizer_tensors_path}")

    optimizer_state_buffers = {
        int(param_id): buffers
        for param_id, buffers in optimizer_metadata["state"].items()
    }

    for tensor_key, tensor_value in optimizer_tensors.items():
        parts = tensor_key.split("_")
        if len(parts) < 3 or parts[0] != "state":
            logger.warning(
                f"Unrecognized tensor key in {prefix}_optimizer: {tensor_key}"
            )
            continue

        param_id = int(parts[1])
        buffer_name = "_".join(parts[2:])

        if param_id not in optimizer_state_buffers:
            optimizer_state_buffers[param_id] = {}

        optimizer_state_buffers[param_id][buffer_name] = tensor_value

    optimizer_state_to_load = {
        "param_groups": optimizer_metadata["param_groups"],
        "state": optimizer_state_buffers,
    }

    try:
        optimizer.load_state_dict(optimizer_state_to_load)
    except Exception as e:
        logger.error(f"Failed to load state_dict for {prefix}_optimizer: {e}")
        logger.warning(f"Continuing without loading {prefix}_optimizer state.")


def save_checkpoint(
    checkpoint_dir_path: str | Path,
    epoch: int,
    generator: nn.Module,
    generator_optimizer: optim.Optimizer,
    metrics: Metrics,
    generator_scaler: GradScaler | None = None,
    generator_scheduler: MultiStepLR | None = None,
    discriminator: nn.Module | None = None,
    discriminator_optimizer: optim.Optimizer | None = None,
    discriminator_scaler: GradScaler | None = None,
    discriminator_scheduler: MultiStepLR | None = None,
) -> None:
    checkpoint_dir_path = Path(checkpoint_dir_path)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    save_file(generator.state_dict(), checkpoint_dir_path / "generator.safetensors")

    generator_optimizer_metadata = _save_optimizer_state(
        generator_optimizer,
        checkpoint_dir_path,
        "generator",
    )

    if discriminator and discriminator_optimizer:
        save_file(
            discriminator.state_dict(),
            checkpoint_dir_path / "discriminator.safetensors",
        )
        discriminator_optimizer_metadata = _save_optimizer_state(
            discriminator_optimizer,
            checkpoint_dir_path,
            "discriminator",
        )

    full_metadata = {
        "epoch": epoch,
        "metrics": asdict(metrics),
        "generator_optimizer_metadata": generator_optimizer_metadata,
        "discriminator_optimizer_metadata": discriminator_optimizer_metadata
        if discriminator
        else None,
        "generator_scaler_state_dict": generator_scaler.state_dict()
        if generator_scaler
        else None,
        "discriminator_scaler_state_dict": discriminator_scaler.state_dict()
        if discriminator_scaler
        else None,
        "generator_scheduler_state_dict": generator_scheduler.state_dict()
        if generator_scheduler
        else None,
        "discriminator_scheduler_state_dict": discriminator_scheduler.state_dict()
        if discriminator_scheduler
        else None,
    }

    with open(checkpoint_dir_path / "training_state.json", "w") as f:
        json.dump(full_metadata, f, indent=4)

    logger.debug(f'Checkpoint was saved to "{checkpoint_dir_path}" after {epoch} epoch')


def load_checkpoint(
    checkpoint_dir_path: str | Path,
    generator: nn.Module,
    test_mode: bool = False,
    metrics: Metrics | None = None,
    generator_optimizer: optim.Optimizer | None = None,
    generator_scaler: GradScaler | None = None,
    generator_scheduler: MultiStepLR | None = None,
    discriminator: nn.Module | None = None,
    discriminator_optimizer: optim.Optimizer | None = None,
    discriminator_scaler: GradScaler | None = None,
    discriminator_scheduler: MultiStepLR | None = None,
    device: Literal["cuda", "cpu"] = "cpu",
) -> int:
    checkpoint_dir_path = Path(checkpoint_dir_path)

    generator_path = checkpoint_dir_path / "generator.safetensors"
    discriminator_path = checkpoint_dir_path / "discriminator.safetensors"
    state_path = checkpoint_dir_path / "training_state.json"

    if generator_path.exists():
        generator.load_state_dict(load_file(filename=generator_path, device=device))
    else:
        logger.warning(
            f"Checkpoint (generator.safetensors or training_state.json) was not found at '{checkpoint_dir_path}', starting from 1 epoch"
        )
        return 1

    if test_mode:
        logger.info(
            f"Loaded Generator weights from '{checkpoint_dir_path}' (test_mode=True)"
        )
        return 1

    if state_path.exists():
        with open(state_path, "r") as f:
            full_metadata = json.load(f)

        if metrics and "metrics" in full_metadata:
            metrics_dict = full_metadata["metrics"]
            metrics.epochs = metrics_dict["epochs"]
            metrics.generator_learning_rates = metrics_dict["generator_learning_rates"]
            metrics.discriminator_learning_rates = metrics_dict[
                "discriminator_learning_rates"
            ]
            metrics.generator_train_losses = metrics_dict["generator_train_losses"]
            metrics.discriminator_train_losses = metrics_dict[
                "discriminator_train_losses"
            ]
            metrics.generator_val_losses = metrics_dict["generator_val_losses"]
            metrics.generator_val_psnrs = metrics_dict["generator_val_psnrs"]
            metrics.generator_val_ssims = metrics_dict["generator_val_ssims"]

        if generator_optimizer:
            _load_optimizer_state(
                generator_optimizer,
                checkpoint_dir_path,
                "generator",
                full_metadata,
                device,
            )

        if generator_scaler and full_metadata.get("generator_scaler_state_dict"):
            generator_scaler.load_state_dict(
                full_metadata["generator_scaler_state_dict"]
            )

        if generator_scheduler and full_metadata.get("generator_scheduler_state_dict"):
            generator_scheduler.load_state_dict(
                full_metadata["generator_scheduler_state_dict"]
            )

        if discriminator and discriminator_optimizer:
            discriminator.load_state_dict(
                load_file(filename=discriminator_path, device=device)
            )

            _load_optimizer_state(
                discriminator_optimizer,
                checkpoint_dir_path,
                "discriminator",
                full_metadata,
                device,
            )

            if discriminator_scaler and full_metadata.get(
                "discriminator_scaler_state_dict"
            ):
                discriminator_scaler.load_state_dict(
                    full_metadata["discriminator_scaler_state_dict"]
                )

            if discriminator_scheduler and full_metadata.get(
                "discriminator_scheduler_state_dict"
            ):
                discriminator_scheduler.load_state_dict(
                    full_metadata["discriminator_scheduler_state_dict"]
                )

        logger.info(f'Checkpoint was loaded from "{checkpoint_dir_path}"')

        return full_metadata["epoch"]
    else:
        logger.error("State path does not exists, can not load model parameters")
        return 0


def format_time(total_seconds: float) -> str:
    if total_seconds < 0:
        total_seconds = 0

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def create_hyperparameters_str() -> str:
    return f"Scaling factor: {config.SCALING_FACTOR} | Crop size: {config.CROP_SIZE} | Batch size: {config.TRAIN_BATCH_SIZE} | Generator learning rate: {config.GENERATOR_LEARNING_RATE} | Discriminator learning rate: {config.DISCRIMINATOR_LEARNING_RATE}| Epochs: {config.EPOCHS} | Number of workers: {config.NUM_WORKERS} | Dev mode: {config.DEV_MODE}"


def plot_training_metrics(
    metrics: Metrics,
    hyperparameters_str: str,
    model_type: Literal["real-esrnet", "real-esrgan"],
) -> None:
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    palette = sns.color_palette("deep")

    epochs = list(range(0, len(metrics.generator_train_losses)))

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    if model_type == "real-esrnet":
        fig.suptitle("Real-ESRNET Training Metrics", fontsize=18)
    else:
        fig.suptitle("Real-ESRGAN Training Metrics", fontsize=18)

    fig.text(0.5, 0.94, hyperparameters_str, ha="center", va="top", fontsize=10)

    sns.lineplot(
        x=epochs,
        y=metrics.generator_train_losses,
        label="Generator train loss",
        ax=axs[0, 0],
        linewidth=2.5,
        color=palette[0],
    )

    if model_type == "real-esrgan":
        sns.lineplot(
            x=epochs,
            y=metrics.generator_val_losses,
            label="Generator val loss",
            ax=axs[0, 0],
            linewidth=2.5,
            color=palette[1],
        )
        axs[0, 0].set_title("Generator training and validation losses")
        axs[0, 0].set_xlabel("Epoch")
        axs[0, 0].set_ylabel("Loss")

        sns.lineplot(
            x=epochs,
            y=metrics.discriminator_train_losses,
            ax=axs[0, 1],
            linewidth=2.5,
            color=palette[2],
        )
        axs[0, 1].set_title("Discriminator training loss")
        axs[0, 1].set_xlabel("Epoch")
        axs[0, 1].set_ylabel("Loss")

    elif model_type == "real-esrnet":
        axs[0, 0].set_title("Generator training loss")
        axs[0, 0].set_xlabel("Epoch")
        axs[0, 0].set_ylabel("Loss")

        sns.lineplot(
            x=epochs,
            y=metrics.generator_val_losses,
            label="Generator val loss",
            ax=axs[0, 1],
            linewidth=2.5,
            color=palette[1],
        )

        axs[0, 1].set_title("Generator val loss")
        axs[0, 1].set_xlabel("Epoch")
        axs[0, 1].set_ylabel("Loss")

    sns.lineplot(
        x=epochs,
        y=metrics.generator_val_psnrs,
        ax=axs[1, 0],
        linewidth=2.5,
        color=palette[1],
    )
    axs[1, 0].set_title("Validation Peak Signal-to-Noise Ratio")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("PSNR")

    sns.lineplot(
        x=epochs,
        y=metrics.generator_val_ssims,
        ax=axs[1, 1],
        linewidth=2.5,
        color=palette[1],
    )
    axs[1, 1].set_title("Validation Structural Similarity Index Measure")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("SSIM")

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])

    output_path = (
        Path("images")
        / f"training_metrics_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()


def apply_network_interpolation(
    model: nn.Module,
    params_psnr: dict,
    params_esrgan: dict,
    alpha: float,
) -> None:
    logger.info("Creating network interpolated model...")

    params_interpolated = dict()

    for key in params_psnr.keys():
        if key in params_esrgan:
            tensor_psnr = params_psnr[key].float()
            tensor_esrgan = params_esrgan[key].float()

            params_interpolated[key] = (1 - alpha) * tensor_psnr + alpha * tensor_esrgan
        else:
            logger.warning(
                f'Key "{key}" not found in Real-ESRGAN model. Copying from Real-ESRNET model...'
            )
            params_interpolated[key] = params_psnr[key].float()

    model.load_state_dict(params_interpolated)


def upscale_img_tiled(
    model: nn.Module,
    lr_img_tensor: Tensor,
    scale_factor: Literal[2, 4, 8] = 4,
    tile_size: int = 512,
    tile_overlap: int = 64,
    device: Literal["cuda", "cpu"] = "cpu",
) -> Tensor:
    batch_size, channels, height_original, width_original = lr_img_tensor.shape

    height_target = height_original * scale_factor
    width_target = width_original * scale_factor

    border_pad = tile_overlap // 2

    lr_img_tensor_padded = F.pad(
        lr_img_tensor, (border_pad, border_pad, border_pad, border_pad), "reflect"
    )

    _, _, height_padded, width_padded = lr_img_tensor_padded.shape

    step_size = tile_size - tile_overlap

    pad_right = (step_size - (width_padded - tile_size) % step_size) % step_size
    pad_bottom = (step_size - (height_padded - tile_size) % step_size) % step_size

    lr_img_tensor_padded = F.pad(
        lr_img_tensor_padded, (0, pad_right, 0, pad_bottom), "reflect"
    )

    _, _, height_final, width_final = lr_img_tensor_padded.shape

    # logger.debug(
    #     f"Original LR: {width_original}x{height_original} | Target SR: {width_target}x{height_target}"
    # )

    final_img_canvas = torch.zeros(
        (batch_size, channels, height_final * scale_factor, width_final * scale_factor),
        dtype=lr_img_tensor.dtype,
        device="cpu",
    )

    count_canvas = torch.zeros_like(final_img_canvas, device="cpu")

    for height in range(0, height_final - tile_size + 1, step_size):
        for width in range(0, width_final - tile_size + 1, step_size):
            lr_img_tensor_tile = lr_img_tensor_padded[
                :, :, height : height + tile_size, width : width + tile_size
            ].to(device, non_blocking=True)

            with torch.inference_mode():
                sr_img_tensor_tile = model(lr_img_tensor_tile).cpu()

            final_height_start = height * scale_factor
            final_width_start = width * scale_factor
            final_height_end = (height + tile_size) * scale_factor
            final_width_end = (width + tile_size) * scale_factor

            final_img_canvas[
                :,
                :,
                final_height_start:final_height_end,
                final_width_start:final_width_end,
            ] += sr_img_tensor_tile

            count_canvas[
                :,
                :,
                final_height_start:final_height_end,
                final_width_start:final_width_end,
            ] += 1

    # logger.debug("All tiles processed. Blending results...")

    output_padded = final_img_canvas / count_canvas

    final_border_pad = border_pad * scale_factor

    final_output = output_padded[
        :,
        :,
        final_border_pad : final_border_pad + height_target,
        final_border_pad : final_border_pad + width_target,
    ]

    return final_output
