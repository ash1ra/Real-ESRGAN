import logging
import math
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal

# Generator settings
GENERATOR_CHANNELS_COUNT = 64
GENERATOR_GROWTHS_CHANNELS_COUNT = 32
GENERATOR_RES_DENSE_BLOCKS_COUNT = 3
GENERATOR_RRDB_COUNT = 23
GENERATOR_LARGE_KERNEL_SIZE = 9
GENERATOR_SMALL_KERNEL_SIZE = 3

# Discriminator settings
DISCRIMINATOR_CHANNELS_COUNT = 64

# Basic parameters
SCALING_FACTOR: Literal[2, 4, 8] = 4
CROP_SIZE = 256
EPOCHS = 250
PRINT_FREQUENCY = 200

GRADIENT_ACCUMULATION_STEPS: Literal[1, 2, 4, 6, 8, 12, 16] = 12
TRAIN_BATCH_SIZE = 48
REAL_TRAIN_BATCH_SIZE = TRAIN_BATCH_SIZE // GRADIENT_ACCUMULATION_STEPS
TEST_BATCH_SIZE = 1

GENERATOR_LEARNING_RATE = 1e-5
DISCRIMINATOR_LEARNING_RATE = 1e-5

SCHEDULER_MILESTONES = [150]
SCHEDULER_SCALING_VALUE = 0.5

NUM_WORKERS = 8

TILE_SIZE = 512
TILE_OVERLAP = 64

# USM-sharpening parameters
USE_USM_SHARPENING = False
USM_AMOUNT = 1.2
USM_RADIUS = 50
USM_THRESHOLD = 10

# Loss calculation parameters
PERCEPTUAL_LOSS_LAYER_WEIGHTS = {
    "conv1_2": 0.1,
    "conv2_2": 0.1,
    "conv3_4": 1.0,
    "conv4_4": 1.0,
    "conv5_4": 1.0,
}

PERCEPTUAL_LOSS_SCALING_VALUE = 1.0
ADVERSARIAL_LOSS_SCALING_VALUE = 0.1
PIXEL_LOSS_SCALING_VALUE = 1.0

# Initialization parameters
WEIGHTS_SCALING_VALUE = 0.1
RESIDUAL_SCALING_VALUE = 0.2

GRADIENT_CLIPPING_VALUE = 0.5
LEAKY_RELU_NEGATIVE_SLOPE_VALUE = 0.2

# Image degradation parameters
BLUR_KERNEL_SIZE_LIST = [i for i in range(7, 22, 2)]

BLUR_KERNEL_PROBABILITY = [0.7, 0.15, 0.15]
BLUR_SIGMA_RANGE = [0.2, 3.0]
BLUR_SIGMA_RANGE_2 = [0.2, 1.5]
BETAG_RANGE = [0.5, 4.0]
BETAP_RANGE = [1, 2]

SINC_PROBABILITY = 0.1
SINC_KERNEL_SIZE = 21
OMEGA_RANGE = [math.pi / 3, math.pi]

# First degradation process
RESIZE_PROBABILITY = [0.2, 0.7, 0.1]
RESIZE_RANGE = [0.15, 1.5]

GAUSSIAN_NOISE_PROBABILITY = 0.5
NOISE_RANGE = [1, 30]
POISSON_SCALE_RANGE = [0.05, 3.0]
GRAY_NOISE_PROBABILITY = 0.4

JPEG_RANGE = [30, 95]

# Second degradation process
SECOND_BLUR_PROBABILITY = 0.8
RESIZE_PROBABILITY_2 = [0.3, 0.4, 0.3]
RESIZE_RANGE_2 = [0.3, 1.2]
GAUSSIAN_NOISE_PROBABILITY_2 = 0.5
NOISE_RANGE_2 = [1, 25]

POISSON_SCALE_RANGE_2 = [0.05, 2.5]
GRAY_NOISE_PROBABILITY_2 = 0.4

JPEG_RANGE_2 = [30, 95]

# Model parameters loading
INITIALIZE_WITH_ESRGAN_CHECKPOINT = False
LOAD_REAL_ESRNET_CHECKPOINT = True
LOAD_BEST_REAL_ESRNET_CHECKPOINT = False

INITIALIZE_WITH_REAL_ESRNET_CHECKPOINT = False
LOAD_REAL_ESRGAN_CHECKPOINT = True
LOAD_BEST_REAL_ESRGAN_CHECKPOINT = False

DEV_MODE = False

# Pathes
TRAIN_DATASET_PATH = Path("data/DF2K_OST.txt")
VAL_DATASET_PATH = Path("data/DIV2K_valid.txt")
TEST_DATASETS_PATHS = [
    Path("data/Set5.txt"),
    Path("data/Set14.txt"),
    Path("data/BSDS100.txt"),
    Path("data/Urban100.txt"),
]

BEST_ESRGAN_CHECKPOINT_DIR_PATH = Path("checkpoints/esrgan_best")
REAL_ESRNET_CHECKPOINT_DIR_PATH = Path("checkpoints/real_esrnet_latest")
BEST_REAL_ESRNET_CHECKPOINT_DIR_PATH = Path("checkpoints/real_esrnet_best")
REAL_ESRGAN_CHECKPOINT_DIR_PATH = Path("checkpoints/real_esrgan_latest")
BEST_REAL_ESRGAN_CHECKPOINT_DIR_PATH = Path("checkpoints/real_esrgan_best")

INFERENCE_INPUT_IMG_PATH = Path("images/inference_img_1.jpg")
INFERENCE_OUTPUT_IMG_PATH = Path("images/sr_img_1.png")
INFERECE_COMPARISON_IMG_PATH = Path("images/comparison_img_1.png")


def create_logger(
    log_level: str,
    caller_file_name: str,
    log_file_name: str | None = None,
    max_log_file_size: int = 5 * 1024 * 1024,
    backup_count: int = 10,
) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y.%m.%d %H:%M:%S"
    )

    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    logger.addHandler(console_handler)

    if not log_file_name:
        current_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        log_file_name = (
            f"logs/real-esrgan_{Path(caller_file_name).stem}_{current_date}.log"
        )

    log_file_path = Path(log_file_name)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        filename=log_file_path,
        maxBytes=max_log_file_size,
        backupCount=backup_count,
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
