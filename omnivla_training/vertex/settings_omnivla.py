import os

from omnivla_training.vertex import utils as vertex_utils


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


# Vertex AI / GCP
PROJECT_ID = _env("VERTEX_PROJECT_ID")
REGION = _env("VERTEX_REGION", "us-central1")
IMAGE_URI = _env("VERTEX_OMNIVLA_IMAGE_URI", f"gcr.io/{PROJECT_ID}/omnivla-training:latest" if PROJECT_ID else "")
STAGING_BUCKET = _env("VERTEX_OMNIVLA_STAGING_BUCKET")
TRAIN_BUCKET = _env("VERTEX_OMNIVLA_TRAIN_BUCKET")

# Hardware defaults
# Default to a single A100 since OpenVLA / OmniVLA 7B is unlikely to fit on smaller cards comfortably.
MACHINE_TYPE = os.getenv("VERTEX_OMNIVLA_MACHINE_TYPE", "a2-highgpu-1g")
ACCELERATOR_TYPE = os.getenv("VERTEX_OMNIVLA_ACCELERATOR_TYPE", "NVIDIA_TESLA_A100")
ACCELERATOR_COUNT = int(os.getenv("VERTEX_OMNIVLA_ACCELERATOR_COUNT", "1"))
BOOT_DISK_TYPE = os.getenv("VERTEX_OMNIVLA_BOOT_DISK_TYPE", "pd-ssd")
BOOT_DISK_SIZE_GB = int(os.getenv("VERTEX_OMNIVLA_BOOT_DISK_SIZE_GB", "500"))
TIMEOUT_DAYS = int(os.getenv("VERTEX_OMNIVLA_TIMEOUT_DAYS", "3"))

# Secrets
WANDB_SECRET = _env("VERTEX_WANDB_SECRET")
WANDB_VERSION = _env("VERTEX_WANDB_SECRET_VERSION", "latest")
HF_TOKEN_SECRET = _env("VERTEX_HF_TOKEN_SECRET")
HF_TOKEN_VERSION = _env("VERTEX_HF_TOKEN_SECRET_VERSION", "latest")

# Run naming
TIMESTAMP = vertex_utils.get_timestamp()
USERNAME = vertex_utils.get_gituser().strip().replace(" ", "_")
JOB_NAME_PREFIX = os.getenv("VERTEX_OMNIVLA_JOB_PREFIX", "omnivla")

TRAIN_JOB_NAME = f"{JOB_NAME_PREFIX}-{TIMESTAMP}"
TRAIN_SAVE_DIR = f"{TRAIN_BUCKET.rstrip('/')}/artifacts/{USERNAME}/{TRAIN_JOB_NAME}"
TRAIN_JOB_DIR = f"{TRAIN_BUCKET.rstrip('/')}/jobs/{TRAIN_JOB_NAME}"
