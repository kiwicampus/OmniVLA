from pathlib import Path
from typing import Optional
import os
import sys

try:
    import typer
    from google.cloud import aiplatform
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing Vertex submit dependencies. Install them with: "
        "`pip install -r omnivla_training/requirements_vertex_submit.txt`"
    ) from exc

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from omnivla_training.vertex import settings_omnivla as vertex_settings
    from omnivla_training.vertex import utils as vertex_utils
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing Vertex submit dependencies. Install them with: "
        "`pip install -r omnivla_training/requirements_vertex_submit.txt`"
    ) from exc


def maybe_access_secret(secret_id: str, version_id: str, project_id: str) -> Optional[str]:
    if not secret_id:
        return None

    try:
        success, payload = vertex_utils.access_secret_version(secret_id, version_id, project_id)
    except Exception as exc:
        print(f"Skipping secret `{secret_id}`: {exc}")
        return None

    if not success:
        print(f"Skipping secret `{secret_id}` due to checksum failure.")
        return None

    return payload.decode("utf-8")


def build_job_paths(job_name: str) -> tuple[str, str]:
    train_bucket = vertex_settings.TRAIN_BUCKET.rstrip("/")
    save_dir = f"{train_bucket}/artifacts/{vertex_settings.USERNAME}/{job_name}"
    job_dir = f"{train_bucket}/jobs/{job_name}"
    return save_dir, job_dir


def validate_vertex_settings() -> None:
    required_values = {
        "VERTEX_PROJECT_ID": vertex_settings.PROJECT_ID,
        "VERTEX_OMNIVLA_IMAGE_URI": vertex_settings.IMAGE_URI,
        "VERTEX_OMNIVLA_STAGING_BUCKET": vertex_settings.STAGING_BUCKET,
        "VERTEX_OMNIVLA_TRAIN_BUCKET": vertex_settings.TRAIN_BUCKET,
    }
    missing = [name for name, value in required_values.items() if not value]
    if missing:
        missing_lines = "\n  ".join(missing)
        raise SystemExit(
            "Missing required Vertex environment variables:\n"
            f"  {missing_lines}\n\n"
            "Set them before submitting a Vertex job. See TRAINING.md for the full setup."
        )


def validate_hardware(
    region: str,
    machine_type: str,
    accelerator_type: str,
    accelerator_count: int,
) -> None:
    normalized_region = region.lower()
    normalized_machine_type = machine_type.lower()
    normalized_accelerator_type = accelerator_type.upper()

    if normalized_region == "us-east4" and (
        normalized_machine_type.startswith("a2-highgpu-")
        or normalized_accelerator_type == "NVIDIA_TESLA_A100"
    ):
        raise SystemExit(
            "Unsupported Vertex AI hardware for region `us-east4`: "
            f"`{machine_type}` with `{accelerator_type}` requests an A100 40GB configuration, "
            "but `us-east4` currently exposes A100 80GB instead.\n"
            "Use one of these options:\n"
            "  1. Keep the current defaults and switch to `--region us-central1`\n"
            "  2. Stay in `us-east4` and submit with "
            "`--machine-type a2-ultragpu-1g --accelerator-type NVIDIA_A100_80GB "
            f"--accelerator-count {accelerator_count}`"
        )


def main(
    config_path: str = typer.Option("config_nav/train_omnivla.yaml", "--config-path"),
    region: Optional[str] = typer.Option(None, "--region"),
    output_dir: Optional[str] = None,
    job_name: Optional[str] = None,
    dataset_repo_id: Optional[str] = None,
    dataset_revision: Optional[str] = None,
    dataset_root: Optional[str] = None,
    model_path: Optional[str] = None,
    steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    grad_accumulation_steps: Optional[int] = None,
    save_freq: Optional[int] = None,
    num_workers: Optional[int] = None,
    accelerator_count: Optional[int] = None,
    accelerator_type: Optional[str] = None,
    machine_type: Optional[str] = None,
    boot_disk_size_gb: Optional[int] = None,
    smoke_test: bool = False,
    sync: bool = False,
):
    validate_vertex_settings()

    resolved_region = region or vertex_settings.REGION
    resolved_job_name = job_name or vertex_settings.TRAIN_JOB_NAME
    resolved_machine_type = machine_type or vertex_settings.MACHINE_TYPE
    resolved_accelerator_type = accelerator_type or vertex_settings.ACCELERATOR_TYPE
    resolved_accelerator_count = accelerator_count or vertex_settings.ACCELERATOR_COUNT

    validate_hardware(
        region=resolved_region,
        machine_type=resolved_machine_type,
        accelerator_type=resolved_accelerator_type,
        accelerator_count=resolved_accelerator_count,
    )

    default_output_dir, job_dir = build_job_paths(resolved_job_name)
    resolved_output_dir = output_dir or default_output_dir

    if config_path.startswith("gs://"):
        config_gcs_path = config_path
    else:
        config_gcs_path = vertex_utils.upload_config(config_path, job_dir, vertex_settings.PROJECT_ID)

    aiplatform.init(project=vertex_settings.PROJECT_ID, location=resolved_region)

    job = aiplatform.CustomContainerTrainingJob(
        command=[
            "python3",
            "/app/omnivla_training/training_scripts/vertex_train_omnivla_wrapper.py",
        ],
        display_name=resolved_job_name,
        staging_bucket=vertex_settings.STAGING_BUCKET,
        container_uri=vertex_settings.IMAGE_URI,
    )

    training_args = [
        "--config-path",
        config_gcs_path,
        "--output-dir",
        resolved_output_dir,
        "--job-name",
        resolved_job_name,
        "--local-run-root",
        "/tmp/omnivla_output",
        "--nproc-per-node",
        str(resolved_accelerator_count),
    ]

    if dataset_repo_id is not None:
        training_args.extend(["--dataset-repo-id", dataset_repo_id])
    if dataset_revision is not None:
        training_args.extend(["--dataset-revision", dataset_revision])
    if dataset_root is not None:
        training_args.extend(["--dataset-root", dataset_root])
    if model_path is not None:
        training_args.extend(["--model-path", model_path])
    if steps is not None:
        training_args.extend(["--steps", str(steps)])
    if batch_size is not None:
        training_args.extend(["--batch-size", str(batch_size)])
    if grad_accumulation_steps is not None:
        training_args.extend(["--grad-accumulation-steps", str(grad_accumulation_steps)])
    if save_freq is not None:
        training_args.extend(["--save-freq", str(save_freq)])
    if num_workers is not None:
        training_args.extend(["--num-workers", str(num_workers)])
    if smoke_test:
        training_args.append("--smoke-test")

    env_variables = {
        "GCP_PROJECT_ID": vertex_settings.PROJECT_ID,
        "PYTHONPATH": "/app:/app/omnivla_training",
        "HF_HOME": "/tmp/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/tmp/huggingface/hub",
        "TRANSFORMERS_CACHE": "/tmp/huggingface/hub",
        "TORCH_HOME": "/tmp/torch",
        "WANDB_DIR": "/tmp/wandb",
        "TOKENIZERS_PARALLELISM": "false",
        "VERTEX_WANDB_SECRET": vertex_settings.WANDB_SECRET,
        "VERTEX_WANDB_SECRET_VERSION": vertex_settings.WANDB_VERSION,
        "VERTEX_HF_TOKEN_SECRET": vertex_settings.HF_TOKEN_SECRET,
        "VERTEX_HF_TOKEN_SECRET_VERSION": vertex_settings.HF_TOKEN_VERSION,
    }

    wandb_api_key = os.getenv("WANDB_API_KEY") or maybe_access_secret(
        vertex_settings.WANDB_SECRET,
        vertex_settings.WANDB_VERSION,
        vertex_settings.PROJECT_ID,
    )
    if wandb_api_key:
        env_variables["WANDB_API_KEY"] = wandb_api_key

    hf_token = os.getenv("HF_TOKEN") or maybe_access_secret(
        vertex_settings.HF_TOKEN_SECRET,
        vertex_settings.HF_TOKEN_VERSION,
        vertex_settings.PROJECT_ID,
    )
    if hf_token:
        env_variables["HF_TOKEN"] = hf_token

    _ = job.run(
        replica_count=1,
        machine_type=resolved_machine_type,
        accelerator_type=resolved_accelerator_type,
        accelerator_count=resolved_accelerator_count,
        boot_disk_type=vertex_settings.BOOT_DISK_TYPE,
        boot_disk_size_gb=boot_disk_size_gb or vertex_settings.BOOT_DISK_SIZE_GB,
        args=training_args,
        environment_variables=env_variables,
        enable_web_access=True,
        sync=sync,
        timeout=vertex_settings.TIMEOUT_DAYS * 24 * 60 * 60,
    )

    training_pipeline_name = None
    training_pipeline_url = None
    try:
        training_pipeline_name = job.resource_name
    except Exception:
        training_pipeline_name = None

    try:
        training_pipeline_url = job._dashboard_uri()
    except Exception:
        training_pipeline_url = None

    print(f"\n{'=' * 60}")
    print("Vertex OmniVLA job submitted successfully.")
    print(f"Region: {resolved_region}")
    print(f"Job name: {resolved_job_name}")
    if training_pipeline_name:
        print(f"Training pipeline: {training_pipeline_name}")
    else:
        print("Training pipeline: pending server-side resource hydration")
    if training_pipeline_url:
        print(f"Training view: {training_pipeline_url}")
    else:
        print(
            "Training view: "
            f"https://console.cloud.google.com/vertex-ai/locations/{resolved_region}/training/custom-jobs?project={vertex_settings.PROJECT_ID}"
        )
    print(f"Image URI: {vertex_settings.IMAGE_URI}")
    print(f"Output directory: {resolved_output_dir}")
    print(
        "Custom jobs view: "
        f"https://console.cloud.google.com/vertex-ai/locations/{resolved_region}/training/custom-jobs?project={vertex_settings.PROJECT_ID}"
    )
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    typer.run(main)
