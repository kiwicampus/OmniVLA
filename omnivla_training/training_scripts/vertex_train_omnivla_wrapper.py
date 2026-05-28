#!/usr/bin/env python3
"""Vertex AI wrapper that materializes cloud inputs and launches the canonical OmniVLA trainer."""

import argparse
import copy
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml
from google.cloud import storage
from huggingface_hub import login
from omnivla_training.vertex import utils as vertex_utils

ROOT = Path(__file__).resolve().parents[2]


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vertex AI OmniVLA training wrapper")
    parser.add_argument("--config-path", type=str, required=True, help="Path to YAML config (local or gs://)")
    parser.add_argument("--output-dir", type=str, required=True, help="Final output directory (local or gs://)")
    parser.add_argument("--job-name", type=str, required=True, help="Vertex job name")
    parser.add_argument("--local-run-root", type=str, default="/tmp/omnivla_output", help="Local run root")
    parser.add_argument("--dataset-repo-id", type=str, default=None, help="Override dataset repo ID")
    parser.add_argument("--dataset-revision", type=str, default=None, help="Override dataset revision")
    parser.add_argument("--dataset-root", type=str, default=None, help="Override dataset root (local or gs://)")
    parser.add_argument("--model-path", type=str, default=None, help="Override model path (HF repo, local, or gs://)")
    parser.add_argument("--steps", type=int, default=None, help="Override training.max_steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size")
    parser.add_argument(
        "--grad-accumulation-steps",
        type=int,
        default=None,
        help="Override training.grad_accumulation_steps",
    )
    parser.add_argument("--save-freq", type=int, default=None, help="Override checkpoint.save_freq")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataset.num_workers")
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="sdpa",
        help="Override model.attn_implementation",
    )
    parser.add_argument("--nproc-per-node", type=int, default=None, help="torchrun nproc-per-node")
    parser.add_argument("--sync-interval-seconds", type=int, default=600, help="Periodic GCS sync interval")
    parser.add_argument("--keep-local-checkpoints", type=int, default=2, help="How many checkpoint dirs to keep")
    parser.add_argument("--smoke-test", action="store_true", help="Run a single optimizer step")
    return parser.parse_args()


def parse_gcs_uri(uri: str) -> Tuple[str, str]:
    match = re.match(r"^gs://([^/]+)(?:/(.*))?$", uri)
    if not match:
        raise ValueError(f"Invalid GCS URI: {uri}")
    return match.group(1), (match.group(2) or "").strip("/")


def get_storage_client() -> storage.Client:
    project_id = os.getenv("GCP_PROJECT_ID")
    return storage.Client(project=project_id or None)


def download_file_from_gcs(gcs_uri: str, local_path: Path) -> Path:
    bucket_name, blob_name = parse_gcs_uri(gcs_uri)
    client = get_storage_client()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{ts()}] Downloading {gcs_uri} -> {local_path}")
    client.bucket(bucket_name).blob(blob_name).download_to_filename(str(local_path))
    return local_path


def download_prefix_from_gcs(gcs_uri: str, local_dir: Path) -> Path:
    bucket_name, prefix = parse_gcs_uri(gcs_uri)
    prefix = prefix.strip("/")
    normalized_prefix = f"{prefix}/" if prefix else ""
    client = get_storage_client()
    bucket = client.bucket(bucket_name)
    blobs = list(client.list_blobs(bucket, prefix=normalized_prefix))
    if not blobs:
        raise FileNotFoundError(f"No objects found under {gcs_uri}")

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{ts()}] Downloading {gcs_uri} -> {local_dir}")
    for blob in blobs:
        relative_name = blob.name[len(normalized_prefix) :] if normalized_prefix else blob.name
        if not relative_name or relative_name.endswith("/"):
            continue
        destination = local_dir / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(destination))
    return local_dir


def load_yaml_config(config_path: str) -> Dict:
    local_path = Path("/tmp/omnivla_vertex_config.yaml")
    source = download_file_from_gcs(config_path, local_path) if config_path.startswith("gs://") else resolve_existing_path(config_path, must_exist=True)
    with source.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Vertex config must be a YAML mapping.")
    return config


def resolve_existing_path(path_str: str, must_exist: bool = False) -> Path:
    path = Path(path_str)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                ROOT / path,
                Path.cwd() / path,
                Path(__file__).resolve().parent / path,
                Path(__file__).resolve().parent.parent / path,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    if must_exist:
        raise FileNotFoundError(f"Unable to resolve local path `{path_str}` inside the training container.")
    return (ROOT / path).resolve()


def materialize_file(path_str: str, destination_root: Path) -> str:
    if path_str.startswith("gs://"):
        return str(download_file_from_gcs(path_str, destination_root / Path(path_str).name))
    return str(resolve_existing_path(path_str, must_exist=True))


def materialize_directory(path_str: str, destination_root: Path) -> str:
    if path_str.startswith("gs://"):
        return str(download_prefix_from_gcs(path_str, destination_root))
    return str(resolve_existing_path(path_str, must_exist=True))


def resolve_model_path(model_path: str) -> str:
    if model_path.startswith("gs://"):
        return str(download_prefix_from_gcs(model_path, Path("/tmp/omnivla_model")))

    path = Path(model_path)
    if path.is_absolute() and path.exists():
        return str(path.resolve())
    if not path.is_absolute():
        for candidate in [ROOT / path, Path.cwd() / path]:
            if candidate.exists():
                return str(candidate.resolve())
    return model_path


def disable_wandb_if_needed(config: Dict) -> None:
    logging_cfg = config.setdefault("logging", {})
    wandb_cfg = logging_cfg.setdefault("wandb", {})
    if os.getenv("WANDB_API_KEY"):
        return
    if wandb_cfg.get("enable", False):
        print(f"[{ts()}] WANDB_API_KEY not set; disabling W&B logging for this run.")
    wandb_cfg["enable"] = False


def maybe_materialize_secret(secret_id: str, version_id: str, output_env_var: str) -> None:
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id or not secret_id or not version_id or os.getenv(output_env_var):
        return

    try:
        success, payload = vertex_utils.access_secret_version(secret_id, version_id, project_id)
    except Exception as exc:
        print(f"[{ts()}] Skipping secret `{secret_id}`: {exc}")
        return

    if not success:
        print(f"[{ts()}] Skipping secret `{secret_id}` due to checksum failure.")
        return

    os.environ[output_env_var] = payload.decode("utf-8")
    print(f"[{ts()}] Loaded secret `{secret_id}` into {output_env_var}.")


def materialize_runtime_secrets() -> None:
    maybe_materialize_secret(
        secret_id=os.getenv("VERTEX_WANDB_SECRET", ""),
        version_id=os.getenv("VERTEX_WANDB_SECRET_VERSION", ""),
        output_env_var="WANDB_API_KEY",
    )
    maybe_materialize_secret(
        secret_id=os.getenv("VERTEX_HF_TOKEN_SECRET", ""),
        version_id=os.getenv("VERTEX_HF_TOKEN_SECRET_VERSION", ""),
        output_env_var="HF_TOKEN",
    )


def prepare_runtime_config(config: Dict, args: argparse.Namespace) -> Dict:
    runtime_cfg = copy.deepcopy(config)

    dataset_cfg = runtime_cfg.setdefault("dataset", {})
    model_cfg = runtime_cfg.setdefault("model", {})
    training_cfg = runtime_cfg.setdefault("training", {})
    checkpoint_cfg = runtime_cfg.setdefault("checkpoint", {})

    checkpoint_cfg["run_root_dir"] = args.local_run_root
    model_cfg["attn_implementation"] = args.attn_implementation or model_cfg.get("attn_implementation", "sdpa")

    if args.dataset_root is not None:
        dataset_cfg["root"] = materialize_directory(args.dataset_root, Path("/tmp/omnivla_dataset"))
        dataset_cfg["repo_id"] = None
        dataset_cfg["local_files_only"] = True
    elif dataset_cfg.get("root"):
        dataset_cfg["root"] = materialize_directory(str(dataset_cfg["root"]), Path("/tmp/omnivla_dataset"))
        dataset_cfg["local_files_only"] = True

    if args.dataset_repo_id is not None:
        dataset_cfg["root"] = None
        dataset_cfg["local_files_only"] = False
        dataset_cfg["repo_id"] = args.dataset_repo_id
    if args.dataset_revision is not None:
        dataset_cfg["revision"] = args.dataset_revision

    for key in ("episodes_file",):
        value = dataset_cfg.get(key)
        if value:
            dataset_cfg[key] = materialize_file(str(value), Path("/tmp/episode_subsets"))

    if args.model_path is not None:
        model_cfg["vla_path"] = resolve_model_path(args.model_path)
    elif model_cfg.get("vla_path"):
        model_cfg["vla_path"] = resolve_model_path(str(model_cfg["vla_path"]))

    if args.steps is not None:
        training_cfg["max_steps"] = args.steps
    if args.batch_size is not None:
        training_cfg["batch_size"] = args.batch_size
    if args.grad_accumulation_steps is not None:
        training_cfg["grad_accumulation_steps"] = args.grad_accumulation_steps
    if args.save_freq is not None:
        checkpoint_cfg["save_freq"] = args.save_freq
    if args.num_workers is not None:
        dataset_cfg["num_workers"] = args.num_workers

    disable_wandb_if_needed(runtime_cfg)
    return runtime_cfg


def write_runtime_artifacts(config: Dict, args: argparse.Namespace) -> Path:
    run_root = Path(args.local_run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    config_path = run_root / "runtime_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    metadata = {
        "job_name": args.job_name,
        "output_dir": args.output_dir,
        "smoke_test": args.smoke_test,
        "nproc_per_node": args.nproc_per_node,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    (run_root / "vertex_job_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return config_path


def authenticate_huggingface() -> None:
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print(f"[{ts()}] HF_TOKEN not set; continuing without Hugging Face login.")
        return

    print(f"[{ts()}] Logging in to Hugging Face...")
    login(token=hf_token, add_to_git_credential=False)
    print(f"[{ts()}] Hugging Face authentication successful.")


def print_gpu_info() -> None:
    print(f"[{ts()}] --- GPU INFO ---")
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            print(result.stdout.strip())
    except Exception as exc:
        print(f"[{ts()}] nvidia-smi unavailable: {exc}")

    try:
        import torch

        print(f"[{ts()}] torch={torch.__version__} cuda={torch.version.cuda}")
        print(f"[{ts()}] cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available inside the Vertex container.")
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            print(f"[{ts()}] GPU {idx}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    except Exception as exc:
        print(f"[{ts()}] GPU validation failed: {exc}")
        sys.exit(1)
    print(f"[{ts()}] --- END GPU INFO ---")


def detect_nproc_per_node(explicit_value: Optional[int]) -> int:
    if explicit_value is not None:
        return max(1, explicit_value)

    try:
        import torch

        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


def build_training_command(runtime_config_path: Path, nproc_per_node: int, smoke_test: bool) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes",
        "1",
        "--nproc-per-node",
        str(nproc_per_node),
        str(ROOT / "vla-scripts" / "train_omnivla.py"),
        "--config",
        str(runtime_config_path),
    ]
    if smoke_test:
        command.append("--smoke-test")
    return command


def sync_local_dir_to_gcs(local_dir: Path, gcs_dir: str, manifest: Dict[str, Tuple[int, int]]) -> Dict[str, Tuple[int, int]]:
    if not gcs_dir.startswith("gs://") or not local_dir.exists():
        return manifest

    bucket_name, prefix = parse_gcs_uri(gcs_dir)
    prefix = prefix.strip("/")
    client = get_storage_client()
    bucket = client.bucket(bucket_name)
    updated_manifest = dict(manifest)

    print(f"[{ts()}] Syncing {local_dir} -> {gcs_dir}")
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue

        relative_path = path.relative_to(local_dir).as_posix()
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        if manifest.get(relative_path) == signature:
            continue

        blob_name = "/".join(part for part in [prefix, relative_path] if part)
        bucket.blob(blob_name).upload_from_filename(str(path))
        updated_manifest[relative_path] = signature
        print(f"[{ts()}] Uploaded {relative_path}")

    return updated_manifest


def prune_local_checkpoints(run_root: Path, keep_last: int) -> None:
    if keep_last < 0 or not run_root.exists():
        return

    checkpoint_pattern = re.compile(r"--(\d+)_chkpt$")
    checkpoint_dirs = []
    for path in run_root.iterdir():
        if not path.is_dir():
            continue
        match = checkpoint_pattern.search(path.name)
        if match:
            checkpoint_dirs.append((int(match.group(1)), path))

    checkpoint_dirs.sort(key=lambda item: item[0])
    for _, stale_path in checkpoint_dirs[:-keep_last]:
        shutil.rmtree(stale_path, ignore_errors=True)
        print(f"[{ts()}] Pruned local checkpoint {stale_path}")


def ensure_runtime_env() -> None:
    os.environ.setdefault("HF_HOME", "/tmp/huggingface")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/huggingface/hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/huggingface/hub")
    os.environ.setdefault("TORCH_HOME", "/tmp/torch")
    os.environ.setdefault("WANDB_DIR", "/tmp/wandb")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def main() -> None:
    args = parse_args()
    ensure_runtime_env()
    materialize_runtime_secrets()
    print("=" * 60)
    print(f"[{ts()}] OmniVLA Vertex AI Training Wrapper")
    print("=" * 60)

    print_gpu_info()

    print(f"[{ts()}] Loading config from {args.config_path}")
    config = load_yaml_config(args.config_path)
    runtime_config = prepare_runtime_config(config, args)
    nproc_per_node = detect_nproc_per_node(args.nproc_per_node)
    args.nproc_per_node = nproc_per_node
    runtime_config_path = write_runtime_artifacts(runtime_config, args)
    authenticate_huggingface()

    command = build_training_command(runtime_config_path, nproc_per_node, args.smoke_test)
    print(f"[{ts()}] Launch command: {' '.join(command)}")

    process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr, start_new_session=True)
    sync_manifest: Dict[str, Tuple[int, int]] = {}
    last_sync = time.time()
    local_run_root = Path(args.local_run_root)

    def shutdown_training(exit_code: int) -> None:
        nonlocal sync_manifest
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        if args.output_dir.startswith("gs://"):
            sync_manifest = sync_local_dir_to_gcs(local_run_root, args.output_dir, sync_manifest)
        sys.exit(exit_code)

    def handle_sigterm(signum, frame) -> None:
        print(f"[{ts()}] Received signal {signum}; shutting down training.")
        shutdown_training(1)

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    try:
        while process.poll() is None:
            time.sleep(15)
            if args.output_dir.startswith("gs://") and time.time() - last_sync >= args.sync_interval_seconds:
                sync_manifest = sync_local_dir_to_gcs(local_run_root, args.output_dir, sync_manifest)
                prune_local_checkpoints(local_run_root, args.keep_local_checkpoints)
                last_sync = time.time()

        return_code = process.wait()
        if args.output_dir.startswith("gs://"):
            sync_manifest = sync_local_dir_to_gcs(local_run_root, args.output_dir, sync_manifest)

        if return_code == 0:
            print(f"[{ts()}] OmniVLA training completed successfully.")
        else:
            print(f"[{ts()}] OmniVLA training failed with exit code {return_code}.")
        sys.exit(return_code)
    except Exception:
        shutdown_training(1)


if __name__ == "__main__":
    main()
