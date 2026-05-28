#!/usr/bin/env python3
"""Check whether the current machine is ready for OmniVLA training."""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omnivla_training.episode_manifest import load_episode_subset, load_structured_file


REQUIRED_CONFIG_SECTIONS = ("dataset", "model", "training", "checkpoint", "logging")
TRAINING_IMPORTS = (
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "huggingface_hub",
    "pandas",
    "av",
    "lerobot",
    "yaml",
)
VERTEX_IMPORTS = (
    "google.cloud.aiplatform",
    "google.cloud.storage",
    "google.cloud.secretmanager",
    "typer",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate OmniVLA training environment and config.")
    parser.add_argument("--config", type=Path, default=Path("config_nav/train_omnivla.yaml"))
    parser.add_argument("--check-vertex", action="store_true", help="Also check Vertex submit dependencies.")
    parser.add_argument(
        "--min-vram-gb",
        type=float,
        default=16.0,
        help="Minimum GPU memory expected for OmniVLA training checks. Use 0 to disable this check.",
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Only validate files/config/manifests; skip Python package and CUDA readiness checks.",
    )
    return parser.parse_args()


def check(condition: bool, ok_message: str, fail_message: str, failures: list[str]) -> None:
    if condition:
        print(f"[OK] {ok_message}")
    else:
        print(f"[FAIL] {fail_message}")
        failures.append(fail_message)


def import_status(module_name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "")
        return True, str(version)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def validate_config(config_path: Path, failures: list[str]) -> dict[str, Any]:
    check(config_path.exists(), f"Config exists: {config_path}", f"Missing config: {config_path}", failures)
    if not config_path.exists():
        return {}

    payload = load_structured_file(config_path)
    check(isinstance(payload, dict), "Config is a mapping", "Config must be a YAML mapping", failures)
    if not isinstance(payload, dict):
        return {}

    missing_sections = [section for section in REQUIRED_CONFIG_SECTIONS if section not in payload]
    check(
        not missing_sections,
        "Required config sections are present",
        f"Missing config sections: {', '.join(missing_sections)}",
        failures,
    )

    model_cfg = payload.get("model", {})
    if isinstance(model_cfg, dict):
        vla_path = model_cfg.get("vla_path")
        check(bool(vla_path), f"Base checkpoint configured: {vla_path}", "model.vla_path is required", failures)
        check(
            model_cfg.get("use_lora", True) is True,
            "LoRA is enabled",
            "model.use_lora must stay true for the canonical trainer",
            failures,
        )

    dataset_cfg = payload.get("dataset", {})
    if isinstance(dataset_cfg, dict):
        root = dataset_cfg.get("root")
        repo_id = dataset_cfg.get("repo_id")
        check(
            bool(root) or bool(repo_id),
            f"Dataset source configured: {'root=' + str(root) if root else 'repo_id=' + str(repo_id)}",
            "Set either dataset.root or dataset.repo_id",
            failures,
        )
        if root:
            check(Path(root).exists(), f"Dataset root exists: {root}", f"Dataset root does not exist: {root}", failures)

        episodes_file = dataset_cfg.get("episodes_file")
        if episodes_file:
            episodes_path = Path(episodes_file)
            if not episodes_path.is_absolute():
                candidates = [config_path.parent / episodes_path, ROOT / episodes_path, Path.cwd() / episodes_path]
                episodes_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
            check(
                episodes_path.exists(),
                f"Episode manifest exists: {episodes_path}",
                f"Episode manifest does not exist: {episodes_path}",
                failures,
            )
            if episodes_path.exists():
                try:
                    episodes = load_episode_subset(episodes_path)
                    check(
                        len(episodes) > 0,
                        f"Episode manifest expands to {len(episodes)} episodes",
                        f"Episode manifest is empty: {episodes_path}",
                        failures,
                    )
                except Exception as exc:
                    msg = f"Episode manifest failed to load: {episodes_path}: {exc}"
                    print(f"[FAIL] {msg}")
                    failures.append(msg)
        else:
            print("[OK] No episode manifest configured; trainer will use all dataset episodes")

    return payload


def validate_packages(check_vertex: bool, min_vram_gb: float, failures: list[str]) -> None:
    for module_name in TRAINING_IMPORTS:
        ok, detail = import_status(module_name)
        check(ok, f"Import {module_name} {detail}".strip(), f"Cannot import {module_name}: {detail}", failures)

    sibling_torchrun = Path(sys.executable).parent / "torchrun"
    torchrun = shutil.which("torchrun") or (str(sibling_torchrun) if sibling_torchrun.exists() else None)
    check(bool(torchrun), f"torchrun available: {torchrun}", "torchrun is not on PATH", failures)

    torch_ok, _ = import_status("torch")
    if torch_ok:
        import torch

        cuda_available = torch.cuda.is_available()
        check(cuda_available, "CUDA is available", "CUDA is not available; training cannot run here", failures)
        print(f"[INFO] torch={torch.__version__} cuda={getattr(torch.version, 'cuda', None)}")
        print(f"[INFO] cuda_device_count={torch.cuda.device_count()}")

        if cuda_available:
            try:
                device_index = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(device_index)
                total_vram_gb = props.total_memory / (1024**3)
                print(f"[INFO] cuda_device={props.name} capability=sm_{props.major}{props.minor}")
                print(f"[INFO] total_vram_gb={total_vram_gb:.2f}")
                if min_vram_gb > 0:
                    check(
                        total_vram_gb >= min_vram_gb,
                        f"GPU memory is at least {min_vram_gb:g} GB",
                        (
                            f"GPU has {total_vram_gb:.2f} GB VRAM; OmniVLA 7B training is likely to OOM "
                            f"below {min_vram_gb:g} GB"
                        ),
                        failures,
                    )

                probe = torch.ones(1, device="cuda")
                torch.cuda.synchronize()
                check(
                    float(probe.item()) == 1.0,
                    "CUDA kernel probe succeeded",
                    "CUDA kernel probe failed",
                    failures,
                )
            except Exception as exc:
                msg = f"CUDA kernel probe failed: {type(exc).__name__}: {exc}"
                print(f"[FAIL] {msg}")
                failures.append(msg)

    if check_vertex:
        for module_name in VERTEX_IMPORTS:
            ok, detail = import_status(module_name)
            check(ok, f"Import {module_name} {detail}".strip(), f"Cannot import {module_name}: {detail}", failures)


def main() -> int:
    args = parse_args()
    failures: list[str] = []

    print(f"[INFO] python={sys.version.split()[0]} executable={sys.executable}")
    validate_config(args.config, failures)
    if not args.static_only:
        validate_packages(args.check_vertex, args.min_vram_gb, failures)

    if failures:
        print("\nEnvironment is NOT ready:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nEnvironment/config checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
