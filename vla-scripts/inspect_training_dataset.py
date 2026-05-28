#!/usr/bin/env python3

import sys
try:
    import PIL.Image
    if not hasattr(PIL.Image, "Resampling"):
        class _Resampling:
            NEAREST = 0
            LANCZOS = 1
            BILINEAR = 2
            BICUBIC = 3
            BOX = 4
            HAMMING = 5
        PIL.Image.Resampling = _Resampling
except Exception:
    pass

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

_OMNIVLA_ROOT = Path(__file__).resolve().parent.parent
if str(_OMNIVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(_OMNIVLA_ROOT))

from omnivla_training.episode_manifest import load_episode_subset


def _load_processor_or_fallback(vla_path):
    """
    Returns (base_tokenizer, image_transform, action_tokenizer).
    """
    try:
        from transformers import AutoConfig, AutoImageProcessor, AutoProcessor
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
        try:
            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        except Exception:
            pass
        processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
        from prismatic.vla.action_tokenizer import ActionTokenizer
        action_tokenizer = ActionTokenizer(processor.tokenizer)
        return processor.tokenizer, processor.image_processor.apply_transform, action_tokenizer
    except Exception:
        pass  

    print("  (Full processor not available, using minimal tokenizer + transform)")
    from transformers import AutoTokenizer
    import torch
    from torchvision import transforms
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    size = 224
    _trans = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    def image_transform(pil_image):
        t = _trans(pil_image)
        return t.unsqueeze(0) if t.dim() == 3 else t
    from prismatic.vla.action_tokenizer import ActionTokenizer
    action_tokenizer = ActionTokenizer(tokenizer)
    return tokenizer, image_transform, action_tokenizer


def _show_sample(sample, label):
    """Print the content of a sample (keys, shapes, and a summary of values)."""
    print(f"\n--- {label} ---")
    for k, v in sample.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={v.shape}, dtype={getattr(v, 'dtype', type(v))}")
        elif hasattr(v, "__len__") and not isinstance(v, str):
            print(f"  {k}: len={len(v)}")
        else:
            print(f"  {k}: {v!r}")
    print()


def _load_yaml_config(config_path: Optional[Path]) -> Dict[str, Any]:
    if config_path is None:
        return {}
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    import yaml

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return payload


def _resolve_path(path_value: Optional[str], config_path: Optional[Path]) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    candidates = []
    if config_path is not None:
        candidates.append(config_path.parent / path)
    candidates.extend([_OMNIVLA_ROOT / path, Path.cwd() / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def main():
    parser = argparse.ArgumentParser(description="Inspect the dataset described by the canonical OmniVLA training YAML.")
    parser.add_argument("--config", type=Path, default=Path("config_nav/train_omnivla.yaml"), help="Training YAML to inspect.")
    parser.add_argument("--src-dir", type=Path, help="Path to local dataset root.")
    parser.add_argument("--repo-id", type=str, help="Optional HF dataset repo id to resolve into the local cache.")
    parser.add_argument("--revision", type=str, help="Dataset revision when using --repo-id.")
    parser.add_argument("--episodes-file", type=Path, help="Optional JSON file with `clean_episodes` or a raw episode list.")
    parser.add_argument("--sample-idx", type=int, default=-1, help="Sample index to inspect. Default: last sample.")
    parser.add_argument("--vla-path", type=str, help="HF model or local path used to load the processor.")
    args = parser.parse_args()

    try:
        config = _load_yaml_config(args.config)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    dataset_cfg = config.get("dataset", {}) if isinstance(config.get("dataset", {}), dict) else {}
    model_cfg = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}

    src_dir = args.src_dir or _resolve_path(dataset_cfg.get("root"), args.config)
    repo_id = args.repo_id or dataset_cfg.get("repo_id")
    revision = args.revision or dataset_cfg.get("revision", "main")
    vla_path = args.vla_path or model_cfg.get("vla_path", "NHirose/omnivla-original")

    if src_dir is None and repo_id is None:
        print("Error: provide either --src-dir, --repo-id, or a config with dataset.root/dataset.repo_id")
        return 1
    if src_dir is None:
        from huggingface_hub import snapshot_download

        src_dir = Path(snapshot_download(repo_id, repo_type="dataset", revision=revision))

    if not src_dir.is_dir():
        print(f"Error: directory does not exist: {src_dir}")
        return 1

    selected_episodes = None
    episodes_file = args.episodes_file or _resolve_path(dataset_cfg.get("episodes_file"), args.config)
    if episodes_file is not None:
        try:
            selected_episodes = load_episode_subset(episodes_file)
        except Exception as exc:
            print(f"Error: {exc}")
            return 1

    print("Loading processor (or fallback tokenizer+transform)...")
    from prismatic.vla.datasets import KiwiBotDatasetComplete
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder

    base_tokenizer, image_transform, action_tokenizer = _load_processor_or_fallback(vla_path)

    print("Creating dataset in OmniVLA format (KiwiBotDataset)...")
    dataset = KiwiBotDatasetComplete(
        src_dir=src_dir,
        episodes=selected_episodes,
        action_tokenizer=action_tokenizer,
        base_tokenizer=base_tokenizer,
        image_transform=image_transform,
        prompt_builder_fn=PurePromptBuilder,
        predict_stop_token=True,
        context_size=int(dataset_cfg.get("context_size", 5)),
        action_spacing=int(dataset_cfg.get("action_spacing", 1)),
        mbra_image_size=tuple(dataset_cfg.get("aux_image_size", dataset_cfg.get("mbra_image_size", [96, 96]))),
        metric_waypoint_spacing=float(dataset_cfg.get("metric_waypoint_spacing", 0.1)),
        action_key=dataset_cfg.get("action_key", "action"),
        dataset_name=dataset_cfg.get("name", "local_dataset"),
        default_language_instruction=dataset_cfg.get("default_language_instruction", "reach the goal image"),
        modality_id_with_pose=int(dataset_cfg.get("modality_id_with_pose", 8)),
        modality_id_without_pose=int(dataset_cfg.get("modality_id_without_pose", 6)),
        video_backend=dataset_cfg.get("video_backend", "pyav"),
        tolerance_s=dataset_cfg.get("tolerance_s"),
        max_decode_retries=int(dataset_cfg.get("max_decode_retries", 8)),
    )

    n = len(dataset)
    print(f"  len(dataset) = {n}")
    if n == 0:
        print("  No valid samples (episodes with at least NUM_ACTIONS_CHUNK steps).")
        return 0

    sample_idx = args.sample_idx if args.sample_idx >= 0 else n - 1
    if sample_idx >= n:
        print(f"Error: sample_idx {sample_idx} is out of range for dataset of length {n}")
        return 1

    print(f"Getting sample dataset[{sample_idx}]...")
    sample = dataset[sample_idx]
    _show_sample(sample, f"Sample dataset[{sample_idx}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
