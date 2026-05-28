#!/usr/bin/env python3
"""
Canonical OmniVLA fine-tuning entry point.

This trainer intentionally has one narrow contract:
- one LeRobot-format dataset at a time
- LoRA fine-tuning of the OmniVLA/OpenVLA backbone
- fresh pose projector and continuous action head
- raw action supervision converted into OmniVLA waypoint actions
- no MBRA training dependency
"""

import argparse
import filecmp
import json
import math
import os
import random
import shutil
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import tqdm
import wandb
import yaml
from accelerate import PartialState
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omnivla_training.episode_manifest import load_episode_subset
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead_idcat
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.util.data_utils import PaddedCollatorForActionPrediction_Nav_MMN
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, IGNORE_INDEX, NUM_ACTIONS_CHUNK, POSE_DIM
from prismatic.vla.datasets import KiwiBotDatasetComplete

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def model_is_on_hf_hub(model_path: str) -> bool:
    if Path(model_path).exists():
        return False
    try:
        HfApi().model_info(model_path)
        return True
    except Exception:
        return False


def update_auto_map(pretrained_checkpoint: str) -> None:
    checkpoint_path = Path(pretrained_checkpoint)
    config_path = checkpoint_path / "config.json"
    if not config_path.exists():
        return

    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_auto_map = {
        "AutoConfig": "configuration_prismatic.OpenVLAConfig",
        "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
    }
    if config.get("auto_map") == expected_auto_map:
        return

    backup_path = checkpoint_path / f"config.json.back.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(config_path, backup_path)
    config["auto_map"] = expected_auto_map
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {config_path}; backup written to {backup_path}")


def check_model_logic_mismatch(pretrained_checkpoint: str) -> None:
    checkpoint_path = Path(pretrained_checkpoint)
    if not checkpoint_path.is_dir():
        return

    files_to_sync = {
        "configuration_prismatic.py": ROOT / "prismatic" / "extern" / "hf" / "configuration_prismatic.py",
        "modeling_prismatic.py": ROOT / "prismatic" / "extern" / "hf" / "modeling_prismatic.py",
    }
    for filename, source_path in files_to_sync.items():
        destination_path = checkpoint_path / filename
        if not source_path.exists():
            print(f"WARNING: cannot sync missing model logic file {source_path}")
            continue
        if destination_path.exists() and filecmp.cmp(source_path, destination_path, shallow=False):
            continue
        if destination_path.exists():
            backup_path = checkpoint_path / f"{filename}.back.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(destination_path, backup_path)
            print(f"Backed up existing {filename} to {backup_path}")
        shutil.copy2(source_path, destination_path)
        print(f"Synced {filename} into {checkpoint_path}")


def should_retry_snapshot_download(exc: Exception) -> bool:
    message = str(exc)
    retry_markers = (
        "Consistency check failed",
        "file should be of size",
        "Connection error",
        "Read timed out",
    )
    return any(marker in message for marker in retry_markers)


def robust_snapshot_download(**snapshot_kwargs: Any) -> str:
    local_files_only = bool(snapshot_kwargs.get("local_files_only"))
    attempts = 1 if local_files_only else 3
    last_exc: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        current_kwargs = dict(snapshot_kwargs)
        if attempt > 1:
            current_kwargs["force_download"] = True

        try:
            return snapshot_download(**current_kwargs)
        except Exception as exc:  # pragma: no cover - exercised in cloud only
            last_exc = exc
            if attempt == attempts or not should_retry_snapshot_download(exc):
                raise
            repo_id = current_kwargs.get("repo_id", "<unknown>")
            print(
                f"snapshot_download failed for {repo_id} on attempt {attempt}/{attempts}: {exc}. "
                "Retrying with force_download=True."
            )

    raise RuntimeError(f"snapshot_download failed unexpectedly: {last_exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune OmniVLA through the canonical single-dataset route.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument("--smoke-test", action="store_true", help="Build the pipeline and run a single optimizer step.")
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {config_path} must be a YAML mapping.")
    required_sections = ("dataset", "model", "training", "checkpoint", "logging")
    missing_sections = [section for section in required_sections if section not in cfg]
    if missing_sections:
        raise ValueError(f"Missing required config section(s): {', '.join(missing_sections)}")
    return cfg


def resolve_local_path(path_str: Optional[str], config_path: Path) -> Optional[Path]:
    if not path_str:
        return None

    path = Path(path_str)
    if path.is_absolute():
        return path

    candidates = [
        config_path.parent / path,
        ROOT / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def split_episodes(dataset_cfg: Dict[str, Any], config_path: Path) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    episodes_file = resolve_local_path(dataset_cfg.get("episodes_file"), config_path)
    if not episodes_file:
        return None, None

    all_eps = load_episode_subset(episodes_file)
    val_ratio = float(dataset_cfg.get("val_ratio", 0.0))
    if val_ratio <= 0.0:
        return all_eps, None

    split_seed = int(dataset_cfg.get("split_seed", 7))
    shuffled = list(all_eps)
    random.Random(split_seed).shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio))
    val_eps = sorted(shuffled[:val_count])
    train_eps = sorted(shuffled[val_count:])
    if not train_eps:
        raise ValueError("Episode split produced an empty training set; reduce `val_ratio`.")
    return train_eps, val_eps


def resolve_dataset_root(dataset_cfg: Dict[str, Any], config_path: Path) -> Path:
    root = resolve_local_path(dataset_cfg.get("root"), config_path)
    if root is not None:
        return root

    repo_id = dataset_cfg.get("repo_id")
    if not repo_id:
        raise ValueError("Set either `dataset.root` or `dataset.repo_id` in the config.")

    snapshot_kwargs = {
        "repo_id": repo_id,
        "repo_type": "dataset",
    }
    if dataset_cfg.get("revision"):
        snapshot_kwargs["revision"] = dataset_cfg["revision"]
    if dataset_cfg.get("local_files_only"):
        snapshot_kwargs["local_files_only"] = True

    return Path(robust_snapshot_download(**snapshot_kwargs))


def ddp_barrier() -> None:
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()


def wrap_ddp(module: torch.nn.Module, device_id: int) -> torch.nn.Module:
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        return DDP(module, device_ids=[device_id], find_unused_parameters=False, gradient_as_bucket_view=True)
    return module


def unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if hasattr(module, "module") else module


def count_trainable_parameters(module: torch.nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def build_lr_scheduler(optimizer: AdamW, training_cfg: Dict[str, Any]) -> LambdaLR:
    scheduler_type = str(training_cfg.get("scheduler", "cosine")).lower()
    warmup_steps = max(0, int(training_cfg.get("warmup_steps", 0)))
    max_steps = max(1, int(training_cfg["max_steps"]))
    min_lr_ratio = float(training_cfg.get("min_lr_ratio", 0.1))
    min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)

        if scheduler_type == "constant":
            return 1.0

        decay_total = max(1, max_steps - warmup_steps)
        progress = min(1.0, max(0.0, float(current_step - warmup_steps) / float(decay_total)))

        if scheduler_type == "linear":
            return max(min_lr_ratio, 1.0 - (1.0 - min_lr_ratio) * progress)

        if scheduler_type != "cosine":
            raise ValueError(f"Unsupported scheduler `{scheduler_type}`. Expected one of: constant, linear, cosine.")

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def get_cuda_device(device_id: int) -> torch.device:
    return torch.device(f"cuda:{device_id}")


def finalize_batch(batch: Dict[str, Any], pad_token_id: int, model_max_length: int) -> Dict[str, Any]:
    input_ids = pad_sequence(batch["input_ids"], batch_first=True, padding_value=pad_token_id)
    labels = pad_sequence(batch["labels"], batch_first=True, padding_value=IGNORE_INDEX)

    batch["input_ids"] = input_ids[:, :model_max_length]
    batch["labels"] = labels[:, :model_max_length]
    batch["attention_mask"] = batch["input_ids"].ne(pad_token_id)
    batch["attention_mask_label"] = batch["labels"].ne(IGNORE_INDEX)
    batch["goal_mask_select"] = torch.tensor(batch["modality_id"], dtype=torch.long)
    return batch


def safe_masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 1:
        raise ValueError(f"Expected 1D mask, got shape {tuple(mask.shape)}")
    if not mask.any():
        return pred.new_tensor(0.0)
    return torch.nn.functional.mse_loss(pred[mask], target[mask])


def run_forward_pass(
    vla: torch.nn.Module,
    action_head: torch.nn.Module,
    pose_projector: torch.nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
    num_patches: int,
    obj_loss_weight: float,
    smoothness_loss_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    ground_truth_actions = batch["actions"].to(device=device, dtype=torch.bfloat16)
    modality_id = batch["goal_mask_select"].to(device=device, dtype=torch.long).reshape(-1)
    goal_pose = batch["goal_pose"].to(device=device, dtype=torch.bfloat16)
    obj_pose_norm = batch["obj_pose_norm"].to(device=device, dtype=torch.bfloat16)
    temp_dist = torch.as_tensor(batch["temp_dist"], device=device, dtype=torch.float32).reshape(-1)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            attention_mask_label=batch["attention_mask_label"].to(device),
            pixel_values=batch["pixel_values"].to(device=device, dtype=torch.bfloat16),
            modality_id=modality_id.to(dtype=torch.bfloat16),
            labels=batch["labels"].to(device),
            output_hidden_states=True,
            proprio=goal_pose,
            proprio_projector=pose_projector,
            use_film=False,
        )

        ground_truth_token_ids = batch["labels"][:, 1:].to(device)
        current_action_mask = get_current_action_mask(ground_truth_token_ids)
        next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

        last_hidden_states = output.hidden_states[-1]
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        batch_size = batch["input_ids"].shape[0]
        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )
        predicted_actions = unwrap(action_head).predict_action(
            actions_hidden_states,
            modality_id.to(dtype=torch.bfloat16),
        )

        action_loss = torch.nn.functional.mse_loss(predicted_actions, ground_truth_actions)
        action_l1 = torch.nn.functional.l1_loss(predicted_actions, ground_truth_actions)
        final_action_l1 = torch.nn.functional.l1_loss(predicted_actions[:, -1], ground_truth_actions[:, -1])
        final_xy_l1 = torch.nn.functional.l1_loss(predicted_actions[:, -1, 0:2], ground_truth_actions[:, -1, 0:2])
        smoothness_loss = torch.nn.functional.mse_loss(predicted_actions[:, :-1], predicted_actions[:, 1:])

        # The goal image / goal pose in this pipeline correspond to the final frame of the episode.
        # Supervising the last waypoint in an 8-step chunk against that terminal goal only makes sense
        # when the current frame is already within the chunk horizon of the goal.
        valid_obj_mask = (
            ((modality_id == 7) | (modality_id == 8))
            & torch.isfinite(obj_pose_norm).all(dim=1)
            & (temp_dist <= float(NUM_ACTIONS_CHUNK - 1))
        )
        object_loss = safe_masked_mse(
            predicted_actions[:, -1, 0:2],
            obj_pose_norm,
            valid_obj_mask,
        )

        loss = action_loss + smoothness_loss_weight * smoothness_loss + obj_loss_weight * object_loss

    metrics = {
        "loss": float(loss.detach().cpu()),
        "action_loss": float(action_loss.detach().cpu()),
        "action_l1": float(action_l1.detach().cpu()),
        "final_action_l1": float(final_action_l1.detach().cpu()),
        "final_xy_l1": float(final_xy_l1.detach().cpu()),
        "smoothness_loss": float(smoothness_loss.detach().cpu()),
        "object_loss": float(object_loss.detach().cpu()),
        "object_supervision_samples": float(valid_obj_mask.sum().detach().cpu()),
        "object_supervision_rate": float(valid_obj_mask.to(torch.float32).mean().detach().cpu()),
    }
    return loss, metrics


def save_training_checkpoint(
    cfg: Dict[str, Any],
    run_dir: Path,
    global_step: int,
    vla: torch.nn.Module,
    processor: Any,
    pose_projector: torch.nn.Module,
    action_head: torch.nn.Module,
    distributed_state: PartialState,
) -> None:
    checkpoint_cfg = cfg["checkpoint"]
    if checkpoint_cfg.get("save_latest_only", False):
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(f"{run_dir}--{global_step}_chkpt")
        checkpoint_name_suffix = f"{global_step}_checkpoint.pt"

    adapter_dir = checkpoint_dir / "lora_adapter"

    if distributed_state.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir.mkdir(parents=True, exist_ok=True)
    ddp_barrier()

    if distributed_state.is_main_process:
        processor.save_pretrained(checkpoint_dir)
        unwrap(vla).save_pretrained(adapter_dir)
        portable_base_model = cfg["model"].get("requested_vla_path", cfg["model"]["vla_path"])
        adapter_config_path = adapter_dir / "adapter_config.json"
        if adapter_config_path.exists():
            adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
            adapter_config["base_model_name_or_path"] = portable_base_model
            adapter_config_path.write_text(json.dumps(adapter_config, indent=2) + "\n", encoding="utf-8")
        (checkpoint_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        torch.save(unwrap(pose_projector).state_dict(), checkpoint_dir / f"pose_projector--{checkpoint_name_suffix}")
        torch.save(unwrap(action_head).state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")

    ddp_barrier()

    if checkpoint_cfg.get("merge_lora_during_training", False):
        base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg["model"]["vla_path"],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()
        if distributed_state.is_main_process:
            merged_vla.save_pretrained(checkpoint_dir)
        ddp_barrier()


def load_vla_and_processor(
    model_cfg: Dict[str, Any],
    distributed_state: PartialState,
    device: torch.device,
):
    vla_path = model_cfg["vla_path"].rstrip("/")
    attn_implementation = model_cfg.get("attn_implementation", "sdpa")
    load_from_hub = model_is_on_hf_hub(vla_path)
    if load_from_hub:
        vla_path = robust_snapshot_download(repo_id=vla_path)
    else:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)

    if distributed_state.is_main_process:
        update_auto_map(vla_path)
        check_model_logic_mismatch(vla_path)
    ddp_barrier()

    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)

    if load_from_hub:
        index_file = Path(vla_path) / "model.safetensors.index.json"
        index = json.loads(index_file.read_text(encoding="utf-8"))
        from safetensors.torch import load_file

        state_dict = {}
        for filename in sorted(set(index["weight_map"].values())):
            state_dict.update(load_file(str(Path(vla_path) / filename)))

        config_openvla = AutoConfig.from_pretrained(vla_path, trust_remote_code=True)
        if hasattr(config_openvla, "_attn_implementation"):
            config_openvla._attn_implementation = attn_implementation
        if hasattr(config_openvla, "text_config") and hasattr(config_openvla.text_config, "_attn_implementation"):
            config_openvla.text_config._attn_implementation = attn_implementation
        vla = OpenVLAForActionPrediction_MMNv1(config_openvla)
        vla.load_state_dict(state_dict, strict=False)
    else:
        vla = AutoModelForVision2Seq.from_pretrained(
            vla_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
        )

    vla.vision_backbone.set_num_images_in_input(int(model_cfg.get("num_images_in_input", 2)))
    vla.to(dtype=torch.bfloat16, device=device)
    return vla_path, vla, processor


def build_run_id(cfg: Dict[str, Any], dataset_cfg: Dict[str, Any]) -> str:
    dataset_label = dataset_cfg.get("name") or dataset_cfg.get("repo_id") or Path(dataset_cfg["root"]).name
    dataset_label = str(dataset_label).replace("/", "_")
    model_label = Path(cfg["model"]["vla_path"].rstrip("/")).name
    return (
        f"{model_label}+{dataset_label}"
        f"+b{cfg['training']['batch_size'] * cfg['training'].get('grad_accumulation_steps', 1)}"
        f"+lr-{cfg['training']['learning_rate']}"
    )


def train(cfg: Dict[str, Any], config_path: Path, smoke_test: bool) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OmniVLA training.")

    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    checkpoint_cfg = cfg["checkpoint"]
    logging_cfg = cfg["logging"]

    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    device = get_cuda_device(device_id)
    world_size = distributed_state.num_processes
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    resolved_dataset_root = resolve_dataset_root(dataset_cfg, config_path)
    train_episodes, val_episodes = split_episodes(dataset_cfg, config_path)

    run_root = resolve_local_path(checkpoint_cfg["run_root_dir"], config_path)
    run_id = build_run_id(cfg, dataset_cfg)
    run_dir = run_root / run_id
    if distributed_state.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    if logging_cfg.get("wandb", {}).get("enable", False) and distributed_state.is_main_process:
        wandb.init(
            entity=logging_cfg["wandb"].get("entity"),
            project=logging_cfg["wandb"].get("project", "omnivla-local"),
            name=run_id,
            config=cfg,
        )

    print(f"Dataset root: {resolved_dataset_root}")
    print(f"Train episodes: {'all' if train_episodes is None else len(train_episodes)}")
    print(f"Holdout episodes not used for optimization: {0 if val_episodes is None else len(val_episodes)}")
    print(f"Run dir: {run_dir}")
    print(f"World size: {world_size}, local rank: {device_id}")

    requested_vla_path = model_cfg["vla_path"].rstrip("/")
    resolved_vla_path, vla, processor = load_vla_and_processor(model_cfg, distributed_state, device)
    cfg["model"]["requested_vla_path"] = requested_vla_path
    cfg["model"]["vla_path"] = resolved_vla_path

    if not model_cfg.get("use_lora", True):
        raise ValueError("The canonical OmniVLA trainer saves LoRA adapters; keep `model.use_lora: true`.")

    target_modules = [name for name, module in vla.named_modules() if isinstance(module, torch.nn.Linear)]
    lora_config = LoraConfig(
        r=int(model_cfg.get("lora_rank", 32)),
        lora_alpha=min(int(model_cfg.get("lora_rank", 32)), 16),
        lora_dropout=float(model_cfg.get("lora_dropout", 0.0)),
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora_config)
    vla.print_trainable_parameters()

    vla = wrap_ddp(vla, device_id)
    pose_projector = wrap_ddp(
        ProprioProjector(llm_dim=unwrap(vla).llm_dim, proprio_dim=POSE_DIM).to(device),
        device_id,
    )
    action_head = wrap_ddp(
        L1RegressionActionHead_idcat(
            input_dim=unwrap(vla).llm_dim,
            hidden_dim=unwrap(vla).llm_dim,
            action_dim=ACTION_DIM,
        ).to(device=device, dtype=torch.bfloat16),
        device_id,
    )

    print(f"Trainable VLA params: {count_trainable_parameters(vla)}")
    print(f"Trainable pose projector params: {count_trainable_parameters(pose_projector)}")
    print(f"Trainable action head params: {count_trainable_parameters(action_head)}")

    vla_trainable_params = [param for param in vla.parameters() if param.requires_grad]
    head_trainable_params = [param for param in pose_projector.parameters() if param.requires_grad]
    head_trainable_params += [param for param in action_head.parameters() if param.requires_grad]

    base_learning_rate = float(training_cfg["learning_rate"])
    vla_learning_rate = float(training_cfg.get("vla_learning_rate", base_learning_rate))
    head_learning_rate = float(training_cfg.get("head_learning_rate", base_learning_rate))
    weight_decay = float(training_cfg.get("weight_decay", 0.01))

    optimizer = AdamW(
        [
            {"params": vla_trainable_params, "lr": vla_learning_rate},
            {"params": head_trainable_params, "lr": head_learning_rate},
        ],
        weight_decay=weight_decay,
    )
    scheduler = build_lr_scheduler(optimizer, training_cfg)

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    collator = PaddedCollatorForActionPrediction_Nav_MMN(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
        num_img=int(model_cfg.get("num_images_in_input", 2)),
    )

    train_dataset = KiwiBotDatasetComplete(
        src_dir=resolved_dataset_root,
        episodes=train_episodes,
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
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
    print(f"Training samples: {len(train_dataset)}")
    effective_batch_size = int(training_cfg["batch_size"]) * max(1, world_size)
    if len(train_dataset) < effective_batch_size:
        raise ValueError(
            f"Training set has {len(train_dataset)} samples, which is smaller than the effective batch size "
            f"({effective_batch_size}). Reduce `training.batch_size` or increase the dataset subset."
        )

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=device_id, shuffle=True)
    num_workers = int(dataset_cfg.get("num_workers", 4))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collator,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    num_patches = unwrap(vla).vision_backbone.get_num_patches() * unwrap(vla).vision_backbone.get_num_images_in_input()
    num_patches += 1  # reserved proprio token

    log_freq = int(logging_cfg.get("log_freq", 10))
    metrics_window_steps = max(1, int(logging_cfg.get("metrics_window_steps", log_freq)))
    save_freq = int(checkpoint_cfg.get("save_freq", 1000))
    grad_accumulation_steps = int(training_cfg.get("grad_accumulation_steps", 1))
    max_steps = 1 if smoke_test else int(training_cfg["max_steps"])
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))

    metric_names = (
        "loss",
        "action_loss",
        "action_l1",
        "final_action_l1",
        "final_xy_l1",
        "smoothness_loss",
        "object_loss",
        "object_supervision_samples",
        "object_supervision_rate",
    )
    recent_metrics = {
        key: deque(maxlen=metrics_window_steps)
        for key in (*metric_names, "grad_norm")
    }
    step_metric_sums = {key: 0.0 for key in metric_names}

    vla.train()
    pose_projector.train()
    action_head.train()
    optimizer.zero_grad()

    global_step = 0
    micro_step = 0
    epoch = 0
    sampler.set_epoch(epoch)
    train_iter = iter(train_loader)

    with tqdm.tqdm(total=max_steps, disable=not distributed_state.is_main_process) as progress:
        while global_step < max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = finalize_batch(batch, processor.tokenizer.pad_token_id, processor.tokenizer.model_max_length)
            loss, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                pose_projector=pose_projector,
                batch=batch,
                device=device,
                num_patches=num_patches,
                obj_loss_weight=float(training_cfg.get("object_loss_weight", 0.1)),
                smoothness_loss_weight=float(training_cfg.get("smoothness_loss_weight", 0.1)),
            )

            (loss / grad_accumulation_steps).backward()
            micro_step += 1
            for key in step_metric_sums:
                step_metric_sums[key] += metrics[key]

            if micro_step % grad_accumulation_steps != 0:
                continue

            step_metrics = {
                key: value / float(grad_accumulation_steps)
                for key, value in step_metric_sums.items()
            }
            for key in step_metric_sums:
                step_metric_sums[key] = 0.0

            grad_norm = float("nan")
            if max_grad_norm > 0:
                grad_norm = float(
                    clip_grad_norm_(vla_trainable_params + head_trainable_params, max_grad_norm).detach().cpu()
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            progress.update(1)

            for key, value in step_metrics.items():
                recent_metrics[key].append(value)
            if math.isfinite(grad_norm):
                recent_metrics["grad_norm"].append(grad_norm)

            smooth_metrics = {
                key: sum(values) / len(values)
                for key, values in recent_metrics.items()
                if len(values) > 0
            }
            if distributed_state.is_main_process and (global_step == 1 or global_step % log_freq == 0):
                print(
                    f"[step {global_step}] "
                    f"loss={smooth_metrics['loss']:.4f} "
                    f"action={smooth_metrics['action_loss']:.4f} "
                    f"l1={smooth_metrics['action_l1']:.4f} "
                    f"final_xy_l1={smooth_metrics['final_xy_l1']:.4f} "
                    f"smooth={smooth_metrics['smoothness_loss']:.4f} "
                    f"grad={smooth_metrics['grad_norm']:.4f} "
                    f"obj={smooth_metrics['object_loss']:.4f} "
                    f"obj_rate={smooth_metrics['object_supervision_rate']:.3f}"
                )
                if logging_cfg.get("wandb", {}).get("enable", False):
                    wandb.log(
                        {
                            "train/loss": smooth_metrics["loss"],
                            "train/action_loss": smooth_metrics["action_loss"],
                            "train/action_l1": smooth_metrics["action_l1"],
                            "train/final_action_l1": smooth_metrics["final_action_l1"],
                            "train/final_xy_l1": smooth_metrics["final_xy_l1"],
                            "train/smoothness_loss": smooth_metrics["smoothness_loss"],
                            "train/object_loss": smooth_metrics["object_loss"],
                            "train/object_supervision_samples": smooth_metrics["object_supervision_samples"],
                            "train/object_supervision_rate": smooth_metrics["object_supervision_rate"],
                            "train/grad_norm": smooth_metrics["grad_norm"],
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "train/vla_learning_rate": scheduler.get_last_lr()[0],
                            "train/head_learning_rate": scheduler.get_last_lr()[1],
                        },
                        step=global_step,
                    )

            if global_step > 0 and global_step % save_freq == 0:
                save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    global_step=global_step,
                    vla=vla,
                    processor=processor,
                    pose_projector=pose_projector,
                    action_head=action_head,
                    distributed_state=distributed_state,
                )

    if not smoke_test and global_step > 0 and global_step % save_freq != 0:
        save_training_checkpoint(
            cfg=cfg,
            run_dir=run_dir,
            global_step=global_step,
            vla=vla,
            processor=processor,
            pose_projector=pose_projector,
            action_head=action_head,
            distributed_state=distributed_state,
        )

    if distributed_state.is_main_process and logging_cfg.get("wandb", {}).get("enable", False):
        wandb.finish()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg, args.config.resolve(), args.smoke_test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
