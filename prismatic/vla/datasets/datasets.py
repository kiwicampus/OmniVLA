"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import utm
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, STOP_INDEX

# RLDS imports are lazy (inside RLDSDataset / EpisodicRLDSDataset) so that loading KiwiBotDatasetComplete
# or other non-RLDS code does not require dlimp/tensorflow RLDS stack.

@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, current_action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        actions = rlds_batch["action"]
        
        #print("img", img)
        #print("lang", lang)
        #print("actions", actions.shape, len(action_chunk_string), actions)
        
        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")

        # Get future action chunk
        future_actions = rlds_batch["action"][1:]
        future_actions_string = ''.join(self.action_tokenizer(future_actions))

        # Get action chunk string
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)
        #print("actions", actions.shape, len(action_chunk_string), actions)
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        #print("prompt_builder.get_prompt()", prompt_builder.get_prompt())
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        #print("all", self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True))
        #print("input_ids", input_ids)
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        #print("check", input_ids.size(), labels.size())
        #print(img.size)
        pixel_values = self.image_transform(img)
        #print(pixel_values.size())

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
        #print("labels", -(action_chunk_len + 1), labels)
        #print("In dataloader", labels.size())
        return_dict = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name, actions=actions)

        # Add additional inputs
        if self.use_wrist_image:
            all_wrist_pixels = []
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    pixel_values_wrist = self.image_transform(img_wrist)
                    all_wrist_pixels.append(pixel_values_wrist)
            return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"]
            return_dict["proprio"] = proprio

        return return_dict


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        from prismatic.vla.datasets.rlds import make_interleaved_dataset
        from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights

        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        if "aloha" in self.data_mix:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        else:
            load_camera_views = ("primary", "wrist")

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=NUM_ACTIONS_CHUNK-1,      # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        from prismatic.vla.datasets.rlds import make_interleaved_dataset
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        from prismatic.vla.datasets.rlds import make_single_dataset

        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

# Our dataset

def _parse_lerobot_step(data_item: Dict[str, Any]) -> Tuple[Dict[str, Any], np.ndarray, str]:
    """
    Parse one step from a LeRobot data_item into observation dict, action array, and language_instruction.
    Mirrors lerobot2rlds.parse_step: images (C,H,W) [0,1] -> (H,W,C) uint8; actions and task extracted.
    """
    observation_info = {}
    for k, v in data_item.items():
        if "observation.image" in k and "depth" not in k:
            # LeRobot image is (C, H, W) in [0, 1]
            arr = np.asarray(v) if not hasattr(v, "numpy") else v.numpy()
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            observation_info[k.split(".")[-1]] = np.transpose(arr, (1, 2, 0))
        elif "observation.state" in k:
            key = "_".join(k.split(".")[2:]) or k.split(".")[-1]
            # Observaciones pueden ser numéricas (tensor/array) o str (ej. surface, weather, time_of_day)
            if isinstance(v, str):
                observation_info[key] = np.array([v], dtype=object)
            elif hasattr(v, "numpy"):
                observation_info[key] = v.numpy()
            else:
                observation_info[key] = np.asarray(v)

    action_info = {}
    for k, v in data_item.items():
        if "action" in k:
            key = "_".join(k.split(".")[2:]) or k.split(".")[-1]
            action_info[key] = v.numpy() if hasattr(v, "numpy") else np.asarray(v)
    action_arr = (
        list(action_info.values())[0]
        if len(action_info) == 1
        else np.concatenate([np.atleast_1d(v) for v in action_info.values()])
    )
    action_arr = np.atleast_1d(np.asarray(action_arr, dtype=np.float32))

    lang = data_item.get("task", "")
    if hasattr(lang, "decode"):
        lang = lang.decode()
    language_instruction = str(lang) if lang else ""

    return observation_info, action_arr, language_instruction


def convert_velocity_chunk_to_waypoints(
    velocity_chunk: np.ndarray,
    dt: float = 0.1,
    metric_waypoint_spacing: float = 0.1,
) -> np.ndarray:
    """
    Convert a chunk of (linear_vel, angular_vel) into a chunk of waypoints (x_norm, y_norm, cos(theta), sin(theta))
    in robot local frame. Step 0 = current pose (0, 0, 1, 0); steps 1..7 = pose after 1..7 dt.

    velocity_chunk: shape (NUM_ACTIONS_CHUNK, 2), each row (linear_vel, angular_vel). dt = 100ms between steps.
    Returns: shape (NUM_ACTIONS_CHUNK, 4). waypoint[0] = (0, 0, 1, 0); waypoint[k] for k>=1 = pose after k dt.
    """
    n = velocity_chunk.shape[0]
    velocity_chunk = np.asarray(velocity_chunk, dtype=np.float64)
    v_lin = velocity_chunk[:, 0]
    v_ang = velocity_chunk[:, 1]

    # x[0],y[0],theta[0] = current pose (0,0,0). x[k],y[k],theta[k] = pose after applying (v_lin[k-1], v_ang[k-1]) for dt.
    x = np.zeros(n + 1, dtype=np.float64)
    y = np.zeros(n + 1, dtype=np.float64)
    theta = np.zeros(n + 1, dtype=np.float64)
    for k in range(n):
        theta[k + 1] = theta[k] + v_ang[k] * dt
        x[k + 1] = x[k] + v_lin[k] * math.cos(theta[k]) * dt
        y[k + 1] = y[k] + v_lin[k] * math.sin(theta[k]) * dt

    waypoints = np.zeros((n, 4), dtype=np.float32)
    # Step 0 = current pose (no dt applied)
    waypoints[0, 0] = 0.0
    waypoints[0, 1] = 0.0
    waypoints[0, 2] = 1.0
    waypoints[0, 3] = 0.0
    
    for k in range(1, n):
        waypoints[k, 0] = x[k] / metric_waypoint_spacing
        waypoints[k, 1] = y[k] / metric_waypoint_spacing
        waypoints[k, 2] = math.cos(theta[k])
        waypoints[k, 3] = math.sin(theta[k])

    return waypoints


def _get_primary_image_from_observation(observation: Dict[str, Any]) -> np.ndarray:
    """Return primary RGB image (H, W, C) uint8. Prefer 'main', then 'image', then first image key."""
    for key in ("main", "image", "image_0"):
        if key in observation:
            return np.asarray(observation[key], dtype=np.uint8)
    for k, v in observation.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] in (3, 4):
            return np.asarray(v, dtype=np.uint8)
    raise KeyError(f"No image found in observation keys: {list(observation.keys())}")

class KiwiBotDatasetComplete(Dataset):
    """
    Map-style Dataset that loads LeRobot-format data and returns the same structure as Dummy_Dataset
    (dummy_dataset.py): pixel_values, pixel_values_goal, input_ids, labels, dataset_name, modality_id,
    actions, action_select_mask, goal_pose, obj_pose_norm, img_PIL, gimg_PIL,
    cur_image, goal_image_8, temp_dist, lan_prompt.
    Uses last frame of the episode as goal image for pixel_values_goal / goal_image_8.
    """

    def __init__(
        self,
        src_dir: Path,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        predict_stop_token: bool = True,
        image_size: int = 224,
        context_size: int = 1,
        mbra_image_size: Tuple[int, int] = (96, 96),
        action_spacing: int = 1,
    ):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

        self.src_dir = Path(src_dir)
        self.context_size = context_size  #Number of past frames to include in cur_image for MBRA-style inputs
        self.action_spacing = action_spacing  # Step between consecutive future actions (1=consecutive frames, 3=every 3rd frame, etc.)
        self.action_tokenizer = action_tokenizer #Action tokenizer to convert action arrays to strings (for prompts)
        self.base_tokenizer = base_tokenizer #Tokenizer for language instructions and prompt construction
        self.image_transform = image_transform #Image transform to get pixel_values for current and goal images
        self.prompt_builder_fn = prompt_builder_fn #Function to create PromptBuilder instances for constructing language prompts
        self.predict_stop_token = predict_stop_token #Whether to include the stop token in the labels for loss calculation (if False, stop token is ignored)
        self.image_size = image_size #Resolution for current and goal images after transform (pixel_values_current, pixel_values_goal)
        self.mbra_image_size = tuple(mbra_image_size) #Resolution for images in cur_image and goal_image_8 used for MBRA-style inputs (smaller for efficiency)
        self._LeRobotDataset = LeRobotDataset 
        self._LeRobotDatasetMetadata = LeRobotDatasetMetadata

        meta = LeRobotDatasetMetadata("", root=self.src_dir)
        episodes = meta.episodes
        self._episode_lengths = {}

        for i in range(len(episodes)):
            row = episodes[i]
            ep_id = int(row.get("episode_index", row.get("index", i)))
            length = row.get("length")
            if length is not None:
                self._episode_lengths[ep_id] = int(length)
            print("Loaded episode", ep_id, "length", length)

        self._episode_ids = sorted(self._episode_lengths.keys())

        self._index: List[Tuple[int, int]] = []
        for ep_idx in self._episode_ids:
            L = self._episode_lengths[ep_idx]
            # Include any frame that has at least 8 future steps. If 8*action_spacing fit, use spacing; else fallback to 8 consecutive (spacing 1).
            max_start = max(0, L - (NUM_ACTIONS_CHUNK - 1))  # need frame_idx+7 valid, so frame_idx in 0..L-8
            for frame_idx in range(max_start):
                self._index.append((ep_idx, frame_idx))

    def __len__(self) -> int:
        return len(self._index)

    def _resize_norm(self, image_tensor: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """Resize image tensor (C,H,W) to size for MBRA-style inputs."""
        return TF.resize(image_tensor, size)

    @staticmethod
    def _calculate_relative_position(x_a: float, y_a: float, x_b: float, y_b: float) -> Tuple[float, float]:
        """Delta from (x_a, y_a) to (x_b, y_b) in UTM."""
        return x_b - x_a, y_b - y_a

    @staticmethod
    def _rotate_to_local_frame(delta_x: float, delta_y: float, heading_rad: float) -> Tuple[float, float]:
        """Rotate (delta_x, delta_y) from world to robot local frame (same as Dummy_Dataset)."""
        rel_x = delta_x * math.cos(heading_rad) + delta_y * math.sin(heading_rad)
        rel_y = -delta_x * math.sin(heading_rad) + delta_y * math.cos(heading_rad)
        return rel_x, rel_y

    def _goal_pose_from_waypoints(
        self,
        current_lon: float,
        current_lat: float,
        goal_lon: float,
        goal_lat: float,
        current_compass_deg: float = 0.0,
        goal_compass_deg: float = 0.0,
        thres_dist: float = 30.0,
        metric_waypoint_spacing: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute goal_pose [X, Y, cos(yaw), sin(yaw)] and obj_pose_norm [X, Y] in local frame,
        same convention as Dummy_Dataset (dummy_dataset.py).
        """
        cur_utm = utm.from_latlon(current_lat, current_lon)
        goal_utm = utm.from_latlon(goal_lat, goal_lon)
        cur_compass = -float(current_compass_deg) / 180.0 * math.pi
        goal_compass = -float(goal_compass_deg) / 180.0 * math.pi

        delta_x, delta_y = self._calculate_relative_position(
            cur_utm[0], cur_utm[1], goal_utm[0], goal_utm[1]
        )
        relative_x, relative_y = self._rotate_to_local_frame(delta_x, delta_y, cur_compass)
        radius = np.sqrt(relative_x ** 2 + relative_y ** 2)
        if radius > thres_dist:
            relative_x *= thres_dist / radius
            relative_y *= thres_dist / radius

        goal_pose = np.array([
            relative_y / metric_waypoint_spacing,
            -relative_x / metric_waypoint_spacing,
            np.cos(goal_compass - cur_compass),
            np.sin(goal_compass - cur_compass),
        ], dtype=np.float32)
        obj_pose_norm = goal_pose[0:2].copy()
        return goal_pose, obj_pose_norm

    def _load_episode_steps(self, episode_index: int) -> List[Dict[str, Any]]:
        """Load one episode and return list of parsed steps, ordered by frame_index."""
        # Use pyav backend to avoid torchcodec (requires PyTorch 2.4+ with register_fake)
        ds = self._LeRobotDataset(
            "", self.src_dir, episodes=[episode_index], video_backend="pyav"
        )
        steps_with_idx = []
        for data_item in ds:
            ep = data_item.get("episode_index", episode_index)
            ep_id = int(ep.item() if hasattr(ep, "item") else ep)
            if ep_id != episode_index:
                continue
            fi = data_item.get("frame_index", len(steps_with_idx))
            frame_index = int(fi.item() if hasattr(fi, "item") else fi)
            obs, action, lang = _parse_lerobot_step(data_item)
            steps_with_idx.append((frame_index, {"observation": obs, "action": action, "language_instruction": lang}))
        steps_with_idx.sort(key=lambda x: x[0])
        return [s for _, s in steps_with_idx]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep_idx, frame_idx = self._index[idx]
        steps = self._load_episode_steps(ep_idx)
        expected_len = self._episode_lengths.get(ep_idx)
        if expected_len is not None and len(steps) != expected_len:
            raise RuntimeError(
                f"Episodio {ep_idx}: metadata dice length={expected_len} pero _load_episode_steps devolvió {len(steps)} steps."
            )

        current_step = steps[frame_idx]
        lang = current_step.get("language_instruction") or ""
        image_current = _get_primary_image_from_observation(current_step["observation"])
        image_pil_current = Image.fromarray(image_current)

        # Goal image = last frame of episode (goal-conditioned)
        last_idx = len(steps) - 1
        goal_step = steps[last_idx]
        image_goal = _get_primary_image_from_observation(goal_step["observation"])
        image_pil_goal = Image.fromarray(image_goal)

        # Save current and goal images for this item
        save_dir = Path("kiwibot_complete") / "images" / str(ep_idx)
        save_dir.mkdir(parents=True, exist_ok=True)
        image_pil_current.save(save_dir / f"{frame_idx}_current.png")
        image_pil_goal.save(save_dir / f"{frame_idx}_goal.png")

        # At least 8 future steps (guaranteed by index). Use action_spacing if enough room; else fallback to 8 consecutive.
        last_action_frame_spaced = frame_idx + (NUM_ACTIONS_CHUNK - 1) * self.action_spacing
        if last_action_frame_spaced < len(steps):
            # 8 actions at frame_idx, frame_idx+action_spacing, ..., frame_idx+7*action_spacing
            chunk = [steps[frame_idx + k * self.action_spacing] for k in range(NUM_ACTIONS_CHUNK)]
            dt_step = 0.1 * self.action_spacing
            effective_spacing = self.action_spacing
        else:
            # Fallback: 8 consecutive frames (spacing 1) when not enough room for full action_spacing
            chunk = [steps[frame_idx + k] for k in range(NUM_ACTIONS_CHUNK)]
            dt_step = 0.1
            effective_spacing = 1

        actions = np.stack([s["action"] for s in chunk], axis=0).astype(np.float32)

        # Debug: print 8 raw actions (velocities) and then 8 actions as pose (waypoints)
        # print("[KiwiBotDatasetComplete] 8 raw actions (velocities) [linear_vel, angular_vel] per step (effective_spacing=%d):" % effective_spacing)
        # for i in range(actions.shape[0]):
        #     print(f"  step {i}: {actions[i].tolist()}")
        # actions = convert_velocity_chunk_to_waypoints(
        #         actions, dt=dt_step, metric_waypoint_spacing=0.1
        #     )
        # print("[KiwiBotDatasetComplete] 8 actions transformed to pose (waypoints) [x_norm, y_norm, cos(theta), sin(theta)]:")
        # for i in range(actions.shape[0]):
        #     print(f"  step {i}: {actions[i].tolist()}")

        # Goal from velocity integration: last waypoint (index 7) in same convention as goal_pose [y_norm, -x_norm, cos, sin]
        goal_pose_from_velocity = np.array([
            actions[7, 1],   # y_norm
            -actions[7, 0],  # -x_norm
            actions[7, 2],  # cos(theta)
            actions[7, 3],  # sin(theta)
        ], dtype=np.float32)

        current_action = actions[0]
        future_actions = actions[1:]
        future_actions_string = "".join(self.action_tokenizer(future_actions))
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = torch.tensor(self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids)
        labels = input_ids.clone()
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        # For MBRA
        image_obs_list = []
        for offset in range(self.context_size + 1):
            hist_idx = max(0, frame_idx - self.context_size + offset)
            step_hist = steps[hist_idx]
            img_hist = _get_primary_image_from_observation(step_hist["observation"])
            pil_hist = Image.fromarray(img_hist)
            t = TF.to_tensor(pil_hist)
            image_obs_list.append(self._resize_norm(t, self.mbra_image_size))
        image_obs = torch.cat(image_obs_list, dim=0)
        goal_image_8 = self._resize_norm(TF.to_tensor(image_pil_goal), self.mbra_image_size)

        pixel_values_current = self.image_transform(image_pil_current)
        pixel_values_goal = self.image_transform(image_pil_goal)

        modality_id = 8
        action_select_mask = torch.tensor(1.0, dtype=torch.float32)
        dataset_name = "kiwibot"

        waypoints_key = "waypoints"  # _parse_lerobot_step stores observation.state.waypoints as "waypoints"
        cur_wp = current_step.get("observation", {}).get(waypoints_key)
        goal_wp = goal_step.get("observation", {}).get(waypoints_key)

        # Chunk-end frame: goal = last frame used in the 8-step velocity chunk (frame_idx + 7*action_spacing)
        goal_frame_chunk_idx = frame_idx + (NUM_ACTIONS_CHUNK - 1) * effective_spacing
        goal_step_chunk = steps[goal_frame_chunk_idx] if goal_frame_chunk_idx < len(steps) else None
        goal_wp_chunk = goal_step_chunk.get("observation", {}).get(waypoints_key) if goal_step_chunk is not None else None

        if (
            cur_wp is not None
            and goal_wp is not None
            and hasattr(cur_wp, "reshape")
            and hasattr(goal_wp, "reshape")
        ):
            cur_flat = np.asarray(cur_wp, dtype=np.float64).reshape(-1)
            goal_flat = np.asarray(goal_wp, dtype=np.float64).reshape(-1)
            if cur_flat.size >= 2 and goal_flat.size >= 2 and not (np.any(np.isnan(cur_flat[:2])) or np.any(np.isnan(goal_flat[:2]))):
                current_lon, current_lat = float(cur_flat[0]), float(cur_flat[1])
                goal_lon, goal_lat = float(goal_flat[0]), float(goal_flat[1])
                goal_pose, obj_pose_norm = self._goal_pose_from_waypoints(
                    current_lon, current_lat, goal_lon, goal_lat,
                    current_compass_deg=0.0,
                    goal_compass_deg=0.0,
                )
            else:
                goal_pose = np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float32)
                obj_pose_norm = np.array([np.nan, np.nan], dtype=np.float32)
        else:
            goal_pose = np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float32)
            obj_pose_norm = np.array([np.nan, np.nan], dtype=np.float32)

        # Temporal distance: number of steps from current to goal frame
        temp_dist = float(max(0, last_idx - frame_idx))

        return dict(
            pixel_values=pixel_values_current,
            pixel_values_goal=pixel_values_goal,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            modality_id=modality_id,
            actions=torch.as_tensor(actions),
            action_select_mask=action_select_mask,
            goal_pose=goal_pose,
            obj_pose_norm=obj_pose_norm,
            img_PIL=image_pil_current,
            gimg_PIL=image_pil_goal,
            cur_image=image_obs,
            goal_image_8=goal_image_8,
            temp_dist=temp_dist,
            lan_prompt=lang,
        )
