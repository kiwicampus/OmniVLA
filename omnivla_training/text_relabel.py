"""Per-frame text labels from the recorded turn ahead, drawn from the dataset's own sentences.

Pure numpy and pyarrow, so it runs without torch or the dataset class.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq

WINDOW_FRAMES = 50  # 5 s at 10 Hz
TURN_DEG = 30.0
STRAIGHT_DEG = 10.0
POOL_SIZE = 5


def text_class(text: str) -> Optional[str]:
    """left, right or straight from an instruction's words; None when mixed or unclear."""
    low = text.lower()
    left, right = "left" in low, "right" in low
    if left != right:
        return "left" if left else "right"
    if not left and "straight" in low:
        return "straight"
    return None


def motion_class(rows, idx: int, window: int = WINDOW_FRAMES, dt: float = 0.1, key: str = "action") -> Optional[str]:
    """left, right or straight from the heading change over `window` frames from `idx`; None in between."""
    end = min(idx + window, len(rows))
    rates = [float(np.reshape(np.asarray(rows[i][key], float), -1)[1]) for i in range(idx, end)]
    change = math.degrees(sum(rates) * dt)
    if change >= TURN_DEG:
        return "left"
    if change <= -TURN_DEG:
        return "right"
    if abs(change) < STRAIGHT_DEG:
        return "straight"
    return None


def label_pools(root: Path, size: int = POOL_SIZE) -> Dict[str, List[str]]:
    """The `size` most frequent instructions per class, counted over frames."""
    tasks = pq.read_table(root / "meta" / "tasks.parquet").to_pydict()
    text_col = next(c for c in tasks if c != "task_index")
    names = dict(zip(tasks["task_index"], tasks[text_col]))
    counts = pc.value_counts(
        pads.dataset(root / "data", format="parquet").to_table(columns=["task_index"])["task_index"]
    ).to_pylist()
    ranked = sorted(counts, key=lambda c: -c["counts"])
    pools: Dict[str, List[str]] = {"left": [], "right": [], "straight": []}
    for c in ranked:
        text = names.get(c["values"])
        cls = text_class(text) if text else None
        if cls and len(pools[cls]) < size:
            pools[cls].append(text)
    empty = [k for k, v in pools.items() if not v]
    if empty:
        raise ValueError(f"no dataset instruction for class {empty} under {root}")
    return pools


def relabel(rows, episode: int, frame: int, share: float, pools: Dict[str, List[str]], fallback: str) -> str:
    """With probability `share`, a pool sentence matching the motion ahead; otherwise `fallback`."""
    rng = random.Random(f"text:{episode}:{frame}")
    if rng.random() >= share:
        return fallback
    cls = motion_class(rows, frame)
    return rng.choice(pools[cls]) if cls else fallback


def class_shares(root: Path, episodes: List[int]) -> Dict[str, float]:
    """Share of frames per motion class (None = unclear) in `episodes`, read from the data table."""
    table = pads.dataset(root / "data", format="parquet").to_table(
        columns=["episode_index", "frame_index", "action"], filter=pads.field("episode_index").isin(episodes)
    )
    ep, fr = table["episode_index"].to_numpy(), table["frame_index"].to_numpy()
    yaw = pc.list_flatten(table["action"]).to_numpy(zero_copy_only=False).reshape(len(ep), -1)[:, 1]
    counts: Dict[str, int] = {}
    for e in np.unique(ep):
        sel = np.flatnonzero(ep == e)
        rows = [{"action": (0.0, r)} for r in yaw[sel[np.argsort(fr[sel])]]]
        for i in range(len(rows)):
            cls = str(motion_class(rows, i))
            counts[cls] = counts.get(cls, 0) + 1
    total = sum(counts.values())
    return {k: v / total for k, v in sorted(counts.items())}


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Per-frame text label shares and sentences for a LeRobot dataset.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    total = json.loads((args.root / "meta" / "info.json").read_text())["total_episodes"]
    sample = random.Random(args.seed).sample(range(total), min(args.episodes, total))
    print("frame shares:", {k: f"{v:.1%}" for k, v in class_shares(args.root, sample).items()})
    print("sentences:", json.dumps(label_pools(args.root), indent=2))
