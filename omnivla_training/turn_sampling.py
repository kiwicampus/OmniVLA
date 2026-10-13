"""Per-sample weights that favour frames where the robot is turning.

Pure numpy and pyarrow, so it runs without torch or the dataset class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as pads


def turn_rates(data_dir: Path, index: Sequence[Tuple[int, int]], width: int) -> np.ndarray:
    """Mean angular.z over frames i..i+width-1 for each (episode, frame) in `index`."""
    keys = np.asarray(index, dtype=np.int64).reshape(-1, 2)
    table = pads.dataset(data_dir, format="parquet").to_table(
        columns=["episode_index", "frame_index", "action"],
        filter=pads.field("episode_index").isin(np.unique(keys[:, 0]).tolist()),
    )
    episode = table["episode_index"].to_numpy()
    frame = table["frame_index"].to_numpy()
    yaw_rate = pc.list_flatten(table["action"]).to_numpy(zero_copy_only=False)
    yaw_rate = yaw_rate.reshape(len(episode), -1)[:, 1]

    order = np.lexsort((frame, episode))
    episode, frame, yaw_rate = episode[order], frame[order], yaw_rate[order]
    rows = np.arange(len(episode))
    starts = np.flatnonzero(np.r_[True, episode[1:] != episode[:-1]])
    lengths = np.diff(np.r_[starts, len(episode)])
    row_start = np.repeat(starts, lengths)
    if not np.array_equal(frame, rows - row_start):
        raise ValueError("frame_index must run 0..length-1 inside every episode")

    ends = np.minimum(rows + width, row_start + np.repeat(lengths, lengths))
    csum = np.concatenate([[0.0], np.cumsum(yaw_rate)])
    mean_rate = (csum[ends] - csum[rows]) / (ends - rows)
    return mean_rate[starts[np.searchsorted(episode[starts], keys[:, 0])] + keys[:, 1]]


def turn_sample_weights(
    data_dir: Path,
    index: Sequence[Tuple[int, int]],
    width: int,
    rate_threshold: float,
    boost: float,
) -> np.ndarray:
    """`boost` for each (episode, frame) whose |turn rate| exceeds `rate_threshold`, else 1."""
    return np.where(np.abs(turn_rates(data_dir, index, width)) > rate_threshold, float(boost), 1.0)
