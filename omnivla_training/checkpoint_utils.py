"""Shared conventions for the `<run_dir>--<step>_chkpt` checkpoint layout.

`vla-scripts/train_omnivla.py` writes one directory per save, named
`<run_dir>--<step>_chkpt`, as a sibling of `run_dir` rather than a child of
it. This module is the single place that knows that naming convention, so
the trainer (resume) and the Vertex wrapper (pruning) don't each carry their
own copy of the same regex.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

CHECKPOINT_STEP_PATTERN = re.compile(r"--(\d+)_chkpt$")

# Written last by `save_training_checkpoint`, once every artifact in a
# checkpoint directory has landed. Its absence means the checkpoint is
# either still being written or predates checkpoint resume.
CHECKPOINT_COMPLETE_FILE = "_CHECKPOINT_COMPLETE"


def checkpoint_step(checkpoint_dir: Path) -> Optional[int]:
    """Parse the step number out of a `<run_dir>--<step>_chkpt` directory name."""
    match = CHECKPOINT_STEP_PATTERN.search(checkpoint_dir.name)
    return int(match.group(1)) if match else None


def is_checkpoint_complete(checkpoint_dir: Path) -> bool:
    return (checkpoint_dir / CHECKPOINT_COMPLETE_FILE).exists()


def find_latest_complete_checkpoint(run_root: Path, run_id: str) -> Optional[Path]:
    """Return the highest-step complete checkpoint dir for `run_id`, or None.

    `run_root` is `checkpoint.run_root_dir`; checkpoints for `run_id` live at
    `run_root / f"{run_id}--{step}_chkpt"`.
    """
    if not run_root.is_dir():
        return None

    prefix = f"{run_id}--"
    candidates = []
    for child in run_root.iterdir():
        if not child.is_dir() or not child.name.startswith(prefix):
            continue
        step = checkpoint_step(child)
        if step is None or not is_checkpoint_complete(child):
            continue
        candidates.append((step, child))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]
