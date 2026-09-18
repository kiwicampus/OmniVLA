from pathlib import Path
import tempfile
import unittest

from omnivla_training.checkpoint_utils import (
    CHECKPOINT_COMPLETE_FILE,
    checkpoint_step,
    find_latest_complete_checkpoint,
)


def _make_checkpoint(run_root: Path, run_id: str, step: int, complete: bool = True) -> Path:
    checkpoint_dir = run_root / f"{run_id}--{step}_chkpt"
    checkpoint_dir.mkdir(parents=True)
    if complete:
        (checkpoint_dir / CHECKPOINT_COMPLETE_FILE).touch()
    return checkpoint_dir


class CheckpointUtilsTests(unittest.TestCase):
    def test_checkpoint_step_parses_suffix(self):
        self.assertEqual(checkpoint_step(Path("omnivla-original+my_dataset+b8+lr-2e-05--5000_chkpt")), 5000)

    def test_checkpoint_step_none_for_non_matching_name(self):
        self.assertIsNone(checkpoint_step(Path("omnivla-original+my_dataset+b8+lr-2e-05")))

    def test_find_latest_complete_checkpoint_picks_highest_complete_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir)
            run_id = "omnivla-original+my_dataset+b8+lr-2e-05"
            _make_checkpoint(run_root, run_id, 500)
            _make_checkpoint(run_root, run_id, 1000)
            # A newer, still-being-written checkpoint without the sentinel must be skipped.
            _make_checkpoint(run_root, run_id, 1500, complete=False)

            latest = find_latest_complete_checkpoint(run_root, run_id)
            self.assertEqual(latest, run_root / f"{run_id}--1000_chkpt")

    def test_find_latest_complete_checkpoint_ignores_other_run_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir)
            _make_checkpoint(run_root, "other-run", 9000)

            self.assertIsNone(find_latest_complete_checkpoint(run_root, "omnivla-original+my_dataset+b8+lr-2e-05"))

    def test_find_latest_complete_checkpoint_missing_run_root(self):
        self.assertIsNone(find_latest_complete_checkpoint(Path("/nonexistent/path"), "any-run"))


if __name__ == "__main__":
    unittest.main()
