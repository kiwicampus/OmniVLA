import math
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from omnivla_training.text_relabel import class_shares, label_pools, motion_class, relabel, text_class

POOLS = {"left": ["Turn left."], "right": ["Turn right."], "straight": ["Go straight."]}


def _rows(total_deg: float, frames: int = 50):
    rate = math.radians(total_deg) / (frames * 0.1)
    return [{"action": [1.0, rate]} for _ in range(frames)]


class TextClassTests(unittest.TestCase):
    def test_words(self):
        self.assertEqual(text_class("Turn LEFT at the corner."), "left")
        self.assertEqual(text_class("turn right"), "right")
        self.assertEqual(text_class("Proceed straight on the path."), "straight")
        self.assertIsNone(text_class("Turn left, then right"))
        self.assertIsNone(text_class("Follow the sidewalk"))


class MotionClassTests(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(motion_class(_rows(45), 0), "left")
        self.assertEqual(motion_class(_rows(-45), 0), "right")
        self.assertEqual(motion_class(_rows(5), 0), "straight")
        self.assertIsNone(motion_class(_rows(20), 0))

    def test_window_is_clipped_at_the_episode_end(self):
        rows = [{"action": [1.0, 0.0]}] * 40 + _rows(45, frames=10)
        self.assertEqual(motion_class(rows, 40), "left")
        self.assertEqual(motion_class(rows, 0, window=40), "straight")


class RelabelTests(unittest.TestCase):
    def test_share_zero_keeps_the_episode_label_and_one_follows_the_motion(self):
        rows = _rows(-60)
        self.assertEqual(relabel(rows, 3, 0, 0.0, POOLS, "episode text"), "episode text")
        self.assertEqual(relabel(rows, 3, 0, 1.0, POOLS, "episode text"), "Turn right.")

    def test_unclear_motion_keeps_the_episode_label(self):
        self.assertEqual(relabel(_rows(20), 3, 0, 1.0, POOLS, "episode text"), "episode text")

    def test_same_sample_same_answer_and_share_is_respected(self):
        rows = _rows(60, frames=60)
        picks = [relabel(rows, 7, f, 0.5, POOLS, "keep") for f in range(10)]
        self.assertEqual(picks, [relabel(rows, 7, f, 0.5, POOLS, "keep") for f in range(10)])
        many = [relabel(_rows(60), e, 0, 0.5, POOLS, "keep") for e in range(400)]
        self.assertAlmostEqual(many.count("Turn left.") / 400, 0.5, delta=0.08)


class LabelPoolsTests(unittest.TestCase):
    def test_most_frequent_sentences_per_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meta").mkdir()
            (root / "data" / "chunk-000").mkdir(parents=True)
            texts = ["Turn left at the corner.", "Turn left here.", "Turn right.", "Go straight.", "Left then right."]
            pq.write_table(pa.table({"task_index": list(range(5)), "task": texts}), root / "meta" / "tasks.parquet")
            frames = [0] * 3 + [1] * 5 + [2] + [3] * 2 + [4] * 9
            pq.write_table(pa.table({"task_index": frames}), root / "data" / "chunk-000" / "file-000.parquet")
            pools = label_pools(root, size=1)
            self.assertEqual(
                pools, {"left": ["Turn left here."], "right": ["Turn right."], "straight": ["Go straight."]}
            )
            self.assertEqual(label_pools(root)["left"], ["Turn left here.", "Turn left at the corner."])

    def test_missing_class_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meta").mkdir()
            (root / "data").mkdir()
            pq.write_table(pa.table({"task_index": [0], "task": ["Turn left."]}), root / "meta" / "tasks.parquet")
            pq.write_table(pa.table({"task_index": [0]}), root / "data" / "file-000.parquet")
            with self.assertRaises(ValueError):
                label_pools(root)


class ClassSharesTests(unittest.TestCase):
    def test_shares_per_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            rate = math.radians(90) / 5.0  # 90 degrees over 50 frames
            yaw = [rate] * 50 + [0.0] * 50
            pq.write_table(
                pa.table(
                    {
                        "episode_index": [1] * 100,
                        "frame_index": list(range(100))[::-1],
                        "action": [[1.0, r] for r in yaw[::-1]],
                    }
                ),
                root / "data" / "file-000.parquet",
            )
            shares = class_shares(root, [1])
            self.assertAlmostEqual(sum(shares.values()), 1.0)
            self.assertGreater(shares["left"], 0.3)
            self.assertGreater(shares["straight"], 0.3)


if __name__ == "__main__":
    unittest.main()
