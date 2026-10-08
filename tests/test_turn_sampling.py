from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from omnivla_training.turn_sampling import turn_sample_weights


def _write(root: Path, episodes, frames, yaw_rates, fixed_size: bool = False) -> Path:
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    rows = [[1.0, r] for r in yaw_rates]
    action_type = pa.list_(pa.float32(), 2) if fixed_size else pa.list_(pa.float32())
    pq.write_table(
        pa.table(
            {
                "episode_index": episodes,
                "frame_index": frames,
                "action": pa.array(rows, type=action_type),
            }
        ),
        data_dir / "file-000.parquet",
    )
    return root / "data"


class TurnSamplingTests(unittest.TestCase):
    def test_boosts_frames_whose_next_window_turns(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Episode 3 turns from frame 2; episode 5 goes straight. Rows are shuffled.
            data_dir = _write(
                Path(tmp),
                episodes=[3, 5, 3, 3, 5, 3],
                frames=[1, 0, 2, 0, 1, 3],
                yaw_rates=[0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            )
            index = [(3, 0), (3, 1), (3, 2), (5, 0)]
            weights = turn_sample_weights(data_dir, index, width=2, rate_threshold=0.3, boost=5.0)
        # Episode 3 yaw by frame: 0, 0, 1, 1. Window 2: 0.0, 0.5, 1.0.
        self.assertEqual(weights.tolist(), [1.0, 5.0, 5.0, 1.0])

    def test_opposite_turns_inside_a_window_cancel(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = _write(Path(tmp), [1, 1], [0, 1], [1.0, -1.0], fixed_size=True)
            weights = turn_sample_weights(data_dir, [(1, 0)], width=2, rate_threshold=0.3, boost=5.0)
        self.assertEqual(weights.tolist(), [1.0])

    def test_rejects_gaps_in_frame_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = _write(Path(tmp), [1, 1], [0, 2], [0.0, 0.0])
            with self.assertRaises(ValueError):
                turn_sample_weights(data_dir, [(1, 0)], width=2, rate_threshold=0.3, boost=5.0)


if __name__ == "__main__":
    unittest.main()
