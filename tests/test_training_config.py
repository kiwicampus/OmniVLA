from pathlib import Path
import tempfile
import unittest

import yaml

from omnivla_training.episode_manifest import collapse_episode_ranges, expand_episode_ranges, load_episode_subset


ROOT = Path(__file__).resolve().parents[1]


class TrainingConfigTests(unittest.TestCase):
    def test_canonical_config_shape(self):
        cfg = yaml.safe_load((ROOT / "config_nav" / "train_omnivla.yaml").read_text(encoding="utf-8"))
        self.assertIn("dataset", cfg)
        self.assertIn("model", cfg)
        self.assertIn("training", cfg)
        self.assertIn("checkpoint", cfg)
        self.assertIn("logging", cfg)
        self.assertEqual(cfg["model"]["vla_path"], "NHirose/omnivla-original")
        self.assertTrue(cfg["model"]["use_lora"])
        self.assertEqual(cfg["dataset"]["repo_id"], "robotcom/single_waypoints")
        self.assertIsNone(cfg["dataset"]["episodes_file"])
        self.assertEqual(cfg["dataset"]["action_key"], "observation.state")
        self.assertEqual(cfg["dataset"]["context_size"], 0)

    def test_episode_range_round_trip(self):
        episodes = [0, 1, 2, 7, 8, 10]
        ranges = collapse_episode_ranges(episodes)
        self.assertEqual(ranges, [[0, 2], [7, 8], [10, 10]])
        self.assertEqual(expand_episode_ranges(ranges), episodes)

    def test_compact_manifest_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.yaml"
            path.write_text("clean_episode_ranges:\n  - [1, 3]\n  - [10, 10]\n", encoding="utf-8")
            self.assertEqual(load_episode_subset(path), [1, 2, 3, 10])


if __name__ == "__main__":
    unittest.main()
