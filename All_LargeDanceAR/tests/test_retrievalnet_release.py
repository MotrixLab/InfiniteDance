import unittest
import tempfile
from pathlib import Path

import numpy as np

from RetrievalNet.configs import get_config
from RetrievalNet.retrieve import _load_motion_embeddings, _style_for, _windows


class RetrievalNetReleaseTest(unittest.TestCase):
    def test_released_config_matches_checkpoint_window(self):
        config = get_config(
            "RetrievalNet/configs/largedance/musicbody/InterCLIP.yaml"
        )
        self.assertEqual(config.TransEncoder.max_seq_len, 384)
        self.assertEqual(config.CondEncoder.max_seq_len, 384)
        self.assertEqual(config.CondEncoder.input_feats, 55)

    def test_short_music_is_repeat_padded(self):
        features = np.arange(10 * 55, dtype=np.float32).reshape(10, 55)
        windows = _windows(features)
        self.assertEqual(windows.shape, (1, 384, 55))
        np.testing.assert_array_equal(windows[0, :10], features)
        np.testing.assert_array_equal(windows[0, 10:20], features)

    def test_style_map_and_name_prefix(self):
        self.assertEqual(_style_for("clip@0_384", {"clip": "Modern"}), "Modern")
        self.assertEqual(_style_for("Popular-Jazz1@0_384", {}), "Popular")

    def test_consolidated_embedding_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "embeddings.npz"
            np.savez(
                archive,
                names=np.asarray(["clip@0_384", "Popular-Jazz1@0_384"]),
                embeddings=np.eye(2, 256, dtype=np.float32),
            )
            names, styles, matrix = _load_motion_embeddings(
                archive, {"clip": "Modern"}
            )
            self.assertEqual(names, ["clip@0_384", "Popular-Jazz1@0_384"])
            self.assertEqual(styles, ["Modern", "Popular"])
            self.assertEqual(matrix.shape, (2, 256))


if __name__ == "__main__":
    unittest.main()
