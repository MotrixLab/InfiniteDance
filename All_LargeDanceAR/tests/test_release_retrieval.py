import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.get_top10_mofea264 import get_top_mofea_specific_style


class ReleaseRetrievalTest(unittest.TestCase):
    def test_cache_path_and_missing_cache_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            retrieval = root / "retrieval"
            motion = root / "motion"
            tokens = root / "tokens"
            meta = motion / "meta"
            retrieval.mkdir()
            meta.mkdir(parents=True)
            tokens.mkdir()

            (retrieval / "song.json").write_text(json.dumps({
                "idx_0": {"Popular": [{"name": "clip@0_384"}]}
            }))
            (root / "styles.json").write_text(json.dumps({"clip": "Popular"}))
            np.save(meta / "Mean.npy", np.zeros(264, dtype=np.float32))
            np.save(meta / "Std.npy", np.ones(264, dtype=np.float32))
            np.save(motion / "clip.npy", np.zeros((384, 264), dtype=np.float32))
            np.save(tokens / "clip.npy", np.arange(300, dtype=np.int64))

            result = get_top_mofea_specific_style(
                name="song",
                retrieval_path=str(retrieval),
                motion_base=str(motion),
                motiontoken_dir=str(tokens),
                style_map_path=str(root / "styles.json"),
            )
            self.assertEqual(result[0].shape, (384, 264))
            self.assertEqual(len(result[5][0]["tokens"]), 120)

            with self.assertRaisesRegex(FileNotFoundError, "retrieval_s192_l384_style.tar.gz"):
                get_top_mofea_specific_style(
                    name="missing",
                    retrieval_path=str(retrieval),
                )


if __name__ == "__main__":
    unittest.main()
