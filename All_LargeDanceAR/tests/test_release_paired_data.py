import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from torch.utils.data import DataLoader

from RetrievalNet.configs import get_config
from RetrievalNet.paired_data import AudioMotion_mb


class ReleasePairedDataTest(unittest.TestCase):
    def test_real_slices_collate_as_audio_and_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'audio').mkdir()
            (root / 'motion').mkdir()
            audio = np.arange(12 * 55, dtype=np.float32).reshape(12, 55)
            motion = np.arange(12 * 264, dtype=np.float32).reshape(12, 264)
            np.save(root / 'audio/clip.npy', audio)
            np.save(root / 'motion/clip.npy', motion)
            records = [{'seqname': 'clip', 'mode': 'train', 'start_idx': 2, 'end_idx': 10},
                       {'seqname': 'clip', 'mode': 'test', 'start_idx': 0, 'end_idx': 8}]
            (root / 'windows.json').write_text(json.dumps(records))
            dataset = AudioMotion_mb(root / 'motion', root / 'audio', root / 'windows.json', 'train')
            self.assertEqual(len(dataset), 1)
            names, audio_batch, motion_batch = next(iter(DataLoader(dataset, batch_size=1)))
            self.assertEqual(names, ('clip@2_10',))
            np.testing.assert_array_equal(audio_batch.numpy()[0], audio[2:10])
            np.testing.assert_array_equal(motion_batch.numpy()[0], motion[2:10])
            dataset.data_list[0]['end_idx'] = 20
            with self.assertRaises(ValueError):
                dataset[0]

    def test_config_honors_external_data_root(self):
        with tempfile.TemporaryDirectory(prefix='data root ') as tmp:
            with patch.dict(os.environ, {'INFINITEDANCE_DATA_ROOT': tmp}):
                cfg = get_config('RetrievalNet/configs/largedance/musicbody/datasets45.yaml')
            self.assertEqual(cfg.largedance.audio_dir, str(Path(tmp).resolve() / 'music/musicfeature_55_allmusic_pure'))
            self.assertTrue(cfg.largedance.meta_paths.endswith('alldata_mbr_s100_l384.json'))


if __name__ == '__main__':
    unittest.main()
