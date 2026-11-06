from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from DanceVQVAE.release_data import ReleaseMotionDataset
from DanceVQVAE.utils.losses import ReConsLoss
from DanceVQVAE.models.vqvae import RVQVAE
from All_LargeDanceAR.models.vqvae import RVQVAE as GeneratorRVQVAE


class ReleaseVQTest(unittest.TestCase):
    def test_shared_architecture_and_exact_window(self):
        self.assertIs(RVQVAE, GeneratorRVQVAE)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / 'Mean.npy', np.ones(264, dtype=np.float32))
            np.save(root / 'Std.npy', np.full(264, 2, dtype=np.float32))
            motion = np.arange(4 * 264, dtype=np.float32).reshape(4, 264)
            np.save(root / 'clip.npy', motion)
            (root / 'split.txt').write_text('clip\n')
            data = ReleaseMotionDataset(root, root / 'split.txt', root, 4, False)
            np.testing.assert_array_equal(data[0], (motion - 1) / 2)
            (root / 'split.txt').write_text('missing\n')
            with self.assertRaises(FileNotFoundError):
                ReleaseMotionDataset(root, root / 'split.txt', root, 4, False)

    def test_foot_loss_selects_xyz_of_foot_joints(self):
        loss = ReConsLoss(Namespace(vel_decoder=False, rot_decoder=False), 'l2', 22)
        gt = torch.zeros(1, 3, 264)
        pred = gt.clone()
        pred[:, :, -4:] = 1
        pred[:, 1:, 5 + (7 - 1) * 3] = 1  # left ankle x, root excluded from RIC
        self.assertGreater(loss.forward_all(pred, gt)[-1].item(), 0)
        pred[:, :, 5:68] = 0
        pred[:, 1:, 5 + (3 - 1) * 3] = 1  # spine x is not a foot coordinate
        self.assertEqual(loss.forward_all(pred, gt)[-1].item(), 0)


if __name__ == '__main__':
    unittest.main()
