import json
from pathlib import Path
import tempfile
from argparse import Namespace
import unittest
import numpy as np
import torch
from models.vqvae import RVQVAE
from models.motion import load_vqvae_model
from utils.tokens2smpl import motion_detokenizer


class ReleaseDecoderTest(unittest.TestCase):
    def test_base_checkpoint_does_not_apply_untrained_auxiliary_head(self):
        class Decoder:
            use_velocity_decoder = False
            use_rotation_decoder = False
            def decoder(self, latent):
                return torch.ones(latent.shape[0], latent.shape[2] * 4, 264)
            def vel_decoder(self, latent):
                raise AssertionError('Untrained velocity head must not run')
            def rot_decoder(self, latent):
                raise AssertionError('Untrained rotation head must not run')
        output = motion_detokenizer(torch.zeros(1, 3, 8), Decoder(), np.zeros(264), np.ones(264))
        np.testing.assert_array_equal(output, np.ones((12, 264)))

    def test_model_load_does_not_replace_global_forward_or_require_unrelated_stats(self):
        args = Namespace(mu=.99, num_quantizers=3, shared_codebook=False,
            quantize_dropout_prob=0., vel_decoder=False, rot_decoder=False,
            nb_code=8, code_dim=8, output_emb_width=8, down_t=2, stride_t=2,
            width=8, depth=1, dilation_growth_rate=2, vq_act='relu', vq_norm=None)
        model = RVQVAE(args, 264, args.nb_code, args.code_dim, args.output_emb_width,
            args.down_t, args.stride_t, args.width, args.depth, args.dilation_growth_rate,
            args.vq_act, args.vq_norm)
        forward = RVQVAE.forward
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'args.json').write_text(json.dumps(vars(args)))
            torch.save({'net': model.state_dict()}, path / 'model.pth')
            loaded, books, _ = load_vqvae_model(str(path / 'model.pth'), device='cpu')
            self.assertIs(RVQVAE.forward, forward)
            self.assertEqual(len(books), 3)
            with torch.no_grad():
                self.assertEqual(loaded(torch.zeros(1, 16, 264))[0].shape, (1, 16, 264))


if __name__ == '__main__':
    unittest.main()
