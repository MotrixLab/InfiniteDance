import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

METRICS = Path(__file__).resolve().parents[1] / 'metrics'
sys.path.insert(0, str(METRICS))
release = importlib.import_module('evaluate_release')
legacy = importlib.import_module('metrics_largedata_v2')
legacy_ba = importlib.import_module('beat_align_score_joints')


class ReleaseMetricsTest(unittest.TestCase):
    def test_distribution_formula_matches_previous_scorer(self):
        rng = np.random.default_rng(14)
        for dim in (66, 32):
            pred, gt = rng.normal(size=(90, dim)), rng.normal(size=(120, dim))
            new_fid, new_div = release.distribution_metrics(pred, gt)
            norm_gt, norm_pred = legacy.normalize_simple(gt, pred)
            self.assertAlmostEqual(new_fid, float(legacy.calc_fid(norm_pred, norm_gt).real), places=9)
            self.assertAlmostEqual(new_div, legacy.calculate_avg_distance(norm_pred), places=11)

    def test_canonical_selection_handles_prefixes_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['dance_sample_12_15_a', 'dance_sample_12_2_b',
                         'dance_sample_0_0_a', 'unrelated']:
                (root / (name + '.npy')).touch()
            (root / 'kinetic_features').mkdir()
            (root / 'kinetic_features/stale.npy').touch()
            selected, missing, ignored = release.select_predictions(root, ['b', 'a'], 12)
            self.assertEqual([name for name, _ in selected], ['b', 'a'])
            self.assertEqual(missing, [])
            self.assertEqual(len(ignored), 2)
            with self.assertRaisesRegex(ValueError, 'canonical predictions missing'):
                release.select_predictions(root, ['b', 'a', 'c'], 12)
            self.assertEqual(release.select_predictions(root, ['b', 'a', 'c'], 12, True)[1], ['c'])
            (root / 'a.npy').touch()
            with self.assertRaisesRegex(ValueError, 'Multiple predictions'):
                release.select_predictions(root, ['a', 'b'], 12)

    def test_beats_match_previous_formula_and_undefined_cases_are_explicit(self):
        t = np.arange(160)
        joints = np.zeros((160, 22, 3))
        joints[:, :, 0] = np.sin(t[:, None] / 9)
        music = np.zeros((160, 55))
        music[10::20, 53] = 1
        beats, length = legacy_ba.calc_db(joints)
        expected = legacy_ba.BA(np.flatnonzero(music[:length, 53]), beats)
        self.assertAlmostEqual(release.beat_alignment(joints, music), expected, places=12)
        with self.assertRaisesRegex(ValueError, 'undefined'):
            release.beat_alignment(np.zeros_like(joints), music)
        with self.assertRaisesRegex(ValueError, 'undefined'):
            release.beat_alignment(joints, np.zeros_like(music))
        with self.assertRaisesRegex(ValueError, 'cover'):
            release.beat_alignment(joints, music[:4])

    def test_real_feature_path_ignores_stale_caches_and_reports_same_frame_cap(self):
        rng = np.random.default_rng(21)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred, music = root / 'pred', root / 'music'
            pred.mkdir(); music.mkdir()
            kinetic, manual = [], []
            for i, name in enumerate(['alpha', 'beta']):
                skeleton = rng.normal(size=(1, 22, 3)).astype(np.float32)
                t = np.arange(160, dtype=np.float32)
                joints = np.repeat(skeleton, 160, axis=0)
                joints[:, :, 0] += np.sin(t[:, None] / (9 + i))
                path = pred / ('dance_sample_0_15_' + name + '.npy')
                np.save(path, joints)
                features = np.zeros((160, 55)); features[10::20, 53] = 1
                np.save(music / (name + '.npy'), features)
                k, m = release.motion_features(joints[:120])
                kinetic.append(k); manual.append(m)
                legacy.calc_and_save_feats(path.name, root=str(pred), max_frames=120)
                np.testing.assert_array_equal(k, np.load(pred / 'kinetic_features' / path.name))
                np.testing.assert_array_equal(m, np.load(pred / 'manual_features_new' / path.name))
            np.save(pred / 'kinetic_features/stale.npy', np.full(66, np.nan))
            np.save(pred / 'manual_features_new/stale.npy', np.full(32, np.nan))
            gt_names = ['gt%d' % i for i in range(100)]
            np.savez(root / 'gt.npz', names=gt_names,
                     kinetic=np.mean(kinetic, axis=0) + rng.normal(size=(100, 66)),
                     manual=np.mean(manual, axis=0) + rng.normal(size=(100, 32)))
            (root / 'gt.txt').write_text('\n'.join(gt_names))
            (root / 'eval.txt').write_text('alpha\nbeta\n')
            result = release.main(['--pred-root', str(pred), '--music-root', str(music),
                                  '--gt-features', str(root / 'gt.npz'), '--gt-list', str(root / 'gt.txt'),
                                  '--eval-list', str(root / 'eval.txt'), '--output', str(root / 'result.json'),
                                  '--max-frames', '120'])
            self.assertEqual(result, 0)
            report = json.loads((root / 'result.json').read_text())
            self.assertEqual(report['predictions'], 2)
            self.assertEqual(report['gt_features'], 100)
            self.assertTrue(all(row['frames'] == 120 for row in report['samples']))
            self.assertTrue(all(np.isfinite(x) for x in report['metrics'].values()))
            self.assertTrue((pred / 'kinetic_features/stale.npy').exists())


if __name__ == '__main__':
    unittest.main()
