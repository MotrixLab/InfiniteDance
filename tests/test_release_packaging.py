import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class ReleasePackagingTest(unittest.TestCase):
    def test_missing_samples_are_reported_without_changing_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'partition').mkdir()
            split = root / 'partition/All_train.txt'
            split.write_text('present\n\nmissing\n')
            for folder, suffix in [('music/muq_features', '.npy'),
                                   ('dance/Infinite_MotionTokens_512x1024_3layer_cleandata', '.npy'),
                                   ('styles/Alldata', '.json'), ('dance/retrieval_s192_l384_style', '.json')]:
                p = root / folder
                p.mkdir(parents=True)
                (p / ('present' + suffix)).touch()
            report = module('check_release_data').inspect(root, 'All_train')
            self.assertEqual(report['listed'], 2)
            self.assertEqual(report['complete_file_sets'], 1)
            self.assertTrue(all(names == ['missing'] for names in report['missing'].values()))
            self.assertEqual(split.read_text(), 'present\n\nmissing\n')

    def test_launcher_preserves_paths_and_extra_arguments(self):
        with tempfile.TemporaryDirectory(prefix='release test ') as tmp:
            root = Path(tmp)
            fake_python = root / 'fake python'
            capture = root / 'argv.json'
            fake_python.write_text('#!/usr/bin/env python3\nimport os,sys,json\n'
                                   'open(os.environ["CAPTURE"],"w").write(json.dumps(sys.argv[1:]))\n')
            fake_python.chmod(0o755)
            env = dict(os.environ, PY=str(fake_python), CAPTURE=str(capture),
                       DATA_ROOT=str(root / 'data with spaces'), OUT=str(root / 'output'))
            subprocess.run(['bash', str(ROOT / 'All_LargeDanceAR/train.sh'), '--batch_size', '7'],
                           env=env, check=True, stdout=subprocess.PIPE)
            argv = json.loads(capture.read_text())
            self.assertEqual(argv[argv.index('--music_dir') + 1], str(root / 'data with spaces/music/muq_features'))
            self.assertEqual(argv[argv.index('--mean_path') + 1], str(root / 'data with spaces/dance/vqvae_train_20260522/meta/Mean.npy'))
            self.assertEqual(argv[argv.index('--std_path') + 1], str(root / 'data with spaces/dance/vqvae_train_20260522/meta/Std.npy'))
            self.assertEqual(argv[-2:], ['--batch_size', '7'])

    def test_extract_and_reject_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'data.tar.gz'
            with tarfile.open(archive, 'w:gz') as handle:
                member = tarfile.TarInfo('muq_features/clip.npy')
                member.size = 4
                handle.addfile(member, io.BytesIO(b'data'))
            module('prepare_release_data').extract(archive, root / 'music')
            self.assertEqual((root / 'music/muq_features/clip.npy').read_bytes(), b'data')
            with tarfile.open(archive, 'w:gz') as handle:
                member = tarfile.TarInfo('../outside')
                member.size = 1
                handle.addfile(member, io.BytesIO(b'x'))
            with self.assertRaises(ValueError):
                module('prepare_release_data').extract(archive, root / 'music')

    def test_metrics_navigation_link_resolves_code_and_preserves_caller_paths(self):
        with tempfile.TemporaryDirectory(prefix='metrics entry ') as tmp:
            root = Path(tmp)
            link = root / 'metrics.sh'
            link.symlink_to(ROOT / 'All_LargeDanceAR/metrics.sh')
            capture = root / 'arguments.json'
            fake = root / 'fake python'
            fake.write_text('#!/usr/bin/env python3\nimport os,sys,json\n'
                            'open(os.environ["CAPTURE"],"w").write(json.dumps(sys.argv[1:]))\n')
            fake.chmod(0o755)
            env = dict(os.environ, PY=str(fake), CAPTURE=str(capture), DATA_ROOT=str(root / 'data'))
            subprocess.run(['bash', str(link), 'relative predictions', '7', '--allow-partial'],
                           cwd=tmp, env=env, check=True)
            argv = json.loads(capture.read_text())
            self.assertEqual(Path(argv[0]), ROOT / 'All_LargeDanceAR/metrics/evaluate_release.py')
            self.assertEqual(argv[argv.index('--pred-root') + 1], 'relative predictions')
            self.assertEqual(argv[argv.index('--eval-list') + 1], str(root / 'data/partition/test_eval861.txt'))
            self.assertEqual(argv[-1], '--allow-partial')


if __name__ == '__main__':
    unittest.main()
