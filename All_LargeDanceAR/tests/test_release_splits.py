import tempfile
from pathlib import Path
import unittest

from models.release_splits import check_release_splits


class ReleaseSplitTest(unittest.TestCase):
    def test_separate_validation_and_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'All_train.txt').write_text('train\n\n')
            (root / 'All_val.txt').write_text('validation\n')
            (root / 'test_eval861.txt').write_text('evaluation\n')
            self.assertEqual(check_release_splits(root), {'train': 1, 'val': 1, 'test': 1})
            (root / 'All_val.txt').write_text('evaluation\n')
            with self.assertRaisesRegex(ValueError, 'val/test split overlap'):
                check_release_splits(root)


if __name__ == '__main__':
    unittest.main()
