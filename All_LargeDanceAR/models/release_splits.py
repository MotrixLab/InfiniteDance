"""Read the ECCV train/validation/test lists without reassigning samples."""
from pathlib import Path


def check_release_splits(directory):
    directory = Path(directory)
    names = {key: {line.strip() for line in (directory / filename).read_text().splitlines() if line.strip()}
             for key, filename in [('train', 'All_train.txt'), ('val', 'All_val.txt'),
                                   ('test', 'test_eval861.txt')]}
    for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        overlap = sorted(names[left] & names[right])
        if overlap:
            raise ValueError(f"{left}/{right} split overlap ({len(overlap)} clips): {overlap[:10]}")
    return {key: len(value) for key, value in names.items()}
