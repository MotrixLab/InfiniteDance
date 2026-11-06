"""Motion windows for retraining the released RVQ codec."""
from pathlib import Path
import warnings
import numpy as np
from torch.utils.data import Dataset


class ReleaseMotionDataset(Dataset):
    def __init__(self, motion_dir, split_file, meta_dir, window_size=128, training=True):
        self.motion_dir = Path(motion_dir)
        self.window_size = window_size
        self.training = training
        self.mean = np.load(Path(meta_dir) / 'Mean.npy').astype(np.float32)
        self.std = np.load(Path(meta_dir) / 'Std.npy').astype(np.float32)
        if self.mean.shape != (264,) or self.std.shape != (264,) or not np.all(self.std > 0):
            raise ValueError('RVQ normalization must contain 264 values and positive standard deviations')
        names = [name.strip() for name in Path(split_file).read_text().splitlines() if name.strip()]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate identities in motion split')
        self.names = []
        self.short_names = []
        for name in names:
            motion = np.load(self.motion_dir / (name + '.npy'), mmap_mode='r', allow_pickle=False)
            if motion.ndim != 2 or motion.shape[1] != 264:
                raise ValueError('Expected (T,264) motion: ' + name)
            if len(motion) < window_size:
                self.short_names.append(name)
            else:
                self.names.append(name)
        if self.short_names:
            warnings.warn('%d clips are shorter than the requested %d-frame window' %
                          (len(self.short_names), window_size))
        if not self.names:
            raise ValueError('No motion clips contain a complete training window')

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        motion = np.load(self.motion_dir / (self.names[index] + '.npy'), mmap_mode='r', allow_pickle=False)
        start = np.random.randint(len(motion) - self.window_size + 1) if self.training else 0
        window = np.asarray(motion[start:start + self.window_size], dtype=np.float32)
        if not np.isfinite(window).all():
            raise ValueError('Nonfinite motion: ' + self.names[index])
        return (window - self.mean) / self.std
