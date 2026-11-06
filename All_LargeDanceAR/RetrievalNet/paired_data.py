"""Paired music55 / motion264 windows used to train RetrievalNet."""
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class AudioMotion_mb(Dataset):
    def __init__(self, mofea264_dir, audio_dir, meta_paths, split, **kwargs):
        self.motion_dir = Path(mofea264_dir)
        self.audio_dir = Path(audio_dir)
        with open(meta_paths, encoding="utf-8") as handle:
            metadata = json.load(handle)
        self.data_list = [item for item in metadata
                          if split == "all" or item["mode"] == split]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        item = self.data_list[index]
        name, start, end = item["seqname"], item["start_idx"], item["end_idx"]
        motion = np.load(self.motion_dir / (name + ".npy"), mmap_mode="r",
                         allow_pickle=False)[start:end].astype(np.float32)
        audio = np.load(self.audio_dir / (name + ".npy"), mmap_mode="r",
                        allow_pickle=False)[start:end, :55].astype(np.float32)
        if end <= start or motion.shape != (end - start, 264) or audio.shape != (end - start, 55):
            raise ValueError(f"Invalid paired window {name}@{start}_{end}: "
                             f"audio={audio.shape}, motion={motion.shape}")
        return f"{name}@{start}_{end}", audio, motion
