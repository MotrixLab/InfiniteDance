import json
import os
import sys

import librosa
import numpy as np
import torch
from torch.utils import data

from .pipelines import Compose


class AISTPPDataset(data.Dataset):
    def __init__(self, 
                 dataset_path,
                 motion_dir,
                 music_dir,
                 split_dir,
                 split,
                 pipeline=None,
                 ):
        self.dataname = dataset_path
        self.motion_dir = motion_dir
        self.music_dir = music_dir
        self.split_dir = split_dir
        self.pipeline = Compose(pipeline)
        
        train_list, test_list = self.get_train_test_list()        

        if split == 'train':
            self.data_list = train_list
        elif split == 'test':
            self.data_list = test_list
        print(f'AISTPP has {len(self.data_list)} saamples')
        
        molist = os.listdir(motion_dir)
        mulist = os.listdir(music_dir)
        for seqname in self.data_list:
            # 如果不存在则跳过
            if seqname+'.npy' not in molist:
                continue
            if seqname+'.npy' not in mulist:
                continue
        
    def __len__(self):
        return len(self.data_list)
    
    def get_train_test_list(self):
        train = []
        test = []
        train_file = open(os.path.join(self.split_dir, 'All_train.txt') , 'r')
        for fname in train_file.readlines():
            train.append(fname.strip())
        train_file.close()

        test_file = open(os.path.join(self.split_dir, 'All_eval.txt'), 'r')
        for fname in test_file.readlines():
            test.append(fname.strip())
        test_file.close()
      
        return train, test

    def __getitem__(self, item):
        name = self.data_list[item] + '.npy'
        motion = np.load(os.path.join(self.motion_dir, name))
        music = np.load(os.path.join(self.music_dir, name))
        minlength = min(motion.shape[0], music.shape[0])
        motion = motion[:minlength]
        music = music[:minlength]
        assert motion.shape[0] == music.shape[0]
             
        meta_data = {}
        meta_data['framerate'] = 30
        meta_data['dataset_name'] = 'aistpp'
        meta_data['rotation_type'] = 'smpl_rot'

        results = {}
        results.update({
                'music_cond': 1,
                'music_feat': music,
                'motion': motion,
                'motion_metas': meta_data, 
            })
        results = self.pipeline(results)
        return results