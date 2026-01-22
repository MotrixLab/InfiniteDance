import json
# import librosa
import os
import pickle as pkl
import sys
import test

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

# import ipdb
# from .pipelines import Compose



class AudioMotion(data.Dataset):
    def __init__(self, 
                 face_dir,
                 joints_dir,
                 audio_dir,
                 meta_paths, 
                 split,
                 pose_fps=30,
                #  audio_sr=16000,
                 pipeline=None,
                 **kwargs
                 ):
        self.face_dir = face_dir
        self.joints_dir = joints_dir
        self.audio_dir = audio_dir
        # self.pipeline = Compose(pipeline)
        
        vid_meta = json.load(open(meta_paths, "r"))
        if split != 'all':
            self.vid_meta = [item for item in tqdm(vid_meta) if item.get("mode") == split]
        else:
            print('load all data')
            self.vid_meta = [item for item in tqdm(vid_meta)]
        self.data_list = self.vid_meta
        self.mo_fps = pose_fps
        # self.audio_sr = audio_sr
        print(f'AudioMotion has {len(self.vid_meta)} samples')

    def __len__(self):
        return len(self.data_list)
    
    @staticmethod
    def normalize(motion, mean, std):
        return (motion - mean) / (std + 1e-7)
    
    @staticmethod
    def inverse_normalize(motion, mean, std):
        return motion * std + mean

    def __getitem__(self, item):
        # ipdb.set_trace()
        data_item = self.data_list[item]
        sdx, edx = data_item["start_idx"], data_item["end_idx"]
        
        seqname = data_item["seqname"]
        # with open(os.path.join(self.face_dir, seqname +  '.npy'), 'rb') as f:
        #     dataf = pkl.load(f)
        # print('dataf', dataf.keys())
        flame = np.load(os.path.join(self.face_dir, seqname+ '.npy'))[sdx:edx]
        joints = np.load(os.path.join(self.joints_dir, seqname+ '.npy'))[sdx:edx]
        joints = joints[:, :22, :].reshape(-1, 22*3)
        # extracted wev2vec2 1 seconnd is 30 frames feature
        audio = np.load(os.path.join(self.audio_dir, seqname+'.npy'))
        audio = audio[sdx:edx]      # audio帧率和motion一致

        return seqname, audio, joints, flame   
    
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

if __name__ == '__main__':
    motion_dir = '/data2/lrh/dataset/largedance/ourdata_smplx_pro'
    audio_dir  = '/data2/lrh/dataset/largedance/allmusic_librosa55'
    save_json  = '/data2/lrh/dataset/largedance/Dataset_Split/alldata_s120_l120.json'
    split_dir  = '/data2/lrh/dataset/largedance/Dataset_Split'
    joints_dir = '/data2/lrh/dataset/largedance/joints'
    
    slides = 120
    length = 120
    
    train_list = []
    test_list = []
    train_file = open(os.path.join(split_dir, 'All_train.txt') , 'r')
    for fname in train_file.readlines():
        train_list.append(fname.strip())
    train_file.close()
    
    test_file = open(os.path.join(split_dir, 'All_eval.txt'), 'r')
    for fname in test_file.readlines():
        test_list.append(fname.strip())
    test_file.close()
    
    
    # ipdb.set_trace()
    datainfo = []
    for split in ['train', 'test']:
        if split == 'test':
            datalist = test_list
        elif split == 'train':
            datalist = train_list
            
        for seqname in tqdm(datalist):
            mopath = os.path.join(motion_dir, seqname+'.pkl')
            mupath = os.path.join(audio_dir, seqname+'.npy')
            jointspath = os.path.join(joints_dir, seqname+'.npy')
            if not os.path.exists(mopath):
                continue
            if not os.path.exists(mupath):
                continue
            if not os.path.exists(jointspath):
                continue
            
            with open(mopath, 'rb') as f:
                dataf = pkl.load(f)
            # print(dataf.keys())
            music_frame = dataf['expression'].shape[0]
            musicfeas = np.load(mupath)
            motion_frame = musicfeas.shape[0]
            joints_frame = np.load(jointspath).shape[0]
            maxframe = min(music_frame, motion_frame, joints_frame)

            for idx in range(0, maxframe, slides):
                if idx+length <= maxframe:
                    write_one = {}
                    write_one['seqname'] = seqname
                    write_one["mode"] = split
                    write_one['start_idx'] = idx
                    write_one['end_idx'] = idx + length
                    datainfo.append(write_one)
        
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(datainfo, f, ensure_ascii=False, indent=4)
        
        
        
