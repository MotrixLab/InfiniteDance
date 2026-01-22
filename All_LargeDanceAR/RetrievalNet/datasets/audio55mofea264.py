import json
# import librosa
import os
import pickle as pkl
import sys
import test

import ipdb
import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

# from .pipelines import Compose
# sys.path.append('/data1/hzy/HumanMotion/utilsds')
# from common.quaternion import qinv, qrot


def recover_from_ric264(data,joints_num=22):
    r_pos_y=data[:,0]
    rot_vel = data[..., 2]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 3:5]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 0]

    positions = data[..., 5:(joints_num - 1) * 3 + 5]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions

class AudioMotion_mb(data.Dataset):
    def __init__(self, 
                 mofea264_dir,
                 audio_dir,
                 meta_paths, 
                 split,
                 pose_fps=30,
                #  audio_sr=16000,
                 pipeline=None,
                 **kwargs
                 ):
        self.mofea264_dir = mofea264_dir
        self.audio_dir = audio_dir
        # self.pipeline = Compose(pipeline)
        
        vid_meta = json.load(open(meta_paths, "r"))
        if split != 'all':
            self.vid_meta = [item for item in tqdm(vid_meta) if item.get("mode") == split]
        else:
            print('load all data')
            self.vid_meta = [item for item in tqdm(vid_meta)]
        
        
        
        # debug!!!
        # self.vid_meta = self.vid_meta[:900]
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
        # mofea264 = np.load(os.path.join(self.mofea264_dir, seqname+ '.npy'))[sdx:edx].astype(np.float32)
        # audio = np.load(os.path.join(self.audio_dir, seqname+'.npy')).astype(np.float32)
        # audio = audio[sdx:edx,:55]      # audio帧率和motion一致
        seqname = seqname + '@' + str(sdx) + '_' + str(edx)
        
        return seqname
        # mofea264 = np.zeros([2,264], dtype=np.float32)
        # audio = np.zeros([2,55], dtype=np.float32)
        
        return seqname, audio, mofea264   
    
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
    audio_dir = '/data2/hzy/InfiniteDance/InfiniteDanceData/music/musicfeature_55_allmusic_pure'
    motion_dir = '/data2/hzy/InfiniteDance/InfiniteDanceData/dance/alldata_new_joint_vecs264'
    split_dir = '/data2/hzy/InfiniteDance/InfiniteDanceData/partition/'
    slides = 150                #change
    length = 600
    save_json  = f'/data2/hzy/InfiniteDance/InfiniteDanceData/partition/alldata_mbr_s{slides}_l{length}.json'
    
    
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
            mopath = os.path.join(motion_dir, seqname+'.npy')
            mupath = os.path.join(audio_dir, seqname+'.npy')
            # jointspath = os.path.join(joints_dir, seqname+'.npy')
            if not os.path.exists(mopath):
                continue
            if not os.path.exists(mupath):
                continue
            
            try:
                musicfeas = np.load(mupath)
            except:
                print(mupath)
            music_frame = musicfeas.shape[0]
            motionfeas = np.load(mopath)
            motion_frame = motionfeas.shape[0]
            # joints_frame = np.load(jointspath).shape[0]
            maxframe = min(motion_frame, music_frame)    # joints_frame

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
        
        
        
