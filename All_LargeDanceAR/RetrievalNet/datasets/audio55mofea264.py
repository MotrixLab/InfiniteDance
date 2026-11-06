import json
# import librosa
import os
import pickle as pkl
import sys

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

from RetrievalNet.paired_data import AudioMotion_mb

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
        
        
        
