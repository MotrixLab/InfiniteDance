#import sys
import os
import pickle
from os.path import join as pjoin

#import codecs as cs
#import pandas as pd
#import numpy as np
import torch
#from common.quaternion import *
from paramUtil import *
from pytorch3d.transforms import (matrix_to_axis_angle,
                                  quaternion_to_axis_angle,
                                  rotation_6d_to_matrix)
from smplfk import SMPLX_Skeleton
from tqdm import tqdm

# 打开.pkl文件
#with open('/data2/lrh/datasets/Dance/share_dance_datasets/dataset/AIOZ-Gdance/motion_single_pkl/_P-JWcq1ewI_01_187_1358.pkl', 'rb') as f:
    #data = pickle.load(f)
#for key in data.keys():
    #print(key)
#print(data['smpl_poses'].shape) #T,72
#print(data['root_trans'].shape) #T,3






def ax_from_6v(data):
    data = rotation_6d_to_matrix(data)
    data = matrix_to_axis_angle(data)
    return data


#处理aioz pkl2joints
if __name__ == '__main__':

    # -------------------------------Step1-----------------------------------
    #AIOZ pkl路径，包含了smplx的信息
    modir = '/data2/lrh/datasets/Dance/share_dance_datasets/dataset/AIOZ-Gdance/motion_single_pkl'
    save_dir ='/data2/ss/momask-codes-main/dataset/AIOZ/mofea266/joints'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fps = 30
    #这是什么？
    smplxfk = SMPLX_Skeleton()

    for file in tqdm(os.listdir(modir)):
        if file[-3:] != 'pkl':
            continue
        #路径设置
        mofile = pjoin(modir, file)
        #读取dict
        with open(mofile,'rb') as f:
            modict=pickle.load(f)
        #modata = np.load(mofile)
        #modata = modata[:,4:]       # T,135 3+55*4=223
        trans = modict['root_trans']
        raw_axis = modict['smpl_poses'].reshape(-1, 24, 3) 
        axis = np.zeros((raw_axis.shape[0], 55, 3), dtype=raw_axis.dtype)
        axis[:, :24, :] = raw_axis
        #qua = torch.from_numpy(qua)
        trans = torch.from_numpy(trans)
        axis = torch.from_numpy(axis).view(-1,165)
        #print(trans.shape)
        #axis = quaternion_to_axis_angle(qua).view(modata.shape[0], 165)
        #print(axis.shape)
        data = smplxfk.forward(axis, trans).squeeze(0).detach().cpu().numpy()
        #print("data.shape",data.shape)
        #print(file[:-3])
        np.save(pjoin(save_dir, file[:-3]+'npy'), data)

    # -------------------------------Step2-----------------------------------



