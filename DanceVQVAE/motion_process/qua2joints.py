#import codecs as cs
#import pandas as pd
#import numpy as np
#import sys
import os
from os.path import join as pjoin

import torch
#from common.quaternion import *
from paramUtil import *
from pytorch3d.transforms import (matrix_to_axis_angle,
                                  quaternion_to_axis_angle,
                                  rotation_6d_to_matrix)
from smplfk import SMPLX_Skeleton
from tqdm import tqdm


def ax_from_6v(data):
    data = rotation_6d_to_matrix(data)
    data = matrix_to_axis_angle(data)
    return data



if __name__ == '__main__':

    # -------------------------------Step1-----------------------------------
    
    modir = "/data2/ss/T2M-GPT-main/dataset/NEW/QUA_30fps"
    save_dir ='/data2/ss/T2M-GPT-main/dataset/NEW/joints'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fps = 30
    #这是什么？
    smplxfk = SMPLX_Skeleton()

    for file in tqdm(os.listdir(modir)):
        if file[-3:] != 'npy':
            continue
        mofile = pjoin(modir, file)
        modata = np.load(mofile)
        #modata = modata[:,4:]       # T,135 3+55*4=223
        trans = modata[:,:3] 
        qua = modata[:,3:].reshape(modata.shape[0], 55, 4) 

        qua = torch.from_numpy(qua)
        trans = torch.from_numpy(trans)
        #print(trans.shape)
        axis = quaternion_to_axis_angle(qua).view(modata.shape[0], 165)
        #print(axis.shape)
        data = smplxfk.forward(axis, trans).squeeze(0).detach().cpu().numpy()
        print("data.shape",data.shape)

        np.save(pjoin(save_dir, file), data)

    # -------------------------------Step2-----------------------------------



