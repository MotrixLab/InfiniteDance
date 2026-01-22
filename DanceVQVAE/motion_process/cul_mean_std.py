import os
import sys
from os.path import join as pjoin

import numpy as np


# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def mean_variance(data_dir, save_dir, joints_num):
    file_list = os.listdir(data_dir)
    data_list = []

    for file in file_list:
        data = np.load(pjoin(data_dir, file))
        if np.isnan(data).any():
            print(file)
            continue
        data_list.append(data)

    data = np.concatenate(data_list, axis=0)
    print(data.shape)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)
    # Std[0:3] = Std[0:3].mean() / 1.0
    # Std[3:4] = Std[3:4].mean() / 1.0
    # Std[4:6] = Std[4:6].mean() / 1.0
    # # Std[3:4] = Std[3:4].mean() / 1.0
    # Std[6: 6+(joints_num - 1) * 3] = Std[6: 6+(joints_num - 1) * 3].mean() / 1.0
    # Std[6+(joints_num - 1) * 3: 6+(joints_num - 1) * 9] = Std[6+(joints_num - 1) * 3: 6+(joints_num - 1) * 9].mean() / 1.0
    # Std[6+(joints_num - 1) * 9: 6+(joints_num - 1) * 9 + joints_num*3] = Std[6+(joints_num - 1) * 9: 6+(joints_num - 1) * 9 + joints_num*3].mean() / 1.0
    # Std[6 + (joints_num - 1) * 9 + joints_num * 3: ] = Std[6 + (joints_num - 1) * 9 + joints_num * 3: ].mean() / 1.0
    # assert 8 + (joints_num - 1) * 9 + joints_num * 3 == Std.shape[-1]

    np.save(pjoin(save_dir, 'Mean_d+m.npy'), Mean)
    np.save(pjoin(save_dir, 'Std_d+m.npy'), Std)

    return Mean, Std

if __name__ == '__main__':
    # data_dir = '/data2/ss/T2M-GPT-main/dataset/ALL/mofea266/new_joint_vecs/'
    # save_dir = '/data2/ss/T2M-GPT-main/dataset/ALL/mofea266'
    data_dir = '/data1/hzy/HumanMotion/T2M-GPT/dataset/HumanML3D/new_joint_vecs'
    save_dir = '/data1/hzy/HumanMotion/T2M-GPT/dataset/HumanML3D'
    mean, std = mean_variance(data_dir, save_dir, 22)