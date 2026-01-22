import os
import subprocess
import sys

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D

sys.path.append('/data1/hzy/HumanMotion/LDmofea266')
import argparse

from common.quaternion import qinv, qrot


# Recover global joints from root positions
def recover_from_ric264(data, joints_num):
    r_pos_y = data[:, 0]
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

# Main function to process all .npy files
def process_npy_files(input_dir, output_dir, joints_num=22):
    # Check if output directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all .npy files in the input directory
    npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

    for npy_file in npy_files:
        # Load the data from the .npy file
        reconstructed_file_path = os.path.join(input_dir, npy_file)
        data = np.load(reconstructed_file_path)
        print("date.shape",data.shape)
        # Convert to tensor and process
        data_tensor = torch.from_numpy(data).float()
        reconstructed_positions = recover_from_ric264(data_tensor, joints_num)
        
        # Convert the result back to numpy
        reconstructed_positions_np = reconstructed_positions.cpu().numpy()

        # Save the result to the output directory
        output_file_path = os.path.join(output_dir, npy_file)
        np.save(output_file_path, reconstructed_positions_np)
        print(f"Processed and saved: {output_file_path} ,{reconstructed_positions.shape}")

if __name__ == "__main__":
    input_dir = '/data1/hzy/HumanMotion/InfiniteDance/ListenDenoiseAction/results/generated/alldata/'
    output_dir = '/data1/hzy/HumanMotion/InfiniteDance/ListenDenoiseAction/results/generated/alldata/joints'
    
    process_npy_files(input_dir, output_dir)