import ast
import json
import os
import warnings
from datetime import datetime, timedelta
from os.path import join as pjoin

import numpy as np
import torch

warnings.filterwarnings('ignore')

import argparse
import json
from argparse import Namespace

import numpy as np
from tqdm import tqdm

import DanceVQVAE.models.vqvae as vqvae
from All_LargeDanceAR.utils.quaternion import qinv, qrot

# 提取参数并转换为对象

# log_file_path = '/data1/hzy/HumanMotion/T2M-GPT_mofea264/output/exp_momask_finedance_mofea264_250323_1754/run.log'

parser = argparse.ArgumentParser(description='Load motion data and statistics')

# 定义命令行参数
parser.add_argument('--checkpoint_path', type=str, default='/data1/hzy/HumanMotion/T2M-GPT_mofea264/output/exp_momask_aistpp_mofea264_250619_0859/net_best_loss.pth',
                    help='File name for motion data')
parser.add_argument('--data_root', type=str, default='/data1/hzy/HumanMotion/InfiniteDance/All_mofea/ourAISTPP',
                    help='Root directory for data')
parser.add_argument('--mean_file', type=str, default='/data1/hzy/HumanMotion/InfiniteDance/All_mofea/ourAISTPP/Mean.npy',
                    help='File name for mean values')
parser.add_argument('--std_file', type=str, default='/data1/hzy/HumanMotion/InfiniteDance/All_mofea/ourAISTPP/Std.npy',
                    help='File name for standard deviation values')
parser.add_argument('--motion_path', type=str, default='/data1/hzy/HumanMotion/InfiniteDance/All_mofea/ourAISTPP/new_joint_vecs264/test',
                    help='File name for motion data')

args = parser.parse_args()
checkpoint_path=args.checkpoint_path
mean_path = args.mean_file
std_path = args.std_file
mean = np.load(mean_path)
std = np.load(std_path)
motion_path=args.motion_path

opt_path=os.path.join(os.path.dirname(checkpoint_path),"args.json")
with open(opt_path, 'r') as f:
    opt_dict = json.load(f)

# 转换为 Namespace 对象
opt = Namespace(**opt_dict)
name=os.path.basename(os.path.dirname(checkpoint_path))
current_time = (datetime.now() + timedelta(hours=7.92)).strftime(f"25%m%d_%H%M")
output_dir = f"{args.data_root}/test_mofea264_{name}"
os.makedirs(output_dir, exist_ok=True)
motion_files = [f for f in os.listdir(motion_path) if f.endswith('.npy')]
# 初始化模型
dim_pose=264
import models.vqvae as vqvae

net = vqvae.RVQVAE(    opt, ## use args to define different parameters in different quantizers
                       dim_pose,
                       opt.nb_code,
                       opt.code_dim,
                       opt.output_emb_width,
                       opt.down_t,
                       opt.stride_t,
                       opt.width,
                       opt.depth,
                       opt.dilation_growth_rate,
                       opt.vq_act,
                       opt.vq_norm)

checkpoint = torch.load(checkpoint_path, map_location='cpu')
net.load_state_dict(checkpoint['net'], strict=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)
net.eval()  # 设置为评估模式




def recover_from_ric264(data,joints_num):
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

for motion in tqdm(motion_files, desc="Processing motions", unit="file"):
    file_name = os.path.basename(motion).replace('.npy', '_reconstructed.npy')
    motion_path_full = os.path.join(motion_path, motion)

    # 读取并归一化
    motion_data = np.load(motion_path_full)
    motion_data = (motion_data - mean) / std
    motion_tensor = torch.from_numpy(motion_data).float().to(device)
    motion_tensor = motion_tensor.unsqueeze(0)

    # 预测
    with torch.no_grad():
        pred_motion, _,rot_out, _, _,_ = net(motion_tensor)
        pred_motion_np = pred_motion.cpu().numpy()

    # 反归一化
    pred_motion_np = pred_motion_np * std + mean

    # 保存处理后的数据
    file_path = pjoin(output_dir, file_name)
    pred_motion_np = pred_motion_np.squeeze(0)

    # 还原为 22 个关键点
    data_tensor = torch.from_numpy(pred_motion_np).float()
    joints_num = 22
    reconstructed_positions = recover_from_ric264(data_tensor, joints_num)
    reconstructed_positions_np = reconstructed_positions.cpu().numpy()

    np.save(file_path, reconstructed_positions_np)
    
import subprocess

# 设置参数
gt_root = "/data1/hzy/HumanMotion/All_mofea/FineDance/new_joint_vecs264"
pred_root = output_dir

# 调用 metrics_finedance.py 并传递参数
result = subprocess.run(
    [
        "python", "/data1/hzy/HumanMotion/Bailando/utils/metrics_new_aistpp.py",
        "--gt_root", gt_root,
        "--pred_root", pred_root
    ],
    capture_output=True,  # Capture stdout and stderr
    text=True  # Return output as a string instead of bytes
)

# Check if the script ran successfully
if result.returncode != 0:
    # print("Error running script:", result.stderr)
    raise Exception("Script failed")

# Get the output (assuming the script prints the metrics dictionary)
output = result.stdout.strip()
# breakpoint()

# breakpoint()
# Convert the string representation of the dictionary to a Python dictionary
# metrics = json.loads(output)
# Now you can use the metrics
print(output)
metrics = ast.literal_eval(output)
# print("FID Kinetic:", metrics['fid_k'].real)
# print("FID Manual:", metrics['fid_m'])
