import argparse
import json
import os
import time
import warnings
from datetime import datetime, timedelta
from os.path import join as pjoin

import models.vqvae as vqvae
import numpy as np
import options.option_vq as option_vq
import torch
import torch.optim as optim
import utils.losses as losses
import utils.utils_model as utils_model
from dataset import dataset_TM_eval, dataset_VQ
from models.evaluator_wrapper import EvaluatorModelWrapper
from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import paramUtil

warnings.filterwarnings('ignore')
import sys

from utils.word_vectorizer import WordVectorizer

sys.path.append('/data1/hzy/HumanMotion/LDmofea266')
import numpy as np
from common.quaternion import qinv, qrot

# 定义命令行参数解析器
parser = argparse.ArgumentParser()
'''
python /data1/hzy/HumanMotion/T2M-GPT/Eval_Reconstruct.py --batch-size 256 \--lr 2e-4 \--total-iter 300000 \--lr-scheduler 200000 \--nb-code 2048 \--code-dim 1024 \--down-t 2 \--depth 3 \--dilation-growth-rate 3 \--out-dir output \--dataname all \--vq-act relu \--quantizer ema_reset \--loss-vel 0.5 \--recons-loss l1_smooth  \--output-emb-width 1024
'''

# args = parser.parse_args()
args = option_vq.get_args_parser()
# 初始化模型
net = vqvae.HumanVQVAE(args,
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate,
                       args.vq_act,
                       args.vq_norm)

# 加载模型权重
checkpoint_path = '/data1/hzy/HumanMotion/T2M-GPT_mofea264/output/exp_momask_alldata_mofea264_250425_1522/net_best_loss.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
net.load_state_dict(checkpoint['net'], strict=True)

# 将模型移动到GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)
net.eval()  # 设置为评估模式
# 在初始化模型后添加以下代码
print(net)
quantizer = net.vqvae.quantizer  # 获取量化器模块
if hasattr(quantizer, 'codebook'):
    codebook = quantizer.codebook
    print("Codebook shape:", codebook.shape)
    print("Codebook content:", codebook)
else:
    print("The quantizer does not have a 'codebook' attribute.")



# val_dataset = dataset_VQ.MotionDataset(args, split_file='/data1/hzy/HumanMotion/All_mofea/DataSet_Split/ALL_eval.txt',
#                                         data_root='/data1/hzy/HumanMotion/All_mofea')
val_dataset = dataset_VQ.MotionDataset(args, split_file='/data1/hzy/HumanMotion/All_mofea/DataSet_Split/ALL_eval.txt',
                                        data_root='/data1/hzy/HumanMotion/All_mofea')
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, num_workers=4,
                         shuffle=False, pin_memory=True)

# 初始化损失函数
Loss = losses.ReConsLoss(args.recons_loss, args.nb_joints)



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
def evaluate_model(net, val_loader, Loss, device, args):
    net.eval()
    total_loss = 0
    total_commit_loss = 0
    total_perplexity = 0
    num_batches = 0
    with torch.no_grad():
        for batch_data in val_loader:
            batch_data = batch_data.to(device).float()
            # 前向传播
            pred_motion, loss_commit, perplexity = net(batch_data)
            # 计算损失
            loss_motion = Loss(pred_motion, batch_data)
            loss_vel = Loss.forward_vel(pred_motion, batch_data)
            loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel
            total_loss += loss.item()
            total_commit_loss += loss_commit.item()
            total_perplexity += perplexity.item()
            num_batches += 1
    # 计算平均损失
    avg_loss = total_loss / num_batches
    avg_commit_loss = total_commit_loss / num_batches
    avg_perplexity = total_perplexity / num_batches
    return avg_loss, avg_commit_loss, avg_perplexity


# print(f"Reconstructed files are saved in {output_dir}")
parser = argparse.ArgumentParser(description='Load motion data and statistics')

# 定义命令行参数
parser.add_argument('--data_root', type=str, default='/data1/hzy/HumanMotion/All_mofea/FineDance',
                    help='Root directory for data')
parser.add_argument('--mean_file', type=str, default='Mean.npy',
                    help='File name for mean values')
parser.add_argument('--std_file', type=str, default='Std.npy',
                    help='File name for standard deviation values')
parser.add_argument('--motion_path', type=str, default='/data1/hzy/HumanMotion/All_mofea/FineDance/',
                    help='File name for motion data')
args = parser.parse_args()

mean_path = f"{args.data_root}/{args.mean_file}"
std_path = f"{args.data_root}/{args.std_file}"
mean = np.load(mean_path)
std = np.load(std_path)
motion_path=args.motion_path
output_dir = f"{args.data_root}/test_reconstruct"
os.makedirs(output_dir, exist_ok=True)
for motion in os.listdir(motion_path):
    if not motion.endswith('.npy'):
        continue
    file_name = os.path.basename(motion).replace('.npy', '_reconstructed.npy')
    # breakpoint()
    motion=os.path.join(motion_path,motion)
    # print(motion)
    motion = np.load(motion)
    motion=(motion-mean)/std
    motion=torch.from_numpy(motion).float().to(device)
    
    motion=motion.unsqueeze(0)
    with torch.no_grad():
        pred_motion,pred_motion_xz, _, _ ,_= net(motion) #加了局部速度预测头
        # 将预测结果转换为numpy数组
        pred_motion[..., 3:5]=pred_motion_xz
        pred_motion_np = pred_motion.cpu().numpy()
        # 反归一化
        pred_motion_np = pred_motion_np * std + mean

    
    file_path = pjoin(output_dir, file_name)
    pred_motion_np = pred_motion_np.squeeze(0)
    # np.save(file_path, pred_motion_np)
    
    data = pred_motion_np
    data_tensor = torch.from_numpy(data).float()
    joints_num = 22
    reconstructed_positions = recover_from_ric264(data_tensor, joints_num)
    reconstructed_positions_np = reconstructed_positions.cpu().numpy()
    # npy_save_path = f'/data1/hzy/HumanMotion/T2M-GPT/results/reconstruct_joints/{file_name}'
    np.save(file_path, reconstructed_positions_np)

    print(f"Reconstructed result(recovered to 22 points) saved in {file_path}")
    # print("Reconstructed motion shape:", pred_motion_np.shape)
