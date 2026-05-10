"""
多GPU并行特征提取脚本
使用多进程在多个GPU上并行处理文件，大幅加速特征提取
"""
import numpy as np
import torch
import os
import argparse
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import sys

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from features.kinetic import extract_kinetic_features
from features.manual_new import extract_manual_features
from utils.quaternion import qrot, qinv


def recover_from_ric264(data, joints_num):
    """从264维表示恢复3D关节位置"""
    r_pos_y = data[:, 0]
    r_rot_ang = torch.zeros_like(data[:, 2]).to(data.device)
    r_rot_ang[1:] = data[:-1, 2]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=0)
    
    r_rot_quat = torch.zeros([data.shape[0], 4]).to(data.device)
    r_rot_quat[:, 0] = torch.cos(r_rot_ang)
    r_rot_quat[:, 2] = torch.sin(r_rot_ang)
    
    r_pos = torch.zeros([data.shape[0], 3]).to(data.device)
    r_pos[:, 1] = r_pos_y
    r_pos[1:, [0, 2]] = data[:-1, 3:5]
    r_pos = torch.cumsum(r_pos, dim=0)
    
    r_pos_expand = r_pos.unsqueeze(1).repeat(1, joints_num - 1, 1)
    local_joints = data[:, 5:].reshape(data.shape[0], joints_num - 1, 3)
    global_joints = qrot(r_rot_quat.unsqueeze(1).repeat(1, joints_num - 1, 1), local_joints) + r_pos_expand
    
    result = torch.cat([r_pos.unsqueeze(1), global_joints], dim=1)
    return result


def process_single_file(args_tuple):
    """处理单个文件（在子进程中运行）"""
    pkl, root, gpu_id = args_tuple
    
    try:
        # 在子进程中设置CUDA_VISIBLE_DEVICES，确保只使用指定的GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # 检查GPU是否可用
        if not torch.cuda.is_available():
            return False, f"CUDA not available"
        
        # 设置后GPU索引变为0
        device = torch.device('cuda:0')
        
        # 检查是否已存在
        kinetic_path = os.path.join(root, 'kinetic_features', pkl)
        manual_path = os.path.join(root, 'manual_features_new', pkl)
        if os.path.exists(kinetic_path) and os.path.exists(manual_path):
            return True, None
        
        # 加载数据
        data = np.load(os.path.join(root, pkl))
        
        # 处理不同格式的数据
        if len(data.shape) == 2:
            # 264维或266维表示
            if data.shape[1] == 264:
                data = data[:1024, :]
                data = torch.from_numpy(data).to(device).to(torch.float32)
                with torch.no_grad():
                    joint3d = recover_from_ric264(data, 22)[:, :24, :]
            elif data.shape[1] == 266:
                data = data[:1024, :]
                # 去掉第2列（index 1）
                data = np.concatenate([data[:, 1:2], data[:, 3:]], axis=1)
                data = torch.from_numpy(data).to(device).to(torch.float32)
                with torch.no_grad():
                    joint3d = recover_from_ric264(data, 22)[:, :24, :]
            else:
                return False, f"Unsupported 2D data shape: {data.shape}"
        elif len(data.shape) == 3:
            # 3D关节位置 (T, 22, 3)
            if data.shape[1] == 22 and data.shape[2] == 3:
                joint3d = torch.from_numpy(data).to(device).to(torch.float32)
                joint3d = joint3d[:1024, :, :]  # 限制到1024帧
            else:
                return False, f"Unsupported 3D data shape: {data.shape}"
        else:
            return False, f"Unsupported data shape: {data.shape}"
        
        # 转换为numpy
        joint3d = joint3d[:1024, :22, :]
        joint3d = joint3d.reshape(joint3d.shape[0], 22 * 3).detach().cpu().numpy()
        
        # 计算相对位置
        roott = joint3d[:1, :3]
        joint3d = joint3d - np.tile(roott, (1, 22))
        
        joint3d_relative = joint3d.copy()
        joint3d_relative = joint3d_relative.reshape(-1, 22, 3)
        joint3d_relative[:, 1:, :] = joint3d_relative[:, 1:, :] - joint3d_relative[:, 0:1, :]
        
        # 提取特征并保存
        np.save(kinetic_path, extract_kinetic_features(joint3d_relative.reshape(-1, 22, 3)))
        np.save(manual_path, extract_manual_features(joint3d_relative.reshape(-1, 22, 3)))
        
        return True, None
        
    except Exception as e:
        return False, f"Error processing {pkl}: {str(e)}"


def calc_and_save_feats_parallel(root, gpu_ids, max_samples=None, num_workers_per_gpu=2):
    """
    多GPU并行提取特征
    
    Args:
        root: 数据根目录
        gpu_ids: GPU ID列表，如 [0, 1, 2, 3]
        max_samples: 限制处理的文件数量
        num_workers_per_gpu: 每个GPU的worker数量
    """
    # 验证GPU可用性
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，无法使用多GPU模式")
    
    available_gpus = torch.cuda.device_count()
    print(f"检测到 {available_gpus} 个可用GPU")
    
    # 验证所有请求的GPU都可用
    invalid_gpus = [gpu_id for gpu_id in gpu_ids if gpu_id >= available_gpus]
    if invalid_gpus:
        raise ValueError(f"请求的GPU {invalid_gpus} 不可用！系统只有 {available_gpus} 个GPU (0-{available_gpus-1})")
    
    # 创建输出目录
    os.makedirs(os.path.join(root, 'kinetic_features'), exist_ok=True)
    os.makedirs(os.path.join(root, 'manual_features_new'), exist_ok=True)
    
    # 获取文件列表
    file_list = sorted([f for f in os.listdir(root) 
                       if f.endswith('.npy') and not os.path.isdir(os.path.join(root, f))])
    
    if max_samples is not None:
        file_list = file_list[:max_samples]
    
    # 统计需要处理的文件
    files_to_process = []
    files_already_exist = 0
    
    for pkl in file_list:
        kinetic_path = os.path.join(root, 'kinetic_features', pkl)
        manual_path = os.path.join(root, 'manual_features_new', pkl)
        if os.path.exists(kinetic_path) and os.path.exists(manual_path):
            files_already_exist += 1
        else:
            files_to_process.append(pkl)
    
    print(f"\n{'='*60}")
    print(f"特征提取统计:")
    print(f"  总文件数: {len(file_list)}")
    print(f"  已存在: {files_already_exist}")
    print(f"  需处理: {len(files_to_process)}")
    print(f"  使用GPU: {gpu_ids}")
    print(f"  总进程数: {len(gpu_ids) * num_workers_per_gpu}")
    print(f"{'='*60}\n")
    
    if len(files_to_process) == 0:
        print("✅ 所有特征已存在，跳过提取步骤")
        return
    
    # 为每个文件分配GPU
    tasks = []
    for i, pkl in enumerate(files_to_process):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        tasks.append((pkl, root, gpu_id))
    
    # 多进程处理（使用spawn模式以兼容CUDA）
    total_workers = len(gpu_ids) * num_workers_per_gpu
    print(f"🚀 启动 {total_workers} 个worker进程并行处理...")
    
    success_count = 0
    error_count = 0
    
    # 使用spawn上下文创建进程池（CUDA兼容）
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=total_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, tasks),
            total=len(tasks),
            desc="提取特征"
        ))
        
        for success, error_msg in results:
            if success:
                success_count += 1
            else:
                error_count += 1
                if error_msg:
                    print(f"❌ {error_msg}")
    
    print(f"\n{'='*60}")
    print(f"特征提取完成:")
    print(f"  成功: {success_count}")
    print(f"  失败: {error_count}")
    print(f"{'='*60}\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多GPU并行特征提取")
    parser.add_argument("--root", type=str, required=True, help="数据根目录")
    parser.add_argument("--gpus", type=str, default="0", help="GPU ID列表，用逗号分隔，如 '0,1,2,3'")
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数")
    parser.add_argument("--workers_per_gpu", type=int, default=2, help="每个GPU的worker数量")
    
    args = parser.parse_args()
    
    # 解析GPU列表
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    
    print(f"数据目录: {args.root}")
    print(f"GPU列表: {gpu_ids}")
    print(f"每GPU Worker数: {args.workers_per_gpu}")
    
    calc_and_save_feats_parallel(
        root=args.root,
        gpu_ids=gpu_ids,
        max_samples=args.max_samples,
        num_workers_per_gpu=args.workers_per_gpu
    )


if __name__ == '__main__':
    # 必须在主进程中设置，确保spawn模式正常工作
    main()
