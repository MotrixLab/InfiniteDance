import argparse
import io
import os
import multiprocessing as mp
from multiprocessing import Pool

import imageio
import matplotlib
matplotlib.use('Agg')  # 必须在导入 pyplot 之前设置
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm
def qinv(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

def recover_from_ric264(data, joints_num=22):
    """从 264 维数据恢复 22 个关节的全局位置"""
    data_tensor = torch.from_numpy(data).float()
    r_pos_y = data_tensor[:, 0]
    rot_vel = data_tensor[..., 2]
    r_rot_ang = torch.zeros_like(rot_vel)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data_tensor.shape[:-1] + (4,))
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data_tensor.shape[:-1] + (3,))
    r_pos[..., 1:, [0, 2]] = data_tensor[..., :-1, 3:5]
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data_tensor[..., 0]

    positions = data_tensor[..., 5:(joints_num - 1) * 3 + 5]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions.cpu().numpy()
def convert_to_tensor(data):
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).float()  
    elif isinstance(data, dict):
        converted_data = {}
        for key, value in data.items():
            converted_data[key] = convert_to_tensor(value)
        return converted_data
    elif isinstance(data, list):
        return [convert_to_tensor(item) for item in data]
    elif isinstance(data, torch.Tensor):
        return data.float() 
    return data

def custom_wrap(text, width):
    return '\n'.join([text[i:i+width] for i in range(0, len(text), width)])




def plot_3d_motion(args, figsize=(10, 10), fps=120, radius=4):
    # matplotlib.use('Agg') 已在文件开头设置  

    joints, out_name, title = args

    data = joints.copy().reshape(len(joints), -1, 3)

    nb_joints = joints.shape[1]
    full_joint_chain = [
    # 身体关节（22个）
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 

    # 头部关节（3个）
    'jaw', 'left_eye_smplhf', 'right_eye_smplhf', 

    # 左手关节（15个）
    'left_index1', 'left_index2', 'left_index3', 'left_middle1', 'left_middle2', 
    'left_middle3', 'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1', 
    'left_ring2', 'left_ring3', 'left_thumb1', 'left_thumb2', 'left_thumb3', 

    # 右手关节（15个）
    'right_index1', 'right_index2', 'right_index3', 'right_middle1', 'right_middle2', 
    'right_middle3', 'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_ring1', 
    'right_ring2', 'right_ring3', 'right_thumb1', 'right_thumb2', 'right_thumb3'
        ]

    smpl_kinetic_chain =  [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
    left_hand_joints = [
        (20, 25, 26, 27),  # 食指：left_wrist -> left_index1 -> left_index2 -> left_index3
        (20, 28, 29, 30),  # 中指：left_wrist -> left_middle1 -> left_middle2 -> left_middle3
        (20, 31, 32, 33),  # 无名指：left_wrist -> left_pinky1 -> left_pinky2 -> left_pinky3
        (20, 34, 35, 36),  # 小指：left_wrist -> left_ring1 -> left_ring2 -> left_ring3
        (20, 37, 38, 39)   # 拇指：left_wrist -> left_thumb1 -> left_thumb2 -> left_thumb3
    ]

    right_hand_joints = [
        (21, 40, 41, 42),  # 食指：right_wrist -> right_index1 -> right_index2 -> right_index3
        (21, 43, 44, 45),  # 中指：right_wrist -> right_middle1 -> right_middle2 -> right_middle3
        (21, 46, 47, 48),  # 无名指：right_wrist -> right_pinky1 -> right_pinky2 -> right_pinky3
        (21, 49, 50, 51),  # 小指：right_wrist -> right_ring1 -> right_ring2 -> right_ring3
        (21, 52, 53, 54)   # 拇指：right_wrist -> right_thumb1 -> right_thumb2 -> right_thumb3
    ]
    full_kinetic_chain = [
        [0, 2, 5, 8, 11],  # 脊柱链
        [0, 1, 4, 7, 10],  # 左髋链
        [0, 3, 6, 9, 12, 15],  # 右髋链
        [9, 14, 17, 19, 21],  # 左腿链
        [9, 13, 16, 18, 20],  # 右腿链
        # [20, 25, 26, 27],  # 食指：left_wrist -> left_index1 -> left_index2 -> left_index3
        # [20, 28, 29, 30],  # 中指：left_wrist -> left_middle1 -> left_middle2 -> left_middle3
        # [20, 31, 32, 33],  # 无名指：left_wrist -> left_pinky1 -> left_pinky2 -> left_pinky3
        # [20, 34, 35, 36],  # 小指：left_wrist -> left_ring1 -> left_ring2 -> left_ring3
        # [20, 37, 38, 39],  # 拇指：left_wrist -> left_thumb1 -> left_thumb2 -> left_thumb3
        # [21, 40, 41, 42],  # 食指：right_wrist -> right_index1 -> right_index2 -> right_index3
        # [21, 43, 44, 45],  # 中指：right_wrist -> right_middle1 -> right_middle2 -> right_middle3
        # [21, 46, 47, 48],  # 无名指：right_wrist -> right_pinky1 -> right_pinky2 -> right_pinky3
        # [21, 49, 50, 51],  # 小指：right_wrist -> right_ring1 -> right_ring2 -> right_ring3
        # [21, 52, 53, 54]   # 拇指：right_wrist -> right_thumb1 -> right_thumb2 -> right_thumb3
    ]


    # [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4],
    #                       [3, 5, 6, 7], [3, 8, 9, 10]] if nb_joints == 21 else \
    #只考虑前22个关节
    limits = 1000 if nb_joints == 21 else 2  
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)

    #y轴是重力轴
    height_offset = MINS[1]
    ground_offset = 0.5
    data[:, :, 1] -= (height_offset + ground_offset)  
    MINS = data.min(axis=0).min(axis=0)
    trajec = data[:, 0, [0, 2]]  
    data[..., 0] -= data[:, 0:1, 0] 
    data[..., 2] -= data[:, 0:1, 2]  

    def update(index):
        def init():
            ax.set_xlim(-limits, limits)
            ax.set_ylim(-limits, limits)
            ax.set_zlim(0, limits)
            ax.grid(visible=False)

        def plot_xzPlane(minx, maxx, miny, minz, maxz):
            verts = [
                [3*minx, 3*miny, 3*minz],
                [3*minx, 3*miny, 3*maxz],
                [3*maxx, 3*miny, 3*maxz],
                [3*maxx, 3*miny, 3*minz]
            ]
            xz_plane = Poly3DCollection([verts])
            xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
            ax.add_collection3d(xz_plane)

        fig = plt.figure(figsize=(10, 10), dpi=96)
        if title is not None:
            wrapped_title = custom_wrap(title, 40)
            fig.suptitle(wrapped_title, fontsize=16)
        ax = p3.Axes3D(fig, auto_add_to_figure=False)  
        fig.add_axes(ax)  
        init()

        ax.view_init(elev=110, azim=-90)
        ax.dist = 7.5

        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0,
                    MINS[2] - trajec[index, 1], MAXS[2] - trajec[index, 1])

        # 绘制人物骨架
        #,'red','red','red','red','red', 'blue', 'blue', 'blue', 'blue', 'blue'
        for i, (chain, color) in enumerate(zip(full_kinetic_chain, ['red', 'blue', 'green', 'orange', 'purple'])):
            linewidth = 4.0 if i < 5 else 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2],
                    linewidth=linewidth, color=color)

        # 绘制 trajec 轨迹 (在 xz 平面)
        # ax.plot(trajec[:index+1, 0], trajec[:index+1, 1], np.zeros_like(trajec[:index+1, 0]), color='black', linewidth=1)
        # if index > 1:
        #     ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
        #               trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
        #               color='blue')
        #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        if out_name is not None:
            plt.savefig(out_name, dpi=96)
            plt.close()
        else:
            io_buf = io.BytesIO()
            fig.savefig(io_buf, format='raw', dpi=96)
            io_buf.seek(0)
            arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                            newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
            io_buf.close()
            plt.close()
            return arr

    out = []
    length=data.shape[0]
    # length=min(100,length)
    # length=450
    for i in tqdm(range(length), desc="Rendering frames"):
        out.append(update(i))

    out = np.stack(out, axis=0)
    return torch.from_numpy(out)

def draw_to_batch(smpl_joints_batch, title_batch=None, outname=None):
    length = len(smpl_joints_batch)
    out = []
    # skip=3
    for i in range(0,length):
        out.append(plot_3d_motion([smpl_joints_batch[i], None, title_batch[i] if title_batch is not None else None]))
        if outname is not None:
            imageio.mimsave(outname[i], np.array(out[-1]), fps=30)
    out = torch.stack(out, axis=0)
    return out

def process_single_file(args_tuple):
    """处理单个文件的函数，用于多进程"""
    npy_path, joints_dir, overwrite = args_tuple
    
    try:
        folder = os.path.basename(npy_path).split('.npy')[0]
        renfer_dir = joints_dir
        os.makedirs(f'{renfer_dir}/render', exist_ok=True)
        
        final_video_path = f'{renfer_dir}/render/{folder}.mp4'
        
        # 如果文件已存在且不覆盖，跳过
        if os.path.exists(final_video_path) and not overwrite:
            return f"跳过已存在的文件: {folder}"
        
        J_3D = np.load(npy_path)
        J_3D = J_3D.squeeze()
        if J_3D.shape[1] == 264:
            J_3D = recover_from_ric264(J_3D)
        
        draw_to_batch([J_3D], title_batch=[f"{folder}"], outname=[final_video_path])
        return f"成功处理: {folder} -> 保存到: {final_video_path}"
    except Exception as e:
        return f"处理 {os.path.basename(npy_path)} 时出错: {str(e)}"

def parse_args():
    parser = argparse.ArgumentParser(description="Script to process pkl file")
    parser.add_argument(
        '--npy_path',
        type=str,
        default="./test.npy",
        help="Path to the npy file"
    )
    parser.add_argument(
        '--change',

        default=False,
        help="Path to the pickle file (default: /data1/hzy/Alldata_process/final_motion/Ballet/Ballet5/Ballet5.pkl)"
    )
    parser.add_argument(
        '--render_folder',

        default=True,
        help="Path to the pickle file (default: /data1/hzy/Alldata_process/final_motion/Ballet/Ballet5/Ballet5.pkl)"
    )
    parser.add_argument(
        '--joints_dir',
        type=str,
        default="./joints",
        help="Directory containing joint npy files"
    )
    parser.add_argument(
        '--num_processes',
        type=int,
        default=8,
        help="Number of processes to use for parallel processing (default: 8)"
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        default=False,
        help="覆盖已存在的视频文件 (default: False)"
    )
    return parser.parse_args()




if __name__ == "__main__":
    args = parse_args()
    npy_path = args.npy_path
    change= args.change

    render_folder=args.render_folder
    overwrite = args.overwrite
    if render_folder==True:
        joints_dir=args.joints_dir
        num_processes = args.num_processes
        
        # 收集所有需要处理的文件
        npy_files = []
        for file in os.listdir(joints_dir):
            if not file.endswith(".npy"):
                continue
            npy_path = os.path.join(joints_dir, file)
            npy_files.append((npy_path, joints_dir, overwrite))
        
        print(f"找到 {len(npy_files)} 个 npy 文件，使用 {num_processes} 个进程进行处理...")
        print(f"输出目录: {joints_dir}/render/")
        if overwrite:
            print("覆盖模式: 已存在的文件将被重新生成")
        else:
            print("跳过模式: 已存在的文件将被跳过")
        
        # 使用多进程处理
        if num_processes > 1:
            mp.set_start_method('spawn', force=True)  # 确保使用 spawn 模式
            with Pool(processes=num_processes) as pool:
                results = list(tqdm(
                    pool.imap(process_single_file, npy_files),
                    total=len(npy_files),
                    desc="渲染视频"
                ))
            
            # 打印结果
            success_count = sum(1 for r in results if "成功处理" in r)
            skip_count = sum(1 for r in results if "跳过" in r)
            error_count = sum(1 for r in results if "出错" in r)
            
            print(f"\n处理完成!")
            print(f"成功: {success_count} 个文件")
            print(f"跳过: {skip_count} 个文件")
            print(f"错误: {error_count} 个文件")
            print(f"\n所有视频文件保存在: {joints_dir}/render/")
        else:
            # 单进程模式（向后兼容）
            for npy_path, joints_dir, overwrite_flag in tqdm(npy_files, desc="渲染视频"):
                result = process_single_file((npy_path, joints_dir, overwrite_flag))
                print(result)
    else:
        if not os.path.exists(npy_path):
            print(f"文件不存在: {npy_path}")
            exit(1)
        npy_path = args.npy_path
        folder = os.path.basename(npy_path).split('.npy')[0]
        os.makedirs('./output/render', exist_ok=True)
        final_video_path = f'./output/render/{folder}.mp4'        
        J_3D=np.load(npy_path)

        # breakpoint()
        if J_3D.shape[1]==264:
            print("recover jointd from mofea 264")
            J_3D=recover_from_ric264(J_3D)
        J_3D=J_3D[:,0:55]
        J_3D=J_3D.squeeze()
        # folder="test"
        # if not os.path.exists(final_video_path):
        draw_to_batch([J_3D], title_batch=[f"{folder}"], outname=[final_video_path])

        print(f"Animation saved to {final_video_path}")
        print(f"输出目录: /data2/hzy/InfiniteDance/All_LargeDanceAR/output/render/")