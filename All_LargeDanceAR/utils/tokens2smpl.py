'''
tranfer motion tokens obtained from InfiniteDance inference to smpl joints and motion features
'''
import argparse
import io
import multiprocessing as mp
from multiprocessing import Pool
import json
import os
import re
import sys
from argparse import Namespace
from datetime import datetime

# Set root and data directories relative to this script
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(UTILS_DIR, ".."))
DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, "../InfiniteDanceData"))

sys.path.append(ROOT_DIR)

import imageio
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm
import models.vqvae as vqvae

# Set CUDA device (uncomment if needed)
# torch.cuda.set_device(1)

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

def process_initial_data(file_path, increment=512):
    data = np.load(file_path)
    data = data.reshape(-1)
    if data.shape[0] % 3 != 0:
        raise ValueError("Data length is not a multiple of 3, cannot process in triplets!")
    
    data_reshaped = data.reshape(-1, 3)
    
    def process_triplet(a, b, c):
        a_new = a - 4096
        b_new = b - (4096 + increment * 1)
        c_new = c - (4096 + increment * 2)
        return a_new, b_new, c_new
    
    result = np.array([process_triplet(row[0], row[1], row[2]) for row in data_reshaped])
    return result

def load_vqvae_model(checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(ROOT_DIR, "models/checkpoints/dance_vqvae.pth")
    
    opt_path = os.path.join(os.path.dirname(checkpoint_path), "args.json")
    with open(opt_path, 'r') as f:
        opt_dict = json.load(f)
    
    args = Namespace(**opt_dict)
    dim_pose = 264
    net = vqvae.RVQVAE(args, dim_pose, args.nb_code, args.code_dim, args.output_emb_width, args.down_t, args.stride_t, 
                       args.width, args.depth, args.dilation_growth_rate, args.vq_act, args.vq_norm)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    net.load_state_dict(checkpoint['net'], strict=False)  # Use strict=False for compatibility with parameters like vel_decoder
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.eval()
    
    # Debug: Print codebooks shapes and args configuration
    codebooks = [net.quantizer.layers[i].codebook.data for i in range(args.num_quantizers)]
    for i, cb in enumerate(codebooks):
        print(f"Codebook {i} shape: {cb.shape}")
    print(f"args.code_dim: {args.code_dim}, args.dim_pose: {dim_pose}, args.num_quantizers: {args.num_quantizers}")
    
    return net, args

def load_mean_std(mean_path=None, std_path=None):
    if mean_path is None:
        mean_path = os.path.join(DATA_DIR, "dance/alldata_new_joint_vecs264/meta/Mean.npy")
    if std_path is None:
        std_path = os.path.join(DATA_DIR, "dance/alldata_new_joint_vecs264/meta/Std.npy")
        
    mean = np.load(mean_path)
    std = np.load(std_path)
    return mean, std

def motion_embedding(motion_idx, codebooks, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # Ensure motion_idx is on the correct device
    B, L, G = motion_idx.shape
    D = codebooks[0].shape[1]
    quantized_vectors = torch.zeros(B, L, D, device=device)  # Initialize zero tensor on specified device
    for g in range(G):
        # Get the g-th codebook and corresponding indices
        codebook_g = codebooks[g].to(device)  # Shape: [K, D]
        indices_g = motion_idx[:, :, g]       # Shape: [B, L]
        
        # Extract vectors from the codebook and accumulate
        vectors_g = codebook_g[indices_g]     # Shape: [B, L, D]
        quantized_vectors += vectors_g         # Residual accumulation
    
    return quantized_vectors

def motion_detokenizer(quantized_vectors, net, mean, std):
    """Reconstruct motion data from quantized vectors"""
    # Debug: Print input shape
    # print(f"Input to net shape: {quantized_vectors.shape}")
    
    # Call net's decoder and vel_decoder
    pred_motion = net.decoder(quantized_vectors.permute(0, 2, 1))
    pred_motion_xz = net.vel_decoder(quantized_vectors.permute(0, 2, 1))
    
    pred_motion = pred_motion.squeeze(0).cpu().detach().numpy()
    pred_motion_xz = pred_motion_xz.squeeze(0).cpu().detach().numpy()
    pred_motion[..., 3:5] = pred_motion_xz  # Update XZ velocity
    dance_output = pred_motion * std + mean
    
    return dance_output

def reconstruct_from_tokens(motion_idx, output_dir, net, codebooks, mean, std, device, filename_prefix="reconstructed"):
    """Reconstruct motion data from motion tokens and save"""
    if isinstance(motion_idx, np.ndarray):
        motion_idx = torch.from_numpy(motion_idx).to(device).unsqueeze(0)
    elif not isinstance(motion_idx, torch.Tensor):
        raise ValueError("motion_idx must be a numpy array or torch.Tensor!")
    
    if len(motion_idx.shape) != 3:
        raise ValueError("motion_idx must be a 3D tensor of shape [B, L, G]!")
    
    # Debug: Print motion_idx shape
    print(f"motion_idx shape: {motion_idx.shape}")
    quantized_vectors = motion_embedding(motion_idx, codebooks, device)
    dance_output = motion_detokenizer(quantized_vectors, net, mean, std)
    
    os.makedirs(output_dir, exist_ok=True)
    return dance_output

def recover_from_ric264(data, joints_num=22):
    """Recover global positions of 22 joints from 264-dimensional data"""
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

def custom_wrap(text, width):
    return '\n'.join([text[i:i+width] for i in range(0, len(text), width)])

def plot_3d_motion(args, figsize=(10, 10), fps=120, radius=4):
    matplotlib.use('Agg')  

    joints, out_name, title = args

    data = joints.copy().reshape(len(joints), -1, 3)

    nb_joints = joints.shape[1]
    full_kinetic_chain = [
        [0, 2, 5, 8, 11],  # Spine chain
        [0, 1, 4, 7, 10],  # Left hip chain
        [0, 3, 6, 9, 12, 15],  # Right hip chain
        [9, 14, 17, 19, 21],  # Left leg chain
        [9, 13, 16, 18, 20],  # Right leg chain
    ]

    limits = 1000 if nb_joints == 21 else 2  
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)

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

        for i, (chain, color) in enumerate(zip(full_kinetic_chain, ['red', 'blue', 'green', 'orange', 'purple'])):
            linewidth = 4.0 if i < 5 else 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2],
                    linewidth=linewidth, color=color)

        if index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                      trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                      color='blue')

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
    length = data.shape[0]
    for i in tqdm(range(length), desc="Rendering frames"):
        out.append(update(i))

    out = np.stack(out, axis=0)
    return torch.from_numpy(out)

def draw_to_batch(smpl_joints_batch, title_batch=None, outname=None):
    length = len(smpl_joints_batch)
    out = []
    for i in range(0, length):
        out.append(plot_3d_motion([smpl_joints_batch[i], None, title_batch[i] if title_batch is not None else None]))
        if outname is not None:
            imageio.mimsave(outname[i], np.array(out[-1]), fps=30)
    out = torch.stack(out, axis=0)
    return out

def process_npy_file(input_npy_path, final_output_dir, checkpoint_path, mean_path, std_path):
    processed_data = process_initial_data(input_npy_path)
    
    net, args = load_vqvae_model(checkpoint_path=checkpoint_path)
    mean, std = load_mean_std(mean_path=mean_path, std_path=std_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    codebooks = [net.quantizer.layers[i].codebook.data for i in range(args.num_quantizers)]
    
    temp_output_dir = os.path.join(DATA_DIR, "dance/temp/sliding_window_results")
    dance_output = reconstruct_from_tokens(processed_data, temp_output_dir, net, codebooks, mean, std, device)
    
    file_name264 = os.path.basename(input_npy_path).replace('.npy', '_mofea264.npy')
    final_output_dir_joints = os.path.join(final_output_dir, "joints")
    final_output_dir_mofea264 = os.path.join(final_output_dir, "mofea264")
    os.makedirs(final_output_dir_mofea264, exist_ok=True)
    os.makedirs(final_output_dir_joints, exist_ok=True)
    final_output_path264 = os.path.join(final_output_dir_mofea264, file_name264)
    np.save(final_output_path264, dance_output)
    print(f"264-dim mofea saved to: {final_output_path264}")
    
    reconstructed_positions = recover_from_ric264(dance_output)
    
    os.makedirs(final_output_dir, exist_ok=True)
    file_name = os.path.basename(input_npy_path)
    final_output_path_joints = os.path.join(final_output_dir_joints, file_name)
    np.save(final_output_path_joints, reconstructed_positions)
    print(f"Final reconstructed joint positions saved to: {final_output_path_joints}")
    
    return reconstructed_positions, final_output_path_joints, dance_output

def render_npy_file(npy_path):
    if not os.path.exists(npy_path):
        print(f"File does not exist: {npy_path}")
        return False
    
    basename = os.path.basename(npy_path).split('.npy')[0]
    output_dir = os.path.dirname(npy_path)
    final_video_path = os.path.join(output_dir, f"{basename}.mp4")
    if os.path.exists(final_video_path):
        print(f"{final_video_path} exists")
        return final_video_path, basename
    
    J_3D = np.load(npy_path)
    draw_to_batch([J_3D], title_batch=[f"{basename}"], outname=[final_video_path])
    
    print(f"saved to {final_video_path}")
    return final_video_path, basename

def natural_sort_key(filename):
    parts = re.split(r'(\d+)', filename)
    return [int(part) if part.isdigit() else part.lower() for part in parts]

def process_single_file(args):
    file, npy_dir, out_dir, checkpoint_path, mean_path, std_path = args
    try:
        if not file.endswith('.npy'):
            return None
            
        input_npy_path = os.path.join(npy_dir, file)
        print(f"Processing {input_npy_path} in process {os.getpid()}...")
        
        reconstructed_positions, final_output_path, dance_output = process_npy_file(
            input_npy_path=input_npy_path, 
            final_output_dir=out_dir,
            checkpoint_path=checkpoint_path,
            mean_path=mean_path,
            std_path=std_path
        )
        
        return final_output_path
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")
        return None

def process_files_parallel(npy_files, npy_dir, out_dir, checkpoint_path, mean_path, std_path, num_processes):
    task_args = [(file, npy_dir, out_dir, checkpoint_path, mean_path, std_path) for file in npy_files]
    
    with Pool(processes=num_processes) as pool:
        results = list(pool.imap(process_single_file, task_args))
    
    successful_results = [r for r in results if r is not None]
    print(f"Successfully processed {len(successful_results)}/{len(npy_files)} files")
    return successful_results

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True) 
    parser = argparse.ArgumentParser(description="Process and render dance motion data")
    parser.add_argument('--mode', type=str, default="process", choices=["process", "render", "both"],
                        help="Mode of operation: 'process' for processing NPY files, 'render' for rendering, 'both' for both operations")
    parser.add_argument('--npy_dir', type=str, 
                        default=".",
                        help="Directory path for dance motion data")
    parser.add_argument('--checkpoint_path', type=str, 
                        default=os.path.join(ROOT_DIR, "models/checkpoints/dance_vqvae.pth"))
    parser.add_argument('--mean_path', type=str, 
                        default=os.path.join(DATA_DIR, "dance/alldata_new_joint_vecs264/meta/Mean.npy"),
                        help="Path to mean vector file")
    parser.add_argument('--std_path', type=str, 
                        default=os.path.join(DATA_DIR, "dance/alldata_new_joint_vecs264/meta/Std.npy"),
                        help="Path to std vector file")
    parser.add_argument('--num_processes', type=int, default=16,
                        help="Number of processes to use for parallel processing (default: 16)")
    parser.add_argument('--chunk_size', type=int, default=1,
                        help="Number of files to process per worker in one batch (default: 1)")
    
    args = parser.parse_args()
    
    npy_dir = args.npy_dir
    out_dir = os.path.join(npy_dir, "npy")
    os.makedirs(out_dir, exist_ok=True)
    
    # Get list of npy files
    npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
    npy_files.sort(key=natural_sort_key)
    npy_files.reverse()
    
    print(f"Found {len(npy_files)} npy files, using {args.num_processes} processes...")
    
    # Select processing mode
    if args.mode in ["process", "both"]:
        if args.num_processes > 1:
            # Multi-process processing
            successful_files = process_files_parallel(
                npy_files, npy_dir, out_dir, 
                args.checkpoint_path, args.mean_path, args.std_path,
                min(args.num_processes, len(npy_files)) ) # Process count does not exceed file count
        else:
            # Single-process processing (backward compatibility)
            successful_files = []
            for file in npy_files:
                if not file.endswith('.npy'):
                    continue
                input_npy_path = os.path.join(npy_dir, file)
                print(f"Processing {input_npy_path}...")
                reconstructed_positions, final_output_path, dance_output = process_npy_file(
                    input_npy_path=input_npy_path, 
                    final_output_dir=out_dir,
                    checkpoint_path=args.checkpoint_path,
                    mean_path=args.mean_path,
                    std_path=args.std_path
                )
                successful_files.append(final_output_path)
    
    # Rendering mode (if needed)
    if args.mode in ["render", "both"]:
        final_video_paths = []
        # Add multi-process rendering logic here if needed
        for file in npy_files:
            npy_path = os.path.join(out_dir, "joints", file)
            if os.path.exists(npy_path):
                video_path, basename = render_npy_file(npy_path)
                final_video_paths.append(video_path)
    
    print("All processing completed!")
