# 这个版本加快了处理速度

from genericpath import exists
import numpy as np
import pickle

from tqdm  import tqdm
from features.kinetic import extract_kinetic_features
from features.manual_new import extract_manual_features
from scipy import linalg
# kinetic, manual
import torch
import os, sys
import argparse
import multiprocessing
import functools
import time



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


def normalize(feat, feat2, feadir):
    os.makedirs(os.path.join(feadir, 'norm'), exist_ok=True)
    if os.path.exists(os.path.join(feadir, 'norm', 'mean.npy')):
        mean = np.load(os.path.join(feadir, 'norm', 'mean.npy'))
    else:
        mean = feat.mean(axis=0) 
        np.save(os.path.join(feadir, 'norm', 'mean.npy'), mean)
    if os.path.exists(os.path.join(feadir, 'norm', 'std.npy')):
        std = np.load(os.path.join(feadir, 'norm', 'std.npy'))
    else:
        std = feat.std(axis=0)
        np.save(os.path.join(feadir, 'norm', 'std.npy'), std)
        
    return (feat - mean) / (std + 1e-10), (feat2 - mean) / (std + 1e-10)

def normalize_simple(feat, feat2):
    mean = feat.mean(axis=0) 
    std = feat.std(axis=0)
       
    return (feat - mean) / (std + 1e-10), (feat2 - mean) / (std + 1e-10)


def normalize_one(feat):
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    
    return (feat - mean) / (std + 1e-10)

def quantized_metrics(predicted_pkl_root, gt_pkl_root):
    pred_features_k = []
    pred_features_m = []
    gt_freatures_k = []
    gt_freatures_m = []


    time1 = time.time()
    pred_features_k = [np.load(os.path.join(predicted_pkl_root, 'kinetic_features', pkl))  for pkl in os.listdir(os.path.join(predicted_pkl_root, 'kinetic_features'))]
    pred_features_m = [np.load(os.path.join(predicted_pkl_root, 'manual_features_new', pkl)) for pkl in os.listdir(os.path.join(predicted_pkl_root, 'manual_features_new'))]
    
    gt_freatures_k = [np.load(os.path.join(gt_pkl_root, 'kinetic_features', pkl)) for pkl in os.listdir(os.path.join(gt_pkl_root, 'kinetic_features'))]
    gt_freatures_m = [np.load(os.path.join(gt_pkl_root, 'manual_features_new', pkl)) for pkl in os.listdir(os.path.join(gt_pkl_root, 'manual_features_new'))]
    time2 = time.time()
    
    pred_features_k = np.stack(pred_features_k)  # Nx72 p40
    pred_features_m = np.stack(pred_features_m) # Nx32
    gt_freatures_k = np.stack(gt_freatures_k) # N' x 72 N' >> N
    gt_freatures_m = np.stack(gt_freatures_m) # 
    if gt_freatures_k.shape[1] == 72:
        gt_freatures_k = gt_freatures_k[:,:66]
    if pred_features_k.shape[1] == 72:
        pred_features_k = pred_features_k[:,:66]

    # gt_freatures_k, pred_features_k = normalize(gt_freatures_k, pred_features_k, os.path.join(gt_pkl_root,'kinetic_features') )
    # gt_freatures_m, pred_features_m = normalize(gt_freatures_m, pred_features_m, os.path.join(gt_pkl_root,'manual_features_new')) 
    
    gt_freatures_k, pred_features_k = normalize_simple(gt_freatures_k, pred_features_k)
    gt_freatures_m, pred_features_m = normalize_simple(gt_freatures_m, pred_features_m)
    print('pred_features_k', pred_features_k.dtype)
    print('gt_freatures_m', gt_freatures_m.dtype)


    fid_k = calc_fid(pred_features_k, gt_freatures_k)
    fid_m = calc_fid(pred_features_m, gt_freatures_m)
    div_k_gt = '***'
    div_m_gt = '***'
    # div_k_gt = calculate_avg_distance(gt_freatures_k)
    # div_m_gt = calculate_avg_distance(gt_freatures_m)
    div_k = calculate_avg_distance(pred_features_k)
    div_m = calculate_avg_distance(pred_features_m)


    metrics = {'fid_k': fid_k, 'fid_m': fid_m, 'div_k': div_k, 'div_m' : div_m, 'div_k_gt': div_k_gt, 'div_m_gt': div_m_gt}
    time3 = time.time()
    
    print(f"耗时1-2: {time2 - time1:.2f} 秒")
    print(f"耗时2-3: {time3 - time2:.2f} 秒")
    
    return metrics


def calc_fid(kps_gen, kps_gt):
    mu_gen = np.mean(kps_gen, axis=0)
    sigma_gen = np.cov(kps_gen, rowvar=False)

    mu_gt = np.mean(kps_gt, axis=0)
    sigma_gt = np.cov(kps_gt, rowvar=False)

    mu1,mu2,sigma1,sigma2 = mu_gen, mu_gt, sigma_gen, sigma_gt

    diff = mu1 - mu2
    eps = 1e-5
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calc_diversity(feats):
    feat_array = np.array(feats)
    n, c = feat_array.shape
    diff = np.array([feat_array] * n) - feat_array.reshape(n, 1, c)
    return np.sqrt(np.sum(diff**2, axis=2)).sum() / n / (n-1)

def calculate_avg_distance(feature_list, mean=None, std=None):
    feature_list = np.stack(feature_list)
    n = feature_list.shape[0]
    # normalize the scale
    if (mean is not None) and (std is not None):
        feature_list = (feature_list - mean) / std
    dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist += np.linalg.norm(feature_list[i] - feature_list[j])
    dist /= (n * n - n) / 2
    return dist

def calc_and_save_feats(pkl, root=None, max_frames=None):
    """
    计算并保存特征
    Args:
        pkl: 文件名
        root: 根目录
        max_frames: 最大帧数，默认1024。如果为None，则使用完整序列
    """
    if not os.path.exists(os.path.join(root, 'kinetic_features')):
        os.makedirs(os.path.join(root, 'kinetic_features'), exist_ok=True)
    if not os.path.exists(os.path.join(root, 'manual_features_new')):
        os.makedirs(os.path.join(root, 'manual_features_new'), exist_ok=True)
    
    if os.path.isdir(os.path.join(root, pkl)):
        return
     
    if pkl[-3:] == 'pkl':
        data = pickle.load(open(os.path.join(root, pkl), "rb"))
        print(data.keys())
        model_q = torch.from_numpy(data['smpl_poses'] ).to(device)   
        model_x = torch.from_numpy(data['smpl_trans'] ).to(device)    
        print("model_q", model_q.shape)
        print("model_x", model_x.shape)
        model_q156 = torch.cat([model_q, torch.zeros([model_q.shape[0], 90]).to(device) ], dim=-1)
        with torch.no_grad():
            joint3d = smplx_model.forward(model_q156, model_x)[:,:24,:]
            print("joint3d", joint3d.shape)
    elif pkl[-3:] == 'npy':
        # 对于FineDance的Mirror数据，以M开头
        # if pkl[0] == 'M':
        #     continue
        data = np.load(os.path.join(root, pkl),allow_pickle=True)
        # print("f{pkl}",data.shape)
        print("pkl",pkl)
        print(data.shape)
        if data.shape[1]==22:
            if data.shape[2]==3:
                joint3d=torch.tensor(data)
        elif data.shape[1] == 72:
            joint3d = torch.tensor(data).reshape(data.shape[0], 24, 3)
        elif data.shape[1] == 66:
            joint3d = torch.tensor(data).reshape(data.shape[0], 22, 3)
            # Create a (2, 3) tensor of zeros
            zeros = torch.zeros(joint3d.shape[0], 2, 3, device=joint3d.device, dtype=joint3d.dtype)

            # Concatenate along the second dimension (dim=1) to get (N, 24, 3)
            joint3d = torch.cat([joint3d, zeros], dim=1)
        elif data.shape[1]==55:
            if data.shape[2]==3:
                joint3d=torch.tensor(data[:,:22])
        elif data.shape[1] == 266:
            if max_frames is not None:
                data = data[:max_frames,:]
            data = np.concatenate([ data[:, 1:2], data[:, 3:] ], axis = 1 ) 
            data = torch.from_numpy(data).to(device).to(torch.float32)
            joint3d = recover_from_ric264(data, 22)[:,:24,:]
        elif data.shape[1] == 264:
            if max_frames is not None:
                data = data[:max_frames,:]
            data = torch.from_numpy(data).to(device).to(torch.float32)
            joint3d = recover_from_ric264(data, 22)[:,:24,:]
        else:
            if data.shape[1] == 139 or data.shape[1] == 319:
                if max_frames is not None:
                    data = data[:max_frames,:139]
                else:
                    data = data[:,:139]
                data = torch.from_numpy(data).to(device)   
            elif data.shape[1] == 135 or data.shape[1] == 315:
                if max_frames is not None:
                    data = data[:max_frames,:135]
                else:
                    data = data[:,:135]
                data = torch.from_numpy(data).to(device)   
                data = torch.cat([torch.zeros([data.shape[0], 4]).to(data),  data], dim=1) 
            # print(data.shape)
            assert data.shape[-1] == 139
            
            with torch.no_grad():
                joint3d = do_smplxfk(data, smplx_model)[:,:24,:]
    else:
        return

    # 根据max_frames参数决定是否截取
    if max_frames is not None:
        joint3d = joint3d[:max_frames,:22,:]  # 只评估前max_frames帧
    else:
        joint3d = joint3d[:,:22,:]  # 评估完整序列
    assert len(joint3d.shape) == 3
    joint3d = joint3d.reshape(joint3d.shape[0], 22*3).detach().cpu().numpy()    #.astype(np.float32)
            
        
    roott = joint3d[:1, :3]  # the root Tx72 (Tx(24x3))
    joint3d = joint3d - np.tile(roott, (1, 22))  # Calculate relative offset with respect to root

    # relative
    joint3d_relative = joint3d.copy()
    joint3d_relative = joint3d_relative.reshape(-1, 22, 3)
    joint3d_relative[:, 1:, :] = joint3d_relative[:, 1:, :] - joint3d_relative[:, 0:1, :]
    np.save(os.path.join(root, 'kinetic_features', pkl), extract_kinetic_features(joint3d_relative.reshape(-1, 22, 3)).astype(np.float32))
    np.save(os.path.join(root, 'manual_features_new', pkl), extract_manual_features(joint3d_relative.reshape(-1, 22, 3)).astype(np.float32))

def calc_save_feats_dir(modir, max_frames=1024):
    seq_names = [n for n in os.listdir(modir)]
    process = functools.partial(calc_and_save_feats, root=modir, max_frames=max_frames)
    with multiprocessing.Pool(40) as pool:
        for _ in tqdm(pool.imap_unordered(process, seq_names), total=len(seq_names)):
            pass

if __name__ == '__main__':


    # 创建 ArgumentParser
    parser = argparse.ArgumentParser(description="设置 GT 和预测路径")

    # 添加参数，提供默认值

    # parser.add_argument("--pred_root", type=str, default="/data2/lrh/dataset/largedance/joints/test", help="预测数据目录")
    # parser.add_argument("--gt_root", type=str, default="/data1/hzy/HumanMotion/InfiniteDance/All_mofea/FineDance/new_joint_vecs264", help="Ground Truth 数据目录")
    # parser.add_argument("--pred_root", type=str, default="/data1/hzy/HumanMotion/All_LargeDanceAR/output/m2d_llama/infer_wav/exp_m2d_wav_250512_1308/dance/sample1/mofea264_2", help="预测数据目录")
    # parser.add_argument("--pred_root", type=str, default="/data1/hzy/HumanMotion/All_LargeDanceAR/output/LODGE/puremusic/npy/test", help="预测数据目录")
    # parser.add_argument("--pred_root", type=str, default="/data1/hzy/HumanMotion/All_LargeDanceAR/output/m2d_llama/infer_wav_RAG/exp_m2d_wav_RAG_250517_1738/dance/mofea264", help="预测数据目录")
    # parser.add_argument("--pred_root", type=str, default="/data1/hzy/HumanMotion/All_LargeDanceAR/output/exp_m2d_token2token/dance/mofea264", help="预测数据目录")
    # parser.add_argument("--pred_root", type=str, default="/data1/hzy/HumanMotion/All_LargeDanceAR/output/m2d_llama/infer_wav/exp_m2d_wav_250520_1529_onlystyle/dance/mofea264", help="预测数据目录")
    # parser.add_argument("--pred_root", type=str, default="/data1/hzy/HumanMotion/InfiniteDance/All_mofea/FineDance/new_joint_vecs264/test", help="预测数据目录")
    
    # parser.add_argument("--pred_root", type=str, default="/data1/hzy/HumanMotion/InfiniteDance/All_LargeDanceAR/output/infer/alldata/exp_m2d_wav_RAG_alldata_mid_10250715_1749/dance/npy/joints", help="预测数据目录")
    
    # parser.add_argument("--pred_root", type=str, default="/data1/hzy/HumanMotion/InfiniteDance/All_LargeDanceAR/output/infer/alldata/dance500_510250716_1658/dance/npy/joints", help="预测数据目录")
    # parser.add_argument("--pred_root", type=str, default="/data1/hzy/HumanMotion/InfiniteDance/All_mofea/ourData_smplx_22_smooth/new_joint_vecs264_vel/test", help="预测数据目录")
    # parser.add_argument("--pred_root", type=str, default="/data1/hzy/HumanMotion/InfiniteDance/All_LargeDanceAR/output/infer/alldata/dance100_110_true250716_2213/dance/npy/joints", help="预测数据目录")
    parser.add_argument("--gt_root", type=str, default="/data2/hzy/InfiniteDance/InfiniteDanceData/dance/ourData_smplx_22_smooth_new/new_joint_vecs264_vel", help="Ground Truth 数据目录")

    parser.add_argument("--pred_root", type=str, default="/data2/hzy/InfiniteDance_opensource/All_LargeDanceAR/infer/infer_unseen/dance_250129_1057/dance/npy/joints/over60s")
    parser.add_argument("--max_frames", type=int, default=0, 
                       help="最大评估帧数，默认1024。设置为0或负数表示评估完整序列")
    

    # 解析参数
    args = parser.parse_args()
    device = f"cuda:4"
    
    gt_root = args.gt_root
    pred_root=args.pred_root
    
    # 处理max_frames参数：0或负数表示完整序列
    max_frames = args.max_frames if args.max_frames > 0 else None
    if max_frames is None:
        print("评估模式: 完整序列（全局）")
    else:
        print(f"评估模式: 前{max_frames}帧")

    # calc_save_feats_dir(gt_root, max_frames=max_frames)
    calc_save_feats_dir(pred_root, max_frames=max_frames)

    print(quantized_metrics(pred_root, gt_root))

