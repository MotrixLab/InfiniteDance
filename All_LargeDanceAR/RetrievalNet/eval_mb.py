import sys

sys.path.append(sys.path[0]+r"/../")
from collections import OrderedDict
from datetime import datetime
from os.path import join as pjoin

import numpy as np
import torch
from configs import get_config
from datasets import (EvaluatorModelWrapper, get_dataset_motion_loader,
                      get_motion_loader)
from models import *
from tqdm import tqdm
from utils.metrics import *
from utils.plot_script import *
from utils.utils import *

os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"

torch.multiprocessing.set_sharing_strategy('file_system')
promotiondir = '/data2/lrh/dataset/HHI/InterHuman/motions_processed/'


def build_models(cfg):
    if cfg.NAME == "InterGen":
        model = InterGen(cfg)
    return model

def evaluate_matching_score(motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    # print(motion_loaders.keys())
    print('========== Evaluating MM Distance ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        print("motion_loader_name",motion_loader_name)
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        mm_dist_sum = 0
        top_k_count = 0
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(motion_loader)):
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(batch)
                # print(text_embeddings.shape)
                # print(motion_embeddings.shape)
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                     motion_embeddings.cpu().numpy())
                # print(dist_mat.shape)
                mm_dist_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                # print(argsmax.shape)

                top_k_mat = calculate_top_k(argsmax, top_k=2)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            mm_dist = mm_dist_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = mm_dist
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] MM Distance: {mm_dist:.4f}')
        print(f'---> [{motion_loader_name}] MM Distance: {mm_dist:.4f}')

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line)

    return match_score_dict, R_precision_dict, activation_dict


def evaluate_fid(groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(groundtruth_loader)):
            motion_embeddings = eval_wrapper.get_motion_embeddings(batch)
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    # print(gt_mu)
    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        # print(mu)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}')
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
    return eval_dict


def evaluate_multimodality(mm_motion_loaders, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                batch[2] = batch[2][0]
                batch[3] = batch[3][0]
                batch[4] = batch[4][0]
                motion_embedings = eval_wrapper.get_motion_embeddings(batch)
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        eval_dict[model_name] = multimodality
    return eval_dict


def get_metric_statistics(values):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(log_file):
    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({'MM Distance': OrderedDict({}),
                                   'R_precision': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'Diversity': OrderedDict({}),
                                   'MultiModality': OrderedDict({})})
        for replication in range(replication_times):
            motion_loaders = {}
            mm_motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                if motion_loader_name != 'ground truth':
                    continue
                motion_loader, mm_motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader

            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(motion_loaders, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(gt_loader, acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mm_score_dict = evaluate_multimodality(mm_motion_loaders, f)

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

            for key, item in mat_score_dict.items():
                if key not in all_metrics['MM Distance']:
                    all_metrics['MM Distance'][key] = [item]
                else:
                    all_metrics['MM Distance'][key] += [item]

            for key, item in R_precision_dict.items():
                if key not in all_metrics['R_precision']:
                    all_metrics['R_precision'][key] = [item]
                else:
                    all_metrics['R_precision'][key] += [item]

            for key, item in fid_score_dict.items():
                if key not in all_metrics['FID']:
                    all_metrics['FID'][key] = [item]
                else:
                    all_metrics['FID'][key] += [item]

            for key, item in div_score_dict.items():
                if key not in all_metrics['Diversity']:
                    all_metrics['Diversity'][key] = [item]
                else:
                    all_metrics['Diversity'][key] += [item]

            for key, item in mm_score_dict.items():
                if key not in all_metrics['MultiModality']:
                    all_metrics['MultiModality'][key] = [item]
                else:
                    all_metrics['MultiModality'][key] += [item]


        # print(all_metrics['Diversity'])
        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)

            for model_name, values in metric_dict.items():
                # print(metric_name, model_name)
                mean, conf_interval = get_metric_statistics(np.array(values))
                # print(mean, mean.dtype)
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)


if __name__ == '__main__':
    mm_num_samples = 100
    mm_num_repeats = 30
    mm_num_times = 10

    diversity_times = 100
    replication_times = 1      # 20

    # batch_size is fixed to 96!!
    batch_size = 96
    
    eval_motion_loaders = {}
    data_cfg = get_config("/data1/hzy/HumanMotion/RetrievalNet/configs/largedance/musicbody/datasets93.yaml").largedance    #_test
    device = torch.device('cuda:%d' % 0 if torch.cuda.is_available() else 'cpu')
    gt_loader, gt_dataset = get_dataset_motion_loader(data_cfg, batch_size)
    # evalmodel_cfg = get_config("configs/eval_model.yaml")
    # evalmodel_cfg = get_config("configs/interclip/InterCLIP_RP516.yaml")
    evalmodel_cfg = get_config("/data1/hzy/HumanMotion/RetrievalNet/configs/largedance/musicbody/InterCLIP.yaml")
    
    eval_wrapper = EvaluatorModelWrapper(evalmodel_cfg, device)  # InterClip

    log_file = f'/data1/hzy/HumanMotion/RetrievalNet/evaluation_{1}.log'
    evaluation(log_file)



# # import sys
# # sys.path.append(sys.path[0]+r"/../")
# # import numpy as np
# # import torch
# # import os
# # from datetime import datetime
# # from datasets import get_dataset_motion_loader, get_motion_loader
# # from models import *
# # from utils.metrics import *
# # from datasets import EvaluatorModelWrapper
# # from collections import OrderedDict
# # from utils.plot_script import *
# # from utils.utils import *
# # from configs import get_config
# # from os.path import join as pjoin
# # from tqdm import tqdm

# # os.environ['WORLD_SIZE'] = '1'
# # os.environ['RANK'] = '0'
# # os.environ['MASTER_ADDR'] = 'localhost'
# # os.environ['MASTER_PORT'] = '12345'
# # os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"

# # torch.multiprocessing.set_sharing_strategy('file_system')
# # promotiondir = '/data2/lrh/dataset/HHI/InterHuman/motions_processed/'


# # def build_models(cfg):
# #     if cfg.NAME == "InterGen":
# #         model = InterGen(cfg)
# #     return model


# # def evaluate_matching_score(motion_loaders, file):
# #     match_score_dict = OrderedDict({})
# #     R_precision_dict = OrderedDict({})
# #     activation_dict = OrderedDict({})
# #     print('========== Evaluating MM Distance ==========')
# #     for motion_loader_name, motion_loader in motion_loaders.items():
# #         print("motion_loader_name", motion_loader_name)
# #         all_motion_embeddings = []
# #         all_text_embeddings = []
# #         score_list = []
# #         all_size = 0
# #         mm_dist_sum = 0
# #         top_k_count = 0
# #         with torch.no_grad():
# #             for idx, batch in tqdm(enumerate(motion_loader)):
# #                 text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(batch)
# #                 dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
# #                                                     motion_embeddings.cpu().numpy())
# #                 mm_dist_sum += dist_mat.trace()
# #                 argsmax = np.argsort(dist_mat, axis=1)
# #                 top_k_mat = calculate_top_k(argsmax, top_k=2)
# #                 top_k_count += top_k_mat.sum(axis=0)
# #                 all_size += text_embeddings.shape[0]
# #                 all_motion_embeddings.append(motion_embeddings.cpu().numpy())
# #                 all_text_embeddings.append(text_embeddings.cpu().numpy())

# #             all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
# #             all_text_embeddings = np.concatenate(all_text_embeddings, axis=0)
# #             mm_dist = mm_dist_sum / all_size
# #             R_precision = top_k_count / all_size
# #             match_score_dict[motion_loader_name] = mm_dist
# #             R_precision_dict[motion_loader_name] = R_precision
# #             activation_dict[motion_loader_name] = all_motion_embeddings
# #             activation_dict[motion_loader_name + '_text'] = all_text_embeddings

# #         print(f'---> [{motion_loader_name}] MM Distance: {mm_dist:.4f}')
# #         print(f'---> [{motion_loader_name}] MM Distance: {mm_dist:.4f}')

# #         line = f'---> [{motion_loader_name}] R_precision: '
# #         for i in range(len(R_precision)):
# #             line += '(top %d): %.4f ' % (i+1, R_precision[i])
# #         print(line)
# #         print(line)

# #     return match_score_dict, R_precision_dict, activation_dict


# # def evaluate_fid(groundtruth_loader, activation_dict, file):
# #     eval_dict = OrderedDict({})
# #     gt_motion_embeddings = []
# #     print('========== Evaluating FID ==========')
# #     with torch.no_grad():
# #         for idx, batch in tqdm(enumerate(groundtruth_loader)):
# #             motion_embeddings = eval_wrapper.get_motion_embeddings(batch)
# #             gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
# #     gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
# #     gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

# #     for model_name, motion_embeddings in activation_dict.items():
# #         if '_text' in model_name:
# #             continue
# #         mu, cov = calculate_activation_statistics(motion_embeddings)
# #         fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
# #         print(f'---> [{model_name}] FID: {fid:.4f}')
# #         print(f'---> [{model_name}] FID: {fid:.4f}')
# #         eval_dict[model_name] = fid
# #     return eval_dict


# # def evaluate_diversity(activation_dict, file):
# #     eval_dict = OrderedDict({})
# #     print('========== Evaluating Diversity ==========')
# #     for model_name, motion_embeddings in activation_dict.items():
# #         if '_text' in model_name:
# #             continue
# #         diversity = calculate_diversity(motion_embeddings, diversity_times)
# #         eval_dict[model_name] = diversity
# #         print(f'---> [{model_name}] Diversity: {diversity:.4f}')
# #         print(f'---> [{model_name}] Diversity: {diversity:.4f}')
# #     return eval_dict


# # def evaluate_multimodality(mm_motion_loaders, file):
# #     eval_dict = OrderedDict({})
# #     print('========== Evaluating MultiModality ==========')
# #     for model_name, mm_motion_loader in mm_motion_loaders.items():
# #         mm_motion_embeddings = []
# #         with torch.no_grad():
# #             for idx, batch in enumerate(mm_motion_loader):
# #                 batch[2] = batch[2][0]
# #                 batch[3] = batch[3][0]
# #                 batch[4] = batch[4][0]
# #                 motion_embeddings = eval_wrapper.get_motion_embeddings(batch)
# #                 mm_motion_embeddings.append(motion_embeddings.unsqueeze(0))
# #         if len(mm_motion_embeddings) == 0:
# #             multimodality = 0
# #         else:
# #             mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
# #             multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
# #         print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
# #         print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
# #         eval_dict[model_name] = multimodality
# #     return eval_dict


# # def save_top10_similar_motions(activation_dict, input_audio_file, device, output_dir='/data1/hzy/top10_motions/'):
# #     """
# #     根据输入的音频文件，找出相似度最高的前10个动作并保存
    
# #     Args:
# #         activation_dict: 包含动作嵌入的字典
# #         input_audio_file: 输入音频特征文件路径
# #         device: 计算设备
# #         output_dir: 输出目录
# #     """
# #     print('========== 保存 Top 10 相似动作 ==========')
# #     os.makedirs(output_dir, exist_ok=True)

# #     # 加载输入音频特征
# #     audio_data = np.load(input_audio_file).astype(np.float32)
    
# #     # 打印音频特征信息
    
    
# #     # 确保音频特征维度正确 (取前384维)
# #     if audio_data.shape[0] >= 384:
# #         audio_data = audio_data[:384]  # 取前 384 维
# #     else:
# #         # 如果维度不足，则填充零
# #         padding = np.zeros(384 - audio_data.shape[0], dtype=np.float32)
# #         audio_data = np.concatenate([audio_data, padding])
# #     print(f"音频特征形状: {audio_data.shape}")
# #     # audio_data = audio_data.reshape(1, -1)  # 调整形状为 (1, 384)
# #     # audio_data=audio_data.permute(1,0)
# #     audio_data = torch.from_numpy(audio_data).float()  # 转为 float Tensor（可选根据你的模型需要）
# #     audio_data = audio_data.unsqueeze(0)  # 添加 batch 维度
# #     audio_data=audio_data.cpu().numpy()
# #     print(f"音频特征形状: {audio_data.shape}")
# #     # 模拟数据集 batch 格式
# #     # breakpoint()
# #     seqname = os.path.basename(input_audio_file).replace('.npy', '')
# #     dummy_mofea = np.zeros((1,384, 264), dtype=np.float32)  # 占位动作特征
    
# #     # 创建简化的batch结构 (仅包含3个元素)
# #     simple_batch = (
# #         [seqname],  # 序列名 - 列表形式与原始数据集一致
# #         torch.from_numpy(audio_data).to(device),  # 音频特征
# #         torch.from_numpy(dummy_mofea).to(device)   # 占位动作特征
# #     )
    
# #     # 获取文本嵌入 (音频嵌入)
# #     with torch.no_grad():
# #         try:
# #             text_embedding, _ = eval_wrapper.get_co_embeddings(simple_batch)
# #             print(f"获取文本嵌入成功，形状: {text_embedding.shape}")
# #             # 确保转换为numpy数组
# #             if torch.is_tensor(text_embedding):
# #                 text_embedding = text_embedding.cpu().numpy()
# #         except Exception as e:
# #             # print(f"获取文本嵌入失败: {e}")
# #             # print("尝试检查evaluator.py中get_co_embeddings函数的参数...")
# #             # # 如果无法通过简化batch获取嵌入，尝试从已有的嵌入字典中获取参考嵌入
# #             # for model_name, embeddings in activation_dict.items():
# #             #     if '_text' in model_name:
# #             #         text_embedding = embeddings[0:1]  # 使用第一个文本嵌入作为参考
# #             #         if torch.is_tensor(text_embedding):
# #             #             text_embedding = text_embedding.cpu().numpy()
# #             #         print(f"使用已有嵌入 '{model_name}' 作为参考，形状: {text_embedding.shape}")
# #             #         break
# #             # else:
# #                 # print("无法获取有效的文本嵌入，退出")
# #                 # return
# #             print("无法获取有效的文本嵌入，退出")
# #             return 0
    
# #     # 确保text_embedding是numpy数组
# #     if not isinstance(text_embedding, np.ndarray):
# #         print(f"警告: text_embedding不是numpy数组，类型为: {type(text_embedding)}")
# #         text_embedding = np.array(text_embedding)
    
# #     # 保存输入文本嵌入，便于后续使用
# #     input_embedding_file = os.path.join(output_dir, f'input_{seqname}_embedding.npy')
# #     np.save(input_embedding_file, text_embedding)
# #     print(f"已保存输入嵌入: {input_embedding_file}")

# #     # 处理每个模型的动作嵌入
# #     for model_name, motion_embeddings in activation_dict.items():
# #         if '_text' in model_name:
# #             continue  # 跳过文本嵌入
        
# #         print(f"处理模型 '{model_name}' 的动作嵌入，形状: {motion_embeddings.shape}")

# #         # 计算输入文本嵌入与所有动作嵌入的欧几里得距离
# #         dist_mat = euclidean_distance_matrix(text_embedding, motion_embeddings)
# #         dist_mat = dist_mat.flatten()
        
# #         # 取前10个最小距离的索引
# #         top_count = min(10, len(dist_mat))
# #         top_indices = np.argsort(dist_mat)[:top_count]
# #         top_distances = dist_mat[top_indices]
        
# #         print(f"找到 {top_count} 个最相似的动作，距离范围: {top_distances[0]:.4f} - {top_distances[-1]:.4f}")

# #         # 保存 top N 动作嵌入
# #         for rank, idx in enumerate(top_indices):
# #             motion_embedding = motion_embeddings[idx]
# #             distance = top_distances[rank]
            
# #             # 创建更有序的文件名
# #             output_filename = os.path.join(
# #                 output_dir, 
# #                 f'{seqname}_top{rank+1:02d}_idx{idx:05d}_dist{distance:.4f}.npy'
# #             )
            
# #             # np.save(output_filename, motion_embedding)
# #             # print(f'已保存第 {rank+1} 名动作: {output_filename}')
        
# #         # 保存所有top N动作到一个npy文件
# #         all_top_file = os.path.join(output_dir, f'{seqname}_all_top{top_count}.npy')
# #         top_motion_embeddings = np.array([motion_embeddings[idx] for idx in top_indices])
# #         np.save(all_top_file, top_motion_embeddings)
# #         print(f'已保存所有前 {top_count} 名动作: {all_top_file}')
        
# #         # 保存结果索引和距离
# #         result_file = os.path.join(output_dir, f'{seqname}_results.npy')
# #         result_data = {
# #             'top_indices': top_indices,
# #             'top_distances': top_distances,
# #             'input_embedding': text_embedding
# #         }
# #         np.save(result_file, result_data)
# #         print(f'已保存结果数据: {result_file}')


# # def save_original_motion_data(gt_loader, indices, save_dir):
# #     """
# #     保存指定索引的原始动作数据
    
# #     Args:
# #         gt_loader: 数据加载器
# #         indices: 要保存的动作索引列表
# #         save_dir: 保存目录
# #     """
# #     print('========== 保存原始动作数据 ==========')
# #     os.makedirs(save_dir, exist_ok=True)
    
# #     # 将索引按照批次组织
# #     batch_size = 96  # 固定批次大小
# #     batch_indices = {}
# #     for idx in indices:
# #         batch_idx = idx // batch_size
# #         in_batch_idx = idx % batch_size
# #         if batch_idx not in batch_indices:
# #             batch_indices[batch_idx] = []
# #         batch_indices[batch_idx].append((idx, in_batch_idx))
    
# #     # 遍历数据加载器，找到并保存指定索引的动作
# #     with torch.no_grad():
# #         for batch_idx, batch in tqdm(enumerate(gt_loader), desc="保存原始动作数据"):
# #             if batch_idx in batch_indices:
# #                 # 获取当前批次中需要保存的索引
# #                 for global_idx, in_batch_idx in batch_indices[batch_idx]:
# #                     try:
# #                         # 假设batch[2]是动作数据
# #                         if len(batch) > 2 and torch.is_tensor(batch[2]):
# #                             motion_data = batch[2][in_batch_idx].cpu().numpy()
# #                             # 获取序列名
# #                             seqname = batch[0][in_batch_idx] if isinstance(batch[0], list) and len(batch[0]) > in_batch_idx else f"unknown_{global_idx}"
# #                             # 保存动作数据
# #                             save_file = os.path.join(save_dir, f'original_motion_{global_idx}.npy')
# #                             np.save(save_file, motion_data)
# #                             print(f'已保存原始动作数据: {save_file}')
# #                         else:
# #                             print(f"警告: 批次 {batch_idx} 中没有动作数据")
# #                     except Exception as e:
# #                         print(f"保存索引 {global_idx} 的动作数据时出错: {e}")


# # def get_metric_statistics(values):
# #     mean = np.mean(values, axis=0)
# #     std = np.std(values, axis=0)
# #     conf_interval = 1.96 * std / np.sqrt(replication_times)
# #     return mean, conf_interval


# # def evaluation(log_file, input_audio_file=None):
# #     with open(log_file, 'w') as f:
# #         all_metrics = OrderedDict({'MM Distance': OrderedDict({}),
# #                                    'R_precision': OrderedDict({}),
# #                                    'FID': OrderedDict({}),
# #                                    'Diversity': OrderedDict({}),
# #                                    'MultiModality': OrderedDict({})})
# #         for replication in range(replication_times):
# #             motion_loaders = {}
# #             mm_motion_loaders = {}
# #             motion_loaders['ground truth'] = gt_loader
# #             for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
# #                 if motion_loader_name != 'ground truth':
# #                     continue
# #                 motion_loader, mm_motion_loader = motion_loader_getter()
# #                 motion_loaders[motion_loader_name] = motion_loader
# #                 mm_motion_loaders[motion_loader_name] = mm_motion_loader

# #             print(f'==================== Replication {replication} ====================')
# #             print(f'==================== Replication {replication} ====================', file=f, flush=True)
# #             print(f'Time: {datetime.now()}')
# #             print(f'Time: {datetime.now()}', file=f, flush=True)
# #             mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(motion_loaders, f)

# #             # 如果提供了输入音频文件，保存 top 10 相似动作
# #             if input_audio_file is not None:
# #                 save_top10_similar_motions(acti_dict, input_audio_file, device)

# #             print(f'Time: {datetime.now()}')
# #             print(f'Time: {datetime.now()}', file=f, flush=True)
# #             fid_score_dict = evaluate_fid(gt_loader, acti_dict, f)

# #             print(f'Time: {datetime.now()}')
# #             print(f'Time: {datetime.now()}', file=f, flush=True)
# #             div_score_dict = evaluate_diversity(acti_dict, f)

# #             print(f'Time: {datetime.now()}')
# #             print(f'Time: {datetime.now()}', file=f, flush=True)
# #             mm_score_dict = evaluate_multimodality(mm_motion_loaders, f)

# #             print(f'!!! DONE !!!')
# #             print(f'!!! DONE !!!', file=f, flush=True)

# #             for key, item in mat_score_dict.items():
# #                 if key not in all_metrics['MM Distance']:
# #                     all_metrics['MM Distance'][key] = [item]
# #                 else:
# #                     all_metrics['MM Distance'][key] += [item]

# #             for key, item in R_precision_dict.items():
# #                 if key not in all_metrics['R_precision']:
# #                     all_metrics['R_precision'][key] = [item]
# #                 else:
# #                     all_metrics['R_precision'][key] += [item]

# #             for key, item in fid_score_dict.items():
# #                 if key not in all_metrics['FID']:
# #                     all_metrics['FID'][key] = [item]
# #                 else:
# #                     all_metrics['FID'][key] += [item]

# #             for key, item in div_score_dict.items():
# #                 if key not in all_metrics['Diversity']:
# #                     all_metrics['Diversity'][key] = [item]
# #                 else:
# #                     all_metrics['Diversity'][key] += [item]

# #             for key, item in mm_score_dict.items():
# #                 if key not in all_metrics['MultiModality']:
# #                     all_metrics['MultiModality'][key] = [item]
# #                 else:
# #                     all_metrics['MultiModality'][key] += [item]

# #         for metric_name, metric_dict in all_metrics.items():
# #             print('========== %s Summary ==========' % metric_name)
# #             print('========== %s Summary ==========' % metric_name, file=f, flush=True)

# #             for model_name, values in metric_dict.items():
# #                 mean, conf_interval = get_metric_statistics(np.array(values))
# #                 if isinstance(mean, np.float64) or isinstance(mean, np.float32):
# #                     print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
# #                     print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
# #                 elif isinstance(mean, np.ndarray):
# #                     line = f'---> [{model_name}]'
# #                     for i in range(len(mean)):
# #                         line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
# #                     print(line)
# #                     print(line, file=f, flush=True)
        
# #         return all_metrics


# # def extract_top_motions(input_audio_file, output_dir='/data1/hzy/top10_motions/'):
# #     """
# #     提取指定音频文件对应的前10个相似动作
    
# #     Args:
# #         input_audio_file: 输入音频特征文件路径
# #         output_dir: 输出目录
# #     """
# #     # 加载数据
# #     print("加载数据和模型...")
# #     data_cfg = get_config("/data1/hzy/HumanMotion/RetrievalNet/configs/largedance/musicbody/datasets93.yaml").largedance_test
# #     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# #     gt_loader, gt_dataset = get_dataset_motion_loader(data_cfg, 96)
# #     evalmodel_cfg = get_config("/data1/hzy/HumanMotion/RetrievalNet/configs/largedance/musicbody/InterCLIP.yaml")
# #     eval_wrapper = EvaluatorModelWrapper(evalmodel_cfg, device)
    
# #     # 收集所有嵌入向量
# #     print("收集所有嵌入向量...")
# #     motion_loaders = {'ground truth': gt_loader}
# #     mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(motion_loaders, sys.stdout)
    
# #     # 提取相似动作
# #     print("提取相似动作...")
# #     save_top10_similar_motions(acti_dict, input_audio_file, device, output_dir)
    
# #     # 尝试保存原始动作数据
# #     for model_name, embeddings in acti_dict.items():
# #         if '_text' in model_name:
# #             continue
            
# #         # 加载结果数据获取索引
# #         result_file = os.path.join(output_dir, f'{os.path.basename(input_audio_file).replace(".npy", "")}_results.npy')
# #         if os.path.exists(result_file):
# #             try:
# #                 result_data = np.load(result_file, allow_pickle=True).item()
# #                 top_indices = result_data['top_indices']
# #                 save_dir = os.path.join(output_dir, f'original_motions_{os.path.basename(input_audio_file).replace(".npy", "")}')
# #                 save_original_motion_data(gt_loader, top_indices, save_dir)
# #                 break
# #             except Exception as e:
# #                 print(f"保存原始动作数据时出错: {e}")
    
# #     return acti_dict


# # if __name__ == '__main__':
# #     mm_num_samples = 100
# #     mm_num_repeats = 30
# #     mm_num_times = 10
# #     diversity_times = 300
# #     replication_times = 1
# #     batch_size = 96
    
# #     eval_motion_loaders = {}
# #     data_cfg = get_config("/data1/hzy/HumanMotion/RetrievalNet/configs/largedance/musicbody/datasets93.yaml").largedance_test
# #     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# #     gt_loader, gt_dataset = get_dataset_motion_loader(data_cfg, batch_size)
# #     evalmodel_cfg = get_config("/data1/hzy/HumanMotion/RetrievalNet/configs/largedance/musicbody/InterCLIP.yaml")
    
# #     eval_wrapper = EvaluatorModelWrapper(evalmodel_cfg, device)
    
# #     # 指定输入文件和输出目录
# #     input_audio_file = '/data1/hzy/AllDataset/allmusic_librosa55/083.npy'
# #     output_dir = '/data1/hzy/HumanMotion/RetrievalNet/top_motions_083'
    
# #     # 创建输出目录
# #     os.makedirs(output_dir, exist_ok=True)
    
# #     # 提取相似动作
# #     print(f"使用输入文件: {input_audio_file}")
# #     print(f"输出目录: {output_dir}")
    
# #     # 调用函数提取相似动作
# #     extract_top_motions(input_audio_file, output_dir)
    
# #     print("完成！")


# import sys
# sys.path.append(sys.path[0]+r"/../")
# import numpy as np
# import torch
# import os
# from datetime import datetime
# from datasets import get_dataset_motion_loader, get_motion_loader
# from models import *
# from utils.metrics import *
# from datasets import EvaluatorModelWrapper
# from collections import OrderedDict
# from utils.plot_script import *
# from utils.utils import *
# from configs import get_config
# from os.path import join as pjoin
# from tqdm import tqdm

# os.environ['WORLD_SIZE'] = '1'
# os.environ['RANK'] = '0'
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12345'
# os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"

# torch.multiprocessing.set_sharing_strategy('file_system')
# promotiondir = '/data2/lrh/dataset/HHI/InterHuman/motions_processed/'


# def build_models(cfg):
#     if cfg.NAME == "InterGen":
#         model = InterGen(cfg)
#     return model


# def evaluate_matching_score(motion_loaders, file):
#     match_score_dict = OrderedDict({})
#     R_precision_dict = OrderedDict({})
#     activation_dict = OrderedDict({})
#     print('========== Evaluating MM Distance ==========')
#     for motion_loader_name, motion_loader in motion_loaders.items():
#         print("motion_loader_name", motion_loader_name)
#         all_motion_embeddings = []
#         all_text_embeddings = []
#         score_list = []
#         all_size = 0
#         mm_dist_sum = 0
#         top_k_count = 0
#         with torch.no_grad():
#             for idx, batch in tqdm(enumerate(motion_loader)):
#                 text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(batch)
#                 dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
#                                                     motion_embeddings.cpu().numpy())
#                 mm_dist_sum += dist_mat.trace()
#                 argsmax = np.argsort(dist_mat, axis=1)
#                 top_k_mat = calculate_top_k(argsmax, top_k=2)
#                 top_k_count += top_k_mat.sum(axis=0)
#                 all_size += text_embeddings.shape[0]
#                 all_motion_embeddings.append(motion_embeddings.cpu().numpy())
#                 all_text_embeddings.append(text_embeddings.cpu().numpy())

#             all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
#             all_text_embeddings = np.concatenate(all_text_embeddings, axis=0)
#             mm_dist = mm_dist_sum / all_size
#             R_precision = top_k_count / all_size
#             match_score_dict[motion_loader_name] = mm_dist
#             R_precision_dict[motion_loader_name] = R_precision
#             activation_dict[motion_loader_name] = all_motion_embeddings
#             activation_dict[motion_loader_name + '_text'] = all_text_embeddings

#         print(f'---> [{motion_loader_name}] MM Distance: {mm_dist:.4f}')
#         print(f'---> [{motion_loader_name}] MM Distance: {mm_dist:.4f}')

#         line = f'---> [{motion_loader_name}] R_precision: '
#         for i in range(len(R_precision)):
#             line += '(top %d): %.4f ' % (i+1, R_precision[i])
#         print(line)
#         print(line)

#     return match_score_dict, R_precision_dict, activation_dict


# def evaluate_fid(groundtruth_loader, activation_dict, file):
#     eval_dict = OrderedDict({})
#     gt_motion_embeddings = []
#     print('========== Evaluating FID ==========')
#     with torch.no_grad():
#         for idx, batch in tqdm(enumerate(groundtruth_loader)):
#             motion_embeddings = eval_wrapper.get_motion_embeddings(batch)
#             gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
#     gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
#     gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

#     for model_name, motion_embeddings in activation_dict.items():
#         if '_text' in model_name:
#             continue
#         mu, cov = calculate_activation_statistics(motion_embeddings)
#         fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
#         print(f'---> [{model_name}] FID: {fid:.4f}')
#         print(f'---> [{model_name}] FID: {fid:.4f}')
#         eval_dict[model_name] = fid
#     return eval_dict


# def evaluate_diversity(activation_dict, file):
#     eval_dict = OrderedDict({})
#     print('========== Evaluating Diversity ==========')
#     for model_name, motion_embeddings in activation_dict.items():
#         if '_text' in model_name:
#             continue
#         diversity = calculate_diversity(motion_embeddings, diversity_times)
#         eval_dict[model_name] = diversity
#         print(f'---> [{model_name}] Diversity: {diversity:.4f}')
#         print(f'---> [{model_name}] Diversity: {diversity:.4f}')
#     return eval_dict


# def evaluate_multimodality(mm_motion_loaders, file):
#     eval_dict = OrderedDict({})
#     print('========== Evaluating MultiModality ==========')
#     for model_name, mm_motion_loader in mm_motion_loaders.items():
#         mm_motion_embeddings = []
#         with torch.no_grad():
#             for idx, batch in enumerate(mm_motion_loader):
#                 batch[2] = batch[2][0]
#                 batch[3] = batch[3][0]
#                 batch[4] = batch[4][0]
#                 motion_embeddings = eval_wrapper.get_motion_embeddings(batch)
#                 mm_motion_embeddings.append(motion_embeddings.unsqueeze(0))
#         if len(mm_motion_embeddings) == 0:
#             multimodality = 0
#         else:
#             mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
#             multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
#         print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
#         print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
#         eval_dict[model_name] = multimodality
#     return eval_dict


# def save_top10_similar_motions(activation_dict, input_audio_file, device, output_dir='/data1/hzy/top10_motions/'):
#     """
#     根据输入的音频文件，找出相似度最高的前10个动作并保存
    
#     Args:
#         activation_dict: 包含动作嵌入的字典
#         input_audio_file: 输入音频特征文件路径
#         device: 计算设备
#         output_dir: 输出目录
#     """
#     print('========== 保存 Top 10 相似动作 ==========')
#     os.makedirs(output_dir, exist_ok=True)

#     # 加载输入音频特征
#     audio_data = np.load(input_audio_file).astype(np.float32)
    
#     # 确保音频特征维度正确 (取前384维)
#     if audio_data.shape[0] >= 384:
#         audio_data = audio_data[:384]  # 取前 384 维
#     else:
#         # 如果维度不足，则填充零
#         padding = np.zeros(384 - audio_data.shape[0], dtype=np.float32)
#         audio_data = np.concatenate([audio_data, padding])
#     print(f"音频特征形状: {audio_data.shape}")
#     audio_data = torch.from_numpy(audio_data).float()  # 转为 float Tensor
#     audio_data = audio_data.unsqueeze(0)  # 添加 batch 维度
#     audio_data = audio_data.cpu().numpy()
#     print(f"音频特征形状: {audio_data.shape}")
    
#     # 模拟数据集 batch 格式
#     seqname = os.path.basename(input_audio_file).replace('.npy', '')
#     dummy_mofea = np.zeros((1, 384, 264), dtype=np.float32)  # 占位动作特征
    
#     # 创建简化的batch结构
#     simple_batch = (
#         [seqname],  # 序列名
#         torch.from_numpy(audio_data).to(device),  # 音频特征
#         torch.from_numpy(dummy_mofea).to(device)  # 占位动作特征
#     )
    
#     # 获取文本嵌入 (音频嵌入)
#     with torch.no_grad():
#         try:
#             text_embedding, _ = eval_wrapper.get_co_embeddings(simple_batch)
#             print(f"获取文本嵌入成功，形状: {text_embedding.shape}")
#             if torch.is_tensor(text_embedding):
#                 text_embedding = text_embedding.cpu().numpy()
#         except Exception as e:
#             print(f"获取文本嵌入失败: {str(e)}")
#             print(f"批次结构: {[type(x) for x in simple_batch]}")
#             return 0
    
#     # 确保text_embedding是numpy数组
#     if not isinstance(text_embedding, np.ndarray):
#         print(f"警告: text_embedding不是numpy数组，类型为: {type(text_embedding)}")
#         text_embedding = np.array(text_embedding)
    
#     # 保存输入文本嵌入
#     input_embedding_file = os.path.join(output_dir, f'input_{seqname}_embedding.npy')
#     np.save(input_embedding_file, text_embedding)
#     print(f"已保存输入嵌入: {input_embedding_file}")

#     # 处理每个模型的动作嵌入
#     for model_name, motion_embeddings in activation_dict.items():
#         if '_text' in model_name:
#             continue  # 跳过文本嵌入
        
#         print(f"处理模型 '{model_name}' 的动作嵌入，形状: {motion_embeddings.shape}")

#         # 计算输入文本嵌入与所有动作嵌入的欧几里得距离
#         dist_mat = euclidean_distance_matrix(text_embedding, motion_embeddings)
#         dist_mat = dist_mat.flatten()
        
#         # 取前10个最小距离的索引
#         top_count = min(10, len(dist_mat))
#         top_indices = np.argsort(dist_mat)[:top_count]
#         top_distances = dist_mat[top_indices]
        
#         print(f"找到 {top_count} 个最相似的动作，距离范围: {top_distances[0]:.4f} - {top_distances[-1]:.4f}")

#         # 保存 top N 动作嵌入
#         for rank, idx in enumerate(top_indices):
#             motion_embedding = motion_embeddings[idx]
#             distance = top_distances[rank]
#             output_filename = os.path.join(
#                 output_dir, 
#                 f'{seqname}_top{rank+1:02d}_idx{idx:05d}_dist{distance:.4f}.npy'
#             )
#             np.save(output_filename, motion_embedding)
#             print(f'已保存第 {rank+1} 名动作: {output_filename}')
        
#         # 保存所有top N动作到一个npy文件
#         all_top_file = os.path.join(output_dir, f'{seqname}_all_top{top_count}.npy')
#         top_motion_embeddings = np.array([motion_embeddings[idx] for idx in top_indices])
#         np.save(all_top_file, top_motion_embeddings)
#         print(f'已保存所有前 {top_count} 名动作: {all_top_file}')
        
#         # 保存结果索引和距离
#         result_file = os.path.join(output_dir, f'{seqname}_results.npy')
#         result_data = {
#             'top_indices': top_indices,
#             'top_distances': top_distances,
#             'input_embedding': text_embedding
#         }
#         np.save(result_file, result_data)
#         print(f'已保存结果数据: {result_file}')


# def save_original_motion_data(gt_loader, save_dir):
#     """
#     保存数据加载器中的所有原始动作数据
    
#     Args:
#         gt_loader: 数据加载器
#         save_dir: 保存目录
#     """
#     print('========== 保存所有原始动作数据 ==========')
#     os.makedirs(save_dir, exist_ok=True)
    
#     global_idx = 0  # 全局索引，用于文件名
#     batch_size = 96  # 固定批次大小
    
#     with torch.no_grad():
#         for batch_idx, batch in tqdm(enumerate(gt_loader), desc="保存原始动作数据", total=len(gt_loader)):
#             try:
#                 # 检查批次是否包含动作数据
#                 if len(batch) > 2 and torch.is_tensor(batch[2]):
#                     motion_data = batch[2].cpu().numpy()  # 获取整个批次的动作数据
#                     seqnames = batch[0] if isinstance(batch[0], list) else [f"batch_{batch_idx}" for _ in range(motion_data.shape[0])]
                    
#                     # 遍历批次中的每个样本
#                     for i in range(motion_data.shape[0]):
#                         sample_motion = motion_data[i]
#                         seqname = seqnames[i] if i < len(seqnames) else f"unknown_{global_idx}"
#                         # 使用序列名和全局索引生成文件名
#                         save_file = os.path.join(save_dir, f'original_motion_{seqname}_{global_idx}.npy')
#                         np.save(save_file, sample_motion)
#                         print(f'已保存原始动作数据: {save_file}')
#                         global_idx += 1
#                 else:
#                     print(f"警告: 批次 {batch_idx} 中没有动作数据")
#             except Exception as e:
#                 print(f"保存批次 {batch_idx} 的动作数据时出错: {str(e)}")


# def get_metric_statistics(values):
#     mean = np.mean(values, axis=0)
#     std = np.std(values, axis=0)
#     conf_interval = 1.96 * std / np.sqrt(replication_times)
#     return mean, conf_interval


# def evaluation(log_file, input_audio_file=None):
#     with open(log_file, 'w') as f:
#         all_metrics = OrderedDict({'MM Distance': OrderedDict({}),
#                                    'R_precision': OrderedDict({}),
#                                    'FID': OrderedDict({}),
#                                    'Diversity': OrderedDict({}),
#                                    'MultiModality': OrderedDict({})})
#         for replication in range(replication_times):
#             motion_loaders = {}
#             mm_motion_loaders = {}
#             motion_loaders['ground truth'] = gt_loader
#             for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
#                 if motion_loader_name != 'ground truth':
#                     continue
#                 motion_loader, mm_motion_loader = motion_loader_getter()
#                 motion_loaders[motion_loader_name] = motion_loader
#                 mm_motion_loaders[motion_loader_name] = mm_motion_loader

#             print(f'==================== Replication {replication} ====================')
#             print(f'==================== Replication {replication} ====================', file=f, flush=True)
#             print(f'Time: {datetime.now()}')
#             print(f'Time: {datetime.now()}', file=f, flush=True)
#             mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(motion_loaders, f)

#             if input_audio_file is not None:
#                 save_top10_similar_motions(acti_dict, input_audio_file, device)

#             print(f'Time: {datetime.now()}')
#             print(f'Time: {datetime.now()}', file=f, flush=True)
#             fid_score_dict = evaluate_fid(gt_loader, acti_dict, f)

#             print(f'Time: {datetime.now()}')
#             print(f'Time: {datetime.now()}', file=f, flush=True)
#             div_score_dict = evaluate_diversity(acti_dict, f)

#             print(f'Time: {datetime.now()}')
#             print(f'Time: {datetime.now()}', file=f, flush=True)
#             mm_score_dict = evaluate_multimodality(mm_motion_loaders, f)

#             print(f'!!! DONE !!!')
#             print(f'!!! DONE !!!', file=f, flush=True)

#             for key, item in mat_score_dict.items():
#                 if key not in all_metrics['MM Distance']:
#                     all_metrics['MM Distance'][key] = [item]
#                 else:
#                     all_metrics['MM Distance'][key] += [item]

#             for key, item in R_precision_dict.items():
#                 if key not in all_metrics['R_precision']:
#                     all_metrics['R_precision'][key] = [item]
#                 else:
#                     all_metrics['R_precision'][key] += [item]

#             for key, item in fid_score_dict.items():
#                 if key not in all_metrics['FID']:
#                     all_metrics['FID'][key] = [item]
#                 else:
#                     all_metrics['FID'][key] += [item]

#             for key, item in div_score_dict.items():
#                 if key not in all_metrics['Diversity']:
#                     all_metrics['Diversity'][key] = [item]
#                 else:
#                     all_metrics['Diversity'][key] += [item]

#             for key, item in mm_score_dict.items():
#                 if key not in all_metrics['MultiModality']:
#                     all_metrics['MultiModality'][key] = [item]
#                 else:
#                     all_metrics['MultiModality'][key] += [item]

#         for metric_name, metric_dict in all_metrics.items():
#             print('========== %s Summary ==========' % metric_name)
#             print('========== %s Summary ==========' % metric_name, file=f, flush=True)

#             for model_name, values in metric_dict.items():
#                 mean, conf_interval = get_metric_statistics(np.array(values))
#                 if isinstance(mean, np.float64) or isinstance(mean, np.float32):
#                     print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
#                     print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
#                 elif isinstance(mean, np.ndarray):
#                     line = f'---> [{model_name}]'
#                     for i in range(len(mean)):
#                         line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
#                     print(line)
#                     print(line, file=f, flush=True)
        
#         return all_metrics


# def extract_top_motions(input_audio_file, output_dir='/data1/hzy/top10_motions/'):
#     """
#     提取指定音频文件对应的前10个相似动作，并保存所有原始动作数据
    
#     Args:
#         input_audio_file: 输入音频特征文件路径
#         output_dir: 输出目录
#     """
#     # 加载数据
#     print("加载数据和模型...")
#     data_cfg = get_config("/data1/hzy/HumanMotion/RetrievalNet/configs/largedance/musicbody/datasets93.yaml").largedance_test
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     gt_loader, gt_dataset = get_dataset_motion_loader(data_cfg, 96)  
#     evalmodel_cfg = get_config("/data1/hzy/HumanMotion/RetrievalNet/configs/largedance/musicbody/InterCLIP.yaml")
#     eval_wrapper = EvaluatorModelWrapper(evalmodel_cfg, device)
    
#     # 收集所有嵌入向量
#     print("收集所有嵌入向量...")
#     motion_loaders = {'ground truth': gt_loader}
#     mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(motion_loaders, sys.stdout)
    
#     # 提取相似动作
#     print("提取相似动作...")
#     save_top10_similar_motions(acti_dict, input_audio_file, device, output_dir)
    
#     # 保存所有原始动作数据
#     print("保存所有原始动作数据...")
#     seqname = os.path.basename(input_audio_file).replace('.npy', '')
#     save_dir = os.path.join(output_dir, f'original_motions_{seqname}')
#     save_original_motion_data(gt_loader, save_dir)
    
#     return acti_dict


# if __name__ == '__main__':
#     mm_num_samples = 100
#     mm_num_repeats = 30
#     mm_num_times = 10
#     diversity_times = 300
#     replication_times = 1
#     batch_size = 96
    
#     eval_motion_loaders = {}
#     data_cfg = get_config("/data1/hzy/HumanMotion/RetrievalNet/configs/largedance/musicbody/datasets93.yaml").largedance_test
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     gt_loader, gt_dataset = get_dataset_motion_loader(data_cfg, batch_size)
#     evalmodel_cfg = get_config("/data1/hzy/HumanMotion/RetrievalNet/configs/largedance/musicbody/InterCLIP.yaml")
    
#     eval_wrapper = EvaluatorModelWrapper(evalmodel_cfg, device)
    
#     # 指定输入文件和输出目录
#     input_audio_file = '/data1/hzy/AllDataset/allmusic_librosa55/083.npy'
#     output_dir = '/data1/hzy/HumanMotion/RetrievalNet/top_motions_083'
    
#     # 创建输出目录
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 提取相似动作并保存所有原始动作数据
#     print(f"使用输入文件: {input_audio_file}")
#     print(f"输出目录: {output_dir}")
    
#     extract_top_motions(input_audio_file, output_dir)
    
#     print("完成！")