
import json
import os
import sys

import numpy as np
import torch

from collections import Counter

from tqdm import tqdm

motionembedding_dir = '/data2/hzy/InfiniteDance/InfiniteDanceData/dance/motionembeding'
config_path = '/data2/hzy/InfiniteDance/All_LargeDanceAR/RetrievalNet/checkpoints/AInterClip_Audio55Motion264/0512/train/bc256_s100l384_drop0.2_lr1e-4/InterCLIP.yaml'
device = 'cuda:4'

# 初始化模型包装器
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists
def sort_by_dist(data_list, reverse=False):
    sortlist = sorted(data_list, key=lambda x: x['dist'], reverse=reverse)
    for one in sortlist:
        del one['dist']
    return sortlist

def get_top10_similar_mofea_features(audio_feat, eval_wrapper, motionembedding_dir, device='cuda:0'):
    assert audio_feat.shape == (384, 55), "Input audio feature shape must be (384, 55)"
    audio_feat_tensor = torch.from_numpy(audio_feat).unsqueeze(0).to(device).to(torch.float32)

    audio_batch = ("query_sample", audio_feat_tensor, torch.zeros([1, 55], device=device))
    # breakpoint()
    with torch.no_grad():
        audio_embedding = eval_wrapper.get_co_embeddings(audio_batch, "audio").cpu().numpy()

    all_motion_embeddings = []
    seqnames = []
    for file in os.listdir(motionembedding_dir):
        if file.endswith('.npy'):
            motion_emb = np.load(os.path.join(motionembedding_dir, file))
            all_motion_embeddings.append(motion_emb)
            seqnames.append(file.replace('.npy', ''))

    all_motion_embeddings = np.stack(all_motion_embeddings, axis=0)

    dist_mat = euclidean_distance_matrix(audio_embedding, all_motion_embeddings)
    argsmax = np.argsort(dist_mat[0])[:10]

    top10 = []
    for idx in argsmax:
        top10.append({
            'name': seqnames[idx],
            'dist': dist_mat[0][idx],
        })

    sorted_top10 = sort_by_dist(top10)
    return sorted_top10

def get_top_mofea(name: str = None,
                              motion: np.ndarray = None,
                              idx: int = 0,
                              eval_wrapper=None,
                              motionembedding_dir="/data2/hzy/InfiniteDance/InfiniteDanceData/dance/motionembeding",
                              retrieval_path="/data2/hzy/InfiniteDance/InfiniteDanceData/dance/retrieval_s192_l384",
                              motion_base="/data2/hzy/InfiniteDance/InfiniteDanceData/dance/alldata_new_joint_vecs264",
                              style_map_path="/data2/hzy/InfiniteDance/InfiniteDanceData/styles/all_style_map.json",
                              style="Popular",
                              meta_path="meta",
                              device='cuda:0'):
    """
    支持两种输入：
    - name + idx：优先读取对应 JSON
    - motion: 若 JSON 不存在，则根据 motion 计算 top10
    """

    if name !=None:
        json_path = os.path.join(retrieval_path, f"{name}.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            # n = len(data)
            # if idx < 0:
            #     print(f"[BAD IDX] {idx}, fallback -> 0")
            #     idx = 0
            # elif idx >= n:
            #     print(f"[BAD IDX] {idx}, fallback -> {n-1}")
            #     idx = n - 1

            # musiclist = data[idx][:10]
            # print("data[idx]",data[idx])
            musiclist = data[idx][:10] if len(data[idx]) > 0 else [data[idx][0]] * 10

            # musiclist = data[idx][-10:]
            top10_results = [{'name': item['name']} for item in musiclist]
        else:
            # breakpoint()
            from RetrievalNet.configs import get_config
            from RetrievalNet.datasets import EvaluatorModelWrapper
            eval_wrapper = EvaluatorModelWrapper(get_config(config_path), device)
            if motion is None:
                # print("name",name)
                motion=np.load(f"/data2/hzy/InfiniteDance/InfiniteDanceData/music/musicfeature_55_allmusic_pure/{name}.npy")[idx*96:idx*96+384]
                if motion.shape[0] < 384:
                    pad_length = 384 - motion.shape[0]
                    motion = np.pad(motion, ((0, pad_length), (0, 0)), mode='wrap')

            # breakpoint()
            top10_results = get_top10_similar_mofea_features(motion, eval_wrapper, motionembedding_dir, device=device)
    else:
        from RetrievalNet.configs import get_config
        from RetrievalNet.datasets import EvaluatorModelWrapper
        eval_wrapper = EvaluatorModelWrapper(get_config(config_path), device)
        if motion is None:
            # print("name",name)
            motion=np.load(f"/data2/hzy/InfiniteDance/InfiniteDanceData/music/musicfeature_55_allmusic_pure/{name}.npy")[idx*96:idx*96+384]
            if motion.shape[0] < 384:
                pad_length = 384 - motion.shape[0]
                motion = np.pad(motion, ((0, pad_length), (0, 0)), mode='wrap')


        top10_results = get_top10_similar_mofea_features(motion, eval_wrapper, motionembedding_dir, device=device)

    # 提取 motion 片段
    results = []
    for item in top10_results:
        full_name = item['name']
        if '@' in full_name:
            video_part, frame_part = full_name.rsplit('@', 1)
            if '_' in frame_part:
                start_frame, end_frame = map(int, frame_part.split('_'))
                results.append((video_part, start_frame, end_frame))

        # 加载 style map
    with open(style_map_path, 'r') as f:
        style_map = json.load(f)

    genres = []
    for video_name, _, _ in results:
        genre = style_map.get(video_name, 'Unknown')
        genres.append(genre)

    genre_counts = Counter(genres)
    total = len(genres)

    # Calculate genre proportions
    genre_proportions = {genre: count / total for genre, count in genre_counts.items()}

    # Sort genres by proportion in descending order and take top 5
    top_5_genres = sorted(genre_proportions.items(), key=lambda x: x[1], reverse=True)[:5]

    # Assign weights (proportional to their proportions, normalized to sum to 1)
    total_proportion = sum(proportion for _, proportion in top_5_genres)
    weighted_genres = {genre: proportion / total_proportion for genre, proportion in top_5_genres}

    # Define priority genres
    GENRES = ["Ballet", "Popular", "Latin", "Modern", "Folk", "Classic"]

    # Check if name contains any priority genre (case-insensitive)
    name_lower = name.lower() if name else ""
    priority_genre = None
    for genre in GENRES:
        if genre.lower() in name_lower:
            priority_genre = genre
            break

    # Set top_genre: use priority genre if found, otherwise use highest-proportion genre
    if priority_genre:
        top_genre = priority_genre
    else:
        top_genre = top_5_genres[0][0] if top_5_genres else 'Popular'

    # Load Mean & Std
    mean = np.load(os.path.join(motion_base, meta_path, "Mean.npy"))
    std = np.load(os.path.join(motion_base, meta_path, "Std.npy"))

    mofea264_list = []
    for video_name, start, end in results:
        motion_path = os.path.join(motion_base, f"{video_name}.npy")
        if not os.path.exists(motion_path):
            print(f"Warning: {motion_path} not found, skipping...")
            continue

        motion_data = np.load(motion_path)
        if end > motion_data.shape[0]:
            print(f"Warning: {video_name} range ({start}-{end}) exceeds motion length {motion_data.shape[0]}")
            continue

        segment = motion_data[start:end]
        normed_segment = (segment - mean) / std
        mofea264_list.append(normed_segment)

    if not mofea264_list:
        print("No valid motion segments found.")
        return None, genre_proportions, top_genre, top10_results

    weights = np.array([10 - i for i in range(min(len(mofea264_list), 10))], dtype=np.float32)
    weights /= weights.sum()

    min_length = min(segment.shape[0] for segment in mofea264_list)
    aligned_segments = [segment[:min_length] for segment in mofea264_list]

    weighted_sum = np.zeros_like(aligned_segments[0], dtype=np.float32)
    for i, segment in enumerate(aligned_segments):
        weighted_sum += weights[i] * segment
    top10_names_indices = []
    for item in top10_results:
        full_name = item['name']
        if '@' in full_name:
            video_part, frame_part = full_name.rsplit('@', 1)
            if '_' in frame_part:
                start_idx, end_idx = map(int, frame_part.split('_'))
                top10_names_indices.append((video_part, start_idx, end_idx))
                
    # top10_names_indices
    # motiontoken_dir = "/data1/hzy/HumanMotion/All_mofea/Alldata/MotionTokens_512_vel_processed"
    motiontoken_dir = "/data2/hzy/InfiniteDance/InfiniteDanceData/dance/Infinite_MotionTokens_512_vel_processed"
    token_segments = []

    for name, start_idx, _ in top10_names_indices:
        token_file = os.path.join(motiontoken_dir, f"{name}.npy")
        if not os.path.exists(token_file):
            print(f"Warning: Token file {token_file} not found.")
            continue

        tokens = np.load(token_file)  # shape: (N,)

        token_index = round(start_idx / 96) * 72
        if token_index + 60 > len(tokens):
            print(f"Warning: Token index {token_index} out of range for file {name}.npy (len={len(tokens)}).")
            continue

        token_segment = tokens[token_index:token_index + 60]
        token_segments.append({
            "name": name,
            "start_idx": start_idx,
            "token_index": token_index,
            "tokens": token_segment.tolist(),  # or keep as np.array if needed
        })

    # 输出结果
    # print("\nExtracted token segments (first 60 tokens each):")
    # for i, item in enumerate(token_segments):
    #     print(f"Top {i+1}: {item['name']} @ token_index={item['token_index']}, tokens[:5]={item['tokens'][:5]}")
    # print("top_genre",top_genre)

    return weighted_sum, genre_proportions, top_genre, top10_results,top10_names_indices,token_segments


def get_items_by_style_and_idx(retrieval_filepath: str, 
                                    target_style: str, 
                                    target_idx: int) -> list:
    """
    从 *预处理后* 的文件中，瞬时 (O(1)) 获取匹配的条目列表。
    
    参数:
    retrieval_filepath (str): *新* 文件夹中的文件路径
                             (e.g., ".../retrieval_fast_lookup/music_file.json")
    target_style (str): 你要查找的目标流派 (e.g., "Popular")
    target_idx (int): 你要查找的指定索引 (e.g., 0)
    
    返回:
    list: 包含所有匹配的 item 字典的列表 (相对顺序被保留)
    """
    
    try:
        with open(retrieval_filepath, 'r') as f:
            # data 是 { "idx_0": { "Style": [...] }, ... }
            data = json.load(f)
                    
        # 1. 直接查找 idx (e.g., "idx_0")
        style_dict = data.get(f"idx_{target_idx}", {})
        
        # 2. 直接查找 style (e.g., "Popular")
        # .get(..., []) 确保如果流派不存在, 也返回一个空列表, 而不是报错
        items_list = style_dict.get(target_style, [])
        items_list_top10=[item["name"] for item in items_list][:10]

        # breakpoint()
        items_list_top10_2=[{'name': item['name']} for item in items_list][:10]
        # print("items_list_top10",items_list_top10_2)
        return items_list_top10,items_list_top10_2
        
    except FileNotFoundError:
        print(f"错误: 找不到指定的检索文件: {retrieval_filepath}")
        return []
    except json.JSONDecodeError:
        print(f"错误: 无法解析检索文件: {retrieval_filepath}")
        return []
    except Exception as e:
        print(f"发生意外错误: {e}")
        return []



def get_top_mofea_specific_style(name: str = None,
                              motion: np.ndarray = None,
                              idx: int = 0,
                              eval_wrapper=None,
                              motionembedding_dir="/data2/hzy/InfiniteDance/InfiniteDanceData/dance/motionembeding",
                              retrieval_path="/data2/hzy/InfiniteDance/InfiniteDanceData/dance/retrieval_s192_l384_style",
                              motion_base="/data2/hzy/InfiniteDance/InfiniteDanceData/dance/alldata_new_joint_vecs264",
                              style_map_path="/data2/hzy/InfiniteDance/InfiniteDanceData/styles/all_style_map.json",
                              style="Popular",
                              meta_path="meta",
                              device='cuda:0',
                              windows_length=120,
                              infertype="infinitedance"):
    """
    支持两种输入：
    - name + idx：优先读取对应 JSON
    - motion: 若 JSON 不存在，则根据 motion 计算 top10
    
    Args:
        infertype: "infinitedance" (默认) 使用旧格式 tokens，或 "infinitedanceplus" 使用 v6_2 窗口化 tokens
    """
    if name==None and motion is None:
        raise ValueError("必须提供 name 或 motion 其中之一。")
    if name !=None:
        json_path = os.path.join(retrieval_path, f"{name}.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            # n = len(data)
            # if idx < 0:
            #     print(f"[BAD IDX] {idx}, fallback -> 0")
            #     idx = 0
            # elif idx >= n:
            #     print(f"[BAD IDX] {idx}, fallback -> {n-1}")
            #     idx = n - 1

            # musiclist = data[idx][:10]
            # print("data[idx]",data[idx])
            # musiclist = data[idx][:10] if len(data[idx]) > 0 else [data[idx][0]] * 10
            _,top10_results = get_items_by_style_and_idx(
                    retrieval_filepath=json_path,
                    target_style=style,
                    target_idx=idx
                )
            

            # musiclist = data[idx][-10:]
            # top10_results = [{'name': item['name']} for item in musiclist]
        else:
            # breakpoint()
            from RetrievalNet.configs import get_config
            from RetrievalNet.datasets import EvaluatorModelWrapper
            eval_wrapper = EvaluatorModelWrapper(get_config(config_path), device)
            if motion is None:
                # print("name",name)
                motion=np.load(f"/data2/hzy/InfiniteDance/InfiniteDanceData/music/musicfeature_55_allmusic_pure/{name}.npy")[idx*96:idx*96+384]
                if motion.shape[0] < 384:
                    pad_length = 384 - motion.shape[0]
                    motion = np.pad(motion, ((0, pad_length), (0, 0)), mode='wrap')

            # breakpoint()
            top10_results = get_top10_similar_mofea_features(motion, eval_wrapper, motionembedding_dir, device=device)
    else:
        from RetrievalNet.configs import get_config
        from RetrievalNet.datasets import EvaluatorModelWrapper
        eval_wrapper = EvaluatorModelWrapper(get_config(config_path), device)
        if motion is None:
            # print("name",name)
            motion=np.load(f"/data2/hzy/InfiniteDance/InfiniteDanceData/music/musicfeature_55_allmusic_pure/{name}.npy")[idx*96:idx*96+384]
            if motion.shape[0] < 384:
                pad_length = 384 - motion.shape[0]
                motion = np.pad(motion, ((0, pad_length), (0, 0)), mode='wrap')


        top10_results = get_top10_similar_mofea_features(motion, eval_wrapper, motionembedding_dir, device=device)

    # 提取 motion 片段
    results = []
    for item in top10_results:
        full_name = item['name']
        if '@' in full_name:
            video_part, frame_part = full_name.rsplit('@', 1)
            if '_' in frame_part:
                start_frame, end_frame = map(int, frame_part.split('_'))
                results.append((video_part, start_frame, end_frame))

        # 加载 style map
    with open(style_map_path, 'r') as f:
        style_map = json.load(f)

    genres = []
    for video_name, _, _ in results:
        genre = style_map.get(video_name, 'Unknown')
        genres.append(genre)

    genre_counts = Counter(genres)
    total = len(genres)

    # Calculate genre proportions
    genre_proportions = {genre: count / total for genre, count in genre_counts.items()}

    # Sort genres by proportion in descending order and take top 5
    top_5_genres = sorted(genre_proportions.items(), key=lambda x: x[1], reverse=True)[:5]

    # Assign weights (proportional to their proportions, normalized to sum to 1)
    total_proportion = sum(proportion for _, proportion in top_5_genres)
    weighted_genres = {genre: proportion / total_proportion for genre, proportion in top_5_genres}

    # Define priority genres
    GENRES = ["Ballet", "Popular", "Latin", "Modern", "Folk", "Classic"]

    # Check if name contains any priority genre (case-insensitive)
    name_lower = name.lower() if name else ""
    priority_genre = None
    for genre in GENRES:
        if genre.lower() in name_lower:
            priority_genre = genre
            break

    # Set top_genre: use priority genre if found, otherwise use highest-proportion genre
    if priority_genre:
        top_genre = priority_genre
    else:
        top_genre = top_5_genres[0][0] if top_5_genres else 'Popular'

    # Load Mean & Std
    mean = np.load(os.path.join(motion_base, meta_path, "Mean.npy"))
    std = np.load(os.path.join(motion_base, meta_path, "Std.npy"))

    mofea264_list = []
    for video_name, start, end in results:
        motion_path = os.path.join(motion_base, f"{video_name}.npy")
        if not os.path.exists(motion_path):
            print(f"Warning: {motion_path} not found, skipping...")
            continue

        motion_data = np.load(motion_path)
        if end > motion_data.shape[0]:
            print(f"Warning: {video_name} range ({start}-{end}) exceeds motion length {motion_data.shape[0]}")
            continue

        segment = motion_data[start:end]
        normed_segment = (segment - mean) / std
        mofea264_list.append(normed_segment)

    if not mofea264_list:
        print("No valid motion segments found.")
        return None, genre_proportions, top_genre, top10_results

    weights = np.array([10 - i for i in range(min(len(mofea264_list), 10))], dtype=np.float32)
    weights /= weights.sum()

    min_length = min(segment.shape[0] for segment in mofea264_list)
    aligned_segments = [segment[:min_length] for segment in mofea264_list]

    weighted_sum = np.zeros_like(aligned_segments[0], dtype=np.float32)
    for i, segment in enumerate(aligned_segments):
        weighted_sum += weights[i] * segment
    top10_names_indices = []
    for item in top10_results:
        full_name = item['name']
        if '@' in full_name:
            video_part, frame_part = full_name.rsplit('@', 1)
            if '_' in frame_part:
                start_idx, end_idx = map(int, frame_part.split('_'))
                top10_names_indices.append((video_part, start_idx, end_idx))
                
    # top10_names_indices
    # motiontoken_dir = "/data1/hzy/HumanMotion/All_mofea/Alldata/MotionTokens_512_vel_processed"
    motiontoken_dir = "/data2/hzy/InfiniteDance/InfiniteDanceData/dance/Infinite_MotionTokens_512_vel_processed"
    # v6_2 使用的新路径（窗口化的 tokens）
    v6_2_token_dir = "/data2/hzy/InfiniteDance/InfiniteDanceData/ALL_SD_ID_DATA/motion_264_30fps_tokens_1layer_windowed"
    token_segments = []

    # 根据 infertype 决定是否使用 v6_2 格式
    use_v6_2 = (infertype == "infinitedanceplus")

    for name, start_idx, end_idx in top10_names_indices:
        # 如果 infertype 是 infinitedanceplus，优先尝试 v6_2 的窗口化 token 文件
        # 流程：先得到 retrieval 结果，再得到 token
        # 1. Retrieval 返回: name@start_idx_end_idx (motion 帧索引)
        #    例如: _P-JWcq1ewI_01_187_1358@0_384
        #    例如: Popular-HomeDance804@384_768
        # 2. 转换为 token 索引: token_start = start_idx // 4, token_end = end_idx // 4
        #    例如: 0_384 -> 0_96 (384/4=96)
        #    例如: 384_768 -> 96_192 (384/4=96, 768/4=192)
        # 3. 匹配窗口文件: {name}_{token_start}-{token_end}.npy
        #    例如: _P-JWcq1ewI_01_187_1358_0-96.npy
        #    例如: Popular-HomeDance804_96-192.npy
        v6_2_token_file = None
        if use_v6_2 and os.path.exists(v6_2_token_dir):
            # 将 motion 帧索引转换为 token 索引
            token_start_idx = start_idx // 4
            token_end_idx = end_idx // 4
            
            # 方法1: 先尝试精确匹配（如果存在完全匹配的窗口文件）
            exact_match_file = os.path.join(v6_2_token_dir, f"{name}_{token_start_idx}-{token_end_idx}.npy")
            if os.path.exists(exact_match_file):
                v6_2_token_file = exact_match_file
            else:
                # 方法2: 如果没有精确匹配，列出所有匹配的窗口文件，找到最接近的
                import glob
                pattern = os.path.join(v6_2_token_dir, f"{name}_*.npy")
                matching_files = glob.glob(pattern)
                
                if matching_files:
                    # 解析文件名，找到最接近 token_start_idx 的窗口
                    best_file = None
                    best_distance = float('inf')
                    
                    for file_path in matching_files:
                        filename = os.path.basename(file_path)
                        # 解析文件名: {name}_{start}-{end}.npy
                        # 注意：文件中的 start 和 end 是 token 索引，不是 motion 帧索引
                        try:
                            parts = filename.replace('.npy', '').split('_')
                            if len(parts) >= 2:
                                window_range = parts[-1]  # 例如 "0-96"
                                window_start, window_end = map(int, window_range.split('-'))
                                # 计算距离：如果 token_start_idx 在窗口内，距离=0；否则计算到窗口中心的距离
                                if window_start <= token_start_idx < window_end:
                                    distance = 0
                                else:
                                    window_center = (window_start + window_end) // 2
                                    distance = abs(token_start_idx - window_center)
                                
                                if distance < best_distance:
                                    best_distance = distance
                                    best_file = file_path
                        except (ValueError, IndexError):
                            continue
                    
                    if best_file and os.path.exists(best_file):
                        v6_2_token_file = best_file
        
        if use_v6_2 and v6_2_token_file and os.path.exists(v6_2_token_file):
            # 使用 v6_2 的窗口化 tokens
            tokens = np.load(v6_2_token_file)  # shape: (96,)
            # v6_2 tokens 是相对值 (0-511)，直接使用
            # 注意：窗口文件中的 tokens 已经是完整的窗口（96个tokens），
            # 对应 retrieval 返回的 384 帧 motion (384 / 4 = 96)
            if len(tokens) >= windows_length:
                token_segment = tokens[:windows_length]
            else:
                token_segment = tokens
            token_segments.append({
                "name": name,
                "start_idx": start_idx,  # 原始 motion 帧索引（用于记录）
                "token_index": 0,  # 窗口化文件，从 0 开始
                "tokens": token_segment.tolist(),  # 相对值 (0-511)
            })
        else:
            # 回退到旧的 token 文件格式
            # 旧格式：Infinite_MotionTokens_512_vel_processed/{name}.npy
            # - 文件包含整个视频的 tokens（不是窗口化的）
            # - tokens 是绝对值（4096+），范围约 4096-4607
            # - 索引计算：token_index = round(start_idx / 96) * 72
            token_file = os.path.join(motiontoken_dir, f"{name}.npy")
            if not os.path.exists(token_file):
                print(f"Warning: Token file {token_file} not found.")
                continue

            tokens = np.load(token_file)  # shape: (N,)

            token_index = round(start_idx / 96) * 72
            if token_index + windows_length > len(tokens):
                print(f"Warning: Token index {token_index} out of range for file {name}.npy (len={len(tokens)}).")
                continue

            token_segment = tokens[token_index:token_index + windows_length]
            token_segments.append({
                "name": name,
                "start_idx": start_idx,
                "token_index": token_index,
                "tokens": token_segment.tolist(),  # 绝对值 (4096+)
            })

    # 输出结果
    # print("\nExtracted token segments (first 60 tokens each):")
    # for i, item in enumerate(token_segments):
    #     print(f"Top {i+1}: {item['name']} @ token_index={item['token_index']}, tokens[:5]={item['tokens'][:5]}")
    # print("top_genre",top_genre)

    return weighted_sum, genre_proportions, top_genre, top10_results,top10_names_indices,token_segments

