# '''
# 20250907:修改encoder
# '''
# import json
# import os
# from collections import Counter
# import torch
# import numpy as np
# # def get_top10_dance_cache_retrieval_embeds(name: str, muidx: int,
# #                        retrieval_path="../InfiniteDanceData",
# #                        motion_base="../InfiniteDanceData",
# #                        meta_path="meta", net=None):
# #     '''
# #     new version, use dance embeddings from quantizers
# #     '''
# #     '''
# #     已经跑好的encoder下的quantized embeds在../InfiniteDanceData
# #     '''
# #     if net is None:
# #         raise ValueError("net (VQVAE model) must be provided")
# #     net = net.eval()
    
# #     # 获取当前设备（如果模型已经在某个设备上，就使用该设备）
# #     device = next(net.parameters()).device if next(net.parameters()).is_cuda else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #     net = net.to(device)  # 确保模型在正确的设备上
    
# #     json_path = os.path.join(retrieval_path, f"{name}.json")

# #     if not os.path.exists(json_path):
# #         motion = np.load(f"../InfiniteDanceData")
# #         num_frames, dim = motion.shape

# #         if num_frames >= 384:
# #             motion_fixed = motion[:384]
# #         else:
# #             pad_len = 384 - num_frames
# #             padding = np.zeros((pad_len, dim), dtype=motion.dtype)
# #             motion_fixed = np.concatenate([motion, padding], axis=0)
# #         motion_fixed = torch.tensor(motion_fixed).float().to(device).unsqueeze(0)
# #         mmotion_fixed_quantized = net.forward_quantizer(motion_fixed)
# #         mmotion_fixed_quantized=mmotion_fixed_quantized.squeeze(0).permute(1,0).cpu().numpy()
# #         assert (mmotion_fixed_quantized.shape[1] == 1024)
# #         assert (mmotion_fixed_quantized.shape[0] == 96)
# #         return mmotion_fixed_quantized

# #     # 读取 JSON 数据
# #     with open(json_path, 'r') as f:
# #         data = json.load(f)

# #     # 解析前 10 个结果
# #     musiclist = data[muidx][0:10]
# #     names = [item['name'] for item in musiclist]
# #     result = []
# #     for full_name in names:
# #         if '@' in full_name:
# #             video_part, frame_part = full_name.rsplit('@', 1)
# #             if '_' in frame_part:
# #                 start_frame, end_frame = map(int, frame_part.split('_'))
# #                 result.append((video_part, start_frame, end_frame))

# #     # 加载 Mean 和 Std
# #     mean_path = os.path.join(motion_base, meta_path, "Mean.npy")
# #     std_path = os.path.join(motion_base, meta_path, "Std.npy")
# #     mean = np.load(mean_path)
# #     std = np.load(std_path)

# #     # 读取每个 .npy 文件并截取 & 归一化
# #     mofea264_list = []
# #     for video_name, start, end in result:
# #         motion_path = os.path.join(motion_base, f"{video_name}.npy")
# #         if not os.path.exists(motion_path):
# #             print(f"Warning: {motion_path} not found, skipping...")
# #             continue

# #         motion = np.load(motion_path)  # shape: (T, 264)
# #         if end > motion.shape[0]:
# #             print(f"Warning: {video_name} range ({start}-{end}) exceeds motion length {motion.shape[0]}")
# #             continue

# #         segment = motion[start:end]
# #         normed_segment = (segment - mean) / std
# #         assert (normed_segment.shape[1] == 264)
# #         mofea264_list.append(normed_segment)

# #     if not mofea264_list:
# #         print("No valid motion segments found.")
# #         return None

# #     # 将 NumPy 数组转换为 PyTorch 张量，并移动到与模型相同的设备
# #     mofea264_tensor = torch.stack([torch.from_numpy(x) for x in mofea264_list], dim=0).to(device)
# #     # print("mofea264_tensor", mofea264_tensor.shape)

# #     # 使用 net 的量化器
# #     with torch.no_grad():  # 添加无梯度计算上下文，节省内存
# #         mofea264_tensor_quantized = net.forward_quantizer(mofea264_tensor)
# #     # print("mofea264_tensor_quantized", mofea264_tensor_quantized.shape)
    
# #     # 调整维度
# #     mofea264_tensor_quantized = mofea264_tensor_quantized.permute(0, 2, 1)

# #     # 定义权重：序号越前（0 到 9），权重越大，例如 [10, 9, 8, ..., 1]
# #     # 将权重也移动到相同设备
# #     weights = torch.tensor([10 - i for i in range(min(len(mofea264_tensor_quantized), 10))], 
# #                           dtype=torch.float32, device=device)
# #     weights /= weights.sum()  # 归一化权重，使其和为 1

# #     # 确保所有片段长度一致（取最短长度）
# #     min_length = min(segment.shape[0] for segment in mofea264_tensor_quantized)
# #     aligned_segments = [segment[:min_length] for segment in mofea264_tensor_quantized]

# #     # 加权求和
# #     weighted_sum = torch.zeros_like(aligned_segments[0], dtype=torch.float32, device=device)
# #     for i, segment in enumerate(aligned_segments):
# #         weighted_sum += weights[i] * segment
    
# #     # 将结果移回CPU（如果需要）
# #     weighted_sum = weighted_sum.cpu()
# #     # print("weighted_sum", weighted_sum.shape)
# #     assert (weighted_sum.shape[1] == 1024)
# #     assert (weighted_sum.shape[0] == 96)
# #     # 返回的是 CPU 张量
# #     return weighted_sum  # 返回的是 CPU 张量
# # def get_top10_dance_retrieval_embeds(name: str, muidx: int,
# #                        retrieval_path="../InfiniteDanceData",
# #                        motion_base="../InfiniteDanceData",
# #                        meta_path="meta", net=None):
# #     '''
# #     new version, use dance embeddings from quantizers
# #     '''
# #     if net is None:
# #         raise ValueError("net (VQVAE model) must be provided")
# #     net = net.eval()
    
# #     # 获取当前设备（如果模型已经在某个设备上，就使用该设备）
# #     device = next(net.parameters()).device if next(net.parameters()).is_cuda else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #     net = net.to(device)  # 确保模型在正确的设备上
    
# #     json_path = os.path.join(retrieval_path, f"{name}.json")

# #     if not os.path.exists(json_path):
# #         motion = np.load(f"../InfiniteDanceData")
# #         num_frames, dim = motion.shape

# #         if num_frames >= 384:
# #             motion_fixed = motion[:384]
# #         else:
# #             pad_len = 384 - num_frames
# #             padding = np.zeros((pad_len, dim), dtype=motion.dtype)
# #             motion_fixed = np.concatenate([motion, padding], axis=0)
# #         motion_fixed = torch.tensor(motion_fixed).float().to(device).unsqueeze(0)
# #         mmotion_fixed_quantized = net.forward_quantizer(motion_fixed)
# #         mmotion_fixed_quantized=mmotion_fixed_quantized.squeeze(0).permute(1,0).cpu().numpy()
# #         assert (mmotion_fixed_quantized.shape[1] == 1024)
# #         assert (mmotion_fixed_quantized.shape[0] == 96)
# #         return mmotion_fixed_quantized

# #     # 读取 JSON 数据
# #     with open(json_path, 'r') as f:
# #         data = json.load(f)

# #     # 解析前 10 个结果
# #     musiclist = data[muidx][0:10]
# #     names = [item['name'] for item in musiclist]
# #     result = []
# #     for full_name in names:
# #         if '@' in full_name:
# #             video_part, frame_part = full_name.rsplit('@', 1)
# #             if '_' in frame_part:
# #                 start_frame, end_frame = map(int, frame_part.split('_'))
# #                 result.append((video_part, start_frame, end_frame))

# #     # 加载 Mean 和 Std
# #     mean_path = os.path.join(motion_base, meta_path, "Mean.npy")
# #     std_path = os.path.join(motion_base, meta_path, "Std.npy")
# #     mean = np.load(mean_path)
# #     std = np.load(std_path)

# #     # 读取每个 .npy 文件并截取 & 归一化
# #     mofea264_list = []
# #     for video_name, start, end in result:
# #         motion_path = os.path.join(motion_base, f"{video_name}.npy")
# #         if not os.path.exists(motion_path):
# #             print(f"Warning: {motion_path} not found, skipping...")
# #             continue

# #         motion = np.load(motion_path)  # shape: (T, 264)
# #         if end > motion.shape[0]:
# #             print(f"Warning: {video_name} range ({start}-{end}) exceeds motion length {motion.shape[0]}")
# #             continue

# #         segment = motion[start:end]
# #         normed_segment = (segment - mean) / std
# #         assert (normed_segment.shape[1] == 264)
# #         mofea264_list.append(normed_segment)

# #     if not mofea264_list:
# #         print("No valid motion segments found.")
# #         return None

# #     # 将 NumPy 数组转换为 PyTorch 张量，并移动到与模型相同的设备
# #     mofea264_tensor = torch.stack([torch.from_numpy(x) for x in mofea264_list], dim=0).to(device)
# #     # print("mofea264_tensor", mofea264_tensor.shape)

# #     # 使用 net 的量化器
# #     with torch.no_grad():  # 添加无梯度计算上下文，节省内存
# #         mofea264_tensor_quantized = net.forward_quantizer(mofea264_tensor)
# #     # print("mofea264_tensor_quantized", mofea264_tensor_quantized.shape)
    
# #     # 调整维度
# #     mofea264_tensor_quantized = mofea264_tensor_quantized.permute(0, 2, 1)

# #     # 定义权重：序号越前（0 到 9），权重越大，例如 [10, 9, 8, ..., 1]
# #     # 将权重也移动到相同设备
# #     weights = torch.tensor([10 - i for i in range(min(len(mofea264_tensor_quantized), 10))], 
# #                           dtype=torch.float32, device=device)
# #     weights /= weights.sum()  # 归一化权重，使其和为 1

# #     # 确保所有片段长度一致（取最短长度）
# #     min_length = min(segment.shape[0] for segment in mofea264_tensor_quantized)
# #     aligned_segments = [segment[:min_length] for segment in mofea264_tensor_quantized]

# #     # 加权求和
# #     weighted_sum = torch.zeros_like(aligned_segments[0], dtype=torch.float32, device=device)
# #     for i, segment in enumerate(aligned_segments):
# #         weighted_sum += weights[i] * segment
    
# #     # 将结果移回CPU（如果需要）
# #     weighted_sum = weighted_sum.cpu()
# #     # print("weighted_sum", weighted_sum.shape)
# #     assert (weighted_sum.shape[1] == 1024)
# #     assert (weighted_sum.shape[0] == 96)
# #     # 返回的是 CPU 张量
# #     return weighted_sum  # 返回的是 CPU 张量

# # def get_top10_dance_retrieval_embeds_tensor(name: str, muidx: int,
# #                        retrieval_path="../InfiniteDanceData",
# #                        motion_base="../InfiniteDanceData",
# #                        meta_path="meta",net=None):
# #     '''
# #     new version,use dance embeddings from quantizers
# #     '''
# #     if net is None:
# #         raise ValueError("net (VQVAE model) must be provided")
    

# #     json_path = os.path.join(retrieval_path, f"{name}.json")

# #     if not os.path.exists(json_path):
# #         motion = np.load(f"../InfiniteDanceData")
# #         num_frames, dim = motion.shape

# #         if num_frames >= 384:
# #             motion_fixed = motion[:384]
# #         else:
# #             pad_len = 384 - num_frames
# #             padding = np.zeros((pad_len, dim), dtype=motion.dtype)
# #             motion_fixed = np.concatenate([motion, padding], axis=0)

# #         return motion_fixed

# #     # 读取 JSON 数据
# #     with open(json_path, 'r') as f:
# #         data = json.load(f)

# #     # 解析前 10 个结果
# #     musiclist = data[muidx][0:10]
# #     names = [item['name'] for item in musiclist]
# #     result = []
# #     for full_name in names:
# #         if '@' in full_name:
# #             video_part, frame_part = full_name.rsplit('@', 1)
# #             if '_' in frame_part:
# #                 start_frame, end_frame = map(int, frame_part.split('_'))
# #                 result.append((video_part, start_frame, end_frame))

# #     # 加载 Mean 和 Std
# #     mean_path = os.path.join(motion_base, meta_path, "Mean.npy")
# #     std_path = os.path.join(motion_base, meta_path, "Std.npy")
# #     mean = np.load(mean_path)
# #     std = np.load(std_path)

# #     # 读取每个 .npy 文件并截取 & 归一化
# #     mofea264_list = []
# #     for video_name, start, end in result:
# #         motion_path = os.path.join(motion_base, f"{video_name}.npy")
# #         if not os.path.exists(motion_path):
# #             print(f"Warning: {motion_path} not found, skipping...")
# #             continue

# #         motion = np.load(motion_path)  # shape: (T, 264)
# #         if end > motion.shape[0]:
# #             print(f"Warning: {video_name} range ({start}-{end}) exceeds motion length {motion.shape[0]}")
# #             continue

# #         segment = motion[start:end]
# #         normed_segment = (segment - mean) / std
        
        
# #         mofea264_list.append(normed_segment)

# #     # 加权求和
# #     mofea264_tensor = torch.cat(mofea264_list, dim=0)
# #     mofea264_tensor_quantized=net.forward_quantizer(torch.tensor(mofea264_tensor).float().cuda())
# #     mofea264_tensor_quantized=mofea264_tensor_quantized.cpu().numpy()
# #     mofea264_tensor_quantized=mofea264_tensor_quantized.permute(0,2,1)
    
# #     if not mofea264_list:
# #         print("No valid motion segments found.")
# #         return None

# #     # 定义权重：序号越前（0 到 9），权重越大，例如 [10, 9, 8, ..., 1]
# #     weights = np.array([10 - i for i in range(min(len(mofea264_tensor_quantized), 10))], dtype=np.float32)
# #     weights /= weights.sum()  # 归一化权重，使其和为 1

# #     # 确保所有片段长度一致（取最短长度）
# #     min_length = min(segment.shape[0] for segment in mofea264_tensor_quantized)
# #     aligned_segments = [segment[:min_length] for segment in mofea264_tensor_quantized]

# #     # 加权求和
# #     weighted_sum = np.zeros_like(aligned_segments[0], dtype=np.float32)
# #     for i, segment in enumerate(aligned_segments):
# #         weighted_sum += weights[i] * segment
    
# #     return weighted_sum


# def get_items_by_style_and_idx(retrieval_filepath: str, 
#                                     target_style: str, 
#                                     target_idx: int) -> list:
#     """
#     从 *预处理后* 的文件中，瞬时 (O(1)) 获取匹配的条目列表。
    
#     参数:
#     retrieval_filepath (str): *新* 文件夹中的文件路径
#                              (e.g., ".../retrieval_fast_lookup/music_file.json")
#     target_style (str): 你要查找的目标流派 (e.g., "Popular")
#     target_idx (int): 你要查找的指定索引 (e.g., 0)
    
#     返回:
#     list: 包含所有匹配的 item 字典的列表 (相对顺序被保留)
#     """
    
#     try:
#         with open(retrieval_filepath, 'r') as f:
#             # data 是 { "idx_0": { "Style": [...] }, ... }
#             data = json.load(f)
                    
#         # 1. 直接查找 idx (e.g., "idx_0")
#         style_dict = data.get(f"idx_{target_idx}", {})
        
#         # 2. 直接查找 style (e.g., "Popular")
#         # .get(..., []) 确保如果流派不存在, 也返回一个空列表, 而不是报错
#         items_list = style_dict.get(target_style, [])
#         items_list_top10=[item["name"] for item in items_list][:10]

#         # breakpoint()
        
#         return items_list_top10
        
#     except FileNotFoundError:
#         print(f"错误: 找不到指定的检索文件: {retrieval_filepath}")
#         return []
#     except json.JSONDecodeError:
#         print(f"错误: 无法解析检索文件: {retrieval_filepath}")
#         return []
#     except Exception as e:
#         print(f"发生意外错误: {e}")
#         return []


# def get_top10_mofea264_1(name: str, muidx: int,
#                        retrieval_path="../InfiniteDanceData",
#                        motion_base="../InfiniteDanceData",
#                        meta_path="meta"):
#     '''
#     old version,use 264dim motion features
#     '''

#     json_path = os.path.join(retrieval_path, f"{name}.json")

#     if not os.path.exists(json_path):
#         motion = np.load(f"../InfiniteDanceData")
#         num_frames, dim = motion.shape

#         if num_frames >= 384:
#             motion_fixed = motion[:384]
#         else:
#             pad_len = 384 - num_frames
#             padding = np.zeros((pad_len, dim), dtype=motion.dtype)
#             motion_fixed = np.concatenate([motion, padding], axis=0)

#         return motion_fixed

#     # 读取 JSON 数据
#     with open(json_path, 'r') as f:
#         data = json.load(f)

#     # 解析前 10 个结果
#     musiclist = data[muidx][0:10]
#     names = [item['name'] for item in musiclist]
#     result = []
#     for full_name in names:
#         if '@' in full_name:
#             video_part, frame_part = full_name.rsplit('@', 1)
#             if '_' in frame_part:
#                 start_frame, end_frame = map(int, frame_part.split('_'))
#                 result.append((video_part, start_frame, end_frame))

#     # 加载 Mean 和 Std
#     mean_path = os.path.join(motion_base, meta_path, "Mean.npy")
#     std_path = os.path.join(motion_base, meta_path, "Std.npy")
#     mean = np.load(mean_path)
#     std = np.load(std_path)

#     # 读取每个 .npy 文件并截取 & 归一化
#     mofea264_list = []
#     for video_name, start, end in result:
#         motion_path = os.path.join(motion_base, f"{video_name}.npy")
#         if not os.path.exists(motion_path):
#             print(f"Warning: {motion_path} not found, skipping...")
#             continue

#         motion = np.load(motion_path)  # shape: (T, 264)
#         if end > motion.shape[0]:
#             print(f"Warning: {video_name} range ({start}-{end}) exceeds motion length {motion.shape[0]}")
#             continue

#         segment = motion[start:end]
#         normed_segment = (segment - mean) / std
#         mofea264_list.append(normed_segment)

#     # 加权求和
#     if not mofea264_list:
#         print("No valid motion segments found.")
#         return None

#     # 定义权重：序号越前（0 到 9），权重越大，例如 [10, 9, 8, ..., 1]
#     weights = np.array([10 - i for i in range(min(len(mofea264_list), 10))], dtype=np.float32)
#     weights /= weights.sum()  # 归一化权重，使其和为 1

#     # 确保所有片段长度一致（取最短长度）
#     min_length = min(segment.shape[0] for segment in mofea264_list)
#     aligned_segments = [segment[:min_length] for segment in mofea264_list]

#     # 加权求和
#     weighted_sum = np.zeros_like(aligned_segments[0], dtype=np.float32)
#     for i, segment in enumerate(aligned_segments):
#         weighted_sum += weights[i] * segment

#     return weighted_sum

# def get_top10_mofea264_specific_style(name: str, muidx: int,
#                        retrieval_path="../InfiniteDanceData",
#                        motion_base="../InfiniteDanceData",
#                        style="Popular",
#                        meta_path="meta"):
#     '''
#     old version,use 264dim motion features
#     '''

#     json_path = os.path.join(retrieval_path, f"{name}.json")

#     if not os.path.exists(json_path):
#         motion = np.load(f"../InfiniteDanceData")
#         num_frames, dim = motion.shape

#         if num_frames >= 384:
#             motion_fixed = motion[:384]
#         else:
#             pad_len = 384 - num_frames
#             padding = np.zeros((pad_len, dim), dtype=motion.dtype)
#             motion_fixed = np.concatenate([motion, padding], axis=0)

#         return motion_fixed

#     # 读取 JSON 数据
#     # with open(json_path, 'r') as f:
#     #     data = json.load(f)
        

#     # # 解析前 10 个结果
#     # musiclist = data[muidx][0:10]
#     # names = [item['name'] for item in musiclist]
#     if os.path.exists(json_path):

#             # musiclist = data[idx][:10]

#             # musiclist = data[idx][-10:]
#         musiclist = get_items_by_style_and_idx(
#                     retrieval_filepath=json_path,
#                     target_style=style,
#                     target_idx=muidx
#                 )
#     # names = [item['name'] for item in musiclist]
#     result = []
#     for full_name in musiclist:
#         if '@' in full_name:
#             video_part, frame_part = full_name.rsplit('@', 1)
#             if '_' in frame_part:
#                 start_frame, end_frame = map(int, frame_part.split('_'))
#                 result.append((video_part, start_frame, end_frame))

#     # 加载 Mean 和 Std
#     mean_path = os.path.join(motion_base, meta_path, "Mean.npy")
#     std_path = os.path.join(motion_base, meta_path, "Std.npy")
#     mean = np.load(mean_path)
#     std = np.load(std_path)

#     # 读取每个 .npy 文件并截取 & 归一化
#     mofea264_list = []
#     for video_name, start, end in result:
#         motion_path = os.path.join(motion_base, f"{video_name}.npy")
#         if not os.path.exists(motion_path):
#             print(f"Warning: {motion_path} not found, skipping...")
#             continue

#         motion = np.load(motion_path)  # shape: (T, 264)
#         if end > motion.shape[0]:
#             print(f"Warning: {video_name} range ({start}-{end}) exceeds motion length {motion.shape[0]}")
#             continue

#         segment = motion[start:end]
#         normed_segment = (segment - mean) / std
#         mofea264_list.append(normed_segment)

#     # 加权求和
#     if not mofea264_list:
#         print("No valid motion segments found.")
#         return None

#     # 定义权重：序号越前（0 到 9），权重越大，例如 [10, 9, 8, ..., 1]
#     weights = np.array([10 - i for i in range(min(len(mofea264_list), 10))], dtype=np.float32)
#     weights /= weights.sum()  # 归一化权重，使其和为 1

#     # 确保所有片段长度一致（取最短长度）
#     min_length = min(segment.shape[0] for segment in mofea264_list)
#     aligned_segments = [segment[:min_length] for segment in mofea264_list]

#     # 加权求和
#     weighted_sum = np.zeros_like(aligned_segments[0], dtype=np.float32)
#     for i, segment in enumerate(aligned_segments):
#         weighted_sum += weights[i] * segment

#     return weighted_sum


# # def get_top10_mofea264_style(name: str, muidx: int,
# #                              retrieval_path="../InfiniteDanceData",
# #                              motion_base="../InfiniteDanceData",
# #                              meta_path="meta",
# #                              style_map_path="../InfiniteDanceData"):

# #     # 构造 JSON 文件路径
# #     json_path = os.path.join(retrieval_path, f"{name}.json")
# #     if not os.path.exists(json_path):
# #         raise FileNotFoundError(f"File {json_path} not found.")

# #     # 读取 JSON 数据
# #     with open(json_path, 'r') as f:
# #         data = json.load(f)

# #     # 解析前 10 个结果
# #     musiclist = data[muidx][0:10]
# #     names = [item['name'] for item in musiclist]
# #     result = []
# #     for full_name in names:
# #         if '@' in full_name:
# #             video_part, frame_part = full_name.rsplit('@', 1)
# #             if '_' in frame_part:
# #                 start_frame, end_frame = map(int, frame_part.split('_'))
# #                 result.append((video_part, start_frame, end_frame))

# #     # 加载 style map
# #     with open(style_map_path, 'r') as f:
# #         style_map = json.load(f)

# #     # 计算流派占比
# #     genres = []
# #     for video_name, _, _ in result:
# #         genre = style_map.get(video_name, 'Unknown')
# #         genres.append(genre)
    
# #     genre_counts = Counter(genres)
# #     total = len(genres)
# #     genre_proportions = {genre: count / total for genre, count in genre_counts.items()}

# #     # 找到概率最大的流派
# #     top_genre = max(genre_proportions, key=genre_proportions.get, default='Unknown')

# #     # 加载 Mean 和 Std
# #     mean_path = os.path.join(motion_base, meta_path, "Mean.npy")
# #     std_path = os.path.join(motion_base, meta_path, "Std.npy")
# #     mean = np.load(mean_path)
# #     std = np.load(std_path)

# #     # 读取每个 .npy 文件并截取 & 归一化
# #     mofea264_list = []
# #     for video_name, start, end in result:
# #         motion_path = os.path.join(motion_base, f"{video_name}.npy")
# #         if not os.path.exists(motion_path):
# #             print(f"Warning: {motion_path} not found, skipping...")
# #             continue

# #         motion = np.load(motion_path)  # shape: (T, 264)
# #         if end > motion.shape[0]:
# #             print(f"Warning: {video_name} range ({start}-{end}) exceeds motion length {motion.shape[0]}")
# #             continue

# #         segment = motion[start:end]
# #         normed_segment = (segment - mean) / std
# #         mofea264_list.append(normed_segment)

# #     # 加权求和
# #     if not mofea264_list:
# #         print("No valid motion segments found.")
# #         return None, genre_proportions, top_genre

# #     # 定义权重：序号越前（0 到 9），权重越大，例如 [10, 9, 8, ..., 1]
# #     weights = np.array([10 - i for i in range(min(len(mofea264_list), 10))], dtype=np.float32)
# #     weights /= weights.sum()  # 归一化权重，使其和为 1

# #     # 确保所有片段长度一致（取最短长度）
# #     min_length = min(segment.shape[0] for segment in mofea264_list)
# #     aligned_segments = [segment[:min_length] for segment in mofea264_list]

# #     # 加权求和
# #     weighted_sum = np.zeros_like(aligned_segments[0], dtype=np.float32)
# #     for i, segment in enumerate(aligned_segments):
# #         weighted_sum += weights[i] * segment

# #     return weighted_sum, genre_proportions, top_genre,mofea264_list[0]

# # # weighted_sum, genre_proportions,top_genre,mofea_top1=get_top10_mofea264_style("Ballet8",1)
# # # print(top_genre)
# # # print(mofea_top1.shape)



# def get_top_mofea_v1(name: str = None,
#                               motion: np.ndarray = None,
#                               idx: int = 0,
#                               eval_wrapper=None,
#                               motionembedding_dir="../InfiniteDanceData",
#                               retrieval_path="../InfiniteDanceData",
#                               motion_base="../InfiniteDanceData",
#                               style_map_path="../InfiniteDanceData",
#                               meta_path="meta",
#                               device='cuda:0'):
#     """
#     支持两种输入：
#     - name + idx：优先读取对应 JSON
#     - motion: 若 JSON 不存在，则根据 motion 计算 top10
#     """

#     if name !=None:
#         json_path = os.path.join(retrieval_path, f"{name}.json")
#         if os.path.exists(json_path):
#             with open(json_path, 'r') as f:
#                 data = json.load(f)

#             musiclist = data[idx][:10]
#             # musiclist = data[idx][-10:]
#             top10_results = [{'name': item['name']} for item in musiclist]
#         else:
#             # breakpoint()
#             from RetrievalNet.configs import get_config
#             from RetrievalNet.datasets import EvaluatorModelWrapper
#             eval_wrapper = EvaluatorModelWrapper(get_config(config_path), device)
#             if motion is None:
#                 # print("name",name)
#                 motion=np.load(f"../InfiniteDanceData")[idx*96:idx*96+384]
#                 if motion.shape[0] < 384:
#                     pad_length = 384 - motion.shape[0]
#                     motion = np.pad(motion, ((0, pad_length), (0, 0)), mode='wrap')

#             # breakpoint()
#             top10_results = get_top10_similar_mofea_features(motion, eval_wrapper, motionembedding_dir, device=device)
#     else:
#         from RetrievalNet.configs import get_config
#         from RetrievalNet.datasets import EvaluatorModelWrapper
#         eval_wrapper = EvaluatorModelWrapper(get_config(config_path), device)
#         if motion is None:
#             # print("name",name)
#             motion=np.load(f"../InfiniteDanceData")[idx*96:idx*96+384]
#             if motion.shape[0] < 384:
#                 pad_length = 384 - motion.shape[0]
#                 motion = np.pad(motion, ((0, pad_length), (0, 0)), mode='wrap')


#         top10_results = get_top10_similar_mofea_features(motion, eval_wrapper, motionembedding_dir, device=device)

#     # 提取 motion 片段
#     results = []
#     for item in top10_results:
#         full_name = item['name']
#         if '@' in full_name:
#             video_part, frame_part = full_name.rsplit('@', 1)
#             if '_' in frame_part:
#                 start_frame, end_frame = map(int, frame_part.split('_'))
#                 results.append((video_part, start_frame, end_frame))

#         # 加载 style map
#     with open(style_map_path, 'r') as f:
#         style_map = json.load(f)

#     genres = []
#     for video_name, _, _ in results:
#         genre = style_map.get(video_name, 'Unknown')
#         genres.append(genre)

#     genre_counts = Counter(genres)
#     total = len(genres)

#     # Calculate genre proportions
#     genre_proportions = {genre: count / total for genre, count in genre_counts.items()}

#     # Sort genres by proportion in descending order and take top 5
#     top_5_genres = sorted(genre_proportions.items(), key=lambda x: x[1], reverse=True)[:5]

#     # Assign weights (proportional to their proportions, normalized to sum to 1)
#     total_proportion = sum(proportion for _, proportion in top_5_genres)
#     weighted_genres = {genre: proportion / total_proportion for genre, proportion in top_5_genres}

#     # Define priority genres
#     GENRES = ["Ballet", "Popular", "Latin", "Modern", "Folk", "Classic"]

#     # Check if name contains any priority genre (case-insensitive)
#     name_lower = name.lower() if name else ""
#     priority_genre = None
#     for genre in GENRES:
#         if genre.lower() in name_lower:
#             priority_genre = genre
#             break

#     # Set top_genre: use priority genre if found, otherwise use highest-proportion genre
#     if priority_genre:
#         top_genre = priority_genre
#     else:
#         top_genre = top_5_genres[0][0] if top_5_genres else 'Popular'

#     # Load Mean & Std
#     mean = np.load(os.path.join(motion_base, meta_path, "Mean.npy"))
#     std = np.load(os.path.join(motion_base, meta_path, "Std.npy"))

#     mofea264_list = []
#     for video_name, start, end in results:
#         motion_path = os.path.join(motion_base, f"{video_name}.npy")
#         if not os.path.exists(motion_path):
#             print(f"Warning: {motion_path} not found, skipping...")
#             continue

#         motion_data = np.load(motion_path)
#         if end > motion_data.shape[0]:
#             print(f"Warning: {video_name} range ({start}-{end}) exceeds motion length {motion_data.shape[0]}")
#             continue

#         segment = motion_data[start:end]
#         normed_segment = (segment - mean) / std
#         mofea264_list.append(normed_segment)

#     if not mofea264_list:
#         print("No valid motion segments found.")
#         return None, genre_proportions, top_genre, top10_results

#     weights = np.array([10 - i for i in range(min(len(mofea264_list), 10))], dtype=np.float32)
#     weights /= weights.sum()

#     min_length = min(segment.shape[0] for segment in mofea264_list)
#     aligned_segments = [segment[:min_length] for segment in mofea264_list]

#     weighted_sum = np.zeros_like(aligned_segments[0], dtype=np.float32)
#     for i, segment in enumerate(aligned_segments):
#         weighted_sum += weights[i] * segment
#     top10_names_indices = []
#     for item in top10_results:
#         full_name = item['name']
#         if '@' in full_name:
#             video_part, frame_part = full_name.rsplit('@', 1)
#             if '_' in frame_part:
#                 start_idx, end_idx = map(int, frame_part.split('_'))
#                 top10_names_indices.append((video_part, start_idx, end_idx))
                
#     # top10_names_indices
#     # motiontoken_dir = "/data1/hzy/HumanMotion/All_mofea/Alldata/MotionTokens_512_vel_processed"
#     motiontoken_dir = "../InfiniteDanceData"
#     token_segments = []

#     for name, start_idx, _ in top10_names_indices:
#         token_file = os.path.join(motiontoken_dir, f"{name}.npy")
#         if not os.path.exists(token_file):
#             print(f"Warning: Token file {token_file} not found.")
#             continue

#         tokens = np.load(token_file)  # shape: (N,)

#         token_index = round(start_idx / 96) * 72
#         if token_index + 60 > len(tokens):
#             print(f"Warning: Token index {token_index} out of range for file {name}.npy (len={len(tokens)}).")
#             continue

#         token_segment = tokens[token_index:token_index + 60]
#         token_segments.append({
#             "name": name,
#             "start_idx": start_idx,
#             "token_index": token_index,
#             "tokens": token_segment.tolist(),  # or keep as np.array if needed
#         })

#     # 输出结果
#     # print("\nExtracted token segments (first 60 tokens each):")
#     # for i, item in enumerate(token_segments):
#     #     print(f"Top {i+1}: {item['name']} @ token_index={item['token_index']}, tokens[:5]={item['tokens'][:5]}")
#     # print("top_genre",top_genre)

#     return weighted_sum, genre_proportions, top_genre, top10_results,top10_names_indices,token_segments



# # def get_top_mofea_style(name: str = None,
# #                               motion: np.ndarray = None,
# #                               idx: int = 0,
# #                               eval_wrapper=None,
# #                               motionembedding_dir="../InfiniteDanceData",
# #                               retrieval_path="../InfiniteDanceData",
# #                               motion_base="../InfiniteDanceData",
# #                               style_map_path="../InfiniteDanceData",
# #                               meta_path="meta",
# #                               style="Popular",
# #                               device='cuda:0'):
# #     """
# #     支持两种输入：
# #     - name + idx：优先读取对应 JSON
# #     - motion: 若 JSON 不存在，则根据 motion 计算 top10
# #     """

# #     if name !=None:
# #         json_path = os.path.join(retrieval_path, f"{name}.json")
# #         if os.path.exists(json_path):

# #             # musiclist = data[idx][:10]

# #             # musiclist = data[idx][-10:]
# #             top10_results = get_items_by_style_and_idx(
# #                     retrieval_filepath=json_path,
# #                     target_style=style,
# #                     target_idx=idx
# #                 )
# #         else:
# #             # breakpoint()
# #             from RetrievalNet.configs import get_config
# #             from RetrievalNet.datasets import EvaluatorModelWrapper
# #             eval_wrapper = EvaluatorModelWrapper(get_config(config_path), device)
# #             if motion is None:
# #                 # print("name",name)
# #                 motion=np.load(f"../InfiniteDanceData")[idx*96:idx*96+384]
# #                 if motion.shape[0] < 384:
# #                     pad_length = 384 - motion.shape[0]
# #                     motion = np.pad(motion, ((0, pad_length), (0, 0)), mode='wrap')

# #             # breakpoint()
# #             top10_results = get_top10_similar_mofea_features(motion, eval_wrapper, motionembedding_dir, device=device)
# #     else:
# #         from RetrievalNet.configs import get_config
# #         from RetrievalNet.datasets import EvaluatorModelWrapper
# #         eval_wrapper = EvaluatorModelWrapper(get_config(config_path), device)
# #         if motion is None:
# #             # print("name",name)
# #             motion=np.load(f"../InfiniteDanceData")[idx*96:idx*96+384]
# #             if motion.shape[0] < 384:
# #                 pad_length = 384 - motion.shape[0]
# #                 motion = np.pad(motion, ((0, pad_length), (0, 0)), mode='wrap')


# #         top10_results = get_top10_similar_mofea_features(motion, eval_wrapper, motionembedding_dir, device=device)

# #     # 提取 motion 片段
# #     results = []
# #     for item in top10_results:
# #         full_name = item['name']
# #         if '@' in full_name:
# #             video_part, frame_part = full_name.rsplit('@', 1)
# #             if '_' in frame_part:
# #                 start_frame, end_frame = map(int, frame_part.split('_'))
# #                 results.append((video_part, start_frame, end_frame))

# #         # 加载 style map
# #     with open(style_map_path, 'r') as f:
# #         style_map = json.load(f)

# #     genres = []
# #     for video_name, _, _ in results:
# #         genre = style_map.get(video_name, 'Unknown')
# #         genres.append(genre)

# #     genre_counts = Counter(genres)
# #     total = len(genres)

# #     # Calculate genre proportions
# #     genre_proportions = {genre: count / total for genre, count in genre_counts.items()}

# #     # Sort genres by proportion in descending order and take top 5
# #     top_5_genres = sorted(genre_proportions.items(), key=lambda x: x[1], reverse=True)[:5]

# #     # Assign weights (proportional to their proportions, normalized to sum to 1)
# #     total_proportion = sum(proportion for _, proportion in top_5_genres)
# #     weighted_genres = {genre: proportion / total_proportion for genre, proportion in top_5_genres}

# #     # Define priority genres
# #     GENRES = ["Ballet", "Popular", "Latin", "Modern", "Folk", "Classic"]

# #     # Check if name contains any priority genre (case-insensitive)
# #     name_lower = name.lower() if name else ""
# #     priority_genre = None
# #     for genre in GENRES:
# #         if genre.lower() in name_lower:
# #             priority_genre = genre
# #             break

# #     # Set top_genre: use priority genre if found, otherwise use highest-proportion genre
# #     if priority_genre:
# #         top_genre = priority_genre
# #     else:
# #         top_genre = top_5_genres[0][0] if top_5_genres else 'Popular'

# #     # Load Mean & Std
# #     mean = np.load(os.path.join(motion_base, meta_path, "Mean.npy"))
# #     std = np.load(os.path.join(motion_base, meta_path, "Std.npy"))

# #     mofea264_list = []
# #     for video_name, start, end in results:
# #         motion_path = os.path.join(motion_base, f"{video_name}.npy")
# #         if not os.path.exists(motion_path):
# #             print(f"Warning: {motion_path} not found, skipping...")
# #             continue

# #         motion_data = np.load(motion_path)
# #         if end > motion_data.shape[0]:
# #             print(f"Warning: {video_name} range ({start}-{end}) exceeds motion length {motion_data.shape[0]}")
# #             continue

# #         segment = motion_data[start:end]
# #         normed_segment = (segment - mean) / std
# #         mofea264_list.append(normed_segment)

# #     if not mofea264_list:
# #         print("No valid motion segments found.")
# #         return None, genre_proportions, top_genre, top10_results

# #     weights = np.array([10 - i for i in range(min(len(mofea264_list), 10))], dtype=np.float32)
# #     weights /= weights.sum()

# #     min_length = min(segment.shape[0] for segment in mofea264_list)
# #     aligned_segments = [segment[:min_length] for segment in mofea264_list]

# #     weighted_sum = np.zeros_like(aligned_segments[0], dtype=np.float32)
# #     for i, segment in enumerate(aligned_segments):
# #         weighted_sum += weights[i] * segment
# #     top10_names_indices = []
# #     for item in top10_results:
# #         full_name = item['name']
# #         if '@' in full_name:
# #             video_part, frame_part = full_name.rsplit('@', 1)
# #             if '_' in frame_part:
# #                 start_idx, end_idx = map(int, frame_part.split('_'))
# #                 top10_names_indices.append((video_part, start_idx, end_idx))
                
# #     # top10_names_indices
# #     # motiontoken_dir = "/data1/hzy/HumanMotion/All_mofea/Alldata/MotionTokens_512_vel_processed"
# #     motiontoken_dir = "../InfiniteDanceData"
# #     token_segments = []

# #     for name, start_idx, _ in top10_names_indices:
# #         token_file = os.path.join(motiontoken_dir, f"{name}.npy")
# #         if not os.path.exists(token_file):
# #             print(f"Warning: Token file {token_file} not found.")
# #             continue

# #         tokens = np.load(token_file)  # shape: (N,)

# #         token_index = round(start_idx / 96) * 72
# #         if token_index + 60 > len(tokens):
# #             print(f"Warning: Token index {token_index} out of range for file {name}.npy (len={len(tokens)}).")
# #             continue

# #         token_segment = tokens[token_index:token_index + 60]
# #         token_segments.append({
# #             "name": name,
# #             "start_idx": start_idx,
# #             "token_index": token_index,
# #             "tokens": token_segment.tolist(),  # or keep as np.array if needed
# #         })

# #     # 输出结果
# #     # print("\nExtracted token segments (first 60 tokens each):")
# #     # for i, item in enumerate(token_segments):
# #     #     print(f"Top {i+1}: {item['name']} @ token_index={item['token_index']}, tokens[:5]={item['tokens'][:5]}")
# #     # print("top_genre",top_genre)

# #     return weighted_sum, genre_proportions, top_genre, top10_results,top10_names_indices,token_segments




import json
import os
import sys
import numpy as np
import torch
from collections import Counter
from tqdm import tqdm

# ==========================================
# Global Configurations (From Snippet A)
# ==========================================
motionembedding_dir = '/data2/hzy/InfiniteDance/InfiniteDanceData/dance/motionembeding'
config_path = '/data2/hzy/InfiniteDance/All_LargeDanceAR/RetrievalNet/checkpoints/AInterClip_Audio55Motion264/0512/train/bc256_s100l384_drop0.2_lr1e-4/InterCLIP.yaml'
device = 'cuda:4'

# ==========================================
# Excluded-name helpers (e.g. exclude AIST++ / FineDance from retrieval)
# ==========================================

DEFAULT_AISTPP_FINEDANCE_PARTITION_FILES = [
    "/data2/hzy/InfiniteDance_opensource/InfiniteDanceData/partition/aistpp_train.txt",
    "/data2/hzy/InfiniteDance_opensource/InfiniteDanceData/partition/aistpp_test.txt",
    "/data2/hzy/InfiniteDance_opensource/InfiniteDanceData/partition/aistpp_eval.txt",
    "/data2/hzy/InfiniteDance_opensource/InfiniteDanceData/partition/finedance_train.txt",
    "/data2/hzy/InfiniteDance_opensource/InfiniteDanceData/partition/finedance_eval.txt",
]


def load_excluded_names_from_partitions(partition_files):
    """Read partition .txt files and return a set of video_part names to exclude."""
    excluded = set()
    for fp in partition_files or []:
        if not os.path.exists(fp):
            print(f"Warning: partition file not found: {fp}")
            continue
        with open(fp) as f:
            for line in f:
                name = line.strip()
                if name:
                    excluded.add(name)
    return excluded


def _video_part(full_name: str) -> str:
    """Extract the video portion of a retrieval entry name (the part before '@')."""
    return full_name.rsplit('@', 1)[0] if '@' in full_name else full_name


def _filter_excluded(items, exclude_names):
    """Filter a list of dicts/strings by their video_part not in exclude_names."""
    if not exclude_names:
        return items
    out = []
    for it in items:
        nm = it['name'] if isinstance(it, dict) else it
        if _video_part(nm) not in exclude_names:
            out.append(it)
    return out


# ==========================================
# Helper Functions (Math & Sorting)
# ==========================================

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

def get_top10_similar_mofea_features(audio_feat, eval_wrapper, motionembedding_dir, device='cuda:0', top_k=10, exclude_names=None):
    assert audio_feat.shape == (384, 55), "Input audio feature shape must be (384, 55)"
    audio_feat_tensor = torch.from_numpy(audio_feat).unsqueeze(0).to(device).to(torch.float32)

    audio_batch = ("query_sample", audio_feat_tensor, torch.zeros([1, 55], device=device))
    # breakpoint()
    with torch.no_grad():
        audio_embedding = eval_wrapper.get_co_embeddings(audio_batch, "audio").cpu().numpy()

    all_motion_embeddings = []
    seqnames = []
    # Note: Using the passed motionembedding_dir
    for file in os.listdir(motionembedding_dir):
        if file.endswith('.npy'):
            motion_emb = np.load(os.path.join(motionembedding_dir, file))
            all_motion_embeddings.append(motion_emb)
            seqnames.append(file.replace('.npy', ''))

    all_motion_embeddings = np.stack(all_motion_embeddings, axis=0)

    dist_mat = euclidean_distance_matrix(audio_embedding, all_motion_embeddings)
    sorted_idx = np.argsort(dist_mat[0])

    top_items = []
    for idx in sorted_idx:
        nm = seqnames[idx]
        if exclude_names and _video_part(nm) in exclude_names:
            continue
        top_items.append({
            'name': nm,
            'dist': dist_mat[0][idx],
        })
        if len(top_items) >= top_k:
            break

    sorted_top = sort_by_dist(top_items)
    return sorted_top

def _match_style_from_name(name: str, target_style: str) -> bool:
    if not target_style:
        return True
    name_lower = name.lower()
    style_lower = target_style.lower()
    if name_lower.startswith(style_lower):
        return True
    if '-' in name_lower and name_lower.split('-', 1)[0] == style_lower:
        return True
    return False

def get_items_by_style_and_idx(retrieval_filepath: str,
                                    target_style: str,
                                    target_idx: int,
                                    top_k: int = 10,
                                    exclude_names=None) -> list:
    """
    Unified version: Returns both the list of names and the list of dicts.
    Snippet A used this signature. Snippet B callers will need to unpack just the first return value.
    """
    try:
        with open(retrieval_filepath, 'r') as f:
            # data is { "idx_0": { "Style": [...] }, ... }
            data = json.load(f)
                    
        # Dict format: { "idx_0": { "Style": [...] }, ... }
        if isinstance(data, dict):
            style_dict = data.get(f"idx_{target_idx}", {})
            items_list = style_dict.get(target_style, [])
        # List format: [ [ {"name": ...}, ... ], ... ]
        elif isinstance(data, list):
            if 0 <= target_idx < len(data):
                items_list = data[target_idx]
            else:
                items_list = []
            if target_style:
                filtered = [item for item in items_list if _match_style_from_name(item.get("name", ""), target_style)]
                if filtered:
                    items_list = filtered
        else:
            items_list = []

        if exclude_names:
            items_list = [item for item in items_list if _video_part(item.get("name", "")) not in exclude_names]

        items_list_top10 = [item["name"] for item in items_list][:top_k]
        items_list_top10_2 = [{'name': item['name']} for item in items_list][:top_k]

        return items_list_top10, items_list_top10_2
        
    except FileNotFoundError:
        print(f"Error: Retrieval file not found: {retrieval_filepath}")
        return [], []
    except json.JSONDecodeError:
        print(f"Error: Cannot parse retrieval file: {retrieval_filepath}")
        return [], []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return [], []

# ==========================================
# Main Functions from Snippet A (Absolute Paths & New Features)
# ==========================================

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
                              device='cuda:0',
                              top_k=10,
                              exclude_names=None):
    """
    Snippet A version: Supports absolute paths.
    `top_k` controls the number of retrieved neighbors (default 10).
    `exclude_names`: optional set/list of video_part names to exclude from results
        (e.g. AIST++ / FineDance names). When the JSON cache has fewer surviving
        entries than top_k, returns however many remain.
    """

    if name != None:
        json_path = os.path.join(retrieval_path, f"{name}.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            entries = data[idx]
            if exclude_names:
                entries = [e for e in entries if _video_part(e.get('name', '')) not in exclude_names]
            musiclist = entries[:top_k] if len(entries) > 0 else []
            top10_results = [{'name': item['name']} for item in musiclist]
        else:
            from RetrievalNet.configs import get_config
            from RetrievalNet.datasets import EvaluatorModelWrapper
            eval_wrapper = EvaluatorModelWrapper(get_config(config_path), device)
            if motion is None:
                motion = np.load(f"/data2/hzy/InfiniteDance/InfiniteDanceData/music/musicfeature_55_allmusic_pure/{name}.npy")[idx*96:idx*96+384]
                if motion.shape[0] < 384:
                    pad_length = 384 - motion.shape[0]
                    motion = np.pad(motion, ((0, pad_length), (0, 0)), mode='wrap')

            top10_results = get_top10_similar_mofea_features(motion, eval_wrapper, motionembedding_dir, device=device, top_k=top_k, exclude_names=exclude_names)
    else:
        from RetrievalNet.configs import get_config
        from RetrievalNet.datasets import EvaluatorModelWrapper
        eval_wrapper = EvaluatorModelWrapper(get_config(config_path), device)
        if motion is None:
            motion = np.load(f"/data2/hzy/InfiniteDance/InfiniteDanceData/music/musicfeature_55_allmusic_pure/{name}.npy")[idx*96:idx*96+384]
            if motion.shape[0] < 384:
                pad_length = 384 - motion.shape[0]
                motion = np.pad(motion, ((0, pad_length), (0, 0)), mode='wrap')

        top10_results = get_top10_similar_mofea_features(motion, eval_wrapper, motionembedding_dir, device=device, top_k=top_k, exclude_names=exclude_names)

    # Extract motion segments
    results = []
    for item in top10_results:
        full_name = item['name']
        if '@' in full_name:
            video_part, frame_part = full_name.rsplit('@', 1)
            if '_' in frame_part:
                start_frame, end_frame = map(int, frame_part.split('_'))
                results.append((video_part, start_frame, end_frame))

    with open(style_map_path, 'r') as f:
        style_map = json.load(f)

    genres = []
    for video_name, _, _ in results:
        genre = style_map.get(video_name, 'Unknown')
        genres.append(genre)

    genre_counts = Counter(genres)
    total = len(genres)
    genre_proportions = {genre: count / total for genre, count in genre_counts.items()}
    top_5_genres = sorted(genre_proportions.items(), key=lambda x: x[1], reverse=True)[:5]

    GENRES = ["Ballet", "Popular", "Latin", "Modern", "Folk", "Classic"]
    name_lower = name.lower() if name else ""
    priority_genre = None
    for genre in GENRES:
        if genre.lower() in name_lower:
            priority_genre = genre
            break

    if priority_genre:
        top_genre = priority_genre
    else:
        top_genre = top_5_genres[0][0] if top_5_genres else 'Popular'

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

    n_used = min(len(mofea264_list), top_k)
    weights = np.array([n_used - i for i in range(n_used)], dtype=np.float32)
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

    motiontoken_dir = "/data2/hzy/InfiniteDance/InfiniteDanceData/dance/Infinite_MotionTokens_512_vel_processed"
    token_segments = []

    for name, start_idx, _ in top10_names_indices:
        token_file = os.path.join(motiontoken_dir, f"{name}.npy")
        if not os.path.exists(token_file):
            print(f"Warning: Token file {token_file} not found.")
            continue

        tokens = np.load(token_file)

        token_index = round(start_idx / 96) * 72
        if token_index + 60 > len(tokens):
            print(f"Warning: Token index {token_index} out of range for file {name}.npy (len={len(tokens)}).")
            continue

        token_segment = tokens[token_index:token_index + 60]
        token_segments.append({
            "name": name,
            "start_idx": start_idx,
            "token_index": token_index,
            "tokens": token_segment.tolist(),
        })

    return weighted_sum, genre_proportions, top_genre, top10_results, top10_names_indices, token_segments

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
                              infertype="infinitedance",
                              top_k=10,
                              exclude_names=None):
    """
    Snippet A version: Includes logic for infertype='infinitedanceplus' (v6_2 windowed tokens).
    `top_k` controls the number of retrieved neighbors (default 10).
    `exclude_names`: optional set/list of video_part names to exclude (e.g. AIST++ / FineDance).
    """
    if name==None and motion is None:
        raise ValueError("Must provide name or motion.")
    if name !=None:
        json_path = os.path.join(retrieval_path, f"{name}.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            # Use unified get_items_by_style_and_idx, taking the second return value (dict list) as top10_results
            _, top10_results = get_items_by_style_and_idx(
                    retrieval_filepath=json_path,
                    target_style=style,
                    target_idx=idx,
                    top_k=top_k,
                    exclude_names=exclude_names
                )
        else:
            from RetrievalNet.configs import get_config
            from RetrievalNet.datasets import EvaluatorModelWrapper
            eval_wrapper = EvaluatorModelWrapper(get_config(config_path), device)
            if motion is None:
                motion=np.load(f"/data2/hzy/InfiniteDance/InfiniteDanceData/music/musicfeature_55_allmusic_pure/{name}.npy")[idx*96:idx*96+384]
                if motion.shape[0] < 384:
                    pad_length = 384 - motion.shape[0]
                    motion = np.pad(motion, ((0, pad_length), (0, 0)), mode='wrap')

            top10_results = get_top10_similar_mofea_features(motion, eval_wrapper, motionembedding_dir, device=device, top_k=top_k, exclude_names=exclude_names)
    else:
        from RetrievalNet.configs import get_config
        from RetrievalNet.datasets import EvaluatorModelWrapper
        eval_wrapper = EvaluatorModelWrapper(get_config(config_path), device)
        if motion is None:
            motion=np.load(f"/data2/hzy/InfiniteDance/InfiniteDanceData/music/musicfeature_55_allmusic_pure/{name}.npy")[idx*96:idx*96+384]
            if motion.shape[0] < 384:
                pad_length = 384 - motion.shape[0]
                motion = np.pad(motion, ((0, pad_length), (0, 0)), mode='wrap')

        top10_results = get_top10_similar_mofea_features(motion, eval_wrapper, motionembedding_dir, device=device, top_k=top_k, exclude_names=exclude_names)

    # Extract motion segments
    results = []
    for item in top10_results:
        full_name = item['name']
        if '@' in full_name:
            video_part, frame_part = full_name.rsplit('@', 1)
            if '_' in frame_part:
                start_frame, end_frame = map(int, frame_part.split('_'))
                results.append((video_part, start_frame, end_frame))

    with open(style_map_path, 'r') as f:
        style_map = json.load(f)

    genres = []
    for video_name, _, _ in results:
        genre = style_map.get(video_name, 'Unknown')
        genres.append(genre)

    genre_counts = Counter(genres)
    total = len(genres)

    genre_proportions = {genre: count / total for genre, count in genre_counts.items()}
    top_5_genres = sorted(genre_proportions.items(), key=lambda x: x[1], reverse=True)[:5]
    total_proportion = sum(proportion for _, proportion in top_5_genres)
    weighted_genres = {genre: proportion / total_proportion for genre, proportion in top_5_genres}

    GENRES = ["Ballet", "Popular", "Latin", "Modern", "Folk", "Classic"]
    name_lower = name.lower() if name else ""
    priority_genre = None
    for genre in GENRES:
        if genre.lower() in name_lower:
            priority_genre = genre
            break

    if priority_genre:
        top_genre = priority_genre
    else:
        top_genre = top_5_genres[0][0] if top_5_genres else 'Popular'

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

    n_used = min(len(mofea264_list), top_k)
    weights = np.array([n_used - i for i in range(n_used)], dtype=np.float32)
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

    motiontoken_dir = "/data2/hzy/InfiniteDance/InfiniteDanceData/dance/Infinite_MotionTokens_512_vel_processed"
    v6_2_token_dir = "/data2/hzy/InfiniteDance/InfiniteDanceData/ALL_SD_ID_DATA/motion_264_30fps_tokens_1layer_windowed"
    token_segments = []

    use_v6_2 = (infertype == "infinitedanceplus")

    for name, start_idx, end_idx in top10_names_indices:
        v6_2_token_file = None
        if use_v6_2 and os.path.exists(v6_2_token_dir):
            token_start_idx = start_idx // 4
            token_end_idx = end_idx // 4
            
            exact_match_file = os.path.join(v6_2_token_dir, f"{name}_{token_start_idx}-{token_end_idx}.npy")
            if os.path.exists(exact_match_file):
                v6_2_token_file = exact_match_file
            else:
                import glob
                pattern = os.path.join(v6_2_token_dir, f"{name}_*.npy")
                matching_files = glob.glob(pattern)
                
                if matching_files:
                    best_file = None
                    best_distance = float('inf')
                    
                    for file_path in matching_files:
                        filename = os.path.basename(file_path)
                        try:
                            parts = filename.replace('.npy', '').split('_')
                            if len(parts) >= 2:
                                window_range = parts[-1]
                                window_start, window_end = map(int, window_range.split('-'))
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
            tokens = np.load(v6_2_token_file)
            if len(tokens) >= windows_length:
                token_segment = tokens[:windows_length]
            else:
                token_segment = tokens
            token_segments.append({
                "name": name,
                "start_idx": start_idx,
                "token_index": 0,
                "tokens": token_segment.tolist(),
            })
        else:
            token_file = os.path.join(motiontoken_dir, f"{name}.npy")
            if not os.path.exists(token_file):
                print(f"Warning: Token file {token_file} not found.")
                continue

            tokens = np.load(token_file)

            token_index = round(start_idx / 96) * 72
            if token_index + windows_length > len(tokens):
                print(f"Warning: Token index {token_index} out of range for file {name}.npy (len={len(tokens)}).")
                continue

            token_segment = tokens[token_index:token_index + windows_length]
            token_segments.append({
                "name": name,
                "start_idx": start_idx,
                "token_index": token_index,
                "tokens": token_segment.tolist(),
            })

    return weighted_sum, genre_proportions, top_genre, top10_results, top10_names_indices, token_segments

# ==========================================
# Exclude-AISTPP-FineDance convenience wrappers
# ==========================================

# Cached excluded-names set (lazy build).
_AISTPP_FINEDANCE_EXCLUDE_CACHE = None


def get_aistpp_finedance_excluded_names(partition_files=None, force_reload=False):
    """Return the cached set of AIST++/FineDance video names to exclude from retrieval."""
    global _AISTPP_FINEDANCE_EXCLUDE_CACHE
    if _AISTPP_FINEDANCE_EXCLUDE_CACHE is None or force_reload:
        files = partition_files if partition_files is not None else DEFAULT_AISTPP_FINEDANCE_PARTITION_FILES
        _AISTPP_FINEDANCE_EXCLUDE_CACHE = load_excluded_names_from_partitions(files)
        print(f"[exclude] loaded {len(_AISTPP_FINEDANCE_EXCLUDE_CACHE)} AIST++/FineDance names to exclude")
    return _AISTPP_FINEDANCE_EXCLUDE_CACHE


def get_top_mofea_exclude_aistpp_finedance(*args, partition_files=None, **kwargs):
    """Like get_top_mofea, but skips any retrieval entry whose video belongs to AIST++ or FineDance."""
    excl = get_aistpp_finedance_excluded_names(partition_files)
    if 'exclude_names' in kwargs and kwargs['exclude_names']:
        kwargs['exclude_names'] = set(kwargs['exclude_names']) | excl
    else:
        kwargs['exclude_names'] = excl
    return get_top_mofea(*args, **kwargs)


def get_top_mofea_specific_style_exclude_aistpp_finedance(*args, partition_files=None, **kwargs):
    """Like get_top_mofea_specific_style, but excludes AIST++/FineDance entries from retrieval."""
    excl = get_aistpp_finedance_excluded_names(partition_files)
    if 'exclude_names' in kwargs and kwargs['exclude_names']:
        kwargs['exclude_names'] = set(kwargs['exclude_names']) | excl
    else:
        kwargs['exclude_names'] = excl
    return get_top_mofea_specific_style(*args, **kwargs)


def get_top_mofea_specific_style_only_aistpp_finedance(*args, partition_files=None, **kwargs):
    """Inverse of *_exclude_aistpp_finedance: keep ONLY AIST++/FineDance entries.

    Implementation note: we read the retrieval JSON ourselves, prefilter
    candidates to those whose video_part is in the AIST++/FineDance set, and
    pass the prefiltered list back via a temporary file. To avoid touching the
    underlying loader, we use a thread-local override of the include set.
    """
    inc = get_aistpp_finedance_excluded_names(partition_files)
    import json as _json
    name = kwargs.get('name') or (args[0] if args else None)
    retrieval_path = kwargs.get('retrieval_path',
        "/data2/hzy/InfiniteDance/InfiniteDanceData/dance/retrieval_s192_l384_style")
    if name is None:
        return get_top_mofea_specific_style(*args, **kwargs)
    rfp = os.path.join(retrieval_path, f"{name}.json")
    drop = set()
    try:
        with open(rfp, 'r') as f:
            data = _json.load(f)
        if isinstance(data, dict):
            for _idx_key, styles in data.items():
                if not isinstance(styles, dict):
                    continue
                for _style, cands in styles.items():
                    if not isinstance(cands, list):
                        continue
                    for c in cands:
                        nm = c['name'] if isinstance(c, dict) else c
                        if _video_part(nm) not in inc:
                            drop.add(_video_part(nm))
        elif isinstance(data, list):
            for cands in data:
                if not isinstance(cands, list):
                    continue
                for c in cands:
                    nm = c['name'] if isinstance(c, dict) else c
                    if _video_part(nm) not in inc:
                        drop.add(_video_part(nm))
    except Exception:
        pass

    if 'exclude_names' in kwargs and kwargs['exclude_names']:
        kwargs['exclude_names'] = set(kwargs['exclude_names']) | drop
    else:
        kwargs['exclude_names'] = drop
    out = get_top_mofea_specific_style(*args, **kwargs)
    if isinstance(out, tuple) and len(out) == 4:
        ws, genre_props, top_genre, top10 = out
        if ws is None:
            ws = np.zeros((384, 264), dtype=np.float32)
        return ws, genre_props, top_genre, top10, [], []
    return out


# ==========================================
# Top-K convenience wrappers (top1 / top3 / top20)
# ==========================================

def get_top1_mofea(*args, **kwargs):
    kwargs['top_k'] = 1
    return get_top_mofea(*args, **kwargs)

def get_top3_mofea(*args, **kwargs):
    kwargs['top_k'] = 3
    return get_top_mofea(*args, **kwargs)

def get_top20_mofea(*args, **kwargs):
    kwargs['top_k'] = 20
    return get_top_mofea(*args, **kwargs)

def get_top1_mofea_specific_style(*args, **kwargs):
    kwargs['top_k'] = 1
    return get_top_mofea_specific_style(*args, **kwargs)

def get_top3_mofea_specific_style(*args, **kwargs):
    kwargs['top_k'] = 3
    return get_top_mofea_specific_style(*args, **kwargs)

def get_top20_mofea_specific_style(*args, **kwargs):
    kwargs['top_k'] = 20
    return get_top_mofea_specific_style(*args, **kwargs)

# ==========================================
# Legacy / Relative Path Functions from Snippet B
# ==========================================

def get_top10_mofea264_1(name: str, muidx: int,
                       retrieval_path="../InfiniteDanceData",
                       motion_base="../InfiniteDanceData",
                       meta_path="meta"):
    '''
    old version, use 264dim motion features
    '''

    json_path = os.path.join(retrieval_path, f"{name}.json")

    if not os.path.exists(json_path):
        motion = np.load(f"../InfiniteDanceData")
        num_frames, dim = motion.shape

        if num_frames >= 384:
            motion_fixed = motion[:384]
        else:
            pad_len = 384 - num_frames
            padding = np.zeros((pad_len, dim), dtype=motion.dtype)
            motion_fixed = np.concatenate([motion, padding], axis=0)

        return motion_fixed

    with open(json_path, 'r') as f:
        data = json.load(f)

    musiclist = data[muidx][0:10]
    names = [item['name'] for item in musiclist]
    result = []
    for full_name in names:
        if '@' in full_name:
            video_part, frame_part = full_name.rsplit('@', 1)
            if '_' in frame_part:
                start_frame, end_frame = map(int, frame_part.split('_'))
                result.append((video_part, start_frame, end_frame))

    mean_path = os.path.join(motion_base, meta_path, "Mean.npy")
    std_path = os.path.join(motion_base, meta_path, "Std.npy")
    mean = np.load(mean_path)
    std = np.load(std_path)

    mofea264_list = []
    for video_name, start, end in result:
        motion_path = os.path.join(motion_base, f"{video_name}.npy")
        if not os.path.exists(motion_path):
            print(f"Warning: {motion_path} not found, skipping...")
            continue

        motion = np.load(motion_path)
        if end > motion.shape[0]:
            print(f"Warning: {video_name} range ({start}-{end}) exceeds motion length {motion.shape[0]}")
            continue

        segment = motion[start:end]
        normed_segment = (segment - mean) / std
        mofea264_list.append(normed_segment)

    if not mofea264_list:
        print("No valid motion segments found. Returning zeros.")
        return np.zeros((384, mean.shape[0]), dtype=np.float32)

    weights = np.array([10 - i for i in range(min(len(mofea264_list), 10))], dtype=np.float32)
    weights /= weights.sum()

    min_length = min(segment.shape[0] for segment in mofea264_list)
    aligned_segments = [segment[:min_length] for segment in mofea264_list]

    weighted_sum = np.zeros_like(aligned_segments[0], dtype=np.float32)
    for i, segment in enumerate(aligned_segments):
        weighted_sum += weights[i] * segment

    return weighted_sum

def get_top10_mofea264_specific_style(name: str, muidx: int,
                       retrieval_path="/data2/hzy/InfiniteDance/InfiniteDanceData/dance/retrieval_s192_l384",
                       motion_base="/data2/hzy/InfiniteDance/InfiniteDanceData/dance/alldata_new_joint_vecs264",
                       style="Popular",
                       meta_path="/data2/hzy/InfiniteDance/InfiniteDanceData/dance/meta"):
    '''
    old version, use 264dim motion features (with style)
    '''

    json_path = os.path.join(retrieval_path, f"{name}.json")

    if os.path.isabs(meta_path):
        mean_path = os.path.join(meta_path, "Mean.npy")
        std_path = os.path.join(meta_path, "Std.npy")
    else:
        mean_path = os.path.join(motion_base, meta_path, "Mean.npy")
        std_path = os.path.join(motion_base, meta_path, "Std.npy")
    mean = np.load(mean_path)
    std = np.load(std_path)

    if not os.path.exists(json_path):
        fallback_motion_path = os.path.join(motion_base, f"{name}.npy")
        if os.path.exists(fallback_motion_path):
            motion = np.load(fallback_motion_path)
            start = muidx * 96
            end = start + 384
            segment = motion[start:end]
            if segment.shape[0] < 384:
                pad_len = 384 - segment.shape[0]
                padding = np.zeros((pad_len, segment.shape[1]), dtype=segment.dtype)
                segment = np.concatenate([segment, padding], axis=0)
            return (segment - mean) / std
        print(f"[WARN] 检索文件缺失，且未找到 motion：{fallback_motion_path}，返回全零 (384, {mean.shape[0]})")
        return np.zeros((384, mean.shape[0]), dtype=np.float32)

    if os.path.exists(json_path):
        # NOTE: Updated to unpack tuple from unified get_items_by_style_and_idx
        musiclist, _ = get_items_by_style_and_idx(
                    retrieval_filepath=json_path,
                    target_style=style,
                    target_idx=muidx
                )
    
    result = []
    for full_name in musiclist:
        if '@' in full_name:
            video_part, frame_part = full_name.rsplit('@', 1)
            if '_' in frame_part:
                start_frame, end_frame = map(int, frame_part.split('_'))
                result.append((video_part, start_frame, end_frame))

    mofea264_list = []
    for video_name, start, end in result:
        motion_path = os.path.join(motion_base, f"{video_name}.npy")
        if not os.path.exists(motion_path):
            print(f"Warning: {motion_path} not found, skipping...")
            continue

        motion = np.load(motion_path)
        if end > motion.shape[0]:
            print(f"Warning: {video_name} range ({start}-{end}) exceeds motion length {motion.shape[0]}")
            continue

        segment = motion[start:end]
        normed_segment = (segment - mean) / std
        mofea264_list.append(normed_segment)

    if not mofea264_list:
        print("No valid motion segments found.")
        return None

    weights = np.array([10 - i for i in range(min(len(mofea264_list), 10))], dtype=np.float32)
    weights /= weights.sum()

    min_length = min(segment.shape[0] for segment in mofea264_list)
    aligned_segments = [segment[:min_length] for segment in mofea264_list]

    weighted_sum = np.zeros_like(aligned_segments[0], dtype=np.float32)
    for i, segment in enumerate(aligned_segments):
        weighted_sum += weights[i] * segment

    return weighted_sum

def get_top_mofea_v1(name: str = None,
                              motion: np.ndarray = None,
                              idx: int = 0,
                              eval_wrapper=None,
                              motionembedding_dir="../InfiniteDanceData",
                              retrieval_path="../InfiniteDanceData",
                              motion_base="../InfiniteDanceData",
                              style_map_path="../InfiniteDanceData",
                              meta_path="meta",
                              device='cuda:0'):
    """
    Snippet B version: Uses relative paths (../InfiniteDanceData).
    """

    if name !=None:
        json_path = os.path.join(retrieval_path, f"{name}.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)

            musiclist = data[idx][:10]
            top10_results = [{'name': item['name']} for item in musiclist]
        else:
            from RetrievalNet.configs import get_config
            from RetrievalNet.datasets import EvaluatorModelWrapper
            eval_wrapper = EvaluatorModelWrapper(get_config(config_path), device)
            if motion is None:
                motion=np.load(f"../InfiniteDanceData")[idx*96:idx*96+384]
                if motion.shape[0] < 384:
                    pad_length = 384 - motion.shape[0]
                    motion = np.pad(motion, ((0, pad_length), (0, 0)), mode='wrap')

            top10_results = get_top10_similar_mofea_features(motion, eval_wrapper, motionembedding_dir, device=device)
    else:
        from RetrievalNet.configs import get_config
        from RetrievalNet.datasets import EvaluatorModelWrapper
        eval_wrapper = EvaluatorModelWrapper(get_config(config_path), device)
        if motion is None:
            motion=np.load(f"../InfiniteDanceData")[idx*96:idx*96+384]
            if motion.shape[0] < 384:
                pad_length = 384 - motion.shape[0]
                motion = np.pad(motion, ((0, pad_length), (0, 0)), mode='wrap')

        top10_results = get_top10_similar_mofea_features(motion, eval_wrapper, motionembedding_dir, device=device)

    results = []
    for item in top10_results:
        full_name = item['name']
        if '@' in full_name:
            video_part, frame_part = full_name.rsplit('@', 1)
            if '_' in frame_part:
                start_frame, end_frame = map(int, frame_part.split('_'))
                results.append((video_part, start_frame, end_frame))

    with open(style_map_path, 'r') as f:
        style_map = json.load(f)

    genres = []
    for video_name, _, _ in results:
        genre = style_map.get(video_name, 'Unknown')
        genres.append(genre)

    genre_counts = Counter(genres)
    total = len(genres)

    genre_proportions = {genre: count / total for genre, count in genre_counts.items()}
    top_5_genres = sorted(genre_proportions.items(), key=lambda x: x[1], reverse=True)[:5]
    total_proportion = sum(proportion for _, proportion in top_5_genres)
    weighted_genres = {genre: proportion / total_proportion for genre, proportion in top_5_genres}

    GENRES = ["Ballet", "Popular", "Latin", "Modern", "Folk", "Classic"]
    name_lower = name.lower() if name else ""
    priority_genre = None
    for genre in GENRES:
        if genre.lower() in name_lower:
            priority_genre = genre
            break

    if priority_genre:
        top_genre = priority_genre
    else:
        top_genre = top_5_genres[0][0] if top_5_genres else 'Popular'

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
                
    motiontoken_dir = "../InfiniteDanceData"
    token_segments = []

    for name, start_idx, _ in top10_names_indices:
        token_file = os.path.join(motiontoken_dir, f"{name}.npy")
        if not os.path.exists(token_file):
            print(f"Warning: Token file {token_file} not found.")
            continue

        tokens = np.load(token_file)

        token_index = round(start_idx / 96) * 72
        if token_index + 60 > len(tokens):
            print(f"Warning: Token index {token_index} out of range for file {name}.npy (len={len(tokens)}).")
            continue

        token_segment = tokens[token_index:token_index + 60]
        token_segments.append({
            "name": name,
            "start_idx": start_idx,
            "token_index": token_index,
            "tokens": token_segment.tolist(),
        })

    return weighted_sum, genre_proportions, top_genre, top10_results, top10_names_indices, token_segments