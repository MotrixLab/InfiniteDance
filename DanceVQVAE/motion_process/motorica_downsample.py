import os

import numpy as np

# 目录路径，假设文件在此路径下
directory_path = '/data2/ss/T2M-GPT-main/dataset/NEW/QUA/'
save_path='/data2/ss/T2M-GPT-main/dataset/NEW/QUA_30fps/'
# 获取该目录下的所有.npy文件
npy_files = [f for f in os.listdir(directory_path) if f.endswith('.npy')]

# 遍历所有文件
for npy_file in npy_files:
    # 构造文件的完整路径
    file_path = os.path.join(directory_path, npy_file)

    # 加载.npy文件
    arr = np.load(file_path)

    # 确保数组的形状是 [T, 223]
    if arr.shape[1] == 223:
        # 下采样至 [T//4, 223]：这里我们假设 T 是能被4整除的
        # 通过选择每4个元素取一次样本进行下采样
        downsampled_arr = arr[::4, :]  # 每4行选择1行

        # 检查下采样后的形状
        print(f"Downsampling {npy_file}: {arr.shape} -> {downsampled_arr.shape}")

        # 保存下采样后的文件
        downsampled_file_path = os.path.join(save_path, f"{npy_file}")
        print(downsampled_file_path)
        np.save(downsampled_file_path, downsampled_arr)
    else:
        print(f"File {npy_file} does not have the expected shape [T, 223]. Skipping.")
