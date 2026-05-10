import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
import json
from argparse import Namespace
import sys
sys.path.append('.')
import models.vqvae as vqvae
import numpy as np
import torch

def load_vqvae_model(checkpoint_path="/data1/hzy/HumanMotion/T2M-GPT_mofea264/output/exp_momask_alldata_mofea264_250425_1522/net_best_loss.pth",mean_path="/data2/hzy/InfiniteDance/InfiniteDanceData/dance/alldata_new_joint_vecs264/meta/Mean.npy",std_path="/data2/hzy/InfiniteDance/InfiniteDanceData/dance/alldata_new_joint_vecs264/meta/Std.npy",device=None):
    # checkpoint_path = '/data1/hzy/HumanMotion/T2M-GPT_mofea264/output/exp_momask_aistpp_mofea264_250619_0859/net_best_loss_train.pth'
    # print(f"Loading VQVAE model from {checkpoint_path}")
    opt_path = os.path.join(os.path.dirname(checkpoint_path), "args.json")
    with open(opt_path, 'r') as f:
        opt_dict = json.load(f)
    mean= np.load(mean_path)
    std= np.load(std_path)
    opt = Namespace(**opt_dict)
    dim_pose = 264
    net = vqvae.RVQVAE(
        opt,
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
        opt.vq_norm
    )
    
    def forward(self, x):
        # x= (x-mean)/std #normalize
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5)
        x_out = self.decoder(x_quantized)
        return x_out, commit_loss, perplexity, x_quantized, code_idx
    
    vqvae.RVQVAE.forward = forward
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    net.load_state_dict(checkpoint['net'], strict=False)
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.eval()
    
    for param in net.parameters():
        param.requires_grad = False
    
    codebooks = [
        net.quantizer.layers[i].codebook.data
        for i in range(opt.num_quantizers)
    ]
    
    return net, codebooks, opt

# MotionEmbedding function using global codebooks
def MotionEmbedding(motion_idx,codebooks, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    if codebooks is None:
        raise ValueError("Codebooks not initialized. Ensure VQVAE model is loaded in main.")
    
    motion_idx = motion_idx.to(device)
    B, L, G = motion_idx.shape
    D = codebooks[0].shape[1]
    all_vectors = []
    
    for g in range(G):
        codebook_tokens = motion_idx[:, :, g]
        codebook_g = codebooks[g].to(device)
        vectors_g = codebook_g[codebook_tokens]
        all_vectors.append(vectors_g)
    
    vectors = torch.cat(all_vectors, dim=0)
    vectors = vectors.permute(1, 0, 2)         # shape: (L, 3, D)
    vectors = vectors.reshape(-1, vectors.shape[-1])  # shape: (L * 3, D)
    return vectors

# Dance embedding function
def get_motion_embeddings(token,codebooks):
    if isinstance(token, list):
        token = np.array(token)
    
    if token.ndim > 1:
        token = token.flatten()
    
    seq_len = len(token)
    if seq_len % 3 != 0:
        raise ValueError(f"Dance data length {seq_len} is not a multiple of 3 for triplet structure.")
    
    a_range = (4096, 4096 + 512 - 1)
    b_range = (4096 + 512, 4096 + 2*512 - 1)
    c_range = (4096 + 2*512, 4096 + 3*512 - 1)
    offsets = [4096, 4096 + 512, 4096 + 2*512]
    
    for i in range(0, seq_len, 3):
        a, b, c = token[i], token[i+1], token[i+2]
        if not (a_range[0] <= a <= a_range[1] and
                b_range[0] <= b <= b_range[1] and
                c_range[0] <= c <= c_range[1]):
            raise ValueError(f"Invalid dance triplet at index {i}: a={a}, b={b}, c={c}")
    
    token_adjusted = np.copy(token)
    for i in range(0, seq_len, 3):
        token_adjusted[i] -= offsets[0]
        token_adjusted[i+1] -= offsets[1]
        token_adjusted[i+2] -= offsets[2]
    
    for i in range(0, seq_len, 3):
        a, b, c = token_adjusted[i], token_adjusted[i+1], token_adjusted[i+2]
        if not (0 <= a <= 511 and 0 <= b <= 511 and 0 <= c <= 511):
            raise ValueError(f"Adjusted dance triplet out of range at index {i}: a={a}, b={b}, c={c}")
    
    data_reshaped = token_adjusted.reshape(1, seq_len // 3, 3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    motion_idx = torch.from_numpy(data_reshaped).to(device)
    motion_embeds = MotionEmbedding(motion_idx,codebooks)
    return motion_embeds

def get_motion_embeddings_ta2m(token,codebooks):
    if isinstance(token, list):
        token = np.array(token)
    
    if token.ndim > 1:
        token = token.flatten()
    
    seq_len = len(token)
    if seq_len % 3 != 0:
        raise ValueError(f"Dance data length {seq_len} is not a multiple of 3 for triplet structure.")
    
    # a_range = (4096, 4096 + 512 - 1)
    # b_range = (4096 + 512, 4096 + 2*512 - 1)
    # c_range = (4096 + 2*512, 4096 + 3*512 - 1)
    # offsets = [4096, 4096 + 512, 4096 + 2*512]
    
    # for i in range(0, seq_len, 3):
    #     a, b, c = token[i], token[i+1], token[i+2]
    #     if not (a_range[0] <= a <= a_range[1] and
    #             b_range[0] <= b <= b_range[1] and
    #             c_range[0] <= c <= c_range[1]):
    #         raise ValueError(f"Invalid dance triplet at index {i}: a={a}, b={b}, c={c}")
    
    # token_adjusted = np.copy(token)
    # for i in range(0, seq_len, 3):
    #     token_adjusted[i] -= offsets[0]
    #     token_adjusted[i+1] -= offsets[1]
    #     token_adjusted[i+2] -= offsets[2]
    
    for i in range(0, seq_len, 3):
        a, b, c = token[i], token[i+1], token[i+2]
        if not (0 <= a <= 511 and 0 <= b <= 511 and 0 <= c <= 511):
            raise ValueError(f"Adjusted dance triplet out of range at index {i}: a={a}, b={b}, c={c}")
    
    data_reshaped = token.reshape(1, seq_len // 3, 3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    motion_idx = torch.from_numpy(data_reshaped).to(device)
    motion_embeds = MotionEmbedding(motion_idx,codebooks)
    return motion_embeds
def test():
    # 加载预训练的 VQ-VAE 模型
    net, codebooks, opt = load_vqvae_model(
        checkpoint_path="./models/checkpoints/dance_vqvae.pth"
    )
    net.cuda().eval()  # 将模型移动到GPU并设置为评估模式

    # 定义目录路径
    source_dir = "../InfiniteDanceData"
    target_dir = "../InfiniteDanceData"
    
    # 创建目标目录（如果不存在）
    os.makedirs(target_dir, exist_ok=True)

    # 获取所有.npy文件[2,7](@ref)
    npy_files = [f for f in os.listdir(source_dir) if f.endswith('.npy')]
    
    print(f"找到 {len(npy_files)} 个.npy文件需要处理")

    # 处理每个文件
    for file_name in npy_files:
        try:
            # 构建完整的文件路径[2](@ref)
            source_path = os.path.join(source_dir, file_name)
            target_path = os.path.join(target_dir, file_name)
            
            # 跳过已处理的文件（可选）
            # if os.path.exists(target_path):
            #     print(f"跳过已处理文件: {file_name}")
            #     continue
            
            # 加载.npy文件[1,3](@ref)
            data = np.load(source_path)
            print(f"处理文件: {file_name}, 原始数据形状: {data.shape}")
            
            # 转换为torch张量并移动到GPU
            input_tensor = torch.tensor(data).float().cuda()
            input_tensor=input_tensor.unsqueeze(0)
            # 使用模型进行前向传播和量化
            with torch.no_grad():
                quantized = net.forward_quantizer(input_tensor)
            
            # 将结果移回CPU并转换为numpy数组
            quantized_np = quantized.squeeze(0).permute(1,0).cpu().numpy()
            print(f"量化后形状: {quantized_np.shape}")
            
            # 保存量化后的结果[1](@ref)
            np.save(target_path, quantized_np)
            print(f"已保存: {target_path}")
            
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {str(e)}")
    
    print("所有文件处理完成！")



def test_ta2m():
    # 加载预训练的 VQ-VAE 模型
    net, codebooks, opt = load_vqvae_model(
        checkpoint_path="./models/checkpoints/dance_vqvae.pth"
    )
    net.cuda().eval()  # 将模型移动到GPU并设置为评估模式

    # 定义目录路径
    source_dir = "/data2/shushi/datasd/processed_data/shangdian/pair/emotion_motion_264_30fps"
    target_dir = "/data2/shushi/datasd/processed_data/shangdian/pair/emotion_motion_264_30fps_quantized_embeds"
    
    # 创建目标目录（如果不存在）
    os.makedirs(target_dir, exist_ok=True)

    # 获取所有.npy文件[2,7](@ref)
    npy_files = [f for f in os.listdir(source_dir) if f.endswith('.npy')]
    
    print(f"找到 {len(npy_files)} 个.npy文件需要处理")

    # 处理每个文件
    for file_name in npy_files:
        try:
            # 构建完整的文件路径[2](@ref)
            source_path = os.path.join(source_dir, file_name)
            target_path = os.path.join(target_dir, file_name)
            
            # 跳过已处理的文件（可选）
            # if os.path.exists(target_path):
            #     print(f"跳过已处理文件: {file_name}")
            #     continue
            
            # 加载.npy文件[1,3](@ref)
            data = np.load(source_path)
            print(f"处理文件: {file_name}, 原始数据形状: {data.shape}")
            
            # 转换为torch张量并移动到GPU
            input_tensor = torch.tensor(data).float().cuda()
            input_tensor=input_tensor.unsqueeze(0)
            # 使用模型进行前向传播和量化
            with torch.no_grad():
                quantized = net.forward_quantizer(input_tensor)
            
            # 将结果移回CPU并转换为numpy数组
            quantized_np = quantized.squeeze(0).permute(1,0).cpu().numpy()
            print(f"量化后形状: {quantized_np.shape}")
            
            # 保存量化后的结果[1](@ref)
            np.save(target_path, quantized_np)
            print(f"已保存: {target_path}")
            
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {str(e)}")
    
    print("所有文件处理完成！")
# test_ta2m()