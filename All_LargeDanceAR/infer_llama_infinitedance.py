import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import math
import multiprocessing as mp
import random
import re
import subprocess
from datetime import datetime, timedelta
import traceback
import numpy as np
import torch
from tqdm import tqdm
from utils.get_top10_mofea264 import (
    get_top_mofea, get_top_mofea_specific_style,
    get_top_mofea_specific_style_exclude_aistpp_finedance,
    get_top_mofea_specific_style_only_aistpp_finedance,
)
if os.environ.get('RETRIEVAL_EXCLUDE_AISTPP_FINEDANCE', '0') == '1':
    print("[INFER_SAFE] retrieval excludes AIST++/FineDance")
    get_top_mofea_specific_style = get_top_mofea_specific_style_exclude_aistpp_finedance
elif os.environ.get('RETRIEVAL_ONLY_AISTPP_FINEDANCE', '0') == '1':
    print("[INFER_SAFE] retrieval keeps ONLY AIST++/FineDance")
    get_top_mofea_specific_style = get_top_mofea_specific_style_only_aistpp_finedance
from models.dancellama import Music2DanceLlamaModel
from models.motion import load_vqvae_model


# Constants
MUSIC_VOCAB_SIZE = 4096
DANCE_TOKEN_START = MUSIC_VOCAB_SIZE
DANCE_TOKEN_INCREMENT = 512 * 3
DANCE_RANGES_COUNT = 1
SPECIAL_TOKENS_COUNT = 256
TOTAL_VOCAB_SIZE = DANCE_TOKEN_START + DANCE_TOKEN_INCREMENT + SPECIAL_TOKENS_COUNT
GENRES = ["Ballet", "Popular", "Latin", "Modern", "Folk", "Classic"]
GENRE_TO_IDX = {genre: idx for idx, genre in enumerate(GENRES)}
DANCE_TOKEN_RANGES = [
    (DANCE_TOKEN_START, DANCE_TOKEN_START + 511),
    (DANCE_TOKEN_START + 512, DANCE_TOKEN_START + 1023),
    (DANCE_TOKEN_START + 1024, DANCE_TOKEN_START + 1535),
]

# Global codebooks
codebooks = None

def _sample_one(logits_row, valid_range, temperature, top_k, top_p, device,
                history=None, repetition_penalty=1.0, max_repeats=999, ngram_size=0):
    """logits_row: (V,)
    history: 之前生成的同 slot token list(可选,for repetition_penalty + max_repeats + ngram block)
    repetition_penalty: token-level penalty(>1 抑制 history 中出现过的 token)
    max_repeats: 同 slot 内某 token 出现 >= 这个次数,直接 ban
    ngram_size: 禁止最近 ngram_size 个 token 序列重复(0 关闭)
    """
    next_token_logits = logits_row.clone() / temperature
    V = next_token_logits.shape[0]
    valid_mask = torch.ones(V, device=device, dtype=torch.bool)
    valid_mask[:valid_range[0]] = False
    valid_mask[valid_range[1] + 1:] = False

    # === 1. repetition_penalty(对最近 history 中出现过的 token 降权)===
    if history and repetition_penalty != 1.0:
        recent = set(history[-30:])  # 最近 30 个 slot 内 token
        for tok in recent:
            if 0 <= tok < V:
                # 标准 HF 实现:if logit > 0: logit /= penalty,else: logit *= penalty
                if next_token_logits[tok] > 0:
                    next_token_logits[tok] = next_token_logits[tok] / repetition_penalty
                else:
                    next_token_logits[tok] = next_token_logits[tok] * repetition_penalty

    # === 2. max_repeats(单 token 在最近 N 帧出现 >= max_repeats,直接 ban)===
    if history and max_repeats < 999:
        recent_window = history[-(max_repeats + 5):]
        from collections import Counter
        ctr = Counter(recent_window)
        for tok, cnt in ctr.items():
            if cnt >= max_repeats and 0 <= tok < V:
                valid_mask[tok] = False  # 屏蔽该 token

    # === 3. ngram block(禁止最近 ngram_size 个 token 序列重复)===
    if history and ngram_size > 0 and len(history) >= ngram_size:
        last_ngram = tuple(history[-(ngram_size - 1):])
        # 找 history 中前面的 (ngram_size-1) 个序列匹配 last_ngram 的位置,ban 紧跟的下一个 token
        banned = set()
        for i in range(len(history) - ngram_size + 1):
            seg = tuple(history[i:i + ngram_size - 1])
            if seg == last_ngram:
                next_tok_pos = i + ngram_size - 1
                if next_tok_pos < len(history):
                    banned.add(history[next_tok_pos])
        for tok in banned:
            if 0 <= tok < V:
                valid_mask[tok] = False

    filtered = next_token_logits.masked_fill(~valid_mask, float('-inf'))
    top_k_eff = min(top_k, max(1, int(valid_mask.sum().item())))
    top_v, top_i = torch.topk(filtered, top_k_eff, dim=-1)
    probs = torch.softmax(top_v, dim=-1)
    sp, si = torch.sort(probs, descending=True)
    cum = torch.cumsum(sp, dim=-1)
    cut = cum > top_p
    cut[..., 1:] = cut[..., :-1].clone()
    cut[..., 0] = False
    sp = sp.masked_fill(cut, 0.0)
    sp = sp / (sp.sum() + 1e-10)
    pos = torch.multinomial(sp, num_samples=1)
    idx_in_topk = si.gather(0, pos)
    nxt = top_i.gather(0, idx_in_topk).item()
    if not (valid_range[0] <= nxt <= valid_range[1]):
        nxt = int(torch.randint(valid_range[0], valid_range[1] + 1, (1,), device=device).item())
    return nxt


def generate_dance_sequence(model, music_features, style_idx, dance_retrieval_cond, prev_dance_tokens=None, max_dance_length=288, temperature=1.0, top_k=50, top_p=0.9, device='cuda', disable_tqdm=True):
    """KV-cache + bridge cache 加速版.

    优化:
      1. music_embeds 一次性算,缓存(全程不变)
      2. dance_bridge MLP per-frame,新 token 来时只算一帧 bridge
      3. LLaMA 用 past_key_values,首次喂 [music + prefix dance] 整段,之后每次只喂 1 个新 token 的 hidden
    """
    model.eval()
    if dance_retrieval_cond is None:
        raise ValueError("dance_retrieval_cond cannot be None")

    music_features = torch.tensor(music_features, dtype=torch.float32, device=device).unsqueeze(0)
    style_idx_t = torch.tensor([style_idx], dtype=torch.long, device=device)
    dance_retrieval_cond = torch.tensor(dance_retrieval_cond, dtype=torch.float32, device=device).unsqueeze(0)

    music_len_local = music_features.size(1)

    # ----- 处理 prev_dance_tokens(retrieval prefix) -----
    initial_dance_tokens = []
    if prev_dance_tokens is not None and len(prev_dance_tokens) >= 3:
        num_tokens_to_use = min(60, len(prev_dance_tokens))
        num_tokens_to_use = (num_tokens_to_use // 3) * 3
        initial_dance_tokens = list(prev_dance_tokens[:num_tokens_to_use])
        if len(initial_dance_tokens) >= 3 and 0 <= initial_dance_tokens[0] <= 511:
            for i in range(0, len(initial_dance_tokens), 3):
                if i + 2 < len(initial_dance_tokens):
                    initial_dance_tokens[i] += DANCE_TOKEN_RANGES[0][0]
                    initial_dance_tokens[i + 1] += DANCE_TOKEN_RANGES[1][0]
                    initial_dance_tokens[i + 2] += DANCE_TOKEN_RANGES[2][0]
        # validate
        ok = True
        for i in range(0, len(initial_dance_tokens), 3):
            if i + 2 >= len(initial_dance_tokens):
                break
            a, b, c = initial_dance_tokens[i], initial_dance_tokens[i + 1], initial_dance_tokens[i + 2]
            if not (DANCE_TOKEN_RANGES[0][0] <= a <= DANCE_TOKEN_RANGES[0][1] and
                    DANCE_TOKEN_RANGES[1][0] <= b <= DANCE_TOKEN_RANGES[1][1] and
                    DANCE_TOKEN_RANGES[2][0] <= c <= DANCE_TOKEN_RANGES[2][1]):
                ok = False
                break
        if not ok:
            initial_dance_tokens = []
    # ★ fast 版:不像旧版那样塞 start_token 占位(会破坏 triplet 对齐)
    # 直接从 music_embeds 末尾的 logits 开始采(那个位置训练时正好预测 dance slot 0)
    generated = list(initial_dance_tokens)

    with torch.no_grad():
        # ----- (1) music_embeds 一次性算(永远不变) -----
        # 复刻 model.forward 里的 music_embeds 路径(不走 batch 维)
        style_emb = model.style_embedding(style_idx_t)  # (1, D_style)
        dance_cond = dance_retrieval_cond.permute(0, 2, 1)  # (1, 264, 384)
        dance_cond = model.cond_projection(dance_cond)      # (1, 264, music_len)
        dance_cond = dance_cond.permute(0, 2, 1)            # (1, music_len, 264)

        style_emb_exp = style_emb.unsqueeze(1).expand(-1, music_len_local, -1)
        dance_cond_exp = dance_cond.expand(-1, music_len_local, -1)
        combined = torch.cat([music_features, style_emb_exp, dance_cond_exp], dim=-1)
        music_embeds = model.music_bridge(combined)  # (1, music_len, hidden)
        if music_embeds.dim() == 2:
            music_embeds = music_embeds.unsqueeze(0)
        music_embeds = music_embeds.to(dtype=torch.float32)

        # ----- (2) initial dance hidden(retrieval prefix 那段)-----
        valid_tokens_count = (len(initial_dance_tokens) // 3) * 3
        prefix_dance_tokens = initial_dance_tokens[:valid_tokens_count]
        if valid_tokens_count > 0:
            from models.motion import get_motion_embeddings
            prefix_features = get_motion_embeddings(np.array(prefix_dance_tokens, dtype=np.int64), model.codebooks)  # (T, in_dim)
            if prefix_features.device != music_embeds.device:
                prefix_features = prefix_features.to(music_embeds.device)
            prefix_dance_embeds = model.dance_bridge(prefix_features)  # MLP per-frame
            if prefix_dance_embeds.dim() == 3:
                prefix_dance_embeds = prefix_dance_embeds.squeeze(0)
            prefix_dance_embeds = prefix_dance_embeds.to(dtype=torch.float32)
        else:
            prefix_dance_embeds = torch.zeros(0, music_embeds.size(-1), device=device, dtype=torch.float32)

        # ----- (3) 首次 forward:整段 [music + prefix dance] -----
        first_input = torch.cat([music_embeds.squeeze(0), prefix_dance_embeds], dim=0).unsqueeze(0)  # (1, T0, hidden)
        out = model.llama(inputs_embeds=first_input, use_cache=True, return_dict=True)
        past_kv = out.past_key_values
        # 取最后一个位置 logits 作为预测下一个 token 的依据
        last_logits = out.logits[0, -1, :]  # (V,)

        remaining_length = max_dance_length - len(generated)
        if remaining_length <= 0:
            return np.array(generated, dtype=np.int64)

        # ----- (4) 自回归 KV-cache 循环 -----
        # 关键观察:训练时 dance 部分每个 token 对应一个 LLaMA hidden(motion_embeddings 输出 288 个 hidden 给 288 个 token)
        # 单 token id (在 group g 范围内) → codebook[g][rel_idx] → 1 个 D 维 motion feature → dance_bridge MLP → 1 个 hidden
        codebooks_dev = [cb.to(device) for cb in model.codebooks]

        # ★ Anti-mode-collapse 配置(可通过环境变量调整)
        SAFE_REP_PEN = float(os.environ.get('SAFE_REP_PENALTY', '1.15'))   # 1.0=关
        SAFE_MAX_REP_S0 = int(os.environ.get('SAFE_MAX_REP_S0', '12'))     # slot0 同 token 最多 12 次
        SAFE_MAX_REP_OTHER = int(os.environ.get('SAFE_MAX_REP_OTHER', '20')) # slot 1/2 宽松点
        SAFE_NGRAM_S0 = int(os.environ.get('SAFE_NGRAM_S0', '4'))          # slot0 ngram block 4
        SAFE_TEMP_BOOST = float(os.environ.get('SAFE_TEMP_BOOST', '1.5'))  # 检测重复时升温度倍数

        # slot 历史:0=root vel/pose, 1=mid layer, 2=fine
        slot_history = [[], [], []]   # 仅 slot 内的 token

        # 用于动态温度:检测最近 6 个 slot 0 token unique 数
        for _ in tqdm(range(remaining_length // 3), desc="Generating tokens", disable=disable_tqdm, leave=False):
            triplet_vals = []
            triplet_ok = True
            for i in range(3):
                # 动态温度:slot 0 检测重复,如果最近 6 个 unique<=2,临时升温
                cur_temp = temperature
                if i == 0 and len(slot_history[0]) >= 6:
                    recent6 = slot_history[0][-6:]
                    if len(set(recent6)) <= 2:
                        cur_temp = temperature * SAFE_TEMP_BOOST

                # 限制:slot 0 严格(max_repeats=12, ngram=4),slot 1/2 宽松
                if i == 0:
                    max_rep = SAFE_MAX_REP_S0
                    ngram = SAFE_NGRAM_S0
                else:
                    max_rep = SAFE_MAX_REP_OTHER
                    ngram = 0  # slot 1/2 不用 ngram block(可能太严)

                nxt_value = _sample_one(last_logits, DANCE_TOKEN_RANGES[i], cur_temp, top_k, top_p, device,
                                         history=slot_history[i],
                                         repetition_penalty=SAFE_REP_PEN,
                                         max_repeats=max_rep, ngram_size=ngram)
                slot_history[i].append(nxt_value)
                triplet_vals.append(nxt_value)
                if not (DANCE_TOKEN_RANGES[i][0] <= nxt_value <= DANCE_TOKEN_RANGES[i][1]):
                    triplet_ok = False
                    break

                # 直接查 codebook 拿单 token 的 motion feature
                rel_idx = nxt_value - DANCE_TOKEN_RANGES[i][0]
                motion_feat = codebooks_dev[i][rel_idx]  # (D,)
                motion_feat = motion_feat.unsqueeze(0)   # (1, D),per-frame MLP 输入

                # MLP per-frame 算 LLaMA hidden
                new_emb = model.dance_bridge(motion_feat)  # (1, hidden) 或 (1,1,hidden)
                if new_emb.dim() == 2:
                    new_emb = new_emb.unsqueeze(0)  # (1, 1, hidden)
                new_emb = new_emb.to(dtype=torch.float32)

                # KV-cache LLaMA forward,只送这 1 个新 hidden
                out = model.llama(inputs_embeds=new_emb, past_key_values=past_kv, use_cache=True, return_dict=True)
                past_kv = out.past_key_values
                last_logits = out.logits[0, -1, :]

            if not triplet_ok:
                break
            generated.extend(triplet_vals)
            if len(generated) >= max_dance_length:
                break

    dance_tokens = np.array(generated, dtype=np.int64)
    if len(dance_tokens) % 3 != 0:
        valid_length = (len(dance_tokens) // 3) * 3
        dance_tokens = dance_tokens[:valid_length]

    for i in range(0, len(dance_tokens), 3):
        if i + 2 < len(dance_tokens):
            a, b, c = dance_tokens[i], dance_tokens[i+1], dance_tokens[i+2]
            if not (DANCE_TOKEN_RANGES[0][0] <= a <= DANCE_TOKEN_RANGES[0][1] and
                    DANCE_TOKEN_RANGES[1][0] <= b <= DANCE_TOKEN_RANGES[1][1] and
                    DANCE_TOKEN_RANGES[2][0] <= c <= DANCE_TOKEN_RANGES[2][1]):
                dance_tokens = dance_tokens[:i]
                break

    return dance_tokens

def generate_full_dance_sequence(model, music_features, base_name, style_idx, music_length=320, dance_length=288, overlap=96, temperature=1.0, top_k=50, top_p=0.9, device='cuda', infertype="infinitedance", style_consistency_check=True):
    total_music_length = len(music_features)
    total_dance_length = total_music_length * 9 // 10
    full_dance_sequence = []
    dance_window_size = dance_length
    dance_step = dance_window_size - overlap
    music_window_size = music_length
    num_windows = max(1, (total_dance_length - dance_window_size) // dance_step + 1)
    
    target_style = GENRES[style_idx]
    # Replaced print with just logic, or use tqdm.write if really needed. Keeping it silent for loop cleanliness.
    # print(f"Generating full dance sequence...") 

    for win_idx in range(num_windows):
        dance_start_idx = win_idx * dance_step
        dance_end_idx = dance_start_idx + dance_window_size
        music_start_idx = dance_start_idx * 10 // 9
        music_end_idx = dance_end_idx * 10 // 9
        music_end_idx = min(music_end_idx, total_music_length)
        music_features_sub = music_features[music_start_idx:music_end_idx]

        if len(music_features_sub) < music_window_size:
            reps = math.ceil(music_window_size / len(music_features_sub))
            tiled_features = np.tile(music_features_sub, (reps, 1))
            music_features_sub = tiled_features[:music_window_size]

        music_retrieval_idx = music_start_idx // 320
        # 通过环境变量控制 retrieval top_k(用于 K sweep 实验)
        _retrieval_top_k = int(os.environ.get('RETRIEVAL_TOP_K', '10'))
        dance_retrieval_cond, genre_proportions, top_genre, mofea_top1, top10_names_indices, token_segments = get_top_mofea_specific_style(
            name=base_name,
            idx=music_retrieval_idx,
            style=target_style,
            infertype=infertype,
            top_k=_retrieval_top_k,
        )
        
        if dance_retrieval_cond is None:
            raise ValueError(f"Window {win_idx + 1}: Unable to get dance_retrieval_cond")
        
        if style_consistency_check and top_genre and top_genre != target_style:
             if genre_proportions and target_style in genre_proportions:
                target_style_proportion = genre_proportions[target_style]
                # if target_style_proportion < 0.3:
                    # tqdm.write(f"Warning: Low style proportion...") # Optional logging

        window_prev_dance = None
        if win_idx == 0:
            if token_segments and len(token_segments[0]["tokens"]) >= 3:
                num_tokens = min(60, len(token_segments[0]["tokens"]))
                num_tokens = (num_tokens // 3) * 3
                window_prev_dance = np.array(token_segments[0]["tokens"][:num_tokens])
        else:
            if len(full_dance_sequence) >= overlap:
                prev_window_tokens = np.array(full_dance_sequence[-overlap:])
                if len(prev_window_tokens) >= 3:
                    num_tokens = min(60, len(prev_window_tokens))
                    num_tokens = (num_tokens // 3) * 3
                    window_prev_dance = prev_window_tokens[:num_tokens]
                elif token_segments and len(token_segments[0]["tokens"]) >= 3:
                    num_tokens = min(60, len(token_segments[0]["tokens"]))
                    num_tokens = (num_tokens // 3) * 3
                    window_prev_dance = np.array(token_segments[0]["tokens"][:num_tokens])
            elif token_segments and len(token_segments[0]["tokens"]) >= 3:
                num_tokens = min(60, len(token_segments[0]["tokens"]))
                num_tokens = (num_tokens // 3) * 3
                window_prev_dance = np.array(token_segments[0]["tokens"][:num_tokens])

        # Pass disable_tqdm=True to avoid clutter
        dance_tokens = generate_dance_sequence(
            model,
            music_features_sub,
            style_idx,
            dance_retrieval_cond,
            prev_dance_tokens=window_prev_dance,
            max_dance_length=dance_window_size,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
            disable_tqdm=True 
        )

        if win_idx == 0:
            full_dance_sequence.extend(dance_tokens)
        else:
            full_dance_sequence.extend(dance_tokens[overlap:])

    return np.array(full_dance_sequence)

def process_music_files(process_id, gpu_id, music_files, args, output_dir, dance_output_dir, pbar_position):
    try:
        #         os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        torch.cuda.set_device(gpu_id)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        seed = args.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        global codebooks
        _, codebooks, _ = load_vqvae_model(checkpoint_path=args.vqvae_checkpoint_path,mean_path=args.mean_path,std_path=args.std_path, device=device)
        # Using tqdm.write to avoid breaking the progress bar
        tqdm.write(f"Process {process_id}: VQ-VAE model loaded.")

        model = Music2DanceLlamaModel(
            pretrained_model_name="models/Llama3.2-1B",
            vocab_size=TOTAL_VOCAB_SIZE,
            dance_token_start=DANCE_TOKEN_START,
            dance_token_increment=DANCE_TOKEN_INCREMENT,
            dance_ranges_count=DANCE_RANGES_COUNT,
            num_styles=len(GENRES),
            style_embedding_dim=64,
            music_enc_dim=1024,
            dance_enc_dim=1024,
            llama_hidden_dim=2048,
            bridge_hidden_dim=2048,
            bridge_num_heads=8,
            dance_retrieval_cond_dim=264,
            music_len=320,
            retrieval_dance_len=384,
            codebooks=codebooks
        )
        state_dict = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        model.eval()
        tqdm.write(f"Process {process_id}: Model loaded.")


        pbar = tqdm(music_files, desc=f"GPU {gpu_id} | Proc {process_id}", position=pbar_position, leave=True)

        for music_path in pbar:
            music_file_basename = os.path.basename(music_path).replace('.npy', '')
            
            # Update bar postfix to show current file
            pbar.set_postfix_str(f"File: {music_file_basename[:15]}...")

            if not os.path.exists(music_path):
                tqdm.write(f"Process {process_id}: Error: Music file {music_file_basename} does not exist.")
                continue

            music_features = np.load(music_path).squeeze()

            if len(music_features) < args.music_length:
                reps = math.ceil(args.music_length / len(music_features))
                tiled_features = np.tile(music_features, (reps, 1))
                music_features = tiled_features[:args.music_length]

            existing_dance_files = [f for f in os.listdir(dance_output_dir) if re.match(rf"dance_sample_\d+_\d+_{music_file_basename}\.npy", f)]
            existing_samples = set()
            for f in existing_dance_files:
                match = re.match(rf"dance_sample_(\d+)_\d+_{music_file_basename}\.npy", f)
                if match:
                    sample_num = int(match.group(1))
                    existing_samples.add(sample_num)

            first_retrieval_idx = 0
            _, genre_proportions, top_genre_from_retrieval, _, _, _ = get_top_mofea(
                name=music_file_basename,
                idx=first_retrieval_idx
            )

            matched_genre = None
            for genre in GENRES:
                if genre.lower() in music_file_basename.lower():
                    matched_genre = genre
                    break

            if matched_genre:
                final_genre = matched_genre
            elif top_genre_from_retrieval and top_genre_from_retrieval in GENRE_TO_IDX:
                final_genre = top_genre_from_retrieval
            else:
                final_genre = args.style

            if final_genre not in GENRE_TO_IDX:
                final_genre = args.style
            
            style_idx = GENRE_TO_IDX[final_genre]
            
            # Skip loop if all samples exist to save time
            if len(existing_samples) >= args.num_samples:
                continue

            for j in range(1, args.num_samples + 1):
                if j in existing_samples:
                    continue

                if not args.single_sequence:
                    dance_tokens = generate_full_dance_sequence(
                        model,
                        music_features,
                        music_file_basename,
                        style_idx,
                        music_length=args.music_length,
                        dance_length=args.dance_length,
                        overlap=96,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        device=device,
                        infertype=getattr(args, 'infertype', 'infinitedance'),
                        style_consistency_check=getattr(args, 'style_consistency_check', True)
                    )
                else:
                    max_start_idx = len(music_features) - args.music_length
                    possible_start_indices = list(range(0, max_start_idx + 1, 32))
                    if not possible_start_indices:
                        continue
                    start_idx = random.choice(possible_start_indices)
                    music_features_sub = music_features[start_idx:start_idx + args.music_length]

                    music_retrieval_idx = start_idx // 320
                    dance_retrieval_cond, genre_proportions, top_genre, mofea_top1, top10_names_indices, token_segments = get_top_mofea_specific_style(
                        name=music_file_basename,
                        idx=music_retrieval_idx,
                        style=GENRES[style_idx],
                        infertype=getattr(args, 'infertype', 'infinitedance')
                    )
                    
                    if dance_retrieval_cond is None:
                        continue

                    window_prev_dance = None
                    if token_segments and len(token_segments[0]["tokens"]) >= 3:
                        num_tokens = min(60, len(token_segments[0]["tokens"]))
                        num_tokens = (num_tokens // 3) * 3
                        window_prev_dance = np.array(token_segments[0]["tokens"][:num_tokens])

                    dance_tokens = generate_dance_sequence(
                        model,
                        music_features_sub,
                        style_idx,
                        dance_retrieval_cond,
                        prev_dance_tokens=window_prev_dance,
                        max_dance_length=args.dance_length,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        device=device,
                        disable_tqdm=True # Disable inner bar
                    )

                output_file = os.path.join(dance_output_dir, f"dance_sample_{j}_{process_id}_{music_file_basename}.npy")
                np.save(output_file, dance_tokens)


    except Exception as e:
        tqdm.write(f"Process {process_id} failed on GPU {gpu_id}, error: {e}")
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Music2Dance LLaMA model inference script with multi-process support")
    current_time = (datetime.now()).strftime("25%m%d_%H%M")
    out_dir = "infer_finedance_"
    out_dir = os.path.join(out_dir, f'dance_{current_time}')
    # parser.add_argument("--music_path", required=True, type=str)
    # parser.add_argument("--mean_path", required=True, type=str)
    # parser.add_argument("--std_path", required=True, type=str)
    # parser.add_argument("--checkpoint_path", required=True, type=str)
    # parser.add_argument("--vqvae_checkpoint_path", required=True, type=str)
    parser.add_argument("--music_path", default="/data2/hzy/InfiniteDance/InfiniteDanceData/music/muq_features/test_finedance", type=str)
    
    parser.add_argument("--mean_path", default="/data2/hzy/InfiniteDance/InfiniteDanceData/dance/alldata_new_joint_vecs264/meta/Mean.npy", type=str)
    parser.add_argument("--std_path", default="/data2/hzy/InfiniteDance/InfiniteDanceData/dance/alldata_new_joint_vecs264/meta/Std.npy", type=str)
    
    parser.add_argument("--checkpoint_path", default="/data2/hzy/InfiniteDance_opensource/All_LargeDanceAR/output/exp_m2d_infinitedance/best_model_stage2.pt", type=str)
    parser.add_argument("--vqvae_checkpoint_path", default="/data2/hzy/InfiniteDance/All_LargeDanceAR/models/checkpoints/dance_vqvae.pth", type=str)
    parser.add_argument("--output_dir", default=out_dir, type=str)
    parser.add_argument("--music_length", type=int, default=320)
    parser.add_argument("--dance_length", type=int, default=288)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--style", type=str, default="Popular", choices=GENRES)
    parser.add_argument("--single_sequence", action="store_true", default=False)
    parser.add_argument("--processes_per_gpu", type=int, default=4, help="Number of processes per GPU")
    parser.add_argument("--infertype", type=str, default="infinitedance", choices=["infinitedance", "infinitedanceplus"],
                        help="Inference type: infinitedance uses old format tokens, infinitedanceplus uses v6_2 windowed tokens")
    parser.add_argument("--style_consistency_check", action="store_true", default=True,
                        help="Enable style consistency check to ensure all windows use the same style (enabled by default)")
    parser.add_argument("--no_style_consistency_check", dest="style_consistency_check", action="store_false",
                        help="Disable style consistency check")
    args = parser.parse_args()
    
    # ... directory creation ...
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "args_log.txt")
    with open(log_file, 'w') as f:
        f.write("Arguments for the current run:\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    print(f"Arguments saved to {log_file}")
    
    print("Starting inference")
    dance_output_dir = os.path.join(args.output_dir, "dance")
    os.makedirs(dance_output_dir, exist_ok=True)

    if args.music_path:
        if os.path.isdir(args.music_path):
            music_files_to_process = [os.path.join(args.music_path, f) for f in os.listdir(args.music_path) if f.endswith('.npy')]
            if not music_files_to_process:
                raise ValueError(f"No .npy files found in {args.music_path}")
            print(f"Processing {len(music_files_to_process)} music files from {args.music_path}")
        elif os.path.isfile(args.music_path) and args.music_path.endswith('.npy'):
            music_files_to_process = [args.music_path]
            print(f"Processing single music file: {args.music_path}")
        else:
            raise ValueError(f"Invalid music_path: {args.music_path}")
    else:
        raise ValueError("Must provide --music_path argument")

    gpus = [int(g) for g in os.environ.get('LOGICAL_GPUS', '0').split(',') if g]
    processes_per_gpu = args.processes_per_gpu
    total_processes = len(gpus) * processes_per_gpu

    files_per_process = math.ceil(len(music_files_to_process) / total_processes)
    process_file_splits = [music_files_to_process[i:i + files_per_process] for i in range(0, len(music_files_to_process), files_per_process)]

    while len(process_file_splits) < total_processes:
        process_file_splits.append([])

    process_args = []
    # Modified loop to pass position (which is i)
    for i in range(total_processes):
        gpu_id = gpus[i // processes_per_gpu]
        files = process_file_splits[i]
        # Pass i as the last argument for tqdm position
        process_args.append((i, gpu_id, files, args, args.output_dir, dance_output_dir, i))

    print(f"Starting {total_processes} processes on GPU {gpus}")
    
    # Using a list to hold processes
    processes = []
    for pargs in process_args:
        p = mp.Process(target=process_music_files, args=pargs)
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("\nAll processes completed.")
    # postprocess_dance(dance_output_dir)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
