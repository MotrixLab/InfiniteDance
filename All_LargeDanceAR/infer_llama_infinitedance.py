
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
from utils.get_top10_mofea264 import get_top_mofea,get_top_mofea_specific_style
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

def generate_dance_sequence(model, music_features, style_idx, dance_retrieval_cond, prev_dance_tokens=None, max_dance_length=288, temperature=1.0, top_k=50, top_p=0.9, device='cuda', disable_tqdm=True):
    model.eval()
    
    # Validate dance_retrieval_cond is not None
    if dance_retrieval_cond is None:
        raise ValueError("dance_retrieval_cond cannot be None")
    
    # Convert inputs to tensors
    music_features = torch.tensor(music_features, dtype=torch.float32, device=device).unsqueeze(0)  # [1, music_len, music_enc_dim]
    style_idx = torch.tensor([style_idx], dtype=torch.long, device=device)
    dance_retrieval_cond = torch.tensor(dance_retrieval_cond, dtype=torch.float32, device=device).unsqueeze(0)  # [1, retrieval_dance_len, dance_retrieval_cond_dim]

    # Initialize generated sequence
    generated = torch.tensor([], dtype=torch.long, device=device).unsqueeze(0)  # [1, 0]

    # Process initial dance tokens
    if prev_dance_tokens is not None and len(prev_dance_tokens) >= 3:
        num_tokens_to_use = min(60, len(prev_dance_tokens))
        num_tokens_to_use = (num_tokens_to_use // 3) * 3
        initial_dance_tokens = prev_dance_tokens[:num_tokens_to_use].copy()
        
        if len(initial_dance_tokens) >= 3:
            sample_val = initial_dance_tokens[0]
            if 0 <= sample_val <= 511:
                # print(f"Detected relative value tokens (0-511), converting...") # Commented out for cleaner bar
                for i in range(0, len(initial_dance_tokens), 3):
                    if i + 2 < len(initial_dance_tokens):
                        initial_dance_tokens[i] += DANCE_TOKEN_RANGES[0][0]
                        initial_dance_tokens[i+1] += DANCE_TOKEN_RANGES[1][0]
                        initial_dance_tokens[i+2] += DANCE_TOKEN_RANGES[2][0]
        
        valid_tokens = True
        for i in range(0, len(initial_dance_tokens), 3):
            if i + 2 >= len(initial_dance_tokens):
                break
            a, b, c = initial_dance_tokens[i], initial_dance_tokens[i+1], initial_dance_tokens[i+2]
            if not (DANCE_TOKEN_RANGES[0][0] <= a <= DANCE_TOKEN_RANGES[0][1] and
                    DANCE_TOKEN_RANGES[1][0] <= b <= DANCE_TOKEN_RANGES[1][1] and
                    DANCE_TOKEN_RANGES[2][0] <= c <= DANCE_TOKEN_RANGES[2][1]):
                # print(f"Warning: Initial dance triplet invalid...") # Commented out
                valid_tokens = False
                break
        
        if valid_tokens:
            initial_tensor = torch.tensor(initial_dance_tokens, dtype=torch.long, device=device).unsqueeze(0)
            generated = initial_tensor
    
    if generated.shape[1] == 0:
        start_token = torch.tensor([[DANCE_TOKEN_RANGES[0][0]]], dtype=torch.long, device=device)
        generated = start_token

    with torch.no_grad():
        remaining_length = max_dance_length - generated.shape[1]
        if remaining_length <= 0:
            return generated[0].cpu().numpy()
        
        for _ in tqdm(range(remaining_length // 3), desc="Generating tokens", disable=disable_tqdm, leave=False):
            dance_triplet = []
            current_generated = generated
            for i in range(3):
                outputs = model(
                    input_ids=current_generated,
                    labels=None,
                    music_features=music_features,
                    style_indices=style_idx,
                    dance_retrieval_cond=dance_retrieval_cond,
                    stage=2,
                    infer=True
                )
                next_token_logits = outputs.logits[:, -1, :] / temperature
                valid_range = DANCE_TOKEN_RANGES[i]
                valid_mask = torch.ones(next_token_logits.shape[-1], device=device, dtype=torch.bool)
                valid_mask[:valid_range[0]] = False
                valid_mask[valid_range[1] + 1:] = False
                filtered_logits = next_token_logits.masked_fill(~valid_mask, float('-inf'))
                
                top_k_values, top_k_indices = torch.topk(filtered_logits, top_k, dim=-1)
                
                probs_topk = torch.nn.functional.softmax(top_k_values, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs_topk, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                sorted_probs = sorted_probs.masked_fill(sorted_indices_to_remove, 0.0)
                sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-10)
                
                next_token_idx_in_sorted = torch.multinomial(sorted_probs, num_samples=1)
                idx_in_topk = sorted_indices.gather(1, next_token_idx_in_sorted)
                next_token = top_k_indices.gather(1, idx_in_topk)
                next_token_value = next_token.item()
                
                if not (valid_range[0] <= next_token_value <= valid_range[1]):
                    next_token_value = torch.randint(valid_range[0], valid_range[1] + 1, (1,), device=device).item()
                
                dance_triplet.append(next_token_value)
                current_generated = torch.cat([current_generated, torch.tensor([[next_token_value]], dtype=torch.long, device=device)], dim=1)

            if len(dance_triplet) == 3:
                for j, token in enumerate(dance_triplet):
                    if not (DANCE_TOKEN_RANGES[j][0] <= token <= DANCE_TOKEN_RANGES[j][1]):
                        dance_triplet = []
                        break
            else:
                break

            if dance_triplet:
                triplet_tensor = torch.tensor(dance_triplet, dtype=torch.long, device=device).unsqueeze(0)
                generated = torch.cat([generated, triplet_tensor], dim=1)
            else:
                break

    dance_tokens = generated[0].cpu().numpy()
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
        dance_retrieval_cond, genre_proportions, top_genre, mofea_top1, top10_names_indices, token_segments = get_top_mofea_specific_style(
            name=base_name,
            idx=music_retrieval_idx,
            style=target_style,
            infertype=infertype
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
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
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
    out_dir = "infer_"
    out_dir = os.path.join(out_dir, f'dance_{current_time}')
    parser.add_argument("--music_path", required=True, type=str)
    parser.add_argument("--mean_path", required=True, type=str)
    parser.add_argument("--std_path", required=True, type=str)
    parser.add_argument("--checkpoint_path", required=True, type=str)
    parser.add_argument("--vqvae_checkpoint_path", required=True, type=str)
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
    parser.add_argument("--processes_per_gpu", type=int, default=2, help="Number of processes per GPU")
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

    gpus = [0]
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