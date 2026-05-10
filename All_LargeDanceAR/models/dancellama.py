import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.bridge import MoEMusicBridgeN2 as MoEMusicBridge, DanceBridgeMLP
from torch.utils.data import Sampler
from transformers import AutoConfig, LlamaForCausalLM
# 老 helper get_top10_mofea264 已不可用(bug + 改名 + 路径错位)。
# 用现代 get_top_mofea_specific_style 包装,只取 weighted_sum (dance_retrieval_cond)。
import numpy as _np
from utils.get_top10_mofea264 import get_top_mofea_specific_style as _get_top_mofea_specific_style

_RETRIEVAL_PATH_DEFAULT = "/data2/hzy/InfiniteDance/InfiniteDanceData/dance/retrieval_s192_l384_style"
_MOTION_BASE_DEFAULT = "/data2/hzy/InfiniteDance/InfiniteDanceData/dance/alldata_new_joint_vecs264"
_STYLE_MAP_DEFAULT = "/data2/hzy/InfiniteDance/InfiniteDanceData/styles/all_style_map.json"

def get_top10_mofea264(base_name, music_retrieval_idx, style="Popular"):
    """training-time dance_retrieval_cond getter; falls back to zeros (384,264) on miss."""
    try:
        ws, _, _, _, _, _ = _get_top_mofea_specific_style(
            name=base_name, idx=music_retrieval_idx, style=style,
            retrieval_path=_RETRIEVAL_PATH_DEFAULT,
            motion_base=_MOTION_BASE_DEFAULT,
            style_map_path=_STYLE_MAP_DEFAULT,
            infertype="infinitedance",
        )
        if ws is None:
            return _np.zeros((384, 264), dtype=_np.float32)
        if ws.shape[0] < 384:
            ws = _np.pad(ws, ((0, 384 - ws.shape[0]), (0, 0)), mode='constant')
        return ws[:384].astype(_np.float32)
    except Exception:
        return _np.zeros((384, 264), dtype=_np.float32)
from models.motion import get_motion_embeddings, load_vqvae_model

class Music2DanceLlamaModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name="All_LargeDanceAR/models/Llama3.2-1B",
        vocab_size=4096+4096+256,
        pad_id=0,
        dance_token_start=4096,
        dance_token_increment=512*3,
        dance_ranges_count=1,
        num_styles=6,
        style_embedding_dim=32,
        dance_retrieval_cond_dim=264,
        music_enc_dim=1024,
        music_len=320,
        dance_enc_dim=1024,
        llama_hidden_dim=2048,
        bridge_hidden_dim=2048,
        bridge_num_heads=8,
        retrieval_dance_len=384,
        codebooks=None,
        llama_config_path="/data2/hzy/InfiniteDance/All_LargeDanceAR/models/Llama3.2-1B/config.json",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.codebooks = codebooks
        self.pad_id = pad_id
        self.dance_token_start = dance_token_start
        self.dance_token_increment = dance_token_increment
        self.dance_ranges_count = dance_ranges_count
        self.llama_config_path=llama_config_path

        # ★ 从 Meta 官方 LLaMA-3.2-1B 加载预训练权重(safetensors),不是只用 config 随机初始化
        # llama_config_path 是 /path/to/Llama3.2-1B/config.json,from_pretrained 会读父目录的所有权重
        import os as _os
        llama_dir = _os.path.dirname(llama_config_path)
        try:
            self.llama = LlamaForCausalLM.from_pretrained(llama_dir)
            print(f"[Music2DanceLlamaModel] LLaMA loaded from {llama_dir}")
        except Exception as _e:
            # fallback:旧路径,随机初始化
            print(f"[Music2DanceLlamaModel] from_pretrained failed ({_e}), fallback to random init")
            config = AutoConfig.from_pretrained(llama_config_path)
            self.llama = LlamaForCausalLM(config=config)

        # 扩 vocab(原 LLaMA-3.2 是 128K vocab,扩到 5888 不会有 token 增加而是缩小)
        # 但我们想要 dance token IDs 在 4096-5887 范围 → vocab_size=5888,需要把 LLaMA 的 embed/lm_head 缩到 5888
        self.llama.resize_token_embeddings(vocab_size)

        self.music_bridge = MoEMusicBridge(
            input_dim=music_enc_dim + style_embedding_dim + dance_retrieval_cond_dim,
            output_dim=llama_hidden_dim,
            num_heads=bridge_num_heads,
            hidden_dim=bridge_hidden_dim,
            n_bins=2  # 频段数，与 MoEDanceBridge 一致
        )

        self.dance_bridge = DanceBridgeMLP(
            input_dim=dance_enc_dim,
            output_dim=llama_hidden_dim,
            hidden_dim=bridge_hidden_dim,
        )

        self.style_embedding = nn.Embedding(num_styles, style_embedding_dim)
        self.cond_projection = nn.Linear(retrieval_dance_len, music_len)

        self.style_classifier = nn.Linear(llama_hidden_dim, num_styles)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.005)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.MultiheadAttention):
            for param in module.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def forward(self, input_ids, labels, music_features, style_indices, dance_retrieval_cond, attention_mask=None, stage=1,infer=False):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        hidden_states_list = []

        style_emb = self.style_embedding(style_indices.to(device))
        dance_cond = dance_retrieval_cond.to(device).permute(0, 2, 1)
        dance_cond = self.cond_projection(dance_cond)
        dance_cond = dance_cond.permute(0, 2, 1)

        for b in range(batch_size):
            sample_input = input_ids[b]

            is_dance = (sample_input >= self.dance_token_start) & (sample_input < self.dance_token_start + self.dance_token_increment)
            music_indices = torch.where(~is_dance)[0]
            dance_indices = torch.where(is_dance)[0]

            music_embeds = None
            if len(music_indices) > 0:
                music_features_sample = music_features[b:b+1].to(dtype=torch.float32, device=device)
                style_emb_expanded = style_emb[b:b+1].unsqueeze(1).expand(-1, music_features.size(1), -1)
                dance_cond_expanded = dance_cond[b:b+1].expand(-1, music_features.size(1), -1)
                combined_features = torch.cat([music_features_sample, style_emb_expanded, dance_cond_expanded], dim=-1)
                music_embeds = self.music_bridge(combined_features)  # 移除 style_indices 参数

            dance_embeds = None
            if len(dance_indices) > 0:
                sorted_dance_indices = torch.sort(dance_indices)[0]
                dance_tokens = sample_input[sorted_dance_indices].cpu().numpy()
                num_dance_tokens = len(sorted_dance_indices)
                valid_tokens_count = (num_dance_tokens // 3) * 3
                if valid_tokens_count > 0:
                    valid_dance_tokens = dance_tokens[:valid_tokens_count]
                    dance_features = get_motion_embeddings(valid_dance_tokens, self.codebooks)
                    dance_embeds = self.dance_bridge(dance_features)

            music_seq_len = len(music_indices) if music_embeds is not None else 0
            dance_seq_len = valid_tokens_count if dance_embeds is not None else 0

            total_seq_len = music_seq_len + dance_seq_len

            sample_hidden_states = torch.zeros(
                total_seq_len, self.llama.config.hidden_size,
                device=device, dtype=torch.float32
            )

            offset = 0
            if music_embeds is not None:
                sample_hidden_states[offset:offset + music_seq_len] = music_embeds.squeeze(0)
                offset += music_seq_len
            if dance_embeds is not None:
                sample_hidden_states[offset:offset + dance_seq_len] = dance_embeds

            hidden_states_list.append(sample_hidden_states)

        max_length = max(h.size(0) for h in hidden_states_list)
        padded_hidden_states = []
        attention_masks = []

        for h in hidden_states_list:
            current_length = h.size(0)
            if current_length < max_length:
                padding = torch.zeros(
                    max_length - current_length, self.llama.config.hidden_size,
                    device=device, dtype=torch.float32
                )
                padded_h = torch.cat([h, padding], dim=0)
                mask = torch.ones(max_length, device=device)
                mask[current_length:] = 0
            else:
                padded_h = h
                mask = torch.ones(max_length, device=device)

            padded_hidden_states.append(padded_h)
            attention_masks.append(mask)

        hidden_states = torch.stack(padded_hidden_states)
        batch_attention_mask = torch.stack(attention_masks)

        outputs = self.llama(
            inputs_embeds=hidden_states,
            attention_mask=batch_attention_mask,
            labels=labels,
            return_dict=True,
        )
        style_logits = self.style_classifier(hidden_states.mean(dim=1))
        style_loss = F.cross_entropy(style_logits, style_indices)
        if infer:
            return outputs
        total_loss = outputs.loss + 0.15 * style_loss
        outputs.loss = total_loss

        return outputs

class MusicDanceDataset:
    def __init__(
        self,
        music_dir,
        dance_dir,
        data_split_dir,
        style_dir,
        split='train',
        music_length=320,
        dance_length=288,
        max_samples=None,
        verbose=True,
        window_stride=32,
        dance_token_start=4096,
        dance_token_end=4096 + 512*3 - 1,
        dancedata="finedance",
        vqvae_ckpt_path=None
    ):
        self.vqvae_ckpt_path = vqvae_ckpt_path
        if vqvae_ckpt_path is None:
            raise ValueError("vqvae_ckpt_path must be provided")

        self._net = None
        if self._net is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._net, *_ = load_vqvae_model(self.vqvae_ckpt_path, device=None)
            self._net.to(device).eval()
            if self._net is None:
                raise ValueError("VQVAE model not loaded")
        self.music_dir = music_dir
        self.dance_dir = dance_dir
        self.style_dir = style_dir
        self.music_length = music_length
        self.dance_raw_length = dance_length
        self.dance_token_start = dance_token_start
        self.dance_token_end = dance_token_end
        self.total_dance_length = dance_length
        self.window_stride = window_stride
        if dancedata == "finedance":
            self.genres = ["Popular", "Folk", "Classic"]
        else:
            self.genres = ["Ballet", "Popular", "Latin", "Modern", "Folk", "Classic"]

        self.genre_to_idx = {genre: idx for idx, genre in enumerate(self.genres)}
        self.max_samples = max_samples

        split_file = os.path.join(data_split_dir, f"{dancedata}_{split}.txt")

        with open(split_file, 'r') as f:
            base_names = [line.strip() for line in f.readlines()]

        self.music_files = []
        self.dance_files = []
        self.base_names = []
        self.genres_list = []
        self.windows = []

        valid_count = 0
        a_range = (dance_token_start, dance_token_start + 512 - 1)
        b_range = (dance_token_start + 512, dance_token_start + 2*512 - 1)
        c_range = (dance_token_start + 2*512, dance_token_start + 3*512 - 1)

        for base_name in base_names:
            music_path = os.path.join(music_dir, f"{base_name}.npy")
            dance_path = os.path.join(dance_dir, f"{base_name}.npy")
            style_path = os.path.join(style_dir, f"{base_name}.json")

            if not os.path.exists(music_path) or not os.path.exists(dance_path) or not os.path.exists(style_path):
                continue

            try:
                dance_data = np.load(dance_path).squeeze().flatten()

                with open(style_path, 'r') as f:
                    style_data = json.load(f)
                genre = style_data.get("style")
                if genre not in self.genres:
                    if verbose:
                        print(f"Invalid genre {genre} in {base_name}.json")
                    continue

                dance_min, dance_max = dance_data.min(), dance_data.max()
                if dance_min < self.dance_token_start or dance_max > self.dance_token_end:
                    if verbose:
                        print(f"Warning: Dance tokens out of range in {base_name}.npy: min={dance_min}, max={dance_max}")
                    continue

                if len(dance_data) % 3 != 0:
                    if verbose:
                        print(f"Warning: Dance data length {len(dance_data)} in {base_name}.npy is not a multiple of 3")
                    continue

                valid_triplets = True
                for i in range(0, len(dance_data), 3):
                    if i + 2 >= len(dance_data):
                        valid_triplets = False
                        break
                    a, b, c = dance_data[i], dance_data[i+1], dance_data[i+2]
                    if not (a_range[0] <= a <= a_range[1] and
                            b_range[0] <= b <= b_range[1] and
                            c_range[0] <= c <= c_range[1]):
                        if verbose:
                            print(f"Warning: Invalid dance triplet in {base_name}.npy at index {i}: a={a}, b={b}, c={c}")
                        valid_triplets = False
                        break

                if not valid_triplets:
                    if verbose:
                        print(f"Skipping {base_name}.npy due to invalid dance triplets")
                    continue

                music_features = np.load(music_path).squeeze()
                music_len = music_features.shape[0]
                if music_len >= self.music_length and len(dance_data) >= self.dance_raw_length:
                    self.music_files.append(music_path)
                    self.dance_files.append(dance_data)
                    self.base_names.append(base_name)
                    self.genres_list.append(genre)

                    dance_len = len(dance_data)
                    for dance_start in range(0, dance_len - self.dance_raw_length + 1, self.window_stride):
                        if dance_start % 3 != 0:
                            continue
                        music_start = int(dance_start * 10 / 9)
                        if (music_start + self.music_length > music_len) or (dance_start + self.dance_raw_length > dance_len):
                            continue
                        self.windows.append((len(self.music_files) - 1, dance_start))

                    valid_count += 1

                if self.max_samples is not None and valid_count >= self.max_samples:
                    break
            except Exception as e:
                if verbose:
                    print(f"Error loading {base_name}: {e}")

        self.genre_windows = {genre: [] for genre in self.genres}
        for idx, (file_idx, dance_start) in enumerate(self.windows):
            genre = self.genres_list[file_idx]
            self.genre_windows[genre].append(idx)

        if verbose:
            print(f"Loaded {len(self.music_files)} valid files for {split} split")
            from collections import Counter
            genre_counts = Counter(self.genres_list)
            print(f"Genre distribution (files): {genre_counts}")
            for genre in self.genres:
                total_windows = len(self.genre_windows[genre])
                print(f"Genre {genre}: {len(set(file_idx for file_idx, _ in self.windows if self.genres_list[file_idx] == genre))} files, {total_windows} windows")

        if len(self.music_files) == 0:
            raise ValueError(f"No valid files found for {split} split")

    def __len__(self):
        return min(len(self.windows), self.max_samples or 100000)

    def __getitem__(self, idx):
        file_idx, dance_start = self.windows[idx]

        music_path = self.music_files[file_idx]
        dance_data = self.dance_files[file_idx]
        base_name = self.base_names[file_idx]
        genre = self.genres_list[file_idx]
        music_data = np.load(music_path).squeeze()
        music_len = len(music_data)
        dance_len = len(dance_data)

        music_start = int(dance_start * 10 / 9)
        music_retrieval_idx = round(music_start / 320)

        if self._net is None:
            raise ValueError("VQVAE model not loaded")
        dance_retrieval_cond = get_top10_mofea264(base_name, music_retrieval_idx)

        music_window = music_data[music_start:music_start + self.music_length]
        dance_window = dance_data[dance_start:dance_start + self.dance_raw_length]

        music_placeholder = np.arange(self.music_length)
        combined_tokens = np.concatenate([music_placeholder, dance_window])
        
        labels = np.full_like(combined_tokens, -100)
        labels[self.music_length:] = combined_tokens[self.music_length:]

        style_idx = self.genre_to_idx[genre]
        if isinstance(dance_retrieval_cond, np.ndarray):
            dance_retrieval_cond = torch.from_numpy(dance_retrieval_cond)
        elif isinstance(dance_retrieval_cond, list):
            dance_retrieval_cond = torch.tensor(dance_retrieval_cond)
        dance_retrieval_cond = dance_retrieval_cond.clone().detach().to(torch.float32)

        return {
            "input_ids": torch.tensor(combined_tokens, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "base_name": base_name,
            "style_idx": torch.tensor(style_idx, dtype=torch.long),
            "dance_retrieval_cond": dance_retrieval_cond,
            "music_features": music_window
        }

class GenreBalancedDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas, rank, shuffle=True, seed=42, total_samples=36000):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.total_samples = total_samples

        self.genres = dataset.genres
        self.genre_windows = dataset.genre_windows
        self.samples_per_genre = self.total_samples // len(self.genres)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = []

        for genre in self.genres:
            genre_indices = self.genre_windows.get(genre, [])
            num_available = len(genre_indices)

            if num_available >= self.samples_per_genre:
                sampled = np.random.choice(
                    genre_indices,
                    size=self.samples_per_genre,
                    replace=False
                ).tolist()
            else:
                full = genre_indices * (self.samples_per_genre // num_available)
                remainder = self.samples_per_genre % num_available
                remainder_sampled = np.random.choice(
                    genre_indices,
                    size=remainder,
                    replace=False if remainder <= num_available else True
                ).tolist()
                sampled = full + remainder_sampled

            indices.extend(sampled)

        remaining = self.total_samples - len(indices)
        if remaining > 0:
            all_indices = [idx for genre in self.genres for idx in self.genre_windows[genre]]
            extra = np.random.choice(all_indices, size=remaining, replace=True).tolist()
            indices.extend(extra)

        if self.shuffle:
            indices = torch.tensor(indices)
            indices = indices[torch.randperm(len(indices), generator=g)].tolist()

        indices = indices[self.rank::self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.total_samples // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch


def variable_length_collate_fn(batch):
    max_len = max(b["input_ids"].size(0) for b in batch)

    input_ids_list, labels_list, attention_mask_list = [], [], []
    for b in batch:
        L = b["input_ids"].size(0)
        pad = max_len - L
        input_ids_list.append(
            torch.cat([b["input_ids"], torch.zeros(pad, dtype=torch.long)])
        )
        labels_list.append(
            torch.cat([b["labels"], torch.full((pad,), -100, dtype=torch.long)])
        )
        attention_mask_list.append(
            torch.cat([
                torch.ones(L, dtype=torch.float32),
                torch.zeros(pad, dtype=torch.float32),
            ])
        )

    return {
        "input_ids":            torch.stack(input_ids_list),
        "labels":               torch.stack(labels_list),
        "attention_mask":       torch.stack(attention_mask_list),
        "style_idx":            torch.stack([b["style_idx"]            for b in batch]),
        "dance_retrieval_cond": torch.stack([b["dance_retrieval_cond"] for b in batch]),
        "music_features":       torch.stack([
            torch.tensor(b["music_features"], dtype=torch.float32)
            if not isinstance(b["music_features"], torch.Tensor)
            else b["music_features"]
            for b in batch
        ]),
        "base_name":            [b["base_name"] for b in batch],
    }