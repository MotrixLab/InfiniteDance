# 💃💃InfiniteDance: Scalable 3D Dance Generation Towards in-the-wild Generalization（ECCV 2026）💃💃

[![ECCV 2026](https://img.shields.io/badge/ECCV-2026-3b5998.svg)](https://arxiv.org/abs/2603.13375)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/2603.13375)
[![Project Page](https://img.shields.io/badge/Project-Homepage-008080?logo=googlechrome&logoColor=white)](https://infinitedance.github.io/#/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data%20%26%20Weights-ffc107?labelColor=yellow)](https://huggingface.co/huuuuuuuuu/InfiniteDance)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)



**InfiniteDance** is a framework for scalable 3D music-to-dance generation with high-quality in-the-wild generalization.

---

## 📂 Repository Structure

```text
InfiniteDance
├── All_LargeDanceAR/              # Main generation module
├── DanceVQVAE/                    # VQ-VAE for motion quantization (follows MoMask)
└── InfiniteDanceData/             # Dataset directory (Should be placed at root)
    ├── dance/                     # Motion tokens (.npy)
    ├── music/                     # Music features (.npy)
    ├── partition/                 # Data splits (train/val/test)
    └── styles/                    # Style metadata

```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone git@github.com:MotrixLab/InfiniteDance.git

cd InfiniteDance

# Install dependencies
pip install -r requirements.txt

```

---

## 📥 Downloads (Data & Weights)

All weights and data are hosted on Hugging Face:
**[🤗 huuuuuuuuu/InfiniteDance](https://huggingface.co/huuuuuuuuu/InfiniteDance)**

The HF layout mirrors this repo — download into the repo root and extract the tarballs in place.

### File map (HF → local)

| File on HF | Size | Place at (relative to repo root) |
|---|---|---|
| `models/checkpoints/dance_vqvae.pth` (3-layer Residual VQ-VAE) | 586 MB | `All_LargeDanceAR/models/checkpoints/dance_vqvae.pth` |
| `models/checkpoints/args.json` | 2 KB | `All_LargeDanceAR/models/checkpoints/args.json` |
| `output/exp_m2d_infinitedance/best_model_stage2.pt` | 2.15 GB | `All_LargeDanceAR/output/exp_m2d_infinitedance/best_model_stage2.pt` |
| `InfiniteDanceData/dance/alldata_new_joint_vecs264/meta/{Mean,Std}.npy` | 2 KB ea | same path under repo root |
| `InfiniteDanceData/DanceVQVAE/body_models/smpl/*` | 40 MB | same path under repo root |
| `InfiniteDanceData/partition/*.txt` | <1 MB | same path under repo root |
| `InfiniteDanceData/styles/all_style_map.json` | 0.5 MB | same path under repo root |
| `InfiniteDanceData/Infinite_MotionTokens_512x1024_3layer_cleandata.tar.gz` | 15 MB | extract → `InfiniteDanceData/dance/Infinite_MotionTokens_512x1024_3layer_cleandata/` |
| `InfiniteDanceData/muq_features_test_infinitedance.tar.gz` | 2.6 GB | extract → `InfiniteDanceData/music/muq_features/test_infinitedance/` |
| `InfiniteDanceData/musicfeature_55_allmusic_pure.tar.gz` | 3.0 GB | extract → `InfiniteDanceData/music/musicfeature_55_allmusic_pure/` |
| `InfiniteDanceData/retrieval_s192_l384_style.tar.gz` | 839 MB | extract → `InfiniteDanceData/dance/retrieval_s192_l384_style/` |
| `InfiniteDanceData/alldata_new_joint_vecs264_ft_balanced.tar.gz` | 8.7 GB | extract → `InfiniteDanceData/dance/alldata_new_joint_vecs264_ft_balanced/` |

> `best_model_stage2.pt` already contains the full LLaMA-3.2-1B backbone — no separate download from Meta needed.

> **Motion features** (`alldata_new_joint_vecs264_ft_balanced/`, 10,870 clips, HumanML3D-style 264-d): an integrated corpus of **InfiniteDance (9,706) + AIST++ (911) + FineDance (156) + Motorica (97)**. Cleaned with Savitzky-Golay smoothing (window 11, polyorder 3) + rule-based artifact/tail removal for low foot-slide and jitter. Enables retrieval-conditioned inference and training from these features.

### One-shot download

```bash
# from the repo root
pip install -U "huggingface_hub[cli]"

# downloads the entire HF repo on top of your local clone — paths match,
# so files land in the right place automatically
huggingface-cli download huuuuuuuuu/InfiniteDance \
    --repo-type model \
    --local-dir . \
    --local-dir-use-symlinks False

# extract the four tarballs in place
cd InfiniteDanceData
mkdir -p dance music/muq_features
tar -xzf Infinite_MotionTokens_512x1024_3layer_cleandata.tar.gz -C dance/
tar -xzf retrieval_s192_l384_style.tar.gz              -C dance/
tar -xzf musicfeature_55_allmusic_pure.tar.gz          -C music/
tar -xzf muq_features_test_infinitedance.tar.gz        -C music/muq_features/
cd ..
```

### Expected layout after download

```text
InfiniteDance/
├── All_LargeDanceAR/
│   ├── models/
│   │   ├── checkpoints/dance_vqvae.pth                # ← 3-layer Residual VQ-VAE
│   │   ├── checkpoints/args.json                      # ← VQ-VAE architecture config
│   │   └── Llama3.2-1B/config.json                    # architecture only
│   └── output/
│       └── exp_m2d_infinitedance/
│           └── best_model_stage2.pt                   # ← main ckpt (incl. LLaMA)
└── InfiniteDanceData/
    ├── dance/
    │   ├── alldata_new_joint_vecs264/meta/{Mean,Std}.npy
    │   ├── Infinite_MotionTokens_512x1024_3layer_cleandata/  # ← extracted (matches released 3-layer RVQVAE)
    │   └── retrieval_s192_l384_style/                 # ← extracted
    ├── music/
    │   ├── muq_features/test_infinitedance/           # ← extracted (MuQ test set)
    │   └── musicfeature_55_allmusic_pure/             # ← extracted (BA metric)
    ├── partition/
    ├── styles/
    └── DanceVQVAE/body_models/smpl/
```

### What you can reproduce with this release

| Task | Status | Notes |
|---|---|---|
| Inference on the released MuQ test set | ✅ | `bash infer.sh` |
| Inference on your own audio (mp3 / wav) | ✅ | via `utils/extract_muq.py` |
| Beat-Align (BA) metric | ✅ | needs `musicfeature_55_allmusic_pure` |
| Retrieval ablations | ✅ | uses `retrieval_s192_l384_style` |
| **FID-k / FID-m / Div-k / Div-m** | ⚠️ partial | requires GT joints (`ourData_smplx_22_smooth_new/`), which are **not yet released**; we will add them in a follow-up upload |
| **Training from scratch** | ✅ | 264-d motion features released as `alldata_new_joint_vecs264_ft_balanced/` (10,870 clips), plus the tokenized version and `Mean.npy` / `Std.npy` |
| **Per-dance SMPL/SMPLX parameters (InfiniteDance clips)** | ❌ not yet released | Only the retrieved AIST++/Motorica clips ship with their own upstream SMPL(X); the ~9.7k InfiniteDance-collected clips' own SMPL fits are **TODO** |

---

## 💃 Usage

### 1. Inference & Reproduction

The model takes per-frame **MuQ embeddings** as input (`(T, 1024)` float32 `.npy`, ~30 fps).
`infer.sh` defaults to the released test set. For your own audio, convert it first:

```bash
cd All_LargeDanceAR
python utils/extract_muq.py --in_dir /path/to/audio --out_dir ../InfiniteDanceData/music/muq_features/my_songs
MUSIC_PATH=../InfiniteDanceData/music/muq_features/my_songs bash infer.sh
```

#### Option A: Quick Start (Recommended)

`infer.sh` runs Inference → tokens-to-SMPL → rendering, with anti-collapse decoding on by default.

```bash
cd All_LargeDanceAR
DATA_ROOT=../InfiniteDanceData \
CHECKPOINT_PATH=./output/exp_m2d_infinitedance/best_model_stage2.pt \
bash infer.sh
```

Common overrides: `GPU_ID`, `PROCESSES_PER_GPU`, `STYLE`, `MUSIC_LENGTH`, `DANCE_LENGTH`, `TEMPERATURE`, `TOP_K`, `TOP_P`, `SEED` (see comments at the top of `infer.sh` for anti-collapse tuning).

#### Option B: Manual Execution

```bash
cd All_LargeDanceAR

python infer_llama_infinitedance.py \
    --music_path ../InfiniteDanceData/music/muq_features/test_infinitedance \
    --checkpoint_path ./output/exp_m2d_infinitedance/best_model_stage2.pt \
    --vqvae_checkpoint_path ./models/checkpoints/dance_vqvae.pth \
    --output_dir ./infer_results \
    --style Popular --music_length 320 --dance_length 288 \
    --temperature 0.8 --top_k 15 --top_p 0.95 --seed 42
```

**Visualization** (only needed after manual inference above):

```bash
# 1. Convert tokens to SMPL joints (.npy)
python ./utils/tokens2smpl.py --npy_dir ./infer_results/dance

# 2. Render joints to video (.mp4)
python ./visualization/render_plot_npy.py --joints_dir ./infer_results/dance/npy/joints

```

### 1.1 Metrics

`metrics.sh` runs FID-k / FID-m / Div-k / Div-m and the official Beat-Align score.

Our reported numbers use `partition/test_eval861.txt` (861 clips) as the canonical
evaluation set — the intersection of clips the model can generate for, that have
GT joints, and that have beat-align features. Beat-Align is reproducible now
(`musicfeature_55_allmusic_pure` is released); FID-k/m and Div-k/m additionally
need the GT joints, which are not yet released (see table above).

```bash
cd All_LargeDanceAR
bash metrics.sh <pred_root> [device_id]
# pred_root e.g. ./infer/dance_<TS>/dance/npy/joints
```

### 2. Training

Two-stage training (stage 1: bridges + adapters, LLM frozen; stage 2: full fine-tune)
is run via DDP. Edit `train.sh` (or pass env vars) and launch:

```bash
cd All_LargeDanceAR

# Default: 4 GPUs, bf16, with regularization (weight_decay=0.10,
# llama_dropout=0.15, cond_drop_prob=0.15)
DATA_ROOT=../InfiniteDanceData bash train.sh

# Other GPU counts
GPUS=0,1 WS=2 DATA_ROOT=../InfiniteDanceData bash train.sh

# Warm-start from a previous stage-2 checkpoint
PREV_CKPT=./output/m2d_llama/<run>/epoch_X_stage2.pt bash train.sh
```

---

## 📝 Citation

If you use this code or dataset in your research, please cite our work:

```bibtex
@misc{li2026infinitedancescalable3ddance,
      title={InfiniteDance: Scalable 3D Dance Generation Towards in-the-wild Generalization}, 
      author={Ronghui Li and Zhongyuan Hu and Li Siyao and Youliang Zhang and Haozhe Xie and Mingyuan Zhang and Jie Guo and Xiu Li and Ziwei Liu},
      year={2026},
      eprint={2603.13375},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.13375}, 
}

```

`alldata_new_joint_vecs264_ft_balanced` also integrates motion capture from three external
datasets (AIST++, FineDance, Motorica). If you use this data, please also cite the original
sources:

```bibtex
@inproceedings{li2021aistplusplus,
  author    = {Ruilong Li and Shan Yang and David A. Ross and Angjoo Kanazawa},
  title     = {{AI} Choreographer: Music Conditioned 3D Dance Generation with {AIST++}},
  booktitle = {2021 IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {13381--13392},
  publisher = {IEEE},
  year      = {2021},
  doi       = {10.1109/ICCV48922.2021.01315},
}

@inproceedings{li2023finedance,
  title     = {FineDance: A Fine-grained Choreography Dataset for 3D Full Body Dance Generation},
  author    = {Li, Ronghui and Zhao, Junfan and Zhang, Yachao and Su, Mingyang and Ren, Zeping and Zhang, Han and Tang, Yansong and Li, Xiu},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2023},
}

@article{alexanderson2023listen,
  title     = {Listen, Denoise, Action! Audio-Driven Motion Synthesis with Diffusion Models},
  author    = {Alexanderson, Simon and Nagy, Rajmund and Beskow, Jonas and Henter, Gustav Eje},
  year      = {2023},
  publisher = {ACM},
  volume    = {42},
  number    = {4},
  doi       = {10.1145/3592458},
  journal   = {ACM Trans. Graph.},
  articleno = {44},
  numpages  = {20},
}
```



