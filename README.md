# 💃💃InfiniteDance: Scalable 3D Dance Generation Towards in-the-wild Generalization💃💃

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/2603.13375)
[![Project Page](https://img.shields.io/badge/Project-Homepage-008080?logo=googlechrome&logoColor=white)](https://infinitedance.github.io/#/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data%20%26%20Weights-ffc107?labelColor=yellow)](https://huggingface.co/huuuuuuuuu/InfiniteDance)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Status**: 🚧 **Repository under active development.** We are continuously adding more data and features. More data and features are coming soon!



## 🚀 Overview

**InfiniteDance** is a comprehensive framework for scalable 3D music-to-dance generation, designed for high-quality generalization in-the-wild. 

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

The HF repo layout mirrors this repo exactly — every file's path on HF is
where it should live locally. The only step is to download into the repo
root and extract the tarballs in place.

### File map (HF → local)

| File on HF | Size | Place at (relative to repo root) |
|---|---|---|
| `models/checkpoints/dance_vqvae.pth` | 462 MB | `All_LargeDanceAR/models/checkpoints/dance_vqvae.pth` |
| `output/exp_m2d_infinitedance/best_model_stage2.pt` | 2.3 GB | `All_LargeDanceAR/output/exp_m2d_infinitedance/best_model_stage2.pt` |
| `InfiniteDanceData/dance/alldata_new_joint_vecs264/meta/{Mean,Std}.npy` | 2 KB ea | same path under repo root |
| `InfiniteDanceData/DanceVQVAE/body_models/smpl/*` | 40 MB | same path under repo root |
| `InfiniteDanceData/partition/*.txt` | <1 MB | same path under repo root |
| `InfiniteDanceData/styles/all_style_map.json` | 0.5 MB | same path under repo root |
| `InfiniteDanceData/Infinite_MotionTokens_512_vel_processed.tar.gz` | 14 MB | extract → `InfiniteDanceData/dance/Infinite_MotionTokens_512_vel_processed/` |
| `InfiniteDanceData/muq_features_test_infinitedance.tar.gz` | 2.6 GB | extract → `InfiniteDanceData/music/muq_features/test_infinitedance/` |
| `InfiniteDanceData/musicfeature_55_allmusic_pure.tar.gz` | 3.0 GB | extract → `InfiniteDanceData/music/musicfeature_55_allmusic_pure/` |
| `InfiniteDanceData/retrieval_s192_l384_style.tar.gz` | 839 MB | extract → `InfiniteDanceData/dance/retrieval_s192_l384_style/` |

> The released `best_model_stage2.pt` **already contains the full LLaMA-3.2-1B
> backbone**, so you do *not* need to download anything from Meta. We ship
> the architecture `config.json` in `All_LargeDanceAR/models/Llama3.2-1B/`
> for completeness.

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
tar -xzf Infinite_MotionTokens_512_vel_processed.tar.gz -C dance/
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
│   │   ├── checkpoints/dance_vqvae.pth                # ← VQ-VAE
│   │   └── Llama3.2-1B/config.json                    # architecture only
│   └── output/
│       └── exp_m2d_infinitedance/
│           └── best_model_stage2.pt                   # ← main ckpt (incl. LLaMA)
└── InfiniteDanceData/
    ├── dance/
    │   ├── alldata_new_joint_vecs264/meta/{Mean,Std}.npy
    │   ├── Infinite_MotionTokens_512_vel_processed/   # ← extracted
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
| **Training from scratch** | ⚠️ partial | requires the full 264-d motion features (`alldata_new_joint_vecs264/`), **not yet released**. Only `Mean.npy` / `Std.npy` and the tokenized version (`Infinite_MotionTokens_512_vel_processed/`) are provided so far |

---

## 💃 Usage

### 1. Inference & Reproduction

The model takes per-frame **MuQ embeddings** as input (`(T, 1024)` float32
`.npy`, ~30 frames per second). Two ways to provide them:

* **Use the released test set** — download `muq_features_test_infinitedance.tar.gz`
  from Hugging Face and extract it; this is what `infer.sh` defaults to.
* **Use your own audio** — convert wav / mp3 to MuQ embeddings first:

  ```bash
  cd All_LargeDanceAR
  python utils/extract_muq.py \
      --in_dir  /path/to/your_audio_dir \
      --out_dir ../InfiniteDanceData/music/muq_features/my_songs
  ```

  Then point `infer.sh` at the new directory:

  ```bash
  MUSIC_PATH=../InfiniteDanceData/music/muq_features/my_songs bash infer.sh
  ```

You can run the full inference pipeline (Generation → Post-processing → Visualization) using the provided shell script or by running the python scripts manually.

#### Option A: Quick Start (Recommended)

`infer.sh` runs Inference → tokens-to-SMPL → optional rendering, with
anti-collapse decoding enabled by default.

```bash
cd All_LargeDanceAR
DATA_ROOT=../InfiniteDanceData \
CHECKPOINT_PATH=./output/exp_m2d_infinitedance/best_model_stage2.pt \
bash infer.sh
```

Common overrides: `GPU_ID`, `PROCESSES_PER_GPU`, `STYLE`, `MUSIC_LENGTH`,
`DANCE_LENGTH`, `TEMPERATURE`, `TOP_K`, `TOP_P`, `SEED`. Anti-collapse
decoding is enabled by default; see the comments at the top of `infer.sh`
to tune it.

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

**Visualization Pipeline**:
If you ran the manual inference above, proceed to visualize the results:

```bash
# 1. Convert tokens to SMPL joints (.npy)
python ./utils/tokens2smpl.py --npy_dir ./infer_results/dance

# 2. Render joints to video (.mp4)
python ./visualization/render_plot_npy.py --joints_dir ./infer_results/dance/npy/joints

```

### 1.1 Metrics

`metrics.sh` runs FID-k / FID-m / Div-k / Div-m and the official Beat-Align score.

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


