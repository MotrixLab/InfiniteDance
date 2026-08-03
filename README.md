# 💃💃InfiniteDance: Scalable 3D Dance Generation Towards in-the-wild Generalization（ECCV 2026）💃💃

[![ECCV 2026](https://img.shields.io/badge/ECCV-2026-3b5998.svg)](https://arxiv.org/abs/2603.13375)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/2603.13375)
[![Project Page](https://img.shields.io/badge/Project-Homepage-008080?logo=googlechrome&logoColor=white)](https://infinitedance.github.io/#/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data%20%26%20Weights-ffc107?labelColor=yellow)](https://huggingface.co/huuuuuuuuu/InfiniteDance)
[![License: Non-Commercial](https://img.shields.io/badge/license-Non--Commercial-red.svg)](LICENSE)



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
| `All_LargeDanceAR/models/checkpoints/dance_vqvae.pth` (3-layer Residual VQ-VAE) | 586 MB | same path under repo root |
| `All_LargeDanceAR/models/checkpoints/args.json` | 2 KB | same path under repo root |
| `All_LargeDanceAR/output/exp_m2d_infinitedance/best_model_stage2.pt` | 2.15 GB | same path under repo root |
| `All_LargeDanceAR/models/retrievalnet/retrievalnet_audio55_motion264.ckpt` | 71 MB | same path under repo root |
| `InfiniteDanceData/dance/alldata_new_joint_vecs264/meta/{Mean,Std}.npy` | 2 KB ea | same path under repo root |
| `InfiniteDanceData/DanceVQVAE/body_models/smpl/*` | 40 MB | same path under repo root |
| `InfiniteDanceData/partition/*.txt` | <1 MB | same path under repo root |
| `InfiniteDanceData/styles/all_style_map.json` | 0.5 MB | same path under repo root |
| `InfiniteDanceData/Infinite_MotionTokens_512x1024_3layer_cleandata.tar.gz` | 15 MB | extract → `InfiniteDanceData/dance/Infinite_MotionTokens_512x1024_3layer_cleandata/` |
| `InfiniteDanceData/muq_features_test_infinitedance.tar.gz` | 2.6 GB | extract → `InfiniteDanceData/music/muq_features/test_infinitedance/` |
| `InfiniteDanceData/musicfeature_55_allmusic_pure.tar.gz` | 3.0 GB | extract → `InfiniteDanceData/music/musicfeature_55_allmusic_pure/` |
| `InfiniteDanceData/retrieval_s192_l384_style.tar.gz` | 839 MB | extract → `InfiniteDanceData/dance/retrieval_s192_l384_style/` |
| `InfiniteDanceData/alldata_new_joint_vecs264_ft_balanced.tar.gz` | 8.7 GB | extract → `InfiniteDanceData/dance/alldata_new_joint_vecs264_ft_balanced/` |
| `InfiniteDanceData/dance/retrievalnet_motion_embeddings.npz` | ~60 MB | same path under repo root |
| `InfiniteDanceData/dance/evaluation_features_train8235.npz` | ~4 MB | same path under repo root |
| `InfiniteDanceData/test_eval861_joint_vecs264.tar` | ~1.5 GB | extract → `InfiniteDanceData/dance/test_eval861_joint_vecs264/` |
| `InfiniteDanceData/infinitedance_smplx_smooth.tar` | ~4.2 GB | extract → `InfiniteDanceData/dance/infinitedance_smplx_smooth/` |

> For inference, `best_model_stage2.pt` already contains the full LLaMA-3.2-1B backbone — no separate download from Meta is needed.

> **Motion features** (`alldata_new_joint_vecs264_ft_balanced/`, 10,870 clips, HumanML3D-style 264-d): an integrated corpus of **InfiniteDance (9,706) + AIST++ (911) + FineDance (156) + Motorica (97)**. Cleaned with Savitzky-Golay smoothing (window 11, polyorder 3) + rule-based artifact/tail removal for low foot-slide and jitter. Enables retrieval-conditioned inference and training from these features.

> **SMPL-X format:** each `.pkl` is a `joblib`-serialized dictionary containing
> `body_pose`, `global_orient`, and `transl`; load it with `joblib.load`.

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

# extract the release tarballs in place
cd InfiniteDanceData
mkdir -p dance music/muq_features
tar -xzf Infinite_MotionTokens_512x1024_3layer_cleandata.tar.gz -C dance/
tar -xzf retrieval_s192_l384_style.tar.gz              -C dance/
tar -xzf alldata_new_joint_vecs264_ft_balanced.tar.gz  -C dance/
tar -xf test_eval861_joint_vecs264.tar                 -C dance/
tar -xf infinitedance_smplx_smooth.tar                 -C dance/
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
│   │   ├── retrievalnet/retrievalnet_audio55_motion264.ckpt
│   │   └── Llama3.2-1B/config.json                    # architecture only
│   └── output/
│       └── exp_m2d_infinitedance/
│           └── best_model_stage2.pt                   # ← main ckpt (incl. LLaMA)
└── InfiniteDanceData/
    ├── dance/
    │   ├── alldata_new_joint_vecs264/meta/{Mean,Std}.npy
    │   ├── alldata_new_joint_vecs264_ft_balanced/     # ← released training motion features
    │   ├── Infinite_MotionTokens_512x1024_3layer_cleandata/  # ← extracted (matches released 3-layer RVQVAE)
    │   ├── retrieval_s192_l384_style/                 # ← released test cache
    │   ├── retrievalnet_motion_embeddings.npz         # ← live RetrievalNet corpus embeddings
    │   ├── evaluation_features_train8235.npz          # ← FID/Div GT features
    │   ├── test_eval861_joint_vecs264/                # ← canonical test GT
    │   └── infinitedance_smplx_smooth/                # ← per-clip SMPL-X parameters
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
| Inference on your own audio (mp3 / wav) | ✅ | MuQ extraction plus the released live RetrievalNet pipeline below |
| Beat-Align (BA) metric | ✅ | needs `musicfeature_55_allmusic_pure` |
| Retrieval ablations | ✅ | uses `retrieval_s192_l384_style` |
| **FID-k / FID-m / Div-k / Div-m** | ✅ | precomputed GT kinetic/manual features and canonical split are released; test GT vectors are also included |
| **Training from scratch** | ✅ with Llama access | motion features/tokens are released; obtain the official Llama-3.2-1B weights from Meta before fresh training |
| **Per-dance SMPL-X parameters (InfiniteDance clips)** | ✅ | 9,748 project-owned fits in `infinitedance_smplx_smooth/` |

Inference on the released test music uses the faster precomputed
`retrieval_s192_l384_style` JSON cache. RetrievalNet source, checkpoint,
55-d audio preprocessing, and motion embeddings are also released for new audio.

---

## 💃 Usage

### 1. Inference & Reproduction

The model takes per-frame **MuQ embeddings** as input (`(T, 1024)` float32 `.npy`, ~30 fps).
`infer.sh` defaults to the released test set. For your own audio, convert it first:

```bash
cd All_LargeDanceAR
python utils/extract_muq.py --in_dir /path/to/audio --out_dir ../InfiniteDanceData/music/muq_features/my_songs

# Build a matching retrieval cache for every new song (repeat per file).
python RetrievalNet/retrieve.py /path/to/audio/song.wav \
  ../InfiniteDanceData/dance/retrieval_s192_l384_style/song.json

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

Predictions are selected with `partition/test_eval861.txt` (861 clips), the
intersection with generation outputs, GT motion, and beat features. Following
the reported protocol, FID/Div compare those predictions with the clean training
distribution selected by `partition/All_cleandata_train.txt`; `metrics.sh` uses
the released kinetic/manual GT features directly. The corresponding 861 raw
264-d GT vectors are released for inspection and independent evaluation.

```bash
cd All_LargeDanceAR
bash metrics.sh <pred_root> [device_id]
# pred_root e.g. ./infer/dance_<TS>/dance/npy/joints
```

### 2. Training

Two-stage training (stage 1: bridges + adapters, LLM frozen; stage 2: full fine-tune)
is run via DDP. The InfiniteDance inference checkpoint contains the Llama
backbone, but a fresh training run must start from the official Meta
Llama-3.2-1B weights. Place those files locally and pass `LLAMA_DIR`:

```bash
cd All_LargeDanceAR

# Default: 4 GPUs, bf16, with regularization (weight_decay=0.10,
# llama_dropout=0.15, cond_drop_prob=0.15)
LLAMA_DIR=/path/to/Llama-3.2-1B DATA_ROOT=../InfiniteDanceData bash train.sh

# Other GPU counts
GPUS=0,1 WS=2 DATA_ROOT=../InfiniteDanceData bash train.sh

# Warm-start from a previous stage-2 checkpoint
PREV_CKPT=./output/m2d_llama/<run>/epoch_X_stage2.pt bash train.sh
```

---

## License

InfiniteDance project-owned code, checkpoints, model weights, annotations, and
data resources are available for **non-commercial academic research, education,
and evaluation only** under the [InfiniteDance Non-Commercial Research
License](LICENSE). Commercial use, including internal commercial research,
requires prior written permission. Third-party components and incorporated
datasets remain governed by their original terms.

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
