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

All datasets and pre-trained checkpoints are hosted on Hugging Face. After download, place them in the following locations (relative to the repo root unless you use absolute paths):

**[🤗 Hugging Face CheckPoints: InfiniteDance](https://huggingface.co/huuuuuuuuu/InfiniteDance)**

### 1. Data Setup

Download the `InfiniteDanceData` folder and place it in the repo root:

```bash
# Path: <your path>/InfiniteDance_opensource/InfiniteDanceData

```

### 2. Model Weights Setup

Please place the downloaded weights in their respective directories:

* **VQ-VAE Weights**: `All_LargeDanceAR/models/checkpoints/dance_vqvae.pth`
* **InfiniteDance Fine-tuned Weights**: `All_LargeDanceAR/output/exp_m2d_infinitedance/best_model_stage2.pt`
* **Base LLM**: The released checkpoint already contains the full LLaMA-3.2-1B backbone weights, so you do **not** need to download anything from Meta. We ship the architecture `config.json` in `All_LargeDanceAR/models/Llama3.2-1B/`.

After placement, the expected structure looks like this:

```text
InfiniteDance├── InfiniteDanceData/
│   ├── dance/
│   ├── music/
│   ├── partition/
│   └── styles/
└── All_LargeDanceAR/
    ├── models/
    │   ├── checkpoints/
    │   └── Llama3.2-1B/
    ├── RetrievalNet/
    │   └── checkpoints/
    └── output/
        └── exp_m2d_infinitedance/

```

---

## 💃 Usage

### 1. Inference & Reproduction

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

The default decoding hyper-parameters used to obtain the reported metrics are:

| env var | default | meaning |
|---|---|---|
| `SAFE_REP_PENALTY` | `1.20` | repetition logit penalty |
| `SAFE_MAX_REP_S0`  | `8`    | max consecutive repeats on slot-0 codebook |
| `SAFE_MAX_REP_OTHER` | `16` | max consecutive repeats on slots 1 / 2 |
| `SAFE_NGRAM_S0`    | `5`    | n-gram block size on slot-0 |
| `SAFE_TEMP_BOOST`  | `1.8`  | temperature boost when collapse triggers |

Other useful overrides: `GPU_ID`, `PROCESSES_PER_GPU`, `STYLE`, `MUSIC_LENGTH`,
`DANCE_LENGTH`, `TEMPERATURE`, `TOP_K`, `TOP_P`, `SEED`.

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

# Default: 8 GPUs, bf16, with regularization (weight_decay=0.10,
# llama_dropout=0.15, cond_drop_prob=0.15)
GPUS=0,1,2,3,4,5,6,7 WS=8 DATA_ROOT=../InfiniteDanceData bash train.sh

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


