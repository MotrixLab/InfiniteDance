# 💃💃InfiniteDance: Scalable 3D Dance Generation Towards in-the-wild Generalization💃💃

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org)
[![Project Page](https://img.shields.io/badge/Project-Homepage-008080?logo=googlechrome&logoColor=white)](https://infinitedance.github.io/#/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data%20%26%20Weights-ffc107?labelColor=yellow)](https://huggingface.co/huuuuuuuuu/InfiniteDance)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Status**: 🚧 **Repository under active development.** We are continuously adding more data and features. More data and features are coming soon!



## 🚀 Overview

**InfiniteDance** is a comprehensive framework for scalable 3D music-to-dance generation, designed for high-quality generalization in-the-wild. 

---

## 📂 Repository Structure

```text
InfiniteDance├── All_LargeDanceAR/              # Main LLM generation module
│   ├── models/                    # Model architectures and wrappers
│   │   ├── checkpoints/           # VQVAE and other model weights
│   │   └── Llama3.2-1B/           # Base LLaMA model (Download from HF)
│   ├── RetrievalNet/              # Retrieval-Augmented Generation (RAG) network
│   │   └── checkpoints/           # RetrievalNet pre-trained weights
│   ├── output/                    # Training outputs and fine-tuned weights
│   ├── utils/                     # Token-to-SMPL conversion and utilities
│   ├── visualization/             # Rendering and video generation tools
│   ├── train_infinitedance_start.py # Main training entry point
│   ├── infer_llama_infinitedance.py # Main inference script
│   └── infer.sh                   # All-in-one inference script
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

* **VQ-VAE Weights**: `All_LargeDanceAR/models/checkpoints/`
* **RetrievalNet Weights**: `All_LargeDanceAR/RetrievalNet/checkpoints/`
* **InfiniteDance Fine-tuned Weights**: `All_LargeDanceAR/output/exp_m2d_infinitedance/`
* **Base LLM**: Download [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) and place it in `All_LargeDanceAR/models/Llama3.2-1B/`.

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

Edit `infer.sh` in `All_LargeDanceAR` to set your paths, then run:

```bash
cd All_LargeDanceAR
chmod +x infer.sh
./infer.sh

```

#### Option B: Manual Execution

To generate dance tokens manually from music features:

```bash
cd All_LargeDanceAR

python infer_llama_infinitedance.py \
    --music_path <your path>/InfiniteDanceData/music/muq_features/test_infinitedance \
    --checkpoint_path <your path>/All_LargeDanceAR/output/exp_m2d_infinitedance/best_model_stage2.pt \
    --vqvae_checkpoint_path <your path>/All_LargeDanceAR/models/checkpoints/dance_vqvae.pth \
    --output_dir <your path>/All_LargeDanceAR/infer_results \
    --style Popular \
    --dance_length 288

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

To evaluate metrics, make sure you are in `All_LargeDanceAR`:

```bash
cd All_LargeDanceAR
./metrics.sh <base_path> [device_id]

```

### 2. Training

The training process is divided into two stages:

* **Stage 1**: Train the bridge module and adapters while freezing the LLM backbone.
* **Stage 2**: Full-parameter fine-tuning of the entire system.

```bash
cd All_LargeDanceAR

# Start Training
python train_infinitedance_start.py \
    --dance_dir <your path>/InfiniteDanceData/dance/Infinite_MotionTokens_512_vel_processed \
    --music_dir <your path>/InfiniteDanceData/music/muq_features \
    --vqvae_checkpoint_path <your path>/All_LargeDanceAR/models/checkpoints/dance_vqvae.pth \
    --llama_config_path <your path>/All_LargeDanceAR/models/Llama3.2-1B/config.json \
    --world_size 4 \
    --batch_size 8 \
    --learning_rate1 4e-5 \
    --stage1_epoch 2 \
    --stage2_epoch 50

```

---

## 📝 Citation

If you use this code or dataset in your research, please cite our work:

```bibtex
@article{infinitedance2026,
  title={InfiniteDance: Scalable 3D Dance Generation Towards in-the-wild Generalization},
  author={...},
  journal={arXiv},
  year={2026}
}

```


