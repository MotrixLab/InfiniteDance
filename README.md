# 💃💃InfiniteDance: Scalable 3D Dance Generation Towards in-the-wild Generalization（ECCV 2026）💃💃

[![ECCV 2026](https://img.shields.io/badge/ECCV-2026-3b5998.svg)](https://eccv.ecva.net/virtual/2026/poster/3684)
[![Paper](https://img.shields.io/badge/Paper-ECCV%202026-b31b1b.svg)](https://eccv.ecva.net/virtual/2026/poster/3684)
[![Project Page](https://img.shields.io/badge/Project-Homepage-008080?logo=googlechrome&logoColor=white)](https://huzhongyyuan.github.io/InfiniteDance/)
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
git clone https://github.com/MotrixLab/InfiniteDance.git

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
| `InfiniteDanceData/dance/vqvae_train_20260522/meta/{Mean,Std}.npy` | 1.2 KB ea | same path under repo root |
| `InfiniteDanceData/DanceVQVAE/body_models/smpl/*` | 40 MB | same path under repo root |
| `InfiniteDanceData/partition/*.txt` | <1 MB | same path under repo root |
| `InfiniteDanceData/styles/all_style_map.json` | 0.5 MB | same path under repo root |
| `InfiniteDanceData/styles_Alldata_eccv.tar.gz` | <1 MB | extract → `InfiniteDanceData/styles/Alldata/` |
| `InfiniteDanceData/muq_features_trainval_*.tar.gz` | see training manifest | extract each shard → `InfiniteDanceData/music/muq_features/` |
| `InfiniteDanceData/release_training_manifest.json` | file hashes and availability | same path under repo root |
| `InfiniteDanceData/partition/alldata_mbr_s100_l384.json` | 8.9 MB | original 384-frame RetrievalNet windows |
| `InfiniteDanceData/partition/alldata_mbr_s100_l384_release.json` | see repository | usable paired windows in the public archives |
| `InfiniteDanceData/partition/retrieval_windows_release_audit.json` | see repository | excluded windows and their reasons |
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
pip install -U huggingface_hub
hf download huuuuuuuuu/InfiniteDance --repo-type model --local-dir . --exclude "README.md" "LICENSE"
python prepare_release_data.py

# Also unpack the training music shards and per-clip style JSONs:
python prepare_release_data.py --training
python check_release_data.py --report training_data_check.json
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
    │   ├── vqvae_train_20260522/meta/{Mean,Std}.npy
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
| **Training from scratch** | Cleaned active lists | 10,154 TRAIN / 415 VAL clips have complete source file sets; download all supplementary MuQ shards and styles, then run the check below. Official Llama-3.2-1B access is required |
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

`metrics.sh` evaluates one selected sample for every identity in
`partition/test_eval861.txt` (861 clips), using the released kinetic/manual GT
features selected by `partition/All_cleandata_train.txt` (8,235 available GT
feature rows). Missing predictions and duplicate identities are errors. Extra
predictions and unselected sample indices are listed in the JSON report.

```bash
cd All_LargeDanceAR
bash metrics.sh /absolute/path/to/dance/npy/joints
# CPU parallelism and an explicit report destination:
WORKERS=4 METRICS_REPORT=/path/to/run/metrics.json bash metrics.sh /path/to/joints
# An explicitly partial diagnostic, never a complete 861-clip result:
bash metrics.sh /path/to/joints --allow-partial
```

The scorer recomputes features from decoded world-space `(T,22,3)` joints and
does not reuse feature directories left by earlier evaluations. FID/Div and
Beat-Align use the same first 1,024 frames by default (`MAX_FRAMES=0` uses the
full sequence). Generated names such as `dance_sample_0_15_Ballet119.npy` map to
the original music identity; `--sample-index` selects a single replicate.
Missing beat features fail the run. Undefined beat scores are reported per
sample and produce a nonzero exit, rather than a silently reduced average.

The kinetic/manual extractors, GT normalization, pairwise diversity and
Gaussian beat-matching formulas retain the released definitions. The feature
extractor's historical 60-Hz convention is recorded explicitly; this repair
does not silently reinterpret it as 30 Hz. JSON output includes sample IDs,
frame limits, input hashes and scorer hashes. The previous standalone
`metrics_largedata_v2.py` and `beat_align_score_joints.py` remain historical
utilities; the supported entry is `metrics.sh` / `evaluate_release.py`.
Full-paper numerical reproduction is not established by the CPU regression
tests; changing decoding statistics or sample coverage requires a fresh result.

### 2. Training

The two-stage generator freezes the Llama backbone in stage 1 and fine-tunes it
in stage 2. Fresh training requires the official Meta Llama-3.2-1B weights;
the released generation checkpoint remains usable for inference.

The active generator lists contain **10,154 TRAIN and 415 VAL clips** after
removing 669 TRAIN and 21 VAL entries with missing required files. This is a
cleaned subset of the historical release, not a claim of original full-corpus
coverage. Original lists are archived under
`InfiniteDanceData/legacy/20261106_original_splits/`; every removed entry is
recorded in `partition/removed_incomplete_samples_20261106.json`.
The canonical evaluation list and GT feature selection remain unchanged.

The supplementary MuQ shards retain existing source features, including
available evaluation music. `release_training_manifest.json` records each
file's shape and SHA-256. `release_source_audit.json` records the original
source gaps. After the complete download, check the active lists:

```bash
python check_release_data.py --report training_data_check.json
```

After all required files are present:

```bash
cd All_LargeDanceAR
LLAMA_DIR=/path/to/Llama-3.2-1B DATA_ROOT=../InfiniteDanceData bash train.sh
# Two GPUs; extra CLI arguments are forwarded to the trainer.
GPUS=0,1 WS=2 LLAMA_DIR=/path/to/Llama-3.2-1B \
  DATA_ROOT=../InfiniteDanceData bash train.sh --batch_size 16
```

The generator reads `All_train.txt` for training and `All_val.txt` for validation.
The audit found that the previous public entry used `All_eval.txt` as validation;
that list contains all 861 canonical evaluation clips. The corrected entry
checks train/validation/test disjointness and uses the existing, separate
`All_val.txt`. The canonical evaluation lists and existing checkpoint metrics were preserved;
the historical checkpoint's exact training-list provenance has not been
established by this code audit. Per-clip style JSONs are read from
`styles/Alldata/`; the consolidated style map is used for retrieval.
Missing required inputs raise an error. Retrieval neighbors are selected from
available motion spans in the stored ranking before applying top-k; exclusions
are reported. This fixes queries whose highest-ranked neighbors are missing,
while retaining later valid candidates. The original cache is preserved. These
repairs are not a claim of bitwise replay of historical generated outputs.

File availability and usable default training windows are different counts:
with the released 320-music-frame / 288-token window, 1,465 TRAIN and 34 VAL
clips are too short. Their files and identities remain available. The default
window set therefore uses 8,689 TRAIN / 381 VAL clips (46,107 / 1,576 windows).

### Codec decoding and retraining

The released 512-code, 1024-d, three-layer RVQ checkpoint uses statistics in
`dance/vqvae_train_20260522/meta/`. The earlier public CLI pointed to different
`alldata_new_joint_vecs264/meta/` statistics; those historical files remain
available, but are no longer the default. Both inference entries and token
conversion now use the checkpoint-matched files. The checkpoint configuration
has `vel_decoder=false` and `rot_decoder=false`; their weights equal seeded
initialization, so decoding uses the trained base decoder. Previously the token
converter unconditionally replaced root velocity with the untrained auxiliary
head. Outputs from that old path are not interchangeable with corrected output.

`DanceVQVAE.models.vqvae` now imports the same implementation used by the
generator. The portable training entry replaces the author-machine launcher,
uses disjoint release train/validation lists and saves checkpoints with the
same parameter names. It also corrects the XYZ indexing of foot-joint loss.
Its lists contain 8,869 TRAIN / 415 VAL clips from the public balanced-motion
archive, with missing motion files recorded in `vq_release_split_audit.json`. It is a retraining
recipe, not a reconstruction of the historical private training pool.

```bash
# From the repository root, after downloading/extracting the motion archive:
bash DanceVQVAE/train_dancevq.sh --data-root ./InfiniteDanceData \
  --output ./DanceVQVAE/output/release_rvq --device cuda:0 --batch-size 256
```

A CPU optimization step on two real training clips, validation on two held-out
clips, and checkpoint saving passed. A checkpoint-compatibility check strictly
loaded every released tensor. Two
real motion/token pairs were used to check normalization and decoder behavior;
these checks do not reproduce the paper's motion-generation metrics. Historical
standalone scripts under `DanceVQVAE/scripts/` remain archival experiment code;
use the documented launcher for the released-data retraining path.

### 2.1 RetrievalNet retraining

The released encoder uses **384 frames**, matching the original
`alldata_mbr_s100_l384.json`. The active retraining metadata is
`alldata_mbr_s100_l384_release.json`: unavailable or truncated windows are
excluded, and `retrieval_windows_release_audit.json` records the exclusions.
The original metadata remains published for provenance. The old 600-frame configuration did not match the
released encoder. Both portable dataset configurations now refer to the
available 384-frame windows and resolve paths through `INFINITEDANCE_DATA_ROOT`.
There are 47,090 usable training windows and 4,557 canonical test windows.
Training and retrieval-corpus construction use only `train` windows; test
windows read the separately released canonical test GT directory.

```bash
cd All_LargeDanceAR
python RetrievalNet/train_release.py \
  --data-root ../InfiniteDanceData \
  --output ./output/retrieval_retrain --device cuda:0 --batch-size 256
```

The entry returns real `(name, audio55, motion264)` batches and trains the
released architecture. It is a portable retraining recipe, not a claim of
bitwise replay of the historical checkpoint; all referenced motion/music
files must exist. A CPU one-step run and loader/configuration regression tests
were checked during release repair. The original weights and reported metrics
were not retrained or replaced by that check.

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
