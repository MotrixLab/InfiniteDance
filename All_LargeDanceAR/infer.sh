#!/bin/bash

# =================================================================
# InfiniteDance Inference & Visualization Pipeline
# =================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# --- Configuration Section ---

# Hardware Settings
#

GPU_ID=0
PROCESSES_PER_GPU=2
Root_path="<your path>"   # parent dir of InfiniteDance repo (repo at Root_path/InfiniteDance)

# Input args (!!!!!Modify these paths as needed!!!!!)
# =================================================================

TIMESTAMP=$(date +"%y%m%d_%H%M")
MUSIC_PATH="$Root_path/InfiniteDance/InfiniteDanceData/music/slow/test1"
CHECKPOINT_PATH="$Root_path/InfiniteDance/All_LargeDanceAR/output/exp_m2d_infinitedance/best_model_stage2.pt"
VQVAE_CHECKPOINT_PATH="$Root_path/InfiniteDance/All_LargeDanceAR/models/checkpoints/dance_vqvae.pth"
MEAN_PATH="$Root_path/InfiniteDance/InfiniteDanceData/dance/alldata_new_joint_vecs264/meta/Mean.npy"
STD_PATH="$Root_path/InfiniteDance/InfiniteDanceData/dance/alldata_new_joint_vecs264/meta/Std.npy"
OUTPUT_ROOT="$Root_path/InfiniteDance/All_LargeDanceAR/infer/dance_${TIMESTAMP}"
DANCE_OUTPUT_DIR="${OUTPUT_ROOT}/dance"

# =================================================================

# --- Pipeline Execution ---

echo "=========================================================="
echo "Starting InfiniteDance Pipeline"
echo "Time: ${TIMESTAMP}"
echo "GPU: ${GPU_ID}"
echo "Output Directory: ${OUTPUT_ROOT}"
echo "=========================================================="
cd $Root_path/InfiniteDance/All_LargeDanceAR

# Step 1: Run Inference
echo ""
echo ">>> [Step 1/3] Generating dance tokens (Inference)..."

# Ensure CUDA_VISIBLE_DEVICES is set for the python script if needed, 
# though the script handles GPU placement internally via multiprocessing.
python $Root_path/InfiniteDance/All_LargeDanceAR/infer_llama_infinitedance.py \
  --music_path "$MUSIC_PATH" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --vqvae_checkpoint_path "$VQVAE_CHECKPOINT_PATH" \
  --mean_path "$MEAN_PATH" \
  --std_path "$STD_PATH" \
  --output_dir "$OUTPUT_ROOT" \
  --processes_per_gpu "$PROCESSES_PER_GPU"

# Step 2: Post-processing (Tokens -> SMPL/Joints)
echo ""
echo ">>> [Step 2/3] Converting tokens to SMPL format..."

if [ -d "$DANCE_OUTPUT_DIR" ]; then
    CUDA_VISIBLE_DEVICES=$GPU_ID python $Root_path/InfiniteDance/All_LargeDanceAR/utils/tokens2smpl.py \
      --npy_dir "$DANCE_OUTPUT_DIR"
else
    echo "Error: Dance output directory not found at $DANCE_OUTPUT_DIR"
    exit 1
fi

# Step 3: Visualization (Render Joints to Video)
# Based on tokens2smpl script, it usually creates an 'npy/joints' subfolder
JOINTS_DIR="${DANCE_OUTPUT_DIR}/npy/joints"

echo ""
echo ">>> [Step 3/3] Rendering visualizations..."

if [ -d "$JOINTS_DIR" ]; then
    CUDA_VISIBLE_DEVICES=$GPU_ID python $Root_path/InfiniteDance/All_LargeDanceAR/visualization/render_plot_npy.py \
      --joints_dir "$JOINTS_DIR"
else
    echo "Warning: Joints directory not found at $JOINTS_DIR. Skipping visualization."
fi

echo ""
echo "=========================================================="
echo "Pipeline Completed Successfully!"
echo "All results are saved in: ${OUTPUT_ROOT}"
echo "=========================================================="