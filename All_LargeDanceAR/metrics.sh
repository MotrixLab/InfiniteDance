#!/bin/bash
# Metrics evaluation script for generated dance sequences
# Usage: ./metrics.sh <base_path> [device_id]

# Set default values
BASE_PATH=${1:-"./output/infer/dance"}
device=${2:-0}

export CUDA_VISIBLE_DEVICES=$device

echo "Evaluating metrics for: $BASE_PATH"

# Convert tokens to SMPL format (if needed)
# python ./utils/tokens2smpl.py --npy_dir ${BASE_PATH}

# Evaluate metrics
python ./metrics/metrics_largedata_v2.py --pred_root ${BASE_PATH}/npy/joints

echo "Metrics evaluation completed for: $BASE_PATH"