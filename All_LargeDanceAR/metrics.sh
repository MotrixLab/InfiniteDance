#!/bin/bash
# Metrics evaluation:
#   FID-k / FID-m / Div-k / Div-m via metrics/metrics_largedata_v2.py
#   BA via metrics/beat_align_score_joints.py
#
# Usage: bash metrics.sh <pred_root> [device_id]
#   pred_root  e.g. ./infer/dance_<TS>/dance/npy/joints
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

PRED_ROOT=${1:-./infer/dance/dance/npy/joints}
DEVICE=${2:-0}
DATA_ROOT=${DATA_ROOT:-../InfiniteDanceData}
GT_ROOT=${GT_ROOT:-${DATA_ROOT}/dance/ourData_smplx_22_smooth_new/new_joint_vecs264_vel}
MUSIC_FEAT_ROOT=${MUSIC_FEAT_ROOT:-${DATA_ROOT}/music/musicfeature_55_allmusic_pure}

export CUDA_VISIBLE_DEVICES=${DEVICE}
export INFINITEDANCE_MUSIC_FEAT_ROOT=${MUSIC_FEAT_ROOT}

PY=${PY:-python}

echo "==> FID / Div"
( cd metrics && ${PY} metrics_largedata_v2.py \
    --pred_root "${PRED_ROOT}" \
    --gt_root "${GT_ROOT}" \
    --max_frames 1024 )

echo ""
echo "==> Beat Align"
( cd metrics && ${PY} -c "
from beat_align_score_joints import calc_ba_score
print('BA =', calc_ba_score('${PRED_ROOT}'))
" )