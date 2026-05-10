#!/usr/bin/env bash
# InfiniteDance training (DDP, bf16) — single MLP-bridge config with strong regularization.
# Optionally resume from a previous stage-2 checkpoint via PREV_CKPT=...
set -e

# Repo root inferred from this script's location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$REPO_DIR"

GPUS=${GPUS:-0,1,2,3,4,5,6,7}
WS=${WS:-8}
BATCH=${BATCH:-8}
ACCUM=${ACCUM:-1}
S2_LR=${S2_LR:-2e-5}
FREEZE=${FREEZE:-0}

STAGE1_EP=${STAGE1_EP:-0}
STAGE2_EP=${STAGE2_EP:-30}
TARGET_ACC=${TARGET_ACC:-0.99}
PRECISION=${PRECISION:-bf16}
WD=${WD:-0.10}
WARMUP=${WARMUP:-50}

LLAMA_DROPOUT=${LLAMA_DROPOUT:-0.15}
COND_DROP_PROB=${COND_DROP_PROB:-0.15}

# Optional warm-start checkpoint (set to "" to train from scratch)
PREV_CKPT=${PREV_CKPT:-}

DATA_ROOT=${DATA_ROOT:-../InfiniteDanceData}

export CUDA_VISIBLE_DEVICES=${GPUS}
export MASTER_PORT=${MASTER_PORT:-17871}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

PY=${PY:-python}
STAMP=$(date +%y%m%d_%H%M)
OUT=${OUT:-./output/m2d_llama/infinitedance_${STAMP}}
mkdir -p "${OUT}"

echo "=========================================="
echo "InfiniteDance training"
echo "  GPUs:        ${GPUS}  (ws=${WS})"
echo "  bs/gpu=${BATCH} accum=${ACCUM} effective=$((BATCH*WS*ACCUM))"
echo "  Stage2:      ${STAGE2_EP}ep, freeze=${FREEZE}, lr=${S2_LR}"
echo "  WD:          ${WD}"
echo "  llama_drop:  ${LLAMA_DROPOUT}"
echo "  cond_drop:   ${COND_DROP_PROB}"
echo "  RESUME:      ${PREV_CKPT:-<from scratch>}"
echo "  out:         ${OUT}"
echo "=========================================="

CMD=(${PY} train_v2/train_infinitedance.py
  --out_dir "${OUT}"
  --world_size ${WS} --MASTER_PORT ${MASTER_PORT}
  --music_dir ${DATA_ROOT}/music/muq_features
  --dance_dir ${DATA_ROOT}/dance/Infinite_MotionTokens_512_vel_processed
  --data_split_dir ${DATA_ROOT}/partition
  --style_dir ${DATA_ROOT}/styles/Alldata
  --dancedata All --n_bins 2
  --style_embedding_dim 64
  --vqvae_checkpoint_path ./models/checkpoints/dance_vqvae.pth
  --mean_path ${DATA_ROOT}/dance/alldata_new_joint_vecs264/meta/Mean.npy
  --std_path  ${DATA_ROOT}/dance/alldata_new_joint_vecs264/meta/Std.npy
  --batch_size ${BATCH}
  --gradient_accumulation_steps ${ACCUM}
  --stage1_epoch ${STAGE1_EP}
  --stage2_epoch ${STAGE2_EP}
  --target_train_acc ${TARGET_ACC}
  --num_train_epochs $((STAGE1_EP + STAGE2_EP))
  --learning_rate  ${S2_LR}
  --learning_rate2 ${S2_LR}
  --weight_decay ${WD}
  --warmup_steps ${WARMUP}
  --label_smoothing 0.0
  --llama_dropout ${LLAMA_DROPOUT}
  --cond_drop_prob ${COND_DROP_PROB}
  --early_token_weight 1.0
  --early_token_count 60
  --freeze_llama_layers ${FREEZE}
  --precision ${PRECISION}
  --num_workers 8
  --save_steps 3000
)

if [ -n "${PREV_CKPT}" ]; then
  CMD+=(--resume_from_checkpoint "${PREV_CKPT}")
fi

"${CMD[@]}"
