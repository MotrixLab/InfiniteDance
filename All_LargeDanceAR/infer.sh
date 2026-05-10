#!/bin/bash
# =================================================================
# InfiniteDance Inference & Visualization Pipeline
#
# Anti-collapse decoding (recommended defaults; override via env if needed):
#   SAFE_REP_PENALTY=1.20    repetition logit penalty
#   SAFE_MAX_REP_S0=8        max consecutive repeats on slot-0 codebook
#   SAFE_MAX_REP_OTHER=16    max consecutive repeats on slots 1/2
#   SAFE_NGRAM_S0=5          n-gram block size on slot-0
#   SAFE_TEMP_BOOST=1.8      sampling temperature boost when collapse triggers
# =================================================================
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

# --- Hardware ---
GPU_ID=${GPU_ID:-0}
PROCESSES_PER_GPU=${PROCESSES_PER_GPU:-2}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${GPU_ID}}

# --- Anti-collapse decoding defaults ---
export SAFE_REP_PENALTY=${SAFE_REP_PENALTY:-1.20}
export SAFE_MAX_REP_S0=${SAFE_MAX_REP_S0:-8}
export SAFE_MAX_REP_OTHER=${SAFE_MAX_REP_OTHER:-16}
export SAFE_NGRAM_S0=${SAFE_NGRAM_S0:-5}
export SAFE_TEMP_BOOST=${SAFE_TEMP_BOOST:-1.8}

# --- Paths (override via env vars or edit here) ---
DATA_ROOT=${DATA_ROOT:-../InfiniteDanceData}
TIMESTAMP=$(date +"%y%m%d_%H%M")

MUSIC_PATH=${MUSIC_PATH:-${DATA_ROOT}/music/muq_features/test_infinitedance}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-./output/exp_m2d_infinitedance/best_model_stage2.pt}
VQVAE_CHECKPOINT_PATH=${VQVAE_CHECKPOINT_PATH:-./models/checkpoints/dance_vqvae.pth}
MEAN_PATH=${MEAN_PATH:-${DATA_ROOT}/dance/alldata_new_joint_vecs264/meta/Mean.npy}
STD_PATH=${STD_PATH:-${DATA_ROOT}/dance/alldata_new_joint_vecs264/meta/Std.npy}

OUTPUT_ROOT=${OUTPUT_ROOT:-./infer/dance_${TIMESTAMP}}
DANCE_OUTPUT_DIR="${OUTPUT_ROOT}/dance"

PY=${PY:-python}

# --- Sampling defaults ---
STYLE=${STYLE:-Popular}
MUSIC_LENGTH=${MUSIC_LENGTH:-320}
DANCE_LENGTH=${DANCE_LENGTH:-288}
TEMPERATURE=${TEMPERATURE:-0.8}
TOP_K=${TOP_K:-15}
TOP_P=${TOP_P:-0.95}
NUM_SAMPLES=${NUM_SAMPLES:-1}
SEED=${SEED:-42}

echo "=========================================================="
echo "InfiniteDance inference (anti-collapse strict)"
echo "  ckpt:        ${CHECKPOINT_PATH}"
echo "  music_path:  ${MUSIC_PATH}"
echo "  GPUs:        ${CUDA_VISIBLE_DEVICES} (procs/gpu=${PROCESSES_PER_GPU})"
echo "  output:      ${OUTPUT_ROOT}"
echo "  rep_penalty=${SAFE_REP_PENALTY} max_rep_s0=${SAFE_MAX_REP_S0} ngram_s0=${SAFE_NGRAM_S0} temp_boost=${SAFE_TEMP_BOOST}"
echo "=========================================================="

# Step 1: Generate dance tokens
echo ""
echo ">>> [Step 1/3] Generating dance tokens..."
${PY} infer_llama_infinitedance.py \
  --music_path           "${MUSIC_PATH}" \
  --checkpoint_path      "${CHECKPOINT_PATH}" \
  --vqvae_checkpoint_path "${VQVAE_CHECKPOINT_PATH}" \
  --mean_path            "${MEAN_PATH}" \
  --std_path             "${STD_PATH}" \
  --output_dir           "${OUTPUT_ROOT}" \
  --style                "${STYLE}" \
  --music_length         "${MUSIC_LENGTH}" \
  --dance_length         "${DANCE_LENGTH}" \
  --temperature          "${TEMPERATURE}" \
  --top_k                "${TOP_K}" \
  --top_p                "${TOP_P}" \
  --num_samples          "${NUM_SAMPLES}" \
  --seed                 "${SEED}" \
  --processes_per_gpu    "${PROCESSES_PER_GPU}"

# Step 2: tokens → SMPL / joints
echo ""
echo ">>> [Step 2/3] Converting tokens to SMPL/joints..."
if [ -d "${DANCE_OUTPUT_DIR}" ]; then
  ${PY} utils/tokens2smpl.py \
    --npy_dir "${DANCE_OUTPUT_DIR}" \
    --checkpoint_path "${VQVAE_CHECKPOINT_PATH}" \
    --mean_path "${MEAN_PATH}" --std_path "${STD_PATH}"
else
  echo "Error: ${DANCE_OUTPUT_DIR} not found" >&2; exit 1
fi

# Step 3: render
JOINTS_DIR="${DANCE_OUTPUT_DIR}/npy/joints"
echo ""
echo ">>> [Step 3/3] Rendering visualizations..."
if [ -d "${JOINTS_DIR}" ] && [ -f "${SCRIPT_DIR}/visualization/render_plot_npy.py" ]; then
  ${PY} visualization/render_plot_npy.py --joints_dir "${JOINTS_DIR}" || \
    echo "(visualization step failed — non-fatal)"
else
  echo "(skipping visualization — ${JOINTS_DIR} or renderer missing)"
fi

echo ""
echo "=========================================================="
echo "Done. Output: ${OUTPUT_ROOT}"
echo "=========================================================="
