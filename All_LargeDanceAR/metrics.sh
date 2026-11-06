#!/usr/bin/env bash
# CPU evaluation of the canonical release split. Extra evaluator args are forwarded.
set -euo pipefail
SCRIPT_FILE=${BASH_SOURCE[0]}
while [[ -L "$SCRIPT_FILE" ]]; do
  LINK_DIR="$(cd -P "$(dirname "$SCRIPT_FILE")" && pwd)"
  SCRIPT_FILE=$(readlink "$SCRIPT_FILE")
  [[ "$SCRIPT_FILE" = /* ]] || SCRIPT_FILE="$LINK_DIR/$SCRIPT_FILE"
done
SCRIPT_DIR="$(cd -P "$(dirname "$SCRIPT_FILE")" && pwd)"
PRED_ROOT=${1:?Usage: bash metrics.sh PRED_ROOT [legacy_device_id] [evaluator arguments]}
shift
# Preserve the documented positional device argument; this evaluator uses CPU.
if [[ ${1:-} =~ ^[0-9]+$ ]]; then shift; fi
DATA_ROOT=${DATA_ROOT:-${SCRIPT_DIR}/../InfiniteDanceData}
PY=${PY:-python}
"${PY}" "${SCRIPT_DIR}/metrics/evaluate_release.py" \
  --pred-root "${PRED_ROOT}" \
  --gt-features "${GT_ROOT:-${DATA_ROOT}/dance/evaluation_features_train8235.npz}" \
  --gt-list "${GT_LIST:-${DATA_ROOT}/partition/All_cleandata_train.txt}" \
  --eval-list "${EVAL_LIST:-${DATA_ROOT}/partition/test_eval861.txt}" \
  --music-root "${MUSIC_FEAT_ROOT:-${DATA_ROOT}/music/musicfeature_55_allmusic_pure}" \
  --output "${METRICS_REPORT:-${PRED_ROOT}/release_metrics.json}" \
  --max-frames "${MAX_FRAMES:-1024}" --workers "${WORKERS:-1}" "$@"
