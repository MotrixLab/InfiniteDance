#!/usr/bin/env bash
# Top-level training entry. Forwards to train_v2/run_infinitedance_train.sh.
# Override defaults via env vars, e.g.:
#   GPUS=0,1,2,3 WS=4 DATA_ROOT=../InfiniteDanceData bash train.sh
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
bash "${SCRIPT_DIR}/train_v2/run_infinitedance_train.sh" "$@"
