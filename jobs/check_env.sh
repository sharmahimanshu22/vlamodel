#!/usr/bin/env bash
# GPU-node sanity check: CUDA, bf16, torchcodec, LIBERO EGL rendering.
#   lsf/submit.sh -W 00:20 jobs/check_env.sh
set -euo pipefail
python scripts/check_env.py 2>&1 | tee "$RUN_DIR/check_env.log"
