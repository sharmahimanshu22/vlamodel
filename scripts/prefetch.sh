#!/usr/bin/env bash
# Download checkpoints, tokenizers and LIBERO assets. Run on the LOGIN node.
#   bash scripts/prefetch.sh                              # defaults
#   bash scripts/prefetch.sh --models lerobot/smolvla_base
set -euo pipefail
source "$(dirname "$0")/../config/cluster.env"
activate_env

mkdir -p "$VLA_STORAGE/logs"
LOG="$VLA_STORAGE/logs/prefetch_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "Logging to $LOG"

if ! hf auth whoami >/dev/null 2>&1; then
    echo "NOTE: not logged in to Hugging Face. Gated tokenizers (PaliGemma, used by pi0/pi0.5)"
    echo "      need a token: run 'source config/cluster.env && activate_env && hf auth login' first."
fi

# The login node caps threads per user. hf_xet spawns a thread pool per parallel file
# download and hits that cap ("failed to spawn thread"), so use plain HTTP downloads.
export HF_HUB_DISABLE_XET=1

python "$VLA_ROOT/scripts/prefetch.py" "$@"
