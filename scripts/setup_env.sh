#!/usr/bin/env bash
# Create the 'lerobot' conda env. Run on the LOGIN node (compute nodes have no internet).
#   bash scripts/setup_env.sh              # create (fails if env already exists)
#   bash scripts/setup_env.sh --recreate   # remove the 'lerobot' env and rebuild it
set -euo pipefail
source "$(dirname "$0")/../config/cluster.env"

TORCH_INDEX="https://download.pytorch.org/whl/cu130"
TORCH_VERSION="2.9.1"
TORCHVISION_VERSION="0.24.1"
TORCHCODEC_SPEC="torchcodec==0.8.*"     # torchcodec 0.8 is the release built against torch 2.9
# CPU build: the cu130 build needs NVIDIA NPP (libnppicc.so.13), which isn't installed, and
# LeRobot decodes dataset videos on CPU in dataloader workers anyway.
TORCHCODEC_INDEX="https://download.pytorch.org/whl/cpu"
LEROBOT_SPEC="lerobot[libero,smolvla,pi]==0.6.1"

mkdir -p "$VLA_STORAGE/logs"
LOG="$VLA_STORAGE/logs/setup_env_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "Logging to $LOG"

source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
    if [[ "${1:-}" == "--recreate" ]]; then
        echo ">> Removing existing env '$CONDA_ENV_NAME'"
        conda env remove -y -n "$CONDA_ENV_NAME"
    else
        echo "ERROR: env '$CONDA_ENV_NAME' already exists. Use --recreate to rebuild it." >&2
        exit 1
    fi
fi

echo ">> Creating env '$CONDA_ENV_NAME' (python 3.12, ffmpeg, cmake)"
# ffmpeg: shared libs for torchcodec video decoding.
# cmake:  robomimic (a LIBERO dependency) builds egl_probe from source.
conda create -y -n "$CONDA_ENV_NAME" -c conda-forge python=3.12 "ffmpeg>=6,<8" cmake
conda activate "$CONDA_ENV_NAME"

python -m pip install --upgrade pip

echo ">> Installing torch $TORCH_VERSION (cu130) first so later installs keep this build"
pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "$TORCH_INDEX"
pip install --no-deps "$TORCHCODEC_SPEC" --index-url "$TORCHCODEC_INDEX"

echo ">> Installing $LEROBOT_SPEC"
# egl_probe's CMakeLists predates CMake 4; this lets CMake 4 still configure it.
export CMAKE_POLICY_VERSION_MINIMUM=3.5
pip install "$LEROBOT_SPEC"

echo ">> Verifying torch was not replaced"
python - <<EOF
import torch
v = torch.__version__
assert v.startswith("$TORCH_VERSION") and v.endswith("+cu130"), f"torch changed to {v}"
print("torch", v, "cuda", torch.version.cuda)
EOF

mkdir -p "$VLA_ROOT/env"
pip freeze > "$VLA_ROOT/env/lerobot-freeze.txt"
conda env export -n "$CONDA_ENV_NAME" --no-builds > "$VLA_ROOT/env/lerobot-conda.yml"
echo ">> Done. Package list saved to env/. Next: bash scripts/prefetch.sh"
