#!/usr/bin/env bash
# Runs on the compute node (invoked by lsf/submit.sh). Sets up an offline environment,
# then runs the job script with RUN_DIR exported.
# Usage: job_wrapper.sh <run_dir> <job_script> [args...]
set -uo pipefail
run_dir=$1 job_script=$2; shift 2

source "$(dirname "$0")/../config/cluster.env"
activate_env

# Compute nodes have no internet: fail fast instead of hanging on downloads.
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export RUN_DIR=$run_dir
cd "$VLA_ROOT"

echo "== host $(hostname)  job ${LSB_JOBID:-?}  start $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "== running: $job_script $*"

start=$(date +%s)
bash "$job_script" "$@"
status=$?

{
    echo "exit_code: $status"
    echo "host: $(hostname)"
    echo "lsf_job_id: ${LSB_JOBID:-}"
    echo "duration_s: $(( $(date +%s) - start ))"
} >> "$run_dir/run_info.txt"
echo "== finished with exit code $status at $(date -Iseconds)"
exit $status
