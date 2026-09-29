#!/usr/bin/env bash
# Submit a job script from jobs/ to LSF. Each submission gets its own run directory:
#   runs/<job>_<timestamp>/   (lsf.out, job metadata, outputs; large files git-ignored)
#
# Usage: lsf/submit.sh [options] <jobs/script.sh> [script args...]
#   -g MODEL   GPU model: a100 | a10080g | h10080g | h100nvl | l40s   (default from cluster.env)
#   -W HH:MM   wall time
#   -m MB      host memory
#   -n N       CPU cores
#   -q QUEUE   LSF queue
#   -J NAME    run name (default: job script name)
#   -d         dry run: print the bsub command, don't submit
set -euo pipefail
source "$(dirname "$0")/../config/cluster.env"

gpu=$LSF_GPU_MODEL walltime=$LSF_WALLTIME mem=$LSF_MEM_MB ncpu=$LSF_NCPU queue=$LSF_QUEUE
name="" dry_run=0
while getopts "g:W:m:n:q:J:d" opt; do
    case $opt in
        g) gpu=$OPTARG ;;
        W) walltime=$OPTARG ;;
        m) mem=$OPTARG ;;
        n) ncpu=$OPTARG ;;
        q) queue=$OPTARG ;;
        J) name=$OPTARG ;;
        d) dry_run=1 ;;
        *) sed -n '2,13p' "$0"; exit 1 ;;
    esac
done
shift $((OPTIND - 1))
[[ $# -ge 1 ]] || { sed -n '2,13p' "$0"; exit 1; }

job_script=$(realpath "$1"); shift
[[ -f $job_script ]] || { echo "No such job script: $job_script" >&2; exit 1; }
name=${name:-$(basename "$job_script" .sh)}
run_id="${name}_$(date +%Y%m%d_%H%M%S)"
run_dir="$VLA_ROOT/runs/$run_id"
mkdir -p "$run_dir"

# Record what was run, so results can be traced back to code.
{
    echo "run_id: $run_id"
    echo "job_script: ${job_script#"$VLA_ROOT"/}"
    echo "args: $*"
    echo "git_commit: $(git -C "$VLA_ROOT" rev-parse --short HEAD 2>/dev/null || echo none)"
    echo "git_dirty: $(git -C "$VLA_ROOT" status --porcelain 2>/dev/null | grep -qv '^??' && echo yes || echo no)"
    echo "lsf: queue=$queue gpu=$gpu ncpu=$ncpu mem_mb=$mem walltime=$walltime"
    echo "submitted: $(date -Iseconds)"
} > "$run_dir/run_info.txt"

cmd=(bsub -P "$LSF_PROJECT" -q "$queue" -n "$ncpu" -gpu "num=1"
     -R "$gpu" -R "rusage[mem=$mem]" -W "$walltime"
     -J "$run_id" -oo "$run_dir/lsf.out"
     "$VLA_ROOT/lsf/job_wrapper.sh" "$run_dir" "$job_script" "$@")

if [[ $dry_run == 1 ]]; then
    printf '%q ' "${cmd[@]}"; echo
    rm -rf "$run_dir"
else
    "${cmd[@]}"
    echo "Run dir: ${run_dir#"$VLA_ROOT"/}"
fi
