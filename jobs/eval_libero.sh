#!/usr/bin/env bash
# Evaluate a policy on LIBERO with lerobot-eval.
#   lsf/submit.sh jobs/eval_libero.sh <policy> <suite(s)> <episodes_per_task> [extra lerobot-eval args...]
#
# Examples:
#   # smoke test: 1 task, 1 episode
#   lsf/submit.sh -W 00:30 jobs/eval_libero.sh lerobot/pi05_libero_finetuned libero_spatial 1 \
#       --env.task_ids='[0]' --policy.n_action_steps=10
#   # full suite, published protocol (10 episodes per task)
#   lsf/submit.sh -W 03:00 jobs/eval_libero.sh lerobot/pi05_libero_finetuned libero_spatial 10 \
#       --policy.n_action_steps=10
set -euo pipefail
policy=$1 suite=$2 episodes=$3; shift 3

# batch_size=1 keeps every env in this process: GPUs here are in exclusive-process
# mode, so subprocess envs opening their own EGL contexts could fail.
lerobot-eval \
    --policy.path="$policy" \
    --env.type=libero \
    --env.task="$suite" \
    --eval.n_episodes="$episodes" \
    --eval.batch_size=1 \
    --env.max_parallel_tasks=1 \
    --output_dir="$RUN_DIR/eval" \
    "$@" 2>&1 | tee "$RUN_DIR/eval.log"
