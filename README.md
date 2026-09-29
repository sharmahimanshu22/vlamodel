# vlamodel

Experiments with vision-language-action (VLA) models in simulation, using
[LeRobot](https://github.com/huggingface/lerobot) and the [LIBERO](https://libero-project.github.io) benchmark.

Code is edited locally and synced with git. Jobs run on Minerva through LSF.
Compute nodes have **no internet access**, so everything is installed and downloaded
on the login node first.

## Layout

```
config/cluster.env    paths, conda env, LSF defaults, cache locations (sourced by everything)
scripts/              login-node setup: setup_env.sh, prefetch.sh; check_env.py
lsf/submit.sh         submit any jobs/*.sh; creates runs/<name>_<timestamp>/
lsf/job_wrapper.sh    runs on the compute node: activates env, sets offline mode
jobs/                 job scripts (check_env.sh, eval_libero.sh, ...)
runs/                 one dir per submitted job; small text results are committed
storage/              git-ignored: HF cache, LIBERO assets, checkpoints, setup logs
env/                  pip freeze / conda export of the env, written by setup_env.sh
```

## One-time setup (login node)

```bash
git clone https://github.com/sharmahimanshu22/vlamodel && cd vlamodel

# 1. Create the 'lerobot' conda env (torch 2.9.1+cu130, lerobot 0.6.1 with LIBERO)
bash scripts/setup_env.sh

# 2. Hugging Face login: pi0/pi0.5 use the gated PaliGemma tokenizer.
#    First accept the license at https://huggingface.co/google/paligemma-3b-pt-224
source config/cluster.env && activate_env && hf auth login

# 3. Download LIBERO assets + default checkpoint, write LIBERO config
bash scripts/prefetch.sh
```

## Running jobs

```bash
lsf/submit.sh -W 00:20 jobs/check_env.sh          # GPU + rendering sanity check
lsf/submit.sh -W 00:30 jobs/eval_libero.sh lerobot/pi05_libero_finetuned libero_spatial 1 \
    --env.task_ids='[0]' --policy.n_action_steps=10
lsf/submit.sh -d ...                               # dry run: print the bsub command
```

Options: `-g` GPU model (`a100`, `a10080g`, `h10080g`, `h100nvl`, `l40s`), `-W` wall time,
`-m` memory (MB), `-n` cores, `-J` run name. Watch with `bjobs` and `tail -f runs/<run>/lsf.out`.

To share results, commit the run directory (`git add runs/<run> && git commit && git push`).
Videos and checkpoints stay on the server.
