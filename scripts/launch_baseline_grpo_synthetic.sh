#!/bin/bash
# Baseline (vanilla) GRPO on the 200 FrontierSmith synthetic problems, Qwen3.5-9B.
# Wraps run_verl_grpo_frontiercs_qwen35_9b_synthetic.sh (paper settings: 16k response,
# lr 1e-6, batch 8 x group 8) with the overrides needed on this host (scripts/launch_common.sh):
#
#   - runs inside the training container (glibc 2.31 host; see run_in_container.sh)
#   - validation on the 172 Frontier-CS problems (data/frontiercs/full.parquet, built from
#     the official repo; see data/sample_lists/frontiercs_val_172.json): every VAL_FREQ
#     steps (default 10), VAL_N samples per problem (default 5), plus a step-0 eval of the
#     base model (VAL_BEFORE_TRAIN). VAL_FREQ=0 disables the periodic validation.
#   - TOTAL_STEPS (default 100, the paper's budget) instead of the script's 60 epochs
#   - no FSDP param offload: 9B sharded over 8x80GB fits; offload only costs time
#   - checkpoints to /dev/shm (host RAM, shared into the container via --ipc host; the
#     shared disk has little free space), keep 1, per CKPT_POLICY: full (default; fp32
#     weights + Adam, ~103 GB, every 10 steps, exact resume), model (weights only, ~35 GB,
#     every 10 steps, resume resets Adam) or final (weights only, last step only). They
#     survive the run being killed but NOT a reboot — merge the final one to disk.
#
# Usage:  bash scripts/launch_baseline_grpo_synthetic.sh [extra hydra overrides...]
#   env: RUN_NAME (qwen35_9b_grpo_val) -> wandb experiment name, rollout/validation dump dirs,
#        checkpoint dir /dev/shm/frontiersmith_ckpt/$RUN_NAME, log prefix;
#        TOTAL_STEPS, VAL_FREQ, VAL_N, VAL_BEFORE_TRAIN, CKPT_POLICY, SHM_MIN_FREE_GB,
#        JUDGE_MAX_WAIT, IDLE_MIN / PREEMPT_IDLE_MIN (supervisor); NO_TMUX=1 runs the
#        supervisor in the foreground (for queues); DRY_RUN=1 only composes the Hydra config;
#        PRINT_ONLY=1 prints the command. Full list: scripts/launch_common.sh.
# Runs scripts/supervise_training.sh (relaunch after preemption, resume from the
# checkpoint). Logs: outputs/logs/$RUN_NAME_*.log and outputs/logs/$RUN_NAME_supervisor.log.

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_NAME=${RUN_NAME:-qwen35_9b_grpo_val}
unset TRAIN_ENTRY ADV_SCALE   # verl's own main_ppo, plain GRPO
ALGO_OVERRIDES=()

source "$PROJECT_ROOT/scripts/launch_common.sh" "$@"
