#!/bin/bash
# "Quantized-reward GRPO": the baseline recipe with each rollout's reward binarized
# against its group mean before the usual GRPO advantage.
#
#   b_i = 1 if r_i > mean_g (up to float rounding: ties count as NOT higher), else 0
#   A_i = (b_i - mean_g(b)) / (std_g(b) + eps)        (algorithm.adv_estimator=grpo_quantized,
#         registered by scripts/adv_estimators.py; norm_adv_by_std_in_grpo as in GRPO)
#   Flat groups get A = 0, as in GRPO. Logged scores (critic/score, validation) stay the
#   raw 0-100 judge scores; only the advantages see the binarized reward.
#   loss_agg_mode is NOT overridden (same token-mean as the baseline).
#
# Everything else is identical to scripts/launch_baseline_grpo_synthetic.sh (same data,
# model, lr, batch 8x8, 16k responses, Frontier-CS validation every VAL_FREQ=10 steps
# with VAL_N=5 plus a step-0 eval, TOTAL_STEPS=100, no param offload, checkpoints to
# /dev/shm per CKPT_POLICY, supervisor with auto-resume): the shared part is
# scripts/launch_common.sh.
#
# Usage:  bash scripts/launch_quantized_grpo.sh [extra hydra overrides...]
#   env: RUN_NAME (qwen35_9b_grpo_quantized), TOTAL_STEPS, VAL_FREQ, VAL_N,
#        VAL_BEFORE_TRAIN, CKPT_POLICY, SHM_MIN_FREE_GB, JUDGE_MAX_WAIT,
#        IDLE_MIN / PREEMPT_IDLE_MIN, NO_TMUX=1 (foreground), DRY_RUN=1 (only print the
#        composed Hydra config, in the container), PRINT_ONLY=1 (print the command).

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_NAME=${RUN_NAME:-qwen35_9b_grpo_quantized}
unset ADV_SCALE
export TRAIN_ENTRY=scripts/main_ppo_plugins.py   # registers grpo_quantized in the Ray workers
ALGO_OVERRIDES=(
    algorithm.adv_estimator=grpo_quantized
)

source "$PROJECT_ROOT/scripts/launch_common.sh" "$@"
