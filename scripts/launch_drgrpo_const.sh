#!/bin/bash
# "Dr. GRPO with a constant scale": the baseline recipe with an unbiased advantage.
#
#   A_i = (r_i - mean_g) / c,  c = ADV_SCALE (default 18 = RMS within-group std of the
#         baseline's rollouts, so RMS(A) matches GRPO's and the KL coef / grad clip
#         keep their meaning)
#   loss_agg_mode = seq-mean-token-sum-norm (per-sequence token SUM over a constant
#         denominator; the default per-sequence token-mean is biased by length)
#
# Everything else is identical to scripts/launch_baseline_grpo_synthetic.sh (same data,
# model, lr, batch 8x8, 16k responses, Frontier-CS validation every VAL_FREQ=10 steps
# with VAL_N=5, TOTAL_STEPS=100, no param offload, checkpoints to /dev/shm per
# CKPT_POLICY, supervisor with auto-resume): the shared part is scripts/launch_common.sh.
#
# Usage:  bash scripts/launch_drgrpo_const.sh [extra hydra overrides...]
#   env: ADV_SCALE (18), RUN_NAME (qwen35_9b_drgrpo_const_val), TOTAL_STEPS, VAL_FREQ,
#        VAL_N, VAL_BEFORE_TRAIN, CKPT_POLICY, SHM_MIN_FREE_GB, JUDGE_MAX_WAIT,
#        IDLE_MIN / PREEMPT_IDLE_MIN, NO_TMUX=1 (foreground), DRY_RUN=1, PRINT_ONLY=1.

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_NAME=${RUN_NAME:-qwen35_9b_drgrpo_const_val}
export ADV_SCALE=${ADV_SCALE:-18}
export TRAIN_ENTRY=scripts/main_ppo_plugins.py   # registers drgrpo_scaled in the Ray workers
ALGO_OVERRIDES=(
    algorithm.adv_estimator=drgrpo_scaled
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm
)

source "$PROJECT_ROOT/scripts/launch_common.sh" "$@"
