#!/bin/bash
# "Adaptive-threshold GRPO": quantized-reward GRPO with each rollout's reward binarized against
# a per-problem threshold carried across steps, instead of against its own group mean.
#
#   first group of problem p:  tau_p = mean of that group
#   every group of problem p:  r_i reaches t iff r_i >= t (up to float rounding) and r_i > 0
#                              if at least ADAPTIVE_FRAC of the group reaches tau_p:
#                                  tau_p = max(tau_p, mean_g(r))
#                              b_i = 1 if r_i reaches tau_p (as just updated), else 0
#                              A_i = (b_i - mean_g(b)) / (std_g(b) + eps)
#   A group that raises the threshold is binarized against the raised one; a group in which everybody
#   reaches tau_p is the special case where the whole group, not only ADAPTIVE_FRAC of it, does: it is
#   split at its mean (flat only if its scores are all equal).
#   (algorithm.adv_estimator=grpo_adaptive_threshold, scripts/adv_adaptive_threshold.py.)
#   Groups where nobody or everybody reaches the threshold they are binarized against get A = 0.
#   Thresholds are saved with every checkpoint and restored on resume; they are logged as
#   adaptive_threshold/* and dumped per step to outputs/adaptive_threshold_$RUN_NAME/<step>.json.
#   Logged scores (critic/score, validation) stay the raw 0-100 judge scores.
#
# Everything else is identical to scripts/launch_quantized_grpo.sh (shared part:
# scripts/launch_common.sh).
#
# Usage:  bash scripts/launch_adaptive_threshold_grpo.sh [extra hydra overrides...]
#   env: ADAPTIVE_FRAC (fraction x of a group that must reach the threshold to raise it,
#        0 < x <= 1; default 0.75), plus the knobs of scripts/launch_common.sh (RUN_NAME
#        default qwen35_9b_grpo_adaptive_threshold).

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_NAME=${RUN_NAME:-qwen35_9b_grpo_adaptive_threshold}
ADAPTIVE_FRAC=${ADAPTIVE_FRAC:-0.75}
python3 -c "import sys; x = float(sys.argv[1]); sys.exit(0 if 0 < x <= 1 else 1)" "$ADAPTIVE_FRAC" 2>/dev/null ||
    { echo "$(basename "$0"): ADAPTIVE_FRAC must be a number in (0, 1], got '$ADAPTIVE_FRAC'" >&2; exit 2; }
unset ADV_SCALE
export TRAIN_ENTRY=scripts/main_ppo_adaptive.py   # registers grpo_adaptive_threshold in the Ray workers
ALGO_OVERRIDES=(
    algorithm.adv_estimator=grpo_adaptive_threshold
    +algorithm.adaptive_threshold_frac="$ADAPTIVE_FRAC"
)

source "$PROJECT_ROOT/scripts/launch_common.sh" "$@"
