#!/bin/bash
# "Noisy-reward GRPO": vanilla GRPO (same estimator, entry point and settings as
# scripts/launch_baseline_grpo_synthetic.sh) trained on a corrupted reward.
#
# The reward is scripts/noisy_reward_fn.py:compute_score (verl custom_reward_function):
# the Frontier-CS judge's 0-100 score plus noise, applied ONLY to the training rows
# (ground_truth "frontiersmith_<n>"); validation rows get the untouched judge result, so
# the Frontier-CS validation metrics are directly comparable with the baseline's. Training
# rows also carry the clean score (acc / clean_score) and the noise in the rollout dumps.
#
# Noise model (env -> reward_kwargs.<kwarg>; only the variables that are set are passed,
# compute_score's defaults apply otherwise; the names are checked against its signature
# before launching):
#   NOISE_TYPE    noise_type     REQUIRED: none | gaussian | random | ... (NOISE_TYPES in the file)
#   NOISE_STD     noise_std      gaussian: r + N(0, std^2)
#   NOISE_P       noise_p        random: with prob p, r -> U[low, high] (required for random:
#                                compute_score's default 0 means no noise)
#   NOISE_LOW     noise_low      random: U[low, high] bounds (0, 100)
#   NOISE_HIGH    noise_high
#   NOISE_ON_ZERO noise_on_zero  True/False: also perturb rows whose clean score is 0
#   NOISE_K       noise_k        subsample mode
#   NOISE_CLIP    clip           True/False: clip the noisy score to [NOISE_CLIP_LOW, NOISE_CLIP_HIGH]
#   NOISE_CLIP_LOW / NOISE_CLIP_HIGH  clip_low / clip_high
#   NOISE_SEED    noise_seed     (default 0)
#
# Usage:  NOISE_TYPE=gaussian NOISE_STD=10 bash scripts/launch_noisy_grpo.sh [extra hydra overrides...]
#   env: RUN_NAME (qwen35_9b_grpo_noisy), TOTAL_STEPS, VAL_FREQ, VAL_N, VAL_BEFORE_TRAIN,
#        CKPT_POLICY, SHM_MIN_FREE_GB, JUDGE_MAX_WAIT, IDLE_MIN / PREEMPT_IDLE_MIN,
#        NO_TMUX=1 (foreground), DRY_RUN=1 (only compose the Hydra config), PRINT_ONLY=1.
#        Full list: scripts/launch_common.sh.

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_NAME=${RUN_NAME:-qwen35_9b_grpo_noisy}
unset TRAIN_ENTRY ADV_SCALE   # verl's own main_ppo, plain GRPO
REWARD_FN="$PROJECT_ROOT/scripts/noisy_reward_fn.py"   # absolute: imported inside the Ray workers

if [ -z "${NOISE_TYPE:-}" ]; then
    echo "$(basename "$0"): NOISE_TYPE must be set (e.g. NOISE_TYPE=gaussian NOISE_STD=10); there is no default noise model." >&2
    exit 2
fi
if [ "$NOISE_TYPE" = random ] && [ -z "${NOISE_P:-}" ]; then
    echo "$(basename "$0"): NOISE_TYPE=random needs NOISE_P (compute_score's default noise_p=0 adds no noise)." >&2
    exit 2
fi
NOISE_SEED=${NOISE_SEED:-0}

ALGO_OVERRIDES=(
    reward.custom_reward_function.path="$REWARD_FN"
    reward.custom_reward_function.name=compute_score
)
NOISE_KWARGS=()
for _pair in NOISE_TYPE:noise_type NOISE_STD:noise_std NOISE_P:noise_p NOISE_LOW:noise_low \
             NOISE_HIGH:noise_high NOISE_ON_ZERO:noise_on_zero NOISE_K:noise_k \
             NOISE_CLIP:clip NOISE_CLIP_LOW:clip_low NOISE_CLIP_HIGH:clip_high NOISE_SEED:noise_seed; do
    _var=${_pair%%:*} _kw=${_pair#*:}
    if [ -n "${!_var:-}" ]; then
        NOISE_KWARGS+=("$_kw")
        ALGO_OVERRIDES+=("+reward.custom_reward_function.reward_kwargs.$_kw=${!_var}")
    fi
done
unset _pair _var _kw

# Fail now, not after the container and vLLM are up: every kwarg must be a compute_score
# parameter (it raises on unknown kwargs) and NOISE_TYPE one of its NOISE_TYPES.
PY="$PROJECT_ROOT/.venv/bin/python"
[ -x "$PY" ] || PY=$(command -v python3 || true)
if [ -n "$PY" ]; then
    "$PY" - "$REWARD_FN" "$NOISE_TYPE" "${NOISE_KWARGS[@]}" <<'EOF' || exit 2
import ast, sys
path, noise_type, *kwargs = sys.argv[1:]
tree = ast.parse(open(path).read(), path)
fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compute_score"), None)
if fn is None:
    sys.exit(f"launch_noisy_grpo.sh: no compute_score() in {path}")
accepted = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
types = next((ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
              and any(getattr(t, "id", None) == "NOISE_TYPES" for t in n.targets)), None)
errors = [f"reward_kwarg {k!r} is not a compute_score() parameter" for k in kwargs if k not in accepted]
if types is not None and noise_type not in types:
    errors.append(f"NOISE_TYPE={noise_type!r} is not one of NOISE_TYPES {types}")
if errors:
    sys.exit("launch_noisy_grpo.sh: " + "; ".join(errors) + f" ({path})")
EOF
else
    echo "$(basename "$0"): warning: no python found, reward_kwargs not checked" >&2
fi

source "$PROJECT_ROOT/scripts/launch_common.sh" "$@"
