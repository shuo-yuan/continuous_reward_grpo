#!/bin/bash
# "Dynamic sampling" GRPO (DAPO): each training step keeps generating until it has enough groups
# with nonzero advantages, then trains on exactly TRAIN_BATCH_SIZE groups.
#
#   1. generate ROLLOUT_N rollouts per prompt of the step's dataloader batch, as usual;
#   2. a group is informative if ESTIMATOR will give it nonzero advantages (grpo: scores not all
#      equal; grpo_quantized: mixed 0/1 after binarizing at the group mean; grpo_adaptive_threshold:
#      mixed 0/1 against the per-problem threshold, which every generated group updates (kept or not)
#      before it is binarized against it; grpo_fixed_threshold: mixed 0/1 against FIXED_THRESHOLD);
#   3. while fewer than DS_TARGET_GROUPS groups are informative, fewer than DS_MAX_GEN_BATCHES batches
#      were used this step and the judge is not failing, pull more batches from the same training
#      dataloader (as in DAPO they count as consumed) and generate them;
#   4. train on the first TRAIN_BATCH_SIZE informative groups, topped up with flat groups when there
#      are fewer (those only contribute the KL term).
#   (+algorithm.dynamic_sampling=True: scripts/dynamic_sampling.py, entry scripts/main_ppo_dynamic.py.)
#
# Logged as dynamic_sampling/* (rounds, batches used, informative fractions, kept/dropped groups,
# cap_hit, outage_stop, score_mean_round1/_all/_kept, epoch, ...). Every generated group (scores,
# response lengths, judge statuses, kept or not; no texts) is dumped per step to
# outputs/dynamic_sampling_$RUN_NAME/<step>.json.
#
# Caveats:
#   * critic/score/*, response_length/*, train-aux/* and outputs/rollout_data_$RUN_NAME describe the
#     trained (filtered) groups only: compare with other runs on dynamic_sampling/score_mean_round1.
#     timing_per_token_ms/gen and perf/throughput are skewed (dropped groups' tokens are not counted);
#     timing_s/gen is the true generation wall time.
#   * training/epoch is wrong; use dynamic_sampling/epoch.
#   * every extra batch is TRAIN_BATCH_SIZE x ROLLOUT_N more judge submissions.
#
# Everything else is identical to the other launchers (shared part: scripts/launch_common.sh).
# Unless PRINT_ONLY=1 or DRY_RUN=1, a missing outputs/rollout_data_$RUN_NAME is created as a symlink
# to /dev/shm/frontiersmith_rollouts/$RUN_NAME: the rollout dumps (~18 MB per step) would fill the
# small root disk. An existing real directory there is refused (move it to /dev/shm and symlink it;
# ALLOW_ROOT_ROLLOUT_DIR=1 keeps it). This happens before launch_common.sh checks its own knobs, so a
# launch it refuses can leave the link behind (harmless).
#
# Usage:  bash scripts/launch_dynamic_sampling_grpo.sh [extra hydra overrides...]
#   env: ESTIMATOR             grpo (default), grpo_quantized, grpo_adaptive_threshold or grpo_fixed_threshold
#        RUN_NAME              default qwen35_9b_${ESTIMATOR}_dynsamp
#        DS_MAX_GEN_BATCHES    dataloader batches per step at most, the step's own included (default 6)
#        DS_TARGET_GROUPS      informative groups to stop pulling at, 1..TRAIN_BATCH_SIZE (unset: TRAIN_BATCH_SIZE;
#                              checked against a data.train_batch_size=... argument instead when one is given)
#        DS_BATCHES_PER_ROUND  batches per extra generation round, or auto (default): enough to reach the
#                              target at the informative rate seen so far this step (taken as at least 10%)
#        DS_OUTAGE_FRAC        stop pulling when more than this fraction of a round's rollouts timed out or
#                              could not reach the judge (judge_status -1/-2; 0 < f <= 1, default 0.5)
#        ADAPTIVE_FRAC         grpo_adaptive_threshold only (default 0.75; see
#                              scripts/launch_adaptive_threshold_grpo.sh); ignored for the other estimators
#        FIXED_THRESHOLD       grpo_fixed_threshold only: c in b = 1[score >= c and score > 0], a number in
#                              [0, 100] written without sign or exponent (default 50; see
#                              scripts/adv_fixed_threshold.py); ignored for the other estimators
#        LR_WARMUP_STEPS       N > 0: actor_rollout_ref.actor.optim.lr_warmup_steps=N. With verl's default
#                              constant scheduler (fsdp_workers.py: get_constant_schedule_with_warmup) the lr
#                              is scaled by s/N at scheduler step s = 0, 1, ..., N-1 and by 1 after; the
#                              scheduler steps once per update_actor, after the update, so training step 1
#                              runs at lr 0 (it only feeds Adam's moments) and step N+1 is the first at the
#                              full lr (logged as actor/lr). The scheduler is checkpointed: with CKPT_POLICY=full
#                              a resume continues it exactly; with CKPT_POLICY=model (Adam not restored) the
#                              rebuilt optimizer keeps lr 0 for the first update after every resume, while
#                              actor/lr shows the scheduled lr.
#                              Unset or 0: no override (constant lr, as in the other launchers).
#        plus the knobs of scripts/launch_common.sh (TRAIN_BATCH_SIZE, ROLLOUT_N, TOTAL_STEPS, ...).

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

_ds_die() { echo "$(basename "$0"): $*" >&2; exit 2; }
_ds_pos_int() { [[ "$2" =~ ^[1-9][0-9]*$ ]] || _ds_die "$1 must be a positive integer, got '$2'"; }
_ds_frac() {
    python3 -c "import sys; x = float(sys.argv[1]); sys.exit(0 if 0 < x <= 1 else 1)" "$2" 2>/dev/null ||
        _ds_die "$1 must be a number in (0, 1], got '$2'"
}

ESTIMATOR=${ESTIMATOR:-grpo}
case $ESTIMATOR in
    grpo|grpo_quantized|grpo_adaptive_threshold|grpo_fixed_threshold) ;;
    *) _ds_die "ESTIMATOR must be grpo, grpo_quantized, grpo_adaptive_threshold or grpo_fixed_threshold," \
           "got '$ESTIMATOR'" ;;
esac
RUN_NAME=${RUN_NAME:-qwen35_9b_${ESTIMATOR}_dynsamp}
[[ "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] ||
    _ds_die "RUN_NAME may only contain letters, digits, '.', '_' and '-', got '$RUN_NAME'"
DS_MAX_GEN_BATCHES=${DS_MAX_GEN_BATCHES:-6}
DS_BATCHES_PER_ROUND=${DS_BATCHES_PER_ROUND:-auto}
DS_OUTAGE_FRAC=${DS_OUTAGE_FRAC:-0.5}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-0}
_ds_pos_int DS_MAX_GEN_BATCHES "$DS_MAX_GEN_BATCHES"
[ "$DS_BATCHES_PER_ROUND" = auto ] || _ds_pos_int DS_BATCHES_PER_ROUND "$DS_BATCHES_PER_ROUND"
_ds_frac DS_OUTAGE_FRAC "$DS_OUTAGE_FRAC"
[[ "$LR_WARMUP_STEPS" =~ ^(0|[1-9][0-9]*)$ ]] ||
    _ds_die "LR_WARMUP_STEPS must be a non-negative integer, got '$LR_WARMUP_STEPS'"

unset ADV_SCALE
export TRAIN_ENTRY=scripts/main_ppo_dynamic.py   # dynamic sampling + every custom estimator in the Ray workers
ALGO_OVERRIDES=(
    algorithm.adv_estimator="$ESTIMATOR"
    +algorithm.dynamic_sampling=True
    +algorithm.dynamic_sampling_max_gen_batches="$DS_MAX_GEN_BATCHES"
    +algorithm.dynamic_sampling_batches_per_round="$DS_BATCHES_PER_ROUND"
    +algorithm.dynamic_sampling_outage_frac="$DS_OUTAGE_FRAC"
    trainer.total_epochs=1000   # fit() stops at TOTAL_STEPS, but a resume trains nothing if step // len(loader) >= this
)
if [ -n "${DS_TARGET_GROUPS:-}" ]; then
    _ds_pos_int DS_TARGET_GROUPS "$DS_TARGET_GROUPS"
    _ds_batch=${TRAIN_BATCH_SIZE:-8} _ds_batch_from=TRAIN_BATCH_SIZE
    for _ds_arg in "$@"; do  # the last data.train_batch_size override wins in Hydra
        case $_ds_arg in
            data.train_batch_size=*|++data.train_batch_size=*)
                _ds_batch=${_ds_arg#*=} _ds_batch_from="the $_ds_arg argument" ;;
        esac
    done
    # a non-integer batch size is launch_common.sh's or Hydra's to refuse
    if [[ "$_ds_batch" =~ ^[1-9][0-9]*$ ]] && [ "$DS_TARGET_GROUPS" -gt "$_ds_batch" ]; then
        _ds_die "DS_TARGET_GROUPS ($DS_TARGET_GROUPS) must be at most the batch size ($_ds_batch, from $_ds_batch_from)"
    fi
    unset _ds_batch _ds_batch_from _ds_arg
    ALGO_OVERRIDES+=(+algorithm.dynamic_sampling_target_groups="$DS_TARGET_GROUPS")
fi
if [ "$ESTIMATOR" = grpo_adaptive_threshold ]; then
    ADAPTIVE_FRAC=${ADAPTIVE_FRAC:-0.75}
    _ds_frac ADAPTIVE_FRAC "$ADAPTIVE_FRAC"
    ALGO_OVERRIDES+=(+algorithm.adaptive_threshold_frac="$ADAPTIVE_FRAC")
fi
if [ "$ESTIMATOR" = grpo_fixed_threshold ]; then
    FIXED_THRESHOLD=${FIXED_THRESHOLD:-50}
    # plain decimals only, so Hydra reads an int or a float (it would keep e.g. 07 as a string)
    [[ "$FIXED_THRESHOLD" =~ ^(100(\.0+)?|[1-9]?[0-9](\.[0-9]+)?)$ ]] ||
        _ds_die "FIXED_THRESHOLD must be a number in [0, 100] such as 50 or 37.5, got '$FIXED_THRESHOLD'"
    ALGO_OVERRIDES+=(+algorithm.fixed_threshold="$FIXED_THRESHOLD")
fi
if [ "$LR_WARMUP_STEPS" -gt 0 ]; then
    ALGO_OVERRIDES+=(actor_rollout_ref.actor.optim.lr_warmup_steps="$LR_WARMUP_STEPS")
fi

if [ "${PRINT_ONLY:-0}" != 1 ] && [ "${DRY_RUN:-0}" != 1 ]; then
    _rollout_link="outputs/rollout_data_$RUN_NAME"
    _rollout_dir="/dev/shm/frontiersmith_rollouts/$RUN_NAME"
    if [ ! -e "$_rollout_link" ] && [ ! -L "$_rollout_link" ]; then
        mkdir -p outputs "$_rollout_dir"
        ln -s "$_rollout_dir" "$_rollout_link"
        echo "$(basename "$0"): created $_rollout_link -> $_rollout_dir (rollout dumps off the root disk)"
    elif [ ! -e "$_rollout_link" ]; then
        [ "$(readlink "$_rollout_link")" = "$_rollout_dir" ] ||
            _ds_die "$_rollout_link is a dangling symlink to $(readlink "$_rollout_link"); fix or remove it"
        mkdir -p "$_rollout_dir"
        echo "$(basename "$0"): recreated $_rollout_dir (target of $_rollout_link)"
    elif [ ! -L "$_rollout_link" ] && [ "${ALLOW_ROOT_ROLLOUT_DIR:-0}" != 1 ]; then
        _ds_die "$_rollout_link is not a symlink: the rollout dumps would fill the root disk;" \
            "move it to $_rollout_dir and symlink it there (or set ALLOW_ROOT_ROLLOUT_DIR=1 to keep it)"
    fi
    unset _rollout_link _rollout_dir
fi

source "$PROJECT_ROOT/scripts/launch_common.sh" "$@"
