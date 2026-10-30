#!/bin/bash
# Keep a preemptible verl training job running until it finishes.
#
# We run at low priority on a shared node: higher-priority jobs `docker stop` our
# container when they need the GPUs. This loop waits until all GPUs are idle,
# (re)launches the command — verl's trainer.resume_mode=auto resumes from the
# latest checkpoint in $CKPT_DIR — and repeats until the command exits 0.
#
#   exit 0                  -> training finished, stop
#   exit 137/143 (signal)   -> preempted: wait until GPUs have been idle for
#                              $PREEMPT_IDLE_MIN minutes (gives the job that preempted
#                              us time to start), relaunch. Repeated preemptions with no
#                              new checkpoint in between double the wait (capped at
#                              $MAX_IDLE_MIN), so we don't fight a higher-priority user
#                              who keeps reclaiming the node.
#   any other exit          -> error: relaunch, but give up after $MAX_ERRORS in a row
#                              without progress (an attempt that saved a new
#                              checkpoint resets the count, so rare transient
#                              failures over a long run don't add up)
#
# It also prunes checkpoints older than the latest complete one (verl doesn't track
# the checkpoint it resumed from, so each resume would otherwise leak one), and
# removes our stale shared-memory files left by killed containers.
#
# Checkpoints live in /dev/shm, which other users' jobs fill too. verl writes a new
# checkpoint completely before it deletes the previous one, so EVERY save briefly holds
# two checkpoints and needs a full checkpoint's worth (~103 GB) of free space on top of
# the existing one (after a resume the old one is removed by the pruner below instead of
# verl); a failed save crashes the attempt. An attempt is therefore only started while
# /dev/shm has at least $SHM_MIN_FREE_GB free. The check only runs at attempt start: if
# others fill /dev/shm during the attempt, a later save can still fail (counted as an
# error and relaunched once space is back, see above).
#
# Validation on resume: verl's trainer.val_before_train=True would re-validate at every
# resumed step (~30 min each). Within a step verl saves the checkpoint BEFORE validating
# and dumps the validation to $VAL_DIR/<step>.jsonl, so with VAL_DIR set each resumed
# attempt (checkpoint step N > 0) gets an extra Hydra override appended to the command
# (the command's trailing args must be Hydra overrides; the last one wins):
#   VAL_FREQ > 0, N % VAL_FREQ == 0 and $VAL_DIR/N.jsonl missing or incomplete
#       -> trainer.val_before_train=True   (step N's validation never finished,
#                                           e.g. preempted mid-validation)
#   otherwise -> trainer.val_before_train=False
# Fresh starts (no checkpoint) and VAL_DIR unset append nothing (the command decides).
# Final checkpoint: verl has no "already finished" check, so resuming from a checkpoint at
# step N >= trainer.total_training_steps trains step N+1 (and saves/validates it). With
# TOTAL_STEPS set, such a resume instead appends trainer.val_before_train=True
# trainer.val_only=True if VAL_DIR is set, VAL_FREQ > 0 and $VAL_DIR/N.jsonl is missing or
# incomplete (verl validates the last step whatever VAL_FREQ), and otherwise the supervisor
# exits 0 without launching (e.g. preempted after the final validation, before exiting).
#
# Usage: supervise_training.sh <command...>
#   env: LOG_PREFIX (log name prefix), CKPT_DIR, CONTAINER_NAME, MAX_ERRORS,
#        SHM_MIN_FREE_GB (110), ADOPT=1 to first wait on an already-running $CONTAINER_NAME,
#        VAL_DIR (= trainer.validation_data_dir; unset: no override), VAL_FREQ (= trainer.test_freq; 0),
#        TOTAL_STEPS (= trainer.total_training_steps; unset: no final-step handling).

set -u

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_PREFIX=${LOG_PREFIX:-train}
CKPT_DIR=${CKPT_DIR:?CKPT_DIR must be set}
CONTAINER_NAME=${CONTAINER_NAME:-frontiersmith-train}
MAX_ERRORS=${MAX_ERRORS:-3}
IDLE_MIN=${IDLE_MIN:-3}                # idle minutes required before a normal (re)launch
PREEMPT_IDLE_MIN=${PREEMPT_IDLE_MIN:-10}  # ... and after being preempted
MAX_IDLE_MIN=${MAX_IDLE_MIN:-120}         # cap for the doubling backoff
SHM_MIN_FREE_GB=${SHM_MIN_FREE_GB:-110}   # free /dev/shm needed to start an attempt (one checkpoint + margin)
VAL_DIR=${VAL_DIR:-}                      # trainer.validation_data_dir; empty = never override val_before_train
VAL_FREQ=${VAL_FREQ:-0}                   # trainer.test_freq
TOTAL_STEPS=${TOTAL_STEPS:-}              # trainer.total_training_steps; empty = no final-step handling
preempt_wait=0                            # 0 = no preemption without progress yet
need_idle=$IDLE_MIN
SUPERVISOR_LOG=outputs/logs/${LOG_PREFIX}_supervisor.log
mkdir -p outputs/logs

log() { echo "[supervisor $(date -u '+%F %T')] $*" | tee -a "$SUPERVISOR_LOG"; }

if ! [[ $VAL_FREQ =~ ^[0-9]+$ ]]; then
    log "VAL_FREQ must be a non-negative integer, got '$VAL_FREQ'"
    exit 2
fi
if [ -n "$TOTAL_STEPS" ] && ! [[ $TOTAL_STEPS =~ ^[0-9]+$ ]]; then
    log "TOTAL_STEPS must be a non-negative integer, got '$TOTAL_STEPS'"
    exit 2
fi

gpus_idle() {
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] &&
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1024 { busy = 1 } END { exit busy }'
}

shm_free_gb() { df --output=avail -BG /dev/shm | tail -1 | tr -dc 0-9; }

wait_for_idle_gpus() {
    # Require $need_idle idle checks (1/min) in a row so we don't race a job that is starting up,
    # and enough free /dev/shm for the attempt's first checkpoint save.
    local idle=0 announced=0 shm_announced=0 free
    while [ $idle -lt "$need_idle" ]; do
        free=$(shm_free_gb)
        if [ "${free:-0}" -lt "$SHM_MIN_FREE_GB" ]; then
            idle=0
            [ $shm_announced -eq 0 ] && log "/dev/shm has only ${free}G free (need ${SHM_MIN_FREE_GB}G for a checkpoint); waiting" && shm_announced=1
        elif gpus_idle; then
            idle=$((idle + 1))
        else
            idle=0
            [ $announced -eq 0 ] && log "GPUs busy; waiting for them to free up" && announced=1
        fi
        sleep 60
    done
}

prune_checkpoints() {
    local latest d n
    latest=$(cat "$CKPT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null) || return 0
    for d in "$CKPT_DIR"/global_step_*; do
        [ -d "$d" ] || continue
        n=${d##*_}
        if [ "$n" -lt "$latest" ]; then
            log "pruning old checkpoint $(basename "$d") (latest complete: $latest)"
            rm -rf "$d"
        fi
    done
}

clean_stale_shm() {
    find /dev/shm -maxdepth 1 -user "$(id -un)" -type f -mmin +1 \
        \( -name 'psm_*' -o -name 'cuda.shm.*' -o -name 'torch_*' -o -name 'sem.*' \) \
        -exec sh -c 'fuser -s "$1" 2>/dev/null || rm -f "$1"' _ {} \;
}

# A complete dump is non-empty and ends with the newline verl writes after the last
# record (a missing final newline means the write was cut short).
val_dump_complete() { [ -s "$1" ] && [ -z "$(tail -c 1 "$1")" ]; }

# Sets the array val_override to the Hydra override(s) for an attempt resuming from
# checkpoint step $1 (see header).
set_val_override() {
    local n=$1 dump what
    val_override=()
    if ! [[ $n =~ ^[0-9]+$ ]] || [ $((10#$n)) -eq 0 ]; then
        # Fresh start (no checkpoint, e.g. a restart from scratch under CKPT_POLICY=final):
        # skip the step-0 validation if an earlier attempt of this run already completed it.
        if [ -n "$VAL_DIR" ] && val_dump_complete "$VAL_DIR/0.jsonl"; then
            val_override=(trainer.val_before_train=False)
            log "fresh start; the step-0 validation is already done ($VAL_DIR/0.jsonl): skipping it (${val_override[*]})"
        fi
        return 0
    fi
    n=$((10#$n))
    dump="$VAL_DIR/$n.jsonl"
    if [ -n "$TOTAL_STEPS" ] && [ "$n" -ge "$TOTAL_STEPS" ]; then
        if [ -n "$VAL_DIR" ] && [ "$VAL_FREQ" -gt 0 ] && ! val_dump_complete "$dump"; then
            val_override=(trainer.val_before_train=True trainer.val_only=True)
            log "checkpoint step $n is the final step (TOTAL_STEPS=$TOTAL_STEPS) and $dump is missing or incomplete: validating it only (${val_override[*]})"
            return 0
        fi
        if [ -n "$VAL_DIR" ]; then what="its validation is done or disabled ($dump, VAL_FREQ=$VAL_FREQ)"; else what="VAL_DIR unset, validation not checked"; fi
        log "checkpoint step $n is the final step (TOTAL_STEPS=$TOTAL_STEPS); $what: training finished, nothing to launch"
        exit 0
    fi
    [ -n "$VAL_DIR" ] || return 0
    if [ "$VAL_FREQ" -gt 0 ] && [ $((n % VAL_FREQ)) -eq 0 ] && ! val_dump_complete "$dump"; then
        val_override=(trainer.val_before_train=True)
        log "step $n is a validation step and $dump is missing or incomplete: re-validating it (${val_override[*]})"
    else
        val_override=(trainer.val_before_train=False)
        if [ "$VAL_FREQ" -gt 0 ] && [ $((n % VAL_FREQ)) -eq 0 ]; then
            log "validation of step $n already done ($dump): skipping it (${val_override[*]})"
        else
            log "step $n is not a validation step (VAL_FREQ=$VAL_FREQ): skipping validation (${val_override[*]})"
        fi
    fi
}

# Returns 0 to keep going, exits the script when done or giving up.
errors=0
attempt_ckpt=$(cat "$CKPT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null)
handle_exit() {
    local code=$1
    prune_checkpoints
    case $code in
        0)
            log "training finished (exit 0); latest checkpoint: $(cat "$CKPT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null)"
            exit 0 ;;
        137|143)
            errors=0
            if [ "$(cat "$CKPT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null)" != "$attempt_ckpt" ]; then
                preempt_wait=$PREEMPT_IDLE_MIN  # made progress before being preempted
            else
                preempt_wait=$((preempt_wait == 0 ? PREEMPT_IDLE_MIN : preempt_wait * 2))
                preempt_wait=$((preempt_wait > MAX_IDLE_MIN ? MAX_IDLE_MIN : preempt_wait))
            fi
            need_idle=$preempt_wait
            log "preempted (exit $code); resuming from checkpoint $(cat "$CKPT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || echo none) after GPUs are idle for ${need_idle} min" ;;
        *)
            if [ "$(cat "$CKPT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null)" != "$attempt_ckpt" ]; then
                errors=0  # this attempt made progress
            fi
            errors=$((errors + 1))
            log "failed (exit $code), consecutive errors: $errors/$MAX_ERRORS"
            if [ $errors -ge "$MAX_ERRORS" ]; then
                log "giving up after $errors consecutive errors"
                exit 1
            fi ;;
    esac
}

# Background pruning while an attempt runs (each save completes before the old one is obsolete).
( while sleep 120; do prune_checkpoints; done ) &
PRUNER=$!
trap 'kill $PRUNER 2>/dev/null' EXIT

if [ "${ADOPT:-0}" = "1" ] && docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    log "adopting running container $CONTAINER_NAME"
    handle_exit "$(docker wait "$CONTAINER_NAME" 2>/dev/null || echo 143)"
fi

while true; do
    # Decided before waiting for the GPUs (nothing changes the checkpoint meanwhile), so a
    # run whose final step is complete exits right away.
    attempt_ckpt=$(cat "$CKPT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null)
    set_val_override "$attempt_ckpt"
    wait_for_idle_gpus
    need_idle=$IDLE_MIN
    clean_stale_shm
    LOG=outputs/logs/${LOG_PREFIX}_$(date +%Y%m%d_%H%M%S).log
    echo "$LOG" > outputs/logs/LATEST
    log "starting attempt (resume from: ${attempt_ckpt:-scratch}), log: $LOG"
    "$@" "${val_override[@]}" 2>&1 | tee -a "$LOG"
    handle_exit "${PIPESTATUS[0]}"
done
