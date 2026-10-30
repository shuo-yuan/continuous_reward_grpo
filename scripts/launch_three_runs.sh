#!/bin/bash
# Queue the three matched 100-step runs back to back in one tmux session ('three_runs'):
# vanilla GRPO, quantized-reward GRPO, noisy-reward GRPO (Qwen3.5-9B, 200 synthetic problems,
# Frontier-CS validation of 172 problems x VAL_N samples every VAL_FREQ steps).
# Each run goes through its launcher with NO_TMUX=1, i.e. through the supervisor
# (preemption-safe, resumes from its own /dev/shm checkpoint); the next run starts when
# the previous one has finished or given up. A failed run (rc != 0) does not stop the queue.
#
# Usage: NOISE_TYPE=gaussian NOISE_STD=10 bash scripts/launch_three_runs.sh
#   ORDER        "baseline quantized noisy"  run order; names: baseline quantized noisy drgrpo
#   STEP0_VAL    all | first | none (all): which runs validate before training
#                (VAL_BEFORE_TRAIN); 'first' = only the first run in ORDER (the supervisor
#                decides about validation on resume separately)
#   RUN_TAG      (3runs) suffix of the default run names:
#                  baseline  qwen35_9b_grpo_$RUN_TAG
#                  quantized qwen35_9b_grpo_quantized_$RUN_TAG
#                  noisy     qwen35_9b_grpo_noisy_${NOISE_TYPE}_$RUN_TAG
#                  drgrpo    qwen35_9b_drgrpo_const_$RUN_TAG
#                RUN_NAME_<name> (e.g. RUN_NAME_noisy) overrides one of them.
#   CKPT_ROOT    (/dev/shm/frontiersmith_ckpt): checkpoints in $CKPT_ROOT/<run name>
#   RESUME=1     allow runs that already have a checkpoint / outputs: a partial run resumes,
#                a complete one (checkpoint step >= TOTAL_STEPS) is skipped, unless its final
#                validation dump is missing (then the supervisor only validates). Without it the
#                queue refuses to start if any run has a checkpoint or a non-empty
#                outputs/{rollout,validation}_data_<run name> (protects against silently
#                resuming / extending an earlier run of the same name).
#   DROP_FINAL_OPTIM (1): with CKPT_POLICY=full, after a run exits 0 delete the optimizer
#                shards (optim_world_size_*_rank_*.pt, ~68 GB) of its final checkpoint;
#                the weights stay. Nothing else is ever deleted by the queue.
#   Shared by all runs (see scripts/launch_common.sh): TOTAL_STEPS (100), VAL_FREQ (10),
#   VAL_N (5), TRAIN_BATCH_SIZE (8), ROLLOUT_N (8), PPO_MINI_BATCH_SIZE, MAX_RESPONSE_LENGTH (16000),
#   USE_FUSED_KERNELS (False), ROLLOUT_TP (2), CKPT_POLICY (full), SHM_MIN_FREE_GB, JUDGE_MAX_WAIT, IDLE_MIN,
#   PREEMPT_IDLE_MIN, MAX_ERRORS, MAX_IDLE_MIN, ADV_SCALE (drgrpo), NOISE_* (noisy).
#   SESSION (three_runs), ALLOW_CONCURRENT=1 (start although another supervisor is running),
#   PRINT_ONLY=1 or DRY_RUN=1: check the settings and print the plan, start nothing.
# Logs: outputs/logs/three_runs_queue.log (queue), outputs/logs/<run name>_supervisor.log.

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

QLOG=outputs/logs/three_runs_queue.log
SESSION=${SESSION:-three_runs}
ORDER=${ORDER:-baseline quantized noisy}
STEP0_VAL=${STEP0_VAL:-all}
RUN_TAG=${RUN_TAG:-3runs}
CKPT_ROOT=${CKPT_ROOT:-/dev/shm/frontiersmith_ckpt}
CKPT_POLICY=${CKPT_POLICY:-full}
TOTAL_STEPS=${TOTAL_STEPS:-100}
VAL_FREQ=${VAL_FREQ:-10}
VAL_N=${VAL_N:-5}
DROP_FINAL_OPTIM=${DROP_FINAL_OPTIM:-1}
declare -A LAUNCHER=(
    [baseline]=launch_baseline_grpo_synthetic.sh
    [quantized]=launch_quantized_grpo.sh
    [noisy]=launch_noisy_grpo.sh
    [drgrpo]=launch_drgrpo_const.sh
)
read -r -a RUNS <<< "$ORDER"

die() { echo "launch_three_runs.sh: $*" >&2; exit 2; }
qlog() { echo "[queue $(date -u '+%F %T')] $*" | tee -a "$QLOG"; }

run_name() {   # run_name <name>
    local var="RUN_NAME_$1"
    if [ -n "${!var:-}" ]; then echo "${!var}"; return; fi
    case $1 in
        baseline)  echo "qwen35_9b_grpo_$RUN_TAG" ;;
        quantized) echo "qwen35_9b_grpo_quantized_$RUN_TAG" ;;
        noisy)     echo "qwen35_9b_grpo_noisy_${NOISE_TYPE}_$RUN_TAG" ;;
        drgrpo)    echo "qwen35_9b_drgrpo_const_$RUN_TAG" ;;
    esac
}

val_before_train() {   # val_before_train <index in ORDER>
    case $STEP0_VAL in
        all) echo True ;;
        none) echo False ;;
        first) if [ "$1" -eq 0 ]; then echo True; else echo False; fi ;;
    esac
}

latest_step() { cat "$1/latest_checkpointed_iteration.txt" 2>/dev/null; }

# complete <run name> <latest step>: the checkpoint is at the final step and its validation is
# done (or disabled). A final checkpoint without its validation dump (e.g. the queue was stopped
# during the final validation) is NOT complete: its supervisor runs only that validation
# (trainer.val_only) and never trains past TOTAL_STEPS.
complete() {
    local f="outputs/validation_data_$1/$2.jsonl"
    [[ "$2" =~ ^[0-9]+$ ]] && [ "$2" -ge "$TOTAL_STEPS" ] &&
        { [ "$VAL_FREQ" = 0 ] || { [ -s "$f" ] && [ -z "$(tail -c 1 "$f")" ]; }; }
}

# Environment for one run's launcher: the shared knobs come from our environment (exported,
# or forwarded into tmux); the per-run ones are set here; stale per-run values are dropped.
run_env() {   # run_env <name> <index>  -> prints nothing; fills RUN_ENV
    local rn
    rn=$(run_name "$1")
    RUN_ENV=(-u SESSION -u LOG_PREFIX -u VAL_DIR -u TRAIN_ENTRY -u DRY_RUN -u PRINT_ONLY -u ADOPT
             NO_TMUX=1 RUN_NAME="$rn" CKPT_DIR="$CKPT_ROOT/$rn" VAL_BEFORE_TRAIN="$(val_before_train "$2")"
             TOTAL_STEPS="$TOTAL_STEPS" VAL_FREQ="$VAL_FREQ" VAL_N="$VAL_N" CKPT_POLICY="$CKPT_POLICY")
}

drop_final_optim() {   # drop_final_optim <ckpt dir>: delete ONLY optim_world_size_*_rank_*.pt of the final checkpoint
    local dir=$1 latest actor size
    local -a optim model
    latest=$(latest_step "$dir")
    if ! [[ "$latest" =~ ^[0-9]+$ ]] || [ "$latest" -lt "$TOTAL_STEPS" ]; then
        qlog "  keeping optimizer state: latest checkpoint '${latest:-none}' in $dir is not the final step $TOTAL_STEPS"
        return
    fi
    actor="$dir/global_step_$latest/actor"
    mapfile -t optim < <(find "$actor" -maxdepth 1 -type f -name 'optim_world_size_*_rank_*.pt' 2>/dev/null | sort)
    mapfile -t model < <(find "$actor" -maxdepth 1 -type f -name 'model_world_size_*_rank_*.pt' 2>/dev/null)
    if [ ${#optim[@]} -eq 0 ]; then
        qlog "  no optimizer shards in $actor; nothing deleted"
        return
    fi
    if [ ${#model[@]} -eq 0 ]; then
        qlog "  no model shards in $actor; keeping the optimizer shards"
        return
    fi
    size=$(du -ch "${optim[@]}" | tail -1 | cut -f1)
    if rm -f -- "${optim[@]}"; then
        qlog "  deleted ${#optim[@]} optimizer shards ($size) from $actor (kept ${#model[@]} model shards + extra/hf)"
    else
        qlog "  failed to delete the optimizer shards in $actor"
    fi
}

# compress_dumps <run name>: after a run exits 0, gzip its rollout/validation dumps (gzip -1 shrinks
# them ~3.7x) to spare the small root disk (200-step runs write ~1 GB of dumps each). The final
# step's validation dump stays uncompressed because complete() and the supervisor's resume logic
# look for <TOTAL_STEPS>.jsonl. Failed runs are left as they are (they may be resumed).
# COMPRESS_DUMPS=0 disables it.
compress_dumps() {
    local rn=$1 before after
    [ "${COMPRESS_DUMPS:-1}" = 1 ] || return 0
    before=$(du -sm "outputs/rollout_data_$rn" "outputs/validation_data_$rn" 2>/dev/null | awk '{s+=$1} END{print s+0}')
    find "outputs/rollout_data_$rn" -maxdepth 1 -type f -name '*.jsonl' -exec gzip -1 -f -- {} + 2>/dev/null
    find "outputs/validation_data_$rn" -maxdepth 1 -type f -name '*.jsonl' ! -name "$TOTAL_STEPS.jsonl" -exec gzip -1 -f -- {} + 2>/dev/null
    after=$(du -sm "outputs/rollout_data_$rn" "outputs/validation_data_$rn" 2>/dev/null | awk '{s+=$1} END{print s+0}')
    qlog "  compressed the dumps of $rn: ${before} MB -> ${after} MB (kept validation_data_$rn/$TOTAL_STEPS.jsonl uncompressed)"
}

worker() {
    local i name rn rc latest
    qlog "queue start: ORDER='$ORDER' STEP0_VAL=$STEP0_VAL CKPT_POLICY=$CKPT_POLICY TOTAL_STEPS=$TOTAL_STEPS VAL_FREQ=$VAL_FREQ VAL_N=$VAL_N BATCH=${TRAIN_BATCH_SIZE:-8}x${ROLLOUT_N:-8} MINI=${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE:-8}} RESP=${MAX_RESPONSE_LENGTH:-16000} FUSED=${USE_FUSED_KERNELS:-False} TP=${ROLLOUT_TP:-2} SAVE_FREQ=${SAVE_FREQ:-policy default} JUDGE_MAX_WAIT=${JUDGE_MAX_WAIT:-unset} SHM_MIN_FREE_GB=${SHM_MIN_FREE_GB:-policy default}"
    for i in "${!RUNS[@]}"; do
        name=${RUNS[$i]}
        rn=$(run_name "$name")
        latest=$(latest_step "$CKPT_ROOT/$rn")
        if complete "$rn" "$latest"; then
            qlog "$name ($rn): checkpoint at step $latest >= $TOTAL_STEPS and its validation is done, already complete; skipping"
            continue
        fi
        run_env "$name" "$i"
        qlog "starting $name (RUN_NAME=$rn, scripts/${LAUNCHER[$name]}, VAL_BEFORE_TRAIN=$(val_before_train "$i"), resume from: ${latest:-scratch}; /dev/shm free: $(df --output=avail -BG /dev/shm | tail -1 | tr -d ' '))"
        env "${RUN_ENV[@]}" PRINT_ONLY=1 bash "scripts/${LAUNCHER[$name]}" >> "$QLOG" 2>&1
        rc=0
        env "${RUN_ENV[@]}" bash "scripts/${LAUNCHER[$name]}" || rc=$?
        qlog "$name ($rn) supervisor exited rc=$rc; latest checkpoint: $(latest_step "$CKPT_ROOT/$rn" || echo none)"
        if [ "$rc" -eq 0 ] && [ "$CKPT_POLICY" = full ] && [ "$DROP_FINAL_OPTIM" = 1 ]; then
            drop_final_optim "$CKPT_ROOT/$rn"
        elif [ "$rc" -ne 0 ]; then
            qlog "$name failed; continuing with the next run"
        fi
        [ "$rc" -eq 0 ] && compress_dumps "$rn"
    done
    qlog "queue finished"
}

# Knobs forwarded into the tmux session (a tmux session gets the tmux server's environment,
# not ours) and inherited by each run's launcher; unset ones are removed explicitly so the
# worker sees exactly what the plan printed.
NOISE_VARS=(NOISE_TYPE NOISE_STD NOISE_P NOISE_LOW NOISE_HIGH NOISE_ON_ZERO NOISE_K NOISE_CLIP
            NOISE_CLIP_LOW NOISE_CLIP_HIGH NOISE_SEED)
forward_env() {
    local v
    FWD_ENV=()
    for v in ORDER STEP0_VAL RUN_TAG CKPT_ROOT CKPT_POLICY TOTAL_STEPS VAL_FREQ VAL_N SAVE_FREQ DROP_FINAL_OPTIM \
             TRAIN_BATCH_SIZE ROLLOUT_N PPO_MINI_BATCH_SIZE MAX_RESPONSE_LENGTH USE_FUSED_KERNELS ROLLOUT_TP \
             SHM_MIN_FREE_GB JUDGE_MAX_WAIT FRONTIERCS_MAX_WAIT IDLE_MIN PREEMPT_IDLE_MIN MAX_ERRORS \
             MAX_IDLE_MIN CONTAINER_NAME TRAIN_IMAGE VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS ADV_SCALE \
             RUN_NAME_baseline RUN_NAME_quantized RUN_NAME_noisy RUN_NAME_drgrpo "${NOISE_VARS[@]}" \
             $(compgen -e | grep -E '^NOISE_[A-Z0-9_]+$' | grep -vxF -f <(printf '%s\n' "${NOISE_VARS[@]}")); do
        if [ -n "${!v+x}" ]; then FWD_ENV+=("$v=${!v}"); else FWD_ENV=(-u "$v" "${FWD_ENV[@]}"); fi
    done
}

main() {
    if [ "${1:-}" = __worker ]; then
        worker
        return
    fi

    # ── checks before anything is started ──
    [ ${#RUNS[@]} -gt 0 ] || die "ORDER is empty"
    local name i rn latest state d seen=" " bad=()
    for name in "${RUNS[@]}"; do
        [ -n "${LAUNCHER[$name]:-}" ] || die "unknown run '$name' in ORDER (known: ${!LAUNCHER[*]})"
        [[ "$seen" == *" $name "* ]] && die "run '$name' appears twice in ORDER"
        seen+="$name "
    done
    case $STEP0_VAL in all|first|none) ;; *) die "STEP0_VAL must be all, first or none, got '$STEP0_VAL'" ;; esac
    case $CKPT_POLICY in full|model|final) ;; *) die "CKPT_POLICY must be full, model or final, got '$CKPT_POLICY'" ;; esac
    [[ "$TOTAL_STEPS" =~ ^[0-9]+$ ]] || die "TOTAL_STEPS must be an integer"
    if [[ " $ORDER " == *" noisy "* ]] && [ -z "${NOISE_TYPE:-}" ]; then
        die "'noisy' is in ORDER but NOISE_TYPE is unset (e.g. NOISE_TYPE=gaussian NOISE_STD=10)"
    fi
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
        die "tmux session '$SESSION' already exists; stop it first (tmux kill-session -t $SESSION)"
    fi
    if [ "${ALLOW_CONCURRENT:-0}" != 1 ] && pgrep -f '[s]upervise_training\.sh' >/dev/null; then
        die "another supervise_training.sh is running (pgrep -af supervise_training.sh); it would fight over the container. ALLOW_CONCURRENT=1 to start anyway"
    fi

    # Each launcher validates its own settings (CKPT_POLICY, NOISE_* kwargs, ...) and prints its command.
    echo "Plan (session '$SESSION', queue log $QLOG):"
    for i in "${!RUNS[@]}"; do
        name=${RUNS[$i]}
        rn=$(run_name "$name")
        latest=$(latest_step "$CKPT_ROOT/$rn")
        if [ -n "$latest" ]; then
            [ "${RESUME:-0}" = 1 ] || bad+=("$CKPT_ROOT/$rn (checkpoint step $latest)")
            if complete "$rn" "$latest"; then state="complete at step $latest: will be SKIPPED"
            elif [ "$latest" -ge "$TOTAL_STEPS" ] 2>/dev/null; then state="final checkpoint $latest without its validation dump: will only validate"
            else state="resumes from step $latest"; fi
        else
            state="from scratch"
            for d in "$CKPT_ROOT/$rn" "outputs/rollout_data_$rn" "outputs/validation_data_$rn"; do
                if [ "${RESUME:-0}" != 1 ] && [ -n "$(ls -A "$d" 2>/dev/null)" ]; then bad+=("$d (not empty)"); fi
            done
        fi
        echo "  $((i + 1)). $name: RUN_NAME=$rn, VAL_BEFORE_TRAIN=$(val_before_train "$i"), $state"
        run_env "$name" "$i"
        env "${RUN_ENV[@]}" PRINT_ONLY=1 bash "scripts/${LAUNCHER[$name]}" | sed 's/^/       /'
        [ "${PIPESTATUS[0]}" -eq 0 ] || die "scripts/${LAUNCHER[$name]} rejected its settings (see above)"
    done
    if [ ${#bad[@]} -gt 0 ]; then
        die "earlier outputs exist: ${bad[*]}. Set RESUME=1 to resume / skip these runs, or choose other names (RUN_TAG / RUN_NAME_<name>)"
    fi
    if [ "${PRINT_ONLY:-0}" = 1 ] || [ "${DRY_RUN:-0}" = 1 ]; then
        echo "PRINT_ONLY/DRY_RUN: nothing started."
        return 0
    fi

    mkdir -p outputs/logs
    forward_env
    tmux new -d -s "$SESSION" -c "$PROJECT_ROOT" \
        bash -c "env ${FWD_ENV[*]@Q} bash scripts/launch_three_runs.sh __worker; echo '[queue exited]'; exec bash"
    echo "Queue started in tmux session '$SESSION' (attach: tmux attach -t $SESSION)"
    echo "Queue log: $QLOG; per-run logs: outputs/logs/<RUN_NAME>_supervisor.log"
}

main "$@"
exit $?
