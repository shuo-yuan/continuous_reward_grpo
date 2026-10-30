# Shared part of the scripts/launch_*.sh training launchers (Qwen3.5-9B GRPO on the 200
# synthetic problems, Frontier-CS validation). SOURCED, not executed:
#
#   PROJECT_ROOT=...; cd "$PROJECT_ROOT"          # the launcher's own setup
#   RUN_NAME=${RUN_NAME:-<launcher default>}
#   ALGO_OVERRIDES=(...)                          # launcher-specific Hydra overrides (may be empty)
#   export TRAIN_ENTRY=... ADV_SCALE=...          # only if the launcher needs them
#   source "$PROJECT_ROOT/scripts/launch_common.sh" "$@"
#
# It adds the common Hydra overrides (on top of run_verl_grpo_frontiercs_qwen35_9b_synthetic.sh:
# paper settings, 16k responses, lr 1e-6, batch 8 x group 8), exports the supervisor's
# environment and launches scripts/supervise_training.sh (auto-resume after preemption).
#
# Env knobs (identical for every launcher):
#   RUN_NAME          -> wandb experiment name, log prefix, outputs/{rollout,validation}_data_$RUN_NAME
#   TOTAL_STEPS (100), VAL_FREQ (10; 0 disables periodic validation), VAL_N (5 samples/problem),
#   TRAIN_BATCH_SIZE (8 prompts/step), ROLLOUT_N (8 samples/prompt), PPO_MINI_BATCH_SIZE
#                     (= TRAIN_BATCH_SIZE: one optimizer step per training step). Micro-batches stay
#                     at 1 sequence/GPU (no dynamic bsz): packing sequences would leak Qwen3.5's
#                     linear-attention state across them.
#   MAX_RESPONSE_LENGTH (16000 tokens; also the validation limit), USE_FUSED_KERNELS (False):
#                     True computes log-probs from hidden states in chunks (verl FusedLinearForPPO,
#                     torch backend) instead of materializing (tokens x 248k-vocab) logits. Required
#                     for 32k responses: unfused, the actor update OOMs (smoke test 2026-10-21) --
#                     and it made the update ~2.5x faster per token.
#   ROLLOUT_TP (2)    vLLM tensor parallel size per rollout replica (8/TP replicas). 1 = one GPU
#                     per replica, no inter-GPU collectives in vLLM: TP=2 replicas hung in an
#                     NCCL collective 3 times on 2026-10-21/22 (watchdog kill after 10 min).
#   VAL_BEFORE_TRAIN (True: validate the step-0 model / the resumed checkpoint)
#   CKPT_POLICY  full  (default) full checkpoint (fp32 weights + Adam, ~103 GiB) every 10 steps,
#                      keep 1; exact resume
#                model weights + extra only (~35 GiB) every 10 steps, keep 1; a resume
#                      reloads the weights and resets Adam
#                final weights + extra only, saved once at the last step; a preemption
#                      restarts the run from step 0
#   SAVE_FREQ         override the policy's save frequency (10, 10, TOTAL_STEPS)
#   SHM_MIN_FREE_GB   free /dev/shm the supervisor needs before an attempt
#                     (default: full 120, model 45, final 45)
#   JUDGE_MAX_WAIT    (unset) -> FRONTIERCS_MAX_WAIT: seconds a reward call waits for a judge
#                     verdict (verl's default 1800)
#   CKPT_DIR (/dev/shm/frontiersmith_ckpt/$RUN_NAME), SESSION (tmux session, $RUN_NAME),
#   IDLE_MIN (3), PREEMPT_IDLE_MIN (10); MAX_ERRORS, MAX_IDLE_MIN, ADOPT, CONTAINER_NAME,
#   TRAIN_IMAGE, VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS are forwarded when set.
#   DRY_RUN=1     compose the Hydra config in the container (--cfg job), no training
#   PRINT_ONLY=1  print the supervisor command line (with its environment) and exit
#   NO_TMUX=1     run the supervisor in the foreground (used by the queues)
#
# Per-run override file outputs/run_overrides/$RUN_NAME.env: KEY=VALUE lines (blank lines and
# # comments allowed; read, never sourced) that win over the environment for that run name only.
# Only VAL_FREQ and VAL_BEFORE_TRAIN (True|False) are accepted. Use: change a setting of runs that
# scripts/launch_three_runs.sh has queued but not started yet (the queue hands every launcher the
# knobs it was started with).
# The override is printed to stderr (the queue copies it into its log with the run's command).
#
# The supervisor also gets VAL_DIR (= trainer.validation_data_dir), VAL_FREQ and TOTAL_STEPS, to
# decide whether a resumed attempt has to re-run validation, and to never resume a finished run
# (verl would train a step past TOTAL_STEPS).

_lc_die() { echo "$(basename "$0"): $*" >&2; exit 2; }
_lc_int() { [[ "$2" =~ ^[0-9]+$ ]] || _lc_die "$1 must be a non-negative integer, got '$2'"; }

: "${PROJECT_ROOT:?set PROJECT_ROOT before sourcing launch_common.sh}"
: "${RUN_NAME:?set RUN_NAME before sourcing launch_common.sh}"
[[ "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || _lc_die "RUN_NAME may only contain letters, digits, '.', '_' and '-', got '$RUN_NAME'"

SESSION=${SESSION:-$RUN_NAME}
CKPT_DIR=${CKPT_DIR:-/dev/shm/frontiersmith_ckpt/$RUN_NAME}

RUN_OVERRIDES_FILE="$PROJECT_ROOT/outputs/run_overrides/$RUN_NAME.env"
if [ -f "$RUN_OVERRIDES_FILE" ]; then
    while IFS= read -r _line || [ -n "$_line" ]; do
        _line=${_line%%#*}
        _line=${_line//[[:space:]]/}
        [ -n "$_line" ] || continue
        [[ "$_line" == *=* ]] || _lc_die "$RUN_OVERRIDES_FILE: expected KEY=VALUE, got '$_line'"
        _key=${_line%%=*} _val=${_line#*=}
        case $_key in
            VAL_FREQ) _lc_int "$_key" "$_val" ;;
            VAL_BEFORE_TRAIN)
                case $_val in True|False) ;; *) _lc_die "$RUN_OVERRIDES_FILE: VAL_BEFORE_TRAIN must be True or False, got '$_val'" ;; esac ;;
            *) _lc_die "$RUN_OVERRIDES_FILE: only VAL_FREQ and VAL_BEFORE_TRAIN can be overridden, got '$_key'" ;;
        esac
        echo "$(basename "$0"): $_key=$_val from $RUN_OVERRIDES_FILE (environment had: ${!_key:-unset})" >&2
        printf -v "$_key" '%s' "$_val"
    done < "$RUN_OVERRIDES_FILE"
    unset _line _key _val
fi

TOTAL_STEPS=${TOTAL_STEPS:-100}
VAL_FREQ=${VAL_FREQ:-10}
VAL_N=${VAL_N:-5}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
_lc_int TOTAL_STEPS "$TOTAL_STEPS"
_lc_int VAL_FREQ "$VAL_FREQ"
_lc_int VAL_N "$VAL_N"
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
ROLLOUT_N=${ROLLOUT_N:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
for _v in TRAIN_BATCH_SIZE ROLLOUT_N PPO_MINI_BATCH_SIZE; do
    _lc_int "$_v" "${!_v}"
    [ "${!_v}" -gt 0 ] || _lc_die "$_v must be positive"
done
[ $((TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE)) -eq 0 ] ||
    _lc_die "TRAIN_BATCH_SIZE ($TRAIN_BATCH_SIZE) must be a multiple of PPO_MINI_BATCH_SIZE ($PPO_MINI_BATCH_SIZE)"
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16000}
_lc_int MAX_RESPONSE_LENGTH "$MAX_RESPONSE_LENGTH"
USE_FUSED_KERNELS=${USE_FUSED_KERNELS:-False}
ROLLOUT_TP=${ROLLOUT_TP:-2}
case $ROLLOUT_TP in 1|2|4|8) ;; *) _lc_die "ROLLOUT_TP must be 1, 2, 4 or 8, got '$ROLLOUT_TP'" ;; esac
case $USE_FUSED_KERNELS in
    True|False) ;;
    *) _lc_die "USE_FUSED_KERNELS must be True or False, got '$USE_FUSED_KERNELS'" ;;
esac
[ $((PPO_MINI_BATCH_SIZE * ROLLOUT_N % 8)) -eq 0 ] ||
    _lc_die "PPO_MINI_BATCH_SIZE x ROLLOUT_N ($((PPO_MINI_BATCH_SIZE * ROLLOUT_N))) must be divisible by the 8 GPUs"
case $VAL_BEFORE_TRAIN in
    True|False|true|false) ;;
    *) _lc_die "VAL_BEFORE_TRAIN must be True or False, got '$VAL_BEFORE_TRAIN'" ;;
esac

CKPT_POLICY=${CKPT_POLICY:-full}
CKPT_OVERRIDES=()
case $CKPT_POLICY in
    full)  _save_freq=10;           _shm_gb=120 ;;
    model) _save_freq=10;           _shm_gb=45 ;;
    final) _save_freq=$TOTAL_STEPS; _shm_gb=45 ;;
    *) _lc_die "CKPT_POLICY must be full, model or final, got '$CKPT_POLICY'" ;;
esac
if [ "$CKPT_POLICY" != full ]; then
    CKPT_OVERRIDES=(
        "actor_rollout_ref.actor.checkpoint.save_contents=['model','extra']"
        "actor_rollout_ref.actor.checkpoint.load_contents=['model','extra']"
    )
fi
SAVE_FREQ=${SAVE_FREQ:-$_save_freq}
SHM_MIN_FREE_GB=${SHM_MIN_FREE_GB:-$_shm_gb}
_lc_int SAVE_FREQ "$SAVE_FREQ"
_lc_int SHM_MIN_FREE_GB "$SHM_MIN_FREE_GB"

if [ -n "${JUDGE_MAX_WAIT:-}" ]; then
    _lc_int JUDGE_MAX_WAIT "$JUDGE_MAX_WAIT"
    FRONTIERCS_MAX_WAIT=$JUDGE_MAX_WAIT
fi

VAL_DIR="$PROJECT_ROOT/outputs/validation_data_$RUN_NAME"
LOG_PREFIX=$RUN_NAME
IDLE_MIN=${IDLE_MIN:-3}
PREEMPT_IDLE_MIN=${PREEMPT_IDLE_MIN:-10}

# Gated-DeltaNet prefill kernel: Triton/FLA instead of vLLM 0.20's default on H100 (FlashInfer's SM90
# kernel). With the default, a vLLM replica now and then hung for good in the first mixed prefill+decode
# step of a generation burst (9 times by 2026-10-22, mostly right at a validation start; one GPU stuck at
# 100% util / idle power, at TP=1 and TP=2 alike). Upstream reports of the same hang on Qwen3.5 / H100 were
# fixed by this backend (vllm #37729, #35502, #36921); EasyPPO's config uses it too.
OVERRIDES=(
    "${ALGO_OVERRIDES[@]}"
    trainer.experiment_name="$RUN_NAME"
    trainer.rollout_data_dir="$PROJECT_ROOT/outputs/rollout_data_$RUN_NAME"
    trainer.validation_data_dir="$VAL_DIR"
    trainer.val_before_train="$VAL_BEFORE_TRAIN"
    trainer.test_freq="$VAL_FREQ"
    data.train_batch_size="$TRAIN_BATCH_SIZE"
    actor_rollout_ref.rollout.n="$ROLLOUT_N"
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE"
    data.max_response_length="$MAX_RESPONSE_LENGTH"
    actor_rollout_ref.model.use_fused_kernels="$USE_FUSED_KERNELS"
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton
    actor_rollout_ref.rollout.val_kwargs.n="$VAL_N"
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.ref.fsdp_config.param_offload=False
    trainer.total_training_steps="$TOTAL_STEPS"
    trainer.default_local_dir="$CKPT_DIR"
    trainer.save_freq="$SAVE_FREQ"
    trainer.max_actor_ckpt_to_keep=1
    "${CKPT_OVERRIDES[@]}"
    "$@"
)

# Environment of the supervisor (and, through run_in_container.sh, of the container).
# Exported for the foreground paths; written into the command for tmux (a tmux session gets
# the tmux server's environment, not ours). Unset variables are removed explicitly so
# nothing stale leaks in from the tmux server's environment.
FWD_VARS=(LOG_PREFIX CKPT_DIR IDLE_MIN PREEMPT_IDLE_MIN SHM_MIN_FREE_GB VAL_DIR VAL_FREQ TOTAL_STEPS
          FRONTIERCS_MAX_WAIT TRAIN_ENTRY ADV_SCALE
          MAX_ERRORS MAX_IDLE_MIN ADOPT CONTAINER_NAME TRAIN_IMAGE VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS)
FWD_ENV=()
for _v in "${FWD_VARS[@]}"; do
    if [ -n "${!_v+x}" ]; then
        export "$_v"
        FWD_ENV+=("$_v=${!_v}")
    else
        FWD_ENV=(-u "$_v" "${FWD_ENV[@]}")
    fi
done
unset _v _save_freq _shm_gb

RUN_SCRIPT=scripts/run_verl_grpo_frontiercs_qwen35_9b_synthetic.sh
CMD=(bash scripts/supervise_training.sh bash scripts/run_in_container.sh bash "$RUN_SCRIPT" "${OVERRIDES[@]}")
CMD_STR="env ${FWD_ENV[*]@Q} ${CMD[*]@Q}"

if [ "${DRY_RUN:-0}" = "1" ]; then   # compose the Hydra config in the container, no training
    exec bash scripts/run_in_container.sh bash "$RUN_SCRIPT" "${OVERRIDES[@]}" --cfg job
fi
if [ "${PRINT_ONLY:-0}" = "1" ]; then
    echo "cd ${PROJECT_ROOT@Q} && $CMD_STR"
    exit 0
fi

mkdir -p "$CKPT_DIR" outputs/logs
if [ "${NO_TMUX:-0}" = "1" ]; then
    exec env "${FWD_ENV[@]}" "${CMD[@]}"
fi

if tmux has-session -t "=$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists; stop it first (tmux kill-session -t $SESSION)." >&2
    exit 1
fi
tmux new -d -s "$SESSION" -c "$PROJECT_ROOT" bash -c "$CMD_STR; echo '[supervisor exited]'; exec bash"
echo "Supervisor started in tmux session '$SESSION' (attach: tmux attach -t $SESSION)"
echo "Supervisor log: outputs/logs/${RUN_NAME}_supervisor.log"
