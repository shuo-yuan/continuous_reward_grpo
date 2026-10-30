#!/bin/bash
# Run a command inside the training container (Ubuntu 22.04 / glibc 2.35).
#
# Needed on hosts with glibc < 2.34 (e.g. Debian 11): vllm 0.20.0's compiled ops
# fail to load there. The project's .venv is reused unchanged — $HOME is mounted
# at the same path, so the uv-managed Python and all project paths resolve
# identically inside the container. The judge is reached via host networking.
#
# VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=600: vLLM kills the engine when a worker does not
# answer within this many seconds (vLLM's default 300). Raised to 1800 after workers stalled
# >300 s (RPC call to sample_tokens timed out) in case they were slow-but-alive (e.g. a kernel
# JIT compile), but none of the 9 stalls seen by 2026-10-22 ever recovered (a GPU kernel that
# never finished, see the gdn_prefill_backend note in launch_common.sh), so 600 s: a hung
# attempt is restarted 20 min sooner, still far above a JIT compile.
#
# VLLM_USE_DEEP_GEMM=0: the DeepGEMM copy bundled in the vllm 0.20.0+cu129 wheel
# is built only for Python 3.12 (this venv is 3.11), yet vLLM still detects it and
# crashes in kernel warmup. DeepGEMM only serves FP8 models; Qwen3.5 here is bf16.
#
# /tmp is a RAM-backed tmpfs (size cap 64g, pages used only as written): Ray's session
# dir (/tmp/ray, ~2 GB of logs per run) and other /tmp caches would otherwise land in the
# container's writable layer on the nearly full root disk. Like the writable layer, it
# is discarded when the container exits.
#
# --ulimit core=0: no core dumps. A crashing vLLM worker (2026-10-21 20:29, NCCL DistBackendError)
# wrote a 2.5 GB core file into the project dir (the container's cwd) and filled the root disk.
#
# Passed through only when set in the caller's environment (docker's bare `-e VAR`):
# TRAIN_ENTRY, ADV_SCALE. FRONTIERCS_MAX_WAIT (seconds to wait for a Frontier-CS judge
# verdict, default 1800) is passed only when non-empty: it is read at import time by
# verl/utils/reward_score/frontiercs.py in the reward workers (Ray actors, which inherit
# the container env), and an empty value would make every reward worker fail on float('').
#
# Usage:
#   bash scripts/run_in_container.sh bash scripts/run_verl_grpo_frontiercs_qwen35_9b_synthetic.sh [overrides...]
#   bash scripts/run_in_container.sh bash        # interactive shell

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE=${TRAIN_IMAGE:-frontiersmith-train}
CONTAINER_NAME=${CONTAINER_NAME:-frontiersmith-train}

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[container] Building $IMAGE ..."
    docker build -t "$IMAGE" -f "$PROJECT_ROOT/docker/Dockerfile.train" "$PROJECT_ROOT/docker"
fi

TTY_ARGS=(-i)
if [ -t 0 ] && [ -t 1 ]; then
    TTY_ARGS+=(-t)
fi

exec docker run --rm "${TTY_ARGS[@]}" \
    --name "$CONTAINER_NAME" \
    --init \
    --gpus all \
    -e NVIDIA_DISABLE_REQUIRE=1 \
    -e VLLM_USE_DEEP_GEMM=0 \
    -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-600} \
    -e TRAIN_ENTRY -e ADV_SCALE ${FRONTIERCS_MAX_WAIT:+-e FRONTIERCS_MAX_WAIT} \
    --tmpfs /tmp:rw,exec,mode=1777,size=64g \
    --network host \
    --ipc host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ulimit core=0 \
    --user "$(id -u):$(id -g)" \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -e HOME="$HOME" \
    -v "$HOME:$HOME" \
    -w "$PROJECT_ROOT" \
    "$IMAGE" "$@"
