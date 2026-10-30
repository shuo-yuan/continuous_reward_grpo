#!/bin/bash
# Queue the matched comparison: baseline GRPO, then Dr.GRPO-const, back to back in one
# tmux session. Each run goes through the supervisor (preemption-safe, resumes from its
# own /dev/shm checkpoint); the second starts when the first has finished (or given up).
#
# Both runs: 200 synthetic problems, 100 steps, Frontier-CS validation (172 problems,
# 5 samples each) at step 0 and every 25 steps (= every epoch), identical settings
# except the advantage estimator / loss aggregation of the second run.
#
# Usage: bash scripts/launch_comparison.sh
#   env: TOTAL_STEPS (100), VAL_FREQ (25), VAL_N (5), SESSION (comparison)

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

SESSION=${SESSION:-comparison}
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists; stop it first (tmux kill-session -t $SESSION)." >&2
    exit 1
fi
export TOTAL_STEPS=${TOTAL_STEPS:-100} VAL_FREQ=${VAL_FREQ:-25} VAL_N=${VAL_N:-5}
QLOG=outputs/logs/comparison_queue.log
mkdir -p outputs/logs

tmux new -d -s "$SESSION" -c "$PROJECT_ROOT" "
  echo \"[queue \$(date -u '+%F %T')] starting baseline (RUN_NAME=qwen35_9b_grpo_val)\" >> $QLOG
  NO_TMUX=1 RUN_NAME=qwen35_9b_grpo_val bash scripts/launch_baseline_grpo_synthetic.sh; rc=\$?
  echo \"[queue \$(date -u '+%F %T')] baseline supervisor exited rc=\$rc\" >> $QLOG
  echo \"[queue \$(date -u '+%F %T')] starting drgrpo_const (RUN_NAME=qwen35_9b_drgrpo_const_val)\" >> $QLOG
  NO_TMUX=1 RUN_NAME=qwen35_9b_drgrpo_const_val bash scripts/launch_drgrpo_const.sh; rc=\$?
  echo \"[queue \$(date -u '+%F %T')] drgrpo_const supervisor exited rc=\$rc\" >> $QLOG
  echo '[queue finished]'; exec bash"

echo "Queue started in tmux session '$SESSION' (attach: tmux attach -t $SESSION)"
echo "Queue log: $QLOG; per-run logs: outputs/logs/qwen35_9b_{grpo,drgrpo_const}_val_*.log"
