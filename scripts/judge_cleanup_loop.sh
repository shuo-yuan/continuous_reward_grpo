#!/bin/bash
# Keep the judge's submission records from filling the shared disk.
#
# The judge (container Competitive-Programming) writes result.json per submission
# under Frontier-CS/algorithmic/submissions/<bucket>/<sid>/, including every test case's
# full program output (SAVE_OUTPUTS only governs the extra case_*.out files). Frontier-CS
# validation problems can emit 100+ MB per case, so one validation writes ~1.5-2 GB on a
# root disk with only a few GB free. A finished result is set in the judge's in-memory
# map BEFORE result.json is written, and GET /result/<sid> serves (and then drops) the
# in-memory copy; the disk is only read when the map has no entry (after a judge restart,
# or a repeat poll after the result was already delivered once). So a record whose job
# has FINISHED (result.json with status done/error) is dead weight once its client has
# polled it, which the reward client does every 2 s. Deleting one before that poll is
# harmless; a poll for a deleted record that is no longer in memory gets 404, which the
# reward client (verl/utils/reward_score/frontiercs.py) treats as "not ready" until
# FRONTIERCS_MAX_WAIT -> judge_status -1. Records without a final result are queued/
# running jobs and are never touched (deleting one makes the worker fail with ENOENT).
# Submission ids come from /app/data/counter.txt, not from the submissions dir, so
# deleting records never reuses an id. Files are root-owned, hence the deletion runs
# inside the container.
#
# Each pass deletes finished records whose result.json is older than MAX_AGE_MIN; if
# free space on / is below EMERGENCY_GB it deletes ALL finished records regardless of
# age (with SAVE_OUTPUTS=true in the judge this could race the case_*.out writes of a
# just-finished job; the judge runs with SAVE_OUTPUTS=false). Empty bucket dirs older
# than 60 min are removed too. Each pass logs the submissions dir size (du -sm).
#
# Usage: bash scripts/judge_cleanup_loop.sh      (runs forever; put it in tmux)
#   env: MAX_AGE_MIN (0.5; fractional minutes allowed, GNU find -mmin; the client polls every 2 s),
#        INTERVAL (20 s), EMERGENCY_GB (2), LOG,
#        DRY_RUN=1: one pass that only prints what would be deleted (to stdout), then exit.

MAX_AGE_MIN=${MAX_AGE_MIN:-0.5}
INTERVAL=${INTERVAL:-20}
EMERGENCY_GB=${EMERGENCY_GB:-2}
DRY_RUN=${DRY_RUN:-0}
JUDGE=Competitive-Programming
LOG=${LOG:-"$(cd "$(dirname "$0")/.." && pwd)/outputs/logs/judge_cleanup.log"}
mkdir -p "$(dirname "$LOG")"

# A validation submits ~850 programs at once and finished records pile up at ~1 GB/min, so the
# peak is roughly that rate x (MAX_AGE_MIN + INTERVAL); 1 min / 60 s let it take 2 GB on 2026-10-21.
if ! [[ $MAX_AGE_MIN =~ ^[0-9]+(\.[0-9]+)?$ ]]; then echo "MAX_AGE_MIN must be a non-negative number, got '$MAX_AGE_MIN'" >&2; exit 2; fi
for v in INTERVAL EMERGENCY_GB; do
    if ! [[ ${!v} =~ ^[0-9]+$ ]]; then echo "$v must be a non-negative integer, got '${!v}'" >&2; exit 2; fi
done

# Runs as root in the judge container. $1: minimum age in minutes or "any"; $2: dry run (1/0).
# Prints one line per deleted (or, dry run, deletable) record, then "<count> <du before> <du after>".
INNER='
age=$1 dry=$2 root=/app/submissions n=0
if [ "$age" = any ]; then agearg=""; else agearg="-mmin +$age"; fi
before=$(du -sm "$root" 2>/dev/null | cut -f1)
for f in $(find "$root" -mindepth 3 -maxdepth 3 -name result.json $agearg 2>/dev/null); do
    d=${f%/result.json}
    st=$(head -c 256 "$f" 2>/dev/null | sed -nE "s/.*\"status\": *\"(done|error)\".*/\1/p" | head -1)
    [ -n "$st" ] || continue
    n=$((n+1))
    if [ "$dry" = 1 ]; then
        echo "would delete $d (status $st, $(du -sm "$d" | cut -f1) MB, result.json from $(date -u -r "$f" "+%F %T") UTC)"
    else
        rm -rf "$d"
    fi
done
[ "$dry" = 1 ] || find "$root" -mindepth 1 -maxdepth 1 -type d -empty -mmin +60 -delete 2>/dev/null
echo "$n ${before:-?} $(du -sm "$root" 2>/dev/null | cut -f1)"
'

while true; do
    stamp=$(date -u '+%F %T')
    free_mb=$(df --output=avail -BM / | tail -1 | tr -dc 0-9)
    if [ "${free_mb:-0}" -lt $((EMERGENCY_GB * 1024)) ]; then
        age=any; what="finished records of ANY age (EMERGENCY: / has ${free_mb}M free < ${EMERGENCY_GB}G)"
    else
        age=$MAX_AGE_MIN; what="finished records older than ${MAX_AGE_MIN} min"
    fi
    if docker ps --format '{{.Names}}' | grep -qx "$JUDGE"; then
        out=$(docker exec "$JUDGE" sh -c "$INNER" sh "$age" "$DRY_RUN" 2>/dev/null)
        read -r n before after <<<"$(tail -1 <<<"$out")"
        if [ "$DRY_RUN" = 1 ]; then
            head -n -1 <<<"$out"
            echo "[$stamp] DRY RUN: would remove ${n:-?} $what; submissions dir ${before:-?} MB; disk free $(df -h / | awk 'NR==2{print $4}')"
        else
            echo "[$stamp] removed ${n:-?} $what; submissions dir ${before:-?} -> ${after:-?} MB (du -sm); disk free $(df -h / | awk 'NR==2{print $4}')" >> "$LOG"
        fi
    else
        echo "[$stamp] judge container not running" | if [ "$DRY_RUN" = 1 ]; then cat; else cat >> "$LOG"; fi
    fi
    [ "$DRY_RUN" = 1 ] && exit 0
    sleep "$INTERVAL"
done
