#!/usr/bin/env bash
# retry.sh -- push the axis kernel as soon as a T4 session frees up.
#
# WHY THIS EXISTS
# ---------------
# Kaggle's free tier allows 2 concurrent batch GPU sessions per account, and this
# study's other agents are using both. `kaggle kernels push` prints
#
#     Kernel push error: Maximum batch GPU session count of 2 reached.
#
# and EXITS 0. An earlier version of run.sh ignored the message, went on to poll
# for a status, saw the PREVIOUS run's COMPLETE, and downloaded its output --
# which is indistinguishable from a successful run of the current code. Two
# rounds of the harness were "measured" that way, against a build that predated
# the fixes. So: the push message is checked, and this loop simply keeps trying
# until it succeeds.
#
# COST
# ----
# No GPU quota until a push actually succeeds. The loop itself is free.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
KAGGLE="${KAGGLE:-$REPO/.venv/bin/kaggle}"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$REPO/out/axis/$STAMP-retry.log"
mkdir -p "$(dirname "$LOG")"

ID=tenghuzhang/mc-axis-t4
MAX="${MAX:-180}"          # attempts; 60s apart is ~3h

for i in $(seq 1 "$MAX"); do
  ts=$(date +%H:%M:%S)
  out=$("$KAGGLE" kernels push -p "$HERE/kernel" -t 1500 2>&1)
  if echo "$out" | grep -q "successfully pushed"; then
    printf '%s  PUSHED (attempt %d)\n' "$ts" "$i" | tee -a "$LOG"
    printf '%s\n' "$out" | tee -a "$LOG"
    # Poll to completion. `kaggle kernels status` prints a SENTENCE
    #   'tenghuzhang/mc-axis-t4 has status "KernelWorkerStatus.COMPLETE"'
    # so the terminal state is extracted with a pattern match on the word. The
    # first version compared the whole line against `COMPLETE*` and therefore
    # never matched anything: it looped 60 times against an already-COMPLETE run
    # and the loop's own success was indistinguishable from waiting.
    for j in $(seq 1 60); do
      line=$("$KAGGLE" kernels status "$ID" 2>/dev/null | head -1 | tr -d '\r')
      st=$(printf '%s' "$line" | grep -oE '(RUNNING|COMPLETE|ERROR|FAIL|CANCELLED|QUEUED|PENDING|INITIALIZING)[A-Z_]*' | head -1)
      printf '%s  %s\n' "$(date +%H:%M:%S)" "${st:-$line}" | tee -a "$LOG"
      case "$st" in
        COMPLETE*|ERROR*|FAIL*|CANCEL*) break ;;
      esac
      sleep 60
    done
    DEST="$REPO/out/axis/$STAMP"
    mkdir -p "$DEST"
    rm -rf "$DEST"
    "$KAGGLE" kernels output "$ID/1" -p "$DEST" -o -q 2>&1 | tee -a "$LOG" || true
    printf '%s  fetched -> %s\n' "$(date +%H:%M:%S)" "$DEST" | tee -a "$LOG"
    ls -la "$DEST" | tee -a "$LOG"
    exit 0
  fi
  printf '%s  attempt %d: %s\n' "$ts" "$i" "$(echo "$out" | head -1)" | tee -a "$LOG"
  sleep 60
done
printf 'gave up after %d attempts; see %s\n' "$MAX" "$LOG" | tee -a "$LOG"
exit 1
