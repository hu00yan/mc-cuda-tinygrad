#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
KDIR="$HERE/kernel_lockstep"
PY="${PY:-$REPO/.venv/bin/python}"
KAGGLE="${KAGGLE:-$REPO/.venv/bin/kaggle}"
OUTROOT="${OUTROOT:-$REPO/out/minecu_lockstep}"
SESSION_MIN="${SESSION_MIN:-40}"
SESSION_SECS=$((SESSION_MIN * 60))
POLL_SECS=25
QUEUE_WAIT_MIN=90
SLUG="tenghuzhang/mc-mineenv-lockstep-t4"
STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTROOT" "$KDIR"
LOG="$OUTROOT/$STAMP-run.log"
say() { printf '%s\n' "$*" | tee -a "$LOG"; }

"$PY" "$HERE/make_lockstep_notebook.py" --repo "$REPO" --out "$KDIR/lockstep_t4.ipynb" | tee -a "$LOG"
say "push ${SESSION_MIN}m"
"$KAGGLE" kernels push -p "$KDIR" -t "$SESSION_SECS" 2>&1 | tee -a "$LOG"
DEADLINE=$(( $(date +%s) + QUEUE_WAIT_MIN * 60 ))
while true; do
  ST="$("$KAGGLE" kernels status "$SLUG" 2>/dev/null | tr '\n' ' ' || true)"
  say "status: $ST"
  case "$ST" in
    *Complete*|*COMPLETE*) break ;;
    *Error*|*ERROR*|*Cancel*|*CANCEL*|*Failed*)
      "$KAGGLE" kernels logs "$SLUG" 2>&1 | tee -a "$LOG" | tail -100
      exit 1 ;;
  esac
  [ "$(date +%s)" -ge "$DEADLINE" ] && { say timeout; exit 2; }
  sleep "$POLL_SECS"
done
OUT="$OUTROOT/$STAMP"
mkdir -p "$OUT"
"$KAGGLE" kernels output "$SLUG" -p "$OUT" 2>&1 | tee -a "$LOG" | tail -10
cat "$OUT/SUMMARY.txt" 2>/dev/null | tee -a "$LOG" || true
"$KAGGLE" kernels logs "$SLUG" 2>/dev/null | "$PY" -c '
import sys,re
raw=sys.stdin.read()
for m in re.finditer(r"\"data\":\"((?:\\\\.|[^\"\\\\])*)\"", raw):
  s=m.group(1).encode().decode("unicode_escape")
  if any(k in s for k in ("LOCKSTEP","FAIL","PASS","reset OK","step ","Error","Traceback","DONE")):
    print(s, end="")
' | tee -a "$LOG"
