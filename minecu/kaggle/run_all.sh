#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
KDIR="$HERE/kernel_all"
PY="${PY:-$REPO/.venv/bin/python}"
KAGGLE="${KAGGLE:-$REPO/.venv/bin/kaggle}"
OUTROOT="${OUTROOT:-$REPO/out/minecu_all}"
STAMP="$(date +%Y%m%d-%H%M%S)"
SESSION_MIN="${SESSION_MIN:-45}"
SESSION_SECS=$((SESSION_MIN * 60))
QUEUE_WAIT_MIN="${QUEUE_WAIT_MIN:-90}"
POLL_SECS="${POLL_SECS:-25}"
SLUG="tenghuzhang/mc-hybrid-all-t4"

mkdir -p "$OUTROOT" "$KDIR"
LOG="$OUTROOT/$STAMP-run.log"
say() { printf '%s\n' "$*" | tee -a "$LOG"; }

say "=== assemble ==="
"$PY" "$HERE/make_all_notebook.py" --repo "$REPO" --out "$KDIR/hybrid_all_t4.ipynb" | tee -a "$LOG"
# kernel-metadata.json already in $KDIR

say "=== push ${SESSION_MIN}m ==="
"$KAGGLE" kernels push -p "$KDIR" -t "$SESSION_SECS" 2>&1 | tee -a "$LOG"

DEADLINE=$(( $(date +%s) + QUEUE_WAIT_MIN * 60 ))
while true; do
  ST="$("$KAGGLE" kernels status "$SLUG" 2>/dev/null | tr '\n' ' ' || true)"
  say "status: $ST"
  case "$ST" in
    *Complete*|*COMPLETE*) break ;;
    *Error*|*ERROR*|*Cancel*|*CANCEL*|*Failed*)
      "$KAGGLE" kernels logs "$SLUG" 2>&1 | tee -a "$LOG" | tail -150
      exit 1
      ;;
  esac
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    say "queue wait exceeded"; exit 2
  fi
  sleep "$POLL_SECS"
done

OUTDIR="$OUTROOT/$STAMP"
mkdir -p "$OUTDIR"
"$KAGGLE" kernels output "$SLUG" -p "$OUTDIR" 2>&1 | tee -a "$LOG" | tail -20
say "=== SUMMARY ==="
cat "$OUTDIR/SUMMARY.txt" 2>/dev/null | tee -a "$LOG" || true
"$KAGGLE" kernels logs "$SLUG" 2>&1 | "$PY" -c '
import sys,re
raw=sys.stdin.read()
for m in re.finditer(r"\"data\":\"((?:\\\\.|[^\"\\\\])*)\"", raw):
    s=m.group(1).encode().decode("unicode_escape")
    if any(k in s for k in ("===","PASS","FAIL","sps","relerr","bench","DONE","obs_parity","parity ok","WARN","Error","Traceback","maxabs","env-steps","reward")):
        print(s, end="")
' | tee -a "$LOG"
