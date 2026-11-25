#!/usr/bin/env bash
# Build notebook, push to Kaggle T4, poll, download SUMMARY.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
KDIR="$HERE/kernel"
PY="${PY:-$REPO/.venv/bin/python}"
KAGGLE="${KAGGLE:-$REPO/.venv/bin/kaggle}"
OUTROOT="${OUTROOT:-$REPO/out/minecu_hybrid}"
STAMP="$(date +%Y%m%d-%H%M%S)"
SESSION_MIN="${SESSION_MIN:-25}"   # minutes; kaggle -t wants seconds
QUEUE_WAIT_MIN="${QUEUE_WAIT_MIN:-60}"
POLL_SECS="${POLL_SECS:-20}"
SESSION_SECS=$((SESSION_MIN * 60))

DRY=0
BUILD_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --build-only) BUILD_ONLY=1; shift ;;
    *) echo "unknown $1" >&2; exit 64 ;;
  esac
done

mkdir -p "$OUTROOT" "$KDIR"
LOG="$OUTROOT/$STAMP-run.log"
say() { printf '%s\n' "$*" | tee -a "$LOG"; }

say "=== local API parity (CPU) ==="
PYTHONPATH="$REPO" "$PY" -m minecu_hybrid.parity_audit 2>&1 | tee -a "$LOG" | sed 's/^/  /'

say "=== assemble notebook ==="
"$PY" "$HERE/make_notebook.py" --repo "$REPO" --out "$KDIR/minecu_hybrid_t4.ipynb" 2>&1 | tee -a "$LOG"
# kernel-metadata.json already lives in $KDIR

if [ "$BUILD_ONLY" = 1 ]; then
  say "build-only; not pushing"
  exit 0
fi
if [ "$DRY" = 1 ]; then
  say "dry-run; not pushing"
  exit 0
fi

say "=== kaggle kernels push (session ${SESSION_MIN}m = ${SESSION_SECS}s) ==="
if ! "$KAGGLE" kernels push -p "$KDIR" -t "$SESSION_SECS" 2>&1 | tee -a "$LOG"; then
  say "push failed"
  exit 3
fi
SLUG="tenghuzhang/mc-minecu-hybrid-t4"
DEADLINE=$(( $(date +%s) + QUEUE_WAIT_MIN * 60 ))
while true; do
  ST="$("$KAGGLE" kernels status "$SLUG" 2>/dev/null | tr '\n' ' ' || true)"
  say "status: $ST"
  case "$ST" in
    *Complete*|*COMPLETE*) break ;;
    *Error*|*ERROR*|*Cancel*|*CANCEL*|*Failed*|*FAILED*)
      "$KAGGLE" kernels logs "$SLUG" 2>&1 | tee -a "$LOG" | tail -120
      exit 1
      ;;
  esac
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    say "queue wait exceeded"
    exit 2
  fi
  sleep "$POLL_SECS"
done

OUTDIR="$OUTROOT/$STAMP"
mkdir -p "$OUTDIR"
"$KAGGLE" kernels output "$SLUG" -p "$OUTDIR" 2>&1 | tee -a "$LOG"
say "=== output ==="
ls -la "$OUTDIR" | tee -a "$LOG"
if [ -f "$OUTDIR/SUMMARY.txt" ]; then
  cat "$OUTDIR/SUMMARY.txt" | tee -a "$LOG"
else
  say "WARN: no SUMMARY.txt — check kernel log"
  "$KAGGLE" kernels logs "$SLUG" 2>&1 | tee -a "$LOG" | tail -100
fi
