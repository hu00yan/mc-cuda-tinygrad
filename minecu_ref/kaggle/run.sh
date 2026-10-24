#!/usr/bin/env bash
# run.sh -- one command, bounded, self-verifying: measure the mine.cu
#           hand-written CUDA baseline on Kaggle's free T4 (sm_75).
#
#   ./run.sh --dry-run     everything except the network calls
#   ./run.sh --build-only  compile for sm_75 in the container, push nothing
#   ./run.sh               build, push, poll, download, print
#
# WHY A KAGGLE KERNEL AND NOT A DATASET. `kaggle kernels push` uploads exactly
# one file: the notebook named by "code_file". Sidecars in the kernel directory
# are not uploaded, so the two CUDA sources have to travel inside the notebook.
# They are base64'd into a single code cell (kernels.cu is 105 KB, bench.cu is
# ~30 KB) and the notebook verifies both sha256 values before compiling. A
# mismatch is a hard failure, not a warning: a timing from a different source
# than the one the note describes is worse than no timing.
#
# COST. One Kaggle GPU run. The battery below is bounded to well under the
# session cap, and the session cap itself is passed to `kernels push -t` so
# Kaggle enforces it server-side. One full run is a small fraction of the ~30
# GPU-hours/week free tier.
#
# NO GPU ON THIS HOST. Nothing here needs one. The build step is a container
# compile; the measurement step happens on Kaggle.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KDIR="$HERE/kaggle/kernel"
PY="${PY:-$HERE/../.venv/bin/python}"
KAGGLE="${KAGGLE:-$HERE/../.venv/bin/kaggle}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUTROOT="${OUTROOT:-$HERE/../out/minecu_ref}"

DRY=0
BUILD_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)    DRY=1; shift ;;
    --build-only) BUILD_ONLY=1; shift ;;
    -h|--help)    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 64 ;;
  esac
done

# --------------------------------------------------------------- the battery
# Bounded: selftest at three shapes, then the measurement battery at the
# project's own headline shape, then a batch sweep. Total in-GPU work is a few
# minutes. Every entry is (label, argv-tail).
BATTERY_S=1500   # total budget in seconds, enforced in the notebook
SESSION_MIN="${SESSION_MIN:-25}"    # Kaggle server-side cap
QUEUE_WAIT_MIN="${QUEUE_WAIT_MIN:-120}"
POLL_SECS="${POLL_SECS:-30}"

mkdir -p "$OUTROOT"
LOG="$OUTROOT/$STAMP-run.log"
say() { printf '%s\n' "$*" | tee -a "$LOG"; }

say "=== minecu_ref T4 baseline: build for sm_75 ==="
say "  staged $(date +%H:%M:%S)"
if [ "$DRY" = 1 ]; then
  say "  would run: $HERE/build.sh sm_75"
  say "  then: compile, hash-check the census, assemble the notebook, push, poll"
  say "  DRY RUN: no build and no network calls made."
  exit 0
fi
"$HERE/build.sh" sm_75 2>&1 | tail -20 | sed 's/^/  /' | tee -a "$LOG"
if ! grep -q "no sm_75-illegal instructions" "$HERE/build/sass_census_sm_75.txt" 2>/dev/null; then
  say "  the sm_75 SASS census did not pass; refusing to push"
  exit 1
fi
say "  sm_75 SASS census passed"
[ "$BUILD_ONLY" = 1 ] && { say "build-only: done"; exit 0; }

if [ ! -x "$PY" ]; then
  say "python not found at $PY (set PY=...)"
  exit 3
fi

# ------------------------------------------------------- assemble the kernel
say ""
say "=== assemble the notebook ==="
"$PY" "$HERE/kaggle/make_notebook.py" \
    --kernels "$HERE/src/kernels.cu" \
    --bench   "$HERE/bench/bench.cu" \
    --out     "$KDIR/minecu_t4.ipynb" \
    --budget-s "$BATTERY_S" 2>&1 | sed 's/^/  /' | tee -a "$LOG"
"$PY" - "$KDIR/minecu_t4.ipynb" <<'PY' 2>&1 | sed 's/^/  /' | tee -a "$LOG"
import json, sys
nb = json.load(open(sys.argv[1]))
assert nb.get("metadata", {}).get("kernelspec"), "no kernelspec"
assert nb.get("nbformat") == 4, f"nbformat {nb.get('nbformat')}"
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        compile("".join(c["source"]), f"cell{i}", "exec")
print(f"notebook valid: {len(nb['cells'])} cells, "
      f"{sum(len(''.join(c['source'])) for c in nb['cells'])} source chars, all code cells compile")
PY

if [ "$DRY" = 1 ]; then
  say ""
  say "DRY RUN: no network calls made. would:"
  say "  kaggle kernels push -p $KDIR -t $((SESSION_MIN * 60))"
  say "  poll kaggle kernels status until COMPLETE, then kaggle kernels output"
  exit 0
fi

# ------------------------------------------------------------------ auth
say ""
say "=== auth ==="
[ -x "$KAGGLE" ] || { say "kaggle CLI not found at $KAGGLE (set KAGGLE=...)"; exit 3; }
if [ -z "${kaggle_API_TOKEN:-}" ] && [ ! -s "$HOME/.kaggle/access_token" ] \
   && [ ! -s "${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json" ]; then
  say "no Kaggle credentials. Authenticate once, either:"
  say "    $KAGGLE auth login"
  say "or write a token to   ~/.kaggle/access_token   (mode 600)"
  say "  get one at https://www.kaggle.com/settings/api"
  exit 2
fi
say "credentials present"

# ------------------------------------------------------------------ push
KID="$(sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
        "$KDIR/kernel-metadata.json" | head -1)"
[ -n "$KID" ] || { say "no \"id\" in kernel-metadata.json"; exit 64; }
OWNER="${OWNER:-${KID%%/*}}"
SLUG="${KID#*/}"
REF="$OWNER/$SLUG"

say ""
say "=== push $REF (session cap ${SESSION_MIN}m) ==="
if ! "$KAGGLE" kernels push -p "$KDIR" -t "$((SESSION_MIN * 60))" 2>&1 \
     | tee -a "$LOG" | sed 's/^/  /'; then
  say "push failed; nothing was started"
  exit 1
fi

# ------------------------------------------------------------------ poll
say ""
say "=== wait (queue up to ${QUEUE_WAIT_MIN}m, session cap ${SESSION_MIN}m) ==="
deadline=$(( $(date +%s) + QUEUE_WAIT_MIN * 60 ))
status=UNKNOWN
last=""
while :; do
  [ "$(date +%s)" -ge "$deadline" ] && { say "  gave up waiting; the run may still be going. stop it at https://www.kaggle.com/code/$REF"; exit 4; }
  out="$("$KAGGLE" kernels status "$REF" 2>&1)" || true
  st="$(printf '%s' "$out" | sed -n 's/.*status "\(.*\)".*/\1/p' | head -1)"
  st="${st##*.}"; [ -z "$st" ] && st=UNKNOWN
  if [ "$st" != "$last" ]; then say "  [$(date +%H:%M:%S)] $st"; last="$st"; fi
  status="$st"
  case "$st" in
    COMPLETE) break ;;
    ERROR|CANCELLED|FAILED) break ;;
    UNKNOWN)
      if printf '%s' "$out" | grep -q "Cannot access kernel"; then
        say "  $out"; say "  the slug in kernel-metadata.json does not exist for $OWNER"
        exit 5
      fi ;;
  esac
  sleep "$POLL_SECS"
done

if [ "$status" != COMPLETE ]; then
  say ""
  say "=== kernel log (this is where the traceback is) ==="
  "$KAGGLE" kernels logs "$REF" 2>&1 | tail -150 | tee -a "$LOG" | sed 's/^/  /'
  say "run did not complete: $status"
  exit 6
fi

# ------------------------------------------------------------------ output
DEST="$OUTROOT/$STAMP"
mkdir -p "$DEST"
say ""
say "=== download ==="
got=0
for attempt in 1 2 3; do
  if "$KAGGLE" kernels output "$REF" -p "$DEST" --force 2>&1 \
       | tee -a "$LOG" | sed 's/^/  /'; then got=1; break; fi
  say "  attempt $attempt/3 failed; sleeping 20s"; sleep 20
done
[ "$got" = 1 ] || { say "all 3 download attempts failed; partial output left in $DEST"; exit 7; }

while IFS= read -r z; do
  [ -n "$z" ] || continue
  unzip -qo "$z" -d "$DEST/$(basename "$z" .zip)" 2>&1 | sed 's/^/    /'
done < <(find "$DEST" -name '*.zip' -type f | sort)

say ""
say "=== what came back ==="
find "$DEST" -type f | sort | while read -r f; do
  printf '    %-56s %8s\n' "${f#$DEST/}" "$(du -h "$f" | cut -f1)"
done | tee -a "$LOG"

for s in "$DEST"/*/SUMMARY.txt "$DEST"/SUMMARY.txt; do
  [ -f "$s" ] || continue
  say ""
  say "--- SUMMARY ---"
  cat "$s" | sed 's/^/  /' | tee -a "$LOG"
  break
done

say ""
say "done. results in $DEST"
say "  notebook as it ran: https://www.kaggle.com/code/$REF"
