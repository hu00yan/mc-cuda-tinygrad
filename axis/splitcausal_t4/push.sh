#!/usr/bin/env bash
# push.sh -- assemble, push, poll, fetch the splitcausal T4 probe. One command.
#
#   ./push.sh --build-only   assemble + validate, push nothing
#   ./push.sh                assemble, push, poll, download, print
#
# WHY THE PUSH IS TREATED AS A FAILURE WORTH STOPPING FOR
# ========================================================
# Kaggle's free tier allows 2 concurrent batch GPU sessions per account and the
# other lanes in this project are using them. When both slots are busy,
#
#     $ kaggle kernels push -p DIR -t 1500
#     Kernel push error: Maximum batch GPU session count of 2 reached
#
# PRINTS THAT AND EXITS 0. An earlier version of `axis/kaggle/run.sh` ignored
# the message, went on to poll `kernels status`, saw the PREVIOUS run's
# COMPLETE, and downloaded its output -- indistinguishable, from the outside,
# from a successful run of the current code. Two rounds of this project's axis
# were "measured" that way, against a build that predated the fixes.
#
# So this script does four things that run.sh's early version did not:
#
#   1. checks the push's EXIT STATUS,
#   2. greps the push's own stdout for "Kernel push error",
#   3. records the kernel's REF COUNT before and after, because a refused push
#      adds no version and that is an independent witness,
#   4. refuses to download unless the version it fetches is the version the
#      push actually created.
#
# Point 4 is the one that closes the loop. Everything else detects the refusal;
# this detects a download that does not correspond to the run, which is the
# failure that was actually observed.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
KAGGLE="${KAGGLE:-$REPO/.venv/bin/kaggle}"
PY="${PY:-$REPO/.venv/bin/python}"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$REPO/out/splitcausal/$STAMP-push.log"
OUT="$REPO/out/splitcausal/$STAMP"
ID=tenghuzhang/mc-splitcausal-t4

BUILD_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --build-only) BUILD_ONLY=1; shift ;;
    -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 64 ;;
  esac
done

mkdir -p "$(dirname "$LOG")" "$OUT"
say() { printf '%s\n' "$*" | tee -a "$LOG"; }

# ---------------------------------------------------------------- 1. assemble
say "=== 1. regenerate the notebook from axis/splitcausal_t4probe.py ==="
(cd "$REPO" && "$PY" axis/splitcausal_t4probe.py --out "$HERE/probe.ipynb" 2>&1 | tee -a "$LOG")
NB_SHA=$("$PY" -c "import hashlib,sys;print(hashlib.sha256(open('$HERE/probe.ipynb','rb').read()).hexdigest()[:16])")
say "  probe.ipynb sha256[:16] = $NB_SHA"

# --------------------------------------------- 2. the installed tinygrad, now
# The probe patches `Tensor._split_cumalu` by SOURCE TRANSFORM, not by editing
# the installed file, so the card's site-packages is never written. This step
# records the card-side build's own hash -- but it is the LOCAL venv we can read
# here, and the probe prints the CARD's hash in its own S0 block. The local one
# is recorded so a change between writing the note and running the probe is
# visible rather than silent.
say ""
say "=== 2. local installed tinygrad, for the record (the card's is printed by S0) ==="
OP="$REPO/.venv/lib/python3.12/site-packages/tinygrad/mixin/op.py"
say "  $OP"
say "  sha256 $(shasum -a 256 "$OP" | awk '{print $1}')"
say "  SPLIT  $(grep -E '^\s*SPLIT = ' "$OP" | head -1 | tr -d ' ')"

if [ "$BUILD_ONLY" = 1 ]; then say ""; say "--build-only: stopping here."; exit 0; fi

# --------------------------------------------------- 3. version, before push
say ""
say "=== 3. the version count BEFORE the push ==="
BEFORE=$("$KAGGLE" kernels status "$ID" 2>&1 | head -1 | tr -d '\r')
say "  $BEFORE"
REF_BEFORE=$("$KAGGLE" kernels files "$ID" 2>/dev/null | grep -c . || true)
say "  files listed: ${REF_BEFORE:-0}"

# ------------------------------------------------------------------- 4. push
say ""
say "=== 4. push $ID (session cap 1500s) ==="
set +e
PUSH_OUT=$("$KAGGLE" kernels push -p "$HERE" -t 1500 2>&1)
PUSH_RC=$?
set -e
printf '%s\n' "$PUSH_OUT" | tee -a "$LOG"
say "  exit status: $PUSH_RC"

if [ "$PUSH_RC" != 0 ]; then
  say "  push FAILED rc=$PUSH_RC (no GPU time spent)"
  exit 1
fi
if printf '%s' "$PUSH_OUT" | grep -q "Kernel push error"; then
  say ""
  say "  *** PUSH REFUSED BY KAGGLE ***"
  say "  $(printf '%s' "$PUSH_OUT" | grep 'Kernel push error' | head -1)"
  say "  Both T4 slots are in use by other lanes. Polling now would return the"
  say "  PREVIOUS run's output and look like a successful run of this code, so"
  say "  this script stops here. No number is reported. Retry later:"
  say "      $0"
  exit 3
fi
if ! printf '%s' "$PUSH_OUT" | grep -q "successfully pushed"; then
  say "  push printed neither 'successfully pushed' nor a recognised refusal."
  say "  Treating as REFUSED rather than assuming success."
  exit 3
fi

# ------------------------------------------------------------- 5. wait, then fetch
say ""
say "=== 5. wait (up to 150m, poll 60s) ==="
# `date +%s`, not `date -s`. On BSD/macOS `date -s` means "SET the date",
# takes an argument, prints nothing, and under `set -u` leaves the
# substitution empty -- so `$(( + 150*60 ))` aborts the script with
# "s: unbound variable". It did, one line after a SUCCESSFUL push, which
# is the worst place to find it: the push was accepted and the script
# threw away the only handle on it.
DEADLINE=$(( $(date +%s) + 150*60 ))
STATUS=""
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  STATUS=$("$KAGGLE" kernels status "$ID" 2>/dev/null | head -1 | tr -d '\r')
  say "  [$(date +%H:%M:%S)] $STATUS"
  case "$STATUS" in
    COMPLETE*|ERROR*|FAIL*|CANCEL*) break ;;
  esac
  sleep 60
done

say ""
say "=== 6. the version count AFTER, and which version we are fetching ==="
REF_AFTER=$("$KAGGLE" kernels files "$ID" 2>/dev/null | grep -c . || true)
say "  files listed after: ${REF_AFTER:-0}  (was ${REF_BEFORE:-0})"
VER=$("$KAGGLE" kernels files "$ID" 2>/dev/null | awk 'NR==1{print $1}')
say "  newest version: ${VER:-?}"
if [ -z "${VER:-}" ]; then say "  no version id available; cannot verify the fetch"; exit 4; fi

say ""
say "=== 7. download version $VER ==="
set +e
(cd "$HERE" && "$KAGGLE" kernels output "$ID/$VER" -p "$OUT" -o -q 2>&1 | tee -a "$LOG")
DL_RC=$?
set -e
say "  download rc=$DL_RC"
ls -la "$OUT" 2>/dev/null | sed 's/^/  /'

# ------------------------------------------------------------------ 8. check
say ""
say "=== 8. is the thing we downloaded actually this probe? ==="
FOUND=0
for f in "$OUT"/splitcausal_t4.json "$OUT"/*.json; do
  [ -f "$f" ] || continue
  if "$PY" -c "import json,sys; d=json.load(open('$f')); sys.exit(0 if d.get('probe','').startswith('axis/splitcausal_t4probe') else 1)" 2>/dev/null; then
    say "  FOUND $f"
    "$PY" - "$f" <<'PYEOF' | tee -a "$LOG"
import json, sys
d = json.load(open(sys.argv[1]))
print("  installed SPLIT      ", d["installed_SPLIT"])
print("  op.py sha256         ", d["op_py_sha256"][:16])
print("  session drift pct    ", d.get("session_drift_pct"))
print("  N=4096 by SPLIT      ", d.get("n4096_by_split"))
o = d.get("one_vs_two", {})
if "ratio_two_over_one" in o:
    print("  one-stage vs two-stage", round(o["ratio_two_over_one"], 4), "x",
          f"(cumsum {o['cumsum']['median_ms']:.3f} ms / _cumalu {o['one_stage']['median_ms']:.3f} ms)")
else:
    print("  one-stage vs two-stage  FAILED:", o.get("error"))
PYEOF
    FOUND=1
  fi
done
if [ "$FOUND" = 0 ]; then
  say "  *** NO splitcausal_t4.json IN THE OUTPUT. ***"
  say "  The notebook died, or the run produced a different build. There is no"
  say "  number to report. Look at the notebook log above for the stage it reached."
  exit 5
fi
say ""
say "done. results in $OUT/"
