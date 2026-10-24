#!/usr/bin/env bash
# run.sh -- assemble, push, poll, fetch. One command, bounded, self-verifying.
#
#   ./run.sh --dry-run     everything except the network calls
#   ./run.sh --build-only  assemble and validate, push nothing
#   ./run.sh               assemble, push, poll, download, print
#
# WHY THE ORDER IS FIXED
# ----------------------
# 1. The host-side CPU gate (`axis/verify.py`) runs on THIS machine first, with
#    no GPU and no network. A rung whose two arms compute different things
#    produces a confident, meaningless ratio, and the on-card run costs quota to
#    discover it. This gate has already caught four real defects in one function
#    -- see the comment block at the top of axis/rungs.py.
# 2. The notebook is GENERATED from the sources and every payload is
#    sha256-checked inside it, so a timing can never come from a source that is
#    not the one on disk.
# 3. Push is the last step that can fail without spending accelerator time.
#
# COST
# ----
# One Kaggle GPU run against a weekly budget. The session cap is passed to
# `kernels push -t` so Kaggle enforces it server-side.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AXIS="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$AXIS/.." && pwd)"
KDIR="$HERE/kernel"
PY="${PY:-$REPO/.venv/bin/python}"
KAGGLE="${KAGGLE:-$REPO/.venv/bin/kaggle}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUTROOT="${OUTROOT:-$REPO/out/axis}"

DRY=0; BUILD_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --build-only) BUILD_ONLY=1; shift ;;
    -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 64 ;;
  esac
done

mkdir -p "$OUTROOT"
LOG="$OUTROOT/$STAMP-run.log"
say() { printf '%s\n' "$*" | tee -a "$LOG"; }

SLUG=tenghuzhang/mc-axis-t4
TITLE=$(python3 -c "import json;print(json.load(open('$KDIR/kernel-metadata.json'))['title'])")
ID=$(python3 -c "import json;print(json.load(open('$KDIR/kernel-metadata.json'))['id'])")
NOTEBOOK=$(python3 -c "import json;print(json.load(open('$KDIR/kernel-metadata.json'))['code_file'])")

# ---------------------------------------------------------------- 1. CPU gate
say "=== 1. the host gate: axis/verify.py, CPU, no network, no GPU ==="
say "    (both arms: the TinyGrad expressions AND an emulation of refkernels.cu)"
if (cd "$REPO" && DEV=CPU "$PY" axis/verify.py 2>&1 | tee -a "$LOG"); then
  say "  gate PASS"
else
  say "  gate FAIL -- an arm disagrees with the float64 oracle, or a re-injected"
  say "  historical defect went undetected."
  say "  Refusing to push. A ratio between two different computations is worse"
  say "  than no ratio (notes/06 5b), and a broken reference is the defect class"
  say "  that has cost this study the most GPU time."
  exit 1
fi

# ------------------------------------------------- 2. compile-check the .cu
# A CUDA syntax error found here costs nothing; found on the card it costs a
# queue. nvcc runs in the repo's existing container, for sm_75, no GPU needed.
say ""
say "=== 2. compile-check refkernels.cu for sm_75 (no GPU) ==="
if docker image inspect archlab/nvcc:12.9 >/dev/null 2>&1; then
  if docker run --rm --platform linux/amd64 -v "$AXIS":/w archlab/nvcc:12.9 \
       /usr/local/cuda/bin/nvcc -arch=sm_75 -O3 --use_fast_math -std=c++17 \
       -diag-suppress 177 -cubin -o /tmp/ref.cubin /w/refkernels.cu 2>&1 | tee -a "$LOG"; then
    say "  sm_75 compile OK"
  else
    say "  sm_75 COMPILE FAILED -- not pushing"
    exit 1
  fi
else
  say "  SKIP: archlab/nvcc:12.9 not present locally. The card compiles it anyway"
  say "  (harness.compile_reference), so this is a fast-fail convenience, not a gate."
fi

# ------------------------------------------------------------ 3. the notebook
say ""
say "=== 3. assemble the notebook ==="
(cd "$REPO" && "$PY" axis/kaggle/make_notebook.py --out "$KDIR/$NOTEBOOK" 2>&1 | tee -a "$LOG")

if [ "$BUILD_ONLY" = 1 ]; then say "  --build-only: stopping here."; exit 0; fi
if [ "$DRY" = 1 ]; then say "  --dry-run: stopping before push."; exit 0; fi

# ------------------------------------------------------------------ 4. push
say ""
say "=== 4. push $ID (session cap 25m) ==="
# `kernels push` takes ONLY -p (folder) and -t (session timeout in seconds,
# enforced server-side). There is no -q/-w, and the kernel is identified by the
# kernel-metadata.json in the folder, not by an argument. `kernels output` is the
# one that takes the owner/slug POSITIONALLY. Both of those were wrong in the
# first version of this script and each cost a push; the CLI's own -h is the
# authority and is quoted here so the next reader does not have to rediscover it.
# The push's exit status is checked. `kaggle kernels push` prints
# "Kernel push error: Maximum batch GPU session count of 2 reached" and exits 0
# when both T4 slots are busy with other agents' runs, and the wait loop then
# polls the PREVIOUS run's COMPLETE status and downloads its output -- which
# looks exactly like a successful run of stale code. That is how two rounds of
# this harness "ran" against a build that predated the fixes. A refused push
# must stop here.
set +e
(cd "$KDIR" && "$KAGGLE" kernels push -p "$KDIR" -t 1500 2>&1 | tee -a "$LOG")
rc=${PIPESTATUS[0]}
set -e
if [ "$rc" != 0 ]; then
  say "  push FAILED rc=$rc (no GPU time spent)"
  exit 1
fi
if grep -q "Kernel push error" "$LOG"; then
  say "  push REFUSED by Kaggle -- both T4 sessions are in use by other runs."
  say "  Polling now would return the PREVIOUS run's output and look like a"
  say "  successful run of stale code. Stopping; retry later."
  exit 3
fi

say ""
say "=== 5. wait (queue up to 180m, poll 60s) ==="
deadline=$(( $(date +%s) + 180*60 ))
status=""
while [ "$(date +%s)" -lt "$deadline" ]; do
  status=$("$KAGGLE" kernels status "$ID" 2>/dev/null | head -1 | tr -d '\r')
  say "  [$(date +%H:%M:%S)] $status"
  case "$status" in
    COMPLETE*|ERROR*|FAIL*) break ;;
  esac
  sleep 60
done

say ""
say "=== 6. download ==="
DEST="$OUTROOT/$STAMP"
mkdir -p "$DEST"
(cd "$KDIR" && "$KAGGLE" kernels output "$ID" -p "$DEST" -o -q 2>&1 | tee -a "$LOG") || true

say ""
say "=== 7. what came back ==="
ls -la "$DEST" 2>/dev/null | sed 's/^/  /'
if [ -f "$DEST/axis_report.json" ]; then
  say ""
  say "--- axis_report.json build block ---"
  "$PY" -c "
import json,sys
d=json.load(open('$DEST/axis_report.json'))
print(json.dumps(d.get('build',{}),indent=2)[:900])
print('floor:',json.dumps(d.get('floor',{})))
for r in d.get('rungs',[]):
    tg=r.get('tg',{}).get('median_ms'); rf=r.get('ref',{}).get('median_ms')
    print(f\"  {r['key']:<9} tg={tg} ref={rf} ratio={r.get('ratio_tg_over_ref')} status={r.get('status')}\")
"
else
  say ""
  say "  NO axis_report.json. The harness did not finish; the notebook log has why."
fi
say ""
say "done. results in $DEST/"
