#!/usr/bin/env bash
# run.sh -- one command, bounded, self-verifying: measure BOTH MODES of the
#           faithful port and the hand-written baseline on Kaggle's free T4.
#
#   ./run.sh --dry-run     everything except the network calls
#   ./run.sh --build-only  assemble and validate the notebook, push nothing
#   ./run.sh               build, push, poll, download, print
#
# WHY THE ORDER IS FIXED
# ----------------------
# 1. the CPU correctness gate runs on THIS host first, with no GPU. A timing
#    from an implementation that does not compute the right thing is a
#    measurement of the wrong thing, and the failure mode is a plausible ratio
#    against a correct baseline.
# 2. so does `battery.py`, tiny. The gate verifies the PORT; the card runs the
#    BATTERY. Between them there was a file nothing on this host ever
#    executed, which is how a NameError survived to the first batch row of a
#    paid-for session. A pre-push gate runs what the push runs.
# 3. `kaggle kernels push` uploads exactly one file -- the notebook named by
#    `code_file`. Sidecars in the kernel directory are not uploaded, so
#    `kernels.cu` (105 KB), `refdrv.cu` and this package travel inside the
#    notebook as base64 and are sha256-checked before anything is compiled. A
#    mismatch is a hard stop, not a warning.
# 4. the notebook's own step order is load-bearing and is documented at the
#    top of `make_notebook.py`: hash-check, `pip install tinygrad`, and only
#    THEN import-check. Reversing the last two fails on a missing tinygrad,
#    which is a real error in the wrong place and a run spent for nothing.
#
# COST
# ----
# One Kaggle GPU run. The notebook's budget is enforced inside the notebook
# and the session cap is passed to `kernels push -t` so Kaggle enforces it
# server-side. The free tier is ~30 GPU-hours/week; one run is a small
# fraction of it.
#
# `-t` IS IN SECONDS, NOT MINUTES
# -------------------------------
# `kaggle kernels push -t` is documented as "Limit the run time of a kernel to
# the given number of seconds", and the CLI help says so. Two pushes of this
# arm passed a bare 50 there and were killed by it: the card's own log shows
# `nbclient.exceptions.CellTimeoutError: A cell timed out while it was being
# executed, after 50 seconds`, immediately after the hash-check and pip install
# finished, i.e. while the on-card correctness gate was still running. Kaggle
# reports that as CANCEL, which reads like a quota or supersede problem and is
# neither. The other lanes in this repo got it right
# (`-t $((SESSION_MIN * 60))`, `minecu_tg/kaggle/run.sh:318`) and one of them
# ran 2532 s in a single cell. This arm's own successful-neighbour evidence is
# `out/minecu_tg/20260928-221859/`. Seconds are computed here, not passed in.
#
# NO GPU ON THIS HOST. Nothing here needs one. The measurement happens on
# Kaggle.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # minecu_tg_faithful/kaggle/
LANE="$(cd "$HERE/.." && pwd)"                           # minecu_tg_faithful/
REPO="$(cd "$LANE/.." && pwd)"                           # repo root
KDIR="$HERE/kernel"
PY="${PY:-$REPO/.venv/bin/python}"
KAGGLE="${KAGGLE:-$REPO/.venv/bin/kaggle}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUTROOT="${OUTROOT:-$REPO/out/minecu_tg_faithful}"

DRY=0; BUILD_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --build-only) BUILD_ONLY=1; shift ;;
    # The whole header block, by a line number found rather than by a range
    # that goes stale the moment a comment is added above it.
    -h|--help) sed -n "2,$(( $(grep -n '^set -' "$0" | head -1 | cut -d: -f1) - 1 ))p" "$0" \
                 | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 64 ;;
  esac
done

mkdir -p "$OUTROOT"
LOG="$OUTROOT/$STAMP-run.log"
say() { printf '%s\n' "$*" | tee -a "$LOG"; }

# ------------------------------------------------------------------ slug
# Kaggle derives the notebook's URL slug from the metadata TITLE. A mismatch
# between the title and the id you expect is a 409 at push time, after the
# whole notebook has been assembled and gated, so it is checked before push.
slug_from_title() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' |
    sed 's/[^a-z0-9]\{1,\}/-/g; s/^-//; s/-$//'
}
TITLE="$(cat "$KDIR/kernel-metadata.json" 2>/dev/null |
  "$PY" -c 'import json,sys; print(json.load(sys.stdin)["title"])')" || {
  echo "cannot read $KDIR/kernel-metadata.json" >&2; exit 65; }
SLUG="$(slug_from_title "$TITLE")"
ID="$(cat "$KDIR/kernel-metadata.json" |
  "$PY" -c 'import json,sys; print(json.load(sys.stdin)["id"])')"
case "$ID" in
  */"$SLUG") : ;;
  *) echo "FATAL: title '$TITLE' slugifies to '$SLUG' but the id is '$ID'." >&2
     echo "       Kaggle derives the slug from the title; fix kernel-metadata.json." >&2
     exit 66 ;;
esac
say "== minecu_tg_faithful: faithful port, both modes, on one T4 =="
say "  kernel id   $ID   (slug '$SLUG' matches the title)"
say "  log         $LOG"

# ------------------------------------------------------- local gate first
say ""
say "=== 1. verify on this host (CPU, no GPU). No push until this passes. ==="
if ! "$PY" "$LANE/verify.py" --quick 2>&1 | tee -a "$LOG" | sed 's/^/  /'; then
  say "  FAILED. Refusing to push: a timing from an unverified port is a"
  say "  measurement of the wrong thing."
  exit 1
fi
say "  correctness gate passed."

# The battery is run here, tiny, for the same reason: it is code the push will
# run and the gate above did not touch. `battery.py` shipped with an f-string
# reading a bare `max_steps` where the local is `a.max_steps` -- a NameError at
# the first batch row, on the card, after the session was paid for. Nothing on
# this host executed that line, because nothing on this host ran the battery.
#
# 8x8 pixels at one step, one iteration, no warmup, no reference: about five
# seconds, and it walks every branch that matters (both modes, the launcher
# census, the timing loop, the step-count build, the observation digest, and
# the clean-vs-faithful row). It is a smoke test, not a measurement, and it is
# not quoted anywhere.
say ""
say "=== 1b. battery.py smoke (tiny; both modes, no GPU timing claimed) ==="
SMOKE="$OUTROOT/$STAMP-battery-smoke.json"
if ! "$PY" "$LANE/battery.py" --batches 1,2 --res 8 --max-steps 4 --world 8 \
       --iters 1 --warmup 0 --json "$SMOKE" 2>&1 | tee -a "$LOG" | sed 's/^/  /'; then
  say "  FAILED. Refusing to push: battery.py runs on the card and this host"
  say "  just showed it does not run here."
  exit 1
fi
say "  battery.py completed both modes; report at $SMOKE"

# ------------------------------------------------------- nvcc pre-flight
# The notebook's EXACT nvcc command, run here first, because push 7 died on it:
#   refdrv.cu:36:24: fatal error: kernels.cu: No such file or directory
# `#include` resolves relative to the including file -- minecu_tg/kaggle/ -- and
# kernels.cu lives in minecu_ref/src/. The file is correct; the path this stage
# owns was not. A pre-flight that does not run the command the notebook runs is
# worse than none, because it looks like a gate (minecu_tg/kaggle/run.sh:237-244
# records the same lesson from the opposite direction).
#
# The `-DMINECU_KERNELS` override is the one refdrv.cu documents for this
# (refdrv.cu:33-38). The paths below are the repo's, not /kaggle/working's,
# because that is what exists here; the notebook substitutes its own OUT. The
# shape of the command is what is under test, not the directory.
if command -v docker >/dev/null && docker image inspect archlab/nvcc:12.9 >/dev/null 2>&1; then
  say ""
  say "=== 1c. nvcc pre-flight: the notebook's command, for sm_75 ==="
  # `grep -c` exits 1 when it counts zero, which is the PASSING case here. It
  # is caught inside the container (`|| true`) and the count is left in a file
  # rather than captured into `$(...)` here, because `VAR="$(cmd)"` under
  # `set -e` inherits the command's status and this script runs with errexit.
  # That is not hypothetical: the first version of this block did exactly that
  # and aborted the push on a pre-flight that had just passed.
  if docker run --rm -v "$REPO:/w" -w /w archlab/nvcc:12.9 bash -c \
      'set -e
       export PATH=/usr/local/cuda-12.9/bin:$PATH
       nvcc -O3 --use_fast_math -std=c++17 -arch=sm_75 -diag-suppress 177 \
         -DMINECU_KERNELS='\''"/w/minecu_ref/src/kernels.cu"'\'' \
         /w/minecu_tg/kaggle/refdrv.cu -o /w/.pf_refdrv
       { cuobjdump -sass /w/.pf_refdrv \
           | grep -cE "HMMA|HGMMA|LDGSTS|REDUX|ELECT|BAR\.RED|F2FP" \
           || true; } > /w/.pf_bad
       echo "built $(stat -c %s /w/.pf_refdrv) bytes; sm_80+ instructions: $(cat /w/.pf_bad)"' \
      2>&1 | tail -6 | tee -a "$LOG" | sed 's/^/  /'; then
    :
  else
    say "  FAILED. Refusing to push: the notebook builds the hand-written side"
    say "  with this command, and it does not build here."
    exit 1
  fi
  BAD="$(cat "$REPO/.pf_bad" 2>/dev/null || echo skip)"
  rm -f "$REPO"/.pf_refdrv "$REPO"/.pf_bad
  if [ "$BAD" = "skip" ]; then
    say "  FAILED: the container produced no opcode count."; exit 1
  fi
  if [ "$BAD" != "0" ]; then
    say "  FAILED: $BAD sm_80+ instructions in the reference binary."; exit 1
  fi
  say "  pre-flight passed: refdrv builds for sm_75 and carries no sm_80+ opcodes"
else
  say ""
  say "  1c. nvcc pre-flight SKIPPED: no archlab/nvcc container on this host."
fi

# ------------------------------------------------------------- assemble
say ""
say "=== 2. assemble the notebook ==="
if ! "$PY" "$HERE/make_notebook.py" --outdir "$KDIR" 2>&1 | tee -a "$LOG" | sed 's/^/  /'; then
  say "  FAILED to assemble."; exit 1
fi
NB_SHA="$("$PY" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$KDIR/minecu_faithful_t4.ipynb")"
say "  notebook sha256 $NB_SHA"
say "  the notebook's payloads are hash-checked on the card, so this is a"
say "  provenance handle: if the code changes, this changes."

if [ "$BUILD_ONLY" = 1 ]; then
  say ""
  say "--build-only: not pushing. notebook at $KDIR/minecu_faithful_t4.ipynb"
  exit 0
fi

if [ "$DRY" = 1 ]; then
  say ""
  say "--dry-run: would push and run. Stopping here."
  exit 0
fi

# ------------------------------------------------------------------ push
# Minutes, because that is how a human thinks about a session. The CLI wants
# seconds; the conversion happens once, at the call site, below.
SESSION_MIN="${SESSION_MIN:-75}"
QUEUE_WAIT_MIN="${QUEUE_WAIT_MIN:-90}"
POLL_SECS="${POLL_SECS:-30}"
SESSION_SECS=$((SESSION_MIN * 60))
# The naive arm measured 2532 s end to end on this free T4 for a comparable
# workload; 75 min leaves headroom for the on-card gate, which is the slowest
# single stage. A cap that is too small does not fail loudly, it cancels.
[ "$SESSION_SECS" -ge 3600 ] || { echo "FATAL: SESSION_MIN=$SESSION_MIN is below an hour." >&2; exit 67; }

say ""
say "=== 3. push (this spends no accelerator time) ==="
say "  session cap ${SESSION_MIN} min = ${SESSION_SECS} s passed to -t"
push_out="$("$KAGGLE" kernels push -p "$KDIR" -t "$SESSION_SECS" 2>&1)"
printf '%s\n' "$push_out" | tee -a "$LOG" | sed 's/^/  /'
case "$push_out" in
  *"Maximum batch GPU session count"*)
    # A shared-quota refusal, not a defect in this payload. Other lanes in
    # this repo push at the same time and the free tier allows two concurrent
    # GPU sessions per account. Said plainly so the log does not read as a
    # failure of this notebook, and so nobody retries it in a tight loop.
    say ""
    say "  REFUSED: the account's concurrent-GPU-session limit is reached."
    say "  This is a quota fact, not a payload fact -- nothing was uploaded and"
    say "  no accelerator time was spent. Re-run when a slot frees."
    exit 75 ;;
  *"successfully pushed"*) : ;;
  *) say "  push FAILED. The slug check above should have caught the usual cause."; exit 1 ;;
esac

say ""
say "=== 4. poll (bounded at ${QUEUE_WAIT_MIN} min; the queue is a separate clock) ==="
deadline=$(( $(date +%s) + QUEUE_WAIT_MIN*60 ))
status=""
while [ "$(date +%s)" -lt "$deadline" ]; do
  # `kaggle kernels status` prints
  #   <id> has status "KernelWorkerStatus.COMPLETE"
  # so the state is the last dotted field inside the quotes, NOT a
  # colon-separated field. Parsing it as the latter yields "unknown" forever
  # and a poll loop that can never see COMPLETE.
  raw="$("$KAGGLE" kernels status "$ID" 2>&1 || true)"
  status="$(printf '%s' "$raw" | sed -n 's/.*KernelWorkerStatus\.\([A-Z]*\).*/\1/p' | head -1)"
  say "  $(date +%H:%M:%S)  status=${status:-unparsed}"
  case "$status" in
    COMPLETE) break ;;
    ERROR) say "  ERROR. fetching logs:"; "$KAGGLE" kernels logs "$ID" 2>&1 | tail -60 | tee -a "$LOG" | sed 's/^/  /'; exit 1 ;;
    # CANCEL* is a terminal state, not a stage. Spinning on it burns the whole
    # ${QUEUE_WAIT_MIN} budget to learn nothing. Stop here, and fetch the card's
    # log before deciding why, because the two causes look identical from the
    # outside and only the log tells them apart:
    #   * CellTimeoutError after N seconds  -> the session cap was too small
    #   * no traceback, notebook simply stops  -> superseded, or out of quota
    CANCEL*) say "  $status: the run was stopped server-side. Fetching the card's"
              say "  log first; the cause is in it, not in the status."
              _lg="$OUTROOT/$STAMP-card-$status.log"
              "$KAGGLE" kernels logs "$ID" >"$_lg" 2>&1 || true
              if grep -q "CellTimeoutError" "$_lg" 2>/dev/null; then
                say "  CAUSE: the card's own log says CellTimeoutError. The run"
                say "  hit the ${SESSION_SECS}s session cap and was killed, not"
                say "  superseded. Raise SESSION_MIN."
              else
                say "  CAUSE: no cell timeout in the log. Superseded by a later"
                say "  push to $ID, or ended by the account's session quota."
              fi
              grep -o "CellTimeoutError[^\"]*" "$_lg" 2>/dev/null | head -2 | sed 's/^/  /'
              tail -25 "$_lg" | sed 's/^/  /'
              say "  full card log: $_lg"
              exit 78 ;;
  esac
  sleep "$POLL_SECS"
done
[ "$status" = COMPLETE ] || { say "  still '${status:-unparsed}' at the deadline; the session cap will end it."; exit 1; }

# ---------------------------------------------------------------- output
say ""
say "=== 5. fetch ==="
DEST="$OUTROOT/$STAMP"
mkdir -p "$DEST"
"$KAGGLE" kernels output "$ID" -p "$DEST" 2>&1 | tail -3 | sed 's/^/  /' || true
say "  fetched into $DEST"
[ -f "$DEST/SUMMARY.txt" ] && { say ""; say "=== SUMMARY ==="; sed 's/^/  /' "$DEST/SUMMARY.txt" | tee -a "$LOG"; }
exit 0
