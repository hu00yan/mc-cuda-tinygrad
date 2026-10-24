#!/usr/bin/env bash
# kaggle_run.sh -- run the probe battery on Kaggle's free accelerator and bring
#                  the results home. No SSH, no tunnel, no long-lived connection.
#
#   ./remote/kaggle_run.sh                 # GPU variant, kernels/probe
#   ./remote/kaggle_run.sh --variant tpu   # TPU variant, kernels/probe-tpu
#   ./remote/kaggle_run.sh --variant gpu --timeout 90
#   ./remote/kaggle_run.sh --smoke         # the 30-second loop test, kernels/smoke
#   ./remote/kaggle_run.sh --dry-run       # everything except the 3 network calls
#   ./remote/kaggle_run.sh --payload-only  # repack + re-upload the dataset, no run
#
# THE LOOP, in four steps, all of them the official CLI over plain HTTP:
#   1. `kaggle kernels push -p <dir>`   creates the kernel on first use and
#      starts a run. There is no `kaggle kernels create` in the CLI: push is
#      create-or-update (verified against kaggle 2.2.4, whose `kernels` verbs
#      are list/files/get/init/push/pull/output/status/logs/delete/topics), and
#      it is idempotent, so re-running this script just makes version N+1.
#   2. poll `kaggle kernels status owner/slug` until COMPLETE or ERROR
#   3. on ERROR, fetch `kaggle kernels logs` -- that is where the traceback is
#   4. on COMPLETE, `kaggle kernels output owner/slug -p <dir>` then unpack
#
# COST DISCIPLINE. The free tier is ~30 GPU-hours and ~20 TPU-hours a week
# (kaggle quota, checked 2026-09-27: both 0.00h used). Three independent bounds:
#   * SESSION_MIN is passed to `kernels push -t`, which Kaggle enforces
#     server-side: the session dies at that mark whatever the notebook does.
#   * QUEUE_WAIT_MIN bounds this script's poll loop.
#   * the notebook's own BATTERY_TOTAL_BUDGET_S bounds the work inside a run.
# The queue is a separate clock from the session: a GPU kernel pushed at 23:00
# on 2026-09-27 was still QUEUED 24 minutes later, so the poll deadline must be
# much larger than the session cap or a slow queue reads as a failure.
#
# AUTH. `kaggle auth login`, or a token at ~/.kaggle/access_token. This script
# never looks for kaggle.json: that is the legacy path, and the current CLI
# accepts it but prints nothing useful when it is missing.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE="$HERE/.venv/bin/kaggle"
PY="$HERE/.venv/bin/python"
STAMP="$(date +%Y%m%d-%H%M%S)"

VARIANT="gpu"
# Measured on a T4, 2026-09-28: the battery's items sum to ~46 min with
# kda_vs_fla given 1800s, plus ~3 min of setup, clone and pip. So the battery
# budget is 50 min and the session cap 75 min: the cap has to sit above the
# budget, because a session that is killed mid-battery never reaches the cell
# that writes SUMMARY.md. One full run is ~0.85h of the 30h weekly GPU quota.
SESSION_MIN="${SESSION_MIN:-75}"
QUEUE_WAIT_MIN="${QUEUE_WAIT_MIN:-90}"  # how long this script waits for a slot
POLL_SECS="${POLL_SECS:-30}"
ITEM_TIMEOUT="${ITEM_TIMEOUT:-900}"          # per-probe, in the notebook
KDA_VS_FLA_TIMEOUT="${KDA_VS_FLA_TIMEOUT:-1800}"
BATTERY_BUDGET="${BATTERY_BUDGET:-3000}"     # total battery wall budget
ONLY="${ONLY:-}"                             # comma-separated probe keys
SMOKE=0
DRY=0
PAYLOAD_ONLY=0
SKIP_PACK=0

usage() {
  sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

options
  --variant gpu|tpu   accelerator lane. gpu -> kernels/probe, tpu -> kernels/probe-tpu
  --session MIN       kernel session cap, enforced by Kaggle. default 75.
  --queue-wait MIN    how long to wait for a slot before giving up. default 90.
  --timeout MIN       set both of the above
  --item-timeout SEC  per-probe cap in the notebook. default 900.
  --kda-timeout SEC   per-probe cap for kda_vs_fla only. default 1800.
  --battery-budget S  total wall budget for the battery. default 3000.
  --only LIST         run only these probes, comma separated, e.g. --only
                      kda_vs_fla. Everything else is recorded as a skip. Use
                      it to give one slow probe a long timeout without paying
                      for the whole battery twice.
  --smoke             the loop test: kernels/smoke, prints nvidia-smi and exits
  --payload-only      repack + re-upload the dataset, run nothing
  --skip-pack         use kernels/payload/payload.tgz as it stands
  --dry-run           do everything except the three network calls
  -h, --help          this text
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --variant) VARIANT="${2:-}"; shift 2 ;;
    --session) SESSION_MIN="${2:-}"; shift 2 ;;
    --queue-wait) QUEUE_WAIT_MIN="${2:-}"; shift 2 ;;
    --timeout) SESSION_MIN="${2:-}"; QUEUE_WAIT_MIN="${2:-}"; shift 2 ;;
    --item-timeout) ITEM_TIMEOUT="${2:-}"; shift 2 ;;
    --kda-timeout) KDA_VS_FLA_TIMEOUT="${2:-}"; shift 2 ;;
    --battery-budget) BATTERY_BUDGET="${2:-}"; shift 2 ;;
    --only) ONLY="${2:-}"; shift 2 ;;
    --smoke)   SMOKE=1; VARIANT=gpu; shift ;;
    --payload-only) PAYLOAD_ONLY=1; shift ;;
    --skip-pack) SKIP_PACK=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 64 ;;
  esac
done

if [ "$SMOKE" = 1 ]; then
  KDIR="$HERE/kernels/smoke"
  LABEL="smoke"
elif [ "$VARIANT" = "tpu" ]; then
  KDIR="$HERE/kernels/probe-tpu"
  LABEL="tpu"
else
  KDIR="$HERE/kernels/probe"
  LABEL="gpu"
fi

# one parse of the kernel id, then split it: OWNER/SLUG. Two separate seds for
# the two halves is how you end up with owner=slug.
KID="$(sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
        "$KDIR/kernel-metadata.json" | head -1)"
if [ -z "$KID" ]; then
  echo "no \"id\" in $KDIR/kernel-metadata.json" >&2
  exit 64
fi
OWNER="${OWNER:-${KID%%/*}}"
SLUG="${KID#*/}"
REF="$OWNER/$SLUG"
OUT="$HERE/out/kaggle/$STAMP-$LABEL"
LOG="$OUT/run.log"
TMPOUT="$OUT/.partial"                 # staged, then renamed: no half-written dir
SESSION_SECS=$(( SESSION_MIN * 60 ))
WAIT_SECS=$(( QUEUE_WAIT_MIN * 60 ))

mkdir -p "$OUT"
say() { printf '%s\n' "$*" | tee -a "$LOG"; }
say "=== kaggle run $STAMP  variant=$LABEL ==="
say "kernel    $REF"
say "dir       $KDIR"
say "out       $OUT"
say "session   ${SESSION_MIN}m cap (kaggle -t), wait up to ${QUEUE_WAIT_MIN}m for a slot"
say "poll      every ${POLL_SECS}s"
[ "$DRY" = 1 ] && say "DRY RUN: no network calls will be made"

# --------------------------------------------------------------------- auth
say ""
say "=== auth ==="
if [ ! -x "$KAGGLE" ]; then
  say "kaggle CLI not found at $KAGGLE"
  say "  uv pip install --python .venv/bin/python kaggle"
  exit 3
fi
if [ -z "${KAGGLE_API_TOKEN:-}" ] && [ ! -s "$HOME/.kaggle/access_token" ] \
   && [ ! -s "${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json" ]; then
  say "no Kaggle credentials found. Authenticate once, with either:"
  say ""
  say "    $KAGGLE auth login"
  say ""
  say "  which does a browser OAuth flow and writes ~/.kaggle/access_token"
  say "  (mode 600). Nothing to place in the repo, nothing to paste into a"
  say "  script. If you would rather use a token, write it to that same file:"
  say ""
  say "    echo -n 'YOUR_TOKEN' > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token"
  say ""
  say "  Get the token at https://www.kaggle.com/settings/api"
  exit 2
fi
say "credentials present"

cfg="$("$KAGGLE" config view 2>&1)"
say "$cfg" | sed 's/^/  /'
if ! printf '%s' "$cfg" | grep -q 'username: '; then
  say "could not read a username from kaggle config; is the token valid?"
  exit 2
fi
if [ "$DRY" != 1 ]; then
  say ""
  say "=== quota (free tier: ~30 GPU-h and ~20 TPU-h per week) ==="
  "$KAGGLE" quota 2>&1 | sed 's/^/  /' | tee -a "$LOG"
fi

# ------------------------------------------------------------------ payload
# `kaggle kernels push` uploads the notebook and nothing else, so the code the
# notebook runs travels as a Kaggle dataset attached to the kernel. Repack and
# re-upload it before every run so the results always match the current tree.
PAYLOAD_DIR="$HERE/kernels/payload"
PAYLOAD_ID="$(sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
               "$PAYLOAD_DIR/dataset-metadata.json" | head -1)"

if [ "$SKIP_PACK" != 1 ] && [ "$SMOKE" != 1 ]; then
  say ""
  say "=== payload ==="
  if [ "$DRY" = 1 ]; then
    say "would run: $PY kernels/pack.py"
  else
    "$PY" "$HERE/kernels/pack.py" 2>&1 | sed 's/^/  /' | tee -a "$LOG"
    packrc=${PIPESTATUS[0]}
    [ "$packrc" != 0 ] && { say "pack failed (rc=$packrc)"; exit 1; }
    say "  dataset     $PAYLOAD_ID"
    if "$KAGGLE" datasets version -p "$PAYLOAD_DIR" -m "payload $STAMP" \
         >"$OUT/dataset-version.log" 2>&1; then
      say "  dataset version published"
    elif grep -qi "identical" "$OUT/dataset-version.log"; then
      say "  dataset unchanged (the API refuses an identical version; fine)"
    else
      # a missing dataset is created on first use
      if "$KAGGLE" datasets create -p "$PAYLOAD_DIR" >"$OUT/dataset-create.log" 2>&1; then
        say "  dataset created"
      else
        say "  could not publish the payload dataset:"
        tail -5 "$OUT/dataset-version.log" "$OUT/dataset-create.log" 2>/dev/null |
          sed 's/^/    /' | tee -a "$LOG"
        say "  the kernel will run WITHOUT the payload and every probe will skip."
      fi
    fi
    grep -h "http" "$OUT"/dataset-*.log 2>/dev/null | sed 's/^/  /' | tee -a "$LOG"

    # A dataset version that is still being created does not mount: the kernel
    # runs with an EMPTY /kaggle/input and every probe skips, which looks
    # exactly like a broken notebook. Observed on 2026-09-27, when the version
    # was published 4 seconds before the push. So wait for `ready`.
    say "  waiting for the dataset to be ready"
    dswait=$(( $(date +%s) + 300 ))
    while :; do
      dsst="$("$KAGGLE" datasets status "$PAYLOAD_ID" 2>&1 | tr -d '[:space:]')"
      if [ "$dsst" = "ready" ]; then
        say "  dataset     ready"
        break
      fi
      if [ "$(date +%s)" -ge "$dswait" ]; then
        say "  dataset still not ready after 5m (status: ${dsst:-<none>})"
        say "  pushing anyway; the notebook will report an empty /kaggle/input"
        break
      fi
      sleep 10
    done
  fi
fi

if [ "$PAYLOAD_ONLY" = 1 ]; then
  say ""
  say "payload-only: done. push the kernel with: $0 --variant $VARIANT"
  exit 0
fi

# ------------------------------------------------------- build + create (network 1)
say ""
say "=== build + create ==="
# Per-item timeouts go into the NOTEBOOK, not into the push: `kaggle kernels push`
# does not forward this shell's environment to the runner, so exporting them
# here would silently do nothing (verified on the second real GPU run).
export BATTERY_ITEM_TIMEOUT="$ITEM_TIMEOUT"
export BATTERY_TOTAL_BUDGET_S="$BATTERY_BUDGET"
export BATTERY_TIMEOUTS="kda_vs_fla=$KDA_VS_FLA_TIMEOUT"
export BATTERY_ONLY="$ONLY"
if ! "$PY" "$HERE/kernels/build_notebooks.py" 2>&1 | sed 's/^/  /' | tee -a "$LOG"; then
  say "could not regenerate the notebooks; not pushing"
  exit 1
fi
say "  battery     item timeout ${ITEM_TIMEOUT}s, kda_vs_fla ${KDA_VS_FLA_TIMEOUT}s, budget ${BATTERY_BUDGET}s"
[ -n "$ONLY" ] && say "  only        $ONLY  (every other probe is recorded as a skip)"

# ----------------------------------------------------------------- validate
# after the build, because the build is what regenerates the notebook
if ! "$PY" "$HERE/kernels/validate_notebooks.py" "$KDIR" 2>&1 | tee -a "$LOG" | sed 's/^/  /'; then
  say "notebook validation failed; not pushing"
  exit 1
fi
say "  metadata, notebook JSON and every code cell are valid Python"

if [ "$DRY" = 1 ]; then
  say "would run: $KAGGLE kernels push -p $KDIR -t $SESSION_SECS"
  say "            (push is create-or-update; there is no 'kernels create' verb)"
  say "  the kernel is public to nobody: is_private=true in the metadata"
else
  # One call, not two. `kernels create` does not exist; `kernels push` creates
  # the kernel when it is absent and creates a new version when it is present,
  # so it is the idempotent create+update step this loop needs.
  if ! "$KAGGLE" kernels push -p "$KDIR" -t "$SESSION_SECS" 2>&1 |
       tee -a "$LOG" | sed 's/^/  /'; then
    say "push failed; nothing was started"
    exit 1
  fi
  say "  pushed. the kernel is $REF"
fi

# ------------------------------------------------------------------ poll
say ""
say "=== wait for $REF (up to ${QUEUE_WAIT_MIN}m for a slot, ${SESSION_MIN}m session cap) ==="
deadline=$(( $(date +%s) + WAIT_SECS ))
status="UNKNOWN"
last=""
while :; do
  now=$(date +%s)
  if [ "$now" -ge "$deadline" ]; then
    say "  gave up after ${QUEUE_WAIT_MIN}m, last status $status"
    say "  the run may still be waiting for, or running on, Kaggle. stop it at"
    say "    https://www.kaggle.com/code/$REF"
    say "  logs so far are in $LOG"
    exit 4
  fi
  if [ "$DRY" = 1 ]; then
    say "  would poll: $KAGGLE kernels status $REF   (every ${POLL_SECS}s)"
    say "  COMPLETE -> $KAGGLE kernels output $REF -p <dir>, then unpack"
    say "  ERROR    -> $KAGGLE kernels logs $REF"
    status="COMPLETE"
    break
  fi
  out="$("$KAGGLE" kernels status "$REF" 2>&1)"
  rc=$?
  if [ $rc -ne 0 ] && [ -z "$out" ]; then
    say "  status call failed with no output; retrying"
    sleep "$POLL_SECS"; continue
  fi
  # the CLI prints:  owner/slug has status "KernelWorkerStatus.<STATE>"
  st="$(printf '%s' "$out" | sed -n 's/.*status "\(.*\)".*/\1/p' | head -1)"
  st="${st##*.}"                       # KernelWorkerStatus.QUEUED -> QUEUED
  [ -z "$st" ] && st="UNKNOWN"
  if [ "$st" != "$last" ]; then
    say "  [$(date +%H:%M:%S)] $st   (${POLL_SECS}s between polls, $(( deadline - now ))s left)"
    last="$st"
  fi
  status="$st"
  case "$st" in
    COMPLETE) say "  COMPLETE"; break ;;
    ERROR|CANCELLED|FAILED|UNKNOWN)
      if [ "$st" = UNKNOWN ]; then
        # a wrong slug prints "Cannot access kernel ..."; that is fatal, not a state
        if printf '%s' "$out" | grep -q "Cannot access kernel"; then
          say "  $out"
          say "  the slug in $KDIR/kernel-metadata.json does not exist for $OWNER"
          exit 5
        fi
        sleep "$POLL_SECS"; continue
      fi
      say "  $st"
      break ;;
    *) sleep "$POLL_SECS" ;;
  esac
done

# ------------------------------------------------------------------ logs
if [ "$status" != "COMPLETE" ] && [ "$DRY" != 1 ]; then
  say ""
  say "=== kernel log (this is where the traceback is) ==="
  "$KAGGLE" kernels logs "$REF" 2>&1 | tail -120 | tee -a "$LOG" | sed 's/^/  /'
  say ""
  say "run did not complete: $status"
  say "full log: $LOG"
  exit 6
fi

# ------------------------------------------------- output (network 3) + unpack
say ""
say "=== output ==="
rm -rf "$TMPOUT"
mkdir -p "$TMPOUT"
if [ "$DRY" = 1 ]; then
  say "would run: $KAGGLE kernels output $REF -p $TMPOUT --force  (3 attempts)"
  say "would then: unpack any *.zip, promote anything else, and print the index"
  say ""
  say "dry run complete. nothing was created, pushed, or downloaded."
  exit 0
fi

# kaggle 2.2.4 downloads the output file by file over kaggleusercontent.com and
# a single TLS hiccup on one file aborts the whole call (seen once on
# 2026-09-27: SSLEOFError on logs/oc_check.log). So: retry, and on total failure
# keep whatever landed. A partial result set beats an empty directory, and the
# retry costs nothing -- there is no accelerator time in downloading.
got=0
for attempt in 1 2 3; do
  say "  attempt $attempt/3"
  if "$KAGGLE" kernels output "$REF" -p "$TMPOUT" --force 2>&1 |
       tee -a "$LOG" | sed 's/^/    /'; then
    got=1
    break
  fi
  say "  attempt $attempt failed; sleeping 20s"
  sleep 20
done

if [ "$got" = 0 ]; then
  say ""
  say "  all 3 attempts failed. Whatever did download is kept in $TMPOUT"
  say "  rather than deleted; retry by hand with:"
  say "    $KAGGLE kernels output $REF -p $TMPOUT --force"
  exit 7
fi

say ""
say "=== unpack ==="
# Both shapes are handled: some CLI/kernel combinations return one zip per
# output, others return loose files. Unpack the zips, then promote the rest.
nz=0
while IFS= read -r z; do
  [ -n "$z" ] || continue
  d="$OUT/$(basename "$z" .zip)"
  mkdir -p "$d"
  if unzip -qo "$z" -d "$d" 2>&1 | sed 's/^/    /' | tee -a "$LOG"; then
    nz=$(( nz + 1 ))
    say "  $(basename "$z") -> $(basename "$d")/"
  else
    say "  $(basename "$z") FAILED to unpack"
  fi
done < <(find "$TMPOUT" -name '*.zip' -type f | sort)
[ "$nz" = 0 ] && say "  no zips: the CLI returned loose files, promoting them"
cp -R "$TMPOUT/." "$OUT/" 2>/dev/null && say "  promoted $TMPOUT into $OUT"
rm -rf "$TMPOUT"
# the run's stdout, as the notebook runner saw it
if [ -f "$OUT/$SLUG.log" ]; then
  say "  $SLUG.log is the runner's own stdout for this version"
fi

# ------------------------------------------------------------------- index
say ""
say "=== what came back ==="
files=$(find "$OUT" -type f ! -name 'run.log' ! -name '*.log' | wc -l | tr -d ' ')
size=$(du -sh "$OUT" | cut -f1)
say "  $files artefact files (+ the runner log), $size, in $OUT"
find "$OUT" -type f ! -name 'run.log' ! -name '*.log' ! -name 'dataset-*.log' |
  sort | while IFS= read -r f; do
    printf '    %-58s %8s\n' "${f#$OUT/}" "$(du -h "$f" | cut -f1)"
  done | tee -a "$LOG"

say ""
for s in "$OUT"/*/SUMMARY.md "$OUT"/SUMMARY.md; do
  [ -f "$s" ] || continue
  say "--- $(basename "$(dirname "$s")")/SUMMARY.md ---"
  cat "$s" | sed 's/^/  /' | tee -a "$LOG"
  break
done

if [ -f "$OUT/logs/00-environment.txt" ]; then
  say ""
  say "--- logs/00-environment.txt ---"
  head -8 "$OUT/logs/00-environment.txt" | sed 's/^/  /'
fi

say ""
say "done. results in $OUT"
say "  the kernel, if you want to read the notebook as it ran:"
say "    https://www.kaggle.com/code/$REF"
