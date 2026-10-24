#!/usr/bin/env bash
# run.sh -- the whole kda_t4 pass, in one card session.
#
# One session, not five, for two reasons that are both about the measurement
# rather than about cost:
#
#   * every number in notes/30 has to be "same card, same session". Splitting
#     the pin, the gate observation and the timing across sessions would make
#     "same card" true and "same process" false, and fla writes
#     TRITON_F32_DEFAULT into os.environ at import -- so a fresh process is a
#     fresh process with a fresh environment. Running them in one process
#     removes the question.
#   * the SASS sweep compiles the same Triton kernel for five architectures.
#     That is CPU time on a box that is being paid for by wall clock, and doing
#     it in the same session as everything else is strictly cheaper.
#
# STAGES, and what each is allowed to do to the others:
#
#   1  INSTALL.sh          idempotent; no-op if already installed
#   2  probe_env.py        the box + the source citations.  Read-only.
#   3  pin_t4.py           runs baseline_pin in SUBPROCESSES, so importing
#                          baseline_pin here cannot perturb stages 4-6
#   4  gate_observed.py    the config collapse, observed
#   5  sass_t4.py          SASS, on the artifact
#   6  time_t4.py          the scoped wall clock
#
# `set -uo pipefail` and NO `set -e`: a stage that fails must not stop the
# others, because the interesting outcome is frequently "this half did not
# work" and the other half still has to be recorded. Each stage's exit code is
# captured and the aggregate is printed at the end.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/out"
LOG="$OUT/run.sh.log"
mkdir -p "$OUT"

: > "$LOG"
log() { printf '%s\n' "$*" | tee -a "$LOG"; }

# Stages may be named as arguments:   bash kda_t4/run.sh install gate sass
# Positional args rather than an env var, because gpush's --cmd collects a list
# and `env K=V cmd` followed by another --cmd makes `env` swallow the next flag
# as its own argument -- which fails with `env: '--cmd': No such file or
# directory` and burns a session to discover.
STAGES="${*:-${KDA_T4_STAGES:-install probe pin gate sass time}}"
declare -a RC
declare -a NAMES

log "kda_t4 run.sh   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "stages: $STAGES"
log ""

for s in $STAGES; do
  log "################################################################"
  log "# stage: $s"
  log "################################################################"
  t0=$(date +%s)
  case "$s" in
    install) bash "$HERE/INSTALL.sh" ;;
    probe)   python3 "$HERE/probe_env.py" ;;
    pin)     python3 "$HERE/pin_t4.py" ;;
    gate)    python3 "$HERE/gate_observed.py" ;;
    sass)    python3 "$HERE/sass_t4.py" ;;
    time)    python3 "$HERE/time_t4.py" ;;
    *)       log "unknown stage '$s'"; RC=("$?"); NAMES=("$s"); continue ;;
  esac
  rc=$?
  t1=$(date +%s)
  RC+=("$rc")
  NAMES+=("$s")
  log ""
  log ">>> stage $s exit=$rc  ($((t1 - t0)) s)"
  log ""
done

log "################################################################"
log "# summary"
log "################################################################"
for i in "${!NAMES[@]}"; do
  printf '  %-10s exit=%s\n' "${NAMES[$i]}" "${RC[$i]}"
done | tee -a "$LOG"
log ""
log "artifacts:"
ls -la "$OUT" 2>&1 | sed 's/^/  /' | tee -a "$LOG" >/dev/null
ls -la "$OUT" | sed 's/^/  /'
log ""
log "wrote $LOG"
# exit 0: a stage failing is a result to be read in the log, not a reason to
# discard the stages that succeeded.  The stage table above is the exit status.
exit 0
