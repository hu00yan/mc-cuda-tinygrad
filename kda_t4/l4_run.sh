#!/usr/bin/env bash
# l4_run.sh -- the command for the L4 reachability/properties session.
#
# ONE probe per session.  Kaggle kills a cell at 1800 s and the runner captures
# stdout only at the end, so this session's entire budget is spent on ONE
# question: which card did the kernel land on, and what can it do.  The KDA
# ladder is a second session and only happens if this one passes.
#
#   1  INSTALL.sh      the study's existing pinned install, unmodified, so the
#                      fla/torch/triton versions are the ones notes/30 used
#   2  l4_probe.py     the properties probe
#
# `set -uo pipefail` and NO `set -e`: a stage that fails must not stop the other,
# because the interesting outcome is frequently "the first half did not work"
# and the second half still has to be recorded.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/out"
mkdir -p "$OUT"
LOG="$OUT/l4_run.log"
: > "$LOG"
log() { printf '%s\n' "$*" | tee -a "$LOG"; }

log "l4_run.sh   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "python: $(python3 -c 'import sys;print(sys.version.split()[0])' 2>&1)"
log ""

t0=$(date +%s)
log "################################################################"
log "# stage: install"
log "################################################################"
bash "$HERE/INSTALL.sh"
rc_install=$?
log ""
log ">>> install exit=$rc_install ($(( $(date +%s) - t0 )) s)"
log ""

t1=$(date +%s)
log "################################################################"
log "# stage: probe"
log "################################################################"
python3 "$HERE/l4_probe.py"
rc_probe=$?
log ""
log ">>> probe exit=$rc_probe ($(( $(date +%s) - t1 )) s)"
log ""

log "################################################################"
log "# summary"
log "################################################################"
log "  install exit=$rc_install"
log "  probe   exit=$rc_probe"
log "  total      $(($(date +%s) - t0)) s"
log ""
log "artifacts:"
ls -la "$OUT" 2>&1 | sed 's/^/  /' | tee -a "$LOG" >/dev/null
ls -la "$OUT"
log ""
log "wrote $LOG"
exit 0
