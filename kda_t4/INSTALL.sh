#!/usr/bin/env bash
# INSTALL.sh -- the pins, executed rather than asserted.
#
# Idempotent: run it twice, the second run is a no-op and says so. Every pip
# invocation is echoed before it runs and its exit status is recorded, so the
# log says what actually happened rather than what was intended.
#
# --no-deps on the fla wheels is deliberate. fla-core 0.5.2 declares
# torch>=2.7.0 and triton>=3.3 under the [cuda] extra, and letting pip resolve
# that would REPLACE the Kaggle image's CUDA build of torch with a
# manylinux wheel -- a 2.5 GB download and a different Triton, silently, in the
# middle of a measurement. The pins below are the fla pins; torch and triton are
# the box's, they get recorded, and probe_env.py fails loudly if they are below
# what fla requires.
#
#   einops  is fla-core's only unconditional runtime dependency
#   transformers  is flash-linear-attention's, and we import fla.ops.kda, not
#           fla.models -- so it is NOT installed. If some fla __init__ pulls it
#           in, probe_env.py will say so and that is a finding.
set -uo pipefail

FLA_VER=0.5.2
TINYGRAD_VER=0.14.0
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/out"
mkdir -p "$OUT"

log() { printf '%s\n' "$*" | tee -a "$OUT/install.log"; }

log "=============================================================="
log "kda_t4 INSTALL  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "python  : $(python3 -c 'import sys;print(sys.version.split()[0])' 2>&1)"
log "pip     : $(python3 -m pip --version 2>&1 | head -1)"
log "=============================================================="

want() {  # want <import-name> <version-spec>  -> 0 if already satisfied
  python3 - "$1" "$2" <<'PY' 2>/dev/null
import sys
try:
    import importlib.metadata as md
    print("   have %s == %s" % (sys.argv[1], md.version(sys.argv[1])))
except Exception:
    sys.exit(1)
PY
}

run() {  # run <label> <pip-args...>
  local label="$1"; shift
  log ""
  log "--> $label"
  log "    \$ python3 -m pip install --disable-pip-version-check $*"
  if python3 -m pip install --disable-pip-version-check --no-input "$@" >>"$OUT/install.log" 2>&1; then
    log "    OK $label"
    return 0
  fi
  log "    FAILED $label  (see $OUT/install.log)"
  return 1
}

if python3 -c "import fla, importlib.metadata as md; md.version('fla-core')" >/dev/null 2>&1; then
  log ""
  log "fla already importable; re-verifying the pin instead of reinstalling"
else
  run "einops" einops
  run "fla-core" --no-deps "fla-core==$FLA_VER"
  run "flash-linear-attention" --no-deps "flash-linear-attention==$FLA_VER"
fi

if python3 -c "import tinygrad, importlib.metadata as md; md.version('tinygrad')" >/dev/null 2>&1; then
  log ""
  log "tinygrad already importable; re-verifying the pin instead of reinstalling"
else
  run "tinygrad" "tinygrad==$TINYGRAD_VER"
fi

log ""
log "=============================================================="
log "installed, as the card sees it"
log "=============================================================="
python3 - <<'PY' 2>&1 | tee -a "$OUT/install.log"
import importlib, importlib.metadata as md
for name, dist in (("torch", "torch"), ("triton", "triton"),
                   ("numpy", "numpy"), ("einops", "einops"),
                   ("fla", "fla-core"), ("flash_linear_attention", "flash-linear-attention"),
                   ("tinygrad", "tinygrad")):
    try:
        v = md.version(dist)
    except Exception as e:
        v = "<%s>" % type(e).__name__
    try:
        mod = importlib.import_module(name)
        where = getattr(mod, "__file__", "?")
    except BaseException as e:
        where = "<import failed: %s: %s>" % (type(e).__name__, str(e)[:80])
    print("  %-22s %-14s %s" % (name, v, where))
print()
try:
    import torch
    print("  torch.cuda.is_available()   ", torch.cuda.is_available())
    print("  device_count                ", torch.cuda.device_count() if torch.cuda.is_available() else 0)
except BaseException as e:
    print("  torch probe failed:", type(e).__name__, str(e)[:100])
PY
log ""
log "done."
exit 0
