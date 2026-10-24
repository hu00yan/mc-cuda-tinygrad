#!/usr/bin/env bash
# run_remote.sh -- run the probes on another machine and bring the results back.
#
#   ./remote/run_remote.sh HOST [USER]
#   ./remote/run_remote.sh 1.2.3.4            # user defaults to $REMOTE_USER
#   REMOTE_DIR=/content/drive/x ./remote/run_remote.sh 1.2.3.4
#
# The remote is treated as disposable. Nothing is assumed to survive between
# runs: the venv is rebuilt if missing, output is written to a fresh timestamped
# directory, and logs are pulled back to ./out/remote/<stamp>/ as they are
# produced. If the box dies mid-run, whatever was logged is already home.
#
# There is no device-specific logic anywhere in this script or in the probes.
# TinyGrad picks CUDA on an NVIDIA box and Metal on this Mac; the code is the
# same either way.
set -uo pipefail

HOST="${1:?usage: run_remote.sh HOST [USER]}"
USER_="${2:-${REMOTE_USER:-$USER}}"
REMOTE_DIR="${REMOTE_DIR:-$HOME/probe}"
STAMP="$(date +%Y%m%d-%H%M%S)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$HERE/out/remote/$STAMP"
REPO="$(basename "$HERE")"
LOG="$OUT/remote.log"

mkdir -p "$OUT"
say() { printf '%s\n' "$*" | tee -a "$LOG"; }

say "host=$HOST user=$USER_ dir=$REMOTE_DIR stamp=$STAMP"
say "local repo: $HERE"
say ""

# ---------------------------------------------------------------- 1. preflight
say "=== remote preflight ==="
ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 "$USER_@$HOST" bash -s <<'REMOTE' 2>&1 | tee -a "$LOG"
set -u
echo "--- whoami / pwd ---"
whoami; pwd
echo "--- python ---"
for p in python3 python; do
  if command -v "$p" >/dev/null 2>&1; then echo "found $p -> $($p -V 2>&1) at $(command -v $p)"; PY=$p; break; fi
done
: "${PY:?no python found on the remote}"
echo "--- nvidia ---"
(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>&1 || echo "nvidia-smi unavailable")
echo "--- libcuda ---"
ldconfig -p 2>/dev/null | grep -i libcuda || find / -name 'libcuda.so*' 2>/dev/null | head -3 || true
echo "--- disk ---"
df -h . | tail -1
echo "--- ram ---"
free -g 2>/dev/null | head -2 || cat /proc/meminfo | head -2
REMOTE

# ---------------------------------------------------------------- 2. upload
say ""
say "=== upload ==="
# Only what is needed: no netherite clone, no .venv, no binaries.
rsync -az --delete \
  --exclude '.venv' --exclude '__pycache__' --exclude 'out' \
  --exclude 'netherite/.git' --exclude 'probes/oc/cells_*.bin' \
  --exclude 'probes/oc/ref.bin' --exclude 'probes/oc/ref_oc' \
  "$HERE/" "$USER_@$HOST:$REMOTE_DIR/$REPO/"
say "rsynced to $REMOTE_DIR/$REPO"

# ---------------------------------------------------------------- 3. run
say ""
say "=== remote run ==="
ssh -o StrictHostKeyChecking=accept-new "$USER_@$HOST" bash -s <<REMOTE 2>&1 | tee -a "$LOG"
set -u
DIR="$REMOTE_DIR/$REPO"
cd "\$DIR" || exit 1

# libcuda is the one thing that is often just missing from the loader path.
if ! ldconfig -p 2>/dev/null | grep -q libcuda; then
  for d in /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64 /usr/lib64; do
    if ls "\$d"/libcuda.so* >/dev/null 2>&1; then
      export LD_LIBRARY_PATH="\$d:\${LD_LIBRARY_PATH:-}"
      echo "added \$d to LD_LIBRARY_PATH"
      break
    fi
  done
fi
export CUDA_PATH="\${CUDA_PATH:-/usr/local/cuda}"

echo "--- venv ---"
if [ ! -x .venv/bin/python ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 .venv 2>&1 | tail -2
    uv pip install --python .venv/bin/python -q tinygrad numpy tabulate 2>&1 | tail -3
    # torch/triton/fla are big and only needed for the comparison; do not fail
    # the whole run if they will not install.
    uv pip install --python .venv/bin/python -q torch triton flash-linear-attention 2>&1 | tail -5 || \
      echo "WARNING: torch/triton/fla install failed; the comparison script will skip them"
  else
    python3 -m venv .venv && .venv/bin/pip install -q tinygrad numpy tabulate
    .venv/bin/pip install -q torch triton flash-linear-attention || \
      echo "WARNING: torch/triton/fla install failed"
  fi
fi
PY=.venv/bin/python

echo "--- preflight ---"
\$PY remote/preflight.py; echo "preflight rc=\$?"

echo "--- camera: correctness (needs clang + the netherite headers) ---"
if command -v clang >/dev/null 2>&1; then
  mkdir -p netherite
  if [ ! -d netherite/blaze ]; then
    git clone --filter=blob:none --sparse --depth 1 \
      https://github.com/Infatoshi/netherite netherite 2>&1 | tail -2 &&
    (cd netherite && git sparse-checkout set blaze docs 2>&1 | tail -1) || \
      echo "could not fetch netherite; skipping the C reference"
  fi
  if [ -d netherite/blaze/core ]; then
    (cd probes/oc && make check 2>&1 | tail -20) || echo "camera check failed"
    (cd probes/oc && make report 2>&1 | tail -45) || echo "camera report failed"
  fi
else
  echo "no clang; the C reference cannot be built, camera correctness skipped"
  echo "(the TinyGrad port itself still runs: probes/oc/run_oc.py --bench 8)"
  (cd probes/oc && \$PY run_oc.py --bench 8 2>&1 | tail -20) || true
fi

echo "--- kda: reference + port + cost model ---"
(cd kda && \$PY test_ref.py 2>&1 | tail -6)
(cd kda && \$PY test_tiny.py 2>&1 | tail -10)
(cd kda && \$PY perf_kda.py 2>&1 | tail -30)
(cd kda && \$PY replay_kda.py 2>&1 | tail -26)
(cd kda && \$PY codegen_diff.py 2>&1 | tail -40)

echo "--- kda vs fla / triton (the N-vs-N comparison) ---"
(cd kda && \$PY kda_vs_fla.py 2>&1 | tail -80)

echo "--- float32 precision on this backend ---"
\$PY probes/fp32_correctness.py 2>&1 | tail -22
\$PY probes/gather_floor.py 2>&1 | tail -24

echo "--- collect generated code ---"
mkdir -p out/remote-codegen
\$PY - <<'PYEOF' 2>&1 | tail -30
import os, sys
sys.path.insert(0, "kda")
from tinygrad import Tensor, Device
import numpy as np
d = Device[Device.DEFAULT]
cap = []
orig = getattr(d.compiler, "compile_cached", None)
if orig is not None:
    def spy(src, *a, **k):
        cap.append(str(src)); return orig(src, *a, **k)
    d.compiler.compile_cached = spy
import test_ref as T
import tiny_kda
q,k,v,g,beta = T.rnd(2,128,2,2,32,32,seed=0,l2n=True)
tiny_kda.kda_chunk(q,k,v,g,beta,chunk_size=32)
if orig is not None: d.compiler.compile_cached = orig
os.makedirs("out/remote-codegen", exist_ok=True)
for i, s in enumerate(cap):
    open(f"out/remote-codegen/kernel{i:03d}.txt","w").write(s)
print(f"wrote {len(cap)} generated kernels to out/remote-codegen/")
PYEOF

echo "--- nvidia-smi after ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 || true
echo "=== REMOTE DONE ==="
REMOTE

# ---------------------------------------------------------------- 4. collect
say ""
say "=== collect ==="
mkdir -p "$OUT/kda" "$OUT/probes"
for f in kda/oc_report.txt probes/oc/oc_report.txt; do :; done
rsync -az "$USER_@$HOST:$REMOTE_DIR/$REPO/out/" "$OUT/" 2>/dev/null && say "pulled out/" || say "no out/ to pull"
for f in perf_kda.py replay_kda.py codegen_diff.py kda_vs_fla.py; do
  rsync -az "$USER_@$HOST:$REMOTE_DIR/$REPO/kda/$f" "$OUT/kda/" 2>/dev/null || true
done
rsync -az "$USER_@$HOST:$REMOTE_DIR/$REPO/probes/" "$OUT/probes/" \
  --exclude 'cells_*.bin' --exclude 'ref.bin' --exclude 'ref_oc' 2>/dev/null || true
say "results in $OUT"
say ""
say "log tail:"
tail -40 "$LOG"
