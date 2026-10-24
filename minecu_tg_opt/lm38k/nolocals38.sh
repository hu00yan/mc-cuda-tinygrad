#!/usr/bin/env bash
# nolocals38.sh -- capture flat_gather's CUDA with NOLOCALS off and on, and
# compile both with ptxas so its own -v report can be read.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SRC="$REPO/.agent-scratch/lm38b/nolocals_src"
OUT="$REPO/.agent-scratch/lm38b/nolocals38.txt"
mkdir -p "$SRC"
"$REPO/.venv/bin/python" "$HERE/capture_nolocals.py" "$SRC"
: > "$OUT"
docker run --rm -v "$REPO:/w" -w /w archlab/nvcc:12.9 bash -c '
export PATH=/usr/local/cuda-12.9/bin:$PATH
P=$(which ptxas)
for t in stock nolocals; do
  f="/w/.agent-scratch/lm38b/nolocals_src/flat_gather_${t}.cu"
  echo "### flat_gather_${t} ###"
  nvcc -ptx -arch=sm_75 -o /tmp/${t}.ptx "$f" 2>&1 | tail -2
  echo "launch_bounds $(grep -o "__launch_bounds__([0-9]*)" "$f" | head -1)"
  $P -arch=sm_75 -v -O3 -o /tmp/${t}.cubin /tmp/${t}.ptx 2>&1 \
    | grep -E "stack frame|Used .* registers" || echo "(no report)"
done' 2>/dev/null | grep -v "does not match" >> "$OUT"
echo "wrote $OUT"
cat "$OUT"
