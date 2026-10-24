#!/usr/bin/env bash
# currencies38.sh -- dump ptxas's own -v report for each rung's captured PTX.
# The ptxas binary is x86-64 Linux, so this runs in the project's own image.
#
# `set -e` is deliberately NOT used inside the container loop: `grep` returns 1
# when it matches nothing, and with `set -e` that kills the whole script after
# the first rung and leaves a half-written file that later looks like "this rung
# has no ptxas report".
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-$REPO/.agent-scratch/lm38b/currencies38.txt}"
mkdir -p "$(dirname "$OUT")"
: > "$OUT"
docker run --rm -v "$REPO:/w" -w /w archlab/ptxas:12.9 bash -c '
P=/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas
for r in naive int_cursor flat_gather incremental lean; do
  f="/w/.agent-scratch/rungtimings/sass/sass-b1024/max_steps_0032__${r}__kernel000.ptx"
  echo "### $r ###"
  if [ -f "$f" ]; then
    $P -arch=sm_75 -v -O3 -o /tmp/k.cubin "$f" 2>&1 \
      | grep -E "stack frame|Used .* registers" || echo "(no report)"
  else
    echo "(no PTX at $f)"
  fi
done' 2>/dev/null | grep -v "does not match" >> "$OUT"
echo "wrote $OUT ($(wc -l < "$OUT") lines)"
