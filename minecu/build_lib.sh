#!/usr/bin/env bash
# Compile Infatoshi/mine.cu kernels.cu (external dep) into libminecu.so.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
OUT="$HERE/libminecu.so"
SM="${SM:-sm_75}"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found; run this on a CUDA Linux box (e.g. Colab T4)." >&2
  exit 1
fi

SRC="$(cd "$ROOT" && PYTHONPATH=. python -m minecu.upstream --print-path)"
if [[ ! -f "$SRC" ]]; then
  echo "missing kernels.cu; run: PYTHONPATH=. python -m minecu.upstream" >&2
  exit 1
fi

echo "nvcc -arch=$SM  $SRC  -> $OUT"
nvcc -O3 --use_fast_math -std=c++17 -diag-suppress 177 \
  -arch="$SM" -Xcompiler -fPIC -shared \
  "$SRC" -o "$OUT"
echo "ok $(wc -c < "$OUT") bytes"
nm -D "$OUT" 2>/dev/null | grep -E 'launch_step_uint8_prebasis_full|launch_generate_world' | head || true
