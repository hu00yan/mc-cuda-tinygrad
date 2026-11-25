#!/usr/bin/env bash
# Compile minecu_ref/src/kernels.cu into a shared library for ctypes.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SRC="$ROOT/minecu_ref/src/kernels.cu"
OUT="$HERE/libminecu.so"
SM="${SM:-sm_75}"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found; run this on a CUDA Linux box (e.g. Colab T4)." >&2
  exit 1
fi
if [[ ! -f "$SRC" ]]; then
  echo "missing $SRC" >&2
  exit 1
fi

echo "nvcc -arch=$SM -> $OUT"
nvcc -O3 --use_fast_math -std=c++17 -diag-suppress 177 \
  -arch="$SM" -Xcompiler -fPIC -shared \
  "$SRC" -o "$OUT"
echo "ok $(wc -c < "$OUT") bytes"
nm -D "$OUT" 2>/dev/null | grep -E 'launch_step_uint8_prebasis_full|launch_generate_world' | head || true
