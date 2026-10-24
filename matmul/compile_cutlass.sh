#!/usr/bin/env bash
# Compile a CUTLASS file to PTX+cubin+SASS for a target arch, inside the
# mc-cuda-nvcc image. Mirrors compile_inspect.sh but with the flags CUTLASS needs
# when clang (not nvcc) is the CUDA frontend.
#
#   ./compile_cutlass.sh <arch> <src.cu> <outdir>
set -uo pipefail

ARCH="${1:?usage: compile_cutlass.sh <arch> <src.cu> <outdir> [extra clang flags...]}"
SRC="${2:?}"
OUT="${3:?}"
shift 3 2>/dev/null || true
EXTRA="$*"          # ALL remaining args, not just $4
CLANG="${CLANG_CUDA:-clang-20}"
HERE="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$OUT"
base=$(basename "$SRC" .cu)
stem="$OUT/${base}"
log="$stem.log"
: > "$log"

run() { echo "+ $*" >> "$log"; "$@" >> "$log" 2>&1; return $?; }

# Two CUTLASS trees are on purpose:
#   cutlass2x/  -- CUTLASS v2.11.0. The 2.x kernels for Sm75/Sm80 (SIMT SGEMM
#                  and tensor-op HMMA). CUTLASS 4.x has no 2.x Sm75 SIMT path and
#                  its Sm80 path does not compile under clang-20.
#   cutlass/    -- CUTLASS 4.8.0 (current main). The 3.x/4.x Hopper (Sm90) and
#                  Blackwell (Sm100) collectives, which have no 2.x equivalent.
# Override with CUTLASS_ROOT=/work/cutlass for the 4.x tree.
ROOT="${CUTLASS_ROOT:-$HERE/cutlass2x}"

# nvcc-only spellings are rejected by clang, so we do not pass them. The
# -D__CUDACC__* flags fake the nvcc version macros CUTLASS branches on.
if ! run "$CLANG" -x cuda --cuda-device-only -S --cuda-gpu-arch="$ARCH" \
        --cuda-path="${CUDA_PATH:-/opt/cuda}" \
        -I"$HERE/cuda_stubs" \
        -I"$ROOT/include" \
        -I"$ROOT/tools/util/include" \
        -I"$HERE/cutlass/examples/common" \
        -include cutlass_clang_compat.h \
        -D__CUDACC__ -D__CUDACC_VER_MAJOR__=12090 -D__CUDACC_VER_MINOR__=86 \
        -O3 -std=c++17 -Wno-unknown-cuda-version $EXTRA \
        -o "$stem.ptx" "$SRC"; then
  echo "STATUS=CLANG_FAILED"
  grep -E "error:" "$log" | head -8
  exit 3
fi
[ -s "$stem.ptx" ] || { echo "STATUS=EMPTY_PTX"; exit 3; }

if ! grep -q "^\.target ${ARCH}\$" "$stem.ptx"; then
  sed -i.bak -E "s/^\.target sm_[0-9]+[af]?/.target ${ARCH}/" "$stem.ptx"
  rm -f "$stem.ptx.bak"
fi

ptxas -arch="$ARCH" -v -o "$stem.cubin" "$stem.ptx" > "$stem.ptxas.txt" 2>&1
if [ ! -s "$stem.cubin" ]; then
  echo "STATUS=PTXAS_FAILED"; cat "$stem.ptxas.txt"; exit 4
fi

nvdisasm -c "$stem.cubin" > "$stem.sass" 2>&1
[ -s "$stem.sass" ] || { echo "STATUS=NVDISASM_FAILED"; exit 5; }

echo "STATUS=OK"
cat "$stem.ptxas.txt"
