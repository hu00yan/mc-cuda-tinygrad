#!/usr/bin/env bash
# Compile a .cu to cubin+SASS for a target arch and report the feature-usage
# matrix. Runs entirely inside the mc-cuda-nvcc image (no GPU, no driver).
#
# Pipeline per (source, arch):
#   clang-20  CUDA C++ -> PTX          (targets .target sm_<arch>)
#   ptxas -arch=<arch> -v              PTX -> cubin, plus register/smem/spill report
#   nvdisasm -c <cubin>                cubin -> SASS text
#
# NOTE: no nvcc. It is not obtainable here (see Dockerfile.toolchain), so clang
# is the CUDA frontend. This is a real substitution and is called out in
# RESULTS.md: nvcc and clang can differ in scheduling and in which intrinsics
# they fold, but the *instruction classes available* at a given -arch are a
# property of ptxas, which is the same NVIDIA assembler in both cases. The
# TinyGrad source is fed to ptxas via clang-produced PTX, so what we measure is
# ptxas's lowering of the source, which is the question being asked.
set -uo pipefail

ARCH="${1:?usage: compile_inspect.sh <arch> <src.cu> <outdir> [extra clang flags...]}"
SRC="${2:?}"
OUT="${3:?}"
shift 3 2>/dev/null || true
EXTRA="$*"
CLANG="${CLANG_CUDA:-clang-20}"

mkdir -p "$OUT"
base=$(basename "$SRC" .cu)
stem="$OUT/${base}"

log="$stem.log"
: > "$log"

run() { echo "+ $*" >> "$log"; "$@" >> "$log" 2>&1; return $?; }

# ---- 1. CUDA C++ -> PTX -------------------------------------------------
if ! run "$CLANG" -x cuda --cuda-device-only -S --cuda-gpu-arch="$ARCH" \
        --cuda-path="${CUDA_PATH:-/opt/cuda}" -I"$(dirname "$0")/cuda_stubs" \
        -O3 -std=c++17 $EXTRA -o "$stem.ptx" "$SRC"; then
  echo "STATUS=CLANG_FAILED"
  grep -E "error:" "$log" | head -5
  exit 3
fi
[ -s "$stem.ptx" ] || { echo "STATUS=EMPTY_PTX"; exit 3; }

# clang emits `.target sm_XX` (no 'a') for the non-conditional archs, and
# `.target sm_XXa` when you ask for an arch-conditional one. Normalise the PTX
# target to exactly the -arch we will hand ptxas, otherwise ptxas rejects it.
if ! grep -q "^\.target ${ARCH}\$" "$stem.ptx"; then
  sed -i.bak -E "s/^\.target sm_[0-9]+[af]?/.target ${ARCH}/" "$stem.ptx"
  rm -f "$stem.ptx.bak"
fi

# ---- 2. PTX -> cubin, with the resource report --------------------------
ptxas_v="$stem.ptxas.txt"
ptxas -arch="$ARCH" -v -o "$stem.cubin" "$stem.ptx" > "$ptxas_v" 2>&1
if [ ! -s "$stem.cubin" ]; then
  echo "STATUS=PTXAS_FAILED"
  cat "$ptxas_v"
  exit 4
fi

# ---- 3. cubin -> SASS ---------------------------------------------------
nvdisasm -c "$stem.cubin" > "$stem.sass" 2>&1
[ -s "$stem.sass" ] || { echo "STATUS=NVDISASM_FAILED"; exit 5; }

echo "STATUS=OK"
echo "PTXAS_BEGIN"
cat "$ptxas_v"
echo "PTXAS_END"
