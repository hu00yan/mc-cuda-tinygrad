#!/usr/bin/env bash
# Reproducible, minimal patch to CUTLASS 2.11 so it compiles under clang-20.
#
# We have no nvcc (see Dockerfile.toolchain for why it is unobtainable here), so
# CUTLASS is compiled by clang. Two genuine CUTLASS 2.11 defects block that:
#
#  1. include/cutlass/matrix.h calls `m.set_slice3x3({...})` inside
#     Matrix::rotation() and Matrix::reflection(). The method does not exist; the
#     real name is `set_slice_3x3`. nvcc never instantiates these 3x3 helpers for
#     a GEMM, and is lenient enough about the unqualified call to swallow it.
#     These are host/device helpers for 3x3 rotation/reflection matrices and
#     have nothing to do with any GEMM inner loop, so correcting the name cannot
#     change the SASS we measure. The second argument of rotation() also has an
#     unrelated arithmetic typo, which we leave alone.
#
#  2. (if present) curand header gap -- handled by cuda_stubs/, not by a patch.
#
# Idempotent: safe to re-run.
set -euo pipefail
ROOT="${1:-$(cd "$(dirname "$0")" && pwd)/cutlass2x}"
F="$ROOT/include/cutlass/matrix.h"

[ -f "$F" ] || { echo "cutlass2x/matrix.h not found at $F" >&2; exit 1; }

n=$(grep -c 'set_slice3x3' "$F" || true)
if [ "$n" -eq 0 ]; then
  echo "matrix.h already patched (no set_slice3x3 remaining)"
  exit 0
fi
sed -i.bak 's/\.set_slice3x3(/.set_slice_3x3(/g' "$F"
rm -f "$F.bak"
echo "patched $n occurrence(s) of set_slice3x3 -> set_slice_3x3 in $F"
