#!/usr/bin/env bash
# Sweep threadblock tiles against the CUTLASS 3.x collective builders for
# sm_90a and sm_100a.
#
# Purpose is measurement, not tuning: the point is to find the boundary of what
# CUTLASS will accept for each arch, because "which tile" is a per-arch human
# choice and the boundary is the evidence. A tile that fails to compile is
# recorded as SMEM/TILE rejection, which is different from a missing toolkit
# target and different from a defect in this harness.
#
# Run inside the cutlass-lab container:
#   docker run --rm --platform linux/amd64 -v "$PWD":/lab -w /lab \
#     cutlass-lab:cuda129 cutlass_lab/harness/build_3x.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB="$(dirname "$HERE")"
CUTLASS="${CUTLASS_DIR:-$LAB/cutlass}"
OUT="${OUT_DIR:-$LAB/build}"
LOGS="$LAB/logs"
mkdir -p "$OUT" "$LOGS"
TIMEOUT_S="${TIMEOUT_S:-1500}"

arch_for() { [ "$1" = "900" ] && echo "sm_90a" || echo "sm_100a"; }

# Candidate tiles: f16 GEMM tiles that appear in NVIDIA's own published CUTLASS
# configurations, plus the two that were found to be rejected, as controls.
# m n k
TILES=(
  "128 256 64"
  "128 128 64"
  "128 128 32"
  "128  64 64"
  "64  128 64"
  "128 256 32"
  "64   64 64"
)

printf 'arch\tarch_flag\ttile_M\ttile_N\ttile_K\tresult\tseconds\tlog\n' > "$LOGS/tile_sweep.tsv"

for sm in 900 1000; do
  arch="$(arch_for "$sm")"
  echo "=== $arch ==="
  for t in "${TILES[@]}"; do
    read -r tm tn tk <<<"$t"
    tag="sm${sm}_${tm}x${tn}x${tk}"
    log="$LOGS/tile_${tag}.log"
    start=$(date +%s)
    {
      echo "# nvcc -arch=$arch -DLAB_SM=$sm -DLAB_TILE_M=$tm -DLAB_TILE_N=$tn -DLAB_TILE_K=$tk"
      timeout --signal=KILL "$TIMEOUT_S" nvcc -arch="$arch" \
        -DLAB_SM="$sm" -DLAB_TILE_M="$tm" -DLAB_TILE_N="$tn" -DLAB_TILE_K="$tk" \
        -std=c++17 -O3 --expt-relaxed-constexpr --expt-extended-lambda \
        -diag-suppress 20012 \
        -I"$CUTLASS/include" -I"$CUTLASS/tools/util/include" -I"$HERE" \
        -Xcompiler -fno-strict-aliasing \
        -o "$OUT/gemm3x_${tag}" "$HERE/gemm_3x.cu" -lcudart 2>&1
      echo "# exit=$?"
    } > "$log" 2>&1
    end=$(date +%s)
    el=$((end - start))
    ex="$(grep -oE '^# exit=[0-9]+' "$log" | tail -1)"

    if [ "${ex:-}" = "# exit=0" ] && [ -x "$OUT/gemm3x_${tag}" ]; then
      res=BUILD_OK
    elif grep -q "SMEM usage exceeded capacity" "$log"; then
      res=REJECTED_SMEM_CAPACITY
    elif grep -q "static assertion failed" "$log"; then
      res=REJECTED_STATIC_ASSERT
    elif [ "${ex:-}" = "# exit=137" ]; then
      res=TIMEOUT
    else
      res=BUILD_FAILED
    fi
    printf '  %-26s %-22s %ss\n' "<${tm}x${tn}x${tk}>" "$res" "$el"
    printf 'sm%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$sm" "$arch" "$tm" "$tn" "$tk" "$res" "$el" "$log" \
      >> "$LOGS/tile_sweep.tsv"
  done
  echo
done

echo "=== tile sweep ($LOGS/tile_sweep.tsv) ==="
column -t -s $'\t' "$LOGS/tile_sweep.tsv" 2>/dev/null || cat "$LOGS/tile_sweep.tsv"
