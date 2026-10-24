#!/usr/bin/env bash
# Compile the CUTLASS SGEMM harness once per rung of the arch ladder.
#
# Every rung is attempted. A rung that fails is recorded as a failure with the
# exact nvcc diagnostic -- it is never skipped silently, because "this toolkit
# cannot target sm_100a" and "my source is wrong" are different findings and the
# study needs them told apart.
#
# Run inside the cutlass-lab container:
#   docker run --rm --platform linux/amd64 -v "$PWD":/lab -w /lab \
#     cutlass-lab:cuda129 cutlass_lab/harness/build.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB="$(dirname "$HERE")"
CUTLASS="${CUTLASS_DIR:-$LAB/cutlass}"
OUT="${OUT_DIR:-$LAB/build}"
LOGS="$LAB/logs"
mkdir -p "$OUT" "$LOGS"

TIMEOUT_S="${TIMEOUT_S:-1800}"

# nvcc -arch spelling, with the 'a' suffix where CUTLASS needs it. CUTLASS
# hard-fails its own arch-conditional MMA code when you build wgmma / tcgen05
# for the non-'a' target, so the suffix is not cosmetic.
arch_for() {
  case "$1" in
    750) echo "sm_75" ;;
    800) echo "sm_80" ;;
    890) echo "sm_89" ;;
    900) echo "sm_90a" ;;
    1000) echo "sm_100a" ;;
    *) echo "sm_$1" ;;
  esac
}

declare -a RESULTS=()

run_one() {
  local sm="$1" op="$2" opname="$3"
  local arch tag log start end exitline elapsed
  arch="$(arch_for "$sm")"
  tag="sm${sm}_${opname}"
  log="$LOGS/build_${tag}.log"
  start=$(date +%s)
  {
    echo "# nvcc -arch=$arch -DLAB_SM=$sm -DLAB_MODE=$op  ($(date -u +%FT%TZ))"
    timeout --signal=KILL "$TIMEOUT_S" nvcc \
      -arch="$arch" \
      -DLAB_SM="$sm" -DLAB_MODE="$op" \
      -std=c++17 -O3 --expt-relaxed-constexpr --expt-extended-lambda \
      -diag-suppress 20012 \
      -I"$CUTLASS/include" -I"$CUTLASS/tools/util/include" -I"$HERE" \
      -Xcompiler -fno-strict-aliasing \
      -o "$OUT/sgemm_${tag}" "$HERE/sgemm.cu" -lcudart 2>&1
    echo "# exit=$?"
  } > "$log" 2>&1
  end=$(date +%s)
  elapsed=$((end - start))
  exitline="$(grep -oE '^# exit=[0-9]+' "$log" | tail -1)"

  if [ "${exitline:-}" = "# exit=0" ] && [ -x "$OUT/sgemm_${tag}" ]; then
    echo "  BUILD OK        ${elapsed}s   -arch=$arch  -> $OUT/sgemm_${tag}"
    RESULTS+=("sm${sm}"$'\t'"${opname}"$'\t'"BUILD_OK"$'\t'"${elapsed}"$'\t'"$log")
  elif [ "${exitline:-}" = "# exit=137" ]; then
    echo "  TIMEOUT         ${elapsed}s   -arch=$arch   (amd64 emulation, ${TIMEOUT_S}s cap)"
    RESULTS+=("sm${sm}"$'\t'"${opname}"$'\t'"TIMEOUT"$'\t'"${elapsed}"$'\t'"$log")
  elif [ "${exitline:-}" = "# exit=1" ] && grep -q "Unsupported gpu architecture" "$log"; then
    echo "  NO_TARGET       ${elapsed}s   -arch=$arch   (this nvcc cannot target it)"
    RESULTS+=("sm${sm}"$'\t'"${opname}"$'\t'"TOOLKIT_CANNOT_TARGET"$'\t'"${elapsed}"$'\t'"$log")
  else
    echo "  BUILD_FAILED    ${elapsed}s   -arch=$arch   [${exitline:-unknown}]"
    grep -E "error|fatal" "$log" | head -3 | cut -c1-200 | sed 's/^/      /'
    RESULTS+=("sm${sm}"$'\t'"${opname}"$'\t'"BUILD_FAILED"$'\t'"${elapsed}"$'\t'"$log")
  fi
}

echo "nvcc     : $(nvcc --version | tail -2 | head -1)"
echo "targets  : $(nvcc --list-gpu-code 2>/dev/null | tr '\n' ' ')"
echo "cutlass  : $CUTLASS"
echo

for op in 0 1; do
  case "$op" in
    0) opname=simt ;;
    1) opname=tensorop ;;
  esac
  echo "--- mode=$opname ---"
  for sm in 750 800 890 900 1000; do
    run_one "$sm" "$op" "$opname"
  done
  echo
done

printf 'sm\tmode\tresult\tseconds\tlog\n' > "$LOGS/build_summary.tsv"
printf '%s\n' "${RESULTS[@]}" >> "$LOGS/build_summary.tsv"
echo "=== summary ($LOGS/build_summary.tsv) ==="
column -t -s $'\t' "$LOGS/build_summary.tsv" 2>/dev/null || cat "$LOGS/build_summary.tsv"
