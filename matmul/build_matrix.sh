#!/usr/bin/env bash
# Build every (kernel, arch) pair and extract the feature-usage matrix.
#
# For each pair we run:
#   clang  : CUDA C++ -> PTX   (.target sm_<arch>)
#   ptxas -v : registers, spill stores, spill loads, stack frame, smem
#   nvdisasm : cubin -> SASS
# and then grep both the PTX and the SASS for the instruction classes the study
# cares about. We report a class as PRESENT only if it appears in SASS (the
# authoritative lowering) or, for the classes ptxas folds away entirely, in the
# PTX.
set -uo pipefail
cd "$(dirname "$0")"
OUT=analysis
mkdir -p "$OUT"

# arch : ptxas target
ARCHES="sm_75 sm_80 sm_89 sm_90a sm_100a"

# Instruction-class detectors. Each entry: name|ptx_regex|sass_regex
# SASS mnemonics are matched with a word-boundary so UMMA does not match UMOV and
# HMMA does not match HMMA2's neighbour, etc.
declare -a CLASS_NAMES=(
  "fp32_fma"
  "tensor_core_mma_sync"
  "int_tensor_core"
  "cp_async"
  "tma"
  "wgmma"
  "tcgen05_umma"
  "ldmatrix"
  "stmatrix"
  "elect"
  "cluster"
  "fp8"
  "block_scaled"
)

det_ptx() { # $1 class
  case "$1" in
    fp32_fma)           echo '(^|[^a-zA-Z0-9_.])fma\.r\.f32' ;;
    tensor_core_mma_sync) echo 'mma\.sync\.aligned' ;;
    int_tensor_core)    echo 'mma\.sync\.aligned\.m[0-9]+n[0-9]+k[0-9]+\.(row|col)\.col\.s[0-9]+' ;;
    cp_async)           echo 'cp\.async\.' ;;
    tma)                echo 'cp\.async\.bulk\.tensor|cp\.async\.bulk\.group' ;;
    wgmma)              echo 'wgmma\.mma_async' ;;
    tcgen05_umma)       echo 'tcgen05\.' ;;
    ldmatrix)           echo 'ldmatrix\.' ;;
    stmatrix)           echo 'stmatrix\.' ;;
    elect)              echo 'elect\.sync' ;;
    cluster)            echo 'clusterlaunch|elect\.sync|barrier\.cluster' ;;
    fp8)                echo 'e4m3|e5m2' ;;
    block_scaled)       echo 'block_scale|scale_vec' ;;
  esac
}

det_sass() {
  case "$1" in
    fp32_fma)           echo '\bFFMA\b' ;;
    tensor_core_mma_sync) echo '\b(HMMA|IMMA|BMMA|QMMA)\b' ;;
    int_tensor_core)    echo '\bIMMA\b' ;;
    cp_async)           echo '\b(LDGSTS|LDGDEPBAR)\b' ;;
    tma)                echo '\b(UTMALDG|UTMAST|UBLKCP)\b' ;;
    wgmma)              echo '\b(HGMMA|QGMMA|DGMMA|UGMMA)\b' ;;
    # Blackwell tensor-core MMA is UTCHMMA.2CTA in SASS (UTensorCore + CHMMA),
    # NOT "UMMA" and not "tcgen05" -- tcgen05 is the PTX spelling only. The
    # tensor-memory load/store are UTMALDG/UTMAST.
    tcgen05_umma)       echo '\b(UTCHMMA|UTCGMA|UTCDMA|UTMALDG|UTMAST|UTCCP)\b' ;;
    ldmatrix)           echo '\bLDSM\b' ;;
    stmatrix)           echo '\bSTSM\b' ;;
    elect)              echo '\bELECT\b' ;;
    cluster)            echo '\b(ELECT|CLUSTER|CCTL)\b' ;;
    fp8)                echo '\b(E4M3|E5M2|F8)\b' ;;
    block_scaled)       echo '\b(TCSCALE|MMA.*SCALE|UBLKCP.*SCALE)\b' ;;
  esac
}

# grep -c prints a count but exits 1 when the count is 0, so `grep -c ... || echo 0`
# yields "0\n0". This returns a single clean integer instead.
cnt() { grep -cE "$1" "$2" 2>/dev/null | head -1 || true; }

row() { # $1=side $2=arch $3=kind $4=label $5=ptx $6=sass $7=ptxas_txt
  local side=$1 arch=$2 kind=$3 label=$4 ptx=$5 sass=$6 ptxas_txt=$7
  local classes="" regs="" spill_st="" spill_ld="" stack="" smem="" status="ok"
  [ -s "$sass" ] || status="NO_SASS"
  if [ ! -s "$ptxas_txt" ] && [ "$status" = "ok" ]; then status="NO_PTXAS"; fi

  # ptxas -v fields
  if [ -s "$ptxas_txt" ]; then
    regs=$(grep -oE "Used [0-9]+ registers" "$ptxas_txt" | grep -oE "[0-9]+" | head -1)
    spill_st=$(grep -oE "[0-9]+ bytes spill stores" "$ptxas_txt" | grep -oE "^[0-9]+" | head -1)
    spill_ld=$(grep -oE "[0-9]+ bytes spill loads" "$ptxas_txt" | grep -oE "^[0-9]+" | head -1)
    stack=$(grep -oE "[0-9]+ bytes stack frame" "$ptxas_txt" | grep -oE "^[0-9]+" | head -1)
    smem=$(grep -oE "smem[^0-9]*([0-9]+) bytes" "$ptxas_txt" | grep -oE "[0-9]+" | head -1)
  fi
  # nvdisasm prints "Used N bytes smem[0], M bytes smem[1]" style in the header
  if [ -z "$smem" ] && [ -s "$sass" ]; then
    smem=$(grep -oE "smem\[[0-9]+\][^,]*|[0-9]+ bytes smem" "$sass" | grep -oE "^[0-9]+ bytes" | grep -oE "^[0-9]+" | head -1)
  fi

  for c in "${CLASS_NAMES[@]}"; do
    local np ns v
    np=$(cnt "$(det_ptx "$c")" "$ptx");  np=${np:-0}
    ns=$(cnt "$(det_sass "$c")" "$sass"); ns=${ns:-0}
    if   [ "$ns" -gt 0 ]; then v="sass:$ns"
    elif [ "$np" -gt 0 ]; then v="ptx-only:$np"
    else v="absent"; fi
    classes="$classes \"$c\": {\"ptx\": $np, \"sass\": $ns, \"v\": \"$v\"},"
  done
  classes=${classes%,}

  # load vector width: widest LDG/STG in SASS
  local vload="none"
  if [ -s "$sass" ]; then
    vload=$(grep -oE '\bLDG\.E(\.[0-9]+)?|\bLDG\.E\.64|\bLDG\.E\.128|\bLDGSTS\.E(\.[0-9]+)?' "$sass" \
      | grep -oE '\.[0-9]+$' | grep -oE '[0-9]+' | sort -n | tail -1)
    vload=${vload:-32}
  fi

  # software-pipeline depth actually emitted: distinct cp.async stages is not
  # directly countable from SASS, so we report the CUTLASS schedule's stage count
  # when it appears in the mangled kernel name, and 0 when there is no cp.async
  # pipeline at all.
  local stages=0
  if [ -s "$sass" ]; then
    stages=$(grep -oE 'UmmaWarpSpecializedILi[0-9]+|WarpSpecializedCooperativeILi[0-9]+|MmaPipelinedILi[0-9]+|Mainloop[A-Za-z0-9]*ILi[0-9]+' "$sass" 2>/dev/null | grep -oE 'ILi[0-9]+' | grep -oE '[0-9]+' | head -1)
    stages=${stages:-0}
  fi
  [ -z "$stages" ] && stages=0

  printf '{"side":"%s","arch":"%s","kind":"%s","label":"%s","status":"%s","registers":%s,"spill_stores":%s,"spill_loads":%s,"stack_frame":%s,"smem_bytes":%s,"max_load_bits":%s,"pipeline_stages":%s,"classes":{%s}}\n' \
    "$side" "$arch" "$kind" "$label" "$status" \
    "${regs:-null}" "${spill_st:-null}" "${spill_ld:-null}" "${stack:-null}" "${smem:-null}" \
    "$vload" "$stages" "$classes" >> "$OUT/matrix.jsonl"
}

echo "{\"generated\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"" > "$OUT/matrix.jsonl"

# ---------------------------------------------------------------- TinyGrad --
# Cross-compile: TinyGrad's target arch (what it *thinks* the GPU is) is
# sm_90 / sm_100; we assemble that same source for the arch-conditional
# hardware targets sm_90a / sm_100a, which is the hardware question.
run_tg() { # $1=ptxas_arch $2=gen file (path relative to repo root of matmul/)
  local pa=$1 f=$2
  [ -f "$f" ] || { echo "skip (missing $f)" >&2; return; }
  ./compile_inspect.sh "$pa" "$f" "$OUT" >/dev/null 2>&1
  local base=$(basename "$f" .cu)
  [ -s "$OUT/$base.sass" ] && row tinygrad "$pa" generated "$f" "$OUT/$base.ptx" "$OUT/$base.sass" "$OUT/$base.ptxas.txt"
}

for f in gen/*.cu; do
  b=$(basename "$f" .cu)
  case "$b" in
    sm_75_*) pa=sm_75  ;; sm_80_*) pa=sm_80 ;; sm_89_*) pa=sm_89 ;;
    sm_90_*) pa=sm_90a ;; sm_100_*) pa=sm_100a ;; *) continue ;;
  esac
  case "$b" in *_beam1__*) continue ;; esac   # byte-identical to default; see RESULTS
  run_tg "$pa" "$f"
done

# ----------------------------------------------------------------- CUTLASS --
for pa in sm_75 sm_80 sm_89; do
  tga=$(( ${pa#sm_} ))
  for sel in 0 1; do
    d="$OUT/cl2x_s$sel"
    CUTLASS_ROOT=$PWD/cutlass2x ./compile_cutlass.sh "$pa" cutlass_baseline_tgarch.cu "$d" -DTG_ARCH=$tga -DSELECT=$sel >/dev/null 2>&1
    lbl=$([ $sel -eq 0 ] && echo "cutlass_${pa}_simt_fp32" || echo "cutlass_${pa}_tensorop_fp16")
    [ -s "$d/cutlass_baseline_tgarch.sass" ] && \
      row cutlass "$pa" 2x "$lbl" "$d/cutlass_baseline_tgarch.ptx" "$d/cutlass_baseline_tgarch.sass" "$d/cutlass_baseline_tgarch.ptxas.txt"
  done
done
CUTLASS_ROOT=$PWD/cutlass ./compile_cutlass.sh sm_90a  cutlass_baseline_sm90a.cu  "$OUT/cl3x" >/dev/null 2>&1
[ -s "$OUT/cl3x/cutlass_baseline_sm90a.sass" ] && \
  row cutlass sm_90a 3x cutlass_sm90a "$OUT/cl3x/cutlass_baseline_sm90a.ptx" "$OUT/cl3x/cutlass_baseline_sm90a.sass" "$OUT/cl3x/cutlass_baseline_sm90a.ptxas.txt"
# Blackwell: CUTLASS 4.8 gates all sm_100a tensor-core support on
# `#if !CUTLASS_CLANG_CUDA` (cutlass/arch/config.h:87), and refuses to emit
# tcgen05 when clang is the frontend. We have no nvcc here, and clang-20 *can*
# target sm_100a (it defines __CUDA_ARCH_FEAT_SM100_ALL), so we define the
# feature macro directly. See patch_cutlass.sh for why that alone is not enough.
CUTLASS_ROOT=$PWD/cutlass ./compile_cutlass.sh sm_100a cutlass_baseline_sm100a.cu "$OUT/cl3x" \
  -DCUTLASS_ARCH_MMA_SM100A_ENABLED=1 >/dev/null 2>&1
[ -s "$OUT/cl3x/cutlass_baseline_sm100a.sass" ] && \
  row cutlass sm_100a 3x cutlass_sm100a "$OUT/cl3x/cutlass_baseline_sm100a.ptx" "$OUT/cl3x/cutlass_baseline_sm100a.sass" "$OUT/cl3x/cutlass_baseline_sm100a.ptxas.txt"

echo "rows: $(wc -l < "$OUT/matrix.jsonl")"
