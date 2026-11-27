#!/usr/bin/env bash
# build.sh -- compile Infatoshi/mine.cu (external) + this directory's bench.cu
#             for one or more GPU architectures, in a container, with no GPU.
#
# WHY A CONTAINER. ptxas/nvcc/cuobjdump/nvdisasm are x86-64 Linux binaries. This
# host is arm64 macOS. The repo's existing lean image already has the whole
# toolchain plus a CUDA runtime, so nothing is installed on the host.
#
#   ./build.sh                 # sm_75 (the T4 / Kaggle target) and sm_86
#   ./build.sh sm_75           # just one
#   ./build.sh sm_75 sm_80 sm_86 sm_90
#   ./build.sh --sweep         # the full list the study cares about
#   SM=sm_75 ./build.sh
#
# OUTPUT, all under build/:
#   kernels_sm<arch>.o        the object
#   minecu_bench_sm<arch>     the linked benchmark binary (runs on the GPU only)
#   kernels_sm<arch>.sass     cuobjdump -sass, for instruction-level inspection
#   resources_sm<arch>.txt    ptxas -v: registers, spills, smem, barriers
#   build_sm<arch>.log        the full nvcc transcript, including every warning
#   MANIFEST.txt              sha256 of every input and every output
#
# The SASS dump is the point of building on a GPU-less host: it is how the
# sm_75 validity question is settled without a GPU. A kernel that emits no
# HMMA/HGMMA/LDGSTS/BAR.RED and no register spills for sm_75 will run on a T4.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${CUDA_IMAGE:-archlab/compile:12.9}"
PLATFORM="${PLATFORM:-linux/amd64}"
OUT="$HERE/build"
ROOT="$(cd "$HERE/.." && pwd)"
KERNELS="$(cd "$ROOT" && PYTHONPATH=. python -m minecu.upstream --print-path)"
# path as seen inside the docker mount of $ROOT at /w
K_DOCKER="/w/${KERNELS#$ROOT/}"
BENCH="$HERE/bench.cu"

# nvcc flags, copied from the upstream setup.py so the binary is the same
# optimisation the project ships:
#   setup.py -> extra_compile_args nvcc: ["-O3", "--use_fast_math"]
# plus -std=c++17, which setup.py leaves to the torch default.
NVCC_FLAGS=(-O3 --use_fast_math -std=c++17 -diag-suppress 177)
HOST_FLAGS=(-O3 -std=c++17)

if [ "${1:-}" = "--sweep" ]; then
  ARCHS=(sm_75 sm_80 sm_86 sm_89 sm_90)
else
  ARCHS=("$@")
  [ ${#ARCHS[@]} -eq 0 ] && ARCHS=(${SM:-sm_75} sm_86)
fi

mkdir -p "$OUT"
echo "image    $IMAGE  ($PLATFORM)"
echo "nvcc     -O3 --use_fast_math -std=c++17"
echo "arches   ${ARCHS[*]}"
echo

for A in "${ARCHS[@]}"; do
  LOG="$OUT/build_${A}.log"
  echo "=== $A ==="
  docker run --rm --platform "$PLATFORM" \
    -v "$ROOT":/w -w /w/bench_cuda -e PATH=/usr/local/cuda/bin:/usr/bin:/bin \
    "$IMAGE" bash -c "
      set -e
      set -o pipefail
      echo '--- nvcc version ---'
      nvcc --version | tail -2
      echo
      echo '--- compile kernels.cu (with -Xptxas -v) ---'
      nvcc ${NVCC_FLAGS[*]} -arch=$A -Xptxas -v -c "$K_DOCKER" -o build/kernels_${A}.o
      echo
      echo '--- compile + link bench.cu ---'
      nvcc ${NVCC_FLAGS[*]} ${HOST_FLAGS[*]} -arch=$A \
           "$K_DOCKER" bench.cu -o build/minecu_bench_${A}
      echo
      echo '--- cuobjdump -sass ---'
      cuobjdump -sass build/kernels_${A}.o > build/kernels_${A}.sass
      echo \"sass lines: \$(wc -l < build/kernels_${A}.sass)\"
    " 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}
  if [ "$rc" != 0 ]; then
    echo "!!! $A FAILED, rc=$rc. See $LOG"
    exit "$rc"
  fi

  # ptxas -v output, separated out for reading
  grep -E "Compiling entry|Used [0-9]+ registers|stack frame|bytes smem" "$LOG" \
    | sed 's/ptxas info    : //' > "$OUT/resources_${A}.txt" || true

  # instruction-mix census, the sm_75 admissibility test
  python3 - "$OUT/kernels_${A}.sass" "$OUT/sass_census_${A}.txt" "$A" <<'PY'
import collections, re, sys
sass_path, out_path, arch = sys.argv[1], sys.argv[2], sys.argv[3]
txt = open(sass_path).read()
parts = re.split(r"Function : (\S+)", txt)
funcs = {parts[i]: parts[i + 1] for i in range(1, len(parts), 2)}
op_re = re.compile(r"/\*[0-9a-f]{4}\*/\s+(?:@!?P\d\s+)?([A-Z][A-Z0-9._]*)")
total = collections.Counter()
per = {}
for name, body in funcs.items():
    ops = op_re.findall(body)
    per[name] = len(ops)
    total.update(ops)

# Anything below sm_75 is illegal on a T4. Anything above sm_80 is a lie about
# portability. This list is the check.
GATES = {
    "HMMA":   "tensor core, fp16, sm_80+",
    "HGMMA":  "tensor core, fp_16 on sm_90a only",
    "IMMA":   "tensor core, integer, sm_75+ (legal on T4)",
    "MMA":    "generic tensor core tag",
    "LDGSTS": "cp.async, sm_80+ (Turing has no async copy)",
    "LDGSTS.E.BYPASS": "cp.async bypass, sm_80+",
    "LDSM":   "ldmatrix, sm_75+ (legal on T4)",
    "BAR.RED": "warp reduce barrier, sm_80+ (Volta+ has BAR.SYNC only)",
    "REDUX":  "redux.sync, sm_80+",
    "ELECT":  "warp ballot elect, sm_80+",
    "F2FP":   "fp8 conversion, sm_89+",
    "E4M3":   "fp8 type, sm_89+",
    "STG.E.U8.SYS": "uint8 global store (expected: this is the obs write)",
    "LDG.E.S8.CONSTANT.SYS": "int8 __ldg read (expected: this is the voxel read)",
}
lines = []
lines.append(f"arch                {arch}")
lines.append(f"functions           {len(funcs)}")
lines.append(f"sass instructions   {sum(per.values())}")
lines.append("")
lines.append("portability gates (count must be 0 for sm_75-illegal ops)")
for op, why in GATES.items():
    n = sum(v for k, v in total.items() if k == op)
    if op in ("STG.E.U8.SYS", "LDG.E.S8.CONSTANT.SYS"):
        lines.append(f"  {op:26s} {n:5d}   expected: {why}")
    else:
        lines.append(f"  {op:26s} {n:5d}   {'OK' if n == 0 else 'PRESENT'}   {why}")
lines.append("")
lines.append("per-kernel sass instruction counts")
for n, c in sorted(per.items(), key=lambda kv: -kv[1]):
    lines.append(f"  {c:6d}  {n}")
lines.append("")
lines.append("top 30 opcodes")
for o, c in total.most_common(30):
    lines.append(f"  {c:6d}  {o}")
open(out_path, "w").write("\n".join(lines) + "\n")
print(f"census -> {out_path}")
bad = sum(total.get(o, 0) for o in ("HMMA", "HGMMA", "LDGSTS", "REDUX", "ELECT", "BAR.RED",
                                    "F2FP"))
# The verdict goes IN the file, not only to stdout. kaggle/run.sh gates its push
# on grepping this file, and a gate that can only be satisfied by watching the
# build log scroll past is not a gate.
verdict = "no sm_75-illegal instructions" if bad == 0 else f"ILLEGAL: {bad} sm_75-illegal instructions"
lines.insert(4, f"verdict             {verdict}")
open(out_path, "w").write("\n".join(lines) + "\n")
if arch == "sm_75" and bad:
    print(f"!!! {verdict}")
    sys.exit(1)
print(f"{arch}: {verdict}")
PY
done

# ------------------------------------------------------------------ manifest
{
  echo "# bench_cuda build manifest"
  echo "# generated by build.sh; inputs and outputs with sha256"
  echo
  echo "## inputs"
  shasum -a 256 "$KERNELS" "$BENCH" 2>/dev/null | sed 's/^/  /'
  echo
  echo "## upstream (Infatoshi/mine.cu via minecu.upstream)"
  echo "  $KERNELS"
  echo
  echo "## outputs"
  for f in "$OUT"/*.o "$OUT"/*.sass "$OUT"/resources_*.txt "$OUT"/sass_census_*.txt \
           "$OUT"/minecu_bench_*; do
    [ -e "$f" ] && shasum -a 256 "$f" | sed 's/^/  /'
  done
} > "$OUT/MANIFEST.txt"

echo
echo "=== manifest ==="
sed 's/^/  /' "$OUT/MANIFEST.txt"
echo
echo "built: ${ARCHS[*]}"
echo "next: run the binary on a GPU, e.g."
echo "  docker run --rm --gpus all --platform linux/amd64 -v \"$HERE\":/w -w /w \\"
echo "    $IMAGE ./build/minecu_bench_sm_75 --selftest-only"
echo "or push it: bench_cuda/kaggle/run.sh"
