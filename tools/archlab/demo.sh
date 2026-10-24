#!/usr/bin/env bash
# The archlab demo, end to end. Everything here runs with no GPU.
#
#   1. prove ptxas assembles and analyses a Hopper-only WGMMA kernel for
#      sm_90a with no GPU present                        <- the key go/no-go
#   2. run the instrument over a contrast pair: the WGMMA path vs a naive
#      scalar shared-memory tiling loop
#   3. close the loop on the other two producers:
#        TinyGrad CUDA C   -> nvcc -arch=sm_90a -cubin
#        Triton kernel     -> triton AOT at GPUTarget("cuda", 90, 32)
#      and show exactly where the toolchain runs out (sm_100a).
#
# Needs the archlab/compile:12.9 image -- see build_image.sh.
#
# Mounts the repo read-only at /w and out/archlab read-write at /demo. The
# tools are staged into container-local /tmp/lab, so a run leaves no script
# copies behind in the repo.
set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
IMG=${IMG:-archlab/compile:12.9}
PLATFORM="--platform linux/amd64"

hr() { printf '\n\033[1m%s\033[0m\n' "$1"; printf '=%.0s' $(seq 1 "${#1}"); echo; }

run() {  # run a bash script (on stdin) inside the container
  docker run --rm -i $PLATFORM \
    -v "$ROOT:/w:ro" -v "$ROOT/out/archlab:/demo" \
    -e PATH=/usr/local/cuda-12.9/bin:/usr/bin:/bin \
    "$IMG" bash -s <<< "$STAGE

$1"
}

STAGE='mkdir -p /tmp/lab
cp /w/tools/archlab/*.py /tmp/lab/ 2>/dev/null
cp /w/tools/archlab/*.json /tmp/lab/ 2>/dev/null
cp /w/out/archlab/triton/fla_kda_chunk_h.py /tmp/lab/ 2>/dev/null'

hr "0. toolchain inside the container, and proof there is no GPU"
run 'ptxas --version | tail -2; echo; nvcc --version | tail -2; echo; cuobjdump --version | tail -1
python3 -c "import triton, tinygrad; print(\"triton\", triton.__version__)"
echo
echo "--- is there a GPU? ---"
python3 - <<PY
try:
    from triton.runtime import driver
    print("triton active driver:", driver.active)
except Exception as e:
    print("triton active driver: ABSENT ->", str(e)[:80])
PY'

hr "1. GPU-free sm_90a: ptxas analyses a Hopper WGMMA kernel"
run 'ptxas -arch=sm_90a -v /demo/probe.ptx -o /tmp/probe.cubin 2>&1
echo
echo "--- SASS mnemonic histogram ---"
cuobjdump -sass /tmp/probe.cubin | grep -oE "/\*[0-9a-f]{4}\*/[[:space:]]+[A-Z][A-Z0-9._:@!]+" \
  | awk "{print \$2}" | sort | uniq -c | sort -rn | head -14
echo
echo "--- the WGMMA and its Hopper companions ---"
cuobjdump -sass /tmp/probe.cubin | grep -iE "HGMMA|WARPGROUP|LDGSTS"'

hr "2a. instrument: WGMMA path (out/archlab/probe.ptx)"
python3 tools/archlab/inspect_ptx.py out/archlab/probe.ptx --arch sm_90a --block-size 128

hr "2b. instrument: naive scalar/shared tiling loop (out/archlab/naive.ptx)"
python3 tools/archlab/inspect_ptx.py out/archlab/naive.ptx --arch sm_90a --block-size 128

hr "3a. TinyGrad generated CUDA C -> nvcc -arch=sm_90a -cubin"
run 'cd /tmp/lab && python3 tinygrad_cuda.py --arch sm_90a --op all --outdir /demo/tg 2>&1 | tail -34'

hr "3b. Triton AOT: FLA KDA kernel at GPUTarget(cuda, 90, 32)"
run 'cd /tmp/lab && python3 triton_aot.py --file /tmp/lab/fla_kda_chunk_h.py \
  --signature /tmp/lab/fla_kda_chunk_h_sig.json --cc 90 --outdir /demo/tri 2>&1 | tail -20'

hr "3c. the same Triton kernel at cc=100 -- and where the toolchain runs out"
run 'cd /tmp/lab && python3 triton_aot.py --file /tmp/lab/fla_kda_chunk_h.py \
  --signature /tmp/lab/fla_kda_chunk_h_sig.json --cc 100 --outdir /demo/tri100 2>&1 \
  | grep -E "driver|target:|RESULT|ptx header"
echo
echo "--- and ptxas 12.9 on that PTX ---"
head -8 /demo/tri100/*.ptx | grep -E "^\.version|^\.target"
ptxas -arch=sm_100a -v /demo/tri100/*.ptx -o /tmp/z.cubin 2>&1 | head -3'

hr "done"
echo "artifacts under out/archlab/:  probe.ptx  naive.ptx  triton/  tg/  tri/  tri100/"
