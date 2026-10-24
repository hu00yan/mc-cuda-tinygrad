#!/bin/bash
# Quick single-probe compile check (dev loop). The authoritative grid is
# src/run_cutlass.py; this exists so a source fix can be tested in seconds.
#   usage: probe_one.sh <probe.cu> <arch> [extra -D flags...]
set -u
SRC="$1"; shift
ARCH="$1"; shift
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
docker run --rm --platform linux/amd64 -v "$REPO:/w" -w /w archlab/nvcc:12.9 sh -lc "
export PATH=/usr/local/cuda-12.9/bin:\$PATH
INC='-I/w/cutlass_lab/cutlass/include -I/w/cutlass_lab/cutlass/tools/util/include'
if nvcc -std=c++17 -arch=$ARCH $* \$INC -O2 -cubin -o /tmp/x.cubin /w/portability/src/cu/$SRC 2>/tmp/e; then
  echo 'OK   $SRC @$ARCH  cubin='\$(stat -c %s /tmp/x.cubin)' bytes'
else
  echo 'FAIL $SRC @$ARCH'
  grep -m4 -iE 'error' /tmp/e | sed 's|/w/portability/src/cu/||; s|/w/cutlass_lab/cutlass/include/||' | cut -c1-200
fi"
