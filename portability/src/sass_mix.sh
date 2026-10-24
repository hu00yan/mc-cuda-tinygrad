#!/bin/bash
# Compile one CUTLASS probe for several arches and print the SASS instruction mix.
#
# This is the COMPILES-DEGRADED evidence: same source, same -O2, only -arch differs.
# If nvcc exits 0 on every arch and only the instruction histogram moves, the
# fallback is silent.
#
#   usage: sass_mix.sh <probe.cu> <arch> [arch...]
set -u
SRC="$1"; shift
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
docker run --rm --platform linux/amd64 -v "$REPO:/w" -w /w archlab/nvcc:12.9 sh -lc "
export PATH=/usr/local/cuda-12.9/bin:\$PATH
INC='-I/w/cutlass_lab/cutlass/include -I/w/cutlass_lab/cutlass/tools/util/include'
for a in $*; do
  if nvcc -std=c++17 -arch=\$a \$INC -O2 -cubin -o /tmp/s.cubin /w/portability/src/cu/$SRC 2>/tmp/e; then
    printf 'COMPILED  %-8s cubin=%s bytes\n' \$a \$(stat -c %s /tmp/s.cubin)
    nvdisasm -c /tmp/s.cubin 2>/dev/null \
      | grep -oE '\b(LDGSTS|LDG\.E[A-Z0-9.]*|STG\.E[A-Z0-9.]*|STS\.[A-Z0-9.]*|UTMALDG[A-Z0-9.]*|UBLKCP[A-Z0-9.]*|WGMMMA[A-Z0-9.]*|HMMA[A-Z0-9.]*|MMA[A-Z0-9.]*|CALL[A-Z0-9.]*|BRX|BPT|TRAP|DEPBAR[A-Z0-9.]*)\b' \
      | sed 's/\..*//' | sort | uniq -c | sort -rn | head -7 | sed 's/^/            /'
  else
    printf 'REFUSED   %-8s %s\n' \$a \"\$(grep -m1 -iE 'error' /tmp/e | sed 's|^.*error|error|' | cut -c1-110)\"
  fi
done"
