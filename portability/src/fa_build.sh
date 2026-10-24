#!/bin/bash
# Try to build a real FlashAttention translation unit for a given arch.
#
# FA2 (2.8.3) and FA3 (main) are the actual study subjects, so a real compile is
# worth more than a transcription of their arch guards.  Both need CUTLASS; the
# repo's cutlass_lab/cutlass is 4.8.0 and is used read-only here.
#
#   usage: fa_build.sh <repo-dir> <path-under-/w> <arch> [extra nvcc flags...]
set -u
REPO="$1"; SRC="$2"; ARCH="$3"; shift 3
EXTRA="$*"
D=$(cd "$(dirname "$0")/.." && pwd)      # portability/ -- subj/ lives here
CROOT=$(cd "$D/.." && pwd)               # repo root -- cutlass_lab/ lives here
docker run --rm --platform linux/amd64 -v "$CROOT:/repo" -v "$D:/w" -w /w archlab/nvcc:12.9 sh -lc "
export PATH=/usr/local/cuda-12.9/bin:\$PATH
C=/repo/cutlass_lab/cutlass
INC=\"-I\$C/include -I\$C/tools/util/include -I/w/subj/fa/$REPO/csrc -I/w/subj/fa/$REPO/csrc/flash_attn/src -I/w/subj/fa/$REPO/csrc/flash_attn -I/w/subj/fa/$REPO/hopper\"
D=\"-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTE_DEVICE_KERNEL_NAMESPACE=cutlass -DNDEBUG\"
err=\$(nvcc -std=c++17 -arch=$ARCH \$INC \$D $EXTRA -O2 -cubin -o /tmp/fa.cubin \\
      /w/$SRC 2>&1)
if [ \$? -eq 0 ] && [ -s /tmp/fa.cubin ]; then
  echo 'COMPILED  $SRC @$ARCH  cubin='\$(stat -c %s /tmp/fa.cubin)' bytes'
  nvdisasm -c /tmp/fa.cubin 2>/dev/null | grep -oE '\b(LDGSTS|UTMALDG[A-Z0-9.]*|UBLKCP[A-Z0-9.]*|WGMMMA[A-Z0-9.]*|HMMA[A-Z0-9.]*|MMA[A-Z0-9.]*|TCGEN05[A-Z0-9]*|LDG\.E[A-Z0-9.]*|STS(\.[A-Z0-9]+)?|DEPBAR[A-Z0-9.]*|BAR\.SYNC[A-Z0-9.]*|CALL\.ABS[A-Z0-9.]*|BRKPT|LDSM[A-Z0-9.]*)\b' \\
    | sort | uniq -c | sort -rn | head -8 | sed 's/^/            /'
else
  echo 'REFUSED   $SRC @$ARCH'
  echo \"\$err\" | grep -m4 -iE 'error' | sed 's|/w/subj/fa/||; s|/w/cutlass_lab/cutlass/include/||' | cut -c1-190 | sed 's/^/            /'
fi"
