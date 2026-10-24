#!/bin/bash
# Assemble every witness at its DECLARED native arch.
#
# A witness that fails here is bad PTX, not a portability finding.  The whole
# matrix is meaningless unless this prints SELFVALID FAIL for nothing.
#
#   usage: selfcheck.sh [ptx-subdir] [out-subdir]   (paths relative to repo root,
#                                                   which is mounted at /w)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PTX="${1:-ptx}"
OUTD="${2:-out/selfval}"
mkdir -p "$REPO/$OUTD"
docker run --rm --platform linux/amd64 -v "$REPO:/w" archlab/nvcc:12.9 sh -lc "
export PATH=/usr/local/cuda-12.9/bin:\$PATH
P=/w/$PTX; O=/w/$OUTD
rm -f \$O/*.cubin
fail=0; n_ok=0; n_known=0
for f in \$P/*.ptx; do
  base=\$(basename \"\$f\" .ptx)
  case \"\$base\" in
    *'!neg') continue;;
    *'!unobtainable')
      echo \"KNOWN-UNOBTAINED \$base\"; n_known=\$((n_known+1)); continue;;
  esac
  arch=\${base##*@}
  err=\$(ptxas -arch=\$arch \"\$f\" -o \$O/\$base.cubin 2>&1)
  if [ \$? -eq 0 ]; then
    n_ok=\$((n_ok+1)); echo \"SELFVALID OK   \$base\"
  else
    fail=\$((fail+1))
    echo \"SELFVALID FAIL \$base :: \$(echo \"\$err\" | grep -iE 'error' | head -1 | sed 's|^.*error|error|')\"
  fi
done
echo '---'
echo \"ok=\$n_ok fail=\$fail known_unobtainable=\$n_known\"
[ \$fail -eq 0 ]"
