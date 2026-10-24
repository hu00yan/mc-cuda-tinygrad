#!/usr/bin/env bash
# sass_build.sh -- a census directory's generated CUDA -> sm_75 SASS, offline.
#
# WHY A CONTAINER. ptxas and cuobjdump are x86-64 Linux binaries. This host is
# arm64 macOS with no CUDA, so the only way to get real sm_75 SASS for
# TinyGrad's generated code is the `archlab/nvcc:12.9` linux/amd64 image that
# `tools/archlab/build_image.sh` builds. Nothing is installed on the host.
#
# WHY THE SAME FLAGS AS THE REFERENCE. nvcc 12.9, -O3 --use_fast_math,
# -arch=sm_75 -- exactly what `minecu_ref/build.sh` uses for the hand-written
# side. A register count and an instruction count are properties of a compiler
# version as much as of a source, so both sides must be built by the same
# toolchain or the columns do not mean the same thing. This is the same reason
# `minecu_tg/sass_tg.sh` uses the same image and flags, and the reason it is
# worth having two scripts that both say so: a flag difference between the two
# sides of a study is invisible in the output and invalidates every row.
#
#   ./sass_build.sh <census-dir> <outdir>     [ARCH=sm_75]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
SRC="${1:?usage: sass_build.sh <census-dir> <outdir>}"
OUT="${2:?usage: sass_build.sh <census-dir> <outdir>}"
ARCH="${ARCH:-sm_75}"
PY="${PY:-$REPO/.venv/bin/python}"

[ -d "$SRC" ] || { echo "not a directory: $SRC" >&2; exit 64; }
command -v docker >/dev/null || { echo "docker not found" >&2; exit 3; }
docker image inspect archlab/nvcc:12.9 >/dev/null 2>&1 || {
  echo "archlab/nvcc:12.9 is not built. Run: bash tools/archlab/build_image.sh" >&2
  exit 3
}

STAGE="$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT").stage"
rm -rf "$STAGE"; mkdir -p "$STAGE/src" "$STAGE/out"
cp -R "$SRC"/. "$STAGE/src"/

# The container script is written to a file rather than piped through a
# heredoc: `bash -s` plus `docker run` is a quoting minefield, and a silently
# empty script here would look exactly like "no kernels compiled".
cat > "$STAGE/_build.sh" <<'EOS'
set -u
export PATH=/usr/local/cuda-12.9/bin:$PATH
ARCH="$1"
ok=0; bad=0
for f in $(find src -name '*.cu' | sort); do
  b="$(printf '%s' "$f" | tr '/.' '__')"
  if nvcc -arch="$ARCH" -O3 --use_fast_math -std=c++17 -cubin "$f" -o "out/$b.cubin" \
       2>"out/$b.err"; then
    cuobjdump -sass "out/$b.cubin" > "out/$b.sass" 2>/dev/null || true
    # ptxas does not take -cubin as an input, so the -v register/spill report
    # has to come from the PTX. `ptxas -v -cubin ...` is accepted silently by
    # some versions and writes "ptxas fatal : Unknown option 'cubin'" to the
    # report file otherwise, which parses as a kernel with no register
    # information -- a missing number that looks like a measurement of zero.
    # Two stages, and the second is the one that carries reg/spill.
    if nvcc -arch="$ARCH" -O3 --use_fast_math -std=c++17 -ptx "$f" -o "out/$b.ptx" \
         2>>"out/$b.err"; then
      ptxas -arch="$ARCH" -v "out/$b.ptx" -o "out/$b.re.cubin" 2> "out/$b.ptxas" || true
    fi
    ok=$((ok+1))
  else
    bad=$((bad+1)); echo "FAIL $f"; head -4 "out/$b.err" | sed 's/^/       /'
  fi
done
echo "COMPILED $ok  FAILED $bad"
EOS

echo "=== generated CUDA -> $ARCH SASS (nvcc 12.9 -O3 --use_fast_math) ==="
echo "  source dir: $SRC"
docker run --rm -v "$STAGE:/w" -w /w archlab/nvcc:12.9 \
  bash /w/_build.sh "$ARCH" 2>&1 | sed 's/^/  /'

rm -rf "$OUT"; mv "$STAGE" "$OUT"
echo "  wrote $OUT"

# The census, through the one categoriser, so the columns mean the same thing
# on the hand-written side. `sass.py` reads the same file for the reference.
if [ -f "$OUT/out/hot_src__cu.sass" ] || ls "$OUT"/out/*.sass >/dev/null 2>&1; then
  echo ""
  echo "  --- per-kernel census ---"
  for s in "$OUT"/out/*.sass; do
    [ -e "$s" ] || continue
    b="$(basename "$s" .sass)"
    printf '  [%s]\n' "$b"
    p="${s%.sass}.ptxas"
    if [ -f "$p" ]; then
      "$PY" "$HERE/sass.py" --generated "$s" --ptxas "$p" 2>&1 |
        sed -n '/this port/,$p' | tail -n +2 | sed 's/^/    /'
    else
      "$PY" "$HERE/sass.py" --generated "$s" 2>&1 |
        sed -n '/this port/,$p' | tail -n +2 | sed 's/^/    /'
    fi
  done
fi
