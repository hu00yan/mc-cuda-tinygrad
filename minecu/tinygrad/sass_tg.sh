#!/usr/bin/env bash
# sass_tg.sh -- compile TinyGrad's generated CUDA to sm_75 SASS, offline.
#
# WHY A CONTAINER. ptxas and cuobjdump are x86-64 Linux binaries. This host is
# arm64 macOS with no CUDA, so the only way to produce real sm_75 SASS for the
# generated code is a linux/amd64 container, which is what
# tools/archlab/build_image.sh builds. Nothing is installed on the host.
#
# WHAT IT DOES. Takes a directory of captured kernels (as written by
# minecu/tinygrad/capture_cuda.py or by the T4 run's codegen stage), compiles every
# .cu it finds for the requested architecture, dumps SASS, and runs the SAME
# categoriser over the result that minecu/tinygrad/sass_matrix.py runs over the
# reference's SASS. One categoriser, so the columns mean the same thing in
# every row of the table.
#
#   ./minecu/tinygrad/sass_tg.sh <kernel-dir> <arch> [outdir]
#
# Point it at the parent directory that contains the max_steps_* subdirectories
# to sweep a whole max_steps range; point it at one of those to do one shape.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SRC="${1:?usage: sass_tg.sh <kernel-dir> <arch> [outdir]}"
ARCH="${2:?usage: sass_tg.sh <kernel-dir> <arch> [outdir]}"
OUT="${3:-}"

[ -d "$SRC" ] || { echo "not a directory: $SRC" >&2; exit 64; }
command -v docker >/dev/null || { echo "docker not found" >&2; exit 3; }
docker image inspect archlab/nvcc:12.9 >/dev/null 2>&1 || {
  echo "archlab/nvcc:12.9 is not built. Run: bash tools/archlab/build_image.sh" >&2
  exit 3
}

STAGE="$(mktemp -d "${TMPDIR:-/tmp}/sass_tg.XXXXXX")"
cleanup() { [ -n "$OUT" ] || rm -rf "$STAGE"; }
trap cleanup EXIT

cp -R "$SRC"/. "$STAGE"/
mkdir -p "$STAGE/out"

# The container script is written to a file and run, rather than piped through a
# heredoc: a `bash -s` heredoc plus `docker run` is a quoting minefield, and a
# silently-empty script here would look exactly like "no kernels compiled".
cat > "$STAGE/_build.sh" <<'EOS'
set -u
export PATH=/usr/local/cuda-12.9/bin:$PATH
ARCH="$1"
n=0
for f in $(find . -name '*.cu' -not -path './out/*' | sort); do
  n=$((n+1))
  b=$(printf '%s' "$f" | tr '/.' '__')
  if nvcc -arch="$ARCH" --use_fast_math -std=c++17 -cubin "$f" -o "out/$b.cubin" \
       2>"out/$b.err"; then
    echo "OK   $f"
    cuobjdump -sass "out/$b.cubin" > "out/$b.sass" 2>/dev/null || true
    ptxas -arch="$ARCH" -v -cubin "out/$b.cubin" -o /dev/null 2> "out/$b.ptxas" \
      || nvcc -arch="$ARCH" --use_fast_math -std=c++17 -ptx "$f" -o "out/$b.ptx" \
           2>/dev/null && ptxas -arch="$ARCH" -v "out/$b.ptx" -o "out/$b.cubin" \
           2>"out/$b.ptxas" || true
  else
    echo "FAIL $f"
    head -6 "out/$b.err" | sed 's/^/       /'
  fi
done
echo "TOTAL_SOURCES $n"
EOS

echo "=== TinyGrad generated CUDA -> $ARCH SASS ==="
echo "  source dir: $SRC"
docker run --rm -v "$STAGE:/w" -w /w archlab/nvcc:12.9 \
  bash /w/_build.sh "$ARCH" 2>&1 | sed 's/^/  /'

if [ -n "$OUT" ]; then
  rm -rf "$OUT"
  mkdir -p "$OUT"
  cp -R "$STAGE"/. "$OUT"/
  echo "  wrote $OUT"
fi

echo ""
echo "  --- SASS census, via minecu/tinygrad/sass_matrix.py ---"
PY="${PY:-$(dirname "$(command -v python3)")/python3}"
for s in "$STAGE"/out/*.sass; do
  [ -e "$s" ] || continue
  base="$(basename "$s" .sass)"
  echo ""
  echo "  [$base]"
  "$PY" "$HERE/sass_matrix.py" --sass "$s" 2>&1 | tail -n +3 | sed 's/^/    /'
  if [ -f "$STAGE/out/$base.ptxas" ] && grep -q "Used .* registers" "$STAGE/out/$base.ptxas" 2>/dev/null; then
    "$PY" "$HERE/sass_matrix.py" --sass "$s" --res "$STAGE/out/$base.ptxas" 2>&1 \
      | tail -n +3 | sed 's/^/    /'
  fi
done
