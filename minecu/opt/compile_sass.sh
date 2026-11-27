#!/usr/bin/env bash
# compile_sass.sh -- compile captured generated CUDA to sm_75 SASS, offline.
#
# WHY A CONTAINER. ptxas and cuobjdump are x86-64 Linux binaries and this host
# is arm64 macOS with no CUDA, so the only way to produce real sm_75 SASS for
# the generated code is a linux/amd64 container, which is what
# tools/archlab/build_image.sh builds. Nothing is installed on the host.
#
# WHY IT IS NOT A `mktemp` STAGE. minecu/tinygrad/sass_tg.sh stages through
# `mktemp -d "${TMPDIR:-/tmp}"`. This repo's working rules forbid writing
# scratch outside the working tree, so this script stages inside the repository
# and every byte it produces is evidence that the note can point at.
#
#   ./minecu/opt/compile_sass.sh <captured-dir> <arch>
#
# <captured-dir> is a tree of *.cu -- in practice `max_steps_NNNN/`, whose
# subdirectories are the ablation rungs. Every file is compiled for <arch> and
# the SASS, the `ptxas -v` resource report and the compile errors land in
# <captured-dir>/_sass, named after the source's path within the tree so the
# rung a SASS file came from is in its own file name.
set -euo pipefail

SRC="${1:?usage: compile_sass.sh <captured-dir> <arch>}"
ARCH="${2:?usage: compile_sass.sh <captured-dir> <arch>}"
# `SRC` is normally one `max_steps_NNNN` directory. The counts are a slope
# against max_steps, so the sweep is taken by pointing this at the PARENT --
# the dir that holds every max_steps_NNNN -- and the shape name then survives
# into the output file names. Pointed at a single shape, it still works and the
# file names carry whatever directory name they have.

[ -d "$SRC" ] || { echo "not a directory: $SRC" >&2; exit 64; }
command -v docker >/dev/null || { echo "docker not found" >&2; exit 3; }
docker image inspect archlab/nvcc:12.9 >/dev/null 2>&1 || {
  echo "archlab/nvcc:12.9 is not built. Run: bash tools/archlab/build_image.sh" >&2
  exit 3
}

case "$SRC" in /*) ABS="$SRC" ;; *) ABS="$PWD/$SRC" ;; esac
PARENT="$(cd "$ABS/.." && pwd)"
LEAF="$(basename "$ABS")"
OUT="$ABS/_sass"
rm -rf "$OUT"
mkdir -p "$OUT"

echo "=== generated CUDA -> $ARCH SASS ==="
echo "  source: $ABS"
echo "  output: $OUT"

# The tree is mounted at its parent's depth and the walk is rooted AT the leaf,
# so only this shape's sources are compiled -- but the output name is prefixed
# with the leaf's own directory name, `max_steps_0032`, because the per-step
# counts are a slope against max_steps and a SASS file that has lost which
# shape it came from cannot be placed on that slope at all. (Walking the parent
# instead compiles every shape at once and picks up the capture's staging
# directories, which is a silent way to get a table full of duplicates.)
docker run --rm -v "$PARENT:/w" -w "/w/$LEAF" archlab/nvcc:12.9 bash -c '
set -u
export PATH=/usr/local/cuda-12.9/bin:$PATH
ARCH="$1"; LEAF="$2"
n=0
for f in $(find . -name "*.cu" -not -path "./_sass/*" | sort); do
  n=$((n+1))
  b="${LEAF}__$(printf "%s" "$f" | sed "s|^\./||; s|/|__|g; s|\.cu$||")"
  if nvcc -arch="$ARCH" --use_fast_math -std=c++17 -cubin "$f" -o "_sass/$b.cubin" \
       2> "_sass/$b.err"; then
    cuobjdump -sass "_sass/$b.cubin" > "_sass/$b.sass" 2>/dev/null || true
    # ptxas -v is the authoritative source for registers and spills, and it
    # names the entry function, which is what lets the parser match by name
    # instead of by position -- pairing two lists by index is how a register
    # count ends up attached to the wrong kernel.
    if nvcc -arch="$ARCH" --use_fast_math -std=c++17 -ptx "$f" -o "_sass/$b.ptx" \
         2>> "_sass/$b.err"; then
      ptxas -arch="$ARCH" -v "_sass/$b.ptx" -o "_sass/$b.ptxas.cubin" \
            > "_sass/$b.ptxas" 2>&1 || true
    fi
    printf "  OK   %-52s sass %9d B\n" "$f" "$(wc -c < "_sass/$b.sass" 2>/dev/null || echo 0)"
  else
    printf "  FAIL %s\n" "$f"
    head -4 "_sass/$b.err" | sed "s|^|         |"
  fi
done
echo "  sources: $n"
' _ "$ARCH" "$LEAF" 2>&1 | grep -v "does not match the detected host" | sed 's/^/  /'

echo ""
printf '  sass files:   %s\n' "$(ls -1 "$OUT"/*.sass 2>/dev/null | wc -l | tr -d ' ')"
printf '  ptxas -v:     %s\n' "$(ls -1 "$OUT"/*.ptxas 2>/dev/null | wc -l | tr -d ' ')"
printf '  compile errs: %s\n' "$(cat "$OUT"/*.err 2>/dev/null | grep -c . || true)"
