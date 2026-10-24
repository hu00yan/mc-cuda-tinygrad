#!/usr/bin/env bash
# sass.sh -- the generated CUDA -> real sm_75 SASS, offline, in the archlab container.
#
#   ./minecu_tg_knobs/sass.sh <census-dir> [arch] [outdir]
#
# WHY A CONTAINER. ptxas and cuobjdump are x86-64 Linux binaries. This host is
# arm64 macOS with no CUDA, so the only way to get real sm_75 SASS out of the
# generated code is a linux/amd64 container -- which is what
# tools/archlab/build_image.sh builds. Nothing is installed on the host.
#
# WHAT IT DOES, AND WHY IT IS NOT notes/19's SASS
# ----------------------------------------------
# It takes the `.cu` files that `codegen.py` captured -- the exact text whose
# sha256 is in the census table, written from the same call that hashed them --
# compiles every one of them, and runs ONE categoriser over the result. The
# categoriser is `tools/archlab/inspect_ptx.py:CATEGORIES`, the same vocabulary
# the reference's SASS was counted in, so the columns mean the same thing on
# both sides of the comparison.
#
# The difference from notes/19's SASS matrix is that THIS runs on arm B's port
# and on every knob configuration, not on arm A's port at one configuration.
# The method is otherwise the same and the flags are the same
# (`--use_fast_math`, `-arch=sm_75`), so a kernel here is comparable to a
# kernel there.
#
# THE BUG THIS INHERITS, AND THE FIX, BOTH STATED
# ------------------------------------------------
# notes/19 §4 records that a `{4}`-hex-digit SASS offset pattern reports EXACTLY
# 4096 instructions for any larger kernel, because the offset field grows to five
# digits after the first 4096 instructions. Three kernels there came out as
# "4096 instructions". The fix is to accept `{4,8}`. This script re-derives the
# count with the corrected pattern and, as a self-check, prints the count the
# BROKEN `{4}` pattern would have reported next to it, so a reader can see
# whether the corrected number is in the region where the bug bites.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

SRC="${1:?usage: sass.sh <census-dir> [arch] [outdir]}"
ARCH="${2:-sm_75}"
OUT="${3:-}"

[ -d "$SRC" ] || { echo "not a directory: $SRC" >&2; exit 64; }
command -v docker >/dev/null || { echo "docker not found" >&2; exit 3; }
docker image inspect archlab/nvcc:12.9 >/dev/null 2>&1 || {
  echo "archlab/nvcc:12.9 is not built. Run: bash tools/archlab/build_image.sh" >&2
  exit 3
}

STAGE="$REPO/.agent-scratch/sass_stage"
rm -rf "$STAGE"
mkdir -p "$STAGE"

# The census dir is `<label>/<port>/*.cu`. Copy the whole thing and let the
# container walk it; the relative path is preserved in the output name so a
# kernel can be traced back to its (port, knob configuration) row.
cp -R "$SRC"/. "$STAGE"/

# The container script goes in a file rather than a `bash -s` heredoc: a heredoc
# piped into `docker run` is a quoting minefield, and a silently-empty script
# here would look exactly like "no kernels compiled".
cat > "$STAGE/_build.sh" <<'EOS'
set -u
export PATH=/usr/local/cuda-12.9/bin:$PATH
ARCH="$1"
# The output dir has to exist BEFORE the first compile, and it has to exist
# inside the mounted volume rather than in the image: the stage is a host
# directory bind-mounted at /w, so a directory created in the image's own /w
# would be discarded. Without this, every `2>out/...` redirect fails, nvcc
# reports the redirect failure as the compile failure, and the run reads
# "TOTAL_SOURCES 143  OK 0  FAIL 143" -- which looks like 143 bad kernels and
# is actually one missing mkdir. That is exactly the "silently-empty looks like
# a real result" failure the file header warns about, so it is created here.
mkdir -p out
n=0; ok=0; fail=0
: > out/sass_index.txt
for f in $(find . -name '*.cu' -not -path './out/*' -not -path './_build.sh' | sort); do
  n=$((n+1))
  b=$(printf '%s' "$f" | sed 's|^\./||' | tr '/' '_' | sed 's|\.cu$||')
  if nvcc -arch="$ARCH" --use_fast_math -std=c++17 -cubin "$f" -o "out/$b.cubin" \
       2>"out/$b.err"; then
    ok=$((ok+1))
    cuobjdump -sass "out/$b.cubin" > "out/$b.sass" 2>/dev/null || true
    # ptxas -v needs the PTX, not the cubin, so the cubin is recompiled from
    # PTX for the register/spill census. Same flags, so the same compiler.
    nvcc -arch="$ARCH" --use_fast_math -std=c++17 -ptx "$f" -o "out/$b.ptx" 2>/dev/null \
      && ptxas -arch="$ARCH" -v "out/$b.ptx" -o /dev/null 2> "out/$b.ptxas" || true
    echo "$b OK" >> out/sass_index.txt
  else
    fail=$((fail+1))
    echo "$b FAIL" >> out/sass_index.txt
    head -4 "out/$b.err" 2>/dev/null | sed "s|^|    |" >> out/sass_index.txt \
      || echo "    (no .err file: the redirect itself failed)" >> out/sass_index.txt
  fi
done
echo "TOTAL_SOURCES $n  OK $ok  FAIL $fail"
# A run that compiles nothing must not be mistaken for a run that found
# nothing. `OK 0` is a failure of this script, not a result about the kernels.
[ "$ok" -gt 0 ] || { echo "FATAL: nothing compiled. This is a harness failure, not a finding."; exit 1; }
EOS

echo "=== arm B generated CUDA -> $ARCH SASS ==="
echo "  census dir: $SRC"
docker run --rm -v "$STAGE:/w" -w /w archlab/nvcc:12.9 \
  bash /w/_build.sh "$ARCH" 2>&1 | sed 's/^/  /'

# --------------------------------------------------------------------------
# One categoriser, run over every .sass in the staged tree.
# --------------------------------------------------------------------------
echo ""
echo "=== instruction census, per (port, knob configuration) ==="
"$REPO/.venv/bin/python" "$HERE/sass_census.py" --stage "$STAGE" --arch "$ARCH"

if [ -n "$OUT" ]; then
  rm -rf "$OUT"
  mkdir -p "$OUT"
  cp -R "$STAGE"/. "$OUT"/
  echo ""
  echo "  wrote $OUT"
fi
