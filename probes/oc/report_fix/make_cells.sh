#!/bin/sh
# make_cells.sh - the corrected `cells` recipe from probes/oc/Makefile.
#
# WHY THIS EXISTS
# ---------------
# probes/oc/Makefile's `cells` target was
#
#     for s in $(SEEDS) 1 2 4 5 6 7; do ./ref_oc --dump-cells cells_$s.bin --seeds $s --quiet; done
#
# which emits 9 fixtures: seeds 0..7 plus 12345.  oc_report.py batches B=32 and
# reads cells_0.bin .. cells_31.bin, so `make report` died with
# FileNotFoundError: 'cells_8.bin' on any machine, from a clean checkout.
# B=32 is the configuration published in notes/01 (39.79 ms/frame GPU-only,
# 1,365 kernels, 101 reduces, "the region buffer is 170 MB" = 32*1,327,976*4 B
# = 169.98 MB), so B=32 is right and the recipe was under-producing.
#
# The required fixture set is therefore {0..B-1} U {correctness seeds}, i.e.
# 0..31 plus 12345 = 33 files, 87.6 MB, all gitignored.
#
# This script is the patch in executable form: it takes the output directory as
# $1 (default: the probes/oc dir it ships next to) and the batch size as $2
# (default: whatever OC_B says, else 32).
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
OC_DIR=$(dirname "$HERE")
OUT=${1:-$OC_DIR}
B=${2:-${OC_B:-32}}
SEEDS=${SEEDS:-"0 3 12345"}
REF=${REF:-$OC_DIR/ref_oc}

if [ ! -x "$REF" ]; then
  echo "make_cells: $REF not built; run 'make ref_oc' first" >&2
  exit 1
fi

mkdir -p "$OUT"

# NENVS_SEEDS is 0..B-1, the contiguous run the batched benchmark needs.
# SEEDS is the correctness set and may sit outside it (12345 does).
nenv_seeds=""
i=0
while [ "$i" -lt "$B" ]; do
  nenv_seeds="$nenv_seeds $i"
  i=$((i + 1))
done

made=0
for s in $SEEDS $nenv_seeds; do
  if [ ! -f "$OUT/cells_$s.bin" ]; then
    "$REF" --dump-cells "$OUT/cells_$s.bin" --seeds "$s" --quiet 2>/dev/null
    made=$((made + 1))
  fi
done

# Prove the set is complete rather than assuming it: anything the report asks
# for and this loop did not make is a bug in the recipe, not a missing file.
missing=0
for s in $SEEDS $nenv_seeds; do
  if [ ! -f "$OUT/cells_$s.bin" ]; then
    echo "make_cells: MISSING cells_$s.bin" >&2
    missing=1
  fi
done
[ "$missing" -eq 0 ] || exit 1

echo "make_cells: $made new, B=$B, $(echo $SEEDS $nenv_seeds | tr ' ' '\n' | sort -nu | wc -l) distinct seeds in $OUT"
