#!/usr/bin/env bash
# Reproducible, minimal patch to CUTLASS 4.8 so its Blackwell tcgen05 path can be
# compiled by clang-20.
#
# Background
# ----------
# CUTLASS 4.8 gates all Blackwell tensor-core support on
#
#     #if !CUTLASS_CLANG_CUDA && (CUDA >= 12.8)     // cutlass/arch/config.h:87
#         ... #define CUTLASS_ARCH_MMA_SM100A_ENABLED 1
#
# i.e. it refuses to emit tcgen05 when the CUDA frontend is clang. We have no
# nvcc in this environment (see Dockerfile.toolchain for why it is unobtainable
# here), and clang-20 *can* target sm_100a and does define
# __CUDA_ARCH_FEAT_SM100_ALL, so the exclusion is over-conservative for our case.
# We therefore pass -DCUTLASS_ARCH_MMA_SM100A_ENABLED=1 on the command line;
# cute/arch/config.hpp only tests `defined(...)`, so that is sufficient to switch
# the tcgen05 paths back on.
#
# Doing so exposes a second, genuine CUTLASS 4.8 defect in
# include/cute/arch/copy_sm100.hpp (two sites, ~line 380 and ~line 399):
#
#     asm volatile ("tcgen05.cp.cta_group::1.128x256b [%0], %1;"
#     :
#     : "r"(dst_addr)  "l"(src_addr));
#
# The asm statement is missing its output-operand list: the `:` after the
# template string is followed immediately by another `:`, so the operand lists do
# not parse ("expected ')'"). It should be `: : "r"(dst), "l"(src)`. This code is
# unreachable in any build we can otherwise make (it is inside
# CUTE_ARCH_TCGEN05_TMEM_ENABLED), which is why it has survived; clang parses
# template bodies strictly and rejects it.
#
# The fix is whitespace plus one comma in an inline-asm *constraint string* on
# a zero-output assembly statement. It cannot change the generated SASS for any
# other kernel: it touches only the operand-list syntax of two `tcgen05.cp`
# helpers, and after the fix they emit exactly the instruction they already
# named.
#
# Idempotent: safe to re-run.
set -euo pipefail
ROOT="${1:-$(cd "$(dirname "$0")" && pwd)/cutlass}"
F="$ROOT/include/cute/arch/copy_sm100.hpp"

[ -f "$F" ] || { echo "not found: $F" >&2; exit 1; }

# Match the malformed form: a lone ':' line followed by a line starting ': "r"'
# and containing two constraints separated by whitespace rather than a comma.
before=$(grep -cE '^    :$' "$F" || true)
if [ "$before" -eq 0 ]; then
  echo "copy_sm100.hpp already patched"
  exit 0
fi

python3 - "$F" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
# `asm volatile ("..."\n    :\n    : "r"(a)  "l"(b));`  ->  `... : : "r"(a), "l"(b));`
new, n = re.subn(r'\n(\s*):\n\s*:\s*("r"\([^)]*\))\s+("l"\([^)]*\))\);',
                 lambda m: f'\n{m.group(1)}: : {m.group(2)}, {m.group(3)});', s)
assert n > 0, "no sites matched"
# Safety: the only textual change allowed is collapsing ":\n  :" to ": :" and
# replacing the whitespace between the two constraints with a comma. Verify the
# result still contains every tcgen05.cp template string it had before.
assert new.count("tcgen05.cp") == s.count("tcgen05.cp"), "lost a tcgen05.cp site"
open(p, "w").write(new)
print(f"patched {n} inline-asm operand lists in {p}")
PY
echo "--- diff (should be only ':' collapses and one ',' per site) ---"
( cd "$ROOT" && git --no-pager diff --stat -- include/cute/arch/copy_sm100.hpp && \
  git --no-pager diff -U0 -- include/cute/arch/copy_sm100.hpp | grep -cE '^[+-][^+-]' )
