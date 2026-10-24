#!/usr/bin/env bash
# Print the most common SASS opcodes in a compiled binary.
# Usage: sass_top.sh <binary> [count]
set -uo pipefail
b="$1"; n="${2:-20}"
echo "=== $(basename "$b") ==="
cuobjdump -sass "$b" 2>/dev/null \
  | grep -oE '[[:space:]]+[A-Z][A-Z0-9_]*(\.[A-Z0-9_]+)*' \
  | tr -d ' \t' \
  | sort | uniq -c | sort -rn | head -"$n"
