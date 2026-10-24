#!/usr/bin/env bash
# Count occurrences of fixed SASS substrings in a binary's disassembly.
# Usage: sass_grep.sh <binary> <substring> [substring ...]
set -uo pipefail
b="$1"; shift
sass=$(mktemp)
cuobjdump -sass "$b" > "$sass" 2>/dev/null
total=$(grep -cE '/\*[0-9a-fA-F]{4}\*/' "$sass")
echo "=== $(basename "$b"): $total instructions ==="
for pat in "$@"; do
  n=$(grep -cE "/\*[0-9a-fA-F]{4}\*/[[:space:]]+${pat}" "$sass" || true)
  printf '  %-24s %s\n' "$pat" "$n"
done
rm -f "$sass"
