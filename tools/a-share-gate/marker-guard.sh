#!/usr/bin/env bash
# Guard: the marker literal used by `-m "slow"` must actually be declared.
#
# WHY: after fixing the "empty vs unreadable" branch, a DIFFERENT pair still looks
# identical — "genuinely 0 slow tests" and "marker name misspelled". A -m "slow-test"
# collects fine, matches 0, and the gate prints "confirmed: 0" while one exists.
#
# WHY ask pytest instead of grepping pyproject.toml: file layout (quotes, indent,
# line position) is not a contract, and whoever reformats it does not know a parser
# is watching. `pytest --markers` IS a public output. Ask, don't parse.
#
# Usage: MARKER=slow bash marker-guard.sh   -> exit 0 if declared, 1 if not/unreadable
set -u
MARKER="${MARKER:-slow}"

if ! markers=$(uv run --with pytest python -m pytest --markers 2>&1); then
    echo "✗ 无法读取 marker 定义（pytest --markers 失败）——不能据此认为 marker 不存在"
    exit 1
fi
if ! printf '%s\n' "$markers" | grep -qE "^@pytest\.mark\.${MARKER}:"; then
    echo "✗ marker '${MARKER}' 未声明 —— -m \"${MARKER}\" 会静默收集 0 条"
    exit 1
fi
echo "OK: marker '${MARKER}' 已声明"
