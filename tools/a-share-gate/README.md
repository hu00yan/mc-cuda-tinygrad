# verify_gate.py — archived copy (NOT live)

These files were written for `~/mycode/a-share-lab` to prove that repo's
`scripts/gate.sh` can actually go red. They are **archived here, not running there**:
`a-share-lab` belongs to @Tim, and leaving untracked files in it blocked his gate run.

## What they do
- `test_gate_has_teeth.py` — a canary test that **always fails**
- `conftest.py` — `collect_ignore`, so the canary never enters the default suite
  (a permanently red suite equals no suite)
- `verify_gate.py` — extracts `run()` and the fast-layer pytest command **verbatim from
  the real `gate.sh`** (a copy would only test the copy), points them at the canary, and
  asserts the gate reports failure

## Why extraction, not a copy
A copied command tests your restatement of the gate. Extraction tests the gate's body, so
"`run()` swallows failures" is caught too.

## Verified before archiving (3 directions, both sides each)
| break | expected | measured |
|---|---|---|
| `if ! "$@"` → `if true` | red | exit 1, `GATE_WOULD_FAIL` |
| `collect_ignore = []` | error | exit 1, names the conftest |
| pytest line split across a `\` continuation | error | exit 1, `GateShapeError`: "found 0" |
| nothing | green | exit 0 |

`gate.sh` was restored byte-identical after each (sha256 `077b0867...`).

## To use them in a-share-lab
```bash
cp tools/a-share-gate/verify_gate.py  ~/mycode/a-share-lab/scripts/
mkdir -p ~/mycode/a-share-lab/tests/canary
cp tools/a-share-gate/{conftest,test_gate_has_teeth}.py ~/mycode/a-share-lab/tests/canary/
cd ~/mycode/a-share-lab && uv run python scripts/verify_gate.py    # ~60-75s
```
Ask @Tim first — his repo, his call.

## Known gap
The verifier is itself a check and could break silently. I tripped it by hand but did
**not** automate a probe for the probe. **Manually tripped at this layer, not automated.**

## marker-guard.sh (added after the gate fix)

Guards the sibling bug left by the "empty vs unreadable" fix: **"genuinely 0 slow
tests" and "marker name misspelled" look identical.** A `-m "slow-test"` collects
fine, matches 0, and the gate prints "confirmed: 0" while one exists.

Asks `pytest --markers` rather than grepping `pyproject.toml`: **file layout is not a
contract**, and whoever reformats it does not know a parser is watching. Same method
as the gate's own `--collect-only` reverse-collection — ask pytest, don't parse files.

Verified both directions in a-share-lab:
| state | result |
|---|---|
| marker declared (healthy) | `OK: marker 'slow' 已声明`, exit 0 |
| `slow:` renamed to `slowtest:` in pyproject.toml | `✗ 未声明 — 会静默收集 0 条`, exit 1 |

`pyproject.toml` restored byte-identical (sha256 `31696ae7…`) after the negative control.
**Not installed in Tim's repo** — the deliverable stays here.
