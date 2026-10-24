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
