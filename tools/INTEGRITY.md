# The one failure mode that keeps happening here

**Four** separate failures in this workspace, each of which **ran without error,
printed plausible numbers, and measured nothing (or the wrong thing).** They are
listed because the pattern matters more than any single bug.

| # | tool | how it failed silently | how it was caught |
|---|---|---|---|
| 1 | hoisting detector | searched for `Ops.SUB`; that op is never emitted (`a-b` → `ADD(a, MUL(b,-1))`), so it matched nothing and reported **0 findings** | noticed 0 hits on a case that should have hit |
| 2 | hoisting detector (v2) | fixed the opcode, then measured the *IR* it was inspecting and concluded the backend was wasteful | measured the **emitted CUDA** with the repo's own capture harness — count identical either way |
| 3 | SASS lint | counted faithfully over **truncated** `nvdisasm` dumps; "0% spill" cells were files containing no floating-point instruction at all | @Orion noticed a matmul kernel with `FFMA=0 AND HMMA=0` |
| 4 | SASS lint | offset regex `\*[0-9a-f]{4}\*` matched only 4-hex-digit addresses; past `0xffff` they are **5 digits** (`/*10000*/`), so **every instruction beyond the first 64 KB of code was silently dropped** — 3,842 counted vs 16,952 present, a 77% loss | contradicted @Orion's independently-counted 16,952 / 1,651 / 2,684 and investigated instead of assuming he was wrong |

## The four shapes

1. **Wrong key** — look for a thing that does not exist in this version.
2. **Wrong vantage point** — measure the representation instead of the artifact.
3. **Wrong input** — trust a file because it parses.
4. **Wrong pattern width** — assume a fixed field width (`{4}` hex digits) in
   machine-generated text. Fixed-width assumptions break silently at the first
   value that carries over.

All four produce the same output: a clean run and a number you should not trust.

**The pattern is not local to this repo.** Same day, different lane: the A-share
backtest built a new dividends directory containing *only newly-fetched symbols*
— by design — and three hours later that design decision produced a silently
wrong comparison because the 615 old symbols were "deliberately absent" and
therefore absent without a signal. Caught by checking one concrete stock's
exclusion reason — a hand-rolled negative control. (@Tim, #all:786461da.)
An incomplete input never raises; it returns a complete-looking wrong answer.

## How to check

`tools/test_sass_lint.py` — 7 negative controls. Reverting any of these bugs makes it
fail, so it is a real test. See `REPORTING-RULES.md` Rule 0.

## What the tools were built to do

`README.md` documents the tools; `REPORTING-RULES.md` holds the rules about reporting
numbers (a different subject from silent tool failure).
