# tools/ — static analysis of generated GPU kernels

**Read `CALIBRATION.md` before trusting any number here, and `REPORTING-RULES.md`
for how a number may be phrased. `INTEGRITY.md` covers how tools fail silently.**
Every tool in this directory is designed to be killed by evidence rather than
defended. One of them already was.

## The only number that survived triangulation

> **LDL density 7.0x** (T4 15.8% vs sm_80 2.3% of each kernel's instructions,
> i.e. 2,684 vs 63 loads). Reproduced by four independent implementations —
> `sass_lint.py`, `sass_lint_independent.py`, an independent `grep`, and an
> independent `awk` — see `REPORTING-RULES.md` Rule 4.

**Scope, and this is not a formality.** 7.0x is a **kernel property, not a project
conclusion**: the same hand-written source spends 7x more of its instruction
budget on spills on T4 than on sm_80. It is **not** "TinyGrad is 7x worse" and
**not** "hand-written is 7x better". Lead with the density; put the absolute
count (42.6x) in parentheses, because a bare absolute number gets read as
"the compiler is 42x worse" — which is how the earlier `43x` was destroyed.

And the thing this protects:
> **On free cards, "generated vs hand-written" has never actually been compared**
> — not because the result was bad, but because it was never run. Every quotable
> number here is about *how hand-written code behaves on T4*, not about *whether
> generated code is good*.

## The tools

### `REPORTING-RULES.md` — how a number may be phrased
Negative controls, bare multiples, mean-vs-median-vs-max, magnitude-excludes-not-
attributes, degrees of independence, residuals must not be named, probe-width vs
claim-width, and "say unknown". **Read this before quoting anything from here.**

### `factor_lint.py` — find factorisable decay (works, verified)
Flags `exp(a_i - b_j)` where the subtraction is broadcast across a contraction.
That form evaluates the exponential once per `(i,j,d)`; factored into
`exp(a_i) * exp(-b_j)` it is once per `(i,d)` and `(j,d)`. A compiler legitimately
cannot do this — `exp(a-b)` and `exp(a)*exp(-b)` differ in IEEE float — so it has
to be requested.

Verified on the real KDA decay `(g[:,None,:] - g[None,:,:]).exp()` at C=64, K=128:
**524,288 → 16,384 `exp2` evaluations (32×)**. Negative controls (single-array
`exp`, no subtraction) correctly stay silent.

### `sass_lint.py` — count instructions on real dumps (works)
Counts `MUFU.EX2`, `HMMA`/`HGMMA`, `STL`/`LDL` (register spills) and total
instructions per `.sass.txt`. Sorts by spill pressure because that turned out to
be the bigger signal.

Prints every figure with `(kernel, cc, config)` because three different
experiments have already been quoted as if they were one gap. **Never** compare a
number from this tool with one from `notes/16` or `FACTORISATION-FINDING.md`
without matching all three subscripts.

### `test_sass_lint.py` — negative controls for `sass_lint.py` (run this first)
Seven controls: 5-hex-digit offsets, width-suffixed spill opcodes, predicate-guarded
`EXIT`, a truncated dump that must be flagged, a healthy dump that must not be.
Verified by reverting the bugs — 3 controls fail when bug 4 is reintroduced, so
they actually test something.

**The principle: every automated check needs a deliberately broken input it must
reject.** A check that always returns 0 is indistinguishable from one that always
passes.

### `sass_lint_independent.py` — the structurally different third path
Same counts, zero shared machinery: tokenizes with `str.split()`, no `re`, no
quantifiers, no field widths. Exists because a "second implementation" that uses
another regex with the same offset assumption is **not** independent — it will
agree with a wrong answer.

Independence is a **degree, not a binary**: as @Orion put it, the standard is
"does not share a failure mode", not "is not the same code". Two implementations
agree only outside the assumptions they share — so ask *which assumptions they
share*, not *whether they differ*. Four paths / three mechanisms now agree
(this tool, `sass_lint.py`, an independent grep, an awk field comparison).

Agrees with `sass_lint.py` on all fields:
`aot16_cc75_BV64_w4_s2 → 16952 / STL 1651 / LDL 2684 / EX2 33`,
`aot16_cc80_BV64_w4_s2 → 2768 / 64 / 63 / 3`.

### `hoist_detect.py` — FALSIFIED, kept as a negative result
Proposed to find loop-invariant expressions recomputed inside a reduction. It
reports them correctly at the UOp level and **that is exactly why it is wrong**:
the UOp graph cannot distinguish "computed inside the reduction" from "hoisted",
and the emitted CUDA has the same `exp2` count either way. tinygrad already
hoists. See `HOIST-FINDING.md`.

## Two traps that cost real time — both produced tools that looked fine

1. **`Ops.SUB` does not exist in this tinygrad.** `a - b` lowers to
   `ADD(a, MUL(b, -1))`. A detector searching for `Ops.SUB` matches nothing and
   reports **0 findings while appearing to run correctly**. An earlier version of
   `factor_lint.py` did exactly this.
2. **Measurement path vs measurement target.** `hoist_detect.py` inspected the IR
   and measured the IR, and concluded the backend was wasteful. The backend was
   not. **Validate against the emitted artifact, never against the IR you are
   inspecting.**
3. **Truncated input.** Dumps that parse are not dumps that are complete. See the
   integrity gate below.
4. **Fixed-width field assumptions.** SASS offsets are **4–8 hex digits**, not 4.
   `/*10000*/` is a real instruction. A `{4}` quantifier silently dropped **77%**
   of a kernel's instructions (3,842 counted vs 16,952 present). Spill opcodes
   likewise carry suffixes: `STL.64`, `LDL.LU`, `LDL.LU.64`.
   **This is why the negative controls exist.**

## The labelling rule

Every SASS/`exp2` figure carries `(kernel, cc, BV/warps, compile-method)`. If you
cannot supply them, write `unlabelled`. The three figures below are **not**
comparable to each other and were, at one point, quoted as though they were:

| source | figure | what it actually is |
|---|---|---|
| `notes/16` §3.3 | 128 vs 3 | Triton offline compile, in a container, one reduce |
| T4 AOT batch (cc75) | 33 / 17 / 2 | Kaggle, varies with BV and num_warps |
| `FACTORISATION-FINDING.md` | 64 → 32 | gram factorisation, code-level, one gram kernel |

## What the SASS data actually showed

On cc100, same kernel, config varied:

| config | exp2 | HMMA | spill | spill% |
|---|---|---|---|---|
| BV32_w2 | 5 | 128 | 383 | 11.1% |
| BV32_w4 | 34 | 0 | 0 | 0.0% |
| BV64_w2 | 4 | 128 | 731 | 20.7% |
| BV64_w4 | 33 | 0 | 127 | 3.7% |

Tensor-core configs keep `exp2` low and pay 11-21% spills. Non-tensor-core
configs accumulate 33-34 `exp2` and barely spill. **These are mutually exclusive
at this shape**, which means the "43× gap" framing may be the wrong question:
there may be no configuration that is good on both, only a choice of which cost
to pay. Settling that needs timing on real hardware.
