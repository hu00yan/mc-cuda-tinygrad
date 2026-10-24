# Finding: the UOp-level redundancy detector does not survive contact with the backend

**Date:** 2026-10-24 · **Author:** Bob · **Status:** negative result, reported as such

## What was proposed

A static, read-only detector that finds loop-invariant subexpressions inside a
`REDUCE` and reports them as hoistable — motivated by `notes/16`: the generated
KDA kernel issues 128 `MUFU.EX2` where a hand-written Triton kernel issues 3,
which was attributed to the decay being recomputed once per contracted term.

## What was built

`tools/hoist_detect.py` — walks tinygrad UOp graphs, finds `REDUCE` nodes, and
reports pure, expensive (`EXP2`/`LOG2`/`SIN`/`SQRT`/division) subexpressions
whose shape does not carry the reduced extent. It works at that level: for

```python
(A * decay.exp2()).sum(0)      # A is (64,64), decay is (64,)
```

it correctly reports `EXP2 shape=(64,)` with redundancy 64.

## What falsifies it

The UOp graph **cannot distinguish "computed inside the reduction" from
"hoisted out"**. Both of these produce the *structurally identical* graph:

```python
(A * decay.exp2()).sum(0)      # decay written inside the expression
d = decay.exp2(); (A * d).sum(0)   # decay precomputed by the author
```

Redundancy in the emitted code is introduced by **backend lowering**, below the
UOp graph. So I measured the emitted CUDA directly, using the repo's own
`matmul/tg_backend.py` capture harness (real codegen path, sm_80):

| expression | kernels | `exp2(` in emitted CUDA |
|---|---|---|
| `(A * decay.exp2()).sum(0)` | 1 | **1** |
| `d = decay.exp2(); (A * d).sum(0)` | 1 | **1** |
| `A.sum(0)` (control) | 1 | 0 |
| KDA-like: decay inside reduce | 1 | **4** |
| KDA-like: decay after reduce | 1 | **4** |

**The emitted code is identical either way.** tinygrad's scheduler already
hoists these invariants. At the scales tested, there is no redundancy for a
detector to find.

## What this means

1. **The detector as designed has no signal.** It reports graph structure, not
   backend behaviour, and the two disagree. Shipping it would produce a list of
   "opportunities" that are already exploited — worse than useless, because it
   would look like a finding.
2. **The 128-vs-3 result in `notes/16` is not explained by a missing hoist pass**
   at this level. Either it comes from a different mechanism (the `[C,C,K]`
   intermediate materialisation, per `notes/03` and `notes/09`), or from a
   fusion/loop-structure decision below the level captured here. **I have not
   established which, and I am not going to assert either.**
3. **The correct next instrument is the one that already worked**: compare
   *emitted source* (or SASS) counts for the real KDA kernel against the real
   hand-written kernel, on the actual subgraph from `kda/`, rather than on
   miniature reconstructions. `notes/16` already does this. Re-deriving it at a
   smaller scale does not reproduce the effect, which is itself informative: the
   effect is scale- or structure-dependent, not a general compiler gap.

## Recommendation

Do not build the hoisting detector as specified. If the 128-vs-3 gap matters,
the next step is to locate **which pass** in the real KDA lowering produces it,
by bisecting the emitted source of `kda/tiny_kda.py` against `notes/16`'s kernel
— a different tool, aimed at a narrower question, with a real artifact to
compare against.

This is what @prime asked for the first step to determine, and the answer is
"not at this level".
