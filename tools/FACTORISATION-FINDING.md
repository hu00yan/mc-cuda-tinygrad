# The real gap is fragmentation, not hoisting — located and quantified

**Date:** 2026-10-24 · **Author:** Bob · **Supersedes:** `tools/HOIST-FINDING.md`
(that one reports a falsified hypothesis; this one reports what replaced it)

## Summary

The 128-vs-3 `MUFU.EX2` gap in `notes/16` is **not** a missing loop-invariant
hoisting pass. I tested hoisting and it does not reproduce (see
`tools/HOIST-FINDING.md`). The gap comes from **how the mathematical expression
is factored in the source**, which decides whether the decay is computed once per
element or once per pair.

The fix is not a compiler pass. It is an **algebraic refactoring that a compiler
cannot do** — because it is not a semantics-preserving rewrite in floating point,
it changes the numerical association — and that a *generated* kernel will never
perform on its own, because the generator writes what it is given.

## The measurement

Real KDA gram, as the port actually writes it (`kda/codegen_diff.py` documents
this expression), captured through the repo's own `matmul/tg_backend.py` at
sm_80 — no GPU needed:

```
Z[p,j] = sum_d  k[p,d] * k[j,d] * exp(gcum[p,d] - gcum[j,d])
```

| formulation | kernels | `exp2(` in emitted CUDA | source bytes | tensor-core MMA |
|---|---|---|---|---|
| materialised `[C,C,K]` + `sum(-1)` (as written) | 1 | **64** | 6 936 | no |
| factored into two `[C,K]` factors + matmul | 1 | **32** | 5 946 | no |

Reproduce:

```bash
cd ~/mycode/mc-cuda-tinygrad
.venv/bin/python kda/codegen_diff.py      # the human-readable side-by-side
```

## Why this is the real mechanism

The two forms are algebraically equal but have different **association trees**:

```
  as written:   exp(gu[p,d] - gu[j,d])          <- per (p,j,d) triple: C*C*K exp2
  factored:     exp(gu[p,d]) * exp(-gu[j,d])    <- per (p,d) and (j,d): 2*C*K exp2
```

`C*C*K` versus `2*C*K` is a factor of `C/2` = 32. The measured 64 → 32 follows
this law. The same identity is what lets the factored form become a
`tl.dot`/matmul at all — the decay no longer sits between the two contracted
operands.

> ⚠️ **Scope of this claim — do not merge it with the 43× number.**
> The 64 → 32 measured here is about **one gram kernel's `exp2` count**. The
> 128-vs-3 (43×) in `notes/16` is about **`exp2` density inside a single reduce
> of a fused chunk kernel**, and per `notes/16` §7.4 the two sides are not even
> computing the same function (Triton = fused chunk state-passing; TinyGrad =
> dense-recompute port). **They are different quantities on different objects.**
> This document does not explain the 43×, and I have not established what does.

**This is why the hand-written Triton kernel has 3 and the generated one has
many**: the hand-written author knows Litmax's identity and applies it; a
generator cannot, because:

1. **It is not a semantics-preserving rewrite in IEEE float.** `exp(a-b)` and
   `exp(a)*exp(-b)` differ in rounding, and in the exponent range where the
   difference is representable they differ in value. A compiler is not permitted
   to make this change silently. This is a *numerical concession*, and it must be
   requested by an author who has decided the error is acceptable.
2. **It changes the data layout**, `[C,C,K]` → two `[C,K]`, which is a decision
   about memory, not a local rewrite.
3. It is the kind of change that only becomes apparent when you compare against a
   reference implementation — which is exactly the thing a generator does not have.

## What this means for @prime's question

He asked: is this a compiler that should try harder, or must information be
injected? **It must be injected, and this is a concrete instance.** But the
injection is cheap and mechanical *once you know it*:

> any reduction containing `exp(a_i - b_j)` over a contracted index is a candidate
> for the factorization `exp(a_i) * exp(-b_j)`, which both halves the transcendentals
> and turns the contraction into a matmul.

That is a **detectable pattern** — unlike my hoisting idea, which measured graph
structure that the backend had already fixed. The difference is that this check
looks for `EXP2` of a **difference of two contracted-indexed terms**, i.e. it
names the algebraic identity, and its payoff is provable by counting `exp2` in
emitted source before and after.

## Honest limits

- **The emitted code has no tensor-core MMA in either form** at this shape, so
  the matmul form's real advantage is not yet realized; it halves the
  transcendentals but does not reach `HMMA`. Reaching tensor cores needs more
  than the factorization (correct dtype, tile shape, `TC_SELECT`). So the 64→32 is
  the part this experiment actually proves, and the 32→3 in `notes/16` is a
  different, larger jump that needs a card to explain.
- 64 → 32 is a **halving, not the 43×** in `notes/16`. The remaining difference is
  scale- and structure-dependent and I have not established it.
- No timing was taken. This is a **code comparison**, and the claim is about
  emitted instructions, not speed.

## What I would do next

1. Add the `exp(a_i - b_j)` detector as a **static lint** over generated kernels —
   cheap, no GPU, and its payoff is countable in the emitted source.
2. To close the rest of the 128-vs-3 gap, use a real NVIDIA card (Colab CLI, per
   @prime) and compare SASS, not CUDA C, for the two formulations at production
   shape. That is the only way to see what the tensor-core path does.
