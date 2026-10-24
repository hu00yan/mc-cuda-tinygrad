# KDA on TPU — the comparison design

Read `research/TPU.md` first. It establishes the facts, the taxonomy mapping, the
prior work, and three corrections to the original brief. This file is only the
runnable part.

**Everything in this directory that can run on CPU, does.** There is no TPU
reachable from this environment and the repo venv has no `jax` (verified:
`import jax` → `ModuleNotFoundError`), so `xla_kda.py` and `pallas_kda.py` are
written but **never executed**. Every number below is analytic or CPU-verified;
none is a TPU measurement, and this file says so wherever a number would
otherwise be mistaken for one.

```bash
cd /Users/huyan00/mycode/mc-cuda-tinygrad
.venv/bin/python kda/tpu/roofline.py      # per-generation peaks + ridge point
.venv/bin/python kda/tpu/shapes.py        # 3 sides, materialisation, speedup bound
.venv/bin/python kda/tpu/tiling_check.py  # Pallas legality rules, with sources
.venv/bin/python kda/tpu/test_design.py   # 16/16 against kda/ref.py in float64
```

## The three sides

| side | file | what it is | variable it isolates |
|---|---|---|---|
| **S1 `xla_jax`** | `xla_kda.py` | KDA transcribed into `jnp` exactly as `kda/tiny_kda.py` is into tinygrad. The two `[C,C,K]` decay tensors are in the graph. `jax.jit`'d. | baseline: *does a much better whole-graph compiler fix the CUDA study's finding?* |
| **S2 `pallas_hand`** | `pallas_kda.py` | Same algebra, hand-tiled. Exp folded into the contracting operand so C²K never exists; C=128 with BC=16; grid `(HV, NT)`; `dimension_semantics=(PARALLEL, ARBITRARY)`; `buffer_count=3`; explicit `scratch_shapes`; fp32 accumulators. | *a human reached for a low-level dialect* |
| **S3 `jax_rewrite`** | (the `folded_gram` rewrite in `test_design.py`, transliterated) | Pure `jnp`, same rewrite applied to the graph **before XLA sees it**. | *the dialect bought anything beyond the rewrite* |

**S3 is the control.** `S2 − S1` is the boring comparison and restates
`research/TAXONOMY.md` item 6. `S2 − S3` is the real question: how much of the
hand-written win is the **algorithm** and how much is the **dialect**? If
S3 ≈ S2, the TPU story is entirely item 6 and Pallas was decoration — a
publishable result, and consistent with XLA's measured fusion strength. If
S3 ≪ S2, tile-level control has independent value on an architecture with no
warps.

**Pre-registered: S3 captures most of S2's win, and S2/S3 < 1.5×.** Reasons are
in `research/TPU.md` §4c. A residual above 1.5× is a surprise and gets reported
as one.

## Shapes

Two regimes, because JAXBench's ≥60%-MXU-utilisation sizing rule and "the
intermediate must be visible" cannot both be satisfied — KDA's `[C,C,K]` tensor
is 64× larger than everything else in the operator.

| regime | shape | one C²K tensor (bf16) | note |
|---|---|---|---|
| A | `B=2, T=1024, H=HV=4, K=V=64, C=64` | 67.1 MB | all three sides run; ~26× traffic gap |
| B | `B=1, T=4096, H=HV=8, K=V=128, C=128` | 1.07 GB | S1's traffic is 26× compulsory; S1 may OOM on a 16 GiB v5e-8 |

The CUDA study's own target (`B=4, T=4096, H=32, K=128, C=64`) needs 8.6 GB per
intermediate — 17.9 GB of S1 traffic. It does not fit a free TPU. `shapes.py`
prints the whole table including that row.

## Metrics, cheapest first

Two of these need no TPU time at all. Do them before spending a quota hour.

| # | metric | how | arch-dependent? |
|---|---|---|---|
| **M1** | HBM bytes + FLOPs in the compiled program | `jax.jit(f).lower(*a).compile().cost_analysis()` | **no** |
| **M2** | `dma.hbm_to_vmem` count + bytes in the LLO | `--xla_jf_dump_to`, `--xla_jf_dump_llo_text` | no |
| **M3** | VLIW bundle count | same dump | no |
| **M4** | fusion / custom-call count in the HLO | `lower().compile().as_text()` | no |
| **M5** | static constraint count | `tiling_check.py` over each side's plan | no |
| **M6** | correctness at 1e-12 / 1e-6 / 1e-2, **and the final state S** | vs `kda/ref.py` | no |
| **M7** | MXU utilisation = FLOPs / (t·peak) | Perfetto device events, median of 50 | yes, as a fraction |
| **M8** | `t / t_hard_floor` | M7 against `max(t_mem, t_flop)` | fraction |

**M1 is the go/no-go gate.** If S1's byte count is not ~25× S2's, the rest of the
experiment is not worth the hour — stop and find out why. `shapes.py` predicts
279.2 MB vs 10.8 MB at the regime-A shape; if the measured numbers disagree with
that by more than the fudge for the `[C,C]` blocks XLA might legitimately fuse,
the materialisation model is wrong and nothing downstream is trustworthy.

**No raw milliseconds, and no cross-generation absolute comparison.** Kaggle's
v3-8 is deprecated and the current accelerator is v5e-8
(`research/TPU.md` §1b), so a raw ms on an unnamed generation is not a result.

## Two results already established on CPU

Both from `test_design.py` (16/16 pass) and both changed the design.

**1. C = 64 → 128 is numerically free.** The `(8,128)` minor-lane constraint means
a `[C,C]` fp32 accumulator is physically `[64,128]` — 2× the VMEM plus a
relayout on every access. The obvious objection is that a wider chunk widens the
exponent range. Measured, on a gate drawn from `kda/SPEC.md` §4.1: with the
`BC=16` sub-chunking that `fla/ops/kda/chunk_intra.py` already does, the
exponentiated range is **independent of C — identical to 2.8e-14 at C=64, 128 and
256**. So raise C, keep BC=16, change nothing else. The thing XLA cannot tell
you is that C=64 was wasting half of every vector register, and that fact is in
the *tile shape*, not the *program*.

**2. The rewrite is exact but its numerical bill is not uniform.** The
factorisation `exp(gc_i − gc_j) = exp(gc_i − gn)·exp(gn − gc_j)` is exact for any
per-channel reference, so C²K can always be eliminated — verified against
`kda/ref.py` in float64 to 2.5e-16. What `BC` buys is a *bound*, and the bound
only holds for the diagonal sub-chunk blocks: the farthest off-diagonal block's
right operand needs **2.5×** the diagonal bound. `fla` hand-schedules around
exactly this ("4 diagonal row-loops + 6 off-diagonal matmuls",
`kda/SPEC.md` §4.4). **A Pallas port must re-derive that decomposition and
Mosaic will not do it.** This is where the hand-written side should actually earn
its keep — not in fusion, which XLA wins.

## What is pre-registered as the answer

- KDA is **memory bound on every TPU generation at every shape we can afford**:
  OI ≈ 113 FLOP/byte against a v6e ridge of 560 and a v3 ridge of 137. Folded or
  not. MXU utilisation will be low even for a perfect kernel, so reporting it as
  the score would be a mistake. The score is bytes.
- The speedup is **bounded analytically at ~25.9×** (S1's 279.2 MB of unavoidable
  traffic against S2's 10.8 MB) at the regime-A shape. A measured gap much
  smaller than that means the intermediate is not the binding constraint, and
  that is a finding.
- The C²K tensor is in S1's graph and not in S2's or S3's, because it is in the
  *graph* and the graph is the semantics. XLA will not fuse it away: reading
  `jax/_src/nn/functions.py`, even `jax.nn.dot_product_attention(implementation='xla')`
  has no online softmax and no tiling — it materialises `[B,N,T,S]` and relies on
  fusion plus a large VMEM.

## Runbook (9 h per session, 20 h/week, and a TPU is not guaranteed)

**Step 0 is not a formality.** Another agent's Kaggle run in this repo
(`out/kaggle/tpu-facts-v2/`) asked for a TPU and got `jax_sees_tpu: false`,
`jax_devices: ["cpu x1"]`, no `/dev/accel*`. A session is an *opportunity*, not
a resource. Build so that a session is worth spending — which is what M1 is for.

1. `pip install -U "jax[tpu]"`, then `tpu_info()`. Record the generation and fill
   `roofline.GENERATIONS[gen]['vmem_mb']` and `['hbm_gib']`. Without this, M7 is
   meaningless and the VMEM check is a warning rather than a result.
2. Re-run the CPU checks in the notebook to confirm shapes before touching the
   TPU. **Free, and it is what you do while waiting for a session.**
3. **M1 for all three sides. No timing yet.** Seconds of compute, no TPU time
   needed. Go/no-go gate: if S1's byte count is not ~25× S2's, stop.
4. Dump LLO for all three. M2, M3. This is the number we most want.
5. Only then M7, and only as a fraction of the roofline.
6. Sweep **one knob at a time**, in this order: `buffer_count ∈ {2,3,4}` at fixed
   blocks → block shapes → `C ∈ {64,128}`. The first two are (ii) and the third
   is a different algorithm. The CUDA taxonomy's own cautionary result — an
   A100-tuned kernel running 100× off on GH200 with no correctness signal — is
   why.
7. Save the dumps. An LLO dump pair (XLA vs Mosaic) for the same operator is the
   most reusable artefact this study can produce, and it does not go stale the
   way a millisecond figure does.

## Files

| file | role | runs today? |
|---|---|---|
| `../research/TPU.md` | facts, taxonomy mapping, citations, prior work, unresolved questions | — |
| `roofline.py` | per-generation peaks (one URL each), ridge point, `classify()`, hard time floors. Handles the chips-vs-TensorCores distinction that silently 2×'s every fraction. | **yes** |
| `shapes.py` | the three sides; FLOP/byte counts; the C²K materialisation model; the 25.9× bound; VMEM vs pipeline depth; the MXU-fill calc for `[C,C]` blocks. | **yes** |
| `tiling_check.py` | Pallas TPU rules R1–R8 with source quotes, plus the `(8,128)` minor-lane check BlockSpec does *not* catch. 10 worked examples. | **yes** |
| `test_design.py` | 16 checks against `kda/ref.py` in float64. | **yes** |
| `xla_kda.py` | S1. | **no** — no jax |
| `pallas_kda.py` | S2, with every forced decision annotated as `DECISION`. | **no** — no jax, no TPU |

## Limits

- **No TPU measurement exists.** Everything above is analytic or CPU-verified.
- **No backward pass.** Forward only, as in the CUDA study.
- **The Pallas kernel is a design sketch.** The open questions are listed in
  `research/TPU.md` §6; the load-bearing one is whether `pl.when` survives
  unrolling for a C=128 UT transform (127 dependent steps), which is the most
  likely thing to be slow in it.
- **v3, v5e, v6e VMEM capacities are UNVERIFIED.** `roofline.py` carries `None`
  and the checker downgrades to a warning. Get them from `tpu-info`.
- **S3 has no file.** It is `test_design.py::folded_gram` transliterated to
  `jnp`. Write it when a TPU is available; it is the side that decides the study.
