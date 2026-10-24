# 07 — The crossover thesis, stated, attacked, and partly replaced

**Owner:** `notes/07-crossover-thesis.md` and `crossover/`. Nothing else in the
repo was modified.

**What this is.** The brief asked for a note that turns a hypothesis into a
testable prediction with quantities. Doing that honestly required first trying
to break it, and it broke in three places. This document states the claim,
derives predictions from cited capacities, reports what the corpus already
collected does to the claim, and then gives the version of the claim that
survives. The surviving version is not the one in the brief.

**Every number here is produced by a script in `crossover/`.** Nothing is
transcribed from another agent's report. Where a figure in the record is quoted
it is quoted *as the thing under test*, and the test either reproduces it or
says so. Re-running is one command:

```bash
cd /Users/huyan00/mycode/mc-cuda-tinygrad
.venv/bin/python crossover/test_control.py      # 64 checks, the suite that fails
.venv/bin/python crossover/devices.py          # capacities, one source per field
.venv/bin/python crossover/claim.py            # the predictions
.venv/bin/python crossover/literature.py       # every prior number, re-derived
.venv/bin/python crossover/attn_control.py     # the control experiment
.venv/bin/python crossover/residency_probe.py  # step or knee, measured
```

Status tags are the study's, from `notes/06-corrections-ledger.md`:
`[MEASURED]` executed here · `[MODELLED]` arithmetic on cited inputs ·
`[PREDICTED]` never a result · `[READ-FROM-SOURCE]` read from a document, not
executed · `[UNVERIFIED]` carried as unknown.

---

## 1. The claim, as given

> "The crossover between 'the compiler wins' and 'the human wins' is not a
> quality gradient at all. It is set by whether the materialised intermediate
> fits in on-chip memory, and it is a step function rather than a slope."

Stated as a prediction, for operator X on device D with intermediate size `s`:

| | claim |
|---|---|
| **P1** | winner(D, X, s) = compiler if `s ≤ C(D)`, human if `s > C(D)` |
| **P2** | `C` is a *capacity*, so moving to a device with a bigger `C` moves the crossover to a bigger `s`, and nothing else does |
| **P3** | the transition in the win-ratio is a **step**, not a slope |
| **P4** | the arch matters far less than `s / C(D)` |

If that is right it explains the whole spread in the literature: 0.95× parity
(small imaging pipelines, intermediates fit), 2.08× Pallas geomean
(memory-bound attention), 0.96× on GEMM (compute-bound, nothing to hide), the
35× counter-example at n=4096, the 380×/1347× authoring cliffs, and this study's
own KDA result. It also predicts the arch barely matters. That is a sharp,
falsifiable, heretical claim and I took it seriously.

**It does not survive intact.** Three specific things break, in increasing order
of seriousness: a sign error in the brief's reading of the TPU table, a
structural property of the byte accounting that makes P3 impossible, and a
result in the study's own corpus that the explanatory variable cannot reach.

---

## 2. What the brief got wrong about its own evidence

The brief's summary of `research/TPU.md` §1c reads:

> "the hand-written tiling was 35× slower than XLA's fused path, **overtaking
> only at n ≥ 8192** when the score matrix stops fitting on chip."

Re-derived from the table at `research/TPU.md:139-145`
(`crossover/literature.py` §1):

| n | scores fp32 | XLA fused (ms) | hand-written (ms) | **hand / XLA** |
|---|---|---|---|---|
| 512 | 1.0 MiB | 0.076 | 0.082 | **1.08** |
| 1024 | 4.0 MiB | 0.055 | 0.133 | **2.42** |
| 2048 | 16.0 MiB | 0.067 | 0.522 | **7.79** |
| 4096 | 64.0 MiB | 0.072 | 2.509 | **34.85** |
| 8192 | 256.0 MiB | 1.189 | 14.052 | **11.82** |
| 16384 | 1024.0 MiB | 4.445 | 89.567 | **20.15** |
| 32768 | 4096.0 MiB | 17.123 | 103.016 | **6.02** |

`[MODELLED]` from the table's own numbers.

**The hand-written version is slower at every n in the sweep. It never
overtakes, anywhere.** The 35× is real (2.509/0.072 = 34.85) but it is the
**maximum** of a hump with an interior maximum, not a threshold crossing. This
is the fourth time this study has carried a figure into prose that the source
does not support (`notes/06` §3 records three), and it is the most consequential
one, because the 35× was the thesis's central empirical support.

Two further consequences, both from the same table:

**(a) The ratio is non-monotone, which no capacity threshold produces.** R(n)
rises 32× from n=512 to n=4096, then goes 11.8, 20.2, 6.0. A threshold predicts
a monotone decline past the crossing, because the only thing past the crossing
is more bytes for the materialised form. And the score tensor at n=4096 (64 MiB)
is *smaller* than at n=8192 (256 MiB), where the penalty is 3× smaller. Whatever
the hand-written version's problem is at n=4096, it is not a byte count.

**(b) The capacity threshold *is* real on TPU — in XLA's own time curve.** XLA's
time is flat at 0.055–0.072 ms across a 64× range of n, then jumps 16.5× at
n=8192. For the flat part to be flat, the score tensor cannot be reaching HBM:
3 passes over 64 MiB in 0.072 ms would be 2796 GB/s, **3.41× the v5e's entire
819 GB/s HBM bandwidth**. And 3 passes over 256 MiB in 1.189 ms is 677 GB/s =
0.83× peak, which is a plausible HBM-bound number. Sweeping the pass count,
only **2 or 3** satisfy both constraints simultaneously — so the source table is
internally consistent, and the pass count (a modelling choice, criterion R-E) is
pinned to a narrow band by data nobody had checked it against.

That last check yields a derived bracket, which is a contribution:
`research/TPU.md` §6 lists v5e VMEM capacity as **UNVERIFIED** and says "Get
this from `tpu-info` before quoting any VMEM number." The table's own
fits/doesn't-fit column brackets the effective fused-intermediate capacity of
this v5e at **[64, 256) MiB** without any `tpu-info` at all. `[MODELLED]`. It
is a factor of 4 wide, so it cannot place a crossover inside it — but it is
narrower than nothing, and it is consistent with the table's own timings.

**The thesis needs the ratio between the two implementations. The ratio is
non-monotone and never crosses 1. The threshold is in a different quantity.
Conflating them is how the brief's sentence came to exist.**

---

## 3. The structural objection: both forms are O(n²), so there is nothing to
cross

This is the serious one, and it is arithmetic, not interpretation
(`crossover/claim.py` §1b, `crossover/attn_control.py` §2).

The thesis' mechanism is: the intermediate stops fitting, so the human's
advantage grows with `n`. That requires the two forms' traffic to have
*different growth rates*. They do not.

```
materialised:   score tensor written and read `passes` times      ->  O(n^2 * es)
online/tiled:   score tile on chip, BUT q and k are re-read once
                per key block                                      ->  O(n^2 * d * es / Bc)
```

Both are **O(n²)**. They differ in the *constant*, set by `(passes, Bc, d)`:

```
byte_ratio  =  passes * Bc / (2 * d)
```

and `Bc` is capped by on-chip capacity, `Bc ≤ C_onchip_per_SM / (d · es)`. So
capacity does enter — **but it caps the human's advantage, and a bigger cache
makes the human better, not the compiler. The sign is the reverse of the brief's.**

Two corollaries, both `[PREDICTED]` and both falsifiable:

1. **The human's advantage has a ceiling and no `n` at which it is exceeded.**
   `crossover/claim.py` computes it: **6.0× on a T4, 21.4× on an H100 or B200**
   (fp32, d=64, `passes`=3). Any measured ratio above ~21× refutes this and
   means something else is going on.
2. **H100 and B200 predict the same ceiling (21.4× each)** because both have
   228 KiB of shared memory per SM, while their L2 differs by 2.52×. Model 1
   (L2) predicts the opposite. **Two cards, one sweep, and the models are
   separated by a factor of 2.5 in opposite directions.**

**P4 dies here.** "The arch matters far less than the intermediate's size
relative to the cache hierarchy" is true in the normalised variable and
decorative in the real one. The arch sets `C_onchip_per_SM`, which sets `Bc`,
which sets the ceiling — so the arch sets the answer, and the L2 the brief names
is not in the expression at all. The hierarchy has two levels and the thesis
picks the wrong one. On a fused CUDA kernel the working set is *per-SM* (a
row-slab is produced and consumed on the SM that owns it), so the device-wide
total is the wrong denominator.

### 3.1 And P3 is refutable by a measurement taken today

`crossover/residency_probe.py` measures, on the machine in front of me, whether
"does the working set fit" produces a **step** or a **knee** in time. Two
float32 arrays, `np.vdot` (memory-bound; `a.sum()` is reduction-bound and
measures a flat 23 GB/s at *every* size, which is a null result about numpy's
reduction loop, not about caches — noted because the first version of this file
made exactly that error), 16 passes, best of 3.

`[MEASURED]`, Apple M1 Pro, 8 cores, 12 MiB L2 per P-core:

| working set | GB/s | × peak |
|---|---|---|
| 4.46 MiB | 259.0 | 1.00 |
| 6.24 MiB | 169.7 | 0.66 |
| 8.74 MiB | 122.6 | 0.47 |
| 12.24 MiB | 72.4 | 0.28 |
| 17.13 MiB | 57.6 | 0.22 |
| 66 MiB … 3.6 GiB | ~55 | 0.21 |

**VERDICT: KNEE.** The transition is spread over five sampled sizes and a 4.7×
bandwidth range. A step function would put the whole change between two
*adjacent* samples. The capacity is bracketed to [6.24, 8.74) MiB of working
set.

This does not settle anything about GPUs — different capacities, block sizes,
and compiler decisions. But a thesis that predicts a **discontinuity** in a
quantity that is **continuous everywhere it has actually been measured** is wrong
about the quantity, not just about the threshold. R-A in `crossover/claim.py` §4
is written in advance and says a gradual transition refutes the step claim.

**Two things to take from that measurement beyond the verdict:**

- **The usable capacity of a real cache is a fraction of its datasheet
  capacity** — here roughly half. Every capacity in `crossover/devices.py` is a
  *nominal* vendor number. So every crossover prediction below is computed
  against a denominator known to be too large by an unmeasured factor, and the
  real crossovers are **earlier** than predicted. That is a bias with a sign,
  not noise.
- The transition in a real memory hierarchy is a **kink in a continuous
  function**, i.e. `T_materialised = max(T_compute, bytes/BW)` crossing
  `T_online`. A crossing of two smooth branches is continuous. "Step, not
  slope" was the wrong sharpening: the capacity half of the claim is defensible,
  the step half has no support and one direct refutation.

---

## 4. The predictions, per device

Capacities in `crossover/devices.py`, one source per field. The four relevant
ones:

| device | arch | SMs | L2 | smem/SM | on-chip aggregate | DRAM |
|---|---|---|---|---|---|---|
| T4 | sm_75 | 40 | 4.00 MiB | 64 KiB | 2.50 MiB | 320 GB/s |
| P100 | sm_60 | 56 | 4.00 MiB | 96 KiB ⚠ | 5.25 MiB | 732 GB/s |
| L4 | sm_89 | 58 | 48.0 MiB | 100 KiB | 5.66 MiB | 300 GB/s |
| H100 | sm_90a | 132 | 50.0 MiB | 228 KiB | 29.39 MiB | 3.35 TB/s |
| B200 | sm_100a | 148 | 126 MiB | 228 KiB | 32.95 MiB | 8.0 TB/s ⚠ |

⚠ **P100's L2 is a finding.** `bench/selftest.py:103` gives
`L2_CACHE_SIZE = 24 << 20` (24 MiB) in a table whose own comment says the values
"are the published values for the three cards the study uses." NVIDIA's *Inside
Pascal* spec table gives **4096 KB = 4 MiB** for both P100 SKUs, corroborated by
arithmetic on the same page: eight 512-bit memory controllers × the Pascal
whitepaper's "each memory controller is attached to 512 KB of L2 cache" = 4096 KB.
**The repo's value is 6× too large**, and every capacity-derived crossover on
that card is linear in it. `bench/` is owned by another agent and is **not
modified here**; the discrepancy is recorded in `devices.CONFLICTS` and asserted
in `crossover/test_control.py` so it cannot be forgotten.

⚠ P100's 96 KiB smem/SM is itself `[CONFLICTING]`: the CUDA C Programming Guide
says 96 KiB for CC 6.0, *Inside Pascal* says 64 KiB for GP100. Both are carried
and both are printed.

### 4.1 Attention: the crossover does not exist in `n`

| device | M1 n\* (score = L2) | M2 n\* (per-SM slab loses tile shape) | M4 ceiling (no n) |
|---|---|---|---|
| T4 | 1024 | 256 | **6.0×** |
| P100 | 1024 | 384 | **9.0×** |
| L4 | 3547 | 400 | **9.4×** |
| H100 | 3620 | 912 | **21.4×** |
| B200 | 5747 | 912 | **21.4×** |

`[PREDICTED]`, fp32, B=H=1, d=64, `passes`=3. Three models, because the brief's
"C" is underspecified and the readings disagree by more than an order of
magnitude:

- **M1** — the score matrix equals device L2. The reading the brief's wording
  most naturally suggests.
- **M2** — the per-SM row-slab can no longer be a full-height tile:
  `C_onchip_per_SM / (M_native · es)`. This is what actually bounds a fused
  CUDA kernel, and it is a much stronger constraint than M1. **H100 and B200
  predict the same n\*** (912) because smem/SM is 228 KiB on both, while M1
  separates them by 2.52×.
- **M3** — pure bandwidth, no capacity: `n = s_comp · eff_ratio / (passes · B · H · es)`.
  **Device-independent** unless `eff_ratio` (achieved bandwidth, compiler ÷
  human) ≠ 1. A 16× change in `eff_ratio` moves the predicted crossover by 16×
  in `n` — more than moving across the entire device list (M1: 5.6×, M2: 3.6×).
  **`eff_ratio` is the quantity nobody in this corpus has measured**, and any
  version of the thesis that omits it is incomplete, not merely imprecise.
- **M4** — §3. The byte ratio, and the reason there is no crossover in `n`.

**Answer to "the predicted crossover point for the KDA operator on each
device":** there is no crossover in `n` for KDA either, for a sharper reason.
The KDA port's intermediate is `[B, NT, HV, C, C, K]` with `NT = T/C`, so its
element count is `B·H·T·C·K` — **linear in `T` and in `C`**, with the `C²`
cancelled by `NT`. A sweep in `T` and a sweep in `C` trace the same line, and
the threshold is a statement about the *shape*, not about the problem size. The
real KDA prediction is in §5.

---

## 5. KDA: the per-device prediction, and three findings about the study's own numbers

### 5.1 The 8.00 GiB is a different shape from the benchmark

`8.00 GiB` re-derives **exactly**, and it is `kda/cost.py:17`'s shape in bf16,
one decay tensor:

| shape | `[B,NT,HV,C,C,K]` | fp32 | bf16 |
|---|---|---|---|
| `kda/perf_kda.py:90` — **the benchmarked one** | 16,777,216 = 2²⁴ | 64.0 MiB | 32.0 MiB |
| `kda/cost.py:17` — **the cost-modelled one** | 4,294,967,296 = 2³² | 16.0 GiB | **8.00 GiB** |

`[MODELLED]`. Both numbers are in the record; neither says which shape the "the
hand-written side wins by construction" claim is about. **They are 128× apart,
and the claim is only true of one of them.**

### 5.2 The crossover, per device, at the benchmarked shape

`B=2, T=512, H=4, K=64, C=64, fp32` — the shape `kda/perf_kda.py` actually runs.

| capacity | T4 | P100 | L4 | H100 | B200 |
|---|---|---|---|---|---|
| **port intermediate / L2** | 16.0× | 16.0× | 1.33× | 1.28× | **0.51×** |
| verdict | human | human | human | human | **compiler may win** |
| **kernel intermediate / L2** | 0.25× | 0.25× | 0.021× | 0.020× | 0.008× |
| **port / on-chip aggregate** | 25.6× | 12.2× | 11.3× | 2.18× | 1.94× |
| **port / smem per SM** | 1024× | 683× | 655× | 287× | 287× |

`[PREDICTED]`. **The crossover is not a claim about a ratio. It is a per-device
statement, and it says the same operator at the same shape is "human wins,
obviously" on a T4 and "the compiler might win" on a B200.** The kernel side
fits everywhere, by 4× to 250×.

Threshold in `T` at `C=64` fp32, i.e. **where the benchmarked `T=512` sits**
`[PREDICTED]`:

| device | T\* = L2 / per-token-bytes | T=512 relative to it |
|---|---|---|
| T4 | 32 | 16.0× past → human |
| P100 | 32 | 16.0× past → human |
| L4 | 384 | 1.33× past → human |
| H100 | 400 | 1.28× past → human |
| B200 | 1008 | 0.51× of it → **compiler may win** |

At the cost-modelled shape, `C\* < 1` on every device and `fla`'s `chunk_kda`
raises unless `C ∈ {32, 64}`, so the port is over every device's L2 at every
**legal** chunk size: there the human wins by construction, at 4 GiB of avoidable
traffic. **At the benchmarked shape there is no such thing.**

### 5.3 The dtype flag moves the study across the threshold

Same shape, one flag apart `[PREDICTED]`:

| device | fp32 (64 MiB) | bf16 (32 MiB) | |
|---|---|---|---|
| T4, P100 | over | over | — |
| L4 | over | **fits** | **flips** |
| H100 | over | **fits** | **flips** |
| B200 | fits | fits | — |

`notes/06` §2 already says the KDA measurement must be fp32 because "at bf16 one
decay tensor is 33.55 MB, comfortably inside L2 on these cards… a bf16 run
cannot distinguish the two stories." This quantifies it: **bf16 flips the
predicted verdict on two of the five devices.** A dtype flag is not a tuning
knob, it is a move on the axis the thesis is about.

### 5.4 The 256× is cross-scope, and the honest ratio is 64×

`[MODELLED]`, exact arithmetic, zero measurement:

| | per (chunk, head) | whole operator |
|---|---|---|
| port | `C²K` = 262,144 | 16,777,216 |
| kernel, all NC=4 sub-chunks | `NC·BC·K = C·K` = 4,096 | 262,144 |
| kernel, **one** sub-chunk | `BC·K` = 1,024 | 65,536 |

- same scope, both sides over a whole chunk: **64×** (= `C`)
- kernel side counted per **sub-chunk**: **256×** (= `C²/BC`)

**256 is the study's headline, and it is obtained by comparing the port's whole
chunk against the kernel's one of four sub-chunks — a cross-scope comparison.**
`notes/06` C-03 correctly records that 256× is a slab ratio and not a byte or
time ratio, and correctly records that "256×" has two incompatible derivations in
the record. What it does not record is that the ratio *itself* compares two
different scopes, so 256 is not "the slab ratio" — it is the slab ratio with a
factor of `NC` thrown away. **At the same scope it is 64×, and 64× is what
`notes/03-kda.md` already said** ("16,777,216 / 262,144 = 64"). The 256 is
therefore not a new finding and should not be quoted as one.

`BC=16`, `NC=4` are `[READ-FROM-SOURCE]` from `fla/ops/kda/chunk_intra.py` and
have not been executed — no GPU, no Triton in this environment.

### 5.5 Two arithmetic-intensity figures in `notes/06` do not reproduce

All ten **byte** figures in `notes/06` §2/§3 reproduce exactly (second
independent derivation). The two **arithmetic-intensity** figures do not:

| | ported | folded |
|---|---|---|
| `notes/06` §3, delivered | 0.8079 | 10.5412 |
| re-derived from `kda/cost.py`'s step table | **0.9802** | **12.7882** |

`[MODELLED]`. Solving for the FLOP count the delivered pair implies gives
234.87 MFLOP against 284.95 MFLOP from the step table, a factor of 1.213. At
this shape `K == V`, so every matmul term is the same size (524,288 FLOPs per
chunk-head) and the counts are **8.4922 units vs 6.9997 units** — the delivered
count is short by one full matmul term *and* the 0.4922-unit UT substitution.

**I can state the size of the gap to four decimal places and I cannot say which
matmul was dropped** — the step table lists ten terms and eight are
indistinguishable at `K == V`, so the number does not identify the omission. I am
not claiming to have found the cause.

**Consequence for an open prediction.** `notes/06` P-3 reads "The KDA port is
~31× memory bound in fp32 at this shape (0.81 FLOP/byte against a 25.3 FLOP/byte
T4 ridge)" and lists as its failure condition "The FLOP count is wrong by a
large factor… it invalidates the denominator every other KDA claim uses. Must be
settled before any KDA ratio is published."

- P-3's **conclusion is robust**: memory-bound, by 31× or 26× depending on which
  FLOP count you take. It does not need settling.
- The **denominator does**, and the two candidates differ by 1.21×.
- P-3's stated remedy — recheck the FLOP count — is the right move and has **not
  been done**. This pass is a partial version of it and it does not converge.
- The measurement P-3 actually wants (`sm__throughput` vs `dram__throughput`,
  fp32, T4) is untouched and still outstanding.

---

## 6. Which prior results the thesis explains, and which it contradicts

`[MODELLED]` throughout, from `crossover/literature.py`.

| result | verdict | why |
|---|---|---|
| **Toulme / Splash, 37.6× byte reduction** (re-derives: 526/14 = 37.57) | **EXPLAINS** | measured bytes, same algorithm, the 128 MB attention matrix is what goes away. The corpus's best case, because the byte counts were *measured* rather than modelled |
| JAXBench Flash Attention 3.91×, Ragged Paged 6.91× | **EXPLAINS** | the attention matrix is the thing being removed |
| JAXBench GEMM 0.96× | **EXPLAINS, weakly** | a GEMM output tile fits in L2 everywhere, so M1 says "compiler wins" and JAXBench says 0.96×. But plain arithmetic intensity explains it too, with no capacity in it — this data does not discriminate between the thesis and its boring competitor |
| **JAXBench MLA 0.94×, Sparse 0.86×, GEMM 0.96×** | **DOES NOT EXPLAIN** | three of eight rows are sub-1×. A capacity threshold has nothing to say about a kernel that is 6% slower when everything fits, because its variable is not a variable there. **A step function with a 3/8 bad tail is not a step function** — and the headline geomean is *floored at 1×*, so the statistic cannot report that tail at all (§6.3) |
| **arXiv 2012.07145, 0.95×** | **CANNOT ADJUDICATE** | see §6.2 |
| **Zhang / TPU, 35× at n=4096** | **CONTRADICTS** | never overtakes; the 35× is the peak of a hump; MXU utilisation 20.1% vs 94.3% (Zhang's own tick-level emulator, quoted at `research/TPU.md` §5c) explains it instead. 20.1/94.3 = 0.213, and 1/0.213 = 4.7 — the same order as the 5.8× fall in the penalty from n=4096 to n=32768. SystolicAttention (arXiv:2507.11331) puts the same 4.83× from an architecture paper |
| **2607.04454, 380× / 1347× LayerNorm** | **CONTRADICTS** | see below — the best counter-case in the corpus |
| this study's KDA **8.00 GiB** | **EXPLAINS** | 8 GiB is past every L2 in the study by 65× to 2048× |
| this study's KDA at the **benchmarked** shape | **UNDERMINES** | 64 MiB is 16× over a T4's L2 and *under* a B200's |
| this study's **256×** | **UNDERMINES** | cross-scope; 64× same-scope, 48.8× byte |
| this study's **0.8079 / 10.5412 FLOP/byte** | **UNDERMINES** | 0.9802 / 12.7882; the discrepancy is in the FLOP count, and the ledger's own remedy would not find it |
| **JAXBench's 2.08× convention** | **UNDERMINES** | it is a *floored* geomean: 2.0803 with the three sub-1× rows clipped to 1.0, vs 2.0154 raw. The flagship statistic cannot report a human loss |

### 6.1 The best counter-case: the LayerNorm cliffs

A fused LayerNorm materialises its **row statistics** — row mean, row variance,
reciprocal σ. All `[B,T]` or `[B,T,1]`. They do **not** scale with the hidden
dimension `H`; they scale with `B·T`. `[MODELLED]`:

- at `B=32, T=4096`: 512 KiB, which is **8× inside a T4's 4 MiB L2 and 252×
  inside a B200's 126 MiB**
- to push `[B,T]` past even the largest L2 in the study you need `B·T > 33 M`,
  e.g. `B=64, T=524288` — a half-million-token sequence, an order of magnitude
  past anything anyone trains or serves

**So the ratio is not a free parameter: for LayerNorm it is pinned to "fits", by
a wide and growing margin, at every reachable shape. And the cliff there is 380×
on an A100, 1347× untuned, 1541× on a GH200.** Under the thesis' own rule that
cell reads "everything fits, so the compiler wins, ratio ≈ 1". The measurement
says 380. The intermediate cannot be made bigger to rescue it.

**And this is the part that really hurts: the corpus contains 380× *and* 1347×
*and* a 0.95× geomean, for the same operator on the same device.** A capacity
threshold assigns one position on the `s/C` axis to (LayerNorm, A100) and
therefore one predicted ratio. The corpus reports three that differ by three
orders of magnitude for that one cell — and `s` does not even change between
them, because all three run the same program. **No function of `s` and capacity
can produce 380, 1347 and 1.06 from one input.**

What *does* produce all three is a scheduling defect: 1347× untuned
(`T.serial` not converted to `T.reduce`), 380× tuned (still not converted), 0.95×
the autoscheduler (which converts it). The intermediate size is identical in all
three. **It is not a variable, so it cannot be the explanandum.**

### 6.2 Why 0.95× cannot adjudicate anything

`[MODELLED]` from the Table 1 figures quoted at `research/PARITY-SCOPE.md:126-131`
— both of PARITY-SCOPE's ratios re-derive (16.82×, 72.33×).

0.95× is a **geomean over 17 imaging programs on one V100**. A geomean of a step
function is not a step function; it is a smooth functional of the distribution of
`s/C` across the 17 programs, and those intermediate sizes appear nowhere in the
literature. **The thesis cannot explain 0.95× in the strong sense ("these programs
all had intermediates that fit, so the machine won, as predicted").** It can only
restate it. This is a *worse* version of the objection `PARITY-SCOPE.md` §1.2
already makes: that document's complaint is that the suite lacks the right
operators; mine is that **the suite does not report the intermediate sizes the
thesis' explanatory variable is made of.** The thesis is unfalsifiable on the one
result it most wants to explain.

**A finding about JAXBench's headline number.** All 8 per-kernel ratios re-derive
from the reported timings, but the raw geomean of the 8 is **2.0154**, not 2.08.
The table header says "geomean (**floor 1×**)", and flooring the three sub-1×
rows at 1.0 gives **2.0803** — a 0.01% match `[MODELLED]`. So the published 2.08×
is fully reproducible, and the convention is exactly what the header claims.

**Which makes it a stronger objection to the brief, not a weaker one.** The
flagship number is a *floored* geomean: a statistic that **cannot report "the
human lost"**. Three of eight rows — MLA 0.94, Sparse 0.86, GEMM 0.96 — are
cases where the expert Pallas kernel was slower, and the headline statistic is
constructed so that those cases contribute exactly 1.0. The brief cites 2.08× as
evidence for a bimodal, step-like distribution of compiler-vs-human outcomes; the
number is a summary in which one entire half of the distribution has been clipped
away. Anyone using it as support for a step function is leaning on a statistic
engineered to be smooth.

### 6.3 The flagship geomean is a floored geomean

| | value |
|---|---|
| raw geomean of the 8 reported ratios | **2.0154** |
| geomean with the 3 sub-1× rows floored at 1.0 | **2.0803** |
| published "geomean (floor 1×)" | **2.08** |

`[MODELLED]`. A 0.01% match. The convention is exactly what the table header
claims, and I was wrong to doubt it — recording that because the doubt was itself
a re-derivation failure in the other direction.

The consequence is the interesting part. **A floored geomean is a statistic that
cannot report "the human lost."** Three of eight rows are losses; in the headline
number each contributes exactly 1.0. So the corpus's flagship evidence for a
bimodal, step-like distribution of compiler-vs-human outcomes is summarised by a
statistic with one entire half of the distribution clipped away. A step function
with a 3/8 bad tail, summarised by a metric that cannot see the tail, is
indistinguishable from a clean step — which is precisely why the brief reads
2.08× the way it does.

---

## 7. The right claim

The thesis as given does not survive. This does:

> **W (the defensible residue).** For a fixed operator and a fixed pair of
> implementations, the ratio of the materialising form to the non-materialising
> one is dominated by one quantity: the bytes that must cross the DRAM boundary.
> That quantity is `O(n²)` for *both* forms, so the ratio is **scale-invariant
> in `n`** and is set by `(passes, Bc, d)` — where `Bc`, the key-block width, is
> what on-chip capacity buys. Enlarging the intermediate enlarges both forms'
> traffic proportionally and moves the ratio not at all.
>
> Consequently:
> - the crossover is **not** at an intermediate size;
> - the human's advantage has a **ceiling** — 6× on a T4, 21× on an H100 or B200
>   at d=64 fp32 — and no `n` exceeds it;
> - **capacity helps the human, not the compiler**; the sign is the reverse of
>   the brief's;
> - the relevant capacity is **per-SM shared memory**, not device-wide L2, and on
>   a CUDA kernel the device total is the wrong denominator;
> - the transition is a **knee** in a continuous function, measured as such on
>   real hardware today;
> - and far from the knee — everything comfortably resident — the byte term is
>   ~0 and what remains is code quality, which is where 380× and 1347× live.
>
> The crossover is the point where the byte term and the quality term are equal.
> That is a **crossing of two smooth curves**, not a step.

**The heretical residue, and it is the part worth keeping:** the study's
qualitative intuition — that on-chip residency is what decides these fights — is
supported, and supported more strongly than the brief argues, because the
*gradient* intuition fails while the *magnitude* intuition holds. Near the knee
the byte term dominates everything and the ratio is 10–100×; far from it, the
byte term is ~0 and the ratio is 1.0–1.2. **The crossover is not a quality
gradient; it is a byte-count gradient. But it is a gradient in the *constant*,
not a step in the *problem size*, and its sign runs the other way.**

### 7.1 The pre-registered refutation criteria, written before any data

`crossover/claim.py` §4, printed by every run. In brief:

| | criterion | STEP supported if | refuted if |
|---|---|---|---|
| **R-A** | the step (tests P3) | R(n) flat within 1.3× over a decade below the predicted crossover, rising monotonically through it, ≥2× within one doubling past | R(n) already varies >2× across a decade **entirely on one side** of the predicted crossover |
| **R-B** | which capacity (tests P2) | — | M1 if the H100↔B200 crossover moves by ~1.59× in `n`; M2/M4 if it does not move at all |
| **R-C** | the natural-lowering premise | `dram__bytes.sum` within 1.3× of modelled materialised traffic | the compiler fuses it away at every `n`, and the "compiler side" does not exist |
| **R-D** | the control is a control | — | the two implementations disagree, or the online form depends on `Bc`. **Runs today, and passes** |
| **R-E** | the pass count | — | any predicted ratio quoted without a measured pass count. **Currently unmet: every ratio in this document is linear in it** |

**A gradual transition is a refutation of R-A, and it is the expected outcome.**
It is written down now so the finding cannot be re-described as a step later.

---

## 8. The confound, stated as a limitation of this document and not of a footnote

`crossover/attn_control.py` prints this in its output, and it belongs here too.

**The materialising implementation and the avoiding implementation are not the
same program.** They cannot be: the entire content of the experiment is that one
of them does not build the tensor the other builds. A difference in time between
them is a difference between **implementations**, and nothing in this study
establishes that the materialising one is what a compiler produces from a
reasonable source program. That is not a caveat, it is the load-bearing premise,
it is unmeasured, and it is criterion R-C.

Concretely, what else differs that nothing here controls:

- the materialising form allocates a large array, so it pays allocator,
  page-fault and cache-pollution costs;
- the online form's inner loop is `n/Bc` separate calls, so its time includes
  `n/Bc` dispatch overhead a real fused kernel would not pay;
- the online form's rescaling of the accumulator is real arithmetic the
  materialising form does not do — **counted, not assumed**: 0.8% of the total at
  d=64 (`crossover/attn_control.py` §2), rising as `1/d`;
- BLAS chooses its own blocking for the materialising form's matmuls, so the
  "materialised" version may already be partly tiled and on-chip.

**Every one of those pushes the two closer together than the byte counts say they
should be.** So a flat R(n) on any device, including a GPU, is weak evidence for
the thesis' underlying intuition and strong evidence against its strong form.
The CPU run gives exactly that: **R = 1.15, 1.08, 1.04, 1.01, 1.15** at
n = 256…4096 `[MEASURED]` — flat to within 1.15× while the modelled byte ratio
moves 1.25× → 1.47×.

**What the comparison supports:** "at this `n`, on this device, the two
implementations cost R, the modelled byte ratio is B, and R tracks B to within
X." **What it does not support, and what nobody in this study has shown:**
"therefore the compiler/human crossover is a step function of s/capacity." That
last step needs a compiler-produced materialising implementation, a capacity
measured on the device rather than read off a datasheet, and an `eff_ratio`.
**Two of those three are unavailable today and the third has never been measured
by anyone.**

---

## 9. What I ran, and what I did not

**Ran, today, on this machine** `[MEASURED]`:

- `crossover/test_control.py` — **64/64 checks pass.** Shapes, byte accounting,
  reference semantics (materialised vs online agree to 1.35e-15 across 3 shapes ×
  causal/dense), block-size invariance (1.39e-15), all ten delivered KDA byte
  figures, the capacity table's arithmetic, the P100 L2 discrepancy, and every
  literature re-derivation.
- `crossover/attn_control.py` — the control, at n = 256…4096.
- `crossover/residency_probe.py` — the knee, on an Apple M1 Pro.

**Did not run, and no number in this document pretends otherwise:**

- **Every GPU timing.** No NVIDIA GPU, no CUDA, no ROCm. `torch` is absent from
  the venv, so no torch-derived figure could be produced even accidentally.
  `tinygrad`'s default device here is **METAL** — real hardware, but not one of
  the five devices in `crossover/devices.py`, and not a CUDA device.
- **Every TPU timing.** `import jax` → `ModuleNotFoundError` in the repo venv
  (verified). The code in `kda/tpu/` is written but not executed, and the 35×,
  the VMEM bracket and every TPU number in §2 are arithmetic on a table
  transcribed into `research/TPU.md` from a blog post I did not re-read.
  **These are `[UNVERIFIED]` at one remove and must be labelled as such
  wherever they are cited.**
- **Criterion R-C** — whether a compiler emits the materialising form at all.
- **Criterion R-E** — the pass count.
- **`eff_ratio`** — how well each side drives memory.

**Two things I got wrong while doing this, recorded because the study has a
documented pattern of exactly that:**

1. The byte model in my first version of `crossover/literature.py` charged I/O
   as `2K + 2V` per token per head, omitting that `q`, `k` and `g` are three
   separate K-wide tensors. That made the compulsory figure 17.0 MB instead of
   5.5 and the fold ratio 10.1× instead of 13.0 — it would have read as a
   refutation of the study's own number. **It is the fourth time this study has
   produced an apparent refutation that was a bug in the checker** (`notes/06`
   §3 records three). The pattern now has a count of four.
2. `crossover/residency_probe.py`'s first version used `a.sum()` as the
   bandwidth kernel. It measured a flat 23 GB/s at *every* size from 64 KiB to
   2 GiB and reported "no knee found" — which reads as a null result about cache
   hierarchies and is in fact a null result about numpy's float32 pairwise
   summation, which is reduction-bound at ~23 GB/s. Switching to `np.vdot` (BLAS,
   memory-bound) shows the knee immediately: 259 → 55 GB/s. **A null result from
   a probe that cannot resolve the effect is not a null result.**

---

## 10. The single most efficient measurement that would kill the thesis

Not R-A. R-A needs a sweep on a GPU, and the outcome is a gradient either way.

**Measure `dram__bytes.sum` for one attention call at a fixed `n`, with and
without the materialised intermediate, on an H100 and on a B200, and compare the
two byte counts to the per-SM shared-memory capacity.** One `ncu` invocation
twice, on two cards, about twenty minutes.

It kills the thesis if, as `crossover/attn_control.py` §2's byte accounting
predicts, **the two forms' DRAM traffic ratio comes out the same on the two
cards** (≈21×, set by 228 KiB of smem per SM, identical on both) rather than
tracking their 2.52× L2 difference. That single number decides between:

- **M1 (device L2)** — the ratio should differ by ~2.52× between H100 and B200,
  because their L2 differs by 2.52× and nothing else that matters does;
- **M2/M4 (per-SM on-chip)** — the ratio should be *identical*, because smem/SM
  is 228 KiB on both.

The two hypotheses differ by a factor of 2.52 in opposite directions, on two
cards anyone can rent, with a metric that needs one counter. **It is the
cheapest available kill, and if it lands the way the accounting says it will, the
thesis' named variable is the wrong one and the study should say so before
spending a TPU session on it.**

The runner-up, if only one card is available: measure `dram__bytes.sum` for the
KDA port at the **benchmarked** shape on a B200. The prediction is ~0.51× of L2
— 64 MiB against 126 MiB — so the decay tensor should be *absorbed*, and the
predicted 13× fold should collapse. That is `notes/06` P-2, and this document
sharpens its refutation threshold from "< 58 MB" to a per-device table.

---

## 11. Files

| file | role | runs on CPU today |
|---|---|---|
| `crossover/devices.py` | the capacity table, one source per field, plus `CONFLICTS` and `UNVERIFIED` | **yes** |
| `crossover/claim.py` | the predictions: 4 crossover models, per-device KDA thresholds, and the pre-registered refutation criteria | **yes** |
| `crossover/literature.py` | every prior-work number re-derived, and the verdict table | **yes** |
| `crossover/attn_control.py` | **the control**: materialised vs online, same arithmetic; the confound statement; the sweep | **yes** |
| `crossover/residency_probe.py` | step or knee, measured on local hardware | **yes** |
| `crossover/test_control.py` | 64 assertions; the suite that fails when a number moves | **yes** |
| `crossover/README.md` | how to run it, and what each file is not | — |

**Relationship to the other notes.** `notes/06` is the register of record for
corrections; the five findings here that belong in it are the P100 L2 value
(§4), the cross-scope 256× (§5.4), the intensity discrepancy and its effect on
P-3 (§5.5), the "overtakes at n≥8192" misreading (§2), and the shape ambiguity
behind "8.00 GiB" (§5.1). They are **not** written into `notes/06` — that file is
owned by another agent — and `crossover/test_control.py` asserts each of them so
that they cannot be quietly dropped.
