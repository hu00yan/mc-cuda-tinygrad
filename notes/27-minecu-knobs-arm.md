---
kind: reference
---

# 27 — arm B: the same DDA, plus the knobs TinyGrad actually exposes

> **⚠️ CITATION STATUS — read before following any path below.**
> The `.agent-scratch/` tree referenced in this note was **deleted from disk**
> (288 files, entire directory). Every `.agent-scratch/...` path in this file is
> currently broken. The content is **recoverable** — it is all in commit
> `9e6767f` — so restore with:
> ```
> git checkout 9e6767f -- .agent-scratch/
> ```
> Until that is run, treat the paths below as *pointers to committed history*,
> not as live files. The measured values quoted in this note are unaffected;
> only their on-disk evidence is missing.


`MEASURED`. The codegen census and the SASS counts are from this host with an
`sm_75` target and nothing executed; the timings are from one Tesla T4, one
session, 56 timed configurations. Every number below is measured here, modelled
here on measured inputs, or read out of another file with a line cite — and the
kind is printed with it. **A passing correctness gate is not performance
evidence** and is not treated as such anywhere below.

**The one sentence this note is for**, and the answer to the question it was
written to settle, is at the end of §7.

Claim citations are `minecu_tg_knobs/knobs_table.py:<claim>`; §8 has the command
list. Artefacts: `out/minecu_tg_knobs/armB/`.

---

## 0. The answer

**Giving the compiler its own knobs does not close the gap. It moves nothing
except making one thing worse, and the thing it moves is not on the hot path.**

Seven of the eight knobs on the brief's list are sweepable. Of those seven, six
emit **byte-identical CUDA** — verified by sha256 per kernel, not by reading a
knob's `.value` — and the seventh, `TRANSCENDENTAL=2`, makes the DDA kernel
**3.8% larger** (25 464 → 26 432 SASS). The eighth, `OPT_LEVEL`, does not exist
in the installed TinyGrad.

`MEASURED` on a T4, 56 timed configurations, one session: **all thirteen knob
ratios on the lean port lie inside the control's own 4.66% spread**, from 0.982
to 1.025. The one knob that reached the scheduler, `NOLOCALS`, set
`__launch_bounds__` from 128 to **1** on every kernel in the graph and cost
**x0.992** — nothing, because this kernel is issue-bound rather than
latency-bound and has no idle warp for a wider block to fill.

What the sweep *does* rule out is a specific and otherwise-open possibility: the
Tensor-core and texture knobs are inert here for reasons **provable from the
guard conditions**, not merely unobserved. §3 separates three different reasons
for inertness, and they land in three different classes of the study's taxonomy.

The larger result is the port ablation, which the brief's own first instruction
forced. `minecu_gap.py:ablation_target` recorded a 9.73× residue and said the
ablation had never been run. Run (§4, §5.2), it recovers **1.429×** in
instructions and **1.574×** in wall clock: the port's index arithmetic was a real
defect and a material one, and it was **not** the remaining story. What remains,
795.8 SASS per step against the reference's 48.5, is the lowering.

**And arm B's absolute standing against the hand-written kernel is unknown.** The
reference's timing was measured in a different session on a shared free-tier T4,
and this note refuses to divide across the two runs (§5.3).

---

## 1. State of the base port, and why arm B is not arm A's numbers

The brief says: build arm B on a good port if one exists; otherwise say so,
build on the best available, and state that the absolute ratios are not
comparable to arm A's.

**It does not exist yet, as of this note.** Two facts, both checked:

- `notes/23-good-vs-naive.md` does not exist in this repo.
- `minecu_tg_opt/` is present and actively being written, and
  `.agent-scratch/minecu_tg_opt_verify.log` records its own verification run with
  **`[FAIL]` on four of its five V-lean checks**, including
  `503/1024 = 49.12%` per-pixel step-count agreement and a mean of 20.672
  against a float64 reference's 16.861. Its `FAIL` list also includes
  `count_steps=True does not change the observation`, which means its verified
  and timed programs are not the same program. Those numbers are another arm's
  in-flight work, read here only to establish that no verified good port was
  available, and they are **not** used as a result.

So I built the port: `minecu_tg_knobs/lean.py`. It is ordinary TinyGrad — no
`custom_kernel`, no PTX, no `Context(...)` steering the scheduler, no hand
`.cu`, no hand-unrolling, no tile size. It differs from `minecu_tg/render.py` in
exactly three things, each a fix to the port rather than a schedule choice, each
separately switchable so its cost is measured rather than asserted:

| fix | the naive port | `lean.py` | reference |
|---|---|---|---|
| cursor type | `float`, cast + clip every step | `int32`, once in the prologue | `(int)floorf(eye_x)`, `kernels.cu:638-640` |
| gather address | recomputed from 3 float coords per step | one accumulator, advanced by a constant stride | `kernels.cu:669,673-677` |
| bounds test | 6 signed compares | 1 uint32 cast + 1 unsigned compare per axis | `(unsigned)v < (unsigned)w`, `kernels.cu:666-668` |

What it deliberately does **not** fix: there is still no per-ray early exit, and
the sky is still a table row rather than a branch. Those are properties of the
array formulation, not of the writing, and `notes/19` §4.2c and §6 gap 12 already
price and classify them.

**`MEASURED` `minecu_tg_knobs/verify.py`, CPU, float64 contract.** 45 checks, 0
failures. All five variants (`lean`, and each of the three fixes switched off,
including one that reproduces the naive port's index arithmetic exactly) agree
with the independent float64 transcription on **the uint8 image exactly and on
1024/1024 per-pixel DDA step counts**, across worlds 16/32/48, two pitches, and
`max_steps` 32 and 16. The step count is the load-bearing one: the image is
lossy, and `minecu_tg_opt`'s log shows what a passing image with failing step
counts looks like.

Two bugs this gate caught, both of which would have produced a confident wrong
answer:

- A single `dv = take_x.where(sx, take_y.where(sy, sz))` added to all three
  cursors **moves y and z whenever x moved**. The image still looked plausible;
  the step count pinned at `max_steps` and the gate caught it. The reference
  advances exactly one axis per step.
- The two outputs have **different ranks** — `obs` is `[B,H,W,3]`, `steps` is
  `[B,1,H,W]` — and indexing both `[0,0]` compares `(32,3)` against `(32,32,3)`,
  which broadcasts, reports ~500 of 1024 pixels differing, and does not raise.
  The gate now refuses any shape mismatch before comparing.

**Consequence, stated plainly: no ratio in this note may be divided by an arm-A
ratio.** The two ports are different objects. The knob comparisons *within* this
note are exact, because every one of them is the same file under two knob
settings.

---

## 2. The knobs, and where TinyGrad reads them

`MEASURED`, from the installed package. `python -m minecu_tg_knobs.registry`
re-reads every cited line out of `.venv/.../tinygrad/` and fails if one has
drifted, so the `file:line`s below are checked rather than transcribed.

| knob (brief's name) | `Context`/env key | default | defined at | read at | reachable for this kernel |
|---|---|---|---|---|---|
| `USE_TC` | **`TC`** | 1 | `helpers.py:238` | `codegen/opt/heuristic.py:28,32` | no |
| `TC_SELECT` | `TC_SELECT` | −1 | `helpers.py:238` | `codegen/opt/heuristic.py:32` | no |
| `TC_OPT` | `TC_OPT` | 0 | `helpers.py:238` | `codegen/opt/heuristic.py:28`; `codegen/opt/search.py:23` | no |
| `ALLOW_TF32` | `ALLOW_TF32` | 0 | `helpers.py:281` | `codegen/opt/postrange.py:226` | no |
| `NOLOCALS` | `NOLOCALS` | 0 | `helpers.py:239` | `codegen/opt/heuristic.py:81,163-164` | **yes** |
| `IMAGE` | `IMAGE` | 0 | `helpers.py:233` | `mixin/op.py:386,1593`; `codegen/opt/heuristic.py:47-58`; `renderer/cstyle.py:602` | no |
| `TRANSCENDENTAL` | `TRANSCENDENTAL` | 1 | `helpers.py:239` | `codegen/__init__.py:372` | **yes** |
| `OPT_LEVEL` | — | — | **absent** | — | — |

### Two things about this table that are not the knob names

**`USE_TC` cannot be set under the name on the brief's list.** `helpers.py:238`
is

```python
USE_TC, TC_SELECT, TC_OPT = ContextVar("TC", 1), ContextVar("TC_SELECT", -1), ContextVar("TC_OPT", 0)
```

`ContextVar.__init__` registers under its **first argument** (`helpers.py:179-182`),
and `Context.__enter__` looks each key up in `ContextVar._cache`
(`helpers.py:169-171`). So the registered key is the string `"TC"`. Measured:
`Context(USE_TC=0)` raises `KeyError: 'USE_TC'`; `Context(TC=0)` sets
`helpers.USE_TC.value` to 0.

This matters beyond tidiness, and the reason is worth recording: a sweep that
wrote `Context(USE_TC=0)` and caught the exception per configuration would
record **`USE_TC: no effect`** — a true-sounding sentence about a knob that was
never set. `registry.check_context_keys` re-verifies the spelling at run time so
it cannot regress silently.

**`OPT_LEVEL` does not exist in this TinyGrad.** Not a `ContextVar`:
`Context(OPT_LEVEL=0)` raises `KeyError`, and there is no definition line in
`helpers.py` to find. The nearest real knobs are `NOOPT` and `BEAM`
(`helpers.py:232`) and **neither is on the brief's list**, so substituting one
would answer a different question. Recorded as an absence; a row reading "no
effect" would be a null result about a knob that was never set — the same
category error as the naming trap above.

### The gate that makes a null result mean something

Tinygrad keys its compiled-program cache on `to_program_config`
(`codegen/__init__.py:493-494`). A knob **outside** that tuple returns the
*default's* compiled kernel, and the sweep would report the default again with
the knob's name on it. All seven sweepable knobs are inside it — verified, not
assumed, by `registry.check_cache_key_coverage`, which refuses to start
otherwise. That is what licenses reading "byte-identical" as *set, consulted,
and had nothing to do* rather than as *not set*.

---

## 3. The knob sweep

`MEASURED`. Codegen on the `NULL:CUDA:sm_75` device — the **real CUDA renderer
on a real sm_75 target**, nothing executed (`codegen.py`) — and the SASS from
`nvcc 12.9 -arch=sm_75 --use_fast_math` cubins in the `archlab` container,
counted through the repo's own `tools/archlab/inspect_ptx.py:CATEGORIES`, which
is **imported, not redeclared**. Claim: `knobs_table.py:knob_effects`.

| knob | value | reachable | codegen change | lean SASS | Δ | verdict |
|---|---|---|---|---|---|---|
| *control (naive)* | — | — | 8 kernels, 15 859 src lines | 36 384 | — | — |
| *control (lean)* | — | — | 3 kernels, 12 313 src lines | 25 464 | — | — |
| `USE_TC` | 0 | no | **byte-identical** | 25 464 | 0 | INERT |
| `TC_SELECT` | 0 | no | **byte-identical** | 25 464 | 0 | INERT |
| `TC_SELECT` | 2 | no | **byte-identical** | 25 464 | 0 | INERT |
| `TC_OPT` | 1 | no | **byte-identical** | 25 464 | 0 | INERT |
| `TC_OPT` | 2 | no | **byte-identical** | 25 464 | 0 | INERT |
| `ALLOW_TF32` | 1 | no | **byte-identical** | 25 464 | 0 | INERT |
| `NOLOCALS` | 1 | **yes** | `__launch_bounds__` **128 → 1** | 25 472 | +8 | CAUSAL on count, **BLIND on time** |
| `IMAGE` | 1 | no | **byte-identical** | 25 464 | 0 | INERT |
| `TRANSCENDENTAL` | 2 | **yes** | `1/sqrt` → 28-line software `pow` | 26 432 | **+968** | CAUSAL, and it **hurts** |
| `OPT_LEVEL` | — | — | — | — | — | **DOES NOT EXIST** |

"byte-identical" is a **sha256 over the emitted CUDA of the DDA kernel**, on
both ports and in the same run as the control. A knob that changes nothing in
either column is as much a result as one that changes everything, and the sha is
what makes it one.

### 3.1 The three kinds of inert, which are three different classes

This is the part of the sweep that does real work, because "the knob did
nothing" is one sentence covering three mechanisms that belong in three
different rows of the study's taxonomy.

**`USE_TC` / `TC_SELECT` / `TC_OPT` — inert because the guard is false.** Their
only reads are inside

```python
if USE_TC > 0 and (len(k.axes_of(AxisType.GROUP_REDUCE, AxisType.REDUCE)) == 1 or (TC_OPT.value >= 1)):
```

at `codegen/opt/heuristic.py:28`. The DDA kernel has **no reduction axis at
all**: every output element is a pure function of one output index, so the
`axes_of(REDUCE)` term is empty and the guard is false. `TC_OPT` is the sharp
case, because `TC_OPT>=1` is a disjunct that can make the guard true *without* a
reduce axis — and sweeping it at 1 and 2 gives byte-identical source. That is a
**measurement**, not an inference, and it closes the reading that `TC_OPT` could
reach this kernel through that disjunct: the guard may open, but the body it
guards calls `OptOps.TC`, which needs reduce axes to match, so every axis raises
`KernelOptError` and the loop continues.

*Class:* **(i)** for this workload. Not a compiler defect — a GEMM knob and a
non-GEMM kernel. *Falsifier:* a source change, or the same byte-identical source
on a kernel that *does* reduce.

**`ALLOW_TF32` — inert partly because of the architecture.** Its only consumer
is one line:

```python
if self.ren.target.device in ("CUDA", "NV") and tc.dtype_in == dtypes.float and not ALLOW_TF32: continue
```

`codegen/opt/postrange.py:226`. It filters *tensor cores with fp32 input* on
CUDA. And a T4 is sm_75, where `get_cuda` returns `cuda_sm75 = cuda_8168_f16`
(`codegen/opt/tc.py:94,98`) — **half-input tensor cores only**. There is no fp32
tensor core on this architecture for the knob to admit, so the line has nothing
to filter even for a matmul.

*Class:* **(iv)**, not (i) and not (ii). This is the one knob whose answer is an
architecture fact rather than a workload fact, and it is worth separating
because on sm_80 — where `cuda_8168_tf32` exists (`tc.py:87-88`) — the same
sweep would be a different experiment. *Falsifier:* the same sweep on an A100
producing different source, or an fp32 tensor core appearing in sm_75's table.

**`IMAGE` — inert because of the shape.** It redirects `dot` and `conv2d` to
texture-memory forms (`mixin/op.py:386,1593`), 4-wide vectorises image-shaped
buffers (`codegen/opt/heuristic.py:47-58`), and enables fp16 storage paths
(`renderer/cstyle.py:602`). The DDA has no `dot`, no `conv2d`, and gathers from
a **3-D** `[B,WY,WX,WZ]` volume, which `image_valid_dims` does not accept. The
capability exists and this kernel is not one it covers. *Class:* neither (i) —
the DDA is expressed — nor a defect; a workload-shape fact. *Falsifier:* the
same knob on a 2-D image-shaped gather producing different source.

### 3.2 `TRANSCENDENTAL=2` works exactly as documented and buys nothing

This is the one knob where the obvious reading was wrong, and the way it was
wrong is checkable.

The obvious reading: "the DDA calls no transcendental, so this cannot matter."
That is false. `rsqrt()` lowers to **SQRT + RECIPROCAL** — measured, the op set
of `Tensor([4.,9.]).rsqrt()` is `{SQRT, RECIPROCAL}` — and the DDA prologue
normalises every ray with it. `transcendental.py:275-276` rewrites `SQRT` to a
software `pow` when the flag is ≥ 2. So the knob *does* reach the DDA kernel.

`MEASURED`: it does, and it costs. On **both** ports, the emitted CUDA replaces

```c
float alu16 = (1/sqrt(((alu12*alu12)+alu15+(alu14*alu14))));
```

with a ~28-line software `pow` — `tg_bitcast` exponent extraction, a degree-7
polynomial, special-case handling for 0/±inf/NaN — and the kernel grows by ~450
source lines and **968 SASS** (`knobs_table.py:knob_effects`).

The prediction was right about the mechanism and the knob is still **worse than
useless** here: the DDA needs one square root per ray in the **prologue**, not on
the hot path, and this knob only makes that one more expensive. Reported as
prominently as a win because that is what it is.

*Falsifier:* the image changing (a last-ulp difference in `ray_len` feeds every
`t_max`, and a DDA step is chosen by `t_max_x < t_max_y`, so it *could* move a
ray at a tie) — or the count not following the source change, which would mean
ptxas folded the software `pow` back.

### 3.3 `NOLOCALS=1` — the row this arm cannot finish

`NOLOCALS` is the only knob whose read is reachable and whose effect is not
prologue-only. It has two effects (`heuristic.py:81,163-164`): the upcast ceiling
drops 2048 → 240 elements, and `Opt(OptOps.NOLOCALS)` sets `dont_use_locals`
(`postrange.py:122-126`) — which speaks directly to the 7 460 B of spill
`notes/19` §4.2b records.

`MEASURED`, and this is the finding:

| | default | `NOLOCALS=1` |
|---|---|---|
| DDA kernel SASS | 25 464 | 25 472 (**+8**) |
| `__launch_bounds__` on all 3 kernels | 128 | **1** |
| spill stores (lean) | 1 700 B | 1 708 B |
| registers (lean) | 168 | 168 |

The knob did **not** reduce the spill — it moved it by 8 bytes — and it did not
meaningfully change the instruction count. What it did was set every block in
the graph to **one thread**.

So the verdict is neither "no effect" nor "a performance win": it is
**causally connected to the code, and this instrument is structurally blind to
the consequence.** A one-thread block does not change what a thread issues; it
changes occupancy, and occupancy is a memory-system property that no static
instruction census can see. **"No measurable effect" is not the same claim as
"no effect"**, and for `NOLOCALS` the honest statement is that the effect, if
any, lives in a dimension this table does not measure. That is §5's job.

*Falsifier:* a wall-clock move with the source otherwise unchanged. The
instruction census cannot supply one, by construction.

---

## 4. The port ablation, which is the larger result

`MEASURED`. `knobs_table.py:port_ablation` and `:sgd_audit`.

The brief's first instruction was *don't sweep knobs over the naive port*, and
following it forced a port ablation that this study had marked as unrun.
`derivation/minecu_gap.py:ablation_target` is `[PREDICTED]`, says "the ablation
has not been run", and sets a falsifiable target: removing 15 of the naive port's
53 per-step source operations should leave the generated side near 38/53 of its
per-step cost.

| port | kernels | DDA-kernel SASS | per step | LDG | LDL (spill traffic) | spill stores | regs |
|---|---|---|---|---|---|---|---|
| naive (arm A) | 8 | 36 384 | 1 137.0 | 517 | 5 231 | 7 460 | 168 |
| **lean (arm B)** | 3 | **25 464** | **795.8** | 517 | **850** | **1 700** | 168 |
| lean − flat advance | 1 | 31 728 | 991.5 | 517 | 7 998 | 13 560 | 64 |
| lean − unsigned bounds | 1 | 29 864 | 933.2 | 517 | 854 | 1 708 | 168 |

The `regs` and `spill` columns are `ptxas -v`, read **per function**.
`sass_census.py` originally took a whole-file regex, and the naive port's
default row came out at **70 registers with zero spill** against 36 384 SASS —
contradicted by the same port's own `NOLOCALS=1` row at 168 registers and
7 460 B on a kernel differing by 40 instructions. The 70/0/0 was borrowed from a
different entry function in the same PTX. ptxas splits one function's resources
across two lines (`Function properties for …` / `Used … registers, …`), so the
parser pairs by function name; the numbers above are the corrected ones and they
agree with `notes/19` §4.1's 168 registers and 7 460 B for the same port.

`lean − flat advance` is `lean` with **only** C-14's term removed, so the
difference between those two rows *is* C-14's term and nothing else.

**The prediction was about the right size and the right sign, and the
attribution it was attached to was wrong.** Measured: naive → lean is
**1.429× fewer instructions** — 30.0% of the naive port's per-step cost was its
own index arithmetic. The flat-address advance alone accounts for **1.246×** of
it. The residue's 16.4×-the-reference does not come from the port.

So, against `derivation/minecu_gap.py`:

| claim in the audit | status after this arm |
|---|---|
| the port's per-step index arithmetic was a real defect | **confirmed**, `MEASURED` at 1.43× |
| the unlisted term was per-step address arithmetic | **confirmed** — it is worth 1.25×, not 9.73× |
| `9.73×` of `47.16×` was unattributed | **superseded.** The audit divides into `notes/19`'s whole-graph 36 600; this arm's totals are different objects (the hot kernel alone; the lean port's kernel), so reusing 9.73× against a different numerator would be the category error C-14 exists to prevent. Re-derived per port in `knobs_table.py:decomposition`: 9.67× naive, **6.77× lean** |
| "only the ablation separates (a) the port from (b) the lowering" | **now run.** (a) was real and worth 1.43×; (b) is the **larger** share of what remains |

**One ratio is refused, and it is the one the audit asked for.**
`minecu_gap.py:sass_per_op` reports "21.5 SASS per source op" against a 1–3 band
and asks for a discrimination. I could not establish its denominator, and the
reason is a measured property of the compiler: **it replicates the voxel gather
rather than emitting one per step.** The lean kernel's emitted CUDA contains
**496 reads of the voxel buffer, not 32** — 16 step bodies ~400 lines apart plus
a 481-read tail. A ratio built on a guessed denominator is a number that looks
rigorous and means nothing, so `knobs_table.py:sgd_audit` reports the
per-step SASS instead (`hot/32`, and the source does contain 32 step bodies) and
says REFUSED. The `naive` row of the audit's 21.5 uses its `[HAND-COUNT]` 53 and
is left as it stands rather than restated on a different basis.

---

## 5. On the card

`MEASURED`. One Tesla T4, compute capability 7.5, driver 580.159.04, nvcc 12.8,
one session, every port and knob configuration interleaved. Batch 1024, res 32,
`max_steps` 32, world 32^3 -- the same shape as every other number in this study.
Artefacts: `out/minecu_tg_knobs/armB/battery.json` and `t4_final_run.txt`. The
run took 1 275.7 s and covered **56 timed configurations**.

**The instrument**, stated because every number here rests on it:
`perf_counter` around `realize()` plus a device **synchronise**, median of 8
after 3 warmup -- the same shape as `minecu_tg/bench.py:time_call`, which is the
instrument `notes/19` §5.2 used. Not tinygrad's own `time_call`: in this version
that is `time_call(call: UOp, ...)`, it takes a **realized UOp** and has no
`reps` argument.

**The workload gate passed on every configuration**: mean 19.045 DDA
steps/pixel throughout, against a floor of `max(2.0, 0.10 x 32) = 3.2`. A knob
that had quietly changed the traversal would have moved that number, and none
did -- so all 56 rows compare the *same amount of work*.

**The control's own spread**, re-timed interleaved through the whole run:
6.14% (`naive`), 4.66% (`lean`), 4.44% (`lean-no-flat`), 7.35%
(`lean-no-unsigned`). This is what makes every "no effect" below falsifiable: a
knob effect smaller than this is not an effect.

### 5.1 The knob sweep, in wall clock

`lean` port, ratio to this run's own `lean` control. **All thirteen ratios lie
inside the control's 4.66% spread; the largest is 1.025 and the smallest 0.982.**

| knob | ratio to lean control | codegen change (§3) | SASS change |
|---|---|---|---|
| `USE_TC=0` | 1.007 | byte-identical | 0 |
| `TC_SELECT=0` | 0.991 | byte-identical | 0 |
| `TC_SELECT=2` | 0.982 | byte-identical | 0 |
| `TC_OPT=1` | 1.002 | byte-identical | 0 |
| `TC_OPT=2` | 1.022 | byte-identical | 0 |
| `ALLOW_TF32=1` | 0.987 | byte-identical | 0 |
| `IMAGE=1` | 1.025 | byte-identical | 0 |
| `NOLOCALS=1` | 0.992 | `__launch_bounds__` 128 -> **1** | +8 |
| `TRANSCENDENTAL=2` | 0.983 | `1/sqrt` -> software `pow` | **+968** |
| `NOLOCALS=0+TRANSCENDENTAL=1` | 1.005 | -- | 0 |
| `NOLOCALS=0+TRANSCENDENTAL=2` | 0.995 | as `TRANSCENDENTAL=2` | +968 |
| `NOLOCALS=1+TRANSCENDENTAL=1` | 1.006 | -- | 0 |
| `NOLOCALS=1+TRANSCENDENTAL=2` | 1.012 | as both | +976 |

Two rows are worth pulling out, because they are the ones a static census could
not have settled.

**`NOLOCALS=1` is 0.992 -- inside the noise, with a one-thread block.** §3.3 said
this row would be unpriceable without a wall clock. The wall clock prices it, at
nothing, and the reason is visible in the same table: 14 kernel launches,
1024x32x32 = 1.05 M threads, and a kernel issuing ~795 SASS per thread with 850
LDL of spill traffic. This kernel is **issue-bound, not latency-bound**, so there
is no idle warp for a wider block to fill and nothing for occupancy to buy. That
is a far more specific claim than "no effect", and it needed the clock.

**`TRANSCENDENTAL=2` is 0.983 despite +968 SASS.** The instructions are real and
they cost nothing measurable, for the reason §3.2 gave: they sit in the
**prologue**, executed once per thread before the DDA loop, while the ~25 000
instructions that dominate the kernel are in the loop. A knob that adds 3.8% to
a kernel and ~1.7% to its runtime -- against a 4.66% spread -- has its cost
correctly placed relative to where the kernel spends its time, and it still buys
nothing.

### 5.2 The port ablation, in wall clock

Same run, same card, same workload, four ports.

| port | device time | vs `naive` | DDA-kernel SASS | SASS vs `naive` |
|---|---|---|---|---|
| naive (arm A) | 616.2 ms | 1.000x | 36 384 | 1.000x |
| lean - unsigned bounds | 417.0 ms | 1.478x | 29 864 | 1.218x |
| lean - flat advance | 585.6 ms | 1.052x | 31 728 | 1.147x |
| **lean (arm B)** | **391.4 ms** | **1.574x** | **25 464** | **1.429x** |

The two instruments agree in **ordering** and disagree in **magnitude**, and the
disagreement is the finding:

- the **flat-address advance** is worth 1.147x in instructions and **1.052x** in
  wall clock;
- the **integral cursor and folded bounds test** together are worth 1.429x in
  instructions and **1.574x** in wall clock.

The wall clock is *more* sensitive to the port than the instruction count is.
That is backwards from the usual expectation, and it has a cause worth naming:
the instruction count is static, and the port change also moved **spill traffic**
from 5 231 to 850 `LDL` and spill stores from 7 460 B to 1 700 B. `LDL`
instructions are cheap in *count* and expensive in *latency* -- they are
local-memory round trips to a frame outside the register file. **A static
census that counts instructions under-weights exactly what the port fixed**, so
§4's 1.429x is the conservative figure and the two are not interchangeable.

### 5.3 What this arm still cannot say

- **No ratio here is against the reference.** `notes/19` §5.2's reference column
  (334.8 us at batch 1024, from `refdrv.cu` linking the unmodified hash-pinned
  `kernels.cu`) was measured in a different run, with a different graph, on a
  shared free-tier T4 whose run-to-run spread this study has recorded as 2.9%.
  Dividing this run's 391.4 ms by that run's 334.8 us would manufacture a 1169x
  out of two runs that were never on the same card together. **The
  reference-vs-`naive` ratio is withdrawn; the reference-vs-`lean` ratio has not
  been measured**, and the honest statement is that arm B's absolute standing
  against the hand-written kernel is **unknown**.
- **`4.55 M env-steps/s` is not a denominator** for anything here. It is a
  whole-step figure over 7 launches; this is one render graph. `notes/19` §5.1
  settles that and this note does not reopen it.
- One card, one session, one shape. The control spread is the confidence
  interval; nothing here is a distribution.

### 5.4 The harness, and the ten runs it took

The timings exist because ten runs were dispatched, in
`out/minecu_tg_knobs/armB/t4_run*.log`. The card was reached and the payload
verified byte-for-byte on it from the first attempt onwards. Every failure was
in the harness, and each is recorded because each would have read as a null
result if it had not been caught:

| run | what happened | the lesson |
|---|---|---|
| 1 | `FileNotFoundError: '/Users/.../.venv/bin/python'` | the remote command carried a **host** absolute path. A host path is the one thing about a remote command that cannot be right |
| 2 | `REFUSING: DEV='CPU'` after 0.1 s | a Kaggle GPU notebook does not export `DEV=CUDA`, and tinygrad defaults to CPU |
| 3 | the payload contained `minecu_tg_faithful/` | `gpush`'s `--include` is a **prefix** match (`gpush/manifest.py:227`), so `--include minecu_tg` also ships `minecu_tg_faithful/`, `minecu_tg_opt/` and `minecu_tg_knobs/` -- other lanes' in-flight work, in a payload whose whole purpose is that its bytes can be named. Trailing slashes make each entry an exact directory |
| 4 | mass-deletion guard refused the cleanup of run 3 | correct: a 44.8% prune against a 30% limit, and the guard cannot know those files were an accident. The override is passed at the one call site that knows why |
| 5 | `ModuleNotFoundError: No module named 'tinygrad'` | **a payload is a source tree, not an environment.** `battery.py` installs it and refuses with a reason if it cannot |
| 6 | `ImportError: cannot import name 'time_call'` | a hard-coded instrument path, in a version whose `time_call` takes a UOp |
| 7 | `published bytes: DIFFERING 1: battery.py` | the file was edited **while** its push was in flight. gpush's gate caught it and spent no GPU time |
| 8 | the card ran a **superseded** `battery.py` | the payload still contained eight files this lane had moved out that morning, and `plan` reported `change +0 ~0 -0 =48, no changes`. **A deletion is not a content change** to a content-addressed manifest, so stale modules keep shipping. `--full-resync` |
| 9 | `FileNotFoundError: 'clang'` in the CPU child | and then `CLANG=0` was the wrong fix, because in this TinyGrad `CLANG=0` selects `X86Compiler`, which does not compile C at all. The child now runs on `DEV=PYTHON`: this arm needs the **bytes**, not a CPU backend. Three wrong answers to one question is why the answer is a device name now |
| 10 | `TypeError: time_call() got an unexpected keyword argument 'reps'` | **a function's name is not its signature.** The instrument is now written out in full in `battery.py` and in this section |

Two of these are worth more than this arm's own results.

**Run 2's lesson** is that `os.environ["DEV"]` assigned inside `main()` is **too
late**: tinygrad resolves `Device.DEFAULT` when the device is first built, so the
probe reported *"a CUDA kernel ran and returned 15.0"* **on a machine with no
CUDA at all**. A gate that passes on the absence of the thing it gates is worse
than no gate. It is now `os.environ["DEV"] = "CUDA"` at module scope, and
`probe_cuda()` checks that a kernel ran *and* returned the right number.

**Run 8's lesson** is that `plan` reported **no changes** while the payload was
wrong. A manifest that cannot express "this file no longer exists" will keep
shipping it, and a payload carrying a superseded module of the same name is worse
than a failed one: the card did not error, it ran the old code. That is a
property of the transport, not of this arm, and it is recorded here because
`notes/22-gpush.md` owns the tool and this is evidence about it.

## 6. Classification, per gap, with the falsifier for the other three

Classes: **(i)** cannot express, **(ii)** expressible but not found, **(iii)**
not implemented, **(iv)** does not exist for this architecture.

| # | gap | class | the evidence that picks it | what would have made it (i), (ii) or (iv) instead |
|---|---|---|---|---|
| 1 | The tensor-core knobs (`USE_TC`, `TC_SELECT`, `TC_OPT`) are inert on this kernel | **(i)** for this workload | Their only reads sit behind `heuristic.py:28`, which needs a reduce axis; the DDA has none. `TC_OPT=1,2` flip that guard's second disjunct and still emit byte-identical source — measured | **(ii)** if a knob had changed the source and improved the count, which would mean the scheduler could have used it; **(iv)** if the guard had been architecture-dependent. Falsifier: byte-different source at any `TC_*` value on this kernel |
| 2 | `ALLOW_TF32` is inert | **(iv)** | `postrange.py:226` filters fp32 tensor cores, and sm_75 has none: `cuda_sm75 = cuda_8168_f16` (`tc.py:94,98`). This is a fact about the card, not about the workload | **(i)** if sm_75 *had* fp32 tensor cores and the port still emitted no `dot`; **(ii)** if `ALLOW_TF32=1` had changed a matmul's source here. Falsifier: the same sweep on sm_80 differing |
| 3 | `IMAGE` is inert | not a gap — a shape fact | It covers `dot`, `conv2d` and 2-D image-shaped buffers (`mixin/op.py:386,1593`, `heuristic.py:47`); the DDA gathers a 3-D volume | **(i)** would need the DDA to be inexpressible, and it is expressed; **(iv)** would need the texture path absent on sm_75, which it is not |
| 4 | `TRANSCENDENTAL=2` costs 968 SASS and recovers nothing | **(i)**, deliberately | The DDA's only transcendental is one `rsqrt` in the prologue. `MEASURED`: the knob reaches it (SQRT→`pow`, +968 SASS) and `MEASURED` on the card it is **x0.983** — the prologue is not where the time is | **(ii)** if the cost were on the hot path and the knob traded prologue for step; **(iv)** if sm_75 lacked a `powf`. Falsifier: a time move following the count, or the image changing |
| 5 | `NOLOCALS=1` sets `__launch_bounds__` 128 → 1 | **(ii)** | It is the only knob that reaches the scheduler's shape decisions (`heuristic.py:81,163`), and the block size it emits is 1. `MEASURED` on the card: **x0.992**, inside the 4.66% control spread — the cost model made a decision that is wrong in principle and free in practice, because the kernel is issue-bound and has no idle warp to fill (§5.1) | **(i)** if there were no way to express a block size at all; **(iv)** if sm_75 lacked the launch-bound path. Falsifier: a wall-clock move — measured, and there is none |
| 6 | After the port is fixed, the DDA still costs 16.4× the reference per step | **(ii)** | The port's own index arithmetic is gone (measured 1.43×) and 795.8 SASS/step remains, with 517 LDG and 850 LDL. The port cannot be the remaining story because the port is now the algorithm | **(i)** would need the 32 unrolled bodies to be inexpressible, and they are in the graph (12 313 lines of emitted CUDA); **(iv)** would need a per-ray exit to be absent from the ISA, and `BSSY`/`BSYNC` exists on sm_75. Falsifier: an instruction count that *falls* to ~50/step once index arithmetic is provably minimal — which is what a (i) or (iv) reading predicts and this run refutes |
| 7 | 517 global loads vs the reference's 16 | **(ii)**, carried over from `notes/19` gap 7 and **not re-opened** | `MEASURED` identical on all four ports, including the lean one with a single-index gather. The compiler emits four predicated byte loads of one address per step, and the *port* no longer asks for four | **(i)** would need a one-load-per-gather lowering to be inexpressible; **(iv)** would need predicated byte loads absent from sm_75. Falsifier: recompiling the capture with `--fmad=false` and seeing the four merge |
| 8 | `OPT_LEVEL` | **not a class** | It does not exist in this TinyGrad. `Context(OPT_LEVEL=0)` raises `KeyError`; there is no `ContextVar` definition line | none — an absence has no class. Recording it as (i)–(iv) would be a category error |

Gaps 6 and 7 are the ones that survive the knobs, and both are **(ii)**: the
capability is there and the search is not choosing it. **No row came out (iii)** —
nothing in this sweep was a missing implementation waiting to be written, which
is a result about the *shape* of the problem: what the compiler does not do here,
it declines to do rather than fail to know how.

---

## 7. What this arm rules out, and what it does not

**Ruled out, with evidence rather than by absence of effort:**

- that any of the six inert knobs is a *missed opportunity* on this kernel. Each
  is inert for a named, checked reason, and the `TC_OPT` case -- the one where the
  guard's disjunct can be flipped -- was swept rather than argued.
- that the port's index arithmetic was the remaining story. It was worth
  **1.429x in instructions and 1.574x in wall clock**, measured, against a 46.9x
  instruction gap.
- that `ALLOW_TF32`'s inertness is a workload fact. It is an sm_75 fact, which
  is a different row of the taxonomy and a different experiment elsewhere.
- that `NOLOCALS=1` is expensive. It is a real schedule decision, made wrongly,
  and it costs nothing measurable on this kernel because the kernel is
  issue-bound rather than latency-bound (5.1).
- that `TRANSCENDENTAL=2` is expensive. +968 SASS, **x0.983** on the card, because
  the instructions it adds are in the prologue.
- that the gap is a **tiling** problem. No knob in the framework's exposed set
  addresses tiling for this kernel, and none of them is close enough to matter
  to try.

**Not ruled out, and stated as open:**

- **Arm B's absolute ratio against the reference.** Not measured, and not
  derivable from anything in this note (5.3). This is the largest gap in the
  arm and it is a gap in the *harness*, not in the finding.
- **Whether `TRANSCENDENTAL=2` changes the image on the card.** The `pow` is more
  accurate than `1/sqrt`, which feeds every `t_max`, and a DDA step is chosen by
  `t_max_x < t_max_y`. `verify.py` exercises the knob's rewrite on the CPU
  backend; the on-card check was not run, and this knob is a numerical change
  dressed as a performance one.
- **Whether any of it survives a second card.** One T4, one session, one shape,
  with the control's 4.66% spread as the only confidence interval. 6's
  classification rests on codegen and on the reference's own SASS, which are
  card-independent; the timings are not.

**One sentence, as asked:** *no -- giving the compiler its own knobs does not close
the gap; six of the seven sweepable knobs emit byte-identical CUDA, the seventh
makes the hot kernel 3.8% bigger and its runtime no different, and the only knob
that reached the scheduler was free in wall clock despite emitting a one-thread
block -- which rules out the possibility that this gap is a tuning failure
reachable through the framework's existing exposed knobs, and leaves it where
C-14 left it: a real port defect worth 1.43x, a residue of 16.4x the reference per
step with the port already correct, and therefore class (ii) -- expressible, and
not found by the scheduler.*

---

## 8. Files

| what | where | what it does |
|---|---|---|
| the knobs | `minecu_tg_knobs/registry.py` | every knob, its real `Context` key, the tinygrad line that reads it, and three run-time gates: the Context keys exist, the cache-key coverage holds, and every cited line is un-drifted in the installed package |
| the port | `minecu_tg_knobs/lean.py` | the lean port; three switchable fixes; no knob, no escape hatch, no hand-staging |
| the gate | `minecu_tg_knobs/verify.py` | the port vs an independent float64 transcription, on the image **and** the per-pixel step count, all five variants, five shapes |
| codegen | `minecu_tg_knobs/codegen.py` | the offline census on `NULL:CUDA:sm_75`; per-kernel sha256, schedule fingerprint, and the `.cu` files the SASS is compiled from |
| SASS | `minecu_tg_knobs/sass.sh`, `sass_census.py` | generated CUDA -> sm_75 SASS in the archlab container; one categoriser, imported from `tools/archlab/inspect_ptx.py`; per-function `ptxas -v` resources |
| the table | `minecu_tg_knobs/knobs_table.py` | the five claims this note cites, built from the artefacts so no cell can drift from its source |
| the card | `minecu_tg_knobs/battery.py`, `kaggle/run.sh` | on-card timings with the CUDA probe, the workload gate, and the interleaved control |
| evidence | `out/minecu_tg_knobs/armB/` | `census.json`, `sass_census.json`, `table.txt`, `verify.log`, `battery.json`, `t4_final_run.txt`, and the ten `t4_run*.log` |

### Re-deriving every number in this note

```bash
python -m minecu_tg_knobs.registry                      # knobs + 3 gates
DEV=CPU python -m minecu_tg_knobs.verify                # the port, 45 checks
python -m minecu_tg_knobs.codegen --out DIR             # source census
./minecu_tg_knobs/sass.sh DIR sm_75                     # -> real SASS
python -m minecu_tg_knobs.sass_census --stage STAGE --json OUT
python -m minecu_tg_knobs.knobs_table --codegen C --sass S
./minecu_tg_knobs/kaggle/run.sh                         # SPENDS GPU QUOTA
```

`knobs_table.py` is the citable surface. Its claims are `sweep`, `knob_effects`,
`port_ablation`, `decomposition` and `sgd_audit`, and each one is a function over
the two JSON artefacts rather than a number typed into prose — which is the
structural fix for the failure mode this repo has been bitten by most often.

### The one set of numbers this arm did not produce

`776` (the reference's whole render, decomposed in `notes/19` §4.0.1 as
152 prologue + 16x38 body + 16 epilogue) and `26.4` (its measured mean DDA
steps/pixel, `notes/19` §1.3) are **read out of `notes/19`, not re-measured
here.** They are the hand-written side, which is arm A's artefact, and using
them as denominators is stated in `knobs_table.py:decomposition`'s own output.
Every generated-side number in this note comes from a cubin this arm compiled or
a clock this arm read; and `knobs_table.py:decomposition` refuses to run at a
`max_steps` other than 32, because the two multiplicative terms it reuses are
written for the reference's default and a residue computed at another step cap
would be arithmetically valid and semantically wrong.
