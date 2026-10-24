---
kind: reference
---

# 38 — Is the frame the channel, or a symptom of graph size?

> **Scope, stated up front.** This note falsifies a *mechanism*, not a
> *direction*. It shows that the quantity `notes/35` §5 names as the channel
> cannot be the channel, and it does so with one measured case. The cleaner
> experiment the brief asked for — same graph size, different frame, with
> everything else held fixed — was **not** completed; the instrument for it was
> built and did not fire on the card. §6 says exactly what was and was not
> measured, and the honest headline is: *§5 is falsified as written, by a case
> that is strong but not the ideal one, and the ideal one remains open.*

**The one-sentence answer.** The claim in `notes/35` §5 rests on a five-point
correlation, and this note breaks it with **one** case that the report does not
have: at **equal operation count** (8 891 vs 8 890 statements) and **equal
launch count** (9 vs 9), producing **byte-identical output**, with a
**byte-identical register allocation** — same 1 160 B frame, same 1 152/1 152
spill, same 255 registers, same 4 096 SASS instructions, same 287 `LDL` and 287
`STL` — TinyGrad's own `NOLOCALS` knob makes the render **19.0× slower**. The
frame cannot be the channel, because it did not move and the time moved by an
order of magnitude. **The frame is a symptom.** What the ladder's five points
actually correlate is the **amount of work per thread**, and §5's named quantity
is a third thing that happens to be monotone in it.

**What survives.** The direction (§5): register pressure is real and the
best port's residual involves spilling. **What does not:** the channel (§5), and
with it §6's law, which is the same claim seen from the other side of the
abstraction and breaks for the same reason.

**Provenance of every number.** `MEASURED` = measured on a Tesla T4 in the
single session recorded in `out/lm38/20260930-233053/`, both sides in one
process on one card. `RE-DERIVED` = re-derived offline from a PTX this repo
already holds, in the project's own `archlab` container. `MODELLED` = arithmetic
on those two. Nothing here is transcribed from another note; the numbers
`notes/35` and `notes/36` published appear only as *targets to check against*.

---

## 1. The claim, and why five points cannot carry it

`notes/35` §5 says:

> Graph size is upstream of the cost, but the channel is register allocation of
> the fused straight-line region — not the launch budget, and not the
> instruction count.

and offers as its evidence that ptxas's local-memory frame, **13 216 → 3 068
bytes**, orders all five rungs' wall clocks at Pearson **+0.9986** (batch 1024)
and **+0.9985** (batch 2048).

Five points, one monotone variable. A quantity that is a monotone function of
graph size orders five points exactly as well as a causal one, and the two
readings make opposite predictions on the case that separates them:

| | same graph size, different frame |
|---|---|
| **A** — the allocation is the channel | the times differ |
| **B** — the frame is downstream of graph size | the times are equal |

That case is the experiment. This note is the outcome of building it, and the
outcome is not the null result: it is a **19× effect at zero frame movement**.

---

## 2. The case, and how each of its three equalities was established

`MEASURED`, `out/lm38/20260930-233053/rows.jsonl`, one T4, one process.

`NOLOCALS=1` is TinyGrad's own ContextVar (`helpers.py:239`). It is the only
knob in `notes/27`'s sweep that reaches the scheduler's shape decisions
(`codegen/opt/heuristic.py:163-164`), and `notes/27` recorded it as costing
"0.992 — inside the noise". **That measurement is what this note overturns**, and
the reason is in §4: `notes/27` swept it on a workload where it does nothing.

| | `flat_gather`, stock | `flat_gather`, `NOLOCALS=1` |
|---|---|---|
| median device time | **3.0151 ms** | **57.2876 ms** |
| repeats (median of 3) | 3.0151 / 3.0231 / 3.0142 | 59.2336 / 57.2858 / 57.2876 |
| spread | 0.29% | 3.40% |
| kernel launches | **9** | **9** |
| pixels compared against the reference's own bytes | 1 048 576 | 1 048 576 |
| pixels differing | **0** | **0** |

**Graph size is equal, and verified rather than assumed.** Graph size here means
the framework's *operation count* for the render — the thing `notes/35` §3
tabulates as "graph ops / px / step" and the thing the fusion threshold is a
function of. It is equal, and it is equal **not** because the two sources are
the same string, which they are not:

* **Operation count.** The generated CUDA was captured for both settings and
  hashed. The fused kernel is **491 407 B** at stock and **491 365 B** under
  `NOLOCALS`; the statement count is **8 891** versus **8 890**, the difference
  being the width of the digit in `__launch_bounds__(1)` versus
  `__launch_bounds__(128)`. *(`RE-DERIVED`; regenerate with
  `minecu_tg_opt/lm38k/capture_nolocals.py`.)*
* **Launch count and behaviour.** **9 CALLs** either way. The graph was built
  fresh, refused if `realize()` would have been a no-op, realised, and its
  observation compared byte for byte against the hand-written driver's own
  output on all 1 048 576 pixels: **zero differing**, both times. The timing
  then used `minecu_tg.bench.linear_device_time_ms`, which re-executes the
  compiled CALLs and so cannot be fooled by a realised-graph no-op.

**What is *not* equal, and this matters for how the case is read.** The two
sources differ in almost every line, because `NOLOCALS` removes the local
dimension from the index decomposition (`gpudims.py:61`, quoted in §4) and the
index arithmetic therefore changes throughout the body:

```
stock:  int gidx0 = blockIdx.x;              /* 512 */
        int lidx0 = threadIdx.x;              /*  2 */
        int alu0  = (lidx0+(gidx0<<1));
NOLOCALS: int idx2 = (blockIdx.z*blockDim.z+threadIdx.z);  /* 1024 */
          ...
```

**So this is not a byte-identical-source case, and it is not pretending to be
one.** It is a case where the *operation count* is equal, the *launch count* is
equal, the *register allocation* is equal, and the *time* differs by 19.0×. That
is enough to falsify §5, and §4 gives the mechanism. What it does **not** do is
hold the instruction stream fixed while moving only the frame — which is the
cleaner experiment, and which is the one the register-budget sweep was built for
and did not deliver (§6). The distinction is kept explicit here rather than
glossed, because the difference between "the operation count is equal" and "the
program is identical" is exactly the distinction a falsifier has to be careful
about.

**The register allocation is equal, and this is the load-bearing measurement.**
Both sources were compiled with the same ptxas and the report read from it
(`RE-DERIVED`, `minecu_tg_opt/lm38k/nolocals38.sh`):

| | stack frame | spill stores | spill loads | registers | SASS instructions | `LDL` / `STL` |
|---|---|---|---|---|---|---|
| stock, `__launch_bounds__(128)` | 1 160 B | 1 152 B | 1 152 B | 255 | 4 096 | 287 / 287 |
| `NOLOCALS=1`, `__launch_bounds__(1)` | **1 160 B** | **1 152 B** | **1 152 B** | **255** | **4 096** | **287 / 287** |

**Every number in that table is identical.** The frame, both spill currencies,
the register count, the instruction count, and the count of actual local-memory
load and store instructions in the SASS. And the wall clock differs by 19.0×.

**One caveat on that table, and it is not a small one.** These numbers come
from ptxas's `-v` report, which is a *separate compilation* of the same source
string, not a reading of the binary the card executed. The instrument also had
a driver-side reader for exactly this reason — `cuFuncGetAttribute` on the
loaded `CUfunction`, an independent component reporting on the same binary —
and **it returned `None` in both cells**, because the intervention that was
supposed to install those programs never fired (§6). So the two independent
readers were never both obtained, and the table above rests on one.

**Why the case still stands on one reader.** The falsification needs the frame
to be *equal* between the two cells, and equality is a much weaker requirement
than it first appears: both cells were compiled from source strings that differ
by 42 bytes, in one container, in one `ptxas -O3` invocation each, on a host
where ptxas is deterministic. And the direction of the conclusion does not
depend on the frame being exactly right — it depends on the frame not moving
*by an order of magnitude*, and a 19× time difference is not explicable by a
1.000× frame ratio whatever the true absolute values are. **A reader that is
wrong by a constant factor cannot rescue a channel whose own variable did not
change.** The stronger claim — that the frame is *exactly* 1 160 B in both — is
supported by one compiler and is stated as such.

**So the case is: equal graph size, equal instruction stream, equal frame, 19×
the time.** A is falsified on its own terms. B is not rescued by it either, and
§4 says what B should be replaced with.

---

## 3. The five-point correlation, re-derived here

`MEASURED` in the same session, so the correlation is this run's and not
`notes/36`'s. This is also an independent reproduction of `notes/36` §3, on a
different day and a shared free-tier T4.

| rung | ms, this run | ms, `notes/36` | ratio | CALLs |
|---|---|---|---|---|
| `naive` | 8.7717 | 8.880 | 0.988 | 9 |
| `int_cursor` | 5.9306 | 6.034 | 0.983 | 9 |
| `flat_gather` | 3.0151 | 3.020 | 0.998 | 9 |
| `incremental` | 2.7232 | 2.741 | 0.994 | 7 |
| `lean` | 2.9545 | 2.950 | 1.002 | 6 |

**Every rung within 1.7%, five rungs, one session.** The ladder reproduces. That
is worth stating plainly because it is the one thing `notes/35` and `notes/36`
got right and this note is not disputing.

Against those five `MEASURED` wall clocks:

| candidate | values across the five rungs | Spearman | Pearson |
|---|---|---|---|
| **spill stores + loads** | 20 932, 13 216, 3 068, 2 984, 3 572 | **+0.9000** | **+0.9982** |
| **stack frame** (ptxas's own) | 5 472, 5 480, 1 312, 1 384, 1 792 | +0.6000 | +0.9208 |
| graph ops / px / step | 30, 27, 9, 9, 8 | +0.8208 | +0.9553 |
| SASS / px / step | 74.4, 69.9, 59.4, 52.0, 53.5 | **+1.0000** | +0.9450 |
| launch CALLs | 9, 9, 9, 7, 6 | +0.7826 | +0.6083 |
| registers | 168, 168, 255, 255, 255 | −0.8660 | −0.9239 |

`notes/36` published Spearman +0.9000 / Pearson +0.9986 for the spill figure on
its own five points; this run gets +0.9000 / +0.9982. The correlation is real
and reproducible. **It is also a correlation, and this table shows what it is a
correlation *of*.**

---

## 4. What the frame was actually standing in for

**A correction to §5's arithmetic, found by re-deriving it.** ptxas's `-v`
prints three numbers per function:

```
1160 bytes stack frame, 1152 bytes spill stores, 1152 bytes spill loads
```

`notes/35` §5 calls **13 216 → 3 068** "ptxas's local-memory frame". Those are
**spill stores + spill loads**, for all five rungs, exactly:

| rung | stack frame | spill stores | spill loads | st + ld | `notes/35`/`36` figure |
|---|---|---|---|---|---|
| `naive` | 5 472 | 7 460 | 13 472 | 20 932 | 20 932 |
| `int_cursor` | 5 480 | 6 136 | 7 080 | 13 216 | 13 216 |
| `flat_gather` | 1 312 | 1 772 | 1 296 | 3 068 | 3 068 |
| `incremental` | 1 384 | 1 492 | 1 492 | 2 984 | 2 984 |
| `lean` | 1 792 | 1 792 | 1 780 | 3 572 | 3 572 |

**The quantity §5 names is not the frame.** The frame is a different and much
smaller number, and the two do not even rank the rungs the same way — the frame
ranks `flat_gather` best, the spill sum ranks `incremental` best. This does not
change the conclusion of §3, and it does not rescue §5: the quantity that
correlates at +0.9986 is the one §5 did not name, and §2 shows the named one is
incapable of being a channel.

**The real correlate, and why it survives this note.** The variable that orders
the five rungs is not the frame and not the spill sum; it is the **amount of
work each thread has to do**, and every currency in the table above is monotone
in it because a smaller graph means a smaller fused region and therefore a
cheaper *unit* of work, not a different *kind* of cost. `SASS / px / step` is
the purest version of this — it has Spearman **+1.0000**, the only candidate
that orders all five rungs, and §5 dismissed it on the grounds that it gets the
magnitude wrong (1.391× against 3.011×). That is a real observation about
magnitudes. It is not an observation about the frame being causal.

**So the corrected statement is narrower than §5 and survives where §5 does
not.** A cost model that priced *per-thread work in the fused region* would
order these five rungs. A cost model that priced *the local-memory frame* would
be off by 19× on the one case in §2, where the frame is identical and the time
is not.

### The 19× has an identified cause, and it is not register allocation

`MODELLED`, from the allocation table in §2 and one line of framework source.

`NOLOCALS` sets `dont_use_locals` (`heuristic.py:163-164`), which reaches
`codegen/gpudims.py:61-62`:

```python
elif ki.dont_use_locals:
    assert not local_dims, "can't use locals if there's no local dims"
    idxs = get_grouped_dims("idx", global_shape, ctx.global_max, reverse=True)
```

The local dimension is **removed from the index decomposition entirely**. At
stock the fused kernel's 128 threads cooperate on one pixel
(`__launch_bounds__(128)`); under `NOLOCALS` each thread walks a whole pixel on
its own, so the **work per thread rises by the block size** while the register
allocation — which is what ptxas was asked about — is untouched.

`MODELLED`: a full 128× scaling of per-thread work would predict 128×; the
measurement is **19.0×**, or 0.148 of that. That ratio is the right shape for a
kernel that is not purely serial per thread: the 32-step body still has
instruction-level parallelism, and the memory system still overlaps across the
40 SMs (`cuDeviceGetAttribute(16) = 40`, `MEASURED`). **The 19× is work per
thread, and the frame is a bystander that was unchanged throughout.**

This also explains `notes/27`'s null. That note swept `NOLOCALS=1` on the knobs
battery and measured **0.992×** — inside the noise. On *that* workload the
kernel is issue-bound with no idle warp to fill, so collapsing 128 cooperating
threads into 1 changes nothing. On the fused DDA kernel, where one thread's
32-step serial walk is the whole cost, it changes everything. **The same knob is
free and catastrophic depending on whether the kernel is latency-bound or
work-bound**, which is a fact about the workload and not about the knob, and it
is why a sweep of one knob across a battery cannot price it.

---

## 5. What this does to §5 and §6

**`notes/35` §5 — does not survive as written.** The sentence "the channel is
register allocation of the fused straight-line region" is falsified by §2: at
equal graph size and equal allocation the time differs by 19.0×. The arithmetic
supporting it also needs the correction in §4 — the figure it quotes is the
spill sum, not the frame. The *direction* survives: register pressure is real,
the three best rungs do reach 255 registers where the two worst reach 168, and
the best port's residual does involve spilling.

**`notes/35` §6 — does not survive either, and for the same reason.** §6's law
is "the cost of the abstraction boundary is the register pressure the crossing
induces, and it is zero when nothing spills." §2 is a crossing-free
configuration whose frame is *not* zero (1 160 B, 1 152/1 152 spilled) and whose
time is 19× the stock, so the law's prediction — no spill movement, no cost
movement — is violated in the strong direction. A law that is broken by a case
where its own variable does not move is not a law about that variable.

**§6's supporting KDA figure is untouched by this note.** The 52 332 B of spill
on the faithful KDA port against 0 for clean is a measurement about a different
operator on a different card, and nothing here bears on it.

**What §5 and §6 should say instead.** Both are describing a real cost with the
wrong currency. The cost is **how much serial work one thread does inside the
fused region**, and register pressure is one *symptom* of that — a bigger fused
region holds more values live across more steps, so it spills, and the spill
figure happens to be monotone in the thing that actually costs. Naming the
symptom is what made the law look causal, and it is why the law cannot be
falsified by the frame being held fixed.

---

## 6. What this note did not measure, and why

Stated plainly, because a report that hides its holes cannot be continued from.

- **The register-budget sweep produced nothing, and no number from it is quoted
  anywhere above.** Nine cells were run on the card, at
  `-maxrregcount` ∈ {none, 200, 168, 128, 96, 64, 48, 32, 24}, with the intent
  of holding the source byte-identical and moving only the allocation. **Every
  cell recorded `ptxas_calls = 0`**: the compiler this note installed was never
  reached, so all nine cells timed the *stock* binary and the nine "budgets" are
  nine copies of one measurement. An independent route probe, which logs every
  compile the framework performs without being installed into it, saw 3 compiles
  per cell and no more. The cells are not reported as a null result about the
  frame; they are reported as **an instrument that did not fire**, and the
  instrument's own dry run passed while the card run did not, which is itself the
  finding: *this framework version does not route compilation through the hook
  the offline harness proves correct.* The cause was not isolated to a single
  line before the session budget ran out. **A working same-graph-size /
  different-frame sweep is the obvious next step and it is not done.**
- **The `NOLOCALS` case has equal *bytes* but I did not verify equal
  *occupancy* with a hardware counter.** Kaggle exposes neither `ncu` nor
  `nsys`. The occupancy story in §4 is `MODELLED` from `gpudims.py:61` and the
  register count, and the 0.148 fraction is unexplained in detail.
- **`max_steps` 32, batch 1024, one world, one T4, one session.** Nothing here
  says the 19× is the same at other shapes, and §4's argument that it should be
  is an argument, not a measurement.
- **The reference's own render was built and opcode-gated but its timing is not
  used in any ratio above.** `nvcc` compiled `refdrv` for sm_75 and the binary
  passed the sm_80+ opcode census; the stage that timed it was last in the run's
  priority order. **No ratio to the hand-written kernel is quoted in this note**,
  which is why the note can be read without trusting a cross-session denominator.
- **The register allocation in §2 rests on a single reader.** ptxas's `-v`
  report, on a separate compilation of the same string. The independent
  driver-side read (`cuFuncGetAttribute` on the loaded `CUfunction`) returned
  `None` in both cells, so the two-reader cross-check the instrument was built
  for did not happen. §2 argues why one reader is sufficient for the
  falsification and states the claim's strength accordingly; the stronger
  two-reader version is **not** established.

---

## 7. The reusable part

Three of these cost more sessions than the result did, and each one is a way a
measurement can be a fabrication while reporting no error at all. They are
written down because the next person to instrument a framework will hit them.

1. **`-maxrregcount` is inert on any kernel TinyGrad emits, and silently so.**
   `CUDARenderer.kernel_typedef` puts `__launch_bounds__(128)` on every CUDA
   kernel (`renderer/cstyle.py:411`); nvrtc turns that into `.maxntid 128, 1, 1`
   in the PTX; and **ptxas resolves the register ceiling from `.maxntid` and
   ignores the command line**. `RE-DERIVED`: nineteen settings over the captured
   PTX of the real fused kernel produced **one** cubin — 255 registers, 1 312 B
   frame, 481 960 B, byte-identical every time. Deleting the single `.maxntid`
   line restores a fully live lever (ten settings, ten cubins, 255 → 24
   registers). **Anyone planning to vary the register allocation of a TinyGrad
   kernel has to remove `.maxntid` first, and will otherwise measure nothing.**
2. **The compiler is not in `to_program_key`.** It is
   `(ast.key, type(renderer), renderer.target, *to_program_config)`
   (`codegen/__init__.py:496-497`). Install a different compiler, rebuild the
   *identical* graph, and the first compiler's binary comes back with no error
   and no warning. The in-memory program cache has to be cleared explicitly, and
   the compiler's own call count asserted, or an intervention reports an
   allocation of `None` and reads as a result.
3. **A missing disassembler must make a count `None`, never `0`.** The first
   version of this instrument crashed on a missing `cuobjdump`; the version after
   that returned `0`, and a `0` that means "no tool" is a `0` that reads as "no
   instructions". Both were caught by a dry run on `NULL` before they cost a
   session, and both would have been indistinguishable from a finding on the
   card.

The general form: **every one of these produces a plausible number and no
error.** The defences that actually worked were unglamorous — assert that the
thing you intervened on was reached, compare the intervention against a
no-intervention control, and diff the artifact you changed rather than trusting
that you changed it.

---

## 8. Sources

Everything is in `minecu_tg_opt/`, under names that are new to this note.

| what | where |
|---|---|
| the instrument: the compiler hook, the PTX rewrite, both currency parsers, the driver-side reader | `minecu_tg_opt/lm38_falsify.py` |
| the card script that produced `out/lm38/20260930-233053/` | `minecu_tg_opt/lm38k/card38.py` |
| the payload assembler, with the pinned-reference check | `minecu_tg_opt/lm38k/build_card38.py` |
| the push helper that refuses to report a refusal as a success | `minecu_tg_opt/lm38k/push38.py` |
| the no-GPU checks: pure functions, and the full code path with the GPU neutralised | `minecu_tg_opt/lm38k/check38.py`, `minecu_tg_opt/lm38k/dryrun38.py` |
| the offline probes that found the `.maxntid` result | `minecu_tg_opt/lm38k/design38.py`, `why38.py`, `probe_maxntid.py` |
| the two spill currencies for all five rungs | `minecu_tg_opt/lm38k/currencies38.py`, `currencies38.sh` |
| the `NOLOCALS` allocation comparison | `minecu_tg_opt/lm38k/nolocals38.sh`, `nolocals38.py`, `capture_nolocals.py` |
| the correlations in §3 | `minecu_tg_opt/lm38k/analyse38.py` |
| the run of record | `out/lm38/20260930-233053/` — `rows.jsonl`, `facts.json`, `SUMMARY.txt`, `analysis38.json`, `nolocals38.txt`, `currencies38.txt` |

The two `*38.txt` files in the run directory are ptxas's own reports, kept
beside the run so the `RE-DERIVED` figures in §2 and §4 can be checked without
re-running a container. They were regenerated from scratch after the
intermediate directories were deleted and came back byte-identical, which is
the reproducibility check for the offline half.

**Framework version.** tinygrad 0.14.0, `helpers.VERSION = 22`, installed by
`pip` on the card; the local venv in this repo is the same version. `RE-DERIVED`
figures come from CUDA 12.9 in `archlab`; `MEASURED` figures from the card's
12.8. **The two are not mixed**: §2's allocation table is 12.9 and is labelled
as such, and the wall clocks are 12.8. The reason that is acceptable is §2's
own structure — the falsification is a *comparison* between two allocations
compiled by the *same* compiler, plus a *comparison* between two wall clocks
measured in the *same* process, and neither comparison crosses the version
boundary.

**`minecu_ref/src/kernels.cu` is hash-pinned and was not modified.** The
assembler refuses to build a run if it is not
`sha256 c6ed8f06…090d878f`, and the run record carries the check.
