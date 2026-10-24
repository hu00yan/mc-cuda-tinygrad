# 13 — schedule vs decay: three claims that must not be merged

**The decay is fused. The operator is not. Those are two different sentences, and
the record has been at risk of writing them as one.**

`kda/materialisation_verdict.md` established, from the real emitted CUDA source,
that the KDA `[B,NT,HV,C,C,K]` decay tensor is never materialised. That same
capture printed `captured 1 kernels from CUDA`, while `kda/perf_kda.py` measured
**12,082 kernel launches per call** on the same class of machine. "One kernel"
and "12,082 kernels" cannot both be a census of the operator, and the gap was
flagged in two adjacent notes without being closed.

This note closes it, and in doing so answers the one measurement
`kda/materialisation_verdict.md` §7 named as the thing that would settle the
CUDA question — the M2 schedule census with `Device.DEFAULT == CUDA` at the bench
shape, and its acceptance test, *is there a bucket at 67,108,864 B?*

Everything numeric here is produced by **`kda/schedule_census_cuda.py`**, on a
MacBook with **no GPU in it**. Every figure has a status tag. Nothing is
transcribed from another agent's file; where a prior number is quoted it is
quoted with its path, and re-derived figures are recomputed in code.

```
.venv/bin/python kda/schedule_census_cuda.py              # full, ~35 min, exit 0 on this host
.venv/bin/python kda/schedule_census_cuda.py --quick      # bench shape, one backend, ~45 s
.venv/bin/python kda/schedule_census_cuda.py --no-replay  # skip the real-device section
.venv/bin/python kda/schedule_census_cuda.py --no-scaling # skip the NT sweep
.venv/bin/python kda/schedule_census_cuda.py --one-line --dev CUDA --shape bench
```

The full run is 9 schedule censuses (3 shapes x 3 backends), 2 repeat censuses
plus a cold subprocess, 5 `DEVICE_MAX_BUFS` subprocess probes, a 5-point NT
sweep, and 4 real-device Metal subprocesses with the worker pool on and off.
It exits 0.

Status vocabulary, unchanged from `kda/materialisation_verdict.md`:

| tag | meaning |
|---|---|
| `MEASURED` | an instrument on this host, on real tinygrad 0.14.0 |
| `SOURCE` | read out of the installed tinygrad 0.14.0, quoted with `file:line` |
| `MODELLED` | arithmetic on shapes |
| `INFERRED` / `INFERENCE` | a conclusion that needs hardware this host does not have. **Labelled, never presented as a measurement** |

---

## The four answers, up front

| question asked | answer | section |
|---|---|---|
| **Is there a bucket at 67,108,864 B** (the fp32 `[B,NT,HV,C,C,K]` tensor at the bench shape) in the CUDA schedule? | **No.** Four argument sizes, largest 1,048,576 B, at three measured shapes. `MEASURED`, CUDA backend, no GPU | §2 |
| **What is the reconciled kernel count for the operator?** | **12,082** SINK kernels per `kda_chunk` call on CUDA, 12,090 on Metal and CPU — the T4's own number, reproduced to the unit. The notebook's 1 was a multiprocessing worker pool, not the operator | §4, §5 |
| **Are the 12,082 launches launch-bound or doing real work?** | **Not settled.** The schedule's arithmetic (8 µs at the AST level) and its bytes (≤ 28.2 ms) cannot account for the T4's 215.9 ms, and the DEBUG=2 instrument blocks the host on a `cuEventSynchronize` per launch. Three measurements named; the cheapest is three lines of code | §7 |
| **Which of the three claims could not be settled?** | **Claim 3, the graph-level schedule cost.** Claims 1 and 2 are settled: the tensor is absent, the computation is recomputed in registers | §8, §10 |

---

## 0. The three claims, and why they have to be three

| | claim | instrument | level |
|---|---|---|---|
| **1** | the `[C,C,K]` decay **tensor** reaches device memory | M2 census: kernel **argument** buffer sizes, whole schedule | per-kernel |
| **2** | the decay **computation** is recomputed rather than reused | emitted source: `exp2` count, and whether it is stored or re-read | per-kernel |
| **3** | the **operator's schedule** is a fused schedule | launch census: SINK kernels per `kda_chunk` call | graph-level |

1 and 2 are properties of *one kernel*. 3 is a property of *the graph*. All three
are true here, and 1 and 2 are in no way evidence for 3. Collapsing them
produces one specific false sentence — **"the operator is fused"** — which the
repo's own numbers refute in the same breath: `notes/10-t4-results.md` §2
measures 12,082 launches and 215.9 ms of GPU time for one `kda_chunk` call.

The correct statement, which nothing contradicted and nothing previously said in
one breath:

> **The decay is fused within each kernel. The operator is not fused at all.**

---

## 1. The method, because it is the part that makes the rest possible

**`Device.DEFAULT == CUDA` does not require a CUDA device.** This is the finding
that let the whole question be asked on a laptop.

`Tensor.schedule_linear()` (`tinygrad/tensor.py:413`) is the public API for
"give me the schedule without running it". It calls `create_linear_with_vars` →
`memory_plan_rewrite` (`tinygrad/schedule/memory.py:20`), which is a pure UOp
rewrite over **device strings** — it computes lifetimes, runs a TLSF allocator
over integers, and emits `SHRINK`/`BITCAST` views. It never constructs a
`Device`, never allocates a buffer, never launches a kernel.

And `Context(DEV="CUDA")` makes `Device.DEFAULT` read `"CUDA"` without touching
hardware, because `Device.DEFAULT` is `DEV.device or self._select_device`
(`tinygrad/device.py:44`) — the context short-circuits device discovery
entirely. So:

* `Device.DEFAULT == "CUDA"` **is asserted** inside every census in the file;
* every `Tensor` in `kda_chunk` is built on the CUDA backend;
* no CUDA device is opened, because none is needed and none exists.

`kda/materialisation_verdict.md` §7 asked for exactly this — "run the M2
schedule census against `Device.DEFAULT == CUDA` at the bench shape and report
the histogram of kernel argument buffer sizes" — and said "one measurement, on
the box, no new hardware", while noting it "needs only `--quick`-scale GPU
time". **It turns out to need no GPU time and no hardware at all**, because the
question is about the *scheduler* and the scheduler is a few hundred lines of
Python over UOps. What genuinely cannot be done here, and is not claimed, is
rendering: `CUDADevice.__init__` calls `cuInit(0)`
(`tinygrad/runtime/ops_cuda.py:103`), and this host has no `libcuda` to load.
Verified, not assumed:

```
$ DEV=CUDA .venv/bin/python -c "from tinygrad import Device; Device[Device.DEFAULT]"
AttributeError: failed to load library cuda: try setting CUDA_PATH?
```

**Why this is the right instrument and not a shortcut.** M2 is defined
(`kda/materialisation_verdict.md` §1) as "does a kernel **argument** of that
size exist in the generated plan". A buffer carries traffic only if some kernel
reads or writes it, and a kernel can only touch buffers in its argument list.
The argument list is decided by the scheduler, not by the renderer and not by
the card. So the census is the instrument the question was defined in terms of,
run on the backend the question was about.

**The fresh-graph discipline.** `Tensor.realize()` on an already-realized graph
is a silent no-op, so a repeat measurement on the same tensors measures nothing.
This census does not realize at all: it intercepts `kda_chunk`'s two `.numpy()`
calls — the operator's only two realizes, at `kda/tiny_kda.py:252` — and
substitutes `schedule_linear()`. Every census is therefore a fresh graph
(`kda_chunk` takes numpy inputs and rebuilds on every call) driven through the
real scheduler, with no execution and therefore no state to go stale. The file
additionally runs the bench census twice in-process and once in a **cold
subprocess**, and requires the sink counts to agree, because a second call in the
same process is served by `schedule_cache` (`schedule/__init__.py:117`) and is
not an independent measurement of the scheduler.

---

## 2. VERDICT 1 — the decay tensor does not reach device memory. **`NO` BUCKET.**

`MEASURED`, whole schedule, CUDA backend, no GPU.

At the bench shape `kda/perf_kda.py:90` — B=2 T=512 H=HV=4 K=V=64 C=64, NT=8,
`B·NT·HV` = 64 — the fp32 `[B,NT,HV,C,C,K]` decay tensor is

```
B·NT·HV·C²·K·4  =  2·8·4·64·64·64·4  =  67,108,864 B  =  67.1089 MB  =  64.0000 MiB
```

and the census of every kernel argument in the whole schedule finds **four**
distinct sizes and **no bucket at 67,108,864 B**:

| kernel argument | launches | argument bytes at this size |
|---|---|---|
| 256 B (`[C]`, one `eyeT` row) | 20,096 | 5.1 MB |
| 16,384 B (`[B,NT,HV,C]`, `[C,C]`) | 24,604 | 403.1 MB |
| 131,072 B (`[B,HV,K,V]`, the live state `S`) | 111 | 14.5 MB |
| **1,048,576 B** (`B·NT·HV·C²·4`) | 8,191 | 8,588.9 MB |
| **total, 12,221 launches** | | **9,011,691,520 B** |

```
largest argument / decay tensor = 1,048,576 / 67,108,864 = 1/64 = 1/K
identity  B·NT·HV·C·C·4  =  1,048,576 B   ->  MATCH
```

`notes/09-materialisation-verdict.md` §4.3's identity holds on CUDA, and the
ratio is exactly K for a structural reason and not a numerical coincidence: both
sides carry `B·NT·HV·C²` and differ only in the trailing `·K`, so the gap is
independent of B, NT, HV and C. The bucket question
`kda/materialisation_verdict.md` §7 left open is **closed**.

Three further shapes, same answer, with the decisive bucket moving with the
shape so the question is not answered by a coincidence of arithmetic:

| shape | decay tensor fp32 | largest kernel argument | bucket present? |
|---|---|---|---|
| `build_notebooks.py:590` (B=2 T=128 H=HV=2 K=V=32 C=32) | 2,097,152 B | **65,536 B** | **NO** |
| bench (NT=8) | 67,108,864 B | **1,048,576 B** | **NO** |
| bench, NT doubled (NT=16) | 134,217,728 B | **2,097,152 B** | **NO** |
| `kda/cost.py:17` (NT=64, K=V=128) | 17,179,869,184 B | 134,217,728 B `[MODELLED]` | not schedulable here |

Each of the three measured rows is exactly `B·NT·HV·C²·4`, and the ratio to the
decay tensor is exactly K = 32 / 64 / 64. The last row is the identity's
prediction at `kda/cost.py`'s shape and is labelled as such: `notes/09` §6
records 288,406 live UOps at the bench shape and 462,345 on the second realize,
so that shape cannot be scheduled on this host.

**The first row is the important one, and it is a tie to real hardware output.**
`kernels/build_notebooks.py:590` is the shape `captured 1 kernels` ran at. The
one piece of genuinely emitted CUDA in this repo,
`out/kaggle/20260928-014248-gpu/codegen/kernel000.txt`, declares buffers of
4,096 / 4,096 / 16,384 / 16,384 / 4,096 elements — a largest buffer of
**65,536 B**. The device-free census's largest kernel argument at that shape is
**65,536 B**. They agree to the byte, on a host with no NVIDIA card, which is
the strongest available tie between the device-free census and the real
lowering. At that shape the decay is 32× the largest buffer and 11.6× the whole
buffer set, exactly as `kda/materialisation_verdict.md` §2.2 says, and it is
absent from both.

**This confirms `notes/09` §4.2 and `kda/materialisation_verdict.md` §1. It
contradicts neither.** What is new is that the CUDA row is now measured on the
*whole schedule* rather than inferred from one kernel's parameter list, and that
the identity is confirmed with the instrument matched to the artifact.

**What this verdict is not.** It is not a claim about DRAM traffic, which
remains unmeasured and must not be inferred from a byte count. It is a claim
about the program's structure, and that is all it takes to kill the 268.4 MB
model: a tensor that is in no kernel's argument list cannot be moved by any
cache on any card. It is also scoped to tinygrad 0.14.0's scheduler — see §5.1
for the one place that scope bites.

---

## 3. VERDICT 2 — the decay is recomputed, and it costs arithmetic, not bytes

`MEASURED` by the prior agent, `NOT RE-DERIVED` here, and one new negative
result.

The artifact is unchanged and its analysis stands: `kernel000.txt` has **128
`exp2`**, all in `float aluN` register locals, none stored and none re-read; four
stores, all to `data0_4096`; 33 additive terms per stored element. The decay
expression is evaluated inside the accumulation, once per output term. So the
tensor is not materialised *and* the work is not shared: it is recomputed. That
is `kda/materialisation_verdict.md` §2.3 and it is not re-litigated here.

**The new negative result: the `exp2` count cannot be obtained from the
schedule.** I tried, because a schedule-wide transcendental count would be the
right denominator for the 32x claim, and it does not exist at that level. The
static `EXP2` uop count in the entire CUDA schedule is:

```
CUDA   15 outside a REDUCE,  35 inside one      (across 12,082 kernels)
METAL  15 outside a REDUCE,  35 inside one
CPU    15 outside a REDUCE,  35 inside one
```

**50 uops is not an `exp2` count, and that is the finding.** 35 of the 50 are
inside a `REDUCE`, i.e. inside a loop whose trip count is C. The one emitted
kernel we have source for carries 128 `exp2` *per thread* over 256 threads =
32,768 evaluations, and the graph's entire contribution to that is **one** `EXP2`
uop inside a reduce whose trip count is C. A uop count is not an evaluation
count, and the multiplier lives in the renderer, not the scheduler. So:

* the dynamic `exp2` count is a **CODEGEN** property, which is exactly why the
  32x had to be counted in source and why it should stay counted there;
* **do not** use 50 as a denominator for anything, and do not read this as
  reporting a *smaller* `exp2` count than the record already has.

**What claim 2 costs, stated precisely so it stops being conflated with claim 1:**
SFU issue slots and registers. `exp2` is a transcendental, narrower-throughput
than FMA on every NVIDIA arch, and it is not traffic. Re-derived at the capture
shape, where the arithmetic is small enough to check by hand:

```
emitted   128 exp2 per thread, 4 float4 stores per thread
        =  32 exp2 per float4 group  ->  8 exp2 per stored output element
AST        1 EXP2 uop x a reduce trip of C=32
        =  32 exp2 per output element
the 4x gap is the float4: the exp2 does not depend on V, and the emitted code
SHARES one exp2 across four V columns, so it is 4x CHEAPER than the AST expansion
```

A decay tensor that *had* been materialised and loaded would replace **all 32**
of those with a load. The inflation is therefore not a modest constant -- it is
the entire transcendental, paid in issue slots. **It never appears in a
buffer.**

**Verdict 2: RECOMPUTED, paid in arithmetic, not in bytes. Unchanged from
`kda/materialisation_verdict.md` section 2.3, and now with its negative result
recorded.**

---

## 4. The 1-vs-12,082 reconciliation: it is a worker pool

This is the part the two adjacent notes both flagged and neither could close,
and it is not a subtle one.

`kernels/build_notebooks.py:578-597` wraps
`Device[DEFAULT].compiler.compile_cached` **in the calling process** and prints
`len(cap)`. It got **1**. Here is why, in tinygrad's own source
(`MEASURED` as a source read, `MEASURED` as a reproduction):

```
tinygrad/engine/realize.py:263-288   lower_and_compile
  265  collect every CALL that still needs lowering
  269  todo = {compile-cache-key: ast} for those not already compiled
  273  pool = None if len(todo) == 1 or any(beam) else get_worker_pool()
  278  for i, prg in (map if pool is None else pool.imap_unordered)(_compile_kernel, tasks)

tinygrad/engine/worker.py:37-44       get_worker_pool -> a multiprocessing
                                      SpawnContext Pool of PARALLEL processes
tinygrad/helpers.py:269               PARALLEL = NUM_CPU_THREADS by default
```

**When more than one kernel needs compiling, tinygrad compiles them in
subprocesses.** `compile_cached` then runs in a child process, where the
notebook never patched it. An in-process hook sees only the rounds where
`len(todo) == 1`, because that is the only condition that takes the `map`
branch and compiles in-process.

`kda_chunk` makes **four** `lower_and_compile` rounds at the bench shape, with
6,188 / 1 / 6,041 / 1 linear calls. The two 1-call rounds are copies with
nothing to compile. The two large rounds each have a `todo`, and their sizes
are the whole story: at `PARALLEL=0` the 6,188-round compiles **190** kernels
and the 6,041-round compiles **exactly 1**. With the pool on, the 190 go to
subprocesses and the 1 — being the only round where `len(todo) == 1` — is
compiled in the parent. That is the one kernel the notebook wrote to
`kernel000.txt`.

**Reproduced on this host, on Metal, at the notebook's exact shape, with five
counters on one `kda_chunk` call.** `PARALLEL=0` is tinygrad's own switch for
"no worker pool":

| shape | `PARALLEL` | launches | distinct programs | `to_program` | `compile_cached` | pool |
|---|---|---|---|---|---|---|
| `build_notebooks.py:590` | default | 2,978 | 97 | 1 | **1** | yes |
| `build_notebooks.py:590` | 0 | 2,978 | 97 | **97** | **97** | no |
| bench | default | 12,090 | 191 | 1 | **1** | yes |
| bench | 0 | 12,090 | 191 | **191** | **191** | no |

`compile_cached` fires **1** time with the pool on and **191** times with it off,
for the same 12,090 launches. That is the whole gap, and it is not a
CUDA-specific artefact: the pool lives in `engine/`, above the backend, so the
notebook's `captured 1 kernels` reproduces on a machine with no NVIDIA card in
it. The round-by-round breakdown confirms the mechanism rather than merely
correlating with it — at `PARALLEL=0` the bench shape shows

```
round 1:  6,188 linear calls, 190 compiled IN-PROCESS
round 2:        1 linear call,   0 compiled  (a copy)
round 3:  6,041 linear calls,   1 compiled IN-PROCESS  <- the one the notebook saw
round 4:        1 linear call,   0 compiled  (a copy)
```

With the pool on, round 1's 190 go to subprocesses; round 3 is the only round
where `len(todo) == 1`, so it takes the `map` branch and runs in the parent.
That is the one.

**So the reconciled counts are:**

| quantity | value | what it is |
|---|---|---|
| kernels the notebook captured | **1** | the one kernel compiled in the parent process |
| distinct compiled programs, bench shape | **191** | `to_program_cache` entries; the 12,090 launches reuse them |
| launches per call, Metal / CPU | **12,090** | `notes/03-kda.md:41-48`, `notes/04:416` |
| launches per call, CUDA | **12,082** | `notes/10-t4-results.md` §2, reproduced here |

**And `kernel000.txt` is not a sample of the operator — it is its last kernel.**
Its parameter list (`data0_4096` written, `data1_4096` read at exactly the
indices `data0_4096` is written, `data2/3_16384` = `B·NT·HV·C·K` at the capture
shape) and its 33-term store expression identify it as the inter-chunk state
update at `kda/tiny_kda.py:240-242`. `kda/materialisation_verdict.md` §2.2 got
that identification exactly right from the source alone. What has to change is
the *framing*: it is a correct analysis of the final kernel, and it was never
evidence about the schedule.

**The correct interception point for the CUDA backend**, if one wants a full
source census on a T4, in order of preference:

1. **`PARALLEL=0` + the existing `compile_cached` hook.** One environment
   variable. Yields every distinct program (191 at the bench shape), not every
   launch — dedup by `to_program_key` is correct behaviour, not a loss.
2. **Hook `tinygrad.engine.realize.to_program`**, which is the codegen entry and
   is what `lower_and_compile` calls in whichever process is doing the work.
3. **Count launches** at `Device[dev].runtime_t.__call__`, which is
   `CUDAProgram.__call__` on CUDA. This is the complete census: 12,082, and it
   is what `DEBUG=2` already reports.
4. **Parse `DEBUG=2`.** Works, `perf_kda.py` already does it, but it times with a
   per-launch host sync (§7.2) so the numbers need §7's caveat.
5. **Do not** hook `compile_cached` in the parent and believe the count. That is
   the bug.

**Scope, stated plainly.** The table above is `MEASURED` on Metal, on this host.
The mapping to CUDA is an **`INFERENCE`** from the source, and a narrow one:
`lower_and_compile`, `get_worker_pool` and `Compiler.compile_cached` all live
above the backend, and the notebook installs the identical hook on CUDA. What the
T4 run does **not** independently show is that `len(cap)` was 1 *for this
reason* — it shows that it was 1, and the worker pool is the only mechanism in
the installed source that can produce 1 out of thousands. The one-line check on
the box: re-run `build_notebooks.py`'s cell with `PARALLEL=0`; the count must
become 191.

---

## 5. VERDICT 3, part one — the 12,082, reproduced with no GPU, and explained

`MEASURED`, on the CUDA backend, from the scheduler, on a laptop.

The census's SINK-kernel count at the bench shape is **12,082**. That is the
T4's number from `out/kaggle/20260928-014248-gpu/logs/perf_kda.log:31`, to the
unit, from a host with no GPU. The correspondence is exact and it is
mechanical, not lucky:

```
SINK kernels  12,082
COPY kernels     139        <-- kda/perf_kda.py:52 filters `" copy "` out of DEBUG=2
total LINEAR   12,221
```

and the `r_`/`E_` split reproduces too, because `perf_kda.py` reads those off the
kernel name, which tinygrad builds from `self.reduceop is not None`
(`codegen/opt/postrange.py:53`) — i.e. from whether the kernel carries a
`REDUCE`, which is a property of the AST and therefore countable at schedule time:

| | reduce (`r_`) | elementwise (`E_`) | total |
|---|---|---|---|
| device-free census, `DEV=CUDA` | **7,984** | **4,098** | **12,082** |
| T4, `DEBUG=2`, same shape | **7,984** | **4,098** | **12,082** |

> **Correcting a number in the brief.** The `7,984 r_ / 4,106 E_` split is the
> **Metal** split — `notes/03-kda.md:45-46` and `notes/04:418-419`, summing to
> 12,090. The CUDA split is `7,984 / 4,098`, from `perf_kda.log:34-35`, summing
> to 12,082. The brief attached the Metal numbers to the CUDA count.

### 5.1 Why CUDA says 12,082 and Metal says 12,090

They differ by **8 kernels**, and the cause is a hard-coded table:

```python
# tinygrad/schedule/rangeify.py:354, installed 0.14.0
DEVICE_MAX_BUFS = {"METAL": 31, "WEBGPU": 8, "CPU": 31}  # TODO: get from device?
```

`rangeify.py:360-382`'s `_limit_bufs` force-bufferizes a kernel's elementwise
inputs when its AST touches more than `MAX_BUFS-1` buffers, which **splits it
into more kernels**. Metal and CPUs have a 31-buffer kernel-argument limit; CUDA
allows 64. tinygrad's table has **no CUDA entry**, so the lookup at line 363
misses, returns 0, and the function returns `None` — the pass is simply off.
Nobody has filled in the entry, and the source says so in a `# TODO`.

A control that toggles the one entry. Each row is its own process, for the
reason in the note under the table:

| `DEVICE_MAX_BUFS` | SINK | `E_` | rewrites |
|---|---|---|---|
| as shipped, no CUDA entry | **12,082** | 4,098 | 0 |
| `['CUDA'] = 31` | **12,090** | 4,106 | **4** |
| `['CUDA'] = 100` | **12,082** | 4,098 | 0 |
| as shipped, METAL | 12,090 | 4,106 | 4 |
| as shipped, CPU | 12,090 | 4,106 | 4 |

**A positive control in both directions.** Pinning CUDA to 31 reproduces Metal's
count exactly — *and reproduces its `r_`/`E_` split exactly*, because
`_limit_bufs` splits the **elementwise** part of a kernel and the 8 extra Metal
kernels are all elementwise. Pinning it to 100 reproduces CUDA's. Four roots
rewrite and the whole 8-kernel difference is accounted for.

**A methodological note that cost this section a rewrite and is worth keeping.**
Each row above runs in its **own subprocess**, and that is not fussiness.
tinygrad's rewrite passes memoize on UOp identity, so a second run in the same
process re-plays the cached rewrite and `_limit_bufs` never fires. The first
version of this control ran in-process and cheerfully reported **0 rewrites and
12,082 for every configuration, including METAL, which is knowably 4 and
12,090**. A control that cannot fail is not a control. It is the same trap as
`Tensor.realize()` on a realized graph, one level down.

**Why this matters beyond the curiosity.** It is the reason the device-free census
is not an approximation of the CUDA schedule: the device **name** is the only
device-dependent input to `get_kernel_graph`/`memory_plan_rewrite` in this
version, we have found it, and we can toggle it. `notes/09` §4.5 said "two
backends agreeing is not three". That caution was right, and the answer is
sharper than agreement: **the schedules are not the same, the difference is a
constant 8 kernels, and it is attributable to a named line of tinygrad.** They
agree exactly on `max_param` and on the identity, so verdict 1 is unaffected.

---

## 6. VERDICT 3, part two — the launch count is a constant of C, not of T

`MEASURED`, device-free. This is the strongest statement available about claim 3
without a card, and it cuts both ways.

C=64 held, NT swept, `DEV=CUDA`:

| T | NT | `B·NT·HV` | SINK | REDUCE | ELEM | trip | arg MB |
|---|---|---|---|---|---|---|---|
| 128 | 2 | 16 | 12,052 | 7,954 | **4,098** | 1,036,092 | 2,335 |
| 256 | 4 | 32 | 12,062 | 7,964 | **4,098** | 3,013,020 | 4,549 |
| 512 | 8 | 64 | 12,082 | 7,984 | **4,098** | 6,966,876 | 9,012 |
| 1024 | 16 | 128 | 12,122 | 8,024 | **4,098** | 14,874,588 | 18,076 |
| 2048 | 32 | 256 | 12,202 | 8,104 | **4,098** | 30,690,012 | 36,757 |

```
elementwise kernels   : CONSTANT at 4,098 across a 16x span
reduce kernels        : 7,954 -> 8,104, a slope of exactly 5 per chunk
SINK(NT)              ~ 12,042 + 5*NT
```

The constant is the unrolled forward substitution at `kda/tiny_kda.py:178-194`:
`for i in range(C)` containing `for r in range(1, C)`, i.e. `C(C-1) = 4,032`
iterations, each producing a reduce (`s = (A[...,r,:r] * b[...,:r]).sum(-1)`) and
an elementwise (`b = b - e_r * s`). It is a function of **C alone** and does not
move with T. The 5-per-chunk slope is the inter-chunk scan.

**Consequences, and the second one is the uncomfortable one:**

* The launch count is real and does not amortise — ~12,000 independent
  synchronisation points per call.
* **It is not a function of the problem size.** Over a 16× span the work grows
  30× in trip count and 16× in argument bytes while the launch count grows
  **1.2%**. So as the study's target shape grows, the schedule's launch cost
  becomes a *smaller* fraction of runtime, not a larger one. Any claim of the
  form "the unrolled schedule costs a factor N" has to be stated at a stated
  shape, and it gets weaker as the shape grows. At `kda/cost.py:17`'s shape
  (`NT=64`, `C=64`) the same structure predicts **12,362** launches
  `[MODELLED — never scheduled. `notes/09` §6 measures 288,406 live UOps at the
  bench shape and 462,345 on the second realize, so ×128 is ~37M UOps before a
  single kernel launches; that is why this note's census stops at NT=32]`.

---

## 7. Is 12,082 launches a real cost, or launch overhead? **NOT SETTLED — and labelled.**

Everything in this section is either a **source read** or an **`INFERENCE`**.
There is no GPU here. I am not going to present an inference as a measurement,
and this section is deliberately the weakest in the note.

### 7.1 The T4 numbers, parsed out of the log

`out/kaggle/20260928-014248-gpu/logs/perf_kda.log`, `device=CUDA`, bench shape:

```
launches per call   12,082
GPU time            215,880 us = 215.9 ms
per launch             17.868 us
  r_   7,984 launches  159,658 us   20.00 us/launch   74.0%
  E_   4,098 launches   56,222 us   13.72 us/launch   26.0%
launches per second  55,966
```

Internal checks, recomputed: `7,984 + 4,098 = 12,082`; `159,658 + 56,222 =
215,880`; `159,658/215,880 = 73.96%`. All exact.

### 7.2 The measurement is upper-bounded by its own instrument. `SOURCE`.

`kda/perf_kda.py:43-49` runs the graph in a subprocess with `DEBUG=2`, and
`engine/realize.py:318` sets `wait=... or DEBUG>=2`, so **every single launch**
goes through `runtime/ops_cuda.py:26-35`, `cu_time_execution`:

```
28-29  cuEventCreate x2                 per launch, per kernel
30-31  cuEventRecord around cuLaunchKernel on the null stream
32-33  cuEventSynchronize               <-- the HOST BLOCKS, per launch
34-35  cuEventElapsedTime, then cuEventDestroy x2
```

`event1 − event0` therefore contains the **host launch latency** as well as the
kernel: the queue is empty when event0 is recorded, so the GPU timestamps event0
and then waits for the host to issue the launch. **215.9 ms is 12,082
event-bracketed, host-synchronised launches.** It is an upper bound on GPU work,
not a measurement of it. The "17.9 µs per launch" figure the brief flags as
suspicious is suspicious for a reason that is in tinygrad's source, not in the
kernels.

### 7.3 How much work the schedule actually asks for. `MEASURED`, device-free.

```
trip count, sum over the 12,082 kernels of the product of their RANGE extents =  6,966,876
static ALU uops in those bodies (addressing excluded)                        =     45,296
LANE-OPS: uops x trip, reduce axis discounted                                =  57,764,876
argument bytes, all launches                                                 = 9,011,691,520
```

577 iterations per kernel on average, and 12,082 of them. These are not fat
kernels. They are 12,082 very small ones, and that is the shape of the problem.

Two independent bounds:

**(a) Bandwidth.** If *every* argument byte missed *every* cache and went to
DRAM — 9,012 MB of them — then at the T4's 320 GB/s (datasheet, **not measured
here**) that is **28.2 ms**, at most **13%** of 215.9 ms. The operator is **not
bandwidth bound**. This bounds `notes/09` §5's "largest unresolved quantity,
schedule argument bytes" from above, which is the first thing anyone has done
to it. (Its figure, 9,028,599,808 B, is the **Metal** number and reproduces
exactly; CUDA's is 16.9 MB smaller for the same 8-kernel reason as §5.1.)

**(b) Arithmetic.** 57,764,876 lane-ops at the AST level. A T4 retires at most
`40 SM x 128 lanes x ~1.4 GHz = 7.2e12` lane-ops/s (arithmetic on the published
40-SM / 2048-core T4 spec, **`INFERRED`**), so at 100% issue the schedule's
arithmetic is **8 us**.

This is an AST-level count and the emitted program can differ from it in
**either** direction, so both signs are checked rather than assumed:

* **The AST can be HIGH.** For the one kernel with emitted source in this repo,
  the AST expansion asks for 32 `exp2` per output element and the emitted
  `float4` kernel does 8, because the renderer shares the V-independent `exp2`
  across four V columns. There the AST count is 4x too big.
* **The AST can be LOW.** A reduction's body can be re-entered per output
  instead of hoisted, which is the mechanism `notes/04`'s 32x describes at the
  graph-object level. I have emitted source for exactly one of the 12,082
  kernels and can bound neither sign, so I will not pretend to.

**What survives both: 1000x in either direction is 8 ms, under 4% of 215.9 ms.**
The schedule's arithmetic cannot account for the measured time whichever way
the AST-to-source correction goes, and that is the only claim I need from it.

### 7.4 So what is the 215.9 ms? `INFERENCE`, and it is not settled.

Bandwidth bounded at 28.2 ms, the schedule's own arithmetic at 8 µs, measured
total 215.9 ms over 12,082 launches at 17.87 µs each. A T4 kernel launch costs
single-digit microseconds of host time, and the measured figure is an
event-bracketed, host-synchronised one. The residual is per-launch cost and
measurement overhead, not arithmetic. The one other T4 number points the same
way: `logs/replay_kda.log:12` measures **14,990 ms of schedule + launch +
execute** per warm call, of which the 215.9 ms is **1.44%**.

**But this is an inference and I am labelling it as one.** The bounds above
exclude two *specific* bottlenecks; they do not exclude the kernels being
genuinely long-running for a reason the AST does not show (a pathological
access pattern, uncoalesced loads, occupancy collapse at these grid sizes), and
they do not exclude the DEBUG=2 event wrapper being most of the 17.87 µs. The
honest summary is: *the schedule's own arithmetic cannot account for 215.9 ms,
and neither can its bytes; the remainder is per-launch or per-instrument, and
which one is not determined.*

**The measurement that settles it, cheapest first:**

1. **Three lines, no new tool.** Re-run `perf_kda` with `DEBUG=0` and time the
   `realize()` with a **single** `Device["CUDA"].synchronize()` at the end, on a
   warm graph (second call, so compilation is out of it). If the wall time of
   the realize is ~0.2 s the kernels are launch-bound; if it is ~15 s then the
   215.9 ms is mostly the `cu_time_execution` wrapper. This is the whole
   experiment.
2. **`ncu --metrics gpu__time_duration.sum`** over one call.
   `gpu__time_duration` excludes launch latency by construction and is the
   number that would go in the record.
3. **One CUDA graph** replaying the same 12,082 kernels. If it replays in
   « 215.9 ms, the 215.9 ms was launch cost, and the study's actionable result
   is that caching the *scheduled program list* — not just the compiled kernels
   — is what a replay path has to do.

`logs/replay_kda.log:21`'s "~109 ms" is a hard-coded Mac constant, flagged
`[STALE]` in `notes/10` §2.2, and is **not** used anywhere above.

---

## 8. VERDICT 3, restated

> **The operator is not fused at all: 12,082 kernels per `kda_chunk` call on
> CUDA, 12,090 on Metal and CPU, reproducible without a GPU, with a launch count
> that is a constant of `C` rather than of `T`. Whether that is a material cost,
> and what fraction of runtime it is, is NOT SETTLED. The schedule's own
> arithmetic (8 µs) and its own bytes (≤ 28.2 ms) cannot account for the T4's
> 215.9 ms, so the residual is not arithmetic; whether it is per-launch or
> per-instrument is undetermined. Three measurements are named, the cheapest
> being three lines of code on the box that already has the numbers.**

**Quote the count. Do not yet quote a cost.**

---

## 9. Byte figures re-derived, so nothing here is transcribed

`kda/materialisation_verdict.md` §7 was right to insist on re-derivation rather
than verification. `kda/schedule_census_cuda.py` §2b recomputes:

**(a) The bench shape, fp32** — the decisive size, and the numerator of the
withdrawn model:

```
B·NT·HV       = 2·8·4                        =        64
[C,C,K] fp32  = 64·64·64·4                   =  1,048,576 B
one tensor    = 67,108,864 B = 67.1089 MB = 64.0000 MiB  (exactly)
two tensors   = 134,217,728 B
x2 touches    = 268,435,456 B = 268.4355 MB  <- the withdrawn 268.4 MB, exact
largest kernel argument (census) = 1,048,576 B
ratio         = 67,108,864 / 1,048,576 = 64 = K
```

The numerator re-derives exactly. The **denominator** does not, and is not
re-attempted here: `notes/09` §2's 22.2822 MB / 5.5050 MB / 48.7619× / 13.0471×
has now failed to reproduce in two independent attempts, the second being
`kda/materialisation_verdict.md` §3.3's 5,242,880 B and 51.20×. A third attempt
would only produce a third number. **The numerator is exact, the denominator is
unattributable, and every ratio built on it is withdrawn.**

**(b) `kda/NOTES.md`'s shape, fp32** — where "8.2 GB each" is wrong:

```
B=4 T=4096 H=HV=32 K=V=128 C=64 NT=64
B·NT·HV      = 8,192          C·C·K = 524,288
elements     = 4,294,967,296 = 2^32
fp32  one    = 17,179,869,184 B = 16.00 GiB = 17.18 GB
      w + r  = 34,359,738,368 B            = 34.36 GB
bf16  one    =  8,589,934,592 B =  8.00 GiB =  8.59 GB
      w + r  = 17,179,869,184 B            = 17.18 GB
```

`notes/03-kda.md:212` and `kda/NOTES.md:153,162` say **8.2 GB each** and 16.4 GB
for a write + a read. **8.2 GB matches neither dtype** — bf16 is 8.59 GB and
fp32 is 17.18 GB — and the conclusion it supports ("does not fit in this
machine's 32 GB") only holds for the fp32 figure, since 2 × 8.2 GB = 16.4 GB
*would* fit. Corrected: **17.18 GB (16.00 GiB) each in fp32, 34.36 GB for a
write + a read.** `notes/07-crossover-thesis.md:296` and `notes/09` §3.1 already
have this right. Both files are owned elsewhere; flagged, not edited. **Not
carried forward in any form.**

**(c) Nothing else in this note is a byte claim.** §7.3(a) is a *bound*, computed
in code from the census, and is labelled as resting on a datasheet bandwidth
rather than a measurement.

---

## 10. What is safe to say now, and what is not

### Safe

* **The `[B,NT,HV,C,C,K]` decay tensor is never materialised, on the whole CUDA
  schedule, at three measured shapes, with no bucket at the tensor's size at any
  of them.** `MEASURED`, CUDA backend, no GPU. Closes the item
  `kda/materialisation_verdict.md` §7 and `notes/10` §9.2 both named as the
  missing measurement.
* **The identity `largest kernel argument = B·NT·HV·C²·4` holds on CUDA**, and
  the ratio to the decay tensor is exactly K = 64 at the bench shape.
* **At `build_notebooks.py:590`'s shape the census's largest kernel argument is
  65,536 B, equal to the largest buffer the real emitted CUDA declares, to the
  byte.**
* **The decay is recomputed, not reused**, and the cost is arithmetic (SFU issue
  slots), not bytes. Unchanged from `kda/materialisation_verdict.md` §2.3.
* **The operator is not fused: 12,082 CUDA / 12,090 Metal+CPU launches per call**,
  reproduced without a GPU, and the `r_`/`E_` split reproduces exactly too.
* **`captured 1 kernels` is a worker-pool artefact, not a statement about the
  operator.** The fix is `PARALLEL=0`.
* **CUDA's and Metal's schedules are not identical and the 8-kernel difference is
  attributable to `schedule/rangeify.py:354`'s `DEVICE_MAX_BUFS` table having no
  CUDA entry.**
* **The launch count is `≈ 12,042 + 5·NT` — a constant of `C`, not of `T`.**

### Not safe

* **Any fraction-of-runtime figure for the schedule.** §8 is the open claim and
  §7 does not close it. "12,082 kernels is 17.9 µs each" is an **upper bound
  from an instrument that blocks the host per launch**, not a per-kernel cost.
* **"TinyGrad is N× slower because of 12,082 kernels."** `notes/10` §2.1 already
  measured that the 19.5 s warm call is ~23% Python and **~76% TinyGrad's
  rewrite passes**, and §6 above shows the launch count does not grow with the
  problem. Both halves of that sentence have to travel with the claim.
* **`4,106 E_` for CUDA.** That is the Metal split. CUDA is `4,098 E_`. (The
  brief quoted the Metal numbers.)
* **`8.2 GB`, `22.2822 MB`, `5.5050 MB`, `48.7619×`, `13.0471×`, `31.9×`, `4.02
  MiB`.** All unreproduced or corrected in §9. None is used above.
* **"The decay dataflow is the bottleneck" as a bandwidth claim.** Still
  withdrawn as bandwidth; §7.3(a) now bounds the bandwidth share of this
  operator at ≤ 13% of the T4's GPU time. It survives only as an arithmetic
  claim.
* **Any claim about DRAM traffic, or about whether the 8,191 re-reads of the
  1,048,576 B buffer hit L2.** Still unmeasured. §7.3(a) bounds it; it does not
  measure it.
* **Anything about tinygrad's CUDA *codegen* from this note.** The census measures
  the scheduler. It says nothing about what `CUDARenderer` emits for the other
  12,081 kernels. `PARALLEL=0` on the box, plus the census's `max_param` and the
  identity, is the whole of what transfers.

### Relationship to the two adjacent documents

**`notes/09-materialisation-verdict.md` and `notes/10-t4-results.md` are not
contradicted.** Both are confirmed and extended:

* `notes/09` §4.2's M2 row is reproduced for METAL and CPU (12,229 linear calls,
  12,090 SINK + 139 COPY, four argument sizes, largest 1,048,576 B) and now also
  produced for **CUDA** (12,221 linear calls, 12,082 SINK + 139 COPY, the same
  four sizes, largest 1,048,576 B). §4.5's "two backends agreeing is not three"
  was the right caution; §5.1 above is the sharper answer.
* `notes/10` §2's 12,082 / 215.9 ms / 7,984 r_ / 4,098 E_ is reproduced exactly,
  from the CUDA scheduler, on a host with no GPU. §2.1's "it is mostly SCHEDULER"
  is untouched and is reinforced by §6.
* `notes/10` §6's caveat — "`captured 1 kernels` is not a census" — is **the
  correct call**, and §4 above is the closure it was asking for. Its warning that
  `kernel000.txt` "is one kernel, correctly analysed, but not the operator's full
  schedule" is confirmed, and sharpened: it is specifically the operator's *last*
  kernel, and it is last by accident of which round happens to have one kernel
  outstanding.

**One correction to `kda/materialisation_verdict.md`**, which is the
authoritative document for the decay question and wins where they disagree: its
§7 correctly identified the measurement and its acceptance test, and correctly
noted it "needs only `--quick`-scale GPU time". **In fact it needs no GPU time
at all.** The scheduler is pure Python over UOps and `Context(DEV=...)` names
the backend without instantiating it, so no container and no CUDA toolchain are
required. Its §1 row for M2 stands, now measured on the whole schedule rather
than on one kernel's parameter list. Its §7's own acceptance test — *is there a
bucket at 67,108,864 B?* — is the one this note answers, and the answer is
**no**.

---

## 11. Files

| file | what it produced here |
|---|---|
| `kda/schedule_census_cuda.py` | everything in this note. The device-free M2 census on the CUDA backend, the `DEVICE_MAX_BUFS` control, the worker-pool reproduction, the NT sweep, the byte re-derivations |
| `out/kaggle/20260928-014248-gpu/logs/perf_kda.log` | the T4's 12,082 launches / 215.9 ms / 7,984 r_ / 4,098 E_, parsed not transcribed |
| `out/kaggle/20260928-014248-gpu/logs/replay_kda.log` | 14,990 ms of schedule+launch+execute per warm call |
| `out/kaggle/20260928-014248-gpu/codegen/kernel000.txt` | the 65,536 B largest emitted buffer, matched to the census at its own shape |
| `.venv/.../tinygrad/schedule/rangeify.py:354` | `DEVICE_MAX_BUFS`, the whole 8-kernel Metal/CUDA difference |
| `.venv/.../tinygrad/engine/realize.py:263-288, :318` | `lower_and_compile`'s worker-pool branch, and the per-launch `wait` |
| `.venv/.../tinygrad/engine/worker.py:37-44` | the pool itself |
| `.venv/.../tinygrad/runtime/ops_cuda.py:26-35, :103` | `cu_time_execution`'s per-launch event + sync; `cuInit(0)` in `CUDADevice.__init__` |
| `.venv/.../tinygrad/codegen/opt/postrange.py:53` | where `r_`/`E_` comes from, which is why the split is countable without a GPU |
| `kda/tiny_kda.py:44, :178-194, :240-242, :252` | the fp32 dtype, the unrolled UT loop, the state update, the two realizes |
| `kda/perf_kda.py:43-49, :52, :90` | the `DEBUG=2` subprocess, the `copy` filter, the bench shape |
| `kernels/build_notebooks.py:578-597` | the capture that got 1, and the shape it captured at |
| `kda/materialisation_verdict.md` §1, §2b, §7 | the M2 definition, the 128-`exp2` analysis, and the measurement this file ran. Its "needs only `--quick`-scale GPU time" framing is superseded: no GPU time is needed |
| `notes/09-materialisation-verdict.md` §4.2-4.5 | the METAL/CPU M2 row, reproduced, and the backend-agreement caution, answered |
| `notes/10-t4-results.md` §2, §6 | the 12,082, reproduced exactly, and the `captured 1 kernels` caveat, closed |
| `notes/03-kda.md:41-48, :212` | 12,090 / 7,984 / **4,106** (the Metal split), and the wrong 8.2 GB |
