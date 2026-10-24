# Decision log

Newest first. One row per decision, with the evidence that settled it.

| # | date | decision | why | evidence |
|---|---|---|---|---|
| 1 | 2026-09-27 | Target `Infatoshi/netherite`, not `mine.cu` | it is the one whose CUDA layer is a real batched RL env (7,200 lockstep worlds); `mine.cu` is a toy voxel grid | repo structure, `blaze/env/blaze_cuda.cu` 2,534 lines exporting the same C ABI as `blaze_cpu.c` |
| 2 | 2026-09-27 | Sparse-clone `blaze/` + `docs/` only | the repo is 156 MB, of which ~600 MB of `verify/` fixtures we never read | `--filter=blob:none --sparse`, 41 MB checked out |
| 3 | 2026-09-27 | Semantic equivalence, not bit-exact | netherite's value proposition is CPU==CUDA bitwise; TinyGrad cannot deliver that (Finding 1), so chasing it wastes the study | 12/41,472 pixels off by depth ±1, all from FMA contraction |
| 4 | 2026-09-27 | Run on M1 Pro / Metal; take CUDA numbers from netherite's published benchmarks | this Mac has no NVIDIA and netherite itself ships no CUDA path for macOS | `AGENTS.md` platform table; our own measurements are Metal |
| 5 | 2026-09-27 | Start with `obs_camera` as probe 1 | only kernel in the repo that is (a) self-contained, (b) pure, (c) has both a C reference and a bit-verified CUDA driver on this Mac | `blaze/cuda/obs_camera.cu`, `blaze/core/obs_camera.h` |
| 6 | 2026-09-27 | Build a **third** implementation (NumPy) alongside C and TinyGrad | the first two bugs we hit were *shared* transcription errors, so a 2-way diff would have blamed the wrong layer | `np_oc.py` is bit-exact with C; two of four bugs were in code both Python ports copied |
| 7 | 2026-09-27 | Treat the DDA as a fixed 96-step unroll | the trip count is data-dependent, but analytically bounded by `L*sqrt(3)+3 = 86.1` at `OC_FAR=48` | measured max step 84; `--check-steps` asserts it |
| 8 | 2026-09-27 | Keep the region index in **int32**, never float32 | float32 has a 24-bit mantissa; a flat index over 32 envs exceeds 2^24 at B≥16 and silently loses precision | `tiny_oc.py` casts to `dtypes.int32` before any index arithmetic |
| 9 | 2026-09-27 | Batch the ray setup over envs instead of looping in Python | a per-env Python loop multiplies the LUT gathers and frame math by B | `ray_dirs_batched`, pose terms `[B,1]` x pixel terms `[1,2304]` |
| 10 | 2026-09-27 | Report GPU time and wall time separately, always | they differ by 24x here (40 ms vs 943 ms) and conflating them gives the wrong answer in both directions | `DEBUG=2` kernel stats vs `bench_ms`; wall is flat in B, GPU is not |
| 11 | 2026-09-27 | Try `Tensor.custom_kernel` as a plugin route for the indexed load | documented extension point; would have made the existing port work unchanged | built it (`probes/real_gather.py`): does not compile. Coalescer asserts `multiple stores`; `DMC=1` then emits illegal MSL pointer arithmetic. The blocker is TinyGrad's memory model wanting affine addresses |
| 12 | 2026-09-27 | Do not dodge the division bug with a reciprocal-multiply rewrite | looked like a free fix | measured 100% different from the exact reciprocal — it still contains a division plus a second rounding. No division-free exact reciprocal exists in float32 |
| 13 | 2026-09-27 | Treat the two precision findings as upstream bug reports, not workarounds | the decision stands; the stated reason is **amended twice**. FMA is (i) for the caller — there is no knob. The division defect is **not a backend defect**: it is created once in the Tensor API and only *manifests* differently per backend, so "both are backend defects" files the bug at the wrong level. **Second amendment:** the figure this row cited for the division — 12.4–15.4% — was **not a measurement of division at all**. `probes/fp32_correctness.py:57` read `b = regimes[name] if op in "*/" else a`, and `regimes[name]` *is* `a`: the row computed `a/a`. The probe never divided by an independent divisor, and its number was copied into ~58 locations across 19 files without one | `+ - *` exact over **18,874,368** op instances (3 ops × 3 magnitude regimes × 2,097,152); `/` wrong in **25.9%–27.0%** of general `a/b` cases on a divisor-independent draw, always 1 ULP, numpy exact, CPU 0/200,000. **The citable form is a band with its divisor law attached, not a scalar** — `derivation/divide.py:mechanism_sentence` and `:published_rate_table`; the 12.4–15.4% band is the `a/a` rate and must be labelled as such wherever it survives. Mechanism: double rounding in the reciprocal, **not** an approximation (`derivation/divide.py:reciprocal_sensitivity`: a reciprocal 1 ULP worse predicts ~86%, not ~27%). Ledger **C-05**, **C-09**, **C-10**; detail in `02-obs-camera-residuals.md` and `12-divide-finding-correction.md`. NVIDIA: measured, see #19 |
| 14 | 2026-09-27 | Write the KDA port's UT transform as an unrolled forward substitution, not `inverse()` | TinyGrad has `qr` and `svd` but no triangular solve and no general `inverse`; unrolling is also what the real kernel does | `dir(Tensor)` has no `inverse`, no `solve`; trip count C is a compile-time constant so unrolling is legitimate. Matches the recurrence to 6.6e-6 |
| 15 | 2026-09-27 | Mask the decay by biasing the exponent, not by multiplying `exp` by 0 | masking after the exp still evaluates exp on out-of-band entries and builds a different, much larger graph | `Dl` relerr 3.0e-8 vs the reference's masked form; the alternative is `(diff + (mask-1)*1e4).exp()` |
| 16 | 2026-09-27 | Report the KDA wall time as ~unusable and say why | 99% of it is Python graph construction, which an RL loop does not pay per step | `perf_kda.py`: 9,842 ms wall, 109 ms GPU, 12,090 kernels. Not a claim about the operator |
| 17 | 2026-09-27 | Rank the compiler-hardness findings by what actually dominated, not by the pre-registered order | the ranking in NOTES.md was written before any TinyGrad code, and the top item was wrong | the `[C,C,K]` decay intermediates beat both sequential loops: 76.8% of GPU time is in the reduce kernels that consume them (`kda/perf_kda.py`, `DEBUG=2`). Control flow is recoverable by unrolling; a bad intermediate is not. **The byte framing in this row was wrong twice, so the three quantities in circulation are now kept apart:** "33.6 MB × 2" is `B·NT·HV·C²·K` × **2** bytes — the bf16 size — for an **fp32** port, where it is 67.1 MB each; and the tensor is **never materialised** (largest allocation the port makes is 8.42 MB, the `exp` is fused into the contraction — `notes/04-codegen-comparison.md` §2, which names this row). * **256×** = **element-count** ratio, `C²K/(BC·K) = 262,144/1,024`, for *one* `(b, chunk, head)` slab — a count, not a time and not a byte count `[MODELLED]`. * **48.8×** = **compulsory byte traffic** ratio, decay ÷ compulsory = `268.4 / 5.51 MB` — *not* `290.7 / 5.5`, which is 52.8× `[MODELLED]`, and it holds only if the decay reaches DRAM. * **13.0×** = `290.7 / 22.3`, the **predicted time** payoff of the exp-fold, which is a **lower bound** because the folded form reuses the ported shapes `[PREDICTED]`, and whose bandwidth-bound premise is unmeasured. Ledger **C-01 / C-03** |
| 18 | 2026-09-27 | Record the T4 run as a **lost run**, and state plainly that the study has **no measured GPU data yet** | a lost run is a real result about the pipeline, and this one is worth keeping because the pipeline threw away a measurement it had everything for: the hardware, the probe and the quota were all present. It was lost to a packaging race, not to a missing capability — which means it is one re-run away, and it must not be mistaken for a null result | `out/kaggle/gpu-v1/SUMMARY.md` — Tesla T4, 15360 MiB, compute capability 7.5; `payload sha256: n/a`; **all 15** battery items `skip` (`divide_precision \| skip \| 0.0 \| payload has no probes/divide_precision.py`); 40 s of battery wall time. Cause: `out/kaggle/20260927-231908-gpu/dataset-version.log` ends at "Dataset version is being created" — the kernel was pushed before the payload dataset version existed, so it ran against no payload. The payload itself was sound (26 files, `kernels/payload/payload.sha256` verifies against `payload.tgz`, and `probes/divide_precision.py` is in it), which is what makes this a race and not a build bug. Quota at the time: `out/kaggle/20260927-231908-gpu/run.log:20` — `GPU 0.00h used, 30.00h remaining`, refreshes 2026-10-03. Ledger §4 |
| 19 | 2026-09-27 | Expect the float32 division defect to **reproduce on NVIDIA at a comparable rate**, not to vanish — and mark it `[PREDICTED]` | the superseded inference chained a compiler flag to an outcome: "tinygrad's CUDA backend compiles through NVRTC without `--use_fast_math`, so `1.0f/b` becomes `div.rn.f32`, therefore NVIDIA should not reproduce the error." The flag is real and the conclusion does not follow, because **the error is not in the division** — it is introduced by tinygrad's own Tensor-level `a*(1/b)` rewrite, already present in the emitted source before NVRTC runs, so no compiler flag can undo it. The measured control is the **PYTHON** backend: a correctly-rounded reciprocal followed by a float32 multiply. **RESOLVED: the prediction held, on a T4.** `out/kaggle/20260928-014248-gpu/logs/divide_precision.log` — CUDA 54,270 / 200,000 wrong (27.13%), **bit-identical** to `a*fl32(1/b)`; `a/a` 30,748 / 200,000. So P-1 is refuted, P-1a is confirmed, and the defect is **not** Metal/CL-specific. Two clauses of this row are now corrected: the trailing prediction of "a second rounding everywhere, **plus an approximation on Metal and CL**" is **withdrawn** — there is no approximation anywhere, all four backends are double-rounded in one place (`derivation/divide.py:reciprocal_sensitivity`); and the "4 of 5 magnitude regimes" bit-identity claim is **retracted as unreproducible** from the current probe | `runtime/support/compiler_cuda.py:48,51,53` — verified: `NVRTCCompiler` sets only `--gpu-architecture`, `-I` includes and `--minimal`; `use_fast_math` appears **zero** times in the installed package and `prec-div` zero times; `renderer/cstyle.py:425` renders `RECIPROCAL` as `(1/{x})` for fp32, so NVRTC receives `a * (1.0f/b)`; `runtime/ops_cuda.py:121` renderer order `[CUDARenderer, PTXRenderer, NVCCRenderer]`. **No result may be quoted as backend-independent without naming the device and host** — the same tinygrad `CL` backend measures 27.13% on one host and 29.76% on the T4 (`derivation/divide_backends.py`). Ledger **P-1a** (confirmed), **C-04** (scoping retracted) |

## Results vs bugs

Worth stating plainly, because "11 bugs" reads like "the results are garbage".
It is not the reverse. Every bug below was found and fixed **during
development**. The final state is verified:

| artefact | result |
|---|---|
| semantic camera, block ids | **0 mismatches** on 41,472 pixels vs the C reference |
| semantic camera, depth | 9/41,472 off by +-1 (float32 rounding + LUT quantisation) |
| KDA output and state | **6.6e-06** relative vs the float64 per-token recurrence, 6/6 tests |
| KDA reference itself | 5.15e-15 vs the recurrence, 17/17 tests |

A bug found while writing a port is a normal event. The point of listing them
is that **most of them produced finite, plausible output rather than an error**,
so they were only caught by an independently validated reference — which is why
the three-implementation rule is in the conventions.

## Claims I got wrong, and what corrected them

Kept because being wrong in a *measured* way is more useful than being vague.

| wrong claim | correction | how |
|---|---|---|
| "the gather lowers to a one-hot reduction" | it is a real indexed load; above ~256 K source elements a *second, reduce-shaped* kernel appears, and its cost is flat in the number of elements requested | `probes/gather_floor.py` — a one-hot of [64, 1.33M] would need 268 MB of traffic, ~2.7 ms; measured 44.8 µs |
| "the 12 residual pixels are FMA contraction" | FMA is real but inserting barriers everywhere left the result bit-identical. The cause is MC's sin LUT quantisation, with FMA/`/` feeding 1-ULP errors into it | exact-arithmetic trace of `3.1415927 * 10430.378` |
| "TinyGrad's **Metal** division is not correctly rounded" — and the first fix, "the `cstyle` renderer emits `a*(1/b)`" | **neither: it is not a backend defect.** `div()` ends with `d = a.alu(Ops.MUL, b.reciprocal())` at `mixin/elementwise.py:251`, so the rewrite happens in the **Tensor API — above the scheduler and above backend selection** — and no `Ops.FDIV` node ever exists in the IR for any renderer to render (`reciprocal()` is `self.alu(Ops.RECIPROCAL)`, `mixin/elementwise.py:465`). `cstyle` is the wrong discriminator: `ClangRenderer` is also a cstyle renderer, and it is the only renderer among those measured here whose op set contains `Ops.FDIV` (`renderer/cstyle.py:271` deletes `RECIPROCAL`, `:274` adds `FDIV`), which is what lets the CPU backend's decomposition at `codegen/decomp/op.py:133-136` — gated on `if Ops.FDIV in ops` — manufacture the missing `FDIV` and emit a real `/`. That one fact is the whole per-backend split | `probes/divide_precision.py`; generated source re-captured this pass — CPU `val0[0]/val1[0]`, METAL/CL `a*(1/b)`, PYTHON emits nothing at all (`PythonRenderer` is not a cstyle renderer: MRO `PythonRenderer → Renderer → object`, `RECIPROCAL` from the interpreter table at `uop/ops.py:1289`). Re-measured N=200,000: CPU 0/200,000 wrong, METAL and CL 27.11% and 1 ULP max. Grouping the three failing backends by *result* is not grouping by mechanism. Ledger **C-05 / C-06 / C-08** |
| "float32 `/` is wrong in 12.4–15.4% of cases" — the figure was in ~58 places | **it was never a measurement of division.** `probes/fp32_correctness.py:57` set the divisor equal to the dividend (`b = regimes[name] if op in "*/" else a`, and `regimes[name]` *is* `a`), so the `/` row computed `a/a`. General `a/b` on a divisor-independent draw is wrong in **25.9%–27.0%** — **1.77x–1.86x** more often on the same `a`. `a/a` is degenerate in a provable way: its exact quotient is `1.0`, representable, so the second rounding starts at a **corner** of an asymmetric rounding cell and fails only on the lower quarter of the reciprocal's error range | `probes/fp32_correctness.py` (fixed: independent divisor, both rows, law declared); `derivation/divide.py:published_rate_table`, `:corner_predicate`, `:sampling_interval`. **The structural fix is that a number is now derived, never transcribed** — it lives in exactly one executable place and documents cite the claim that prints it |
| "the Metal/CL backends *approximate* the reciprocal; the `1 − 2⁻²⁴` value is the signature" | **there is no approximation.** Every failing backend is bit-identical to `a * fl32(1/b)` — a **correctly rounded** reciprocal and one correctly rounded multiply, i.e. **double rounding**. A reciprocal 1 ULP *worse* than correctly rounded predicts **~86%** wrong against the **~27%** measured, and the only wrong `a/a` value is `0x3f7fffff` with `0x3f800001` above **never** produced — a one-sided, exactly-predicted failure, the opposite of an approximation signature. Consequently the earlier claim that the rate is "insensitive to reciprocal accuracy" is **retracted**: it is load-bearing | `derivation/divide.py:reciprocal_sensitivity`, `:corner_predicate`; `probes/divide_precision.py` step 2 re-measures bit-identity live |
| "the study's claim becomes 'a second rounding everywhere, **plus an approximation on Metal and CL**'" | **the approximation clause is withdrawn**, and the claim is now the single mechanism at `derivation/divide.py:mechanism_sentence` | same |
| "`/ ` never goes through `Ops.RECIPROCAL`" / "`Ops.FDIV` is dead code" | **both backwards.** `/` always goes through `Ops.RECIPROCAL` — that is the defect. And `Ops.FDIV` is the load-bearing last line of the CPU repair: `ClangRenderer` keeps it (`cstyle.py:271,274`) and `codegen/decomp/op.py:133-136` **manufactures** the node the Tensor API never emitted, which is why CPU measures 0 wrong | `mixin/elementwise.py:251,465`; `renderer/cstyle.py:271,274`; `codegen/decomp/op.py:133-136`; generated CPU source `(val0[0]/val1[0])`. Ledger **C-11**, **C-12** |
| "on CUDA, `a/b` will be ≈0 wrong in 200,000 (95% upper bound ≈ 3)" | **retracted, off by ~5e6.** A displacement of `s` ULPs of the result **is** a displacement of `s` rounding cells, so the crossing probability is `s` — order one, not `2⁻²⁴`. The derivation multiplied a *relative* epsilon by an *absolute* one-ULP threshold. Correct figure: the mean displacement in ULPs of the result, 22.7%–27.1%, which equals the rate the same draw produces to within 0.3% at all 18 independent-law points. Measured on the T4: CUDA 27.13% (54,270), bit-identical to `a*fl32(1/b)` | `derivation/divide.py:cell_failure_fraction` (slip factor 2.15e6); `bench/HYPOTHESES.md` H9 rewritten; ledger **C-13** |
| "99% of KDA's wall time is Python building the graph" | 16% is graph construction, **85% is the scheduler** re-running its rewrite passes every `realize()`. Kernels *are* compiled once and cached; the *plan* is not | `kda/replay_kda.py` separates build from realize. **Also fixed there:** the GPU term was a hardcoded `0.109` and COLD a hardcoded `49.5 s` (against a measured 96,837 ms), and `perf_kda.py` printed the literal "Metal" on a CUDA run — three hardcoded literals in output strings, the same class as everything above. All three now measured or printed from the running device |
| KDA perf numbers | measured on a graph producing **NaN**: the benchmark used un-normalised `q`/`k`, so the delta rule diverged (`|S|` -> 1.9e140 in float64, `inf` in float32) | `l2n=True` restores it, relerr 6.3e-07. Graph shape and kernel count unaffected |

## Bugs found, in the order they were hit

Kept because the *process* is the reusable part: each one was a silent wrong
answer, and three of four were found by having a third implementation or a
per-step tracer rather than by reading the code.

1. **`t` taken from the post-update `tmx`.** C does `t = tmx; vx += sx; tmx += tdx`.
   Assigning `t` after updating `tmx` marches one plane too far. Found by
   `trace2_oc.py` showing `v,t` frozen at `(8,79,8), 0.000` for 44 steps.
2. **Three-way branch split as `~a` instead of `~(a & b)`.** C is
   `if (tmx<tmy && tmx<tmz) x elif (tmy<tmz) y else z`. When `tmx<tmy` but
   `tmx>=tmz`, all three of our masks are false and the ray dies. Symptom:
   every pixel reported sky.
3. **Sin table stored angles, not sines.** `tab[16384]` was 1.5707964 and
   `tab.max()` was 6.283. Missing `np.sin()`. Present in *both* Python ports.
   Found only because the NumPy port and the TinyGrad port agreed with each
   other and disagreed with C, which ruled out the port layer.
4. **FMA contraction.** Not a port bug at all, and **not a device-specific one
   either** — it is a property of the target language and there is no
   user-facing knob (`runtime/support/compiler_cuda.py:48,51,53` sets only
   `--gpu-architecture`, `-I` and `--minimal`). `a*b+c` fuses: 27.14% of random
   float32 triples differ from the unfused result, 0.00% from an exact FMA
   emulation. **It is also not the cause of the camera's 12 residual pixels** —
   barriers at every contraction site left the result bit-identical; the cause is
   MC's sin-LUT bucket quantisation. `notes/02-obs-camera-residuals.md`.

## KDA port bugs

Worth listing separately because the failure mode is different: **every one of
these produced a plausible number, not a crash.** All were caught by the test
suite, but only once the suite existed and the reference was independently
validated.

5. **UT transform solved in the wrong orientation.** Subtracting previously
   solved *columns* instead of doing row-wise substitution *within* the column
   being built: relative error 0.69, no error raised. `numpy.linalg.solve` on
   the same matrix gives 2.8e-16.
6. **Column scattered with a row selector.** To place a solved vector into
   column `i` of `M`, the mask must be `e_i` as a `[1,C]` **row**; the `[C,1]`
   column form selects row `i` and yields a matrix where every column is the
   same vector. Silently wrong.
7. **`g_last` indexed on the wrong axis.** `g_cum[:,:,:,-1,:]` on a 5-D
   `[B,NT,HV,C,K]` leaves the K axis → `[B,NT,HV,1,K]`, which mis-broadcasts
   downstream instead of erroring.
8. **Output permuted in the wrong order.** `ref.py` does
   `transpose(0,1,3,2,4)` on `[B,NT,HV,C,V]` before reshaping, so the flat
   token index is `n*C + c`. Permuting to `[B,NT,HV,C,V]` and reshaping
   interleaves the head axis into the token index: every token gets another
   head's output. This was the last bug standing.
9. **`unsqueeze(-3)` on a 5-D tensor lands the axis in the wrong place** (it
   goes before HV, not after). Use explicit `reshape(B,NT,HV,C,1,K)`.
10. **Bare `np.float32` / `np.ndarray` constants hit `ufix()`.** tinygrad
    raises `Could not infer dtype`; every constant that meets an operator has to
    be wrapped in `Tensor(...)`.
11. **Masking `exp` by multiplying its output by 0** still evaluates `exp` on
    the out-of-band entries. Bias the exponent instead.

The general lesson, and the reason for the three-implementation rule: **bugs 5,
6, 7 and 8 all produce finite, plausible output.** Only comparing against an
independently validated reference catches them, and only a per-intermediate
comparison localises them — the final-output relative error was ~1.0 in every
case, which tells you nothing about where.
