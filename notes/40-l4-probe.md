---
kind: reference
---

# 40 — Is an L4 reachable on this Kaggle account, and what the answer costs the KDA subject

**The answer is no, and it is no in a way that is worse than a refusal.**

A kernel pushed with `"machine_shape": "NvidiaL4"` is **accepted, recorded, and
run**, and the machine it runs on has **no GPU on it at all** — no `nvidia-smi`,
`device_count` 0, a CPU build of torch. `kaggle kernels pull -m` then reports
`machine_shape: NvidiaL4` for that kernel, because the field records what was
*requested*. Three L4 requests and one RTX PRO 6000 request were made, through
one identical push path, interleaved with three T4 requests that all landed on
2 × Tesla T4.

| kernel version | `machine_shape` requested | GPU actually attached | cc | devices |
|---|---|---|---|---|
| 1 | `NvidiaL4` | **none** | — | 0 |
| 2 | `NvidiaTeslaT4` | Tesla T4 ×2 | 7.5 | 2 |
| 3 | `NvidiaL4` | **none** | — | 0 |
| 4 | `NvidiaTeslaT4` | Tesla T4 ×2 | 7.5 | 2 |
| 5 | `NvidiaTeslaT4` (control, clean) | Tesla T4 ×2 | 7.5 | 2 |
| 6 | `NvidiaL4` **+ the CUDA image pinned by hand** | **none** | — | 0 |
| 7 | `NvidiaRtxPro6000` | **none** | — | 0 |

`MEASURED`, seven sessions, one probe per session, one kernel slug
(`tenghuzhang/l4-probe`), so versions 2/4/5 and 1/3 differ **only** in the shape
string. Every run's payload was verified by the runner against the fingerprint
that invocation pushed; all seven are `MATCHED` and none is stale. Logs:
`out/gpush/l4/runs/*/`.

**So `notes/39` §3's refusal stands, and §4's "the 4090 stops being an optional
extra" is confirmed rather than relieved:** the KDA subject still has no valid
measurement, the blocker is not the T4's 65,536 B of shared memory, and the
cheapest card that would have removed the blocker is not obtainable here. The
scope label still rides on the number, which is where the study already put it.

> **A shape recorded in a kernel's metadata is a request that was accepted, not
> a card that was granted.** Any study that infers the hardware from that field
> — including this note's own first pass, before the control forced the
> distinction — is reading an intention.

---

## 1. Why the L4 was worth a session, and how the string was found

The P100-sunset announcement names `T4x2` and `L4` as the replacements, and this
account is on `NvidiaTeslaT4`, which is the T4 x2 shape (2 devices per session,
`MEASURED`). So the string is a live question, not a guess.

**The client SDK is not a safe source for it.** `kagglesdk` 2.2.4 documents
`machine_shape` as *"Currently supported options: NvidiaTeslaT4, NvidiaTeslaP100,
Tpu1VmV38"* — three shapes, none of them an L4. Reading that as the table would
have ended this note before it started, and it would have been wrong: a census
of 200 recently-run public kernels turned up `NvidiaRtxPro6000`
(`MEASURED`, 5 of 200), which appears in no docstring.

**The instrument is behavioural.** `kaggle kernels list` does not carry the
field; `kaggle kernels pull -m` does, and the value is written by the server
from the run that actually happened. So a census of recently-run public kernels
reads the server's shape table without spending a session. `kda_t4/l4_shape_census.py`
is that census; it is read-only by construction.

`MEASURED`, 2026-09-30:

| what was asked | server-reported `machine_shape` found |
|---|---|
| 200 most recently run public kernels | `None` × 163, `NvidiaTeslaT4` × 31, `NvidiaRtxPro6000` × 5, `Gpu` × 1 |
| kernels whose text names an L4 | **`NvidiaL4`**, 12 distinct kernels |

**And the row is live, which is the part that makes the negative worth having.**
`NvidiaL4` was reported by kernels that ran `2026-09-29T08:06:55Z` — about
19 hours before this census — as well as `2026-09-11`, `2026-06-16` and
`2026-04-26` (`MEASURED`; `kda_t4/out/census/hunt-6.json`).
`NvidiaRtxPro6000` was reported for a run at `2026-09-29T13:38:05Z`.

So the L4 is **not** a retired row that old metadata still carries, and it is
**not** a capacity stockout on the whole platform: other accounts are being
granted L4s within the day. The blocker is this account's entitlement.

**What that rules out.** The separate capacity report suggesting selectors grey
out to "none" and that "T4 supply is the only GPU option left" is *not* what
happened here. The platform has L4s in production; this account is simply never
given one, and the request fails silently rather than loudly.

## 2. The silent downgrade, and the control that attributes it

A refusal would have been cheap and unambiguous. What happened instead is worse:
the request is accepted, and the run proceeds on a machine with no accelerator.

`MEASURED`, from the run logs:

| version | requested | `nvidia-smi` | torch build | `device_count` | `nvcc` |
|---|---|---|---|---|---|
| 1 | `NvidiaL4` | not present | `2.10.0+cpu` | 0 | absent |
| 3 | `NvidiaL4` | not present | `2.10.0+cpu` | 0 | absent |
| 5 (control) | `NvidiaTeslaT4` | `/opt/bin/nvidia-smi` | `2.10.0+cu128` | 2 | 12.8 |
| 6 | `NvidiaL4` + CUDA image pinned | not present | `2.10.0+cu128` | **0** | 12.8 |
| 7 | `NvidiaRtxPro6000` | not present | `2.10.0+cpu` | 0 | absent |

**The mechanism is the image the server picks.** A T4 request is given
`gcr.io/kaggle-private-byod/python@sha256:37c64f7d…`; an L4 request is given
`gcr.io/kaggle-images/python@sha256:dafd4ce5…`, which is not the CUDA image.
`MEASURED` from the pulled `kernel-metadata.json` of the two versions.

That suggests a fix, and version 6 tested it: the CUDA image was pinned by hand
on the L4 request. **The pin worked and it was not enough.** `torch` became
`2.10.0+cu128` and `nvcc 12.8` appeared — a real CUDA toolchain — and
`device_count` was still **0**. So the blocker is the accelerator attach, not the
container.

**The control is what makes this attributable.** The first L4 attempt was the
slug's very first version, which leaves a cold-start confound. Version 3 repeated
the L4 request on the warm slug and versions 2/4/5 requested a T4 on that same
slug, same payload, same notebook generator, same push code. T4 → GPU, twice
plus a clean third. L4 → no GPU, twice, and again with the image pinned. The
only variable is the shape string.

## 3. The 2-session cap: per-accelerator or global — **not measured**

**Open unknown, and I did not spend the sessions to settle it.** The experiment
needs both T4 slots held simultaneously, which is exactly the over-dispatch this
study has already been burned by, so I left it alone. What the runs do establish:

- **The cap is not what blocks a non-T4 shape.** A `NvidiaL4` and an
  `NvidiaRtxPro6000` request were both accepted while the account had **zero**
  sessions running (`timeReserved: 0s` on `GET /api/v1/kernels/quota` before the
  first push, `MEASURED`). Neither was refused, so the concurrency cap is
  downstream of the entitlement problem and cannot be the cause.
- **The GPU quota bucket looks global across shapes.** Account GPU usage went
  `17.02 h → 17.20 h` across the seven sessions (`MEASURED`), about 11 minutes of
  session wall time, and that includes the four sessions that ran on a CPU box
  and got nothing for it. This is a coarse inference from a 2-decimal-hours
  delta, not a controlled measurement, and it is labelled as such — but if it
  holds, **a downgraded non-T4 request costs quota without returning a card**,
  which is worth knowing before anyone tries a shape sweep.
- The exact experiment: hold two T4 sessions, then push a non-T4 shape. A
  `Maximum batch GPU session count of 2 reached` refusal means the cap is
  **global**; acceptance means **per-accelerator**. One session's cost, and it
  must not be run while the other two lanes need a slot.

`Maximum batch GPU session count of 2 reached` is emitted server-side — it is
absent from the `kaggle` 2.2.4 sources — and the string carries no
accelerator qualifier, so the wording is consistent with a global cap and proves
nothing.

## 4. The card's properties against the three numbers to beat

`notes/39` §5's falsifier needs a `cc=89` device. **None was obtained, so the
falsifier stands untested.** The T4 control is the only card measured here.

| number to beat | source | status |
|---|---|---|
| sm_75 gives 65 536 B opt-in shared memory | on record | **re-derived here, independently** |
| sm_75 gives 0 tensor-core instructions | on record | not re-measured — see below |
| cc=80 gives 128 HMMA | on record | not re-measured |
| cc=89 expected 101 376 B and HMMA present | predicted | **not measured. Open unknown.** |

**`MEASURED`, version 5 (`out/gpush/l4/runs/20260930-115946-NvidiaTeslaT4/`), one
T4 x2 session, all five readers agreeing where they should:**

| reader | value |
|---|---|
| `fla.utils._device.get_all_max_shared_mem()` | `[65536, 65536]` |
| `triton … get_device_properties(0)['max_shared_mem']` | 65 536 |
| `torch … shared_memory_per_block` (**not** opt-in) | 49 152 |
| `torch … shared_memory_per_block_optin` | 65 536 |
| `torch … shared_memory_per_multiprocessor` | 65 536 |
| `cudaDeviceGetAttribute(97)` | 65 536 |

and the gate itself, read off the installed `fla-core` 0.5.2 wheel **on the card**
rather than quoted:

| `Backend` threshold | bytes | `check_shared_mem(…)` on this card |
|---|---|---|
| `ADA` | **101 376** | `False` |
| `AMPERE` | 166 912 | `False` |
| `DEFAULT` | 102 400 | `False` |
| `HOPPER` | 232 448 | `False` |

So both halves of the `notes/39` §4 blocker are re-derived from the device:
**65 536 B available, 101 376 B required**, and the 101 376 figure is confirmed
as the ADA threshold in the wheel actually installed rather than as a line
citation. The 49 152 B non-opt-in value is reported beside it because the T4 has
always produced both and quoting either alone is half the answer.

Card identity, `MEASURED`: 2 × Tesla T4, 15 360 MB each, cc 7.5, 40 SMs,
driver 580.159.04, CUDA runtime 13.0, `nvcc` 12.8, torch 2.10.0+cu128.
tf32 state, `MEASURED`: `torch.backends.cuda.matmul.allow_tf32 = False`,
`cudnn.allow_tf32 = True`, `get_float32_matmul_precision() = "highest"`.

**The 0-tensor-core and 128-HMMA rows were not re-measured, deliberately.**
`kda_t4/l4_probe.py` runs its `tl.dot` SASS census and the `fla` forward only on
`cc >= 8.0`, because on a card this study has already measured they would spend
roughly eight minutes of a *shared* T4's quota to reproduce a number already on
the record. That is a defensible spend decision and it has a cost: **the
probe's positive branch — the one that would run on an Ada card — has never
executed anywhere.** It is written, it passes `py_compile` and `pyflakes`, and
it is unexercised. It should not be trusted on first contact with a real `cc=89`
device without reading it again.

## 5. The KDA three-port ladder: **not run**

Conditional on a card that passes, and no card passed. Nothing here is a
measurement of KDA, `fla`, or any ratio.

What is established about the ladder, so the next person does not re-derive it:

- **The three ports exist and are the same three as `notes/30` §6's table:**
  `kda/tiny_kda.py` (naive), `kda/tiny_kda_opt.py` (good),
  `kda/tiny_kda_faithful.py` (faithful) — verified present on disk, not run.
  So the first rung exists: this is a three-rung ladder, not a two-rung one.
- `kda/baseline_pin.py` has a working `--strict --json` mode, and on the record
  it exits `INVALID` with zero `UNKNOWNs` on the shared-memory gate alone. **That
  exit status is cited from `notes/30`, not re-run here** — but it is derived
  from `fla`'s own gate, whose thresholds §4 of this note re-read off the card,
  so it will re-derive on any card and is the cheapest possible pre-flight.
- **`fp16`, not `bf16`, for this arm.** This is a dtype choice about precision
  rather than a change of operator, and the brief cites the T4 measurement
  (7.148e-04 against 5.714e-03 relative error for the same operator) as the
  reason. **I did not re-derive that pair** and it is cited, not measured here;
  whoever runs the arm should confirm it from `kda_t4/time_t4.py`'s own output
  before quoting it.
- **The config space is half the finding.** On sm_75 the autotune space collapsed
  from 12 to 4. If an `cc=89` device ever appears, the live count is the first
  number to read, *before* any timing — it is read from the `Autotuner.configs`
  on the card and compared against the same list modelled from
  `check_shared_mem`, and a disagreement between those two is reported as a
  disagreement rather than resolved in favour of the one that matches a note.
- `notes/39` §5's falsifier — the KDA ratio landing inside the 25.96× envelope —
  **is untested and remains the reason to want this card.** It is not a reason to
  believe it.

## 6. Six defects in this note's own harness, none found by a measurement

Every one was found by a gate, a control, or by reading a generated artefact.
None was found by a number coming out wrong, which is `notes/33`'s pattern
reproduced on a different set of code.

1. **The payload fingerprint was empty.** `Manifest.build()` does not compute it;
   `gpush/cli.py:683` calls `compute_fingerprint` explicitly and I did not.
   Consequence: `payload_provenance` computes `stale = bool(pushed_fp and ran and
   ran != pushed_fp)`, so an empty pushed fingerprint makes `stale` permanently
   `False` and **silently disables the STALE PAYLOAD check** — the exact
   protection this push was built around, broken by forgetting one call. Found
   by reading a printed fingerprint that was blank. `l4_push.py` now refuses to
   push without one.
2. **The harness overwrote its own evidence.** The log and the fetched artifacts
   went to one fixed path, so push *n* destroyed push *n-1*'s. The T4 control's
   shared-memory section was destroyed by the L4 attempt that followed it — the
   worst possible ordering, because the control is what makes the negative
   attributable and it is what an overwrite deletes. Found by going to look for
   the control's numbers. Each push now gets its own run directory; the logs of
   versions 1 and 3 were recovered afterwards — version 3 from the fixed path,
   version 1 from Kaggle's version-indexed log endpoint — and **version 2's log
   is gone for good.**
3. **A `TypeError` on the branch that only a real GPU reaches.**
   `identity()` stores the capability as a `list` so it round-trips through JSON,
   and the guard compared that list to a tuple. Every L4 attempt died in the
   no-GPU branch before reaching it, so the bug was invisible to the runs that
   motivated the probe. **The T4 control crashed on it.** A code path that has
   only ever executed in its failure mode is not a tested code path.
4. **A report that omitted its own key numbers.** The shared-memory block printed
   each reader as a raw dict, and the gate-threshold loop looked for lower-case
   keys against an upper-case enum — so 101 376 and 166 912 were absent from the
   human-readable block while sitting unformatted in the JSON beside it. Found by
   reading my own output.
5. **A rate-limited null that looked exactly like a result.** A 10-wide parallel
   pull pass returned HTTP 429 on every request and came back "no non-empty
   `machine_shape`" for kernels whose metadata had been read successfully
   minutes earlier. Found because a kernel I had already read came back empty.
   The shipped census is serial with a pause and reports a fetch failure as a
   failure.
6. **A probe reporting its own error text as a GPU's name.**
   `driver_version` is not a valid `--query-gpu` field, so `nvidia-smi` printed
   its error *in place of the whole line* and the first control's `name` was
   `Field "driver_version --format=csv" is not a valid field to query.`. Found by
   reading a name that was obviously not a GPU.

## 7. What could not be measured, and why

| not measured | why |
|---|---|
| `cc=89` shared memory, tf32, tensor cores, `fla`'s config space on `cc=89` | no Ada device was obtainable; the KDA subject is unchanged |
| the KDA/`fla` ratio, and therefore `notes/39` §5's falsifier | conditional on the card above |
| `fla`'s live config space on any non-sm_75 card | same |
| whether the 2-session cap is per-accelerator or global | needs both T4 slots held; would over-dispatch against the other two lanes |
| any claim about other accounts' entitlements | the census reads a shape from metadata; it cannot read a *grant policy* |
| the fp16-vs-bf16 error pair | cited from the brief, not re-derived |
| `l4_probe.py` sections 2–4 | the branch requires `cc >= 8.0`; no such device was obtained, so it has never run |
| the 25.96× envelope's own rungs | not this note's subject, and not re-measured |

## 8. What this changes

- **`notes/39` §3 is confirmed.** The KDA subject's deficit is unquantified
  against a competent reference, and the reason is the card, not the framework.
  The L4 was the cheapest candidate remedy and it does not exist here.
- **The scope label rides on the number, permanently, on the free tier.** Every
  `fla` number in this study is a number measured against a reference that the
  card degraded, and no free card reachable from this account removes that.
- **The standing answer is now stronger and it is measured rather than argued:**
  "no free card" is not a guess about the platform. An L4 request is *accepted*.
  Anyone repeating this must not stop at the acceptance.
- **A second, unrelated card is also refused the same way.**
  `NvidiaRtxPro6000` is a live row with runs on `2026-09-29`, and it is refused
  identically. The entitlement is not "no Ada", it is "this account is a T4
  account".
- **The cheapest remedy is unchanged and still un-tried by this study:** the
  4090 recommendation in `notes/39` §4 stands. It is a local card, it costs
  nothing from the quota this account cannot spend on other shapes, and it is
  the one route that makes class (iv) observable.

## Sources

| claim | where |
|---|---|
| the seven runs, one probe each, per-version archives | `out/gpush/l4/runs/`, `out/gpush/l4/push_*.log` |
| the shape census and the hunt for `NvidiaL4` | `kda_t4/l4_shape_census.py`, `kda_t4/out/census/` |
| the properties probe and its two-skip design | `kda_t4/l4_probe.py` |
| the push path, the refusal scan, the per-run archive | `kda_t4/l4_push.py` |
| the one-session-per-probe wrapper | `kda_t4/l4_run.sh` |
| the fingerprint/`stale` interaction that defect 1 breaks | `gpush/manifest.py:168`, `gpush/cli.py:94-109` |
| the payload-READY gate the push inherits | `gpush/backends/kaggle.py:211` |
| the runner notebook and its self-check | `gpush/backends/runner_nb.py:43-157` |
| the 12-config space and the shared-memory gate being collapsed on sm_75 | `notes/30-fla-on-t4.md` §2, §6 |
| the envelope's worst rung, 25.96×, and the KDA ratios against it | `notes/39-subjects-on-the-axis.md` §1 |
| the 4090's second job, and the 101 376 B vs 65 536 B blocker | `notes/39-subjects-on-the-axis.md` §4 |
