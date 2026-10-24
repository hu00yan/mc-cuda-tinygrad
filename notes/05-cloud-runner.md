# 05 — Running the battery on real Hopper / Blackwell silicon

`remote/cloud_run.py` is the launcher. Everything here is a claim about
**Modal's and Lightning AI's documentation and pricing**, verified by reading
their pages on **2026-09-28**, plus a hardware claim that was settled by
running NVIDIA's own assembler rather than by reading about it.

Reproduce every check:

```bash
# 86 checks, no network, no credentials, $0.00
.venv/bin/python remote/cloud_run.py --provider modal --arch h100 --dry-run

# the feature table, diffed against ptxas itself (needs this repo's archlab image)
.venv/bin/python remote/cloud_run.py --provider modal --arch h100 \
    --verify-feature-table --ptxas-cmd 'docker run --rm --platform linux/amd64 \
      -v {dir}:/w archlab/ptxas:12.9 \
      /usr/local/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas \
      {arch} /w/{name} -o {devnull}'
```

**No run has happened.** There is no account on either platform, no GPU
hardware was touched, and nothing in `out/cloud/` exists yet. What is
validated is the launcher.

---

## 1. The GPU string, and why a wrong one is silent

Modal's docs, verbatim:

> "Modal may automatically upgrade a `gpu="H100"` request to run on an H200.
> This automatic upgrade does _not_ change the cost of the GPU."
> "In cases where an automatic upgrade to H200 would not be helpful (for
> instance, benchmarking) you can pass `gpu=H100!` to avoid it."
> — <https://modal.com/docs/guide/gpu#automatic-upgrades-to-h200s>

> "Use `gpu="B200+"` to allow Modal to run requests on either B200 or B300
> GPUs. B200+ is billed as B200, regardless of which GPU is used. Use this
> option only if your code is compatible with both types of GPUs."
> — <https://modal.com/docs/guide/gpu#opt-in-upgrade-to-b300>

> "Modal may automatically upgrade a `gpu="A100"` request to run on an 80 GB
> A100. This automatic upgrade does _not_ change the cost of the GPU."

So the pinned strings, and the reason each one is that exact string:

| arch | pinned `gpu=` | why that and not the neighbour |
|---|---|---|
| `h100` | `H100!` | **the bang is mandatory.** Bare `H100` may be served as an H200 |
| `h200` | `H200` | no documented substitution, no `!` form exists |
| `b200` | `B200` | **never `B200+`.** `+` may be served as a B300 |
| `b300` | `B300` | requested directly, so no `+` form is reachable |
| `l40s` | `L40S` | no upgrade path, no variant form |
| `rtxpro6000` | `RTX-PRO-6000` | exact spelling from the docs; sm_120 |
| `a100` | `A100-80GB` | **never bare `A100`**; 40 GB vs 80 GB changes OOM outcomes |

An H200 served for an H100 request is the dangerous one, because **H100 and
H200 are both sm_90**. The compute capability check cannot see the
substitution. What differs is 141 GB vs 80 GB of HBM3e and 4.8 TB/s vs
3.35 TB/s — and memory bandwidth is precisely what this study measures. The
substitution would not look like a bug. It would look like a Hopper result.

The script does not trust the string anyway: it reads
`torch.cuda.get_device_capability()` off the device, and a mismatch fails the
run (exit 9) with the warning propagated into `SUMMARY.md` and `MANIFEST.json`
in the results directory.

## 2. Pricing and free tier — verified, with the date

From <https://modal.com/pricing>, read 2026-09-28. Per-second, base, us-east:

| GPU | $/sec | $/hr | $30 buys |
|---|---|---|---|
| B300 | 0.001972 | 7.0992 | 4h13m |
| B200 | 0.001736 | 6.2496 | 4h48m |
| H200 SXM | 0.001261 | 4.5516 | 6h35m |
| **H100 SXM5** | **0.001097** | **3.9492** | **7h36m** |
| RTX PRO 6000 | 0.000842 | 3.0312 | 9h53m |
| A100 80GB | 0.000694 | 2.4984 | 12h00m |
| **L40S** | **0.000542** | **1.9512** | **15h22m** |
| L4 | 0.000222 | 0.7992 | 37h31m |
| T4 | 0.000164 | 0.5904 | 50h48m |

Functions also bill CPU at 0.00003942/core/sec and RAM at
0.00000667/GiB/sec, which the launcher includes — a 45-minute H100 run has a
worst case of **$4.08**, not $2.96, and the difference is the whole reason the
budget gate is arithmetic rather than a hope.

Three things the pricing page does not say loudly:

- **Region is a multiplier.** "Region selection — 1.15 - 1.75x base prices."
  The launcher pins `us-east` (1.0x). Choosing a region by convenience can
  nearly double the bill with no other signal.
- **A card is required even for the free credit.** "Using a GPU requires
  having a valid payment method on file."
  — <https://modal.com/docs/guide/gpu>
- **B300 needs CUDA 13.1+**, which constrains the image, not just the request.

### The budget guarantee is arithmetic, not a spend limit

Modal's workspace budget and spend limit are set **only in the web UI**
(<https://modal.com/docs/guide/budgets>, `https://modal.com/settings/usage`).
There is no documented API, so the launcher cannot set one, cannot read one,
and cannot verify one. Worse:

> "The maximum budget you can set depends on prior successful charges for the
> Workspace."

A brand-new account may be unable to set any budget at all. So the launcher
takes the only route that does not depend on a control it cannot reach: it
computes the worst case and **refuses to start unless it fits inside the free
credit**. `--credit-only` (assert no payment method) is then sufficient, and a
$0.00 outlay becomes a property of the arithmetic rather than of a UI toggle.
`--modal-spend-limit-usd` is supported for operators who set one, and is
checked against the worst case — but it is belt, not braces.

## 3. Preemption cannot be turned off

> "All Modal Functions are subject to preemption by default."
> "**Note: The `nonpreemptible` parameter is not supported for GPU
> Functions.**"
> — <https://modal.com/docs/guide/preemption>

So a GPU run can be interrupted and there is no opt-out. Worse, Modal
"will gracefully terminate the Function and restart it on the same input",
which re-spends the budget. Mitigations, all implemented:

- `retries=0` on the function.
- Per-stage checkpoint to a Modal Volume, so a preempted run's partial results
  are still collectable.
- `DONE` written last, atomically, and a SIGTERM handler that stops launching
  and flushes. A restart sees the marker and exits instead of re-running.
- Preemption is reported as its own exit code (7) with partial results kept,
  and the message says plainly that it is not avoidable.

## 4. The correction: B300 **does** have `tcgen05`

The brief this was built from asserted "tcgen05: sm_100a/sm_101a only — not
sm_103, not sm_120". Half of that is wrong, and the wrong half matters,
because B300 is one of the three Blackwell targets.

Settled by running `ptxas` 12.9 (this repo's `archlab/ptxas:12.9` image), not
by reading:

```
tcgen05.alloc    sm_100a  ASSEMBLES
tcgen05.alloc    sm_100   "Instruction 'tcgen05.alloc' not supported on .target 'sm_100'"
tcgen05.alloc    sm_101a  ASSEMBLES
tcgen05.alloc    sm_103a  ASSEMBLES
tcgen05.alloc    sm_103   "Instruction 'tcgen05.alloc' not supported on .target 'sm_103'"
tcgen05.alloc    sm_120   "Instruction 'tcgen05.alloc' not supported on .target 'sm_120'"
tcgen05.alloc    sm_120a  "Instruction 'tcgen05.alloc' not supported on .target 'sm_120a'"
tcgen05.mma      sm_103a  passes the arch gate (rejected later on operand shape)
```

NVIDIA's PTX ISA 9.4 agrees: its `tcgen05` Target ISA Notes list
`sm_100a, sm_101a (Renamed to sm_110a from PTX ISA version 9.0), sm_103a,
sm_107a, sm_110a`. `sm_12xa` appears in 49 arch lists in that document and
**none** of them are `tcgen05`.

**Corrected rule:** `tcgen05` needs an architecture-conditional target on the
datacenter-Blackwell families — `sm_100a`, `sm_100f`, `sm_101a`, `sm_101f`,
`sm_103a`, `sm_107a`, `sm_110a`. So:

- B200 (sm_100) — **has** it
- B300 (sm_103) — **has** it
- RTX PRO 6000 (sm_120) — **does not**
- bare `sm_100` and bare `sm_103` — **do not**; the `a` suffix is the point

## 5. The finding that matters most: tinygrad cannot reach any of it

`.venv/lib/python3.12/site-packages/tinygrad/runtime/ops_cuda.py:122`

```python
arch=f"sm_{major.value}{minor.value}"
```

No architecture-conditional suffix. So the vendored tinygrad compiles to the
bare target on every GPU, and ptxas refuses exactly the instructions the study
cares about:

| feature | target it needs | what tinygrad emits | result |
|---|---|---|---|
| `wgmma` | `sm_90a` | `sm_90` | **unreachable on H100** |
| `tcgen05` | `sm_100a` | `sm_100` | **unreachable on B200** |
| `tcgen05` | `sm_103a` | `sm_103` | **unreachable on B300** |

Consequence for the study: **with this tinygrad, every architecture produces
the same feature-availability set.** Any cross-architecture difference the
study measures can only come from memory bandwidth, clocks and capacity —
never from a richer instruction set. A report that says "H100 has wgmma" based
on a capability table would be wrong about what was actually compiled.

This does not error. It does not warn. It produces entirely plausible finite
numbers, which is why it is the most dangerous item in the launcher rather
than the GPU-string pinning that the brief foregrounded.

The defence is `--ptx-probe`, which asks the assembler on the box and **fails
the run** (exit 9) if the answer disagrees with the transcribed table. The
dry run proves that guard can fail by corrupting the table on purpose and
checking the verdict moves.

## 6. The feature table, as verified

`Y` = ptxas assembles for the arch-conditional target. `--` = not applicable.

| arch | family | cp.async | FP8 MMA | TMA | wgmma | tcgen05 |
|---|---|---|---|---|---|---|
| sm_75 | turing | no | no | no | -- | -- |
| sm_80 | ampere | **Y** | no | no | -- | -- |
| sm_86 | ampere | Y | no | no | -- | -- |
| sm_89 | ada | Y | **Y** | no | -- | -- |
| sm_90 | hopper | Y | Y | **Y** | **Y** | no |
| sm_100 | blackwell_dc | Y | Y | Y | no | **Y** |
| sm_103 | blackwell_ultra | Y | Y | Y | no | **Y** |
| sm_120 | blackwell_ws | Y | Y | Y | no | **no** |

Boundaries, each from ptxas's own message, not folklore:

- `cp.async` — "Feature 'cp.async' requires .target sm_80 or higher"
- FP8 MMA — "Feature 'mma with FP8 floating point type' requires .target sm_89
  or higher" (so **A10/sm_86 and A100/sm_80 have no FP8 MMA at all**)
- TMA — "Feature 'cp.async.bulk.tensor' requires .target sm_90 or higher"
- `wgmma` — "not supported on .target 'sm_90'", accepted on `sm_90a` only
- `tcgen05` — see §4

`--verify-feature-table` diffs all of this against the assembler across 10
capabilities and both target forms, and exits non-zero on any disagreement.
Re-run it after a toolkit upgrade; a toolkit upgrade is exactly the event that
would change the answers.

## 7. Lightning AI — the weak leg, and why

**Nothing about Lightning's commercial terms could be confirmed.** Its own live
pages carry four mutually inconsistent free-tier figures:

| claim | where on lightning.ai |
|---|---|
| "up to 30 credits" | pricing tier card **and** the comparison table |
| "You get 5 free Lightning credits upon registration" | the same page's FAQ block |
| "Start now with 80 free GPU hours" | homepage hero |
| "80 GPU hours monthly + 15 credits" | a 2025 third-party thread |

Its official consumption-rate table
(<https://lightning.ai/pricing/consumption-rates>) publishes **only**:

| machine | credits/hr |
|---|---|
| CPU (2 vCPU) | 0.05 |
| CPU (8 vCPU) | 0.21 |
| 1 T4 | 0.13 |
| 1 V100 | 0.38 |
| 4 V100 | 1.53 |

**There is no published rate for H100, H200, B200, A100 or L40S** anywhere in
the served HTML. The survey's H100 $/hr figure is therefore unverified and
should not be budgeted against. The survey's flag that the rate table came
from a stale cache is corroborated: the page currently ends with "Taking longer
than expected — try refreshing the page", and the modern-GPU rows do not
render.

Also unresolved:

- **Whether a card is required.** Not stated in Lightning's docs. The only
  "no card" claims found are third-party. Confidence: **low**.
- **Whether the free tier really exposes H100.** The pricing page does list
  "Single GPUs (L40S, A100, H100, H200...)" for Free, with a 4-hour session
  limit on A100/H100/H200, so it is plausible — but the same site contradicts
  itself about the credits that would pay for it. Confidence: **low**.

Consequence for the code: the Lightning adapter **refuses to provision by
default** (exit 2) and requires both `--lightning-machine` (a string it refuses
to guess) and `--accept-unverified-surface`. Its provisioning path has never
been executed, because the SDK is not installed here and installing it was out
of scope. What *is* verified is provider-independent: the argument handling,
the cost arithmetic, the battery plan, and the result collection all run
identically. The device-identity check, not the machine-type string, is the
guard that stands.

## 8. Survey claims: confirmed, corrected, unconfirmed

**Confirmed** (Modal, primary docs, 2026-09-28): Starter $0/mo with $30/mo free
compute; H100 $3.9492/hr; B200 $6.2496/hr; L40S ~$1.9512/hr; RTX PRO 6000
~$3.0312/hr; card required for GPU use; the `H100!` benchmarking form; the
`B200+` opt-in and its billing rule; $30 buys ~7h36m of H100 or ~4h48m of B200.

**Corrected:** `tcgen05` on B300/sm_103. It exists, on `sm_103a` (§4). The
brief's rule is wrong here.

**Could not confirm** (Lightning): the per-GPU-hour rate for any modern GPU;
the free-credit amount (four conflicting first-party figures); whether a card
is required; whether the free tier exposes H100. The specific "~1.07 h of
H100" figure has no first-party basis I could find and should be treated as
unsupported.

**Found that the survey did not mention:** regions cost 1.15–1.75x; Functions
bill CPU and RAM on top of the GPU; `nonpreemptible` is not supported for GPU
functions, so preemption is unavoidable; and a preempted function is *restarted
on the same input*, which re-spends the budget unless you defeat it.

## 9. What `--dry-run` actually proves

86 checks, all passing on all seven archs, with sockets and non-`sys.executable`
subprocesses physically disabled for the duration:

1. argument handling, cost arithmetic, and the budget gate — including that it
   **rejects** an oversized budget and names the affordable ceiling
2. path staging over the real tree: real byte counts and sha256, 2 566 files,
   with the 2.6 MB regenerable blob excluded and files vanishing mid-walk
   tolerated (`matmul/` is being written to while the dry run runs)
3. the generated remote driver is compiled, imported, and executed — and its
   `--selftest` really spawns subprocesses, really handles SIGTERM, and
   really kills a hung stage at 3.1 s
4. the feature table for every capability, plus the two load-bearing rows
5. all 55 probe templates render with `.version` and `.target` filled in
6. the real collector over a synthetic result tree, including that a
   B200-request-served-an-H200 run is a hard failure and that `SUMMARY.md`
   says so in plain words
7. all eight failure modes map to distinct exit codes with distinct remedies
8. two post-run verdicts

Two of those checks exist because the dry run caught real bugs in itself: a
file vanishing mid-walk, and a generated file that compiled but failed to
import because JSON `\n` escapes were un-escaped by the surrounding Python
string literal.

**Not proven, and no claim is made:** that a Modal account works, that the
image builds, that any GPU is available, that the optional torch/triton/fla
installs succeed, or that Lightning's SDK surface exists in the form written.
Those need a real run.
