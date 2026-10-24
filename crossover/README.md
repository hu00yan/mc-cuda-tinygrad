# `crossover/` — the experiment

Tests one claim: **the crossover between "the compiler wins" and "the human
wins" is set by whether the materialised intermediate fits in on-chip memory,
and it is a step function rather than a slope.**

The claim is stated, derived into predictions with quantities, attacked, and
partly replaced. `notes/07-crossover-thesis.md` is the writeup; everything in it
is produced by a script here.

## Run everything

```bash
cd /Users/huyan00/mycode/mc-cuda-tinygrad
.venv/bin/python crossover/test_control.py       # 64 checks; the suite that FAILS
.venv/bin/python crossover/devices.py           # capacities, one source per field
.venv/bin/python crossover/claim.py             # the predictions + refutation criteria
.venv/bin/python crossover/literature.py        # every prior number, re-derived
.venv/bin/python crossover/attn_control.py      # the control experiment
.venv/bin/python crossover/residency_probe.py   # step or knee, MEASURED locally
```

Dependencies: `numpy` only. `tinygrad` is imported if present and never
required. No torch, no CUDA, no jax, no network at run time.

## What each file is, and what it is not

| file | is | is **not** |
|---|---|---|
| `devices.py` | the capacity table, one cited source per field, plus the conflicts and unverified items it could not resolve | a measurement. Every capacity is a datasheet number, and §3.1 of the note shows the *usable* capacity is a fraction of it |
| `claim.py` | four crossover models, per-device KDA thresholds, and the pre-registered refutation criteria | a result. Every "which side wins" is `[PREDICTED]` |
| `literature.py` | every prior-work number in the thesis, re-derived from the tables rather than transcribed | a reading of the papers. The source tables are transcribed into `research/*.md` and are `[UNVERIFIED]` at one remove |
| `attn_control.py` | **the control**: one computation, two implementations, differing only in whether the `[B,H,n,n]` intermediate exists | evidence about any GPU. numpy's matmul is already tiled, so a CPU run cannot exhibit a cache crossover, and it does not |
| `residency_probe.py` | a real measurement of whether residency produces a step or a knee, on whatever machine is in front of you | a measurement on any of the five study devices. It is a CPU, and the capacities are not theirs. Only the *shape* transfers |
| `test_control.py` | 64 assertions covering shapes, bytes, semantics, the device table and every re-derivation | a confirmation of any prediction. A prediction that a test "confirms" is a category error and the suite refuses to do it |

## The control, in one paragraph

`attn_materialised` and `attn_online` compute the same thing — softmax of
`q kᵀ/√d` times `v`, in fp64 — and hold everything else constant: same inputs,
same matmuls, same output (asserted equal to 1.4e-15), and the online form's
key-block size is a *parameter* whose result is invariant (asserted to 1.4e-15,
so it is a knob and not a variable). The only intended difference is that one of
them allocates a `[B,H,n,n]` array and the other never allocates more than
`[B,H,n,Bc]`.

**They are not the same program, and the file says so in its output rather than
in a footnote.** They cannot be. The comparison is between *implementations*, and
the claim is only as strong as the claim that the materialising one is what a
compiler would actually produce — which is unmeasured, and is refutation
criterion R-C.

## The pre-registered refutation criteria

Printed by `claim.py` §4 on every run, before any data:

- **R-A** the step — a gradual transition **refutes** it, and is the expected
  outcome
- **R-B** which capacity matters — H100 vs B200 separates the models by 2.52×
  in opposite directions
- **R-C** the natural-lowering premise — the load-bearing assumption, unmeasured
- **R-D** the control is a control — **runs today, passes**
- **R-E** the pass count — currently unmet; every predicted ratio is linear in it

## Two errors made while building this, kept on the record

1. A byte model that charged I/O as `2K + 2V` instead of `3K + 2V`, which made
   the study's own fold ratio read 10.1× instead of 13.0× — an apparent
   refutation that was a bug in the checker. Fourth such bug in this study.
2. A bandwidth probe using `a.sum()`, which is reduction-bound at ~23 GB/s and
   reported "no knee found" at every size. `np.vdot` is memory-bound and finds
   the knee immediately. **A null result from a probe that cannot resolve the
   effect is not a null result.**

## Findings about other people's files, recorded not fixed

`bench/selftest.py:103` gives the P100's `L2_CACHE_SIZE` as 24 MiB. NVIDIA's
*Inside Pascal* table gives 4096 KB = 4 MiB for both P100 SKUs, corroborated by
8 memory controllers × 512 KB. **6× off**, and linear in every capacity-derived
crossover for that card. `bench/` is owned by another agent and is not modified
here; the discrepancy lives in `devices.CONFLICTS` and is asserted in
`test_control.py`.

## Hardware status, recorded rather than assumed

| | |
|---|---|
| NVIDIA GPU | **none** — no CUDA, no ROCm, no torch in the venv |
| TPU | **none** — `import jax` → `ModuleNotFoundError`; `kda/tpu/` is written but not run |
| this machine | Apple M1 Pro, 8 cores, 32 GiB. `tinygrad` default device is `METAL` |

No GPU number appears anywhere in this directory, and nothing here implies
otherwise.
