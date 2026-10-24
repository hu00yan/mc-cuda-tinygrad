# `kernels/` — running the battery on Kaggle

The GPU and TPU lanes for this study. `kaggle kernels push` → poll `kaggle
kernels status` → `kaggle kernels output`, all over plain HTTP with the
official CLI. No SSH, no tunnel, no long-lived connection, no gcloud.

Everything below was executed, not assumed. Quota figures, accelerator
capabilities and probe verdicts are from real runs on 2026-09-27/28; the
commands are the ones that produced them.

---

## One-time setup: authenticate

There is **no file to place in the repo.** The CLI's own login writes
credentials to your home directory:

```bash
uv pip install --python .venv/bin/python kaggle    # once; the CLI is already there
.venv/bin/kaggle auth login
```

`kaggle auth login` is a browser OAuth flow; it caches a token at
`~/.kaggle/access_token` (mode 600). Check it with:

```bash
.venv/bin/kaggle config view     # - username: <you>   - auth_method: ACCESS_TOKEN
.venv/bin/kaggle quota           # GPU 30.00h / TPU 20.00h per week, and what is left
```

If you would rather paste a token: create one at
<https://www.kaggle.com/settings/api> and write it to the same path —

```bash
printf '%s' 'YOUR_TOKEN' > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
```

`remote/kaggle_run.sh` checks for that file (or `KAGGLE_API_TOKEN`, or a legacy
`~/.kaggle/kaggle.json`) before it does anything, and if none is present it
prints the `kaggle auth login` line and exits. It never asks you to create a
token file in the repo.

## Run it

```bash
./remote/kaggle_run.sh                     # GPU lane,  kernels/probe
./remote/kaggle_run.sh --variant tpu       # TPU lane,  kernels/probe-tpu
./remote/kaggle_run.sh --dry-run           # everything except the 3 network calls
./remote/kaggle_run.sh --smoke             # 30-second loop test, kernels/smoke
```

The GPU run is the one that matters: `--variant gpu` is the default. The exact
command behind the results below was

```bash
./remote/kaggle_run.sh --variant gpu
```

Useful knobs (all in `--help`):

| flag | why |
|---|---|
| `--only kda_vs_fla` | run one probe; everything else is recorded as a skip |
| `--kda-timeout 3600 --session 120` | give the fla comparison an hour (see below) |
| `--item-timeout 600` | tighten every probe |
| `--battery-budget 1800` | stop launching new probes after 30 min of battery |
| `--session 30` | the cap Kaggle enforces server-side; one full run needs ≥ 75 |

Results land in `out/kaggle/<timestamp>-<variant>/`.

## What comes back

`kaggle kernels output` in CLI 2.2.4 downloads **loose files, not a zip** (the
driver handles both shapes, and retries three times because one flaky TLS
connection to `kaggleusercontent.com` aborts the whole call). Into
`out/kaggle/<stamp>-<variant>/`:

```
SUMMARY.md              the table: every item, verdict, seconds, and why it skipped
results.json            the same, machine-readable
codegen/kernel*.txt     the CUDA source TinyGrad emitted for one KDA call
logs/00-environment.txt nvidia-smi, TPU/LD_LIBRARY_PATH/KAGGLE_* env, disk, cpus
logs/01-pip-freeze.txt  exactly what was installed, and which interpreter
logs/<probe>.log        each probe's full stdout+stderr, one file per probe
<slug>.log              the runner's own stdout, as JSON-lines, for the whole session
run.log                 everything this script printed
```

The payload source and the 23 MB of regenerated `probes/oc/cells_*.bin` stay
out of the zip: the notebook unpacks to a temp dir and writes only artefacts to
`/kaggle/working`.

## How the code gets there (and why it is not a download)

`kaggle kernels push` uploads **exactly one file**: the notebook named by
`code_file`. Sidecars in the kernel directory are never uploaded — verified
against `KaggleApi.kernels_push` in 2.2.4, which reads the metadata, reads the
code file, strips outputs and POSTs it as text. So the ~70 KB of probe code
travels as a **Kaggle dataset**,
`tenghuzhang/mc-cuda-tinygrad-payload`, attached to the kernel with
`dataset_sources` and mounted read-only by Kaggle itself. The notebook walks
`/kaggle/input` for `payload.tgz`, checks its sha256 against `payload.sha256`,
checks every file against the hashes in `MANIFEST.json`, and only then runs
anything.

Alternatives, and why not: this repo has no git remote and is not published, so
there is no public URL that exists; base64 in the notebook works but makes the
payload unreviewable and rewrites the notebook on every change. netherite *is*
public, so the notebook sparse-clones it itself (`blaze`, `docs`).

`kernels/pack.py` builds the archive; `kernels/payload/payload.tgz` is
generated and can be deleted. The driver repacks and republishes before every
run, so a result always corresponds to the tree that produced it — the sha256 is
printed in `SUMMARY.md`.

## Layout

```
kernels/
  pack.py               repo -> payload.tgz + sha256 + MANIFEST.json (deterministic)
  build_notebooks.py    generates the .ipynb files from readable source
  validate_notebooks.py metadata + nbformat + every cell compiles + source-list newlines
  local_dryrun.py       execs the notebook's own cells here, no token, no GPU
  payload/              generated: payload.tgz, payload.sha256, dataset-metadata.json
  probe/                probe.ipynb + kernel-metadata.json   (GPU, T4, internet)
  probe-tpu/            tpu-facts.ipynb + kernel-metadata.json (TPU, internet)
  smoke/                smoke.ipynb: 30-second test of the whole loop
```

Edit the generator, not the `.ipynb`: `kernels/build_notebooks.py` embeds the
cells as ordinary Python, which is reviewable, and the driver regenerates and
re-validates them before every push.

Two things the validator checks that are easy to get wrong and cost a real run
each: nbformat `source` lists must carry their own newlines (the CLI joins them
with `"".join`, so a list without them becomes one unreadable line on the
runner), and every code cell must `compile()`.

## Quota

Live, from `kaggle quota` on 2026-09-27:

| resource | weekly total | used after 5 runs |
|---|---|---|
| GPU | 30.00 h | 1.72 h |
| TPU | 20.00 h | 0.00 h |

The TPU figure is what the API reports; 20–30 h/week is the commonly quoted
range for the free tier. Sessions can run for hours — we cap ours at 75 min by
default and 40 min for the TPU facts kernel. A full GPU run is **~50 min of
wall time, ~0.85 h of quota**; four runs fit in a week with room to spare. The
cost discipline is three independent bounds: the session cap Kaggle enforces,
the poll loop's own deadline, and the notebook's battery budget, which stops
launching new probes rather than being killed mid-probe.

The free GPU queue is real: a kernel pushed at 23:19 sat `QUEUED` for 29
minutes, while a TPU kernel started within a minute. Queue time is not session
time, which is why `--queue-wait` (90 min) is much larger than `--session`.

## The T4 is sm_75, and that is the whole story

`nvidia-smi` on the Kaggle GPU image, measured:

```
Tesla T4, 15360 MiB, 7.5        # compute capability 7.5, 15.36 GB, and there are TWO of them
torch 2.10.0+cu128, triton 3.6.0, flash-linear-attention 0.5.2, tinygrad 0.14.0, numpy 2.0.2
bf16 tensor cores: NO (sm<80)
```

Consequences for this comparison, all of them real:

- **No bf16 tensor cores.** flash-linear-attention's KDA uses a tensor-core MMA
  path on sm_80+ and falls back to a non-MMA path below that. On a T4 the
  hand-written Triton kernel is measured on its *slow* path, so any "fla beats
  TinyGrad by Nx" number here is a floor on the ratio, not the ratio an L4 or
  H100 would give. `remote/preflight.py` prints this line and the notebook
  records it, so a result can never be quoted without it.
- **15.36 GB, not 24.** The TinyGrad port's `[B,NT,HV,C,C,K]` intermediate is
  8.2 GB at the target shape, so the interesting shapes are near the edge and
  OOM is a plausible outcome. `kda_vs_fla.py` reports predicted OOMs as data
  points instead of crashing.
- **sm_75 PTX, so generated code is directly comparable to the local Metal
  study** — the cstyle renderer's `a*(1/b)` rewrite shows up verbatim in the
  captured CUDA source.
- Two T4s are allocated but every probe here uses device 0.

Measured on that T4 (2026-09-28, one run, `out/kaggle/20260928-014248-gpu/`):

| probe | verdict | s | what it found |
|---|---|---|---|
| preflight | ok | 53.9 | 2x T4, 32.9 GB RAM, tinygrad DEFAULT=CUDA, KDA port relerr 2.3e-07 |
| test_ref | ok | 0.4 | 17 float64 reference tests |
| test_tiny | ok | 131.8 | 6/6 PASS, worst relerr 8.6e-06 vs the float64 recurrence |
| codegen_diff | ok | 2.1 | generated CUDA vs the hand-written Triton |
| kda_vs_fla | TIMEOUT | 1800.7 | dominated by triton JIT, not by the measurement |
| perf_kda | ok | 317.6 | 12,082 kernels/call, 215.9 ms GPU, 20.3 s wall, 1% GPU / 99% python |
| replay_kda | ok | 349.5 | graph replay: the python tax |
| cost | ok | 0.1 | 96.6 GFLOP, 37.5 FLOP/byte, 7.9x memory bound on an H100 ridge |
| fp32_correctness | ok | 2.3 | float32 arithmetic on CUDA |
| divide_precision | ok | 48.9 | **CUDA: 27.13% of float32 divides are 1 ULP wrong, bit-for-bit `a*fl32(1/b)`; CPU 0.00%.** 27.13% is one divisor law's worth at N=200,000, not a rate for `/` — see `derivation/divide.py:published_rate_table` |
| gather_floor | ok | 18.5 | 64-element gather costs flat to 4 M source, 315 us at 67 MB |
| gather_scale | ok | 10.2 | indexed load is O(output) |
| real_gather | ok | 2.0 | the custom_kernel replacement does not compile; failure documented |
| oc_check | ok | 184.3 | **41,472 pixels, 0 block-id mismatches**, depth 0.022%, edge 0.007% |
| oc_report | rc=2 | 168.4 | pre-existing repo bug, see below |
| oc_port_only | skip | 0.0 | battery budget exhausted; measured in the previous run instead |

Two of those deserve a note:

- **`oc_report` fails everywhere, not just here.** `probes/oc/oc_report.py`
  sets `B = 32` and reads `cells_0.bin` … `cells_31.bin`, but the `cells`
  target in `probes/oc/Makefile` only dumps 8 seeds (0, 3, 12345, 1, 2, 4, 5,
  6, 7). `make report` therefore cannot succeed on any machine until one of
  those two numbers changes. `oc_check` is unaffected and is the one that
  carries the correctness number.
- **`kda_vs_fla` does not finish in 30 minutes on a T4.** Its log is mostly
  triton's JIT compiling launchers; the oracle self-check passes and it reaches
  the correctness stage before the cap. It is being rewritten, so the number
  will move. To give it a real window:

  ```bash
  ./remote/kaggle_run.sh --only kda_vs_fla --kda-timeout 5400 --session 120 --battery-budget 6000
  ```

## The TPU lane

`--variant tpu` pushes `kernels/probe-tpu/tpu-facts.ipynb`, which reports device
facts and nothing else: `/dev/accel*`, `libtpu`, the TPU/PJRT environment, what
`jax` sees, what TensorFlow's cluster resolver says, and what TinyGrad's device
list contains. It installs nothing and attaches no dataset, so it costs seconds.

**Measured 2026-09-27: it is reachable, and it did not get a TPU.**

```
kaggle inputTrue  []
nvidia-smi        unavailable
/dev/accel*       No such file or directory
libtpu            No such file or directory
TPU env           (no TPU_/PJRT_/XLA_ variables)
jax 0.7.2         default_backend=cpu, devices=['cpu'], device count 1
tf logical TPUs   []
tf cluster        ValueError: Please provide a TPU Name to connect to.
```

The metadata asked for TPU (`enable_tpu: true`, `machine_shape: TpuV3-8`) and
the push was accepted, but the session ran on a plain CPU container — the
runner served the TPU slot as a CPU box. So the TPU lane is **not** usable yet,
and the caveats stack up on top of that:

- TPU v3-8 is a **different lane**, not a second GPU: the compiler is XLA, the
  kernels are SPMD over 8 cores, and TinyGrad's TPU backend is a separate code
  path from its CUDA one. Numbers from it cannot be compared to the T4 numbers
  without saying so, and the "what a compiler cannot infer" question has to be
  re-asked in XLA's terms.
- Nothing in the current battery is written for it: `kda_vs_fla.py` is
  torch/CUDA-only, and the probes that hook `Device.compiler` expect a
  GPU-shaped codegen path.
- `machine_shape: TpuV3-8` is a guess. The CLI documents `TpuV5E8`; the push was
  accepted, so the name is at least not rejected, but the run shows the field
  did not do what it says. Worth an experiment with a different `machine_shape`
  before anything is built on this lane.

Treat the TPU lane as unproven infrastructure: the plumbing is ready, the
accelerator is not.

## What happens when something is missing

Every step is independently guarded, and the failure is recorded in
`SUMMARY.md` rather than raised — a raising cell stops the notebook, Kaggle
marks the run `ERROR`, and the output zip is lost. That is not theoretical: it
happened twice during bring-up (a `TypeError` in the timeout handler 30 minutes
into a run, and an `EXDEV` from `os.replace` across `/tmp` → `/kaggle/working`
in the last cell). Both are fixed, and `kernels/local_dryrun.py` exists because
it finds that class of bug locally for free.

Concretely, on the Kaggle GPU image:

| missing | what happens |
|---|---|
| `payload.tgz` not mounted | every probe is recorded `skip: payload has no ...`; the run still COMPLETEs and still returns the reason it looked in |
| sha256 or MANIFEST mismatch | refuses to run the payload, prints expected vs got |
| `python3 -m venv` (no `python3-venv`) | falls back to the system interpreter and **symlinks `.venv/bin/python` at it**, because `probes/oc/Makefile` hardcodes that path and a half-built venv makes every camera probe die with `ModuleNotFoundError: numpy` |
| `clang` (absent from the image) | `apt-get install clang`; if that fails, the camera C lane is skipped *and* `CC` is left unset, because tinygrad's CPU backend passes clang-only flags (`--target=...-none-unknown-elf -march=native`) and `CC=gcc` fails differently |
| `torch` / `fla` | only `kda_vs_fla` is skipped; everything else runs |
| a probe raises | recorded `rc=N` with its full traceback in `logs/<probe>.log`; the battery continues |
| a probe hangs | killed at its timeout, recorded `TIMEOUT`, and the partial output is kept |
| the whole battery runs long | new probes are not launched once `BATTERY_TOTAL_BUDGET_S` is spent; the summary is still written |

## Testing it without a token

```bash
.venv/bin/python kernels/validate_notebooks.py            # JSON, nbformat, every cell
.venv/bin/python kernels/local_dryrun.py --only preflight,oc_check
.venv/bin/python kernels/pack.py --verify                 # re-verify the archive
./remote/kaggle_run.sh --dry-run                          # the driver's whole path, no network
```

`local_dryrun.py` execs the notebook's own cells in order in one namespace,
with `/kaggle/working` redirected to a temp dir, the repo's `.venv` and
`netherite/` symlinked in, and the cwd set the way the runner sets it. It is
what caught the payload-extraction path bug, the `nvidia-smi` "command not
found"-counted-as-a-GPU bug, and the `shell=True` + argv-list bug that turned
`make check` into a bare `make`. It does not prove CUDA works, that fla
installs, or that the TPU appears — only a real run does that.
