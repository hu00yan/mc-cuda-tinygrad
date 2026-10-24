# minecu_ref — the hand-written side of the comparison

`Infatoshi/mine.cu`, vendored so it can be compiled, inspected and measured
without touching `netherite/`, plus a benchmark harness that produces the
hand-written baseline the TinyGrad lane is measured against.

The study compares compiler-generated GPU code against hand-written GPU code.
This directory is the "assembly" side. Nothing here is optimised for the study;
it is the reference exactly as its author wrote it, plus instruments.

```
minecu_ref/
  src/kernels.cu          3245 lines, vendored VERBATIM from
                          netherite/mine.cu/src/kernels.cu
  bench/bench.cu          standalone CUDA benchmark, no Python, no PyTorch
  bench/dda_model.py      CPU reference of the render kernel. Measures NOTHING
                          about any GPU: it counts DDA steps and checks the
                          49.6M claim against an instruction-issue bound
  build.sh                containerised build for one or more -arch values
  kaggle/run.sh           one command: build, push to Kaggle, poll, fetch
  kaggle/make_notebook.py assembles the Kaggle notebook from the two sources
  buried_agent_fix.md     the degenerate-shape bug, the fix, and the numbers
                          on both sides of it
  build/                  generated: objects, binaries, SASS, censuses, manifest
```

## Provenance

| file | sha256 | note |
|---|---|---|
| `netherite/mine.cu/src/kernels.cu` | `c6ed8f06ee64d40a8a8ab7ebce3b3ceea1e3ac93f9ee50cfc1e63d3f090d878f` | upstream, commit `ec2085e` |
| `minecu_ref/src/kernels.cu` | `c6ed8f06ee64d40a8a8ab7ebce3b3ceea1e3ac93f9ee50cfc1e63d3f090d878f` | **byte-identical, unmodified** |
| `netherite/mine.cu/src/bindings.cpp` | `53b0d8c75f8da1986518200f8501a9d6069d47beb09fa602021f2a63f6f4b862` | not vendored; needs libtorch |
| `netherite/mine.cu` remote | `https://github.com/Infatoshi/mine.cu` | public |

`build.sh` re-checks that `minecu_ref/src/kernels.cu` and
`netherite/mine.cu/src/kernels.cu` hash the same on every run, so a silent edit
to the reference is caught rather than measured.

The upstream repo is a PyTorch extension. **This directory does not need
PyTorch.** Every launcher in `kernels.cu` is `extern "C"` and takes raw
pointers, so `bench/bench.cu` declares them itself and links only
`libcudart`. That is what makes the baseline reproducible in a lean CUDA
container rather than needing a torch wheel.

## Build (no GPU needed)

```bash
./build.sh                # sm_75 (the T4 target) and sm_86
./build.sh sm_75          # just one
./build.sh --sweep        # sm_75 sm_80 sm_86 sm_89 sm_90
```

nvcc flags are copied from upstream's `setup.py` (`-O3 --use_fast_math`) so the
binary is the same optimisation the project ships.

Produces, per arch, under `build/`:

- `kernels_sm<arch>.o`, `minecu_bench_sm<arch>`
- `kernels_sm<arch>.sass` — full SASS
- `resources_sm<arch>.txt` — `ptxas -v`: registers, spills, smem, barriers
- `sass_census_sm<arch>.txt` — instruction mix plus a portability gate
- `build_sm<arch>.log` — the full nvcc transcript
- `MANIFEST.txt` — sha256 of every input and output

The portability gate is the sm_75 answer: it fails the build if `HMMA`,
`HGMMA`, `LDGSTS`, `REDUX`, `ELECT`, `BAR.RED` or `F2FP` appear at all. **It
does not appear.** See `notes/17-minecu-spec.md` §6.

## Run (needs a GPU)

Locally, if you have an NVIDIA card and a GPU-enabled container runtime:

```bash
docker run --rm --gpus all --platform linux/amd64 \
  -v "$PWD":/w -w /w archlab/compile:12.9 \
  ./build/minecu_bench_sm_75 --selftest-only
```

On Kaggle's free T4, one command:

```bash
./kaggle/run.sh --dry-run     # build + assemble, no network
./kaggle/run.sh --build-only  # build only
./kaggle/run.sh               # build, push, poll, fetch, print
```

Prerequisites: `kaggle auth login` (or a token at `~/.kaggle/access_token`),
and that the Kaggle image has `nvcc`. Both are checked and both fail loudly.

### What `run.sh` will print

```
=== minecu_ref T4 baseline: build for sm_75 ===
  sm_75 SASS census passed
=== assemble the notebook ===
  wrote kaggle/kernel/minecu_t4.ipynb (227243 B)
    kernels.cu sha256 c6ed8f06...
    bench.cu   sha256 ...
  notebook valid: 3 cells, N source chars, all code cells compile
=== push tenghuzhang/mc-minecu-t4-baseline (session cap 25m) ===
=== wait (queue up to 120m, session cap 25m) ===
  [HH:MM:SS] COMPLETE
=== download ===
=== what came back ===
    SUMMARY.txt
    log_selftest_w16.txt
    log_headline_w16.txt
    ... one log per battery entry ...
    kernels_sm75.sass
    minecu_bench_sm_75
--- SUMMARY ---
  ... the full run log ...
done. results in out/minecu_ref/<stamp>/
```

Inside `SUMMARY.txt`, in order: the provenance block (both payload sha256
values, or a hard stop), the nvcc version and the `-arch=sm_75` compile, the
SASS census (`HMMA=0 HGMMA=0 LDGSTS=0 ...`, `arch = sm_75`), then one block per
battery entry containing the memory floors, the per-kernel table, the whole-step
table and the roofline position, then a summary table of M env-steps/s per
entry, then a five-line reading guide.

The three `selftest_*` entries run first and are a gate: a FAIL makes the run
exit non-zero and `run.sh` reports it. A timing from a kernel that fails its own
correctness check is not a timing.

### The DDA gate

Correctness and workload are different questions, and the selftest asks both.
`dda_probe_kernel` replays the vendored DDA over the device's own voxels and
basis, counting how many voxels each ray actually tests, and two assertions
follow:

* mean tests per pixel `>= max(2.0, 0.10 * max_steps)`, and
* fraction of pixels hitting their **first** voxel `< 20%`.

A buried eye scores 1.000–1.3 on the first and ~97% on the second, so it cannot
pass at any step cap; the fixed configuration scores 26.4 and 3.0%. A FAIL
means the run prints
`selftest FAILED: no timings reported` and exits 2 **before** the timing
battery. The gate is re-checked after the settle that precedes timing, so a
configuration that drifts into degeneracy mid-run also aborts.

`--measure-degenerate` waives the abort for one purpose: reproducing the
pre-fix numbers for the before/after table. It does not lower the floor and
does not skip the check — it stamps `DEGENERATE` in the run header and in every
timing table, and the battery uses it exactly once.

`SUMMARY.txt` carries a `summary: DDA workload actually timed` table, because a
throughput number means nothing without the workload behind it.

## What is measured, and what is deliberately not mixed

| quantity | what it is | why it is separate |
|---|---|---|
| whole step | one captured CUDA graph, `cudaEvent` around the launch | this is what the project's "49.6M env-steps/sec" is a claim about |
| per kernel | one kernel via its own `extern "C"` launcher | this is what a framework can be compared against |
| memory floor | store-only, load-only, RMW, D2D memcpy, on the same card | a "% of peak bandwidth" claim needs a floor measured on the card making it |

`%` figures are printed against the **measured** floor from the same run, not
against a datasheet number. The write and read rows count one direction; the
copy and memcpy rows count two. Dividing a one-direction figure by a
two-direction row is off by exactly 2×, and the table says so.

## Options

```
-B,--batch N      envs per step       (default 32768, the project's headline shape)
-w,--world N      cubic world edge     (default 16)
   --res N        render resolution    (default 32)
   --max-steps N  DDA step cap         (default 32)
   --ground N     ground height        (default world_y/4, see below)
   --iters N      per-kernel iters     (default 200)
   --graph-iters N graph-step iters    (default 200)
   --min-dda-steps N                   DDA gate floor (default max(2.0, 0.10*max_steps))
   --measure-degenerate               print timings for a below-floor config
   --no-selftest  skip the correctness gate
   --selftest-only  gate and exit
```

### Why the default ground is `world_y/4`

`kernels.cu` has two independent notions of the ground and they do not agree:

| | source | value |
|---|---|---|
| visible surface, top face | `kernels.cu:1823` | `y = ground_height` |
| physics collision plane | `kernels.cu:1566` | `y = world_y / 4` |

`physics_kernel` never reads a voxel — there is no voxel collision anywhere in
the physics path — so the agent always comes to rest at `world_y/4`. If that is
below the visible surface, the agent is **inside solid rock**, and
`render_kernel_uint8_prebasis`, which tests the current voxel before stepping
(`kernels.cu:666`), hits on iteration 0. Measured on a T4 at `world=16,
ground=8`: a mean of **1.291 DDA steps against a 32-step cap**, median 1, with
96.9% of pixels hitting on their first voxel test. The render kernel degenerates
into an obs store. The previous default (`ground=8`) did exactly this at every
world size except 32.

The two agree iff `ground_height == world_y/4`, so that is the default, and it
is self-consistent at every world size. `--ground` remains for reproducing
other configurations; passing a disagreeing one is caught by the DDA gate
rather than silently measured. See `minecu_ref/buried_agent_fix.md`.

## What the numbers are not

The project's 49.6M env-steps/s is **the author's claim on an RTX 3090**
(`docs/benchmarks.md`, batch=32768, world=16, 32×32). It is not reproduced by
anything in this directory yet, and nothing here inherits it. See
`notes/17-minecu-spec.md` §7, where an instruction-issue bound says the claim
is not reachable from this source at that shape, and §8 for what would settle
it.

`bench/dda_model.py` runs on the CPU and times nothing. Its only outputs are a
DDA step count and an image sha256, both reproducible offline. Any GPU timing
comes from `bench/bench.cu` and nowhere else.
