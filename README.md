# mc-cuda-tinygrad

Port of [Infatoshi](https://github.com/Infatoshi)’s GPU Minecraft voxel raymarcher
([`mine.cu`](https://github.com/Infatoshi/mine.cu) / [`netherite`](https://github.com/Infatoshi/netherite))
from hand-written CUDA to [TinyGrad](https://github.com/tinygrad/tinygrad), plus
measurements of where TinyGrad’s automatic codegen stops on irregular operators.

## Status (honest)

| claim | result |
|---|---|
| Correctness | **Byte-identical** to the CUDA reference’s own observation on a Tesla T4 (0 differing pixels out of 4 194 304) |
| Best TinyGrad render speed | **~9–11× slower** than the hand-written kernel in the fused regime (~8–11% of CUDA) |
| Naive first port | ~27–37× slower — mostly a worse algorithm, not “the compiler is slow” |
| Closing the rest via TinyGrad knobs | **Ruled out** on T4 (`NOLOCALS`, `TC_*`, … do not close the gap) |
| Hardware | Measured on **Kaggle Tesla T4 (sm_75)**. L4 requests silently got CPU. No H100/B300 numbers |

One-sentence takeaway: the port is **correct**; the remaining gap is **framework expressiveness / register-window behaviour**, not missing tuning knobs. Details: [`notes/35-ladder-report.md`](notes/35-ladder-report.md), [`notes/36-rung-timings.md`](notes/36-rung-timings.md).

## MineCU packages (the resume deliverable)

```
minecu_hybrid/       ★ production-shaped path: Infatoshi CUDA step + Python RL API
minecu_ref/          hand-written CUDA reference (vendored, hash-pinned)
minecu_tg/           naive TinyGrad DDA + verify + T4 battery
minecu_tg_opt/       algorithmic ladder: naive → int_cursor → flat_gather → incremental → lean
minecu_tg_faithful/  “copy the reference strategy” arm (bit-correct, slower in fused regime)
minecu_tg_knobs/     codegen-knob sweep (mostly inert)
notes/17,23,35,36    spec, ablation, handoff, per-rung timings
```

### Hybrid RL env (recommended to run)

**One** `MineCUEnv`: TinyGrad-shaped control flow, with Infatoshi CUDA as
optional **inline accelerators** (`accel.py`) for the hot render/step — like
`asm` inside C, not a separate product that abandons TinyGrad.

```bash
./minecu_hybrid/build_lib.sh && pip install cupy-cuda12x   # NVIDIA
PYTHONPATH=. python -m minecu_hybrid.smoke                 # auto uses CUDA if .so exists
# force pure TinyGrad: MineCUEnv(accel="off", ...)
```

Obs are low-res RGB; `viewer` writes PPM mosaics (watchable, not the Java client).

**TPU:** not supported.

```bash
# CPU correctness (no NVIDIA needed)
.venv/bin/python minecu_tg_opt/verify_opt.py

# Ladder pricing / SASS (see package README)
.venv/bin/python minecu_tg_opt/ablate.py --batch 1024 --steps 4,8,16,32,64
```

## Secondary probes (earlier in the study)

| probe | verdict |
|---|---|
| Semantic camera (`probes/oc/`) | 100% block-id agreement; TinyGrad float32 `/` is 1 ULP wrong ~26–27% of the time (no user knob) |
| KDA (`kda/`) | Ports completely (~6.6e-06 vs float64); materialises large intermediates a hand kernel never builds |

See [`notes/01-obs-camera.md`](notes/01-obs-camera.md), [`notes/03-kda.md`](notes/03-kda.md), [`notes/DECISIONS.md`](notes/DECISIONS.md).

## Layout (what to keep vs archive)

| keep for a public GitHub showcase | archive / ignore (research debris) |
|---|---|
| `minecu_*`, `notes/`, `derivation/`, `probes/`, `kda/`, `gpush/` | `out/` bulk (1.4 GB run dumps), `.seiso_cache/`, `.refdrv*`, `.venv/` |
| curated `out/**/SUMMARY.txt` + `obs_equivalence.json` | `experiment_b/`, `crossover/`, `cutlass_lab/`, `matmul/`, `kernels/`, `archlab/` side labs |
| this README | deleted `.agent-scratch/` (recoverable from git history `9e6767f`) |

Cleanup plan: [`SHOWCASE.md`](SHOWCASE.md).

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python tinygrad numpy tabulate
```

GPU timings need an NVIDIA card (Colab Pro / Kaggle T4). Apple GPU is enough for
correctness gates and SASS-side audits, not for the MineCU speed claims above.

## Conventions

- A number in `notes/` is **derived**, not transcribed — see `derivation/`.
- GPU device time and wall time are always reported separately.
- Corrections live in [`notes/06-corrections-ledger.md`](notes/06-corrections-ledger.md).
