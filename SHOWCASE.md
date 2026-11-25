# Showcase / cleanup plan

Goal: a GitHub repo you can put on a résumé without shipping 1.4 GB of run
debris or implying “we beat hand-written CUDA.”

## Recommended public shape

**Option A — slim showcase (preferred for résumé link)**

New public repo (or a cleaned `main`) containing only:

```
README.md
LICENSE                    # add MIT/Apache-2.0 once chosen
minecu_ref/
minecu_tg/
minecu_tg_opt/
notes/17-minecu-spec.md
notes/23-good-vs-naive.md
notes/35-ladder-report.md
notes/36-rung-timings.md
notes/06-corrections-ledger.md
out/minecu_tg/**/SUMMARY.txt
out/minecu_tg/**/obs_equivalence.json
out/minecu_tg_opt/**/SUMMARY.txt
out/minecu_tg_opt/**/obs_equivalence.json
```

Optional second section (if space): `probes/oc/`, `kda/`, `derivation/divide.py`
as “compiler findings,” not as the headline.

**Option B — research archive (this tree as-is, cleaned)**

Keep the full lab, but:

1. Never track `out/` binaries, `.seiso_cache/`, `.refdrv*`, `.venv/`
2. Leave `.agent-scratch/` deleted (evidence is in commit `9e6767f`)
3. Move side labs under `archive/` or drop from the public remote:
   `experiment_b/`, `crossover/`, `cutlass_lab/`, `matmul/`, `kernels/`,
   `archlab/`, `portability/`, `research/`

## Delete / untrack (safe)

| path | why |
|---|---|
| `.refdrv_preflight`, `.refdrv_test`, `.refdrv.sass` | local binaries (~5 MB), regenerable |
| `.seiso_cache/` | tool cache |
| `.venv/` | already ignored |
| most of `out/gpush/` | Kaggle dump copies |
| `__pycache__/` | regenerable |

## Do not delete yet

| path | why |
|---|---|
| `minecu_tg_faithful/`, `minecu_tg_knobs/` | negative results that support the résumé story |
| `notes/` | the science; résumé claims cite these |
| `gpush/` | Colab/Kaggle runner for the next performance loop |
| curated equivalence JSON under `out/minecu_*` | proof of byte-identity |

## GitHub publish steps

1. Pick Option A or B.
2. Add `LICENSE` + rewrite README (done for B).
3. `git remote add origin …` (this clone currently has **no remote**).
4. Push a clean history, or use `git filter-repo` / orphan branch if `out/`
   blobs already bloated history.
5. Pin one Kaggle/Colab notebook URL in the README for reproducibility.

## Performance conjecture (for résumé; not yet re-measured)

| path | expected share of Infatoshi CUDA render | why |
|---|---|---|
| Pure TinyGrad, no custom ops (today) | **~9–11%** measured | register spill / fusion model; knobs inert |
| Pure TinyGrad, more scheduler work | **~15–25%** optimistic ceiling | still not a scalar fused raymarcher |
| TinyGrad + handwritten DDA op (vLLM-style) | **~80–100%** | you re-enter the same CUDA kernel design space; ceiling ≈ the original |
| Beat the original on the same GPU / algorithm | **unlikely** | same DDA, well-written fused kernel is already near that algo’s peak on T4 |

**Hard ceiling:** matching the original is the realistic best case if you allow custom ops. Beating it needs a better algorithm or a better GPU, not a better port of the same kernel.

**Résumé-safe advantages vs original (true without more work):**
- Correctness-gated port + ablation that separates *algorithm* from *compiler*
- Python / TinyGrad stack: easier env logic outside the hot path; multi-backend story (Metal/AMD) as potential, not measured claim
- Publishable negative result: “automatic codegen ≠ hand CUDA on irregular control flow”

Do **not** claim H100/B300 wins until measured on those cards.
