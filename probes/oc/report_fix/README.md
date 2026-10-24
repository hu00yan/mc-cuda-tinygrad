# report_fix — the `make report` fixture bug, corrected and verified

`probes/oc/` is owned by another agent, so nothing in it was modified. This
directory holds the corrected versions and the patch.

## The bug, in one line

`probes/oc/oc_report.py` hardcoded `B = 32` and loaded `cells_0.bin` …
`cells_31.bin`; the Makefile's `cells` target emits 9 fixtures (seeds 0–7 plus
12345); `.gitignore` documents a third set (`0 1 2 3 12345`). Nothing checked
that the producer and the consumer agreed, so `make report` raised

```
FileNotFoundError: [Errno 2] No such file or directory: 'cells_8.bin'
make: *** [Makefile:16: report] Error 1
```

on every machine, from a clean checkout, 2.5 minutes in — after printing the
whole correctness section.

## Which side was wrong

**The Makefile.** B=32 is the published configuration and re-derives exactly:
`notes/01`'s "at B=32 the region buffer is 170 MB" is
`32 × 113·104·113 × 4 B = 169.98 MB`, and its `1.85 M rays/s` and
`0.08 M rays/s` rows are `73,728 / 39.79 ms` and `73,728 / 952.45 ms`. Dropping
the report to B=8 would have made it pass and silently invalidated all of them.

One trap: `notes/01`'s "1,365 kernels / 101 reduces" is **not** evidence for
B=32. I measured **1365 / 101 at both B=8 and B=32** — those counts are set by the
96-step unroll and are batch-invariant. The B-discriminating evidence is the
`rays/s` rows and the 170 MB buffer.

## Files here

| file | what it is |
|---|---|
| `oc-report-fix.patch` | the fix, as a unified diff against `probes/oc/Makefile`, `probes/oc/oc_report.py`, `.gitignore`, `kernels/pack.py`. `git apply -p1` from the repo root. **Verified to apply cleanly to a pristine copy of all four files and to produce byte-identical output to what `make report` passed with.** |
| `oc_report_fixed.py` | the corrected report, runnable as-is from here against any fixture directory. Identical logic to the patch's version; it just reaches its siblings via `OC_DIR` instead of living beside them. |
| `make_cells.sh` | the corrected `cells` recipe in executable form — `make_cells.sh [outdir] [B]`. The patch's Makefile recipe is the same thing in make syntax. |
| `verify_defs.py` | re-derives 41,472 and 0/9/3 from the fixture files, and checks the corrected report's id/depth/edge categories against `run_oc.py`'s using two independently written loops. |
| `fp32_div_row.py` | the audit finding: `probes/fp32_correctness.py`'s `/` row computed `a/a`, not `a/b`, so the division error rate it published was a self-division figure and understated general division by 1.77x–1.86x on the probe's own draw. Unrelated to the report bug; it was the most consequential thing the audit turned up. Fixed in place; the corrected rate and its divisor law are derived in `derivation/divide.py`. |
| `verified-output-b32-metal.txt` | the corrected report at B=32, run end to end. |
| `verified-make-report-clean-tree.log` | `make report` **rc=0** on a clean tree (no fixtures, no `ref_oc`, no `ref.bin`) with the patched Makefile and report. |

## What was verified, and how

| check | result |
|---|---|
| `make report`, unpatched, this Mac | rc=1, `FileNotFoundError: 'cells_8.bin'` — reproduced **before** any change |
| `make report`, patched, clean tree, no fixtures at all | **rc=0**, full report |
| 33 fixtures regenerated from scratch by the patched recipe | 2.0 s, 33 files, `cells: 33 fixtures, covering seeds 0 3 12345 and 0..31 for B=32` |
| the 9 pre-existing fixtures vs fresh output | sha256 **identical**, all 9 |
| all 33 fixtures distinct | 33 distinct sha256 |
| correctness, corrected report vs `run_oc.py` | per-frame mismatch vectors **element-wise identical**: 0 / 9 / 3 of 41,472 |
| 41,472 re-derived from the `ref.bin` header + body length | `165888 / 9216 = 18` frames × 2304 = 41,472 ✓ |
| the ±1 depth claim, now measured instead of asserted | `max |Δdepth| = 1`, differing edge values `{0.0, 1.0}` |
| `ref.bin` rebuilt with `--poses 64` | guard fires in one line, rc=2; unpatched, it would have reported 442,368 pixels from 18 of 192 frames with the seed labels also wrong |
| missing fixtures, B=32 | refused in **0.12 s** naming all 24 missing seeds and the command to build them, instead of 150 s then a traceback |
| the `%` in the Kaggle log | `9/41472 = 0.022%`, `3/41472 = 0.007%` — matches `oc_check.log` exactly |

## Re-measured vs `notes/01`

| | `notes/01` | re-measured (Metal) |
|---|---|---|
| `np_oc` id/depth/edge | 0 / 0 / 0 | 0 / 0 / 0 |
| `tiny_oc` id/depth/edge | 0 · 9 · 3 | 0 · 9 · 3 |
| unroll / mean / p99 / max | 96 / 19.8 / 83 / 84 | identical |
| waste | 4.8x | 4.8x |
| B=32 GPU-only | 39.79 ms, 1.85 M rays/s | 40.76 ms, 1.81 M rays/s |
| B=32 wall | 952.45 ms, 0.08 M rays/s | 1007.78 ms, 0.07 M rays/s |
| kernels / reduces | 1,365 / 101 | 1,365 / 101 (also at B=8) |
| `ref_oc` | 5.67 M rays/s | 5.64 M rays/s |
| B=1 → B=32 wall | 923 → 998 ms (claimed, never measured) | **978.25 → 1021.78 ms, measured**, 1.04x |

## The six changes to the report, none of which moves a number

1. `B` from `$OC_B` (default 32) instead of a literal, so the Makefile's `NENVS`
   and the report are provably the same number.
2. Fixtures resolve against **one** directory. The original resolved the parent's
   files against `$PWD` and `kernel_us`'s subprocess against `__file__`'s dir.
3. A preflight pass before anything is measured.
4. A check that `ref.bin` holds `SEEDS × CANON` frames before indexing it.
5. The narrative claims are derived from the counts. The original printed
   "all id mismatches are depth +-1" (vacuous at id=0) and blamed FMA, a root
   cause `notes/01:43-49` and `notes/02` had already retracted in favour of the
   `sin` LUT bucket boundary.
6. `--b-sweep` measures the flat-wall claim.

**Unchanged deliberately:** every number, and every definition. `id`, `depth` and
`edge` mean what `run_oc.py` and `notes/01` mean by them — a per-pixel `!=`
against the same `ref.bin`, summed over the same 3 seeds × 6 poses × 2304
pixels. Nothing here reinterprets a category.

## Known gap, not fixed

`probes/oc/run_oc.py:93` has the identical blind `load_regions([f"cells_{i}.bin"
for i in range(B)])`. `--bench` defaults to 0 so `make check` never reaches it;
`--bench 32` dies exactly like the report did. It works now only because the
patched recipe guarantees 33 fixtures. Same failure, different file, and the
preflight guard is not shared with it.

See `notes/11-local-suite-integrity.md` for the full audit.
