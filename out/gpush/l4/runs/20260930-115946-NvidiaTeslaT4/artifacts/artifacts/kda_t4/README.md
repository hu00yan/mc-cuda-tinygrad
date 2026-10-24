# kda_t4 — flash-linear-attention on a Tesla T4 (sm_75), measured

This directory exists because two notes in this repo say a wall-clock comparison
against `flash-linear-attention` is impossible on the free-tier card, and that
claim is wrong. `fla`'s shared-memory gate does not *refuse* on sm_75 — it
**narrows the autotune config space** and flips dot precision to IEEE fp32
process-globally. So `fla` runs on a T4, badly, and a scoped comparison is
possible. The scope has to travel with the number (ledger C-04).

Nothing here modifies `kda/baseline_pin.py` or any `kda/tiny_kda*.py`. It reads
them. Where this directory disagrees with them, the disagreement is a finding
and is written down rather than patched over.

## The pins

| thing | value | why it is pinned |
|---|---|---|
| `flash-linear-attention` | `0.5.2` | the version `kda/baseline_pin.py` was written against (its own docstring, `:28-30`) |
| `fla-core` | `0.5.2` | forced by `flash-linear-attention==0.5.2`'s METADATA (`fla-core==0.5.2`) |
| `tinygrad` | `0.14.0` | what `notes/10-t4-results.md:5` and the local venv carry |
| card | Tesla T4, sm_75, 65,536 B optin shared memory | free Kaggle tier; no paid compute anywhere in this directory |

`fla-core==0.5.2` is the **latest** release of both distributions as of this
write, so "pin 0.5.2" and "install latest" coincide today. They will not
coincide tomorrow, which is why the version is in the log and not in a
comment.

`torch` and `triton` are whatever the Kaggle image ships; `probe_env.py` records
them and every report carries them, because `fla` reads the triton version
(`IS_GATHER_SUPPORTED` is `hasattr(triton.language, 'gather')`,
`fla/utils/_device.py:138`) and refuses below 3.3.0.

## Result, in one line

`fla` **runs** on a T4. `kda/baseline_pin.py` exits **INVALID** with **zero
UNKNOWNs**, the sole cause being 65,536 B of opt-in shared memory against
166,912; the autotune space collapses from 12 configs to 4 (`BV=[32]`,
`num_stages=[1,2]`, `num_warps=[2,4]`) and autotune picks
`BV=32, num_warps=4, num_stages=1`; `TRITON_F32_DEFAULT` goes `None → 'ieee'`
across `import fla`; and **0 tensor-core instructions** appear in any of the 45
cubins `fla` compiles on the card, against 8,192 `FFMA` in the state kernel
where the same source at the same config on sm_80 emits 128 `HMMA` and 0 `FFMA`.
Full account, with the scope on every number, in **`notes/30-fla-on-t4.md`**.

## Files

| file | what it does | costs GPU quota |
|---|---|---|
| `INSTALL.sh` | pinned pip install, idempotent, logs exactly what it did | no (still inside a session) |
| `probe_env.py` | the box: versions, both devices, toolchain, fla file:line checks | seconds |
| `pin_t4.py` | runs `kda/baseline_pin.py` as a subprocess with `--strict` and `--json`, records exit codes | no |
| `gate_observed.py` | **the config collapse, observed**: `check_shared_mem()`, `max_shared_mem` from 4 sources, the live autotune config space *and* the config autotune actually picked, tf32, `TRITON_F32_DEFAULT`, and a functional dot-precision probe read out of the PTX | one compile |
| `time_t4.py` | scoped wall clock: `fla` vs whichever `kda/tiny_kda*.py` ports exist | yes, this is the expensive one |
| `sass_t4.py` | the reference's real SASS: the cubin `fla` itself compiled on this card, plus an AOT per-arch sweep with the same toolchain | a few compiles |

`out/` holds everything the card produced. It comes home with the run, so every
number in `notes/30-fla-on-t4.md` has a log path next to it.

## The three rules this directory is built around

1. **Re-derive, never transcribe.** Every figure is produced by a script in
   here. Nothing is copied out of a note.
2. **Scope on the same line as the number.** Every `fla` timing carries its
   degraded-configuration label inline, because the number without the label
   is a claim about `fla` that the measurement does not support.
3. **A passing correctness gate is not evidence that the baseline was the
   right artifact** (`notes/06` §5b: C-14 and C-15 were both invalidated by
   porting the wrong thing while passing every test). So the agreement check
   here is reported next to, and separately from, the pin — never as support
   for it.
4. **Never write scratch inside this directory.** It gets picked up by the
   payload, and the next push is then refused by gpush's mass-deletion guard
   (removing the scratch would prune 93% of the remote manifest). Scratch for
   this work lives in `out/kda_t4_scratch/`. The refusal is recorded in
   `notes/30-fla-on-t4.md` §7a.

## Running it

```bash
./gpush/gpush kaggle run \
  --dataset tenghuzhang/kda-t4-payload --kernel tenghuzhang/kda-t4 \
  --include kda_t4 --include kda --include tools \
  --session 80 --timeout 4000 --cmd bash kda_t4/run.sh
```

**Use those two slugs, not `tenghuzhang/gpush-smoke`.** The other slugs in this
account are shared with agents working in this repository concurrently, and a
shared slug will mount somebody else's dataset version while still printing
`VERDICT: payload matches the fingerprint that was pushed` — because the
fingerprint it checks is the one baked into the notebook that ran. Measured, not
hypothetical; the incident is in `notes/30-fla-on-t4.md` §7a.

`run.sh` takes stage names as positional arguments, so a cheap re-run of just
the static half is `bash kda_t4/run.sh install gate sass` (about 8 minutes of
card time; the `time` stage alone is ~36).
