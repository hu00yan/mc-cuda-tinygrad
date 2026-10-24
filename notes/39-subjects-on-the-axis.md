---
kind: reference
---

# 39 — Can the two subjects be placed on the axis, and why one of them cannot be

**The question.** `notes/33` measured TinyGrad's own performance envelope as a
seven-rung ladder of synthetic operators, and `notes/35` §8 presents that shape as
the study's central claim. But **neither subject of the study is on that ladder.**
The ladder is a description of the framework across operator *shapes*; the two
subjects are whole programs. Placing them on it is what would make the envelope
mean something about them rather than about the rungs.

**The answer is that one subject can be placed and the other cannot, and the
reason is a property of the reference rather than of either implementation.** That
is worth a note on its own, because the failure mode it describes — comparing a
ratio against a strong reference with a ratio against a weak one and reading the
difference as a property of the framework — is the specific artefact this study
exists to avoid, and I was one step from committing it.

All arithmetic below is from `notes/33` §7 and the two T4 runs, re-derived in
`axis/`-independent scratch; no new GPU time was spent on this note, and none of
the numbers in it are new measurements. They are placements of already-measured
ratios, which is why every one of them carries its source.

---

## 1. The envelope, and where each subject falls

`MEASURED`, one Tesla T4, one run, both arms sharing one `cuEvent` path in one
process (`notes/33` §7):

| rung | what it demands | tg/ref | tg kernels | % of memory floor |
|---|---|---|---|---|
| `gather` | one random address; the operator is one multiply | **0.96×** | 1 | 13% |
| `ew` | nothing — the control | **0.99×** | 1 | 105% |
| `gemm16` | reuse, tensor core present | 1.84× | 1 | — |
| `red` | one reduction, no reuse | 1.92× | 1 | 55% |
| `gemm32` | reuse, **no fp32 tensor core on sm_75** | 2.69× | 1 | — |
| `trisolve` | a carried chain **with** a dense contraction | 2.97× | 585 | — |
| `scan` | a carried chain, **no** contraction | **25.96×** | 3 | 3% |

**The envelope's worst rung is 25.96×.** The two subjects:

| subject | ratio | where it falls |
|---|---|---|
| `mine.cu`, best port (`incremental`), batch 1024 | **11.4×** | **inside** — 0.44× the worst rung |
| KDA, opt port vs `fla` on T4, B1 T512 H4 | 101.6× | outside, 3.92× the worst rung |
| KDA, opt port vs `fla` on T4, B1 T1024 H8 | 64.0× | outside, 2.47× |
| KDA, opt port vs `fla` on T4, B2 T1024 H8 | 36.3× | outside, 1.40× |
| KDA, opt port vs `fla` on T4, B2 T2048 H8 | 28.1× | outside, 1.08× |

**So: the DDA subject sits comfortably inside the envelope, and the KDA subject
sits outside all four of its measured shapes, by 1.08× to 3.92× of the worst rung
the axis found.** And unlike the DDA subject it does not improve with scale — it
*closes* on the envelope as the shape grows, which is the wrong direction for a
deficit that is a fixed property of the implementation.

## 2. The dimension the ladder does not have

**The obvious candidate is program size — the number of low-level operations the
framework emits for one user-level operator — and the ladder's own data refutes
it as a predictor before it can be proposed.** The `kernels` column runs 1, 1, 1,
1, 1, 585, 3. The five single-kernel rungs span 0.96× to 2.69×, and then the
**585**-kernel rung comes back at **2.97×** while the **3**-kernel rung is
**25.96×** — a factor of 8.7 in the wrong direction. Launch count is not a
predictor on this ladder, and `notes/36` independently found the same thing from
the other side: the decisive DDA rewrite removed 18 of 22 graph operations and
**zero** launches.

**The dimension that is actually missing is narrower and it is a count, not a
shape.** `notes/33`'s `red` control prices *one* instance of the C-18 rank-1
habit — `(x*w).sum(-1)` in 1 kernel against `K-1` dependent rank-1 updates — at
**2.26×**, at `R=65536, K=256`. The KDA faithful port contains **4 032** of them.
The ladder prices the idiom and nothing prices the count, which is why the axis
can bound every operator it can write down and still miss the larger of the two
subjects entirely.

**A rung that would fill this is therefore not a new operator. It is one operator
repeated until the framework's emitted-program size crosses whatever budget it
has** — the same shape of question as the DDA fusion crossover, which is a
*program-size* threshold and not an operator property, and which the axis does not
contain either. Nothing in this note measures that, and the number it would need
does not exist yet.

## 3. Why the KDA subject cannot be placed at all, which is the actual finding

**The two sets of ratios are not the same kind of measurement, and the difference
is in the reference.**

| | what the axis compares against | what KDA compares against |
|---|---|---|
| language | hand-written CUDA | `fla`'s hand-written Triton |
| config space offered | full (tuned per shape) | **4 of 12** on sm_75 — `BV=[32]`, stages `[1,2]`, warps `[2,4]` |
| dot precision | as written | **`ieee` fp32** — no `tf32` path on this card |
| tensor cores | present where the ISA has them | **0 tensor-core instructions across all 45 real cubins** |
| registers / spill | the fp16 GEMM reference: 776 instructions, 80 HMMA, **zero spills** | `BV=32,w=2`: 19 328 SASS, 8 192 FFMA, 64 registers, **12 806 / 14 588 B spill** |

**`fla`'s T4 configuration is a spilling kernel, and the axis's references are
not.** On `cc=89` the same Triton source emits 1 944 SASS, 64 HMMA and **24 B**
of spill (`notes/30` §3's own AOT sweep) — two orders of magnitude less spill on a
card the same code was not degraded for.

> **A ratio against a spilling reference and a ratio against a non-spilling
> reference are not the same measurement, and the difference between them is not a
> property of the framework.** The 28–102× therefore cannot be placed on the axis
> and cannot be compared with 25.96×, and the honest position is that the KDA
> subject's deficit is **unquantified against a competent reference** on this card
> — not that it is 28–102×.

This study's own §6 law predicts the direction: a reference that spills is paying
the register-pressure cost, so the ratio is inflated by an amount that is a
property of the *card's* shared memory, not of either kernel. `notes/30` §5
already records the clean-versus-degraded gap as an **open unknown, not
estimated**; this note does not estimate it either, and §1's KDA rows should be
read as *"this is what the ratio was against a degraded reference"*, not as a
position on the axis.

**I nearly got this wrong by doing the arithmetic first and checking the reference
afterwards.** A 3.92×-past-the-envelope result is attractive — it says the axis is
incomplete — and it would have gone into `notes/35` §8 as a finding. It is not a
finding about the framework. It is a finding about the card.

## 4. What this changes, and what it costs

- **`notes/35` §8 gains a limit on its own claim.** The envelope describes
  operator shapes; it contains the DDA subject and does not contain the KDA
  subject, and for the KDA subject the comparison is not yet available. §8's "at
  parity where the framework only moves data, about 2× where it contracts, and one
  cliff at a literal with a TODO next to it" is a statement about seven rungs and
  about one program, and should not be read as a statement about both subjects.
- **The 4090 stops being an optional extra.** It was recommended because it is the
  cheapest card that makes class (iv) observable at all (`notes/35` §7). §3 gives
  it a second and larger job: on `cc=89` the `fla` reference is not degraded, so
  the KDA subject becomes comparable to the axis for the first time. **Without a
  card on which the reference runs properly, one of the study's two subjects has
  no valid measurement at all** — and no amount of free T4 time changes that,
  because the blocker is 65 536 B of shared memory against a requirement of
  101 376 B, measured, in `fla/utils/_device.py:181-185`.
- **The ladder is worth extending, and the extension is cheap on a card we have.**
  A rung of *one operator repeated past the framework's program-size budget* would
  test the count hypothesis that §2 identifies, needs only a T4, and would say
  whether the axis's silence about program size is a limit of the rungs chosen or a
  real property of the framework.

## 5. Falsifiers

Stated before the measurement, so they are not chosen afterwards.

- The KDA ratio, recomputed against `fla` on `cc=89`, lands **inside** the
  25.96× envelope ⇒ §3's reference-quality argument is wrong and the subject simply
  sits above the ladder. (This is the outcome that would vindicate the placement I
  nearly made.)
- A program-size rung reaches parity with its hand-written reference ⇒ the missing
  dimension is not program size, and §2's hypothesis is wrong.
- A program-size rung reproduces the fusion-crossover *shape* — a threshold, with
  the port better on one side and worse on the other — ⇒ program size and fusion
  are one mechanism, and the DDA crossover belongs on the axis as a rung.
- The `fla` `cc=89` kernel still spills materially ⇒ the 4090 does not fix the
  reference and the KDA arm stays unquantified, which makes the search for another
  card a priority instead of a detail.

## Sources

| claim | source |
|---|---|
| the seven rungs, both columns, the memory floors | `notes/33-tinygrad-axis.md` §7 (`MEASURED`, one T4, one run) |
| the `red` control at 2.26× and the 4 032 rank-1 reductions | `notes/33` §7 controls; `notes/29-kda-faithful-port.md` |
| `fla` on T4: 4 of 12 configs, `ieee`, 0 tensor-core instructions, 19 328 SASS, 12 806/14 588 B spill; `cc=89` 1 944 SASS / 64 HMMA / 24 B | `notes/30-fla-on-t4.md` §2, §3, §7a |
| the fp16 GEMM reference at 776 instructions, 80 HMMA, zero spills | `notes/33` §7 gate 2 |
| the DDA subject at 11.4× and the launch-count refutation | `notes/36-rung-timings.md`; `notes/35` §5 |
| the shared-memory gate at `fla/utils/_device.py:181-185` | `notes/34-sibling-references.md`; `kda/baseline_pin.py` `CARD_GATE` |
| the register-pressure law that predicts §3's direction | `notes/35` §6 |
