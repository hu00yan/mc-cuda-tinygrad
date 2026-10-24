# portability/ — the compile-time portability lane

**Does the state of the art exist on an old GPU card, or just run slower there?**

Everything in this directory is a **compile-time** measurement. No GPU is involved,
none was sought, and no credentials or paid compute were used. The premise is that
"does it compile for this arch" is a fact `ptxas`/`nvcc`/Triton will answer on the
CPU, and that this axis is more informative than wall-clock — because it can
distinguish *slower* from *a different kernel* from *does not exist*, and only one
of those three is a performance problem.

**The deliverable is [`MATRIX.md`](MATRIX.md).** Start there. Its conclusion in one
line: on a Turing T4 the current flash-linear-attention KDA forward pass compiles
cleanly, runs, and gets correct numbers with **zero** tensor-core instructions —
2052 `fma.rn.f32` where an A100 build has 192 `mma.sync` — and at the Hopper rung
`wgmma` assembles for exactly one `-arch` string, so FlashAttention-3 is not slow
there, it is absent.

## Layout

```
MATRIX.md          the matrix + the blunt conclusion   <- generated from MATRIX.src.md
MATRIX.src.md      editable source; holds <!--TABLES:X--> markers
ptx/               34 PTX witnesses, one mechanism each  (generated, see below)
src/
  gen_witnesses.py     writes ptx/ ; syntax transcribed from CUTLASS 4.8 cute/arch
  selfcheck.sh         PROOF HARNESS: every witness must assemble at its own arch
  run_ptxas.py         the ISA ladder: fixed PTX, 16 targets, 544 cells
  view_ladder.py       render the ladder
  cu/                  CUTLASS / FA2 probe sources
  run_cutlass.py       CUTLASS 4.8.0 by arch, with SASS classification
  triton_ladder.py     Triton mechanism ladder (AOT, no GPU)
  triton_floor.py      the capability floor + the bundled-ptxas ceiling
  kda_ladder.py        fla KDA kernels: ast-lifted upstream source, AOT compiled
  triton_dot_scaled.py tl.dot_scaled attempt (NOT-OBTAINED; kept for the record)
  run_fa.py            FlashAttention arch probe
  build_matrix.py      render + inject every table from the data files
  fa_build.sh          attempt a real FA2/FA3 build (blocked on libtorch)
  sass_mix.sh          one probe, several arches, SASS histogram
  probe_one.sh         dev loop: compile one probe for one arch
  README-images.md     image provenance and what this network cannot fetch
  dev/                 throwaway debugging scripts
out/                 all measured data: *.tsv, *.json, cubins, disassembly, tables.md
subj/                upstream sources fetched read-only (fla, flash-attention)
```

## Running it

```bash
.venv/bin/python src/gen_witnesses.py && bash src/selfcheck.sh   # must print fail=0
.venv/bin/python src/run_ptxas.py && .venv/bin/python src/view_ladder.py
.venv/bin/python src/run_cutlass.py
.venv/bin/python src/run_fa.py
docker run --rm --platform linux/amd64 -v "$PWD:/w" -e TRITON_CACHE_DIR=/tmp/tc \
  portab/triton:3.4.0 python /w/src/triton_ladder.py /w/out/triton
docker run --rm --platform linux/amd64 -v "$PWD:/w" -e TRITON_CACHE_DIR=/tmp/tc \
  portab/triton:3.4.0 python /w/src/kda_ladder.py
.venv/bin/python src/build_matrix.py --inject
```

`src/README-images.md` has the image build steps; note that `docker build` does not
work in this environment (BuildKit cannot reach the registry) and `docker commit`
over a `docker run` is the route.

## The one rule this lane held to

No cell claims a compile that was not performed or a refusal that was not observed.
Where a subject could not be obtained the cell reads `NOT-OBTAINED` and says why.
Where a claim rests on reading source rather than on a compilation, it says so in the
same breath — FA2 and FA3 are argued from their own `__CUDA_ARCH__` guards, with the
arch requirements of those guards then measured independently.

That discipline is not fussiness. The single most valuable result here is that
CUTLASS's Hopper code paths **compile without a single diagnostic for sm_80** and
emit a runtime `assert(0)` into a cubin that is *larger* than the working one. A
report written from plausibility rather than measurement would very likely have
stated that as a clean compile-time refusal — and would have been wrong. It compiles.
That is the finding.
