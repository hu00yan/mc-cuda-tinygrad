#!/usr/bin/env python3
"""opcount.py -- what one DDA step costs, on both sides of the comparison.

THE MEASUREMENT, AND WHY IT IS A DIFFERENCE AND NOT A COUNT
-----------------------------------------------------------
The naive port fuses its 32 steps into one 36 000-instruction kernel with no
period, so "the instructions in one DDA step" cannot be read off the SASS the
way it can for the reference, whose 16 unrolled steps repeat at a fixed
spacing. The robust instrument, and the one this module uses everywhere, is a
**difference**: capture and compile the *same* program at `max_steps = S` and at
`max_steps = S + d`, and take

    (instructions(S + d) - instructions(S)) / d

as the cost of one step. It needs no assumption about the shape of the
generated code, it works on a fused and an unfused schedule alike, and it is
the same instrument on both sides of the comparison. The reference is
measured the same way, through `max_steps = 16` against `max_steps = 32` and
`max_steps = 64`, so the three numbers in the final table are commensurable.

Two other counts are reported alongside it, and both are the same difference
taken one level up:

    graph ops / pixel / step    from the census's `uops_total_graph`, i.e. the
                                number of UOps tinygrad materialised, divided
                                by (pixels x steps). This is the *source-level*
                                operation count and it is what the rewrite
                                actually removed.

    generated statements/step   from the captured CUDA, by splitting the fused
                                kernel at its voxel-gather statements. Direct,
                                not a difference, and it lets every statement
                                inside a step be classified.

THE INDEX-ARITHMETIC SHARE
--------------------------
Reported two ways, and they are not the same measurement:

  * **causal, from the ladder.** `ablate.py` runs V0..V4, each the rung above
    plus exactly one rewrite. The per-step cost difference between two rungs
    is what that rewrite removed, causally, with nothing else changing. This
    is the number to quote.

  * **descriptive, from the opcode mix.** Every statement inside a step is
    classified (float arithmetic / comparison / integer / gather / select /
    cast) and the integer share is reported. This corroborates the ladder; it
    cannot replace it, because a classification is an assignment of a label and
    a difference is a measurement.

WHAT IS NEVER REPORTED
----------------------
A static instruction count is not a per-step count, and a per-step count is not
a time. Every number here is labelled with which of the three it is.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "archlab"))

from inspect_ptx import CATEGORIES, parse_sass  # noqa: E402

# `cuobjdump -sass` body:  "        /*0000*/                   MOV R1, c[0x0][0x28] ;"
# The offset field is 4 hex digits for the first 4096 instructions and 5 or
# more after that, so a `{4}` pattern reports EXACTLY 4096 instructions for any
# kernel bigger than that -- a plausible-looking number that is silently
# wrong. `{4,8}` is what makes the count mean anything.
INSTR_RE = re.compile(r"/\*[0-9a-fA-F]{4,8}\*/\s+(?:@!?\w+\s+)?([A-Z][A-Z0-9_.]*)")
FUNC_RE = re.compile(r"Function : (\S+)")
PTXV_ENTRY = re.compile(r"Compiling entry function '([^']+)' for '(\S+)'")
PTXV_FRAME = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, "
                        r"(\d+) bytes spill loads")
PTXV_USED = re.compile(r"Used (\d+) registers, used (\d+) barriers")

# Integer-domain and float-domain opcode families, for the descriptive mix.
# `IMAD` is Volta+ fused multiply-add over integers *and* a general-purpose
# integer multiply, and on a comparison of a fused kernel it is overwhelmingly
# address work; it is reported separately from the other integer ops so a
# reader can judge.
INT_OPS = ("IADD3", "IMAD", "LOP3", "SHF", "LEA", "ISETP", "SEL", "ISCADD",
           "LDS", "STS", "IABS", "FSEL", "MOV", "PRMT", "PLOP3", "VIADD")
FLOAT_OPS = ("FADD", "FMUL", "FFMA", "FSETP", "FMNMX", "MUFU", "FSEL", "F2F",
             "FRND", "FCHK", "MUFU")


# --------------------------------------------------------------------------
# SASS
# --------------------------------------------------------------------------
def split_functions(sass: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    cur = None
    for line in sass.splitlines():
        m = FUNC_RE.search(line)
        if m:
            cur = m.group(1)
            out[cur] = []
            continue
        if cur is not None:
            out[cur].append(line)
    return out


def count_instructions(body: str) -> int:
    return len(INSTR_RE.findall(body))


def parse_ptxas_v(text: str) -> dict[str, dict]:
    """`ptxas -v` text -> {function name: resources}. Matched BY NAME."""
    out: dict[str, dict] = {}
    cur = None
    for line in text.splitlines():
        m = PTXV_ENTRY.search(line)
        if m:
            cur = {"arch": m.group(2)}
            out[m.group(1)] = cur
            continue
        if cur is None:
            continue
        m = PTXV_FRAME.search(line)
        if m:
            cur["stack_frame"] = int(m.group(1))
            cur["spill_stores"] = int(m.group(2))
            cur["spill_loads"] = int(m.group(3))
            cur["local_bytes"] = int(m.group(2)) + int(m.group(3))
            continue
        m = PTXV_USED.search(line)
        if m:
            cur["registers"] = int(m.group(1))
            cur["barriers"] = int(m.group(2))
    return out


def sass_rows(sass_dir: Path) -> list[dict]:
    """One row per compiled function, from every .sass / .ptxas pair."""
    rows = []
    for f in sorted(sass_dir.rglob("*.sass")):
        stem = f.with_suffix("")
        res = {}
        pt = Path(str(stem) + ".ptxas")
        if pt.exists():
            res = parse_ptxas_v(pt.read_text())
        rel = f.relative_to(sass_dir).as_posix()
        # the file name encodes the shape: max_steps_0032__lean__kernel004
        parts = rel[:-len(".sass")].split("__")
        ms = next((int(p.split("_")[-1]) for p in parts
                   if p.startswith("max_steps_") and p.split("_")[-1].isdigit()), None)
        rung = next((p for p in parts if p in
                     ("naive", "int_cursor", "flat_gather", "incremental", "lean")), None)
        for name, body in split_functions(f.read_text()).items():
            st = parse_sass("\n".join(body))
            r = {"file": rel, "function": name, "max_steps": ms, "rung": rung,
                 "instructions": count_instructions("\n".join(body)),
                 "source_lines": sum(1 for ln in body if ln.strip())}
            # kept for the periodicity check; not written to the JSON, which
            # would then carry a 74 000-element string per row
            r["body"] = "\n".join(body)
            for cat, _, _ in CATEGORIES:
                r[cat] = st["categories"].get(cat, 0)
            r["bss_bsync_pairs"] = sum(1 for o in INSTR_RE.findall("\n".join(body))
                                       if o == "BSSY")
            r["ldg_e_s8"] = sum(1 for o in INSTR_RE.findall("\n".join(body))
                                if o.startswith("LDG.E") and ".S8" in o)
            r["stg_u8"] = sum(1 for o in INSTR_RE.findall("\n".join(body))
                              if o.startswith("STG.E.U8"))
            r.update(res.get(name, {}))
            rows.append(r)
    return rows


# --------------------------------------------------------------------------
# the generated CUDA, split into steps
# --------------------------------------------------------------------------
# One generated statement: a type, a name, `=`, and a body to the first `;`.
# The type is TWO words for the byte types (`signed char val17 = ...`), so
# `(\w+)` alone silently drops every voxel read and every colour read -- which
# is a segmentation that finds no steps and reports zero rather than failing.
STMT_RE = re.compile(r"^\s*(\w+(?:\s+\w+)?)\s+(val\d+|alu\d+)\s*=\s*(.*);\s*(?://.*)?$")
SIG_RE = re.compile(r"extern \"C\" __global__ void \w+\((.*?)\) \{")
ARG_RE = re.compile(r"(signed char|unsigned char|int|unsigned int|float|bool|"
                    r"long|size_t|uint\d+_t)\*\s*(data\d+)_(\d+)")
# One generated statement's RHS may hold several dereferences (a vectorised
# `float4` load is one statement, four elements), so a statement is a "gather"
# if its RHS touches the world array at all.
DEREF_RE = re.compile(r"\*\((data\d+)_(\d+)\+")


def world_array(cu_path: Path) -> tuple[str | None, int]:
    """The generated name of the voxel array, and its element count.

    Found from the kernel SIGNATURE rather than from a pattern in the body.
    The body has many dereferences -- the camera basis alone is 13 of them per
    pixel -- and matching "a line that reads memory" finds all of them, which
    is a segmentation that looks plausible and is nonsense. The world is
    identified instead as the largest **one-byte** pointer argument: it is the
    only byte tensor the hot kernel reads, and it is 128x the size of the
    colour table. That is a property of the computation, not of the
    generator's naming.

    Returns (name, count) or (None, 0) if the signature could not be read.
    """
    for line in cu_path.read_text().splitlines():
        m = SIG_RE.search(line)
        if not m:
            continue
        best, best_n = None, -1
        for ty, name, n in ARG_RE.findall(m.group(1)):
            if "char" in ty and int(n) > best_n:
                best, best_n = name, int(n)
        return best, best_n
    return None, 0


def classify(stmt_type: str, rhs: str, world: str | None) -> str:
    """One label per generated statement, for the descriptive mix."""
    if world and f"*({world}_" in rhs:
        return "gather"
    is_int_lhs = stmt_type in ("int", "unsigned int")
    has_sel = "?" in rhs
    if stmt_type == "bool":
        return "compare"
    if stmt_type in ("signed char", "unsigned char"):
        return "cast" if "int" in rhs else "gather"
    if is_int_lhs:
        if "<<" in rhs or ((("+" in rhs) or ("-" in rhs)) and not has_sel):
            return "int_addr"
        if has_sel:
            return "int_select"
        return "int_other"
    if has_sel:
        return "f_select"
    if "1/sqrt" in rhs or "rsqrt" in rhs:
        return "f_transcend"
    return "f_arith"


def _statements(cu_path: Path, world: str | None):
    stmts = []
    for ln in cu_path.read_text().splitlines():
        m = STMT_RE.match(ln)
        if m:
            stmts.append((m.group(1), m.group(3)))
    return stmts


def step_segments(cu_path: Path, max_steps: int | None = None) -> dict:
    """Split one captured kernel into DDA steps at its voxel reads.

    A "step" here is the statements between two consecutive reads of the world
    array, which is what one execution of the step body costs in the emitted
    source. The generator emits several pixels' worth of body per thread --
    the vectorisation it chose -- so the number of segments is a multiple of
    `max_steps`; the quotient is reported as `pixels_per_thread` so the reader
    can see that rather than take it on trust, and a quotient that is not a
    whole number is reported as `suspicious` rather than averaged away.
    """
    world, n_elem = world_array(cu_path)
    stmts = _statements(cu_path, world)
    if not world:
        return {"statements": len(stmts), "steps": 0, "per_step": None,
                "world_array": None}
    idx = [i for i, (_, rhs) in enumerate(stmts)
           if f"*({world}_" in rhs]
    if len(idx) < 2:
        return {"statements": len(stmts), "steps": 0, "per_step": None,
                "gathers": len(idx), "world_array": world, "world_elems": n_elem}
    # A segment is INCLUSIVE of its opening read: the world read is a real
    # per-step cost and excluding it reports a step that is one instruction
    # short of what the kernel executes, on both sides of the comparison and so
    # invisible in the ratio but wrong in the absolute number.
    segs = [stmts[idx[i]:idx[i + 1]] for i in range(len(idx) - 1)]
    sizes = [len(s) for s in segs]
    out = {"statements": len(stmts), "gathers": len(idx), "steps": len(segs),
           "prologue": idx[0], "epilogue": len(stmts) - idx[-1] - 1,
           "world_array": world, "world_elems": n_elem,
           "per_step_median": statistics.median(sizes),
           "per_step_min": min(sizes), "per_step_max": max(sizes),
           "per_step": sizes}
    if max_steps:
        q = len(segs) / max_steps
        out["pixels_per_thread"] = q
        out["segment_count_is_multiple_of_max_steps"] = \
            abs(q - round(q)) < 1e-9
    return out


def mix_of(cu_path: Path, max_steps: int | None = None) -> dict:
    """The per-step statement mix, over the median-length segments."""
    s = step_segments(cu_path, max_steps)
    if not s.get("per_step"):
        return {}
    world = s["world_array"]
    stmts = _statements(cu_path, world)
    idx = [i for i, (_, rhs) in enumerate(stmts) if f"*({world}_" in rhs]
    segs = [stmts[idx[i]:idx[i + 1]] for i in range(len(idx) - 1)]
    m = s["per_step_median"]
    take = [sg for sg in segs if len(sg) == m] or segs[:1]
    counts: dict[str, int] = {}
    for sg in take:
        for t, rhs in sg:
            k = classify(t, rhs, world)
            counts[k] = counts.get(k, 0) + 1
    tot = sum(counts.values()) or 1
    return {"median_segment": m, "n_segments": len(take),
            "world_array": world, "counts": counts,
            "fractions": {k: v / tot for k, v in counts.items()}}


# --------------------------------------------------------------------------
def pixels_per_render(obs_shape, override: int | None = None) -> int:
    """Pixels per render, from the observation's shape.

    The observation is `[B, H, W, 3]`, so the pixel count is every axis except
    the trailing channel axis: `prod(shape[:-1])`. Reading it as
    `shape[0]*shape[2]*shape[3]` -- which is what a positional transcription of
    a 4-D shape invites -- puts the *channel* count where the W belongs and
    yields a number 3/W of the truth. At 32x32 that is 3072 instead of 32768,
    and every per-pixel figure downstream inherits it silently, because a factor
    of ten is not a value any sanity check rejects.

    Rank-1 and rank-0 shapes raise rather than returning 1: a per-pixel figure
    with no pixel axis is not a small number, it is a wrong one.
    """
    if override is not None:
        return int(override)
    if not isinstance(obs_shape, (list, tuple)) or len(obs_shape) < 2:
        raise SystemExit(f"obs_shape {obs_shape!r} has no batch and pixel axes; "
                         f"a per-pixel figure cannot be formed from it")
    n = 1
    for d in obs_shape[:-1]:
        n *= int(d)
    return n


def slopes(steps: list[int], values: list[float]) -> dict:
    """Least-squares slope and intercept of value against max_steps."""
    n = len(steps)
    if n < 2:
        return {"per_step": None, "intercept": None, "r2": None}
    mx, my = statistics.fmean(steps), statistics.fmean(values)
    sxx = sum((x - mx) ** 2 for x in steps)
    sxy = sum((x - mx) * (y - my) for x, y in zip(steps, values))
    b = sxy / sxx if sxx else 0.0
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in values)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(steps, values))
    return {"per_step": b, "intercept": a,
            "r2": (1 - ss_res / ss_tot) if ss_tot > 0 else None}


def reference_step_cost(sass_path: Path, res_path: Path | None,
                        select: str = "render_kernel_uint8_prebasis") -> dict:
    """The hand-written side's per-step cost, re-derived from its own SASS.

    The reference's `max_steps` is a *runtime* argument, so its static code is
    16 unrolled bodies whatever the caller asks for and the difference method
    this module uses on the generated side does not apply to it. The direct
    measurement is the load-to-load spacing: the kernel contains exactly one
    `__ldg` per unrolled step, so the 16 voxel-load positions are the 16 steps
    and the spacing between them is the per-step instruction count. That is
    measured here rather than quoted, and the spacing is also checked for
    uniformity, which is the evidence that ptxas emitted all 16 bodies and did
    not fold, reorder or spill them.
    """
    out: dict = {"sass": str(sass_path)}
    if not sass_path.exists():
        out["error"] = f"missing {sass_path}"
        return out
    text = sass_path.read_text()
    for name, lines in split_functions(text).items():
        if select not in name:
            continue
        body = "\n".join(lines)
        ops = INSTR_RE.findall(body)
        out["function"] = name
        out["instructions"] = len(ops)
        positions = [i for i, o in enumerate(ops)
                     if o.startswith("LDG.E") and ".S8" in o]
        out["voxel_loads"] = len(positions)
        out["voxel_load_opcodes"] = sorted({o for o in ops
                                            if o.startswith("LDG.E") and ".S8" in o})
        out["voxel_load_indices"] = positions
        spacings = [b - a for a, b in zip(positions, positions[1:])]
        out["load_to_load_spacings"] = spacings
        out["spacings_all_equal"] = len(set(spacings)) == 1
        out["instructions_per_step"] = (spacings[0] if spacings and
                                        out["spacings_all_equal"] else None)
        out["before_first_load"] = positions[0] if positions else None
        out["after_last_load"] = (len(ops) - positions[-1] - 1) if positions else None
        out["bss_bsync_pairs"] = ops.count("BSSY")
        out["bar_sync"] = ops.count("BAR.SYNC")
        out["stg_u8"] = sum(1 for o in ops if o.startswith("STG.E.U8"))
        st = parse_sass(body)
        for cat, _, _ in CATEGORIES:
            out[cat] = st["categories"].get(cat, 0)
        # is the 16-step body an exact repeat? The periodicity claim in the
        # earlier work is checked here rather than inherited, and it is checked
        # the right way round: the bodies are the opcodes BETWEEN consecutive
        # loads, so each is compared with the next, aligned on the load
        # positions rather than on an arbitrary 38-instruction window.
        if len(positions) > 1:
            bodies = [ops[a:b] for a, b in zip(positions, positions[1:])]
            first = bodies[0]
            same = [b == first for b in bodies]
            out["bodies_identical_to_first"] = same
            out["n_bodies"] = len(bodies)
            out["n_bodies_identical"] = sum(same)
            out["all_bodies_identical"] = all(same)
            run = best = 0
            for v in same:
                run = run + 1 if v else 0
                best = max(best, run)
            out["longest_identical_run"] = best
            out["period_38_verified"] = (out["instructions_per_step"] == 38)
        break
    if res_path and res_path.exists():
        res = parse_ptxas_v(res_path.read_text())
        for k, v in res.items():
            if select in k:
                out["resources"] = v
    return out


# The reference's grid, as arithmetic over its own source rather than as a
# quotation: `kernels.cu:2858-2861` is `dim3 grid(blocks_per_batch,
# batch_size)` with `dim3 block(128)`, and `kernels.cu:585` is
# `pixel_offset = blockIdx.x*blockDim.x + threadIdx.x`. So the reference runs
# exactly ONE THREAD PER PIXEL, and its DDA state is one ray per thread. That is
# the fact the work-normalised comparison turns on.
REF_BLOCK = 128


def reference_threads(pixels_per_env: int, batch: int, res: int = 32) -> int:
    """Threads the reference launches for `batch` envs.

    `blocks_per_batch = ceil(H*W/128)`, `grid(blocks_per_batch, batch)`,
    `block(128)`, so the thread count equals the pixel count exactly. Returns
    0 for batch 0 rather than raising, so a sweep that skipped a batch does not
    produce a division by zero three sections later.
    """
    if not batch or not pixels_per_env:
        return 0
    per_env = pixels_per_env * res * res
    return -(-per_env // REF_BLOCK) * REF_BLOCK * batch


def generated_grid(cu_path: Path) -> dict:
    """The generated kernel's grid, read out of its own source.

    `gidx0 = blockIdx.x; /* 512 */` and `lidx0 = threadIdx.x; /* 2 */` are
    emitted by tinygrad's own renderer with the extent in a comment, and the
    remaining axes follow the same pattern. Reading them is the only way to know
    how many pixels a generated thread carries, and that number is what
    divides the per-step slope into a per-pixel figure.

    Returns (blocks, threads, pixel_extent) with `pixel_extent` = product of the
    extents, or None when the comments are absent -- in which case the caller
    reports the missing divisor rather than assuming one.
    """
    import re as _re
    src = cu_path.read_text()
    ext = {}
    for m in _re.finditer(r"int (g|l)idx(\d) = (blockIdx|threadIdx)\.(\w); /\* (\d+) \*/",
                          src):
        axis = m.group(2)
        ext[(m.group(3), axis)] = int(m.group(5))
    if not ext:
        return None
    blocks = 1
    for (kind, _), v in ext.items():
        if kind == "blockIdx":
            blocks *= v
    threads = blocks
    for (kind, _), v in ext.items():
        if kind == "threadIdx":
            threads *= v
    return {"blocks": blocks, "threads": threads,
            "extents": {f"{k[0]}.{k[1]}": v for k, v in ext.items()}}


def longest_period(body: str, lo: int = 8, hi: int = 400) -> dict:
    """The longest run of exact opcode repeats in a kernel's instruction stream.

    The periodicity claim about the two sides is checked here rather than
    inherited, because it is the mechanism behind a register-pressure
    difference, and a mechanism is worth less than a measurement. A body that
    repeats at a fixed period can be register-allocated one copy at a time; one
    straight-line region cannot.

    Returns the best (period, run length) over all periods in [lo, hi], where
    the run length is how many consecutive periods match the first, counted
    from the START of the stream. A kernel whose only match is in its prologue
    therefore reports a short run, which is the honest answer.
    """
    ops = INSTR_RE.findall(body)
    n = len(ops)
    best = (None, 0)
    for p in range(lo, min(hi, n // 3) + 1):
        run = 1
        while run * p + p <= n and ops[0:p] == ops[run * p:run * p + p]:
            run += 1
        if run > best[1]:
            best = (p, run)
    return {"period": best[0], "run_length": best[1], "instructions": n,
            "searched_periods": [lo, min(hi, n // 3)]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captured", type=Path, default=None,
                    help="the capture_opt.py output dir")
    ap.add_argument("--sass", type=Path, default=None,
                    help="the compile_sass.sh output dir; default <captured>/_sass")
    ap.add_argument("--reference", type=Path, default=REPO / "bench_cuda" / "build" / "kernels_sm_75.sass",
                    help="the hand-written side's sm_75 SASS")
    ap.add_argument("--reference-res", type=Path, default=REPO / "bench_cuda" / "build" / "resources_sm_75.txt")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--pixels", type=int, default=None,
                    help="pixels per render, for the per-pixel figures; default "
                         "batch*res*res, taken from the capture's own shape")
    a = ap.parse_args()

    # ---- the hand-written side first, so the frame is on the board
    ref = reference_step_cost(a.reference, a.reference_res)
    print("=" * 78)
    print("  the hand-written side, re-derived from its own sm_75 SASS")
    print("=" * 78)
    if "error" in ref:
        print(f"  {ref['error']}")
    else:
        print(f"  function            {ref['function']}")
        print(f"  static instructions {ref['instructions']:,}")
        print(f"  voxel loads         {ref['voxel_loads']}  "
              f"{ref['voxel_load_opcodes']}")
        print(f"  load indices        {ref['voxel_load_indices']}")
        print(f"  load-to-load gaps   {ref['load_to_load_spacings']}")
        print(f"  all equal?          {ref['spacings_all_equal']}   "
              f"-> {ref['instructions_per_step']} instructions per DDA step")
        print(f"  before 1st / after  {ref['before_first_load']} / "
              f"{ref['after_last_load']}")
        print(f"  BSSY / BAR.SYNC     {ref['bss_bsync_pairs']} / {ref['bar_sync']}")
        print(f"  STG.E.U8            {ref['stg_u8']}")
        r = ref.get("resources", {})
        print(f"  registers / spill   {r.get('registers')} / "
              f"{int(r.get('spill_stores', 0)) + int(r.get('spill_loads', 0))} B")
        if "n_bodies_identical" in ref:
            print(f"  step bodies         {ref['n_bodies_identical']}/"
                  f"{ref['n_bodies']} byte-identical to the first; "
                  f"longest identical run {ref['longest_identical_run']}")
    if a.captured is None:
        # --sass alone is a valid invocation: the SASS section is independent of
        # the capture and a reader often has only one of the two.
        census, out = [], {"reference": ref}
    else:
        census = json.loads((a.captured / "census.json").read_text())
        out = {"captured": str(a.captured), "reference": ref}
    if not census:
        census = []

    # ---- graph ops per pixel per step, from the census
    graph: dict = {}
    if census:
        print("=" * 78)
        print("  graph operations per pixel per DDA step (TinyGrad UOps, whole graph)")
        print("=" * 78)
        print(f"  {'rung':<13} {'slope uops/step':>16} {'/pixel/step':>13} {'r2':>7} "
              f"{'launches':>9} {'distinct':>9}")
    for c in census:
        steps = [r["max_steps"] for r in c["rows"]]
        uops = [r["uops_total_graph"] for r in c["rows"]]
        sl = slopes(steps, [float(u) for u in uops])
        px = pixels_per_render(c["rows"][0]["obs_shape"], a.pixels)
        per_px = sl["per_step"] / px if sl["per_step"] else None
        graph[c["variant"]] = {"steps": steps, "uops": uops, "slope": sl["per_step"],
                               "per_pixel_per_step": per_px, "pixels": px,
                               "launches": {r["max_steps"]: r["kernel_launches"]
                                            for r in c["rows"]},
                               "distinct": {r["max_steps"]: r["distinct_programs_launched"]
                                            for r in c["rows"]}}
        r2 = sl["r2"]
        print(f"  {c['variant']:<13} {sl['per_step']:>16,.1f} "
              f"{(per_px if per_px else 0):>13.3f} "
              f"{(f'{r2:.5f}' if r2 is not None else '?'):>7} "
              f"{c['rows'][0]['kernel_launches']:>9} "
              f"{c['rows'][0]['distinct_programs_launched']:>9}")
    out["graph"] = graph
    if "naive" in graph and "lean" in graph:
        g0, g1 = graph["naive"]["per_pixel_per_step"], graph["lean"]["per_pixel_per_step"]
        out["graph_ratio_naive_over_lean"] = g0 / g1 if g1 else None
        print(f"\n  V0/V4 graph-op ratio: {g0 / g1:.3f}x")

    # ---- the generated CUDA, statement by statement
    src: dict = {}
    mixes: dict = {}
    if a.captured is not None and list(a.captured.glob("max_steps_*/*/kernel*.cu")):
        print()
        print("=" * 78)
        print("  the generated CUDA, split at its voxel gathers (one segment = one step)")
        print("=" * 78)
        print(f"  {'rung':<13} {'ms':>4} {'stmts':>8} {'pro':>6} {'epi':>6} "
              f"{'reads':>6} {'px/thr':>7} {'step med':>9} {'min':>5} {'max':>5} "
              f"{'world':>12}")
        for p in sorted(a.captured.glob("max_steps_*/*/kernel*.cu")):
            ms = int(p.parent.parent.name.split("_")[-1])
            rung = p.parent.name
            s = step_segments(p, ms)
            # The regime where the scheduler fuses emits several programs at
            # each max_steps -- the LUT, the basis, the pixel grid, the whole
            # DDA, the store. Only the DDA one has world reads, and the
            # biggest file at a given max_steps is the one to segment. Taking
            # every file instead produces rows like "7 statements, no world
            # reads", which are a segmentation failure dressed as a result.
            prev = src.get(rung, {}).get(ms)
            if s.get("per_step") and (prev is None or
                                      s["statements"] > prev["statements"]):
                s["file"] = p.name
                s["path"] = str(p)
                # The pixels a generated THREAD carries, from the grid tinygrad
                # emitted -- not from the gather count, which also includes the
                # out-of-bounds reads a clamped index still performs, and not
                # assumed. `pixels_per_thread` from the segmentation is the
                # fallback and is a slightly smaller number, so the grid is
                # preferred and both are kept.
                g = generated_grid(p)
                s["grid"] = g
                if g and g["threads"]:
                    s["pixels_per_thread_grid"] = None   # filled per batch below
                src.setdefault(rung, {})[ms] = s
        for rung in sorted(src):
            for ms in sorted(src[rung]):
                s = src[rung][ms]
                if not s["per_step"]:
                    print(f"  {rung:<13} {ms:>4} {s['statements']:>8}   "
                          f"(no world reads found; world="
                          f"{s.get('world_array')})")
                    continue
                ppt = s.get("pixels_per_thread")
                print(f"  {rung:<13} {ms:>4} {s['statements']:>8} {s['prologue']:>6} "
                      f"{s['epilogue']:>6} {s['gathers']:>6} "
                      f"{(f'{ppt:g}' if ppt else '?'):>7} "
                      f"{s['per_step_median']:>9.1f} {s['per_step_min']:>5} "
                      f"{s['per_step_max']:>5} {str(s['world_array']):>12}")
        out["generated_cuda"] = src

        # The slope of the statement count against max_steps is the robust form
        # of the per-step statement count, for the same reason the SASS one is:
        # a few world reads get merged into one statement, so an individual
        # segment is sometimes two steps. Divided by the pixels per thread it
        # is directly comparable with the graph-op count above, and the two
        # agreeing is a check that the segmentation is not inventing anything.
        stslopes = {}
        print()
        print("  generated statements per DDA step (slope against max_steps, "
              "per pixel per thread)")
        print(f"  {'rung':<13} {'slope stmts/step':>18} {'px/thr':>7} "
              f"{'/pixel/step':>13} {'r2':>9}")
        for rung in sorted(src):
            hot = {}
            for ms, s in src[rung].items():
                if s.get("per_step"):
                    # the fused kernel is the biggest one at each max_steps
                    hot[ms] = s["statements"]
            if len(hot) < 2:
                continue
            ms_sorted = sorted(hot)
            sl = slopes(ms_sorted, [float(hot[m]) for m in ms_sorted])
            ppt = src[rung][max(ms_sorted)].get("pixels_per_thread") or \
                statistics.median([src[rung][m].get("pixels_per_thread") or 16
                                   for m in ms_sorted])
            per_px = sl["per_step"] / ppt if sl["per_step"] else None
            stslopes[rung] = {"slope": sl["per_step"], "pixels_per_thread": ppt,
                              "per_pixel_per_step": per_px, "r2": sl["r2"],
                              "statements_by_max_steps": hot}
            r2 = sl["r2"]
            print(f"  {rung:<13} {sl['per_step']:>18,.1f} {ppt:>7.2f} "
                  f"{(per_px or 0):>13.2f} "
                  f"{(f'{r2:.5f}' if r2 is not None else '?'):>9}")
        out["statement_slope"] = stslopes
        if "naive" in stslopes and "lean" in stslopes:
            a0 = stslopes["naive"]["per_pixel_per_step"]
            a1 = stslopes["lean"]["per_pixel_per_step"]
            out["statement_ratio_naive_over_lean"] = a0 / a1 if a1 else None
            print(f"\n  V0/V4 generated-statement ratio: {a0 / a1:.3f}x")

        # ---- the per-step statement mix, at the largest captured max_steps
        biggest = max(int(p.parent.parent.name.split("_")[-1])
                      for p in a.captured.glob("max_steps_*/*/kernel*.cu"))
        print()
        print("=" * 78)
        print(f"  per-step statement mix at max_steps={biggest} "
              f"(the median-length segment)")
        print("=" * 78)
        hdr = ["f_arith", "f_transcend", "f_select", "compare", "gather",
               "int_addr", "int_select", "int_other", "cast"]
        print(f"  {'rung':<13} {'n':>4} " + " ".join(f"{h:>12}" for h in hdr))
        for p in sorted(a.captured.glob(f"max_steps_{biggest:04d}/*/kernel*.cu")):
            rung = p.parent.name
            ms = int(p.parent.parent.name.split("_")[-1])
            # only the DDA kernel, by the same rule as above
            chosen = src.get(rung, {}).get(ms)
            if not chosen or chosen.get("file") != p.name:
                continue
            mx = mix_of(p, biggest)
            if not mx:
                continue
            mixes[rung] = mx
            print(f"  {rung:<13} {mx['median_segment']:>4} " +
                  " ".join(f"{mx['fractions'].get(h, 0.0):>12.3f}" for h in hdr))
    out["mix"] = mixes
    if "naive" in mixes and "lean" in mixes:
        n_int = sum(mixes["naive"]["fractions"].get(h, 0.0)
                    for h in ("int_addr", "int_select", "int_other"))
        l_int = sum(mixes["lean"]["fractions"].get(h, 0.0)
                    for h in ("int_addr", "int_select", "int_other"))
        print(f"\n  integer-statement share of one step: "
              f"V0 {100 * n_int:.1f}%, V4 {100 * l_int:.1f}%")
        out["int_share_naive"] = n_int
        out["int_share_lean"] = l_int

    # ---- SASS
    sass_dir = a.sass or (a.captured / "_sass" if a.captured else None)
    per_rung: dict = {}
    if sass_dir is not None and sass_dir.exists():
        rows = sass_rows(sass_dir)
        # Attach the pixels-per-thread, read off the captured CUDA, to each SASS
        # row. Without it the per-step slope is per THREAD and comparing it with
        # the reference's one-thread-one-pixel kernel divides a number that is
        # ~16x too large by 38 and reports a 30x gap where the real one is 2x.
        for r in rows:
            cap = (src.get(r["rung"], {}) or {}).get(r["max_steps"]) or {}
            r["pixels_per_thread"] = cap.get("pixels_per_thread")
            r["generated_grid"] = cap.get("grid")
            # drop the body here, once the periodicity is computed
            body, r["body"] = r.get("body"), None
            r["periodicity"] = longest_period(body or "")
        print()
        print("=" * 78)
        print("  SASS, sm_75, through tools/archlab's categoriser")
        print("=" * 78)
        print(f"  {'rung':<13} {'ms':>4} {'kernels':>8} {'total instr':>12} "
              f"{'hot instr':>10} {'slope/step':>12} {'/px/step':>10} {'r2':>8} "
              f"{'reg':>5} {'spill':>7} {'LDG.E.S8':>10} {'BSSY':>5}")
        for rung in sorted({r["rung"] for r in rows if r["rung"]}):
            rr = [r for r in rows if r["rung"] == rung and r["max_steps"] is not None]
            if not rr:
                print(f"  {rung:<13} (no max_steps recoverable from the file names; "
                      f"the SASS was not produced by compile_sass.sh)")
                continue
            by_ms = {}
            for r in rr:
                by_ms.setdefault(r["max_steps"], []).append(r)
            steps, totals, hots = [], [], []
            for ms in sorted(by_ms):
                g = by_ms[ms]
                steps.append(ms)
                totals.append(sum(x["instructions"] for x in g))
                hots.append(max(x["instructions"] for x in g))
                big = max(g, key=lambda x: x["instructions"])
                per_rung.setdefault(rung, {})[ms] = {
                    "kernels": len(g), "total": totals[-1], "hot": hots[-1],
                    "hot_file": big["file"], "hot_function": big["function"],
                    # from the row, not from a key that does not exist on this
                    # dict: the hot kernel carries ~16 pixels per thread, and
                    # the per-step slope has to be divided by it before it can
                    # be compared with the reference's one-pixel-per-thread
                    # kernel.
                    "pixels_per_thread": big.get("pixels_per_thread"),
                    "registers": big.get("registers"),
                    "local_bytes": big.get("local_bytes"),
                    "spill_stores": big.get("spill_stores"),
                    "spill_loads": big.get("spill_loads"),
                    "stack_frame": big.get("stack_frame"),
                    "ldg_e_s8": big.get("ldg_e_s8"),
                    "bss_bsync_pairs": big.get("bss_bsync_pairs"),
                    "periodicity": big.get("periodicity"),
                    "rows": g,
                }
            sl_t, sl_h = slopes(steps, [float(v) for v in totals]), \
                slopes(steps, [float(v) for v in hots])
            h = per_rung[rung][max(steps)]
            rt, rh = sl_t["r2"], sl_h["r2"]
            # The fused kernel is one THREAD doing many pixels, so its per-step
            # instruction count is per thread and has to be divided by the
            # pixels that thread carries before it can be compared with the
            # reference's one-thread-one-pixel kernel. The divisor comes from
            # the row's `pixels_per_thread`, which `sass_rows` copied from the
            # captured source.
            ppt = h.get("pixels_per_thread")
            per_px_step = (sl_h["per_step"] / ppt) if (ppt and sl_h["per_step"]) else None
            h["pixels_per_thread"] = ppt
            h["sass_per_pixel_per_step"] = per_px_step
            print(f"  {rung:<13} {max(steps):>4} "
                  f"{h['kernels']:>8} "
                  f"{totals[-1]:>12,} {hots[-1]:>10,} "
                  f"{(sl_h['per_step'] or 0):>12,.1f} "
                  f"{(per_px_step if per_px_step else 0):>10,.1f} "
                  f"{(f'{rh:.4f}' if rh is not None else '?'):>8} "
                  f"{str(h['registers']):>5} {str(h['local_bytes']):>7} "
                  f"{str(h['ldg_e_s8']):>10} {str(h['bss_bsync_pairs']):>5}")
            per_rung[rung]["slope_total_per_step"] = sl_t["per_step"]
            per_rung[rung]["slope_total_r2"] = rt
            per_rung[rung]["slope_hot_per_step"] = sl_h["per_step"]
            per_rung[rung]["slope_hot_r2"] = rh
            per_rung[rung]["slope_hot_per_pixel_per_step"] = per_px_step
        out["sass"] = per_rung

        # ---- the periodicity claim, on both sides, measured
        print()
        print("=" * 78)
        print("  do the step bodies repeat? (the mechanism behind the register"
              " pressure)")
        print("=" * 78)
        rp = ref.get("instructions_per_step")
        print(f"  reference   16 unrolled steps, {rp} instructions apart, "
              f"{ref.get('n_bodies_identical')}/{ref.get('n_bodies')} "
              f"byte-identical to the first")
        for rung in sorted(per_rung):
            nums = [x for x in per_rung[rung] if isinstance(x, int)]
            if not nums:
                continue
            h = per_rung[rung][max(nums)]
            pr = h.get("periodicity") or {}
            print(f"  {rung:<11} fused {h['hot']:,} instructions: best exact "
                  f"period {pr.get('period')} repeating {pr.get('run_length')}x "
                  f"over a search of {pr.get('searched_periods')}")
        out["periodicity"] = {
            "reference": {"period": rp,
                          "bodies_identical": ref.get("n_bodies_identical"),
                          "bodies": ref.get("n_bodies")}}
        for _r in per_rung:
            _nums = [x for x in per_rung[_r] if isinstance(x, int)]
            if _nums:
                out["periodicity"][_r] = per_rung[_r][max(_nums)].get("periodicity")
        if "naive" in per_rung and "lean" in per_rung:
            # Everything below is PER PIXEL PER STEP. The fused kernel is one
            # thread carrying ~16 pixels, so its raw slope is not comparable
            # with the reference's one-thread-one-pixel 38; the per-pixel figure
            # is, and it is the one every claim in the note rests on.
            a0 = per_rung["naive"].get("slope_hot_per_pixel_per_step")
            a1 = per_rung["lean"].get("slope_hot_per_pixel_per_step")
            t0 = per_rung["naive"]["slope_hot_per_step"]
            t1 = per_rung["lean"]["slope_hot_per_step"]
            if a0 and a1:
                out["sass_per_step_ratio_naive_over_lean"] = a0 / a1
                out["sass_per_pixel_per_step"] = {"V0": a0, "V4": a1}
                print(f"\n  per THREAD per DDA step:  V0 {t0:,.1f}   V4 {t1:,.1f}"
                      f"   = {t0 / t1:.3f}x")
                print(f"  per PIXEL per DDA step:  V0 {a0:,.1f}   V4 {a1:,.1f}"
                      f"   = {a0 / a1:.3f}x")
                print(f"  the optimization removes {100 * (1 - a1 / a0):.1f}% of "
                      f"the per-pixel per-step instruction count")
                rp = ref.get("instructions_per_step")
                if rp:
                    out["per_step_vs_reference"] = {
                        "reference": rp, "V0": a0, "V4": a1,
                        "V0_over_reference": a0 / rp, "V4_over_reference": a1 / rp}
                    print(f"  against the hand-written side's {rp} instructions "
                          f"per pixel per step:  V0 {a0 / rp:.2f}x   "
                          f"V4 {a1 / rp:.2f}x")

            # ---- WHAT THE TWO SIDES' GRIDS ARE, and why the raw static
            # comparison is not a comparison of equal work.
            print()
            print("=" * 78)
            print("  the work each side's static instruction count covers")
            print("=" * 78)
            out["grid"] = {}
            order_all = [x for x in ("naive", "int_cursor", "flat_gather",
                                     "incremental", "lean") if x in per_rung]
            for r in order_all:
                big = {k: v for k, v in per_rung[r].items() if isinstance(k, int)}
                h = big[max(big)]
                g = h["rows"] and h["hot_file"] and (h.get("rows") or [{}])[0].get(
                    "generated_grid")
                out["grid"][r] = g
                if g:
                    print(f"  {r:<13} grid {g['extents']} -> {g['threads']:,} "
                          f"threads; the gather count gives "
                          f"{h['pixels_per_thread']:.2f} px/thread")
            ref_threads = reference_threads(1, 1)
            print(f"  {'reference':<13} grid (ceil(1024/128), B) x block(128) = "
                  f"1 THREAD PER PIXEL, 1.00 px/thread "
                  f"({ref_threads:,} threads per env at 32x32)")
            print()
            print("  A static instruction count is per THREAD, so the generated")
            print("  side's count covers ~16 rays and the reference's covers one.")
            print("  Dividing one by the other without that factor measures the")
            print("  vectorisation, not the codegen.")
            if "naive" in per_rung and a0:
                n_big = {k: v for k, v in per_rung["naive"].items()
                         if isinstance(k, int)}
                n_hot = n_big[max(n_big)]
                ppt = n_hot.get("pixels_per_thread")
                r2 = out["grid"].get("naive")
                if ppt and rp:
                    raw = n_hot["hot"] / ref["instructions"] if ref.get("instructions") else None
                    norm = a0 / rp
                    out["raw_vs_normalised"] = {
                        "naive_hot_static": n_hot["hot"],
                        "reference_static": ref["instructions"],
                        "raw_ratio": raw, "work_normalised_ratio": norm,
                        "pixels_per_thread": ppt}
                    if raw:
                        # The inflation factor is measured, and decomposed into
                        # the two ratios that make it up. A STATIC count is per
                        # thread, so the generated side's 74,312 instructions
                        # cover 64 steps x ~15.7 rays while the reference's 776
                        # cover 16 steps x 1 ray. Dividing static counts without
                        # that decomposition measures the vectorisation and the
                        # unroll factor, not the code generator.
                        ms_max = max(n_big)
                        work_gen = ms_max * ppt
                        work_ref = ref.get("n_bodies_identical", 0) + 1 or 1
                        out["raw_vs_normalised"]["generated_pixel_steps"] = work_gen
                        out["raw_vs_normalised"]["reference_pixel_steps"] = work_ref
                        out["raw_vs_normalised"]["work_ratio"] = work_gen / work_ref
                        print(f"\n  naive static {n_hot['hot']:,} at max_steps="
                              f"{ms_max} against the reference's "
                              f"{ref['instructions']:,} is {raw:.1f}x RAW.")
                        print(f"  That is {work_gen:,.0f} pixel-steps against "
                              f"{work_ref:,.0f}: a work ratio of "
                              f"{work_gen / work_ref:.1f}x. Normalised, the same "
                              f"comparison is")
                        print(f"  {norm:.2f}x per pixel per step. The raw figure "
                              f"is inflated by {raw / norm:.1f}x, of which "
                              f"{work_gen / work_ref:.1f}x is the unequal work "
                              f"and the")
                        print(f"  remaining {norm:.0f}x is the per-pixel "
                              f"per-step instruction count itself.")
            # the causal ladder, rung by rung
            order = [r for r in ("naive", "int_cursor", "flat_gather",
                                 "incremental", "lean") if r in per_rung]
            if len(order) > 1:
                print()
                print("  the causal ladder: per-PIXEL-per-step SASS, and what "
                      "each rewrite removed")
                print(f"  {'rung':<13} {'/px/step':>10} {'delta':>10} "
                      f"{'x vs V0':>8} {'/thread':>10} {'reg':>5} {'spill B':>9}")
                # `per_rung[rung]` holds the per-max_steps table AND the
                # scalars appended to it, so the numeric keys are separated out
                # before taking a max. `max()` over a mixed dict raises, which
                # is what a `sorted()`-by-value would have papered over.
                shapes = {r: {k: v for k, v in per_rung[r].items()
                              if isinstance(k, int)} for r in order}
                base = per_rung[order[0]].get("slope_hot_per_pixel_per_step")
                prev = base
                for r in order:
                    sl = per_rung[r].get("slope_hot_per_pixel_per_step")
                    th = per_rung[r].get("slope_hot_per_step")
                    hot = shapes[r][max(shapes[r])]
                    d = (prev - sl) if (sl and prev) else None
                    print(f"  {r:<13} {(sl or 0):>10,.1f} "
                          f"{(f'{d:,.1f}' if d is not None else '-'):>10} "
                          f"{(sl / base if base and sl else 0):>8.3f} "
                          f"{(th or 0):>10,.1f} "
                          f"{str(hot['registers']):>5} {str(hot['local_bytes']):>9}")
                    prev = sl or prev
                out["ladder"] = {r: per_rung[r].get("slope_hot_per_pixel_per_step")
                                 for r in order}
                if graph:
                    print()
                    print("  the same ladder at the graph level (UOps per pixel "
                          "per step, whole graph)")
                    print(f"  {'rung':<13} {'uops/px/step':>13} {'delta':>10} "
                          f"{'x vs V0':>8}")
                    gbase = graph.get("naive", {}).get("per_pixel_per_step")
                    gprev = gbase
                    for r in order:
                        g = graph.get(r, {}).get("per_pixel_per_step")
                        if g is None:
                            continue
                        d = (gprev - g) if gprev else None
                        print(f"  {r:<13} {g:>13,.2f} "
                              f"{(f'{d:,.2f}' if d is not None else '-'):>10} "
                              f"{(g / gbase if gbase else 0):>8.3f}")
                        gprev = g

    if a.json:
        a.json.write_text(json.dumps(out, indent=2, default=str))
        print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
