"""Every crossing in one file, with the TinyGrad source that defines it.

A CROSSING IS A DOCUMENTED EDIT POINT, NOT A PASTE
---------------------------------------------------
`Tensor.custom_kernel` is not used anywhere in this package, no `.cu` is
written or compiled alongside, and no PTX is inlined. Everything that runs is
CUDA that TinyGrad's own `CUDARenderer` emitted from a UOp graph built by
`render.py`. A crossing is one of:

  * a `ContextVar` that TinyGrad's `codegen/` or `schedule/` reads -- a real
    lever, cited to its definition and to its read site;
  * an *edit point* inside this package, i.e. a marked, named branch in
    `render.py` that a flag turns on. These exist because a crossing has to
    be a lever TinyGrad reads, not a rewrite that avoids the problem;
  * a **gap**, recorded as a gap. Where a technique cannot be expressed at
    all, the entry says so and the port keeps the nearest expressible
    approximation, labelled. Inventing a source-level restructuring to make
    the boundary disappear would be a different arm.

`file:line` citations are against tinygrad 0.14.0 as installed in this
repo's `.venv`, and each was read at that line, not taken from a name.

THE TWO MODES
-------------
    clean      no crossings. Framework's normal vocabulary only. Every knob
               at its default; the marked edit points all off.
    faithful   every crossing that TinyGrad actually exposes is on, and every
               one that it does not expose is recorded as a gap and left as
               the nearest expressible approximation.

The two differ ONLY in this file's dict. `render.py` reads `effective()` and
nothing else, so there is no second place where a crossing can hide.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. Crossings TinyGrad exposes, with the source that defines and reads them.
# ---------------------------------------------------------------------------
# Each entry: what it is, where it is DEFINED, where it is READ, what it does
# to this render, and whether it changes anything here. "no effect" entries
# are kept, not deleted: a knob that provably does not reach this kernel is a
# result, and a list of knobs that were *not* checked is not.
#
# MEASURED on this host with `probe_knobs.py`, which compiles the graph for
# sm_75 through the real renderer and compares the generated source. The
# "effect" column there is a source diff, which is the right instrument for
# "does this knob reach this graph at all" and the wrong one for anything
# downstream of ptxas.

EXPOSED: dict[str, dict] = {
    # ---- schedule / fusion -------------------------------------------------
    "MAX_KERNEL_BUFFERS": {
        "kind": "ContextVar",
        "defined": "tinygrad/helpers.py:248  MAX_KERNEL_BUFFERS = ContextVar(\"MAX_KERNEL_BUFFERS\", 0)",
        "read": "tinygrad/schedule/rangeify.py:363  if not (MAX_BUFS:=MAX_KERNEL_BUFFERS.value or "
                "DEVICE_MAX_BUFS.get(device, 0)): return None",
        "what": "caps the number of buffers one kernel may take; exceeding it "
                "inserts a bufferize and splits the fusion (rangeify.py:371-381).",
        "effect": "SAME, at 2, 4, 8 and 64. MEASURED by generated-source diff, "
                  "both modes, `probe_knobs.py --steps 4 --batch 8`. This is "
                  "the only knob in TinyGrad that splits a fusion, and it "
                  "splits on *argument count* (`if len(bufs) > MAX_BUFS - 1`, "
                  "rangeify.py:371) -- the DDA needs 2 inputs and 1 output, so "
                  "it is under every cap. A kernel that wants a register budget "
                  "and a kernel that wants few arguments are different problems "
                  "and the framework has the second one covered.",
    },
    "NOLOCALS": {
        "kind": "ContextVar",
        "defined": "tinygrad/helpers.py:239  TRANSCENDENTAL, NOLOCALS = ContextVar(\"TRANSCENDENTAL\", 1), "
                   "ContextVar(\"NOLOCALS\", 0)",
        "read": "tinygrad/codegen/opt/heuristic.py:163  if NOLOCALS: k.apply_opt(Opt(OptOps.NOLOCALS))",
        "what": "forbids LOCAL/shared axes, i.e. per-thread and per-block tiling.",
        "effect": "DIFFERS, small: 284 bytes of generated source out of 87 347 "
                  "(clean) and 88 708 (faithful) at batch 8, ms=4. MEASURED. "
                  "Launch geometry, not arithmetic: the DDA has no reduce axis, "
                  "so there is nothing to tile. This is the closest thing to a "
                  "lever for the shared-memory basis (T2), and the measurement "
                  "says it buys nothing, which is why it is off in both modes.",
    },
    "NOOPT": {
        "kind": "ContextVar",
        "defined": "tinygrad/helpers.py:230  DEV, DEBUG, BEAM, NOOPT = _DEV(\"DEV\", \"\"), "
                   "ContextVar(\"DEBUG\", 0), ContextVar(\"BEAM\", 0), ContextVar(\"NOOPT\", 0)",
        "read": "tinygrad/codegen/opt/postrange.py:346  elif not NOOPT and "
                "(ast.arg is None or ast.arg.applied_opts == ())",
        "what": "skips hand_coded_optimizations entirely.",
        "effect": "DIFFERS, large: 25 594 B against 87 347 B of generated "
                  "source (clean), 26 727 against 88 708 (faithful), at batch 8, "
                  "ms=4 -- a 2.4x reduction. MEASURED. This is the closest "
                  "thing to \"let the schedule be whatever it is\", and it is a "
                  "*worse* schedule for this kernel: with no opts the launches "
                  "are not tiled at all. Off in both modes; recorded because it "
                  "is the only knob that changes the geometry wholesale.",
    },
    "ALLOW_TF32": {
        "kind": "ContextVar",
        "defined": "tinygrad/helpers.py:281  ALLOW_TF32 = ContextVar(\"ALLOW_TF32\", 0)",
        "read": "tinygrad/codegen/opt/postrange.py:226  if self.ren.target.device in (\"CUDA\", \"NV\") "
                "and tc.dtype_in == dtypes.float and not ALLOW_TF32: continue",
        "what": "permits TF32 tensor cores.",
        "effect": "SAME, both modes. MEASURED. There is no matmul in the DDA, "
                  "so the tensor-core path (postrange.py:226) is never reached. "
                  "Listed because it is on the brief's knob list and it is worth "
                  "saying plainly that it cannot matter here.",
    },
    "USE_TC": {
        "kind": "ContextVar, set by the name `TC`",
        "defined": "tinygrad/helpers.py:238  USE_TC, TC_SELECT, TC_OPT = ContextVar(\"TC\", 1), "
                   "ContextVar(\"TC_SELECT\", -1), ContextVar(\"TC_OPT\", 0)",
        "read": "tinygrad/codegen/opt/heuristic.py:28  if USE_TC > 0 and "
                "(len(k.axes_of(AxisType.GROUP_REDUCE, AxisType.REDUCE)) == 1 or (TC_OPT.value >= 1))",
        "what": "tensor-core matching. TC_SELECT picks which TC; TC_OPT how eager.",
        "effect": "SAME for TC=0, TC_OPT=2, TC_SELECT=0 -- all three, both modes. "
                  "MEASURED. Same reason as ALLOW_TF32: the DDA has no reduce "
                  "axis, so the tensor-core path is never reached. The name is "
                  "recorded because `getenv(\"TC\", 1)` is the key a caller sets "
                  "and `USE_TC` is the variable's name; setting `USE_TC=0` "
                  "silently does nothing, which is the same failure as testing "
                  "the wrong knob.",
    },
    "TRANSCENDENTAL": {
        "kind": "ContextVar",
        "defined": "tinygrad/helpers.py:239 (same line as NOLOCALS)",
        "read": "tinygrad/codegen/__init__.py:372  pm_decomp = pm_decomp+get_late_rewrite_patterns(...)+"
                "get_transcendental_patterns(supported_ops, TRANSCENDENTAL>=2)",
        "what": "at >=2, enables the full transcendental rewrite set (polynomial "
                "sin/cos, range reduction).",
        "effect": "DIFFERS, large: 151 271 B against 87 347 B of generated "
                  "source (clean) at batch 8, ms=4. MEASURED. It reaches "
                  "`camera_basis`, not the DDA. Left at its default (1) in both "
                  "modes: this is a numerics knob, and the reference is built "
                  "with `--use_fast_math` while TinyGrad is not, so \"faithful\" "
                  "here would mean *changing the numerics being compared*, not "
                  "matching its schedule. See GAP:USE_FAST_MATH, which is the "
                  "honest form of that finding.",
    },
    "IMAGE": {
        "kind": "ContextVar",
        "defined": "tinygrad/helpers.py:237  IMAGE, FLOAT16, OPENPILOT_HACKS = ContextVar(\"IMAGE\", 0), ...",
        "read": "tinygrad/codegen/late/coalesce.py:6  from tinygrad.helpers import ..., IMAGE, ...",
        "what": "image-dtype handling and float4 upcasting of image shapes.",
        "effect": "SAME, both modes. MEASURED. `is_image_shape` "
                  "(tinygrad/helpers.py:36) requires a rank-3 shape with last "
                  "dim 4; the observation is uint8 [B,H,W,3], so it is not an "
                  "image and IMAGE has nothing to do. A same-name knob in a "
                  "different framework version selects the image *type*; in "
                  "0.14.0 it does not exist, and citing the name without the "
                  "definition would have been wrong.",
    },
    "DISABLE_FAST_IDIV": {
        "kind": "ContextVar",
        "defined": "tinygrad/helpers.py:243  DISABLE_FAST_IDIV = ContextVar(\"DISABLE_FAST_IDIV\", 1)",
        "read": "tinygrad/codegen/__init__.py:368  get_late_rewrite_patterns(supported_ops, "
                "bool(DISABLE_FAST_IDIV))",
        "what": "at 0, allows the magic-number integer-division sequence.",
        "effect": "SAME at 0, both modes. MEASURED. The reference's kernel does "
                  "one integer divide per thread (`pixel_offset / width`, "
                  "kernels.cu:590), which is in the prologue and shared by 16 "
                  "unrolled steps; this port's DDA body has no integer division "
                  "at all once the address is maintained. A prologue cost, not "
                  "a per-step one.",
    },
}

# ---------------------------------------------------------------------------
# 2. Crossings that are edit points inside this package.
# ---------------------------------------------------------------------------
# These are NOT TinyGrad knobs. They are the two places where the reference's
# source says something TinyGrad's source language can say differently but not
# better, and the port exposes both so the boundary map can price them.
#
# `ADDR_BITS` and `UNSIGNED_BOUNDS` are *fidelity* switches: they make the port
# say what the reference says. They are expected to be worth ~nothing on sm_75,
# and the SASS census in `sass.py` reports both so that expectation is checked
# rather than inherited from `notes/19` §3.2.

EDIT_POINTS: dict[str, dict] = {
    "addr_bits": {
        "what": "width of the maintained flat voxel address. 64 makes the "
                "port say `long long` as `kernels.cu:669` does; 32 narrows it.",
        "reference": "kernels.cu:634-636, :669 -- long long throughout (35 "
                     "sites; notes/18 entry 13). `IMAD.WIDE.U32` x4 in the "
                     "reference's hot kernel is this.",
        "reaches": "the gather index dtype. TinyGrad's index lowering "
                   "(uop/weak.py:74-81 pm_lower_index_dtype) NARROWS a long "
                   "index to int when `buf.max_numel()-1 <= int32.max`, which "
                   "is true here, so 64 bits may not survive codegen. That is "
                   "itself a finding and `sass.py` reports what survived.",
    },
    "unsigned_bounds": {
        "what": "the bounds test. True = the reference's "
                "`(unsigned)voxel_x < (unsigned)wx` (kernels.cu:666-668). "
                "False = three signed compares.",
        "reference": "kernels.cu:666-668. In SASS: ISETP.GE.U32.OR.",
        "reaches": "the inside predicate. Both are 3 instructions on sm_75 "
                   "(notes/19 §3.2 finding 1), re-derived in `sass.py`.",
    },
}

# ---------------------------------------------------------------------------
# 3. Gaps. Techniques with no TinyGrad lever. Recorded, not worked around.
# ---------------------------------------------------------------------------
# A gap is not a failure of this port. It is the result. Each entry says what
# the reference does, what TinyGrad offers instead, and what the *consequence*
# is -- ideally with a number, and honestly marked if it is not measured.

GAPS: dict[str, dict] = {
    "EARLY_EXIT": {
        "technique": "T6 -- a per-ray early exit.",
        "reference": "kernels.cu:670  `if (block >= 0) { hit_block = block; "
                     "goto done_pre; }`. Costs 33 `BSSY`/`BSYNC` pairs in the "
                     "reference (measured, `sass.py`) and stops a ray that has "
                     "hit from executing the rest of its unrolled window.",
        "tinygrad": "Nothing. The masked formulation (`live = hit < 0`) is the "
                    "only expressible form, and it means every ray executes "
                    "every one of `max_steps` bodies.",
        "why_no_lever": "A per-ray trip count is not a fixed-size tensor, and "
                        "TinyGrad's IR has no loop: `Ops` (uop/ops.py) has no "
                        "WHILE, LOOP or REPEAT; the only control flow is "
                        "IF/ENDIF, synthesised for gated stores at "
                        "codegen/__init__.py:402-407 and unreachable from the "
                        "Tensor API.",
        "class": "(i) cannot express",
        "consequence": "The generated code is CHEAPER per step (0 BSSY against "
                       "33) and does MORE total work: 32 steps per ray against "
                       "a measured mean of 26.4 (`notes/19` §4.2c). That is a "
                       "1.21x work term in the reference's favour and it is "
                       "structural, not a cost-model miss.",
    },
    "UNROLL_FACTOR": {
        "technique": "T7 -- a 16-copy static window of the DDA body.",
        "reference": "kernels.cu:681-687  `#pragma unroll 4` over a body that "
                     "is itself 4 `DDA_STEP_PRE()` calls = 16 static copies. "
                     "The SASS has 16 `LDG.E.S8.CONSTANT.SYS` at a load-to-load "
                     "spacing of exactly 38, fifteen times, and an exact-match "
                     "run at period 38 (measured, `sass.py`). ptxas allocated "
                     "them at 28 registers with zero spill.",
        "tinygrad": "The loop is inlined whole. There is no unroll factor for "
                    "an elementwise chain: `OptOps.UNROLL` "
                    "(codegen/opt/__init__.py:7) is applied only to "
                    "`unrollable_dims`, which `Scheduler.real_axis` "
                    "(codegen/opt/postrange.py:113) resolves to reduce axes, and "
                    "`hand_coded_optimizations` reaches it only inside "
                    "`if k.unrollable_dims` (codegen/opt/heuristic.py:139-152). "
                    "The DDA has no reduce axis.",
        "why_no_lever": "No ContextVar controls how many copies of an elementwise "
                        "chain are emitted. BEAM (helpers.py:230) searches the "
                        "same `Opt` space and would not find it either, because "
                        "the action does not exist in the space.",
        "class": "(i) cannot express",
        "consequence": "Two possible outcomes, both measured, and neither is "
                       "the reference's: below ~1M pixels one kernel per step "
                       "with the state round-tripping global memory; above it, "
                       "one fused kernel of 36 384 SASS instructions at 168 "
                       "registers with 20 932 B of spill (`notes/19` §4.1, "
                       "re-derived in `sass.py`).",
    },
    "FUSE_AT_REGISTER_BUDGET": {
        "technique": "T8 -- the register footprint, and the schedule that "
                     "keeps it.",
        "reference": "28 registers, 0 spill, 0 stack frame "
                     "(`minecu_ref/build/resources_sm_75.txt`, re-read in "
                     "`sass.py`). ptxas allocates 16 identical steps one at a "
                     "time because they ARE identical -- the exact-match run at "
                     "period 38 is the mechanism.",
        "tinygrad": "The fused kernel's 32 steps are not identical: ptxas "
                    "CSEs and reorders across the whole body, and the longest "
                    "exact match at any period 20..200 is a run of 18, not 32 "
                    "copies (measured, `sass.py:repeat_period`). No knob sets a "
                    "register target: `__launch_bounds__` is emitted from the "
                    "product of the local dims (renderer/cstyle.py:156-157, "
                    "411) and there is no `maxrregcount` path.",
        "why_no_lever": "MAX_KERNEL_BUFFERS counts arguments, not registers "
                        "(rangeify.py:371). NOLOCALS changes launch geometry, "
                        "not the register allocator. NOOPT removes tiling "
                        "entirely. None of them is a register budget.",
        "class": "(i) for the budget; (ii) for the fusion boundary, which is "
                 "where a cost model could help",
        "consequence": "Measured in the boundary map. This is the one where a "
                       "cost model and a lever are both missing, which is why it "
                       "is listed separately from UNROLL_FACTOR.",
    },
    "USE_FAST_MATH": {
        "technique": "T3's numerics, and the prologue's.",
        "reference": "`nvcc -O3 --use_fast_math` (minecu_ref/build.sh). That "
                     "one flag sets --ftz=true --prec-div=false "
                     "--prec-sqrt=false, which is why `1.0f/ray_x` is a "
                     "`MUFU.RCP` and `rsqrtf` is a `MUFU.RSQ`.",
        "tinygrad": "TinyGrad's NVRTC path passes no fast-math flag: "
                    "NVRTCCompiler.compile_options is "
                    "['--gpu-architecture=...', '-I...', '--minimal'] and "
                    "nothing else (runtime/support/compiler_cuda.py:46-56). "
                    "There is no ContextVar for it and no renderer knob. The "
                    "generated source is `1/sqrt(...)` where the reference is "
                    "`rsqrtf` -- a real numerical difference, and `notes/19` "
                    "§1.2 measured the resulting images to be byte-identical on "
                    "the reference's own state, which is a fact about that state "
                    "and not a guarantee.",
        "why_no_lever": "It is a compiler flag on a different compiler. Making "
                        "the port match would mean changing numerics, which "
                        "changes the observation being compared -- so the honest "
                        "form of this crossing is a recorded gap, not a flag.",
        "class": "(i) cannot express",
        "consequence": "Bound in the boundary map by what it is worth: the "
                       "prologue is 152 of the reference's 776 instructions "
                       "(19.6%), and fast-math touches the reciprocal and the "
                       "rsqrt inside it.",
    },
    "SHARED_BASIS": {
        "technique": "T2 -- the camera basis in `__shared__`.",
        "reference": "kernels.cu:581 `__shared__ float s_basis[14]`, loaded by "
                     "the first 14 threads (`:594-596`) and `__syncthreads()` "
                     "at `:597`. One barrier, 64 B smem "
                     "(`resources_sm_75.txt`).",
        "tinygrad": "LOCAL/shared buffers only exist as the accumulator of a "
                    "reduce (`add local buffers` after `remove reduces`, "
                    "codegen/__init__.py:325-327; a LOCAL axis needs "
                    "`ren.has_local`, codegen/opt/heuristic.py:162). An "
                    "elementwise kernel cannot declare one, so the basis is "
                    "read from global memory by every thread.",
        "why_no_lever": "NOLOCALS *removes* locals; it does not add one. There "
                        "is no path from the Tensor API to a shared buffer that "
                        "is not a reduction's scratch.",
        "class": "(i) for the carveout, (ii) for the effect",
        "consequence": "Cheap, and this is a negative result worth having: the "
                       "basis is 416 bytes read by every thread, which is a "
                       "broadcast L1 read. `notes/19` §6 gap 8 already called it "
                       "the cheapest of the argument list; nothing here "
                       "contradicts that, and the boundary map records it as "
                       "not a gap.",
    },
    "GRID_AS_ENV": {
        "technique": "T1 -- `b = blockIdx.y`, one block per env.",
        "reference": "kernels.cu:584, launched at `:2182` as "
                     "`dim3(blocks_per_batch, batch_size)`. Exists so the basis "
                     "is block-uniform, i.e. so T2 is possible.",
        "tinygrad": "The grid is derived from the tensor shape: `add_gpudims` "
                    "(codegen/gpudims.py:41-58) assigns global dims in sorted "
                    "order and `ProgramInfo.from_sink` "
                    "(uop/ops.py:1241-1261) reads the result off the "
                    "`Ops.SPECIAL` nodes. Which axis lands on `blockIdx.y` is a "
                    "function of the shape and the opts, not something the "
                    "program states.",
        "why_no_lever": "No ContextVar selects a grid layout.",
        "class": "(i) for the layout, and inert anyway",
        "consequence": "Only exists to enable T2, which is itself a gap. "
                       "Recorded so the row exists and is closed.",
    },
    "RESTRICT_ON": {
        "technique": "`__restrict__` on every buffer parameter, which is what "
                     "makes nvcc emit `LDG.E.S8.CONSTANT.SYS` (the read-only "
                     "path) and keep values in registers across the store.",
        "reference": "kernels.cu:569-572, 81 sites (notes/18 entry 21). SASS: "
                     "16 of 17 global loads are `LDG.E.S8.CONSTANT.SYS` "
                     "(measured).",
        "tinygrad": "`CUDARenderer` does not set `buffer_suffix`; the base class "
                    "has `buffer_suffix: str = \"\"` (renderer/cstyle.py:121) "
                    "and only `ClangRenderer` sets it to `\" restrict\"` "
                    "(cstyle.py:269). Measured on the generated source: 0 "
                    "occurrences of `restrict` in any captured kernel.",
        "why_no_lever": "There is no ContextVar for it and no dtype or shape "
                        "that triggers it. Reaching it means editing the "
                        "renderer, which is not this port's to edit.",
        "class": "(i) cannot express; the SASS consequence is measurable",
        "consequence": "The generated side's loads are plain `LDG.E.SYS` "
                       "(`sass.py`), not `.CONSTANT.SYS`. Whether that costs time "
                       "on sm_75 is in the boundary map, and the honest answer "
                       "may well be 'nothing' -- which is a result, and a "
                       "different one from 'not measured'.",
    },
}

# ---------------------------------------------------------------------------
# 4. The mode table. This is the whole switch.
# ---------------------------------------------------------------------------

_MODES: dict[str, dict] = {
    "clean": {
        "addr_bits": 32,
        "unsigned_bounds": True,
        "context": {},
    },
    "faithful": {
        "addr_bits": 64,
        "unsigned_bounds": True,
        # No `ContextVar` is ON in faithful mode, and that is the honest
        # outcome rather than an oversight. NOLOCALS is the only candidate
        # crossing for SHARED_BASIS, and `probe_knobs.py` measures that it
        # changes the generated source by 284 bytes out of 88 708 at batch 8 /
        # ms=4 -- launch geometry and nothing else -- and that the DDA has no
        # reduce axis for the shared buffer it would otherwise create.
        # NOOPT is a candidate for UNROLL_FACTOR and it changes 62 KB of
        # source in the wrong direction. MAX_KERNEL_BUFFERS is the only
        # fusion-splitting lever and it is SAME at every value tried, because
        # it counts arguments. Turning any of them on to *look* faithful would
        # be a costume; leaving them off and saying why is the point of this
        # file. The one switch that IS on is `addr_bits`, and it is an edit
        # point, not a knob -- see §6.1 of notes/28 for what it costs.
        "context": {},
    },
}

MODES = tuple(_MODES)


def effective(mode: str, **overrides) -> dict:
    """Resolve a mode into the exact values `render.py` will use.

    The single place the selection happens. Raises on an unknown crossing
    rather than ignoring it, because a typo'd crossing name that silently
    does nothing is exactly how a boundary map acquires a row that was never
    measured.
    """
    if mode not in _MODES:
        raise SystemExit(f"unknown mode {mode!r}; have {sorted(_MODES)}")
    cfg = dict(_MODES[mode])
    bad = set(overrides) - set(EDIT_POINTS)
    if bad:
        raise SystemExit(f"unknown crossing(s) {sorted(bad)}; "
                         f"edit points are {sorted(EDIT_POINTS)}")
    cfg.update(overrides)
    return cfg


def crossings_table() -> str:
    """The crossing inventory as text, for the note. Data, not prose."""
    out = []
    out.append("EXPOSED KNOBS (defined, read, and what they do here)")
    for k, v in EXPOSED.items():
        out.append(f"  {k}")
        out.append(f"    defined: {v['defined']}")
        out.append(f"    read:    {v['read']}")
        out.append(f"    effect:  {v['effect']}")
    out.append("EDIT POINTS (in this package; both are fidelity switches)")
    for k, v in EDIT_POINTS.items():
        out.append(f"  {k}: {v['what']}")
        out.append(f"    reference: {v['reference']}")
        out.append(f"    reaches:   {v['reaches']}")
    out.append("GAPS (no TinyGrad lever; recorded, not worked around)")
    for k, v in GAPS.items():
        out.append(f"  {k}  [{v['class']}]")
        out.append(f"    reference: {v['reference']}")
        out.append(f"    tinygrad:  {v['tinygrad']}")
        out.append(f"    why not:   {v['why_no_lever']}")
        out.append(f"    cost:      {v['consequence']}")
    return "\n".join(out)
