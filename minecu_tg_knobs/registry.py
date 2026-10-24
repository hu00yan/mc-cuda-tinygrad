"""registry.py -- the eight knobs, each with the line that READS it.

Every entry here was found by reading the installed tinygrad, not by working
from a name. For each knob the registry records:

  * the `ContextVar` definition, `helpers.py:<line>`, with its default;
  * the `Context(...)` / environment key, which is **not** always the Python
    name (`USE_TC` is registered as `"TC"` -- see the naming trap below);
  * every site that reads it in the *scheduler or codegen*, with `file:line`
    and the source line itself, quoted;
  * whether the read is reachable for a kernel with no reduction axis;
  * what is predicted, so that a surprise is visible as a surprise.

`python -m minecu_tg_knobs.registry` re-reads every cited line out of the
installed package and prints it next to the claim. A cite that has drifted --
a tinygrad upgrade, a line number that moved -- fails there rather than
surviving into the note as a plausible-looking `file:line`.

THE NAMING TRAP, MEASURED
--------------------------
The brief lists `USE_TC`. `Context(USE_TC=0)` raises `KeyError: 'USE_TC'`.

`helpers.py:238`
    USE_TC, TC_SELECT, TC_OPT = ContextVar("TC", 1), ContextVar("TC_SELECT", -1), ContextVar("TC_OPT", 0)

`Context.__enter__` (helpers.py:169-171) looks each key up in
`ContextVar._cache`, and `ContextVar.__init__` (helpers.py:179-182) registers
under its FIRST ARGUMENT. So the registered key is the string `"TC"`. `USE_TC`
is a Python binding, not a key, and `helpers.py:238` is where the two names
diverge. The reachable spellings are `Context(TC=0)` and the environment
variable `TC=0`; `USE_TC=0` in the environment is read by nothing.

This matters beyond tidiness, and the reason is worth writing down. A sweep
that wrote `Context(USE_TC=0)` would raise. If the raise were caught -- and a
sweep that catches exceptions per configuration is a reasonable-looking thing
to write -- the configuration would be recorded as "USE_TC: no effect", which
is a true-sounding sentence about a knob that was never set. `registry.py`
records the brief's name in every label, sets the ContextVar key, and
`check_context_keys` re-verifies the spelling against `ContextVar._cache` at
run time so this cannot regress silently.

`OPT_LEVEL` DOES NOT EXIST
--------------------------
It is on the brief's list and it is not a `ContextVar` in the installed
tinygrad: `Context(OPT_LEVEL=0)` raises `KeyError`, and there is no
`ContextVar("OPT_LEVEL", ...)` line in `helpers.py` to find. It cannot be
swept, and a sweep that appeared to sweep it would be reporting a null result
about a knob that does not exist in this version. Recorded as an absence
rather than dropped, because "the brief named it and it is not here" is itself
the finding, and `notes/27-minecu-knobs-arm.md` reports it in the sweep table.

THE CACHE GATE
--------------
`to_program_config` (`codegen/__init__.py:493-494`) is the tuple tinygrad keys
its compiled-program cache on. A knob outside it would return the *default's*
compiled kernel, and the sweep would silently report the default again with a
knob label on it. All eight minus `OPT_LEVEL` are in that tuple -- verified,
not assumed, by `check_cache_key_coverage` -- which is why setting any of them
actually re-codegens.
"""

from __future__ import annotations

import dataclasses
from typing import Any

# The installed package's own tuple. Imported rather than transcribed, so a
# tinygrad upgrade that adds or removes a key is caught by the gate below
# instead of by a reader noticing.
def _to_program_config():
    from tinygrad.codegen import to_program_config
    return tuple(v.key for v in to_program_config)


# --------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Read:
    """One site that reads the knob, quoted from the installed source."""
    site: str        # "codegen/opt/heuristic.py:28"
    line: str        # the source line, verbatim
    reachable: bool  # can this workload reach it?
    why: str         # why, or why not


@dataclasses.dataclass(frozen=True)
class Knob:
    name: str             # the name the study brief uses
    context_key: str      # the key `Context(...)` / the environment uses
    default: Any
    sweep: tuple          # the values actually run. Not "could be run".
    defined_at: str       # "helpers.py:238"
    defined_line: str
    reads: tuple          # tuple[Read, ...]
    prediction: str       # so a surprise is visible as a surprise

    @property
    def reachable_here(self) -> bool:
        return any(r.reachable for r in self.reads)


def _K(*reads: Read) -> tuple[Read, ...]:
    return reads

KNOBS: tuple[Knob, ...] = (

    # ---------------------------------------------------------------- TC=0
    Knob(
        name="USE_TC", context_key="TC", default=1, sweep=(0, 1),
        defined_at="helpers.py:238",
        defined_line='USE_TC, TC_SELECT, TC_OPT = ContextVar("TC", 1), ContextVar("TC_SELECT", -1), ContextVar("TC_OPT", 0)',
        reads=_K(
            Read("codegen/opt/heuristic.py:28",
                 "if USE_TC > 0 and (len(k.axes_of(AxisType.GROUP_REDUCE, AxisType.REDUCE)) == 1 or (TC_OPT.value >= 1)):",
                 False,
                 "The guard needs ONE reduce axis (or TC_OPT>=1). The DDA kernel has "
                 "no reduction at all: every output element is a pure function of "
                 "one output index. So the conjunction is false and USE_TC is "
                 "never consulted for this kernel."),
            Read("codegen/opt/heuristic.py:32",
                 "try: rngs = tk.apply_opt(Opt(OptOps.TC, axis, (TC_SELECT.value, TC_OPT.value, USE_TC.value)))",
                 False,
                 "Inside the guard above, and reached only by a kernel the TC path "
                 "accepts."),
        ),
        prediction="no effect on this kernel, and no effect on any kernel in this "
                   "workload. It is the brief's own example of a knob for a "
                   "different problem: a GEMM schedule. If setting TC=0 changed "
                   "this kernel's SASS, the guard analysis above would be wrong and "
                   "that is the finding.",
    ),

    # ------------------------------------------------------------ TC_SELECT
    Knob(
        name="TC_SELECT", context_key="TC_SELECT", default=-1, sweep=(-1, 0, 2),
        defined_at="helpers.py:238",
        defined_line='USE_TC, TC_SELECT, TC_OPT = ContextVar("TC", 1), ContextVar("TC_SELECT", -1), ContextVar("TC_OPT", 0)',
        reads=_K(
            Read("codegen/opt/heuristic.py:32",
                 "try: rngs = tk.apply_opt(Opt(OptOps.TC, axis, (TC_SELECT.value, TC_OPT.value, USE_TC.value)))",
                 False,
                 "Read only while choosing a tensor core, which is behind the "
                 "same one-reduce-axis guard. -1 means 'iterate all and take the "
                 "first match'; 0 and 2 index the table."),
        ),
        prediction="no effect. It selects WHICH tensor core, and there is no tensor "
                   "core to select. Recorded as a no-op rather than omitted, "
                   "because 'the knob exists, is set, and does nothing' is a "
                   "different claim from 'the knob was not run'.",
    ),

    # --------------------------------------------------------------- TC_OPT
    Knob(
        name="TC_OPT", context_key="TC_OPT", default=0, sweep=(0, 1, 2),
        defined_at="helpers.py:238",
        defined_line='USE_TC, TC_SELECT, TC_OPT = ContextVar("TC", 1), ContextVar("TC_SELECT", -1), ContextVar("TC_OPT", 0)',
        reads=_K(
            Read("codegen/opt/heuristic.py:28",
                 "if USE_TC > 0 and (len(k.axes_of(AxisType.GROUP_REDUCE, AxisType.REDUCE)) == 1 or (TC_OPT.value >= 1)):",
                 False,
                 "This is the ONE knob of the three whose value can make the guard "
                 "true without a reduce axis -- the `TC_OPT.value >= 1` "
                 "disjunct. But the body it guards still calls OptOps.TC, which "
                 "needs reduce axes to match, so every axis raises "
                 "KernelOptError and the loop continues to `return` nothing. So "
                 "the guard can be true and the knob can still be inert. That is "
                 "the subtle one, and it is why the sweep measures rather than "
                 "argues."),
            Read("codegen/opt/search.py:23",
                 "actions += [Opt(op=OptOps.TC, axis=axis, arg=(-1, getenv(\"TC_OPT\", 2), getenv(\"TC\", 1))) for axis in range(9)]",
                 False,
                 "BEAM's action list, not the heuristic path. This sweep does not "
                 "enable BEAM -- BEAM is a search, not a knob, and switching it "
                 "on would replace the scheduler under test with a search over "
                 "schedulers, which is arm C's question, not arm B's."),
        ),
        prediction="no effect on this kernel. This is the specific claim the brief "
                   "asks to attack, because TC_OPT>=1 can flip the guard's first "
                   "disjunct. It is swept at 0, 1 and 2 precisely so that a claim "
                   "about the guard is a measurement rather than a reading.",
    ),

    # ----------------------------------------------------------- ALLOW_TF32
    Knob(
        name="ALLOW_TF32", context_key="ALLOW_TF32", default=0, sweep=(0, 1),
        defined_at="helpers.py:281",
        defined_line='ALLOW_TF32 = ContextVar("ALLOW_TF32", 0)  # allow tf32 to be used on NVIDIA GPUs',
        reads=_K(
            Read("codegen/opt/postrange.py:226",
                 "if self.ren.target.device in (\"CUDA\", \"NV\") and tc.dtype_in == dtypes.float and not ALLOW_TF32: continue",
                 False,
                 "The only consumer. It filters TENSOR CORES whose input is fp32, "
                 "on CUDA/NV. A T4 is sm_75 and `get_cuda` "
                 "(codegen/opt/tc.py:98) returns `cuda_sm75 = cuda_8168_f16` "
                 "(tc.py:94) -- half-input tensor cores ONLY. There is no fp32 "
                 "tensor core for sm_75, so this line never filters anything on "
                 "this card even for a matmul. On sm_80+ `cuda_8168_tf32` exists "
                 "and the knob would matter."),
        ),
        prediction="no effect, and for a stronger reason than 'no matmul here': on "
                   "sm_75 there is no fp32 tensor core to permit. This is the one "
                   "knob whose answer is partly an ARCHITECTURE fact rather than a "
                   "workload fact, and it is class (iv) -- it does not exist for "
                   "this architecture -- rather than class (ii). Sweeping it on an "
                   "A100 would be a different experiment and is not this one.",
    ),

    # ------------------------------------------------------------- NOLOCALS
    Knob(
        name="NOLOCALS", context_key="NOLOCALS", default=0, sweep=(0, 1),
        defined_at="helpers.py:239",
        defined_line='TRANSCENDENTAL, NOLOCALS = ContextVar("TRANSCENDENTAL", 1), ContextVar("NOLOCALS", 0)',
        reads=_K(
            Read("codegen/opt/heuristic.py:81",
                 "if resolve(prod(k.output_shape[i] for i in k.upcastable_dims) <= (240 if NOLOCALS else 2048), False):",
                 True,
                 "The upcast ceiling: 2048 elements by default, 240 with NOLOCALS. "
                 "Below the ceiling a dim is upcast to a wider vector type. This "
                 "is a real threshold and the DDA kernel's per-step temporaries "
                 "cross it, so NOLOCALS can change the generated vector types."),
            Read("codegen/opt/heuristic.py:163-164",
                 "k.apply_opt(Opt(OptOps.NOLOCALS))",
                 True,
                 "Forces `dont_use_locals` on every kernel "
                 "(postrange.py:122-126). Local memory in tinygrad is the "
                 "register-spill mechanism: an upcast that cannot live in "
                 "registers becomes a local access. This knob therefore speaks "
                 "directly to the 20 932 B spill notes/19 §4.2b records."),
        ),
        prediction="the ONE knob here that can plausibly change this kernel. It has "
                   "two independent effects -- a lower upcast ceiling and a forced "
                   "no-locals -- and they can move instruction count and register "
                   "count in OPPOSITE directions: fewer wide upcasts means less "
                   "register pressure (less spill) but more scalar instructions. "
                   "The interesting result is whichever way it lands, so the note "
                   "reports both the instruction count and the spill bytes rather "
                   "than one number that hides the trade.",
    ),

    # --------------------------------------------------------------- IMAGE
    Knob(
        name="IMAGE", context_key="IMAGE", default=0, sweep=(0, 1),
        defined_at="helpers.py:233",
        defined_line='IMAGE, FLOAT16, OPENPILOT_HACKS = ContextVar("IMAGE", 0), ContextVar("FLOAT16", 0), ContextVar("OPENPILOT_HACKS", 0)',
        reads=_K(
            Read("mixin/op.py:386", "if IMAGE: return self.image_dot(w, dtype)",
                 False,
                 "Replaces `dot` with an image-dot (texture-memory) form. The DDA "
                 "has no dot: it is a gather."),
            Read("mixin/op.py:1593",
                 "if IMAGE: return self.image_conv2d(weight, bias, groups, stride, dilation, padding, dtype)",
                 False, "Same, for conv2d. Neither op occurs in this graph."),
            Read("codegen/opt/heuristic.py:47-58",
                 "if image_valid_dims(buf.src[0].dtype, buf.src[0].max_numel(), k.ren.target.arch):",
                 False,
                 "Vectorises an image-shaped buffer by 4 when its shape is 4-wide "
                 "and divisible. The voxel grid is [B,WY,WX,WZ] with the Z axis "
                 "innermost, so it is a 3-D volume, not a 2-D image; "
                 "`image_valid_dims` wants a 2-D shape and rejects it."),
            Read("renderer/cstyle.py:602",
                 "if (d != dtypes.float16 or (bool(IMAGE) and bool(FLOAT16))) and d not in dtypes.fp8s+(dtypes.bfloat16,dtypes.double)",
                 False,
                 "FLOAT16 storage paths, which also need FLOAT16. This graph is "
                 "float32/int32/int8/uint8."),
        ),
        prediction="no effect. IMAGE is a texture-memory and 2-D-image-shape knob; "
                   "this kernel is a 3-D volume gather with no dot and no conv. The "
                   "reason is a shape property, so it is a property of the "
                   "workload and not of the compiler's judgement.",
    ),

    # ------------------------------------------------------- TRANSCENDENTAL
    Knob(
        name="TRANSCENDENTAL", context_key="TRANSCENDENTAL", default=1, sweep=(1, 2),
        defined_at="helpers.py:239",
        defined_line='TRANSCENDENTAL, NOLOCALS = ContextVar("TRANSCENDENTAL", 1), ContextVar("NOLOCALS", 0)',
        reads=_K(
            Read("codegen/__init__.py:372",
                 "get_transcendental_patterns(supported_ops, TRANSCENDENTAL>=2)",
                 True,
                 "The ONLY consumer, and it is a >=2 test, so the default 1 is the "
                 "off state. The function "
                 "(codegen/decomp/transcendental.py:268-277) is @functools.cache'd "
                 "on (ops, force_transcendental) and adds rewrite patterns for "
                 "EXP2, LOG2, SIN and SQRT."),
        ),
        prediction="**probably a real effect, and the prediction is the point.** The "
                   "obvious reading -- 'the DDA calls no transcendental, so this "
                   "cannot matter' -- is WRONG for a checkable reason. `rsqrt()` "
                   "lowers to SQRT + RECIPROCAL (measured: the op set of "
                   "`Tensor([4.,9.]).rsqrt()` is {SQRT, RECIPROCAL}), and the DDA "
                   "prologue calls `rsqrt` on every ray to normalise it. "
                   "transcendental.py:275-276 rewrites SQRT to a software `pow` "
                   "when the flag is set. So the knob reaches the DDA kernel's "
                   "prologue, where the naive port spends 152 of its 776-equivalent "
                   "instructions. Prediction: the DDA kernel's SASS GROWS, and the "
                   "image is unchanged because the DDA step decision is a compare "
                   "that a last-ulp change in ray_len does not flip at these ties.",
    ),

    # ----------------------------------------------------------- OPT_LEVEL
    Knob(
        name="OPT_LEVEL", context_key="OPT_LEVEL", default=None, sweep=(),
        defined_at="**absent**",
        defined_line="(no such ContextVar in the installed tinygrad)",
        reads=_K(
            Read("(no site)", "OPT_LEVEL does not appear anywhere in the package",
                 False,
                 "`Context(OPT_LEVEL=0)` raises KeyError('OPT_LEVEL'). The "
                 "closest real knobs are NOOPT (helpers.py:232) and BEAM "
                 "(helpers.py:232); neither is OPT_LEVEL, and NEITHER is on the "
                 "brief's list, so substituting one would be answering a different "
                 "question than the one asked."),
        ),
        prediction="**not sweepable: it does not exist in this version.** Recorded "
                   "as an absence. A sweep row for it that reads 'no effect' would "
                   "be a null result about a knob that was never set, which is the "
                   "same category error as the USE_TC naming trap.",
    ),
)


# --------------------------------------------------------------------------
# Run-time gates. Each of these refuses to report rather than reporting wrong.
# --------------------------------------------------------------------------
def check_context_keys() -> list[str]:
    """Every swept knob's Context key must exist in `ContextVar._cache`.

    This is the gate that would have caught `Context(USE_TC=...)`. A `KeyError`
    here means a knob in the table cannot be set, and a sweep that caught it per
    configuration would record a null result about a knob it never set.
    """
    from tinygrad.helpers import ContextVar
    bad = [k.name for k in KNOBS
           if k.sweep and k.context_key not in ContextVar._cache]
    if bad:
        raise AssertionError(
            f"knob(s) {bad} have a Context key that is not in ContextVar._cache; "
            f"Context(**{{key:...}}) would raise KeyError and this sweep would "
            f"record a null result about a knob it never set. Refusing.")
    return []


def check_cache_key_coverage() -> list[str]:
    """Every swept knob must be in tinygrad's own program-cache key tuple.

    `to_program_config` (`codegen/__init__.py:493-494`) is what the compiled
    program cache is keyed on. A knob outside it does not invalidate the cache,
    so the configuration returns the DEFAULT's compiled kernel and the sweep
    reports the default again with the knob's name on it.
    """
    keyed = set(_to_program_config())
    bad = [k.name for k in KNOBS if k.sweep and k.context_key not in keyed]
    if bad:
        raise AssertionError(
            f"knob(s) {bad} are not in tinygrad's to_program_config "
            f"({sorted(keyed)}); the program cache would return the default's "
            f"compiled kernel. Refusing to report.")
    return []


def check_cites(pkg_root: str | None = None) -> list[str]:
    """Re-read every cited `file:line` out of the installed package.

    A cite that has drifted -- a tinygrad upgrade moved a line -- fails here
    instead of surviving into the note as a plausible-looking reference. The
    check is that the quoted `line` is a substring of the cited line.
    """
    import pathlib
    if pkg_root is None:
        import tinygrad
        pkg_root = pathlib.Path(tinygrad.__file__).parent

    bad: list[str] = []
    for k in KNOBS:
        if k.defined_at == "**absent**":
            # the claim IS the absence; verify it really is absent
            from tinygrad.helpers import ContextVar
            if k.context_key in ContextVar._cache:
                bad.append(f"{k.name}: registry says absent but {k.context_key} "
                           f"IS in ContextVar._cache -- the cite is stale")
            continue
        for r in k.reads:
            if not r.site or r.site.startswith("("):
                continue
            path, _, span = r.site.partition(":")
            try:
                src = (pathlib.Path(pkg_root) / path).read_text().splitlines()
            except OSError:
                bad.append(f"{k.name}: cannot read {path}")
                continue
            # a site may be a single line ("28") or a short range ("47-58"); the
            # quoted text has to appear somewhere INSIDE the span, not only at
            # its first line, or a multi-line cite cannot be checked at all.
            first, _, last = span.partition("-")
            lo, hi = int(first), int(last or first)
            if not (1 <= lo <= hi <= len(src)):
                bad.append(f"{k.name}: {r.site} is out of range "
                           f"({len(src)} lines in {path})")
                continue
            window = "\n".join(src[lo - 1:hi])
            if r.line not in window:
                bad.append(f"{k.name}: {r.site} does not contain\n      {r.line!r}\n"
                           f"    window starts: {src[lo-1].strip()!r}")
    if bad:
        raise AssertionError("cited tinygrad lines have drifted:\n  "
                             + "\n  ".join(bad))
    return []


def check_all() -> dict:
    check_context_keys()
    check_cache_key_coverage()
    check_cites()
    return {
        "knobs": len(KNOBS),
        "sweepable": sum(1 for k in KNOBS if k.sweep),
        "cache_key": _to_program_config(),
    }


def main() -> int:
    info = check_all()
    print("=" * 78)
    print("  minecu_tg_knobs -- the knob registry, and its three gates")
    print("=" * 78)
    print(f"\n  all gates PASS. {info['knobs']} knobs, "
          f"{info['sweepable']} of them sweepable.")
    print(f"\n  tinygrad's program-cache key tuple, imported not transcribed:")
    print(f"    to_program_config = {info['cache_key']}")
    print("    codegen/__init__.py:493-497 -- a knob outside this tuple would")
    print("    return the default's compiled kernel and the sweep would report")
    print("    the default again with the knob's name on it.")

    for k in KNOBS:
        print()
        print("-" * 78)
        state = ("sweep " + ", ".join(repr(v) for v in k.sweep)) if k.sweep \
            else "NOT SWEEPABLE"
        print(f"  {k.name}")
        print(f"    Context key   {k.context_key!r}   default {k.default!r}   {state}")
        if k.context_key != k.name:
            print(f"    ** the Context key is NOT the brief's name. "
                  f"{k.name} is registered as {k.context_key!r}. **")
        print(f"    defined at    {k.defined_at}")
        print(f"      {k.defined_line}")
        print(f"    read at       {len(k.reads)} site(s), "
              f"{sum(1 for r in k.reads if r.reachable)} reachable for this kernel")
        for r in k.reads:
            mark = "REACHABLE" if r.reachable else "not reached"
            print(f"      [{mark}] {r.site}")
            print(f"         {r.line}")
        print(f"    prediction    {k.prediction}")

    print()
    print("=" * 78)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
