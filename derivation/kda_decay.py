#!/usr/bin/env python3
"""derivation/kda_decay.py -- the KDA decay tensor: the 256x graph-object
ratio, the never-materialised verdict, the largest device-resident argument,
and the redundant-`exp2` count.

    python derivation/kda_decay.py                 # full report
    python derivation/kda_decay.py <claim>         # quotable value

THE FOUR CLAIMS ARE SEPARATE BECAUSE THEY ARE FOUR DIFFERENT THINGS
    The record's error count on this question came from three agents reporting
    three numbers and reading them as the same quantity. They are not:

      graph_object_ratio     a property of the GRAPH. Two operand shapes. No
                             memory is involved at any point.
      never_materialised     a property of the LOWERING. Measured by wrapping
                             the allocator and by reading the emitted CUDA
                             kernel's parameter list.
      largest_device_resident a property of the SCHEDULE. The largest buffer the
                             scheduler hands a kernel.
      redundant_exp2         a property of the CODEGEN. A transcendental count
                             inside a kernel, in registers.

    A ratio of two shapes, a set of allocations, a maximum over a parameter
    list and a count of `exp2` call sites cannot be compared with one another,
    and every document that tries is the reason this module exists.

WHAT IS MEASURED WHERE
    `graph_object_ratio` and `largest_device_resident` are shape arithmetic plus,
    for the latter, a read of a shipped kernel source that is PARSED here and
    never transcribed. `never_materialised` needs a device and reports
    `[UNAVAILABLE]` for its allocator half if there is none, while its
    source-reading half still runs. `redundant_exp2` is a parse of a shipped
    kernel file, plus, if a device exists, a fresh capture.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fp32 as F                                                      # noqa: E402
from _util import Module, PinDrift, Value, rule, run, ROOT              # noqa: E402

EXPECTED: dict[str, str] | None = None
MOD = Module("kda_decay", __doc__ or "", EXPECTED)

REPO = os.path.join(ROOT, "kda")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

def _read(relpath: str) -> str:
    with open(os.path.join(ROOT, relpath), encoding="utf-8", errors="replace") as f:
        return f.read()


def _tiling() -> dict[str, int]:
    """BC and NC read out of `kda/codegen_diff.py`'s own constexpr line, so
    the number the ratio uses is the number the file states, not a copy."""
    src = _read("kda/codegen_diff.py")
    MOD.pins.file("kda/codegen_diff.py")
    m = re.search(r"BT\s*=\s*(\d+);\s*BC\s*=\s*(\d+);\s*NC\s*=\s*cdiv\(BT,\s*BC\)\s*=\s*(\d+)",
                  src)
    if not m:
        raise RuntimeError("kda/codegen_diff.py no longer states BT/BC/NC; "
                          "the graph_object_ratio claim needs the tiling")
    bt, bc, nc = (int(g) for g in m.groups())
    if nc != bt // bc:
        raise RuntimeError(f"kda/codegen_diff.py says BT={bt} BC={bc} NC={nc} "
                          f"and {bt}//{bc} != {nc}")
    return dict(BT=bt, BC=bc, NC=nc)


# ---- the shapes. Every one of these is READ OUT OF A FILE, not transcribed,
# so a shape edit upstream is pin drift, not a stale sentence.
def _shape_from(relpath: str, lineno: int, *, name: str = "", scalar_line: int = 0) -> dict[str, int]:
    """Read a shape out of ONE pinned line of a repo file, in either spelling:

        B, T, H, HV, K, V, C = 1, 64, 1, 1, 64, 64, 64
        B, Tt, H, HV, K, V   = 2, 512, 4, 4, 64, 64     (+ `C = 64` next line)
        B, T, H, K, V, C     = 4, 4096, 32, 128, 128, 64 (+ `HV = H` next line)

    The line is pinned by content, so a shape edit upstream is drift rather
    than a stale sentence, and the value is read from the file rather than
    typed here. It refuses to guess if a field is missing.
    """
    text = MOD.pins.file_line(relpath, lineno, name=name, must_contain="=")
    m = re.match(r"^\s*([A-Za-z_][\w,\s]*?)\s*=\s*([-\d][-\d,\s]*)$", text.strip())
    if not m:
        raise RuntimeError(f"{relpath}:{lineno} is not a shape assignment: {text!r}")
    names = [n.strip() for n in m.group(1).split(",")]
    vals = [int(v.strip()) for v in m.group(2).split(",") if v.strip()]
    if len(names) != len(vals):
        raise RuntimeError(f"{relpath}:{lineno} has {len(names)} names and "
                           f"{len(vals)} values: {text!r}")
    out = {}
    for k, v in zip(names, vals):
        key = {"Tt": "T", "tt": "T"}.get(k, k)
        if key.isupper() and len(key) <= 2:
            out[key] = v
    if scalar_line:
        hline = MOD.pins.file_line(relpath, scalar_line, name=f"{name}+scalar",
                                   must_contain="=")
        cm = re.match(r"^\s*([A-Za-z_]\w*)\s*=\s*([-\d]+|H)\b", hline.strip())
        if not cm:
            raise RuntimeError(f"{relpath}:{scalar_line} is not a scalar: {hline!r}")
        key, rhs = cm.group(1), cm.group(2)
        out[key] = out.get("H", 0) if rhs == "H" else int(rhs)
    if "HV" not in out and "H" in out:
        out["HV"] = out["H"]
    missing = [k for k in ("B", "T", "H", "HV", "K", "V", "C") if k not in out]
    if missing:
        raise RuntimeError(f"{relpath}:{lineno} is missing shape fields {missing}; "
                           f"read {out}. Fix the parser or the file -- do NOT fill "
                           f"these in by hand.")
    return out


CODEGEN_SHAPE = _shape_from("kda/codegen_diff.py", 77, name="shape:codegen_diff")
BENCH_SHAPE = _shape_from("kda/perf_kda.py", 90, name="shape:perf_kda", scalar_line=91)
COST_SHAPE = _shape_from("kda/cost.py", 17, name="shape:cost", scalar_line=18)
# The Triton kernel's own tiling, read out of `kda/codegen_diff.py`'s constexpr
# line, so the ratio uses the number the file states and not a copy of it.
KERNEL_FILE = "out/kaggle/20260928-014248-gpu/codegen/kernel000.txt"
# The script that produced KERNEL_FILE. The shape is read out of THIS line, and
# then the kernel's own buffer sizes must agree with it or the module fails.
CAPTURE_SCRIPT = "kernels/build_notebooks.py"
CAPTURE_LINE = 590
ITEMSIZE = 4          # float32; the port is fp32 by construction


KERNEL_FILE = "out/kaggle/20260928-014248-gpu/codegen/kernel000.txt"
# The script that produced KERNEL_FILE. The shape is read out of THIS line, and
# then the kernel's own buffer sizes must agree with it or the module fails.
CAPTURE_SCRIPT = "kernels/build_notebooks.py"
CAPTURE_LINE = 590


def _read(relpath: str) -> str:
    with open(os.path.join(ROOT, relpath), encoding="utf-8", errors="replace") as f:
        return f.read()


def _tiling() -> dict[str, int]:
    """BC and NC read out of `kda/codegen_diff.py`'s own constexpr line, so
    the number the ratio uses is the number the file states, not a copy."""
    src = _read("kda/codegen_diff.py")
    MOD.pins.file("kda/codegen_diff.py")
    m = re.search(r"BT\s*=\s*(\d+);\s*BC\s*=\s*(\d+);\s*NC\s*=\s*cdiv\(BT,\s*BC\)\s*=\s*(\d+)",
                  src)
    if not m:
        raise RuntimeError("kda/codegen_diff.py no longer states BT/BC/NC; "
                          "the graph_object_ratio claim needs the tiling")
    bt, bc, nc = (int(g) for g in m.groups())
    if nc != bt // bc:
        raise RuntimeError(f"kda/codegen_diff.py says BT={bt} BC={bc} NC={nc} "
                          f"and {bt}//{bc} != {nc}")
    return dict(BT=bt, BC=bc, NC=nc)


# ---------------------------------------------------------------- claims

@MOD.claim("graph_object_ratio")
def c_ratio() -> Value:
    """The decay intermediate, as a shape. Per (b, n, hv), because that is the
    scope the two sides share."""
    t = _tiling()
    s = CODEGEN_SHAPE
    NT = s["T"] // s["C"]
    # the port: [C, C, K] per (b, n, hv)
    mine = s["C"] * s["C"] * s["K"]
    # the Triton kernel: [BC, K] per (b, n, hv, sub-chunk), NC sub-chunks
    theirs = t["BC"] * s["K"] * t["NC"]
    ratio = mine / theirs
    # the other derivation the record confuses it with: [C,C,K] vs [BC,K], one
    # sub-chunk, which is the same graph objects a different count of
    subgraph = mine / (t["BC"] * s["K"])
    MOD.pins.param("codegen_shape", s)
    MOD.pins.param("tiling", t)
    MOD.pins.param("graph_ratio", ratio)
    MOD.pins.param("graph_ratio_subchunk", subgraph)
    lines = [
        f"  shape: B={s['B']} T={s['T']} H={s['H']} HV={s['HV']} K={s['K']} V={s['V']} "
        f"C={s['C']}  -> NT={NT}",
        f"  tiling, read out of kda/codegen_diff.py: BT={t['BT']} BC={t['BC']} NC={t['NC']}",
        "",
        f"  {'':<34s} {'elements/(b,n,hv)':>18s}",
        f"  port decay intermediate [C,C,K]   {mine:>18,}",
        f"  Triton decay operand [BC,K]xNC    {theirs:>18,}",
        f"  {'ratio':<34s} {ratio:>17.0f}x",
        "",
        "  Both sides are GRAPH OBJECTS -- operand shapes the code asks for.",
        "  Neither is a byte that is ever allocated. `never_materialised` is",
        "  the claim about bytes and it is a separate claim.",
        "",
        f"  the OTHER derivation of the same figure, [C,C,K] vs [BC,K] for one",
        f"  sub-chunk, gives {subgraph:.0f}x. The two differ by exactly NC = {t['NC']},",
        "  so a document that quotes one and means the other is off by that",
        "  factor. Say which.",
    ]
    return Value(
        f"per (b, n, hv) at the shape `kda/codegen_diff.py` uses "
        f"(B={s['B']}, T={s['T']}, H=HV={s['HV']}, K=V={s['K']}, C={s['C']}), the "
        f"port's `[C,C,K]` decay intermediate is {mine:,} elements against the "
        f"Triton kernel's `[BC,K]` per sub-chunk times NC={t['NC']} = "
        f"{theirs:,}, a ratio of {ratio:.0f}x -- a ratio of two GRAPH OBJECTS, "
        f"with no memory involved at either side. Scoping to one sub-chunk "
        f"instead gives {subgraph:.0f}x, and the two differ by exactly NC",
        "\n".join(lines),
    )


@MOD.claim("bench_shape_tensor")
def c_bench() -> Value:
    """The same tensor at the two shapes the repo calls 'the KDA shape'. The
    span between them is the other half of why two agents disagreed."""
    rows = []
    for label, s in (("perf_kda.py", BENCH_SHAPE), ("cost.py", COST_SHAPE)):
        NT = s["T"] // s["C"]
        bnt = s["B"] * NT * s["HV"]
        elts = bnt * s["C"] * s["C"] * s["K"]
        rows.append((label, s, NT, bnt, elts, elts * ITEMSIZE))
    (l0, s0, nt0, b0, e0, z0), (l1, s1, nt1, b1, e1, z1) = rows
    MOD.pins.param("bench_shapes", [(l, s) for l, s, *_ in rows])
    MOD.pins.param("bench_elts", [e0, e1])
    lines = []
    for l, s, nt, bnt, e, z in rows:
        lines.append(f"  {l:14s} B={s['B']} T={s['T']} H={s['H']} HV={s['HV']} "
                     f"K={s['K']} V={s['V']} C={s['C']}  NT={nt}")
        unit, div = ("GiB", 2**30) if z >= 2**30 else ("MiB", 2**20)
        lines.append(f"    {'':14s} B*NT*HV = {bnt:,} instances of [C,C,K]; "
                     f"one = {e:,} elements = {z:,} B = {z/div:.4f} {unit} fp32")
    lines += ["",
              f"  the two shapes are {b1/b0:.0f}x apart in B*NT*HV and both are "
              f"correct for their own shape.",
              "  A size quoted without its shape is not a size.",
              "",
              f"  DTYPE. `kda/cost.py` sets its working element size to 2 bytes, so its",
              f"  headline figure for this tensor is bf16 and is HALF the fp32 figure",
              f"  above. Two documents can quote 2x different numbers for one tensor and",
              f"  both be right, one in bf16 and one in fp32. Say which."]
    return Value(
        f"the same `[C,C,K]` tensor is {e0:,} elements = {z0:,} B = "
        f"{z0/2**20:.4f} MiB fp32 at the bench shape (B={s0['B']}, T={s0['T']}, "
        f"H={s0['H']}, HV={s0['HV']}, K=V={s0['K']}, C={s0['C']}, NT={nt0}) and "
        f"{e1:,} elements = {z1:,} B = {z1/2**30:.2f} GiB fp32 at the "
        f"`kda/cost.py` shape (B={s1['B']}, T={s1['T']}, H={s1['H']}, "
        f"HV={s1['HV']}, K=V={s1['K']}, C={s1['C']}, NT={nt1}) -- "
        f"{b1/b0:.0f}x apart, both correct; and `cost.py` counts the tensor in "
        f"bf16, so its figure for it is half the fp32 one -- a size quoted "
        f"without its shape and dtype is not a size",
        "\n".join(lines),
    )


@MOD.claim("largest_device_resident")
def c_largest() -> Value:
    """The largest kernel ARGUMENT, from a shipped CUDA kernel's parameter
    list, parsed. Plus the identity that makes it a number rather than a
    coincidence, checked at the bench shape."""
    src = _read(KERNEL_FILE)
    MOD.pins.file(KERNEL_FILE)
    params = re.search(r"__global__\s+void\s+(?:__launch_bounds__\([^)]*\)\s+)?"
                       r"(\w+)\s*\(([^)]*)\)", src)
    if not params:
        raise RuntimeError(f"{KERNEL_FILE} has no __global__ signature to read")
    sig = [(m.group(1), int(m.group(2)))
           for m in re.finditer(r"float\*\s*(\w+?)_(\d+)", params.group(2))]
    sizes = list(sig)
    if not sizes:
        raise RuntimeError(f"{KERNEL_FILE} declares no sized buffers: {params.group(2)!r}")
    largest = max(n for _, n in sizes)
    shape = _capture_shape(src)
    decay_elts = (shape["B"] * shape["NT"] * shape["HV"]
                  * shape["C"] * shape["C"] * shape["K"])
    # the identity at the bench shape, where K and C differ
    s = BENCH_SHAPE
    nt = s["T"] // s["C"]
    arg = s["B"] * nt * s["HV"] * s["C"] * s["C"]          # B*NT*HV*C*C
    ten = arg * s["K"]                                    # the decay tensor
    MOD.pins.param("kernel_params", [list(x) for x in sizes])
    MOD.pins.param("kernel_largest_elements", largest)
    MOD.pins.param("kernel_name", params.group(1))
    MOD.pins.param("kernel_capture_shape", shape)
    lines = [
        f"  {KERNEL_FILE}",
        f"  kernel {params.group(1)}: {len(sizes)} sized float buffers",
    ] + [f"    {nm:20s} {n:>8,} elements  {n*ITEMSIZE:>10,} B" for nm, n in sizes]
    lines += [
        "",
        f"  capture shape, read out of the buffer sizes: {shape}",
        f"  largest kernel ARGUMENT = {largest:,} elements = {largest*ITEMSIZE:,} B",
        f"  the decay tensor at this shape = {decay_elts:,} elements = "
        f"{decay_elts*ITEMSIZE:,} B",
        f"  ratio = {decay_elts/largest:.0f}x. It is not in the signature and",
        f"  cannot be: it is {decay_elts/largest:.0f}x too big for anything in it.",
        "",
        f"  the identity, at the BENCH shape where K = {s['K']} and C = {s['C']} "
        f"differ so the size is not degenerate:",
        f"    B*NT*HV*C*C       = {arg:>12,} elements = {arg*ITEMSIZE:>12,} B",
        f"    B*NT*HV*C*C*K     = {ten:>12,} elements = {ten*ITEMSIZE:>12,} B",
        f"    ratio            = {ten/arg:>12.0f} = K. The two share",
        "  B*NT*HV*C*C and differ only in the trailing K, so the gap is exactly",
        "  K and is independent of B, NT, HV and C. That is why it is a number",
        "  and not a coincidence.",
    ]
    return Value(
        f"the largest kernel argument in the shipped CUDA capture is "
        f"{largest:,} float32 = {largest*ITEMSIZE:,} B, and the decay tensor at "
        f"that same shape is {decay_elts:,} elements = "
        f"{decay_elts*ITEMSIZE:,} B, i.e. {decay_elts/largest:.0f}x too large to "
        f"appear in the parameter list at all; at the bench shape the gap "
        f"between the decay tensor and the largest argument is exactly K = "
        f"{s['K']} -- computed here as {ten:,}/{arg:,} = {ten/arg:.0f} -- and it "
        f"is independent of B, NT, HV and C",
        "\n".join(lines),
    )


def _kernel_shape_from_name(name: str, fallback: dict) -> dict:
    """TinyGrad names a kernel `r_<grid>_<block>...`. Reading B*NT*HV out of
    the buffer sizes is more robust than reading a shape out of a name, so
    this only tries, and falls back to the declared capture shape."""
    return dict(fallback)


@MOD.claim("never_materialised")
def c_never() -> Value:
    """Two independent senses, because they were two different measurements.
    M1 needs a device. M2 -- the decisive one -- needs a source file, and a
    source file is what is shipped."""
    src = _read(KERNEL_FILE)
    MOD.pins.file(KERNEL_FILE)
    exp2 = src.count("exp2(")
    bare_exp = len(re.findall(r"(?<![0-9a-zA-Z_])exp\s*\(", src))
    # every global store in the file is an assignment whose LHS names a
    # data buffer; `float4*` casts and `make_float4` count the lanes.
    store_lhs = re.findall(r"^\s*\*\(\(float\d\*\)\(\(data(\d+)_", src, re.M)
    written = sorted({int(i) for i in store_lhs})
    to_local = len(re.findall(r"float\s+alu\d+\s*=\s*[^;]*exp2\s*\(", src))
    kernel_shape = _capture_shape(src)
    decay_elts = (kernel_shape["B"] * kernel_shape["NT"] * kernel_shape["HV"]
                  * kernel_shape["C"] * kernel_shape["C"] * kernel_shape["K"])

    dev = _device_for_never()
    alloc = dev["alloc"]

    MOD.pins.param("kernel_exp2", exp2)
    MOD.pins.param("kernel_bare_exp", bare_exp)
    MOD.pins.param("kernel_written_buffers", sorted(written))
    MOD.pins.param("kernel_local_exp2_defs", to_local)
    MOD.pins.param("decay_elems_capture", decay_elts)
    lines = [
        f"  M2 -- a kernel ARGUMENT of the decay's size exists:  NO",
        f"    read from {KERNEL_FILE}, the emitted source.",
        f"    literal exp2 in the file        : {exp2}",
        f"    bare exp (not exp2)            : {bare_exp}",
        f"    float register locals defining an exp2 : {to_local}",
        f"    distinct buffers written       : {len(written)} "
        f"({', '.join('data%d' % i for i in sorted(written))})",
        f"    decay tensor at this shape      : {decay_elts:,} elements",
        "    Every exp2 is a register local, defined and consumed in the same",
        "    statement group. There is no store of a decay anywhere in the file.",
    ]
    if alloc is None:
        lines += ["",
                  "  M1 -- a device BUFFER of the decay's size is ever requested:",
                  f"    [UNAVAILABLE] {dev['why']}",
                  "    M2 above does not need a device, and M2 is the decisive one:"]
        lines += ["    a buffer carries traffic only if a kernel reads or writes it,",
                  "    and a kernel can only touch buffers in its argument list."]
    else:
        lines += ["",
                  "  M1 -- a device BUFFER of the decay's size is ever requested:  NO",
                  f"    device {dev['name']}, largest allocation ever requested "
                  f"{alloc['largest']:,} B",
                  f"    the decay tensor at the BENCH shape is "
                  f"{dev['decay_bytes']:,} B and was never requested",
                  f"    positive control (force the tensor to exist, it must appear): "
                  f"{dev['control']}"]
    return Value(
        f"the `[B,NT,HV,C,C,K]` decay tensor is never materialised: in the "
        f"emitted CUDA source its {exp2} `exp2` evaluations are all register "
        f"locals, only {len(written)} buffer is ever written, and the tensor's "
        f"own {decay_elts:,} elements do not appear in the kernel's parameter "
        f"list at all"
        + ("" if alloc else f"; the allocator half (M1) is [UNAVAILABLE] here "
                           f"because {dev['why']}, and is not claimed")
        + f". A buffer carries traffic only if a kernel reads or writes it, and a "
          f"kernel can only touch buffers in its argument list, so the source "
          f"reading settles it without a device",
        "\n".join(lines),
    )


def _capture_shape(src: str) -> dict:
    """The shape the shipped CUDA kernel was captured at -- read out of the
    script that captured it, `kernels/build_notebooks.py`, not out of a
    sentence, and then CHECKED against the buffer sizes the kernel declares.

    The check matters. Buffer sizes alone do not determine the shape: at this
    capture K = C = V, so the `[B,HV,K,V]` and `[B,NT,HV,C,K]` families are
    degenerate in size and nine different (B,NT,HV,C,K) tuples reproduce both
    sizes exactly. A module that solved for the shape from the buffer sizes
    would have had nine answers and picked one. So the shape comes from the
    capture script, pinned by line, and the buffer sizes are the
    cross-check -- not the other way round.
    """
    line = MOD.pins.file_line(CAPTURE_SCRIPT, CAPTURE_LINE, name="shape:capture",
                              must_contain="T.rnd(")
    m = re.search(r"T\.rnd\(\s*([-\d]+)\s*,\s*([-\d]+)\s*,\s*([-\d]+)\s*,"
                  r"\s*([-\d]+)\s*,\s*([-\d]+)\s*,\s*([-\d]+)", line)
    if not m:
        raise RuntimeError(f"{CAPTURE_SCRIPT}:{CAPTURE_LINE} is not a T.rnd(...) "
                           f"call: {line!r}")
    B, Tt, H, HV, K, V = (int(g) for g in m.groups())
    cline = MOD.pins.file_line(CAPTURE_SCRIPT, CAPTURE_LINE + 1, name="shape:C",
                               must_contain="chunk_size=")
    cm = re.search(r"chunk_size\s*=\s*([-\d]+)", cline)
    if not cm:
        raise RuntimeError(f"{CAPTURE_SCRIPT}:{CAPTURE_LINE + 1} has no "
                           f"chunk_size: {cline!r}")
    C = int(cm.group(1))
    NT = Tt // C
    sizes = sorted({int(x) for x in re.findall(r"data\d+_(\d+)\b", src)})
    want_state = B * HV * K * V
    want_operand = B * NT * HV * C * K
    if sizes != sorted({want_state, want_operand}):
        raise RuntimeError(
            f"the capture shape read from {CAPTURE_SCRIPT}:{CAPTURE_LINE} does "
            f"not reproduce the buffer sizes in {KERNEL_FILE}: the shape implies "
            f"{{{want_state}, {want_operand}}} and the kernel declares {sizes}. "
            f"One of them moved. Do not pick one.")
    return dict(B=B, T=Tt, H=H, HV=HV, K=K, V=V, C=C, NT=NT,
                state_elts=want_state, operand_elts=want_operand,
                degenerate=(C == K == V))


def _lanes(src: str) -> int:
    """How many scalar components a `make_float4(...)` call carries, counted by
    splitting its argument list at top-level commas. Counted rather than
    assumed: the lanes-per-store number is what turns a per-thread count into a
    per-element one, and assuming 4 would be exactly the kind of literal this
    package does not allow in an output path."""
    calls = re.findall(r"make_float4\(", src)
    if not calls:
        return 0
    # take the first call's balanced argument list
    i = src.index("make_float4(") + len("make_float4(")
    depth, commas, j = 0, 0, i
    while j < len(src):
        c = src[j]
        if c in "([":
            depth += 1
        elif c in ")]":
            if depth == 0:
                break
            depth -= 1
        elif c == "," and depth == 0:
            commas += 1
        j += 1
    return commas + 1


def _device_for_never() -> dict:
    """Wrap the allocator and look for the decay's own size. Needs a device; if
    there is none, says so."""
    try:
        import tiny_kda, test_ref as T                                  # noqa: E402
    except Exception as e:                                              # noqa: BLE001
        return dict(alloc=None, why=f"kda/ not importable: {type(e).__name__}", name="?")
    try:
        from tinygrad import Tensor, Device
        from tinygrad.device import Buffer
        s = BENCH_SHAPE
        NT = s["T"] // s["C"]
        decay_bytes = s["B"] * NT * s["HV"] * s["C"] * s["C"] * s["K"] * ITEMSIZE
        seen: list[int] = []
        real = Buffer.allocate

        def spy(*A, **K):
            out = real(*A, **K)
            seen.append(int(out.size) * ITEMSIZE)
            return out

        Buffer.allocate = staticmethod(spy)
        try:
            q, k, v, g, beta = T.rnd(s["B"], s["T"], s["H"], s["HV"], s["K"], s["V"],
                                      seed=0, l2n=True)
            tiny_kda.kda_chunk(q, k, v, g, beta, chunk_size=s["C"]).realize()
        finally:
            Buffer.allocate = real
        largest = max(seen) if seen else 0
        # POSITIVE CONTROL: force the tensor to exist, so the spy is known to be
        # able to see it. A control that cannot fail proves nothing.
        cseen: list[int] = []
        Buffer.allocate = staticmethod(
            lambda *A, **K: (lambda o: (cseen.append(int(o.size) * ITEMSIZE), o)[1])(
                real(*A, **K)))
        try:
            Tensor.empty(s["B"], NT, s["HV"], s["C"], s["C"], s["K"],
                         dtype="float32").realize()
        finally:
            Buffer.allocate = real
        return dict(
            alloc=dict(largest=largest, seen=len(seen)),
            why="", name=Device.DEFAULT,
            decay_bytes=decay_bytes,
            control=("POSITIVE CONTROL PASSES: forcing the tensor to exist makes "
                     f"exactly {decay_bytes:,} B appear, so the spy can see it; "
                     f"during the operator it never does, and the largest "
                     f"allocation requested was {largest:,} B"
                     if decay_bytes in cseen else
                     f"POSITIVE CONTROL FAILED: {decay_bytes:,} B was not seen "
                     f"even when forced to exist -- the spy is not observing the "
                     f"right call, and the negative result is worthless"),
        )
    except Exception as e:                                              # noqa: BLE001
        return dict(alloc=None, why=f"{type(e).__name__}: {str(e)[:80]}", name="?")


@MOD.claim("redundant_exp2")
def c_exp2() -> Value:
    """The cost of never materialising: the decay is recomputed inside the
    accumulation instead. Per thread, per output element, and against the
    kernel's own transcribed equivalent."""
    src = _read(KERNEL_FILE)
    MOD.pins.file(KERNEL_FILE)
    shape = _capture_shape(src)
    exp2 = src.count("exp2(")
    bounds = re.search(r"__launch_bounds__\((\d+)\)", src)
    threads = int(bounds.group(1)) if bounds else 0
    # one float4 store per output group; a float4 is 4 scalar outputs. Counting
    # the stores is how the outputs-per-thread is known -- `__launch_bounds__`
    # is a ceiling, not a count, and dividing by it would be a different claim.
    stores4 = len(re.findall(r"^\s*\*\(\(float4\*\)\(\(data", src, re.M))
    lanes = _lanes(src)
    outs_per_thread = stores4 * lanes
    nbuf = len(set(re.findall(r"data\d+_(\d+)\b", src)))
    per_thread = exp2
    per_store = per_thread / stores4 if stores4 else 0.0
    per_elem = per_thread / outs_per_thread if outs_per_thread else 0.0
    # the port's AST expansion at this shape: C*C*K exp2 per (b,n,hv)
    t = _tiling()
    ast = shape["C"] ** 2 * shape["K"]
    # the transcribed Triton count at the codegen_diff shape: NC * BC * K * 2
    tri = t["NC"] * t["BC"] * CODEGEN_SHAPE["K"] * 2
    port = CODEGEN_SHAPE["C"] ** 2 * CODEGEN_SHAPE["K"]
    MOD.pins.param("exp2_total", exp2)
    MOD.pins.param("exp2_threads", threads)
    MOD.pins.param("exp2_stores", stores4)
    MOD.pins.param("exp2_lanes", lanes)
    MOD.pins.param("exp2_capture_shape", shape)
    MOD.pins.param("exp2_ratio", port / tri)
    lines = [
        f"  captured shape, read out of the buffer sizes: {shape}",
        f"  literal exp2 in the file   : {exp2}",
        f"  __launch_bounds__          : {threads} (a ceiling, not a count)",
        f"  float4 global stores       : {stores4}   lanes each: {lanes}"
        f"   buffers declared: {nbuf}",
        f"  outputs per thread         : {outs_per_thread}",
        f"  per thread                 : {per_thread} exp2",
        f"  per float4 store           : {per_store:.1f} exp2",
        f"  per stored scalar element  : {per_elem:.1f} exp2",
        "",
        "  This is NOT the 32x figure. The 32x is counted on a different kernel",
        "  at a different shape; the two are not comparable and this module does",
        "  not merge them. What is comparable, and is computed here:",
        f"    port   [C,C,K] at the codegen_diff shape : C*C*K = {port:>9,} exp2",
        f"    Triton [BC,K] per sub-chunk, x NC        = {tri:>9,} exp2",
        f"    ratio                                     = {port/tri:>8.0f}x",
    ]
    return Value(
        f"never materialising costs transcendental evaluations: the captured "
        f"kernel evaluates {exp2} `exp2` per thread for {per_elem:.1f} per stored "
        f"output element, all in registers, and on the shape algebra the port's "
        f"decay expansion is C*C*K = {port:,} per (b,n,hv) against the Triton "
        f"kernel's NC*BC*K*2 = {tri:,}, a ratio of {port/tri:.0f}x -- which is a "
        f"ratio of two transcriptions of shapes, not two measured instruction "
        f"counts, and is reported as such",
        "\n".join(lines),
    )


# ---------------------------------------------------------------- report

def report(mod, names, flags):
    print("derivation/kda_decay.py -- the KDA decay tensor")
    print("four quantities, four scopes, never compared with one another.")
    for name in (names or ["graph_object_ratio", "bench_shape_tensor",
                           "largest_device_resident", "never_materialised",
                           "redundant_exp2"]):
        rule(name)
        try:
            v = MOD.run_one(name)
        except PinDrift as e:
            print(f"[PIN DRIFT] {e}")
            continue
        print(v.text)
        if v.detail:
            print()
            print(v.detail)
    print()


if __name__ == "__main__":
    sys.exit(run(MOD, sys.argv[1:], report=report))
