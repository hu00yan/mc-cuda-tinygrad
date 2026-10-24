"""common.py - shapes, artifact pinning, and the status vocabulary.

Every experiment in this directory re-derives its numbers. Nothing is
transcribed from a notes file, from a brief, or from another agent's output.
Where an experiment *must* read a captured artifact (a generated CUDA file, a
PTX file, a GPU log) that artifact is pinned by sha256 here, and a hash
mismatch is a hard error, not a warning: a changed artifact means the number
this directory would print is stale, and a stale number printed confidently is
worse than a crash.

THE RULES THIS ENFORCES
------------------------
1.  Re-derive. Every figure in `notes/15-experiment-b.md` is printed by a
    script here. If you cannot regenerate it, it does not belong in the note.
2.  Label the regime. Five tags, used identically in the code and the prose:
      MEASURED   an instrument ran here, on this host, on this artifact
      RE-DERIVED arithmetic recomputed here from shapes / from source text
      MODELLED   arithmetic on shapes, with the premise stated
      INFERRED   a conclusion that needs hardware this host does not have
      ARTIFACT   read out of a pinned file, quoted with its path and sha256
3.  Fail loudly. `require()` raises ArtifactDrift rather than proceeding.
4.  Say where a device is needed. `device_status()` reports what this host can
    and cannot do, from probes, not from assumption.

NO GPU IS REQUIRED BY ANY SCRIPT IN THIS DIRECTORY. That is the point of the
whole exercise, and `run_all.py` verifies it by refusing to run if a CUDA
device happens to be present, so that a green run on a GPU box means the same
thing as a green run here.
"""

from __future__ import annotations

import collections
import hashlib
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KDA = os.path.join(REPO, "kda")

# ---------------------------------------------------------------- status tags

MEASURED = "MEASURED"
RE_DERIVED = "RE-DERIVED"
MODELLED = "MODELLED"
INFERRED = "INFERRED"
ARTIFACT = "ARTIFACT"
NOT_HERE = "NOT-HERE"


# --------------------------------------------------------------------- shapes
# Named once, so no experiment below can invent one. Each cites the file it was
# read out of. `build_notebooks.py:590` is the shape the single captured CUDA
# kernel was rendered at, and it is the ONLY shape for which emitted source
# exists in this repo; every other shape here is a model or a schedule census.

# kda/perf_kda.py:90  -- the shape both the GPU log and every contested byte
# figure are quoted at. NT = T//C = 8.
BENCH = dict(B=2, T=512, H=4, HV=4, K=64, V=64, C=64)
BENCH_TAG = "kda/perf_kda.py:90"

# kernels/build_notebooks.py:590 -- the shape the one captured kernel came from.
CAPTURE = dict(B=2, T=128, H=2, HV=2, K=32, V=32, C=32)
CAPTURE_TAG = "kernels/build_notebooks.py:590"

# kda/codegen_diff.py -- the isolated gram shape the C^2/BC element ratio is
# quoted at. B=1, H=HV=1, K=V=64, C=64, NT=1.
GRAM = dict(B=1, T=64, H=1, HV=1, K=64, V=64, C=64)
GRAM_TAG = "kda/codegen_diff.py"

# fla's [BC, K] decay operand block size. Read out of the vendored source, not
# out of a notes file. See b5_arch_gate.py -- it greps the real constant.
FLA_BC_FALLBACK = 16


def nt(s: dict) -> int:
    return s["T"] // s["C"]


def slabs(s: dict) -> int:
    return s["B"] * nt(s) * s["HV"]


def decay_elems(s: dict) -> int:
    """elements of ONE [B,NT,HV,C,C,K] decay tensor"""
    return slabs(s) * s["C"] * s["C"] * s["K"]


def decay_bytes(s: dict, es: int = 4) -> int:
    """fp32 by construction: kda/tiny_kda.py:44 sets F = dtypes.float32 and
    nothing casts the decay away."""
    return decay_elems(s) * es


def cc_bytes(s: dict, es: int = 4) -> int:
    """the [B,NT,HV,C,C] family"""
    return slabs(s) * s["C"] * s["C"] * es


def shape_str(s: dict) -> str:
    return (f"B={s['B']} T={s['T']} H={s['H']} HV={s['HV']} K={s['K']} "
            f"V={s['V']} C={s['C']}  NT={nt(s)}  B*NT*HV={slabs(s):,}")


# ------------------------------------------------------------------ the bytes
# The one number the whole study turns on: the decay tensor at the bench shape
# in fp32. It is 2^26 exactly, which is why the record keeps writing it as
# "67,108,864" and why the census below is looking for that bucket.
DECAY_BUCKET = 1 << 26
assert decay_bytes(BENCH) == DECAY_BUCKET, decay_bytes(BENCH)


# ------------------------------------------------------------------- printing

def rule(c: str = "=", n: int = 96) -> None:
    print(c * n)


def head(title: str) -> None:
    print()
    print(title)
    rule("-")


def kv(k, v, width: int = 34) -> None:
    print(f"  {str(k):<{width}} {v}")


def verdict(text: str) -> None:
    print()
    rule("=")
    print(f"  {text}")
    rule("=")
    print()


# ----------------------------------------------------------------- artifacts

class ArtifactDrift(RuntimeError):
    """A pinned artifact changed. Every number derived from it is now stale."""


# path -> sha256, taken on the machine that wrote the note. Anything read by a
# script here appears below. If you add a dependency, add its hash, do not skip
# it: an unpinned read is exactly the failure mode this study keeps hitting.
PINS: dict[str, str] = {
    # the ONE emitted CUDA kernel in the repo, from the real NVIDIA backend
    "out/kaggle/20260928-014248-gpu/codegen/kernel000.txt":
        "4de9665cc74474365dd2d04b750418e06dab0ec3f34d4f6545110f112580bc5e",
    # the hand-written reference's PTX, sm_90a and sm_100a, no GPU needed
    "out/archlab/tri/chunk_gated_delta_rule_fwd_kernel_h_blockdim64.ptx":
        "f70614679de96f69144247871030aea735b7280d5323de3492e2c29ac3069587",
    "out/archlab/tri100/chunk_gated_delta_rule_fwd_kernel_h_blockdim64.ptx":
        "5632fe2aa9ebd41f589547496448292c9669d8326e44591df8b45a7c020abe75",
    # its own frontend IR, to show the arch adaptation is decided upstream
    "out/archlab/tri/chunk_gated_delta_rule_fwd_kernel_h_blockdim64.ttir":
        "34e1f4f6e2e68025bd92a3c609ce92a7c1e22614e1a43e3db925543aeca3fa99",
    # the T4 run: the only GPU timings in the repo
    "out/kaggle/20260928-014248-gpu/logs/perf_kda.log":
        "600c6dc5b0eb5bac86e345d9d89ad747e8d5cfa161bad6488add66d130ce0495",
    "out/kaggle/20260928-014248-gpu/logs/replay_kda.log":
        "acc1df272c5dd70a5471a667e7d19ce3ffeab6bbd2a486f8bf0b7f6f6491f464",
    # ... and the run's own manifest, which is what establishes the device was
    # a T4 and therefore that the log's hard-coded "Metal" label is wrong
    "out/kaggle/20260928-014248-gpu/SUMMARY.md":
        "34ee572131e34d94acbd1220ab244ce893ea7ef6401e7839da6bc9b8b25253f3",
    # the port under test
    "kda/tiny_kda.py":
        "3d063c44403485a8f4f3d92eef149b8fedb882cf567c3e977ea84ae0ebdb5c6d",
    "kda/perf_kda.py":
        "bf505ca0bcb4dedabc0afe3ac1319532fa1af1f29a7b62a22e9621bdadf9aa3b",
    "kda/test_ref.py":
        "baafff1926910fbf0a4b0146d3e415b622f323dbab1c65871595e004639548d7",
    # the vendored hand-written baseline, read for its arch gates
    "portability/subj/fla/flash-linear-attention-main/fla/utils/_device.py":
        "1be4c26780c64a02aee4bd7ec1701595acdbd9d4831908402880f3e40e1b0993",
    "portability/subj/fla/flash-linear-attention-main/fla/ops/common/chunk_delta_h.py":
        "92bfbdad0021ebcf04dd811657c4426be93a5840ec40645d4d69d4b5dce8cae0",
    # the installed compiler, read for what it can and cannot emit
    "tinygrad/renderer/cstyle.py":
        "280382aa501da70938f54d19bf5295f9b354b9fbbf3b9aa152b6521ecb70ca82",
    "tinygrad/codegen/opt/tc.py":
        "d4cb33e06877905db9c753938ea9fd4d41e6737351b7074f1e3fc8573a7f3d80",
    "tinygrad/runtime/ops_cuda.py":
        "e3d22944295b54c105c5a531562b04b23473b92acf160028ff05bf86df9ebdc3",
}

# where the three tinygrad pins live. Resolved at import so a venv move does
# not silently unpin them.
_TINYGRAD_ROOTS = [
    os.path.join(REPO, ".venv", "lib"),
    os.path.join(sys.prefix, "lib"),
]


def _resolve(rel: str) -> str | None:
    cand = os.path.join(REPO, rel)
    if os.path.exists(cand):
        return cand
    if rel.startswith("tinygrad/"):
        for root in _TINYGRAD_ROOTS:
            for dirpath, dirnames, _ in os.walk(root):
                if "tinygrad" in dirnames:
                    p = os.path.join(dirpath, rel)
                    if os.path.exists(p):
                        return p
                    break
    return None


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(rel: str, *, quiet: bool = False) -> str:
    """Return the absolute path of a pinned artifact, or raise.

    Raises ArtifactDrift if the file's sha256 no longer matches PINS. A changed
    artifact means the number would be stale, and this directory exists so that
    a stale number cannot be printed.
    """
    p = _resolve(rel)
    if p is None:
        raise FileNotFoundError(
            f"pinned artifact missing: {rel}\n"
            f"  searched under {REPO}\n"
            f"  every number derived from it is now UNKNOWN, not zero.")
    want = PINS.get(rel)
    if want is None:
        raise KeyError(f"{rel} is read by this directory but not pinned in common.PINS")
    got = sha256(p)
    if got != want:
        raise ArtifactDrift(
            f"\nARTIFACT DRIFT -- refusing to print a stale number\n"
            f"  file : {p}\n"
            f"  pinned sha256 {want}\n"
            f"  actual sha256 {got}\n"
            f"  The artifact this experiment was written against has changed.\n"
            f"  Re-derive the affected numbers against the new artifact, then\n"
            f"  re-pin deliberately with:  python -m experiment_b.repin {rel}\n")
    if not quiet:
        pass
    return p


def read(rel: str) -> str:
    with open(require(rel)) as f:
        return f.read()


def pin_report() -> list[tuple[str, str, str]]:
    """(rel, pinned, status) for every pin. status in {OK, MISSING, DRIFT}."""
    out = []
    for rel, want in sorted(PINS.items()):
        p = _resolve(rel)
        if p is None:
            out.append((rel, want, "MISSING"))
            continue
        got = sha256(p)
        out.append((rel, want, "OK" if got == want else f"DRIFT {got}"))
    return out


# ------------------------------------------------------------- device status

def device_status() -> dict:
    """What this host can actually do, probed rather than assumed.

    The point of comparison (b) is that none of it needs a device. This
    function is what makes that checkable instead of asserted: it looks for a
    CUDA driver, a CUDA toolkit, and a GPU, and reports what it found.
    """
    out: dict = {}
    import shutil
    out["nvcc"] = shutil.which("nvcc")
    out["ncu"] = shutil.which("ncu")
    out["cuobjdump"] = shutil.which("cuobjdump")
    out["ptxas"] = shutil.which("ptxas")
    libcuda = any(os.path.exists(os.path.join(p, "libcuda.so"))
                  for p in ("/usr/lib", "/usr/local/cuda/lib64",
                            "/usr/lib/x86_64-linux-gnu"))
    out["libcuda"] = libcuda
    out["gpu_present"] = bool(libcuda or shutil.which("nvidia-smi"))
    return out


def require_no_gpu() -> None:
    """Refuse to run on a machine that has a GPU.

    Not superstition. If a CUDA device is discoverable, tinygrad's own
    `Device.DEFAULT` could silently become "CUDA" somewhere, the schedule
    census would start meaning something slightly different, and a green run
    here would no longer be evidence that the question is device-free. Pass
    --allow-gpu to override, deliberately.
    """
    st = device_status()
    if st["gpu_present"]:
        raise RuntimeError(
            "a GPU is visible on this host. experiment_b is device-free by "
            "construction and its verdicts are only meaningful when nothing "
            "can execute a kernel. Re-run with --allow-gpu if you know what "
            "you are doing.")


# ------------------------------------------------------- the schedule census
# The census needs no device. Two facts make that true, both re-derived in
# b2_schedule.py rather than asserted here:
#
#   * `Tensor.schedule_linear()` -> `create_linear_with_vars` ->
#     `memory_plan_rewrite` (tinygrad/schedule/memory.py) is a pure UOp
#     rewrite driven by DEVICE STRINGS. It allocates nothing and runs nothing.
#   * `Context(DEV="CUDA")` makes `Device.DEFAULT` read literally "CUDA"
#     (`Device.DEFAULT` is `DEV.device or self._select_device`), so device
#     discovery is short-circuited and no `CUDADevice` is ever constructed.
#
# The census therefore asks the real CUDA scheduler a real question and
# executes none of the answer.

def import_kda():
    """Import the port under test with the repo's kda/ on sys.path."""
    if KDA not in sys.path:
        sys.path.insert(0, KDA)
    import numpy as np  # noqa: F401
    import test_ref  # noqa: F401
    import tiny_kda  # noqa: F401
    return test_ref, tiny_kda


def render_kda_sources(arch: str = "sm_90") -> list[str]:
    """Render every distinct CUDA C source tinygrad emits for the KDA operator
    at the CAPTURE shape, for a given architecture, with NO DEVICE.

    Two tinygrad affordances make this work:
      * `DEV=NULL:CUDA:<arch>` -- the NULL device initialises with no hardware
        and accepts an alternative renderer, so the CUDA *renderer* is attached
        to a device that cannot run anything. <arch> selects the tensor-core
        and dtype table.
      * `Compiler.compile_cached` is the single choke point every kernel passes
        through. Replacing it captures the source string and pretends it
        compiled.

    MUST BE CALLED IN A FRESH PROCESS, one arch at a time. `DEV` is read once,
    at first device construction, and tinygrad memoises compiled programs, so
    a second arch in the same process silently gets the first arch's kernels
    and an EMPTY source list. `run_render(archs)` does the forking.
    """
    os.environ["DEV"] = f"NULL:CUDA:{arch}"
    # tinygrad's beam search and TC autotuner run candidate compiles in a
    # spawn-context multiprocessing pool; the workers re-import tinygrad, never
    # see our patch, die constructing CUDARenderer (no NVRTC) and hang the pool
    # forever. PARALLEL=0 makes get_worker_pool() return None.
    os.environ["PARALLEL"] = "0"
    # and the in-memory + on-disk program cache would make a re-render a no-op
    os.environ["CACHELEVEL"] = "0"
    from tinygrad.renderer import cstyle
    from tinygrad.device import Compiler
    orig = cstyle.CUDARenderer.__init__
    cstyle.CUDARenderer.__init__ = (
        lambda self, target, use_nvcc=False: orig(self, target, use_nvcc=True))
    srcs: list[str] = []
    Compiler.compile_cached = lambda self, src: (srcs.append(src), b"")[1]
    from tinygrad import Tensor
    import numpy as np
    Tensor.numpy = (lambda self, *a, **k:
                    (self.realize(), np.zeros((), dtype=np.float32))[1])
    test_ref, tiny_kda = import_kda()
    s = CAPTURE
    q, k, v, g, beta = test_ref.rnd(s["B"], s["T"], s["H"], s["HV"],
                                    s["K"], s["V"], seed=0, l2n=True)
    tiny_kda.kda_chunk(q, k, v, g, beta, chunk_size=s["C"])
    return srcs


def render_worker(arch: str) -> str:
    """Print one arch's rendered program as a JSON line, for `run_render`."""
    import json
    srcs = render_kda_sources(arch)
    return json.dumps(dict(arch=arch, n=len(srcs), srcs=srcs))


def run_render(archs: list[str]) -> dict:
    """One fresh process per architecture. Returns {arch: (srcs, blob, sha256)}.

    Isolation is not tidiness. `DEV` is latched at first device construction
    and tinygrad memoises compiled programs, so rendering two archs in one
    process yields the first arch's kernels twice and an empty list the second
    time. That failure is silent and looks like a real result: an empty program
    has zero wgmma AND zero mma.sync, so a class census would report the same
    all-zeros row for every arch and the arch-invariance claim would appear to
    hold for the wrong reason. Hence the hard failures below.
    """
    import hashlib
    import json
    import subprocess
    me = os.path.abspath(__file__)
    out = {}
    for arch in archs:
        p = subprocess.run([sys.executable, me, "--render-one", arch],
                           capture_output=True, text=True,
                           cwd=os.path.dirname(os.path.dirname(me)))
        if p.returncode != 0 or not p.stdout.strip():
            raise RuntimeError(
                f"render worker for {arch} failed (rc={p.returncode}).\n"
                f"{p.stderr[-2000:]}\n"
                f"An empty program here would silently satisfy 'arch-invariant',\n"
                f"so this is fatal rather than a warning.")
        d = json.loads(p.stdout.strip().splitlines()[-1])
        if not d["srcs"]:
            raise RuntimeError(
                f"render worker for {arch} produced ZERO sources. That is the "
                f"silent-failure mode this function exists to prevent: an empty "
                f"program has zero of every instruction class and would make the "
                f"arch-invariance result appear true for the wrong reason.")
        blob = "".join(d["srcs"])
        out[arch] = (d["srcs"], blob, hashlib.sha256(blob.encode()).hexdigest())
    return out


if __name__ == "__main__":  # the render worker, run as a fresh process
    if len(sys.argv) == 3 and sys.argv[1] == "--render-one":
        print(render_worker(sys.argv[2]))
    else:
        raise SystemExit("usage: python3 common.py --render-one sm_90")


# ---------------------------------------------- instruction classes (matrix)
# One vocabulary, used for both sides of the comparison so the two rows of the
# instruction-class matrix are produced by the same classifier. Ordered, so the
# printed matrix is stable.
CLASSES = [
    "wgmma.mma_async",   # Hopper warpgroup MMA
    "wgmma.wait_group",  # Hopper warpgroup MMA completion
    "wgmma.commit_group",
    "wgmma.fence",
    "tcgen05.mma",       # Blackwell 5th-gen tensor core
    "tcgen05.ld",        # Blackwell tcgen05 loads (the "TMA-in-a-register" path)
    "tcgen05.wait",
    "mma.sync",          # Ampere-and-later warp MMA. TinyGrad's entire ceiling.
    "ldmatrix",
    "stmatrix",
    "cp.async",
    "cp.async.commit_group",
    "mbarrier",
    "tensormap",
    "ex2.approx",
    "ld.global.b16",
    "ld.global.b32",
    "st.global.b32",
    "ld.global.v4",
    "st.global.v4",
    "ld.local",
    "st.local",
    "__shared__",
    "__syncthreads",
]

_PTX_RULES = [
    ("wgmma.mma_async", r"\bwgmma\.mma_async"),
    ("wgmma.wait_group", r"\bwgmma\.wait_group"),
    ("wgmma.commit_group", r"\bwgmma\.commit_group"),
    ("wgmma.fence", r"\bwgmma\.fence"),
    ("tcgen05.mma", r"\btcgen05\.mma"),
    ("tcgen05.ld", r"\btcgen05\.ld"),
    ("tcgen05.wait", r"\btcgen05\.wait"),
    ("mma.sync", r"\bmma\.sync"),
    ("ldmatrix", r"\bldmatrix"),
    ("stmatrix", r"\bstmatrix"),
    ("cp.async.commit_group", r"\bcp\.async\.commit_group"),
    ("cp.async", r"\bcp\.async(?!\.commit_group)"),
    ("mbarrier", r"\bmbarrier\."),
    ("tensormap", r"\btensormap"),
    ("ex2.approx", r"\bex2\.approx"),
    ("ld.global.b16", r"\bld\.global\.b16"),
    ("ld.global.b32", r"\bld\.global\.b32"),
    ("ld.global.v4", r"\bld\.global\.v4"),
    ("st.global.b32", r"\bst\.global\.b32"),
    ("st.global.v4", r"\bst\.global\.v4"),
    ("ld.local", r"\bld\.local"),
    ("st.local", r"\bst\.local"),
]

# CUDA C source uses __-prefixed spellings and inline-asm strings, not PTX
# mnemonics. Same classes, source-level spellings.
_CU_RULES = [
    ("wgmma.mma_async", r"\bwgmma\."),
    ("wgmma.wait_group", r"\bwgmma\.wait_group"),
    ("wgmma.commit_group", r"\bwgmma\.commit_group"),
    ("wgmma.fence", r"\bwgmma\.fence"),
    ("tcgen05.mma", r"\btcgen05\.mma"),
    ("tcgen05.ld", r"\btcgen05\.ld"),
    ("tcgen05.wait", r"\btcgen05\.wait"),
    ("mma.sync", r"\bmma\.sync"),
    ("ldmatrix", r"\bldmatrix"),
    ("stmatrix", r"\bstmatrix"),
    ("cp.async.commit_group", r"\bcp\.async\.commit_group"),
    ("cp.async", r"\bcp\.async(?!\.commit_group)"),
    ("mbarrier", r"\bmbarrier\."),
    ("tensormap", r"\btensormap"),
    ("ex2.approx", r"\bexp2\("),        # C spelling of ex2.approx
    ("__shared__", r"__shared__"),
    ("__syncthreads", r"__syncthreads"),
]


def census_classes(text: str, dialect: str) -> collections.Counter:
    """Count instruction classes in PTX or in generated CUDA C.

    'ptx' counts mnemonics; 'cu' counts the source-level spellings. The class
    NAMES are the same, so the two sides of the matrix are comparable; the
    counts are NOT directly comparable in general, because ptxas has not run on
    the CUDA C and Triton has already run on the PTX. That asymmetry is stated
    at every use site and is the reason the matrix is a class matrix and not an
    instruction-count comparison.
    """
    rules = _PTX_RULES if dialect == "ptx" else _CU_RULES
    c: collections.Counter = collections.Counter()
    for name, pat in rules:
        c[name] = len(re.findall(pat, text))
    if dialect == "cu":
        c["ld.global.v4"] = len(re.findall(r"\(\*\(\(float4\*\)", text))
        c["st.global.v4"] = len(re.findall(r"=\s*make_float4\(", text))
        c["ex2.approx"] = len(re.findall(r"\bexp2\(", text))
    return c


def print_matrix(rows: list[tuple[str, collections.Counter]], note: str = "") -> None:
    names = [c for c in CLASSES if any(r[1].get(c) for r in rows)]
    if not names:
        print("  (no instruction of any class in any row)")
        return
    w = max(len(n) for n in names)
    hdr = "  " + f"{'class':<{w}}" + "".join(f"{n:>18}" for n, _ in rows)
    print(hdr)
    rule("-", len(hdr))
    for n in names:
        print("  " + f"{n:<{w}}" + "".join(f"{r[1].get(n, 0):>18,}" for r in rows))
    if note:
        import textwrap
        print()
        for para in note.strip().split("\n\n"):
            for ln in textwrap.wrap(" ".join(para.split()), width=92):
                print("  " + ln)
