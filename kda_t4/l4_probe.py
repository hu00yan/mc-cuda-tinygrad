#!/usr/bin/env python3
"""
l4_probe.py -- is the card this kernel landed on an L4, and what can its
silicon do?  Everything the study's shared-memory gate turns on, read from the
device.

WHY THIS IS A SEPARATE PROBE, AND WHY IT IS THE ONLY ONE IN THE SESSION
    Kaggle kills a cell at 1800 s and the runner captures stdout only at the end,
    so a killed probe loses everything.  This session therefore spends its
    budget on the cheapest question that everything else depends on -- *which
    card is this* -- and gets the answer in the first seconds.  The KDA ladder is
    a second session, and only if this one says the card passed.

THE ORDER IS DELIBERATE, AND IT IS A FALSIFIER ORDER
    Section 0 prints the device identity from four independent sources and
    STOPS if the card is not cc 8.9.  That is not caution about a slow card; it
    is that a run of this probe on a T4 is a *control*, not a failure, and
    conflating the two is how a T4 row would end up in a table of L4 numbers.
    Everything after section 0 is therefore conditional on section 0, and the
    JSON records which branch was taken.

WHAT IS MEASURED HERE, AND ON WHAT EVIDENCE
    * shared memory, read five independent ways, because the study's T4 number
      (65,536 B) came from readers that agreed and a sixth value (49,152 B, the
      non-optin one) that did not, and quoting either without saying which is
      how a 2x error survives.
    * compute capability, from torch, from triton's own properties dict, and
      from the `.target sm_XX` line ptxas was actually handed.
    * whether tf32 survives, read from the PTX of a dot that names no precision
      (so the answer is a property of the emitted code, not of a variable), and
      separately from the knob and the env var.
    * a real `tl.dot` compiled ON this card, censused with `kda_t4/sass_t4.py`'s
      own categoriser rather than a second parser written for this note: a
      cc=89 row is only comparable to the T4 row if it went through the same
      instrument, and "I wrote an equivalent one" is a defect waiting.
    * fla's own kernel cubin for this card -- the row that decides whether the
      reference is degraded here.

EVERY NUMBER IS WRITTEN TWICE
    Once to stdout (which the runner captures) and once, immediately, to
    `/kaggle/working/l4_probe.jsonl`.  A JSONL file appended as the run proceeds
    survives a kill; captured stdout does not.  The two are written from the same
    call, so they cannot disagree.

Run:  python kda_t4/l4_probe.py
"""
from __future__ import annotations

import ctypes
import json
import os
import re
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

# The durable record.  Appended as the run proceeds so that a kill at minute 14
# still leaves minutes 1..13 on disk.
JL = "/kaggle/working/l4_probe.jsonl" if os.path.isdir("/kaggle/working") \
    else os.path.join(OUT, "l4_probe.jsonl")

R: dict = {"probe": "l4_probe", "started": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                        time.gmtime())}
T0 = time.time()


def emit(record: dict) -> None:
    """Print and persist, in one place, so the two cannot diverge."""
    line = json.dumps(record, sort_keys=True, default=repr)
    print(line, flush=True)
    try:
        with open(JL, "a") as f:
            f.write(line + "\n")
            f.flush()
    except OSError as e:
        print(f"  !! could not append to {JL}: {e}", flush=True)


def say(*a) -> None:
    print(" ".join(str(x) for x in a), flush=True)


def hr(t: str = "") -> None:
    say("")
    say("=" * 76)
    if t:
        say(t)
        say("=" * 76)


def step(name: str) -> float:
    say("")
    say(f"### {name}  (t+{time.time() - T0:6.1f}s)")
    return time.time()


def guard(name: str, fn):
    """Run a section; a failure is recorded, never fatal.

    The failure of one reader is evidence about the reader, not a reason to lose
    the other four -- which is how the study lost a whole session once.
    """
    try:
        return fn()
    except BaseException as e:                                    # noqa: BLE001
        emit({"section": name, "ok": False, "error": f"{type(e).__name__}: {e}",
              "traceback": traceback.format_exc()[-900:]})
        say(f"  !! {name} FAILED {type(e).__name__}: {str(e)[:160]}")
        return None


# ============================================================ 0. the identity
def identity() -> dict:
    """Who am I.  Four independent sources, because one of them lied before.

    `torch.cuda.get_device_name` is a string the driver hands over;
    `get_device_capability` is a driver query; triton's properties dict is a
    separate query path; `nvidia-smi` is a different tool entirely.  They agree
    on a T4 in the recorded run, but agreement is the finding, not a formality.

    THE FOURTH CASE, which the first L4 request produced
        The server accepted `machine_shape: NvidiaL4`, recorded it, and gave
        back a machine with **no GPU on it at all**: no nvidia-smi, a `+cpu`
        torch, no libcudart, device_count 0.  So "the server recorded my shape"
        and "a GPU was attached" are two different facts, and only the second
        makes any number quotable.  This function reports the absence explicitly
        rather than letting a later section fail on a missing module and leave a
        reader to guess which of the two happened.
    """
    hr("0. IDENTITY -- read this before believing anything below")
    d: dict = {}

    guard("nvidia-smi", lambda: d.update(smi=_nvidia_smi()))
    guard("torch", lambda: d.update(torch_info=_torch_info()))
    guard("triton_props", lambda: d.update(triton_props=_triton_props()))
    guard("cuda_runtime", lambda: d.update(cuda_runtime=_cuda_runtime()))

    cap = (d.get("torch_info") or {}).get("capability")
    d["cc"] = list(cap) if cap else None
    d["cc_str"] = f"{cap[0]}.{cap[1]}" if cap else "<unknown>"
    names = {v.get("name") for v in (d.get("smi"), d.get("torch_info"),
                                     d.get("triton_props"))
             if isinstance(v, dict) and v.get("name")}
    d["names_seen"] = sorted(n for n in names if n)
    d["is_ada_89"] = (cap == (8, 9))
    # A GPU is present only if at least two independent sources say so.  One
    # source is not enough: torch can report a device it cannot initialise.
    d["gpu_present"] = bool(
        (d.get("torch_info") or {}).get("is_available")
        and ((d.get("torch_info") or {}).get("device_count") or 0) > 0)
    d["smi_present"] = bool((d.get("smi") or {}).get("name"))
    d["smi_reason"] = (d.get("smi") or {}).get("reason")

    say("")
    say(f"  nvidia-smi name        {(d.get('smi') or {}).get('name')}")
    say(f"  nvidia-smi reason      {d.get('smi_reason') or '-'}")
    say(f"  torch device name      {(d.get('torch_info') or {}).get('name')}")
    say(f"  torch build            {(d.get('torch_info') or {}).get('torch_version')}"
        f"  cuda={((d.get('torch_info') or {}).get('cuda_compiled'))}")
    say(f"  triton props name      {(d.get('triton_props') or {}).get('name')}")
    say(f"  device_count           {(d.get('torch_info') or {}).get('device_count')}")
    say(f"  compute capability     {d['cc_str']}   (want 8.9 for an L4)")
    say(f"  driver / cuda          {(d.get('cuda_runtime') or {}).get('driver_version')}"
        f" / {(d.get('cuda_runtime') or {}).get('cuda_runtime_version')}")
    say(f"  names all sources saw  {d['names_seen']}")
    say(f"  nvidia-smi driver      {(d.get('smi') or {}).get('driver_version_query')}")
    say(f"  GPU PRESENT            {d['gpu_present']}   "
        f"(nvidia-smi present: {d['smi_present']})")
    emit({"section": "identity", "ok": True, **d})

    if not d["gpu_present"]:
        say("")
        say("  VERDICT: NO GPU ON THIS MACHINE.")
        say("  The kernel ran. The server recorded the shape that was requested.")
        say("  Neither fact means a GPU was attached, and this one demonstrably")
        say("  was not: there is no nvidia-smi, torch is a CPU build, and the")
        say("  device count is zero.  A shape recorded in metadata is a REQUEST")
        say("  that was accepted, not a card that was granted.")
        say("  Every later section is skipped; there is nothing for it to read.")
        return d

    if d["is_ada_89"]:
        say("")
        say("  VERDICT: cc 8.9. This is an Ada part. The gate is worth measuring.")
    else:
        say("")
        say(f"  VERDICT: cc {d['cc_str']}, NOT 8.9.")
        say("  This run is a CONTROL for the request, not an L4 measurement, and")
        say("  every number below is labelled control. Nothing in it may be")
        say("  quoted as an L4 property.")
    return d


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return (p.stdout or "") + (p.stderr or "")


def _nvidia_smi_path() -> str | None:
    """nvidia-smi is not always on PATH; find it rather than concluding anything
    from its absence.  A CPU image genuinely has no nvidia-smi anywhere, and a
    CUDA image that merely has a long PATH was the shape of a false negative in
    an earlier round of this project."""
    for cand in ("nvidia-smi", "/usr/bin/nvidia-smi", "/usr/local/bin/nvidia-smi",
                 "/usr/local/nvidia/bin/nvidia-smi"):
        if os.path.sep in cand:
            if os.path.exists(cand):
                return cand
        else:
            import shutil
            w = shutil.which(cand)
            if w:
                return w
    return None


def _nvidia_smi() -> dict:
    exe = _nvidia_smi_path()
    if exe is None:
        return {"name": None, "reason": "nvidia-smi not found on PATH or in "
                                        "/usr/bin, /usr/local/bin, "
                                        "/usr/local/nvidia/bin",
                "searched": True}
    txt = _run([exe, "--query-gpu=name,compute_cap,memory.total",
                     "--format=csv,noheader"])
    rows = [r.strip() for r in txt.splitlines() if r.strip()]
    out = {"exe": exe, "raw": txt.strip()[:400], "n_rows": len(rows)}
    # driver_version is NOT a --query-gpu field; asking for it makes nvidia-smi
    # print an error in place of the whole line, which the first L4 attempt
    # would then have reported as the GPU's "name".  Ask separately.
    try:
        dv = _run([exe, "--query-gpu=driver_version", "--format=csv,noheader"])
        out["driver_version_query"] = dv.strip().splitlines()[0] if dv.strip() else None
    except BaseException as e:                                    # noqa: BLE001
        out["driver_version_query"] = f"<{type(e).__name__}>"
    if rows:
        f = [x.strip() for x in rows[0].split(",")]
        out["name"] = f[0] if len(f) > 0 else None
        out["compute_cap"] = f[1] if len(f) > 1 else None
        out["memory_total"] = f[2] if len(f) > 2 else None
    out["n_rows"] = len(rows)
    return out


def _torch_info() -> dict:
    import torch
    d: dict = {"torch_version": torch.__version__,
               "cuda_compiled": torch.version.cuda,
               "is_available": bool(torch.cuda.is_available()),
               "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0}
    if d["is_available"]:
        p = torch.cuda.get_device_properties(0)
        d["name"] = p.name
        d["capability"] = (int(p.major), int(p.minor))
        d["total_memory"] = int(p.total_memory)
        d["multi_processor_count"] = int(p.multi_processor_count)
    d["tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
    d["tf32_cudnn"] = bool(torch.backends.cudnn.allow_tf32)
    try:
        d["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    except BaseException as e:                                    # noqa: BLE001
        d["float32_matmul_precision"] = f"<{type(e).__name__}>"
    return d


def _triton_props() -> dict:
    import triton
    props = triton.runtime.driver.active.utils.get_device_properties(0)
    d = {"keys": sorted(props.keys())}
    for k in ("name", "max_shared_mem", "multiprocessor_count", "warpSize"):
        if k in props:
            d[k] = props[k]
    d["cc"] = getattr(props, "gcnArchName", None) or d.get("cc")
    return d


def _cuda_runtime() -> dict:
    d: dict = {}
    lib = None
    for cand in ("libcudart.so", "libcudart.so.12", "libcudart.so.13"):
        try:
            lib = ctypes.CDLL(cand)
            d["libcudart"] = cand
            break
        except BaseException:                                     # noqa: BLE001
            continue
    if lib is None:
        d["libcudart"] = "<not loadable>"
        return d
    for attr, key in ((75, "multiprocessor_count"), (97, "shared_optin"),
                      (16, "shared_per_block"), (81, "shared_per_mp"),
                      (86, "l2_cache_size"), (100, "max_threads_per_mp")):
        v = ctypes.c_int(0)
        rc = lib.cudaDeviceGetAttribute(ctypes.byref(v), attr, ctypes.c_int(0))
        d[key] = int(v.value) if rc == 0 else f"rc={rc}"
    # the runtime version, as a string, from cudaRuntimeGetVersion
    v = ctypes.c_int(0)
    rc = lib.cudaRuntimeGetVersion(ctypes.byref(v))
    d["cuda_runtime_version_raw"] = int(v.value) if rc == 0 else f"rc={rc}"
    try:
        txt = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        d["driver_version"] = txt.strip().splitlines()[0]
    except BaseException:                                         # noqa: BLE001
        pass
    return d


# ======================================================== 1. shared memory
def shared_memory() -> dict:
    """max_shared_mem, read five ways, and the two values that are not the same.

    The recorded T4 run produced 65,536 B from fla, triton, torch-optin and
    cudaDeviceGetAttribute(97), and 49,152 B from torch's non-optin field.  Both
    are real numbers about the same card and only one of them is the one the
    gate asks about.  So all of them are reported, with the field that produced
    each, and the gate's own reading is reported beside them rather than
    substituted for them.
    """
    hr("1. SHARED MEMORY -- five readers, and the gate's own verdict")
    got: dict = {}

    guard("fla", lambda: got.update(fla=_fla_smem()))
    guard("triton", lambda: got.update(triton=_triton_smem()))
    guard("torch", lambda: got.update(torch=_torch_smem()))
    guard("attr97", lambda: got.update(attr97=_attr97()))

    fla = got.get("fla") or {}
    gate = fla.get("gate") or {}
    thresholds = fla.get("thresholds") or {}

    say("")
    say("  reader                                        value")
    def _flat(v):
        """One reader's numbers on one line.  A dict of five fields printed as
        `{'a': .., 'b': ..}` under a five-word label is unreadable, and the
        number being looked for is buried in it."""
        if isinstance(v, dict):
            return ", ".join(f"{k}={v[k]}" for k in sorted(v)
                             if isinstance(v[k], (int, float, str, bool)))
        return str(v)

    fl = (got.get("fla") or {})
    say(f"  {'fla.get_all_max_shared_mem':44s} {fl.get('get_all_max_shared_mem')}")
    for k in ("triton", "torch", "attr97"):
        say(f"  {k:44s} {_flat(got.get(k) or {})}")

    say("")
    say("  fla's gate, from the installed fla source, not from a note:")
    # The threshold keys are the Backend enum's NAMES, which are upper case.
    # An earlier version of this loop looked for lower-case keys and printed
    # nothing, so the most decision-relevant numbers in the section were absent
    # from the human-readable block while sitting unformatted in the JSON next to
    # it. A report that silently omits its own key numbers is a defect, not a
    # formatting preference.
    for name, val in sorted(thresholds.items()):
        say(f"    threshold {name:10s} {val:>9,}")
    for backend in ("ampere", "ada", "default", "hopper"):
        if backend in gate:
            say(f"    check_shared_mem({backend!r}) = {gate[backend]}")

    res = {"readers": got, "gate": gate, "thresholds": thresholds}
    emit({"section": "shared_memory", "ok": True, **res})
    return res


def _fla_smem():
    from fla.utils import _device as D
    out: dict = {"file": getattr(D, "__file__", None)}
    for fn in ("get_all_max_shared_mem", "check_shared_mem"):
        if hasattr(D, fn):
            out[fn] = str(getattr(D, fn).__doc__ or "")[:200]
    try:
        out["get_all_max_shared_mem"] = D.get_all_max_shared_mem()
    except BaseException as e:                                    # noqa: BLE001
        out["get_all_max_shared_mem"] = f"<{type(e).__name__}: {e}>"
    out["thresholds"] = {}
    B = getattr(D, "Backend", None)
    if B is not None:
        for b in B:
            try:
                out["thresholds"][b.name] = int(b.value)
            except BaseException:                                 # noqa: BLE001
                pass
    out["gate"] = {}
    for backend in ("ampere", "ada", "default", "hopper"):
        try:
            out["gate"][backend] = bool(D.check_shared_mem(backend))
        except BaseException as e:                                # noqa: BLE001
            out["gate"][backend] = f"<{type(e).__name__}>"
    # The source lines the gate is built on, read on the card.  A scope label
    # built on a line that moved is a scope label about nothing.
    try:
        src = open(D.__file__).read().splitlines()
        out["gate_source"] = [
            f"{i + 1}: {l.rstrip()}"
            for i, l in enumerate(src)
            if "shared_mem" in l or "MAX_" in l
        ][:40]
    except BaseException as e:                                    # noqa: BLE001
        out["gate_source"] = [f"<{type(e).__name__}: {e}>"]
    return out


def _triton_smem():
    import torch
    import triton
    props = triton.runtime.driver.active.utils.get_device_properties(0)
    return {"max_shared_mem": int(props["max_shared_mem"]),
            "all_keys": sorted(props.keys()),
            "n_devices": int(torch.cuda.device_count())}


def _torch_smem():
    import torch
    p = torch.cuda.get_device_properties(0)
    out = {}
    for f in ("shared_memory_per_block", "shared_memory_per_block_optin",
              "shared_memory_per_multiprocessor", "shared_memory_per_block_default"):
        v = getattr(p, f, None)
        if v is not None:
            out[f] = int(v)
    return out


def _attr97():
    lib = None
    for cand in ("libcudart.so", "libcudart.so.12", "libcudart.so.13"):
        try:
            lib = ctypes.CDLL(cand)
            break
        except BaseException:                                     # noqa: BLE001
            continue
    if lib is None:
        return {"error": "no libcudart"}
    out = {}
    for i in range(1):
        v = ctypes.c_int(0)
        rc = lib.cudaDeviceGetAttribute(ctypes.byref(v), 97, ctypes.c_int(i))
        out[f"attr97[{i}]"] = int(v.value) if rc == 0 else f"rc={rc}"
    return out


# ================================================= 2. what the dot really emits
def dot_and_sass(cc: tuple) -> dict:
    """A real tl.dot compiled HERE, then counted.

    Two things are being established, and they are different questions:

      (a) does this card emit a tensor-core instruction for a dot?  That is the
          number the T4 row is 0 at, and it is read from the cubin ptxas was
          handed -- not inferred from a spec sheet and not read from a knob.

      (b) does that dot spill?  Read from `ptxas -v`, which writes to STDERR and
          is the authoritative source for register and spill counts.  The
          T4 reference's 12,806/14,588 B came from there, so this has to come
          from there too or the two are not the same measurement.

    The categoriser is imported from kda_t4/sass_t4.py so the cc=89 row and the
    sm_75 row went through identical code.  If the import fails, the probe says
    so and the census is NOT run by a second parser written here.
    """
    hr("2. A tl.dot COMPILED ON THIS CARD, censused from its cubin")
    res: dict = {}
    try:
        sys.path.insert(0, HERE)
        from sass_t4 import (TC_EXACT, count, parse_sass, ptxas_v)   # noqa: F401
        res["categoriser"] = "kda_t4/sass_t4.py (the study's own)"
    except BaseException as e:                                    # noqa: BLE001
        res["categoriser"] = f"IMPORT FAILED {type(e).__name__}: {e}"
        emit({"section": "dot_sass", "ok": False,
              "error": "could not import kda_t4/sass_t4.py; no census run",
              "detail": str(e)})
        say("  !! kda_t4/sass_t4.py did not import. NO CENSUS IS RUN.")
        say("     Writing a second parser here would make this row incomparable")
        say("     with the sm_75 row, which is the defect this probe exists to")
        say("     avoid. Recorded as a null instead.")
        return res

    # The kernel is the same shape as the T4 probe's: one 32x32x32 tl.dot with no
    # explicit input_precision, so whatever the compiler resolves is the answer.
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _dot_probe(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        ia = tl.arange(0, M)[:, None] * K + tl.arange(0, K)[None, :]
        ib = tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :]
        x = tl.load(a + ia)
        y = tl.load(b + ib)
        z = tl.dot(x, y)
        tl.store(c + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], z)

    M = N = K = 32
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")
    c = torch.empty(M, N, device="cuda")
    kern = _dot_probe[(1,)](a, b, c, M, N, K, num_warps=4, num_stages=2)

    ptx = kern.asm["ptx"]
    cubin_path = os.path.join(OUT, "l4_dot.cubin")
    ptx_path = os.path.join(OUT, "l4_dot.ptx")
    with open(cubin_path, "wb") as f:
        f.write(kern.asm["cubin"])
    with open(ptx_path, "w") as f:
        f.write(ptx)

    res["ptx_target"] = [l.strip() for l in ptx.splitlines()[:12]
                         if l.startswith((".version", ".target"))]
    res["ptx_mma_sync"] = len(re.findall(r"\bmma\.sync\b", ptx))
    res["ptx_mma"] = len(re.findall(r"\bmma\.", ptx))
    res["ptx_fma_rn_f32"] = len(re.findall(r"\bfma\.rn\.f32\b", ptx))
    res["ptx_cvt_tf32"] = len(re.findall(r"\bcvt\.(rn\.|tf32)", ptx))
    res["triton_n_regs"] = int(getattr(kern, "n_regs", -1))
    res["triton_n_spills"] = int(getattr(kern, "n_spills", -1))

    # numeric answer, so a kernel that emitted the wrong thing is caught even
    # though it compiled
    ref = (a.double() @ b.double())
    err = (c.double() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-30)
    res["dot_max_rel_err_fp32"] = err
    res["dot_finite"] = bool(torch.isfinite(c).all().item())

    say("")
    say(f"  PTX .target                 {res['ptx_target']}")
    say(f"  mma.sync in PTX            {res['ptx_mma_sync']}")
    say(f"  fma.rn.f32 in PTX          {res['ptx_fma_rn_f32']}")
    say(f"  cvt.*tf32 in PTX           {res['ptx_cvt_tf32']}")
    say(f"  triton n_regs / n_spills   {res['triton_n_regs']} / {res['triton_n_spills']}")
    say(f"  32x32x32 dot max rel err   {res['dot_max_rel_err_fp32']:.3e} (fp32 inputs)")
    say(f"  categoriser                {res['categoriser']}")

    pv = ptxas_v(ptx_path)
    res["ptxas_v"] = pv
    say("")
    say("  ptxas -v (authoritative for registers and spill):")
    for k, v in sorted(pv.items()):
        say(f"    {k:22s} {v}")

    cen = count(cubin_path, "l4_dot_live")
    res["sass"] = cen
    if cen:
        say("")
        say("  SASS census of the cubin this card produced:")
        say(f"    total instructions       {cen['total']}")
        say(f"    tensor-core mnemonics    {cen['tc_mnemonic_table']}")
        say(f"    HMMA-family              {cen['HMMA']}")
        say(f"    FFMA                     {cen['FFMA']}")
        say(f"    LDL / STL (spill)        {cen['LDL']} / {cen['STL']}")
    emit({"section": "dot_sass", "ok": True, **res})
    return res


# ============================================== 3. fla's own kernel, on this card
def fla_kernel_and_configs() -> dict:
    """fla's config space on this card, and the config autotune picks.

    The collapse from 12 configs to 4 on sm_75 was half of the T4 finding, so
    the live count is recorded here before anything is timed.  Two readings are
    taken and compared: the pruned list on the Autotuner, and the same list
    *modelled* from the gate's return values.  When they disagree, the observed
    one is the fact and the model is the theory -- and this probe says so rather
    than printing only the one that agrees with the note.
    """
    hr("3. fla's CONFIG SPACE on this card, and the config it picks")
    res: dict = {}
    try:
        import fla.ops.common.chunk_delta_h as CDH
    except BaseException as e:                                    # noqa: BLE001
        emit({"section": "fla_configs", "ok": False, "error": str(e)})
        say(f"  !! could not import fla.ops.common.chunk_delta_h: {e}")
        return res

    raw = getattr(CDH, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", None)
    res["kernel_found"] = raw is not None
    if raw is None:
        emit({"section": "fla_configs", "ok": False,
              "error": "chunk_gated_delta_rule_fwd_kernel_h_blockdim64 absent"})
        return res

    fn, chain = _unwrap_autotuner(raw)
    res["decorator_chain"] = chain
    res["autotuner_found"] = fn is not None
    if fn is None:
        emit({"section": "fla_configs", "ok": False, "error": "no Autotuner"})
        return res

    cfgs = getattr(fn, "configs", None)
    res["n_configs"] = len(cfgs) if cfgs else 0
    rows = []
    for c in (cfgs or []):
        rows.append({"BV": c.kwargs.get("BV"), "num_warps": c.num_warps,
                     "num_stages": c.num_stages,
                     "num_ctas": getattr(c, "num_ctas", None)})
    res["configs"] = rows
    res["distinct"] = {"BV": sorted({r["BV"] for r in rows}),
                       "num_warps": sorted({r["num_warps"] for r in rows}),
                       "num_stages": sorted({r["num_stages"] for r in rows})}

    from fla.utils import _device as D
    model_BV = [32, 64] if D.check_shared_mem("ada") else [32]
    model_ns = [2, 3, 4] if D.check_shared_mem("ampere") else [2, 1]
    res["model"] = {"BV": model_BV, "num_stages": model_ns}
    res["model_agrees"] = (set(r["BV"] for r in rows) == set(model_BV)
                           and set(r["num_stages"] for r in rows) == set(model_ns))

    say("")
    say(f"  decorator chain           {' -> '.join(chain)}")
    say(f"  THE CONFIG SPACE, live    {res['n_configs']} configs")
    for r in rows:
        say(f"    BV={r['BV']!s:<4} num_warps={r['num_warps']!s:<3} "
            f"num_stages={r['num_stages']}")
    say(f"  distinct                  {res['distinct']}")
    say(f"  modelled from the gate    BV={model_BV} num_stages={model_ns}")
    say(f"  model vs observed         "
        f"{'AGREE' if res['model_agrees'] else 'DISAGREE'}")
    emit({"section": "fla_configs", "ok": True, **res})
    return res


def _unwrap_autotuner(obj, depth: int = 0):
    chain = [type(obj).__name__]
    fn = None
    o = obj
    while o is not None and depth < 12:
        if hasattr(o, "configs") and hasattr(o, "fn"):
            fn = o
        nxt = getattr(o, "fn", None)
        if nxt is None or nxt is o:
            break
        o = nxt
        chain.append(type(o).__name__)
        depth += 1
    return fn, chain


# ================================================== 4. one real fla forward
def fla_forward_and_cubin() -> dict:
    """Run the op once so autotune PICKED, then census the cubin it compiled.

    This is the row that decides whether the reference is degraded on this card.
    The T4 row is 19,328 SASS instructions, 8,192 FFMA, 64 registers and
    12,806/14,588 B of spill, with zero tensor-core instructions anywhere.  The
    comparison is only meaningful if the census instrument and the dtype match,
    so both are recorded here rather than assumed.
    """
    hr("4. one real fla forward, then the cubin fla itself compiled here")
    res: dict = {}
    try:
        import torch
        from fla.ops.kda import chunk_kda
    except BaseException as e:                                    # noqa: BLE001
        emit({"section": "fla_forward", "ok": False, "error": str(e)})
        say(f"  !! could not import fla.ops.kda: {e}")
        return res

    # fp16, deliberately.  The recorded T4 comparison used fp16 and measured
    # 7.148e-04 against 5.714e-03 relative error for the same operator, and a
    # dtype is a choice about precision, not a change of operator.  It is stated
    # here because a bf16 run is not the same measurement and must not be
    # compared with an fp16 one.
    DTYPE = torch.float16
    res["dtype"] = str(DTYPE)
    say(f"  dtype                     {DTYPE}")
    say("  (fp16, to match the recorded T4 arm; bf16 would not be the same")
    say("   measurement and its error is an order of magnitude larger.)")

    B, T, H, K, V = 1, 512, 4, 128, 128
    res["shape"] = {"B": B, "T": T, "H": H, "K": K, "V": V}
    try:
        torch.manual_seed(0)
        q = torch.randn(B, T, H, K, dtype=DTYPE, device="cuda")
        k = torch.randn(B, T, H, K, dtype=DTYPE, device="cuda")
        v = torch.randn(B, T, H, V, dtype=DTYPE, device="cuda")
        g = torch.rand(B, T, H, dtype=DTYPE, device="cuda")
        beta = torch.rand(B, T, H, dtype=DTYPE, device="cuda")
        o, _ = chunk_kda(q=q, k=k, v=v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
        torch.cuda.synchronize()
        res["forward_ok"] = True
        res["out_finite"] = bool(torch.isfinite(o).all().item())
        res["out_shape"] = list(o.shape)
        res["out_absmax"] = float(o.abs().max().item())
        say("")
        say(f"  forward OK                {res['out_finite']}  shape {res['out_shape']}")
        say(f"  out absmax                {res['out_absmax']:.6g}")
    except BaseException as e:                                    # noqa: BLE001
        res["forward_ok"] = False
        res["forward_error"] = f"{type(e).__name__}: {e}"
        emit({"section": "fla_forward", "ok": False, **res})
        say(f"  !! forward FAILED {type(e).__name__}: {str(e)[:200]}")
        return res

    picked = _selected_configs()
    res["selected_configs"] = picked
    say("")
    say("  configs autotune picked (Autotuner.cache):")
    for kk, v in picked.items():
        say(f"    {kk}: {v}")

    cens = _census_live_cubins()
    res["live_cubins"] = cens
    emit({"section": "fla_forward", "ok": True, **res})
    return res


def _selected_configs():
    """The config autotune chose, read out of the live Autotuner cache.

    Read from the objects fla itself holds, not from a log line and not from a
    guess about which config would win.  Returns {} when no Autotuner can be
    reached, which is a null and not a zero.
    """
    out = {}
    try:
        import fla.ops.common.chunk_delta_h as CDH
        raw = getattr(CDH, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", None)
        fn, _chain = _unwrap_autotuner(raw)
        cache = getattr(fn, "cache", None) or {}
        for k, v in cache.items():
            try:
                out[str(k)] = dict(num_warps=getattr(v, "num_warps", None),
                                   num_stages=getattr(v, "num_stages", None),
                                   num_ctas=getattr(v, "num_ctas", None),
                                   kwargs={k2: str(v2) for k2, v2
                                           in getattr(v, "kwargs", {}).items()})
            except BaseException as e:                            # noqa: BLE001
                out[str(k)] = f"<{type(e).__name__}: {e}>"
    except BaseException as e:                                    # noqa: BLE001
        return {"<error>": f"{type(e).__name__}: {e}"}
    return out


def _census_live_cubins():
    """Every cubin triton actually produced on this card, censused.

    The T4 row is "0 tensor-core instructions across all 45 real cubins", so
    the comparable statement here is a census over *all* of this card's cubins,
    not over one chosen kernel.  The cubins are found by walking triton's own
    cache directory for files this session wrote, and each is identified by
    digest so a stale cache entry from a previous session cannot be counted as
    this card's work.
    """
    try:
        sys.path.insert(0, HERE)
        from sass_t4 import count
    except BaseException as e:                                    # noqa: BLE001
        return {"error": f"no categoriser: {e}"}

    roots = []
    for env in ("TRITON_CACHE_DIR", "TRITON_HOME"):
        v = os.environ.get(env)
        if v and os.path.isdir(v):
            roots.append(v)
    roots.append(os.path.expanduser("~/.triton/cache"))
    seen_dirs = set()
    cubins = []
    for root in roots:
        if root in seen_dirs or not os.path.isdir(root):
            continue
        seen_dirs.add(root)
        for dirpath, _d, files in os.walk(root):
            for fn in files:
                if fn.endswith(".cubin"):
                    p = os.path.join(dirpath, fn)
                    try:
                        if os.path.getmtime(p) < T0 - 60:
                            continue          # not from this session
                    except OSError:
                        continue
                    cubins.append(p)
    say("")
    say(f"  cubins written by THIS session: {len(cubins)}")
    rows = []
    totals = {"n": 0, "tensor_core": 0, "total_instructions": 0,
              "spill_store_bytes": 0, "spill_load_bytes": 0,
              "max_registers": 0, "max_stack_frame": 0}
    for p in sorted(cubins):
        label = "l4live_" + os.path.basename(os.path.dirname(p))[:24] + "_" + \
            os.path.basename(p).split(".")[0][:16]
        cen = count(p, label, keep_text=False)
        if not cen:
            continue
        pv = cen.get("ptxas_v") or {}
        tc = sum((cen.get("tc_mnemonic_table") or {}).values())
        rows.append({"label": label, "total": cen["total"],
                     "tc": tc, "tc_table": cen.get("tc_mnemonic_table"),
                     "HMMA": cen.get("HMMA"), "FFMA": cen.get("FFMA"),
                     "LDL": cen.get("LDL"), "STL": cen.get("STL"),
                     "registers": pv.get("registers"),
                     "spill_store_bytes": pv.get("spill_store_bytes"),
                     "spill_load_bytes": pv.get("spill_load_bytes"),
                     "stack_frame_bytes": pv.get("stack_frame_bytes")})
        totals["n"] += 1
        totals["tensor_core"] += tc
        totals["total_instructions"] += cen["total"]
        totals["spill_store_bytes"] += pv.get("spill_store_bytes") or 0
        totals["spill_load_bytes"] += pv.get("spill_load_bytes") or 0
        totals["max_registers"] = max(totals["max_registers"],
                                      pv.get("registers") or 0)
        totals["max_stack_frame"] = max(totals["max_stack_frame"],
                                        pv.get("stack_frame_bytes") or 0)

    say("")
    say("   label                              inst    TC  HMMA   FFMA   LDL  STL"
        "   regs  spillSt spillLd")
    for r in sorted(rows, key=lambda r: -r["total"])[:30]:
        say(f"   {r['label'][:32]:32s} {r['total']:6d} {r['tc']:5d} "
            f"{r['HMMA'] or 0:6d} {r['FFMA'] or 0:6d} {r['LDL'] or 0:4d} "
            f"{r['STL'] or 0:4d} {str(r['registers'] or '-'):>5s} "
            f"{str(r['spill_store_bytes'] if r['spill_store_bytes'] is not None else '-'):>8s} "
            f"{str(r['spill_load_bytes'] if r['spill_load_bytes'] is not None else '-'):>7s)}")
    say("")
    say(f"  TOTALS over {totals['n']} cubins: {totals['total_instructions']} instructions, "
        f"{totals['tensor_core']} tensor-core, "
        f"max {totals['max_registers']} registers, "
        f"spill stores {totals['spill_store_bytes']} B / loads {totals['spill_load_bytes']} B")
    return {"rows": rows, "totals": totals}


# ================================================================== 5. verdict
def verdict(ident: dict, smem: dict, dot: dict, cfgs: dict, fwd: dict) -> None:
    hr("5. VERDICT, and what may be quoted from this run")
    cap = ident.get("cc_str")
    smem_v = ((smem.get("readers") or {}).get("triton") or {}).get("max_shared_mem")
    sass = dot.get("sass") or {}
    tc = sum((sass.get("tc_mnemonic_table") or {}).values()) if sass else None
    gate = (smem.get("gate") or {}).get("ada")

    say("")
    say(f"  card                        {(ident.get('names_seen') or ['?'])[0]}")
    say(f"  compute capability          {cap}   (8.9 = Ada = what the study needs)")
    say(f"  optin shared memory         {smem_v} B")
    say(f"  fla check_shared_mem('ada') {gate}")
    say(f"  tensor-core in the dot      {tc}   (T4 row: 0)")
    say(f"  fla config space            {cfgs.get('n_configs')} configs "
        f"(T4 row: 4 of 12)")
    say(f"  fla forward ok              {(fwd or {}).get('forward_ok')}")

    passed = (ident.get("is_ada_89") is True and smem_v is not None
              and smem_v >= 101376 and tc is not None and tc > 0)
    say("")
    if passed:
        say("  VERDICT: PASS. The card is Ada, its opt-in shared memory clears the")
        say("  101,376 B that fla's gate asks for, and a dot on it emits")
        say("  tensor-core instructions. The KDA arm is measurable here and the")
        say("  ratio in notes/39 SS5's falsifier can be tested.")
    else:
        why = []
        if ident.get("is_ada_89") is not True:
            why.append(f"cc is {cap}, not 8.9")
        if smem_v is None or smem_v < 101376:
            why.append(f"optin shared memory {smem_v} B < 101,376 B")
        if tc is None or tc == 0:
            why.append(f"tensor-core instructions in the dot: {tc}")
        say("  VERDICT: FAIL -> " + "; ".join(why))
        say("  The KDA arm stays unquantified against a competent reference on")
        say("  this card, exactly as notes/39 SS3 says. Nothing here changes that.")
    emit({"section": "verdict", "ok": True, "passed": bool(passed),
          "cc": cap, "optin_smem": smem_v, "dot_tensor_core": tc,
          "fla_n_configs": cfgs.get("n_configs"),
          "fla_forward_ok": (fwd or {}).get("forward_ok"),
          "control_run": not ident.get("is_ada_89")})


# ======================================================================= main
def main() -> int:
    say("=" * 76)
    say(f"l4_probe   started {R['started']}   python {sys.version.split()[0]}")
    say("=" * 76)
    say(f"durable record: {JL}")

    # fla must be imported only AFTER the pre-import environment is recorded --
    # fla writes TRITON_F32_DEFAULT into os.environ at import time, so a probe
    # that imports first cannot see the change it is meant to measure.
    KEYS = ("TRITON_F32_DEFAULT", "TRITON_F32_DEFAULT_PRECISION")
    pre = {k: os.environ.get(k) for k in KEYS}
    say("")
    say(f"  BEFORE any import: TRITON_F32_DEFAULT={pre['TRITON_F32_DEFAULT']!r}")

    step("0 identity")
    ident = identity()

    if not ident.get("gpu_present"):
        # The first L4 request landed here.  There is nothing to measure on a
        # machine with no GPU, and pretending otherwise by running sections that
        # will fail on a missing module is how a CPU box gets a row in a table
        # of L4 measurements.  Stop, and say so in the durable record.
        emit({"section": "verdict", "ok": True, "passed": False,
              "reason": "no GPU attached to the machine the kernel ran on",
              "requested_shape_recorded_by_server": None,
              "cc": ident.get("cc_str"), "gpu_present": False})
        hr("VERDICT: NO GPU. Stopping before the measurement sections.")
        say("  This run establishes that the machine_shape was accepted and")
        say("  recorded, and that no accelerator was attached. It establishes")
        say("  nothing whatsoever about an L4's shared memory, tensor cores or")
        say("  tf32, and none of it may be quoted as such.")
        _finish()
        return 0

    step("1 shared memory (imports fla)")
    import_error = None
    try:
        import fla  # noqa: F401
    except BaseException as e:                                    # noqa: BLE001
        import_error = f"{type(e).__name__}: {e}"
    post = {k: os.environ.get(k) for k in KEYS}
    say(f"  AFTER  import fla: TRITON_F32_DEFAULT={post['TRITON_F32_DEFAULT']!r}")
    if import_error:
        say(f"  !! import fla FAILED: {import_error}")
    emit({"section": "f32_default", "ok": import_error is None,
          "before": pre, "after": post,
          "fla_import_error": import_error,
          "changed_by_import": pre != post})
    smem = shared_memory()

    # Sections 2-4 cost compiles and a fla forward, and they only produce a
    # number worth having on a card this study has not already measured.  A T4
    # control is here to prove the *request* reached a GPU, and sections 0 and 1
    # prove that in seconds.  Spending eight minutes of a shared T4's quota
    # re-measuring sm_75 -- which notes/30 already measured on the same card --
    # would buy a duplicate row and cost the session cap the other lanes need.
    # A defect that only lives on the branch that never ran. `identity()` stores
    # the capability as a LIST (so it round-trips through JSON without becoming
    # a tuple, which json cannot represent), and this compared that list to a
    # tuple. Every L4 attempt died in the no-GPU branch before reaching it, so
    # the bug was invisible to the runs that motivated the probe; the T4 control
    # crashed on it. Fixed by normalising here, and the lesson is the reason the
    # control exists rather than being a formality.
    cc = tuple(ident.get("cc") or (0, 0))
    if cc < (8, 0):
        emit({"section": "verdict", "ok": True, "passed": False,
              "reason": f"control run on cc {ident['cc_str']}; sections 2-4 "
                        f"skipped by design (the card is already measured)",
              "cc": ident.get("cc_str"),
              "optin_smem": (smem.get("readers") or {}).get("triton", {})
                            .get("max_shared_mem"),
              "gpu_present": True})
        hr(f"VERDICT: CONTROL on cc {ident['cc_str']}. Sections 2-4 skipped by design.")
        say("  A GPU was attached, which is what this run was dispatched to")
        say("  establish. Its shared memory was read above. The dot/SASS census")
        say("  and the fla forward are NOT run on a card the study has already")
        say("  measured, and their absence here is deliberate, not a failure.")
        _finish()
        return 0

    step("2 tl.dot on this card + SASS")
    dot = dot_and_sass(ident.get("cc") or (0, 0))

    step("3 fla config space")
    cfgs = fla_kernel_and_configs()

    step("4 fla forward + the cubins this card produced")
    fwd = fla_forward_and_cubin()

    step("5 verdict")
    verdict(ident, smem, dot, cfgs, fwd)

    _finish()
    return 0


def _finish() -> None:
    R["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    R["elapsed_s"] = round(time.time() - T0, 1)
    say("")
    say(f"  total {R['elapsed_s']} s")
    try:
        with open(os.path.join(OUT, "l4_probe.json"), "w") as f:
            json.dump(R, f, indent=1, default=repr)
        say(f"  wrote {os.path.join(OUT, 'l4_probe.json')}")
    except OSError as e:
        say(f"  !! could not write the summary json: {e}")


if __name__ == "__main__":
    sys.exit(main())
