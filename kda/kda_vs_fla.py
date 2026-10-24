"""
kda_vs_fla.py - NVIDIA-vs-NVIDIA comparison of KDA:
flash-linear-attention's hand-written Triton kernel  vs  this project's
TinyGrad port, on the SAME GPU, at the SAME shapes, with correctness checked
against the same float64 reference.

Self-contained by design: everything except pip packages is in this file, so a
Colab / cloud GPU run is a single cell and finishes in minutes. Nothing here
reads from the local repo.

DESIGN RULES FOR UNATTENDED EXECUTION
    * every torch / triton / fla touchpoint is optional. A missing dependency
      prints one "SKIP: <reason>" line and the run continues, so a torch-less
      laptop still produces the full TinyGrad-only measurement set.
    * nothing prompts. Nothing calls input(). No shape is attempted whose
      intermediate provably exceeds the free memory -- that is reported as a
      *predicted* OOM data point instead, because a real OOM on a 16 GB T4 is
      a fact worth stating, not an abort.
    * every stage is individually wrapped. One failing stage cannot take the
      summary down with it.

Run:
    !pip install -q tinygrad triton flash-linear-attention
    !python kda_vs_fla.py            # or paste into a cell
    ../../.venv/bin/python kda_vs_fla.py    # torch-less: TinyGrad-only numbers

What it prints
    1. environment: backend, device name, free memory, which deps are present
    2. oracle self-check: kda_recurrent against two independent closed forms
    3. correctness: both implementations vs a float64 per-token recurrence
    4. arithmetic intensity: FLOPs and bytes per step, roofline, and the
       decay-intermediate size ratio -- the study's central finding
    5. throughput at several shapes, every number decomposed into
       build / schedule / kernel / copy, with OOM as a data point
    6. the generated CUDA source, captured to out/codegen/ (CUDA only)
    7. a 30-second summary

The one number to read first is in section 4:

    TinyGrad materialises the decayed key-key gram as [B,NT,HV,C,C,K];
    the Triton kernel holds [BC,K] per sub-chunk, in registers.
    At C=64, BC=16 that is a 256x difference, and it is arch-independent --
    it is a property of the two programs, not of the GPU they run on.
"""
import math
import os
import re
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODEGEN_DIR = os.path.join(HERE, "out", "codegen")

# things the stages record for the summary
RESULTS = {}


def hr(title=""):
    if title:
        print()
        print("=" * 78)
        print(title)
        print("=" * 78)
    else:
        print("=" * 78)


# ---------------------------------------------------------------------------
# 0. environment.  every heavy dependency is probed, never imported blindly.
# ---------------------------------------------------------------------------
OPT = {}          # name -> (module | None, error string | None)
DEPS = ("torch", "triton", "fla", "flash_linear_attention")


def dep(name):
    """The module if it imported, else None. Never raises."""
    return OPT.get(name, (None, "not probed"))[0]


def dep_err(name):
    return OPT.get(name, (None, "not probed"))[1]


def probe_deps():
    for name in DEPS:
        try:
            OPT[name] = (__import__(name), None)
        except BaseException as e:                     # noqa: BLE001
            OPT[name] = (None, f"{type(e).__name__}: {str(e)[:64]}")
    # fla is importable under either name; treat them as one feature
    if not dep("fla") and dep("flash_linear_attention"):
        OPT["fla"] = (dep("flash_linear_attention"), None)
        OPT["flash_linear_attention"] = (dep("flash_linear_attention"), None)


def is_cuda():
    """True iff tinygrad's default backend is a CUDA one."""
    try:
        from tinygrad import Device
        d = Device.DEFAULT
        if d.upper().startswith("CUDA"):
            return True
        return "cuda" in type(Device[d]).__module__.lower()
    except Exception:
        return False


def has_torch_cuda():
    t = dep("torch")
    try:
        return bool(t) and bool(t.cuda.is_available())
    except Exception:
        return False


def backend_name():
    try:
        from tinygrad import Device
        return f"tinygrad:{Device.DEFAULT}"
    except Exception:
        return "tinygrad:?"


def device_name():
    """Name of the device the timings below were taken on. Never raises."""
    if has_torch_cuda():
        try:
            return dep("torch").cuda.get_device_name(0)
        except Exception:
            pass
    try:
        from tinygrad import Device
        d = Device[d_ := Device.DEFAULT]
        # metal/hip/etc. carry a real name; CPU does not
        return getattr(d, "device", None) or getattr(d, "name", None) or d_
    except Exception:
        return "unknown"


def compute_capability():
    if has_torch_cuda():
        try:
            c = dep("torch").cuda.get_device_capability(0)
            return f"sm_{c[0]}{c[1]}"
        except Exception:
            pass
    return "n/a (not a CUDA device)"


# Every distinct kernel tinygrad compiles, process-wide. On a fresh machine this
# is the number of launches the port's graph decomposes into, which is the
# clearest single statement of what a static-shape graph costs: fla's KDA is a
# handful of persistent kernels, and this is whatever count lands here.
KERNEL_COMPILES = []


def install_kernel_counter():
    """Count compilations process-wide. tinygrad's on-disk cache means a second
    run of the same graph compiles nothing, so the count is a first-run figure;
    it is reported as such rather than as a per-call cost."""
    try:
        from tinygrad import Device
        comp = getattr(Device[Device.DEFAULT], "compiler", None)
        orig = getattr(comp, "compile_cached", None)
        if orig is None or getattr(comp, "_kda_spied", False):
            return
        def spy(src, *a, **k):
            KERNEL_COMPILES.append(1)
            return orig(src, *a, **k)
        comp.compile_cached = spy
        comp._kda_spied = True
    except Exception:
        pass


def free_memory_bytes():
    """Best-effort free memory. CUDA: real device free. Otherwise host RAM."""
    if has_torch_cuda():
        try:
            return int(dep("torch").cuda.mem_get_info()[0])
        except Exception:
            pass
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except Exception:
        return None


def sync():
    """Block until queued device work is done. No-op if we cannot."""
    try:
        from tinygrad import Device
        Device[Device.DEFAULT].synchronize()
        return True
    except Exception:
        try:
            dep("torch").cuda.synchronize()
            return True
        except Exception:
            return False


def env_report():
    hr("environment")
    print(f"  backend         {backend_name()}")
    print(f"  device          {device_name()}")
    fm = free_memory_bytes()
    print(f"  free memory     {fm/2**30:.1f} GiB" if fm else "  free memory     unknown")
    if is_cuda():
        print(f"  codegen capture ENABLED -> {os.path.relpath(CODEGEN_DIR, HERE)}/")
    else:
        print("  codegen capture SKIP: backend is not CUDA, nothing to capture")

    torch = dep("torch")
    if torch is not None:
        try:
            cap = torch.cuda.get_device_capability(0)
            p = torch.cuda.get_device_properties(0)
            print(f"  compute cap     sm_{cap[0]}{cap[1]}")
            print(f"  vram            {p.total_memory/1e9:.1f} GB")
            print(f"  SMs             {p.multi_processor_count}")
            # bf16 tensor cores arrived in sm_80. T4 is sm_75 and has none, which
            # changes which fla code path is exercised, so it must be reported.
            print(f"  bf16 tensor     "
                  f"{'yes' if cap[0] >= 8 else 'NO (sm<80) -- fla will use a non-MMA path'}")
        except Exception as e:
            print(f"  torch/cuda: SKIP: {type(e).__name__}: {str(e)[:60]}")
    else:
        print(f"  torch           SKIP: {dep_err('torch')}")

    for name, pretty in (("triton", "triton"), ("fla", "flash-linear-attention")):
        if dep(name) is None:
            print(f"  {pretty:<15} SKIP: {dep_err(name)}")
        else:
            m = dep(name)
            print(f"  {pretty:<15} {getattr(m, '__version__', 'present')}")

    try:
        from tinygrad import Device
        print(f"  tinygrad dev    {Device.DEFAULT}")
        try:
            live = [d for d in Device._devices if d]
            print(f"  available       {live}")
        except Exception:
            pass
    except Exception as e:
        print(f"  tinygrad: SKIP: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 1. float64 reference -- the definition, and the arbiter
# ---------------------------------------------------------------------------
def l2norm(x, eps=1e-6):
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + eps)


def kda_recurrent(q, k, v, g, beta, scale=None):
    """Kimi Delta Attention, Eq. 1, applied one token at a time, float64.

        S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
        o_t = S_t^T q_t

    The decay multiplies S_{t-1} FIRST and the delta correction is applied to
    the already-decayed state. That order is not reversible and is the single
    most common transcription error.

    `v` and `g` are separate arguments with different trailing extents (V vs K)
    and must stay separate; `verify_oracle()` proves they are, because the
    closed forms below fail loudly if either is wired to the other.
    """
    F = np.float64
    q = np.asarray(q, F); k = np.asarray(k, F); v = np.asarray(v, F)
    g = np.asarray(g, F); beta = np.asarray(beta, F)
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    G = HV // H
    if scale is None:
        scale = K ** -0.5
    q = np.repeat(q, G, axis=2) * scale
    k = np.repeat(np.asarray(k, F), G, axis=2)
    v = np.asarray(v, F)
    g = np.asarray(g, F)
    S = np.zeros((B, HV, K, V), F)
    o = np.zeros((B, T, HV, V), F)
    for t in range(T):
        S = S * np.exp(g[:, t])[..., None]
        kt, vt = k[:, t], v[:, t]
        resid = vt - np.einsum("bhk,bhkv->bhv", kt, S)
        S = S + (beta[:, t][..., None, None] * kt[..., :, None] * resid[..., None, :])
        o[:, t] = np.einsum("bhk,bhkv->bhv", q[:, t], S)
    return o, S


def rnd(B, T, H, HV, K, V, seed=0, gmax=0.7, dtype=np.float64):
    """Random KDA inputs. q and k are L2-normalised, which is what the real
    KDA layer does and what keeps the operator well conditioned: with raw
    Gaussians, k^T k ~ K, the delta-rule matrix (I - beta k k^T) has
    eigenvalues near 1 - K*beta, and the state diverges."""
    rng = np.random.default_rng(seed)
    f = np.dtype(dtype).type
    q = l2norm(rng.normal(size=(B, T, H, K)).astype(f))
    k = l2norm(rng.normal(size=(B, T, H, K)).astype(f))
    v = rng.normal(size=(B, T, HV, V)).astype(f)
    g = -rng.random((B, T, HV, K)).astype(f) * f(gmax)
    beta = rng.random((B, T, HV)).astype(f)
    return q, k, v, g, beta


# ---------------------------------------------------------------------------
# 2. the oracle's own proof.  The reference is what every number in this file
#    is measured against, so it gets checked against two closed forms that do
#    not share its structure.  This is what caught the v/g mixup.
# ---------------------------------------------------------------------------
def _scalar_oracle(q, k, v, g, beta, scale=None, correct_order=True):
    """V1. The intended math transcribed in plain Python floats and lists:
    B = H = HV = 1, V = 1, K arbitrary. No numpy broadcasting, no einsum, no
    array shapes to get right -- if the array version puts the decay and the
    delta correction in the wrong order, or wires v and g to each other, or
    transposes a contraction, this disagrees.

    The delta correction is the FULL rule, cross terms included:

        S_d  <-  S_d - beta*k_d*(sum_e k_e*S_e) + beta*k_d*v

    Writing it as (1 - beta*k_d^2)*S_d instead is the tempting shortcut, and it
    is wrong for every K > 1: it silently drops the sum over e. It happens to
    be exact at K = 1, which is why a scalar spot-check passes a broken port.
    """
    B, T, H, K = q.shape
    assert (B, H) == (1, 1) and v.shape[3] == 1, "V1 assumes B=H=HV=1, V=1"
    if scale is None:
        scale = K ** -0.5
    S = [0.0] * K
    o = np.zeros((1, T, 1, 1))
    for t in range(T):
        a = [math.exp(float(g[0, t, 0, d])) for d in range(K)]
        kt = [float(k[0, t, 0, d]) for d in range(K)]
        vt = float(v[0, t, 0, 0])
        bt = float(beta[0, t, 0])

        def delta(Sd):
            dot = sum(kt[e] * Sd[e] for e in range(K))
            return [Sd[d] - bt * kt[d] * dot + bt * kt[d] * vt for d in range(K)]

        if correct_order:
            S = delta([a[d] * S[d] for d in range(K)])       # decay FIRST
        else:
            S = [a[d] * sd for d, sd in enumerate(delta(S))]  # decay LAST
        o[0, t, 0, 0] = sum(float(q[0, t, 0, d]) * scale * S[d] for d in range(K))
    return o, S


def _wy_oracle(q, k, v, g, beta, scale=None):
    """V2. g == 0, so the delta rule has the exact algebraic form

        S = K^T (I + strictLower(Diag(beta) K K^T))^-1 V Diag(beta)

    which is a triangular inverse, not a recurrence. Different algebra, same
    operator: it pins the beta placement and the o = S^T q projection. Shapes
    with K != V and G > 1 are used, so an argument swap shows up as an error
    rather than being hidden by equal extents.
    """
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    G = HV // H
    if scale is None:
        scale = K ** -0.5
    q = np.repeat(q, G, axis=2) * scale
    k = np.repeat(k, G, axis=2)
    S = np.zeros((B, HV, K, V))
    o = np.zeros((B, T, HV, V))
    for t in range(T):
        Km, Vm, bt = k[:, :t + 1], v[:, :t + 1], beta[:, :t + 1]
        for bi in range(B):
            for hi in range(HV):
                Kh = Km[bi, :, hi]                        # [t+1, K]
                bh = bt[bi, :, hi]                        # [t+1]
                A = np.eye(t + 1) + np.tril((Kh @ Kh.T) * bh[:, None], -1)
                S[bi, hi] = Kh.T @ np.linalg.inv(A) @ (Vm[bi, :, hi] * bh[:, None])
        o[:, t] = np.einsum("bhk,bhkv->bhv", q[:, t], S)
    return o, S


def _relerr(a, b):
    d = np.abs(b).max()
    return float(np.abs(a - b).max() / d) if d else float(np.abs(a - b).max())


def verify_oracle():
    """Prove kda_recurrent before believing anything computed with it."""
    hr("oracle self-check -- kda_recurrent vs independent closed forms")
    ok = True

    print("  V1  plain-Python transcription (B=H=HV=1, V=1, K=1..32, T=1..40)")
    print("      the 'control' column is the same transcription with the decay and")
    print("      the delta correction swapped; it MUST disagree, or V1 proves nothing")
    print(f"      {'K':>4}{'T':>6}{'relerr':>13}{'control':>13}   verdict")
    rng = np.random.default_rng(11)
    worst, teeth = 0.0, 0.0
    for K in (1, 2, 8, 32):
        for T in (1, 5, 40):
            q = rng.normal(size=(1, T, 1, K))
            k = rng.normal(size=(1, T, 1, K))
            k /= np.linalg.norm(k, axis=-1, keepdims=True)   # keep the state bounded
            v = rng.normal(size=(1, T, 1, 1))
            g = -rng.random((1, T, 1, K)) * 0.9
            beta = rng.random((1, T, 1))
            ro, _ = kda_recurrent(q, k, v, g, beta)
            e = _relerr(ro, _scalar_oracle(q, k, v, g, beta)[0])
            ew = _relerr(ro, _scalar_oracle(q, k, v, g, beta, correct_order=False)[0])
            worst, teeth = max(worst, e), max(teeth, ew)
            bad = "" if (e < 1e-12 and ew > 1e-6) else "   <-- NOT CONCLUSIVE"
            print(f"      {K:>4}{T:>6}{e:>13.3e}{ew:>13.3e}   "
                  f"{'ok' if not bad else 'BAD'}{bad}")
    v1 = worst < 1e-12
    v1b = teeth > 1e-6
    ok &= v1 and v1b
    print(f"      {'PASS' if v1 else 'FAIL'} matches the transcription; "
          f"{'PASS' if v1b else 'FAIL'} control has teeth "
          f"(worst control disagreement {teeth:.2e})")

    print("  V2  WY triangular-inverse closed form, g=0, K!=V, G>1")
    worst = 0.0
    for (B, T, H, HV, K, V) in [(2, 10, 2, 2, 16, 16), (1, 8, 1, 1, 32, 16),
                                (2, 6, 1, 2, 8, 32)]:
        q, k, v, g, beta = rnd(B, T, H, HV, K, V, seed=7)
        g = np.zeros_like(g)
        ro, rS = kda_recurrent(q, k, v, g, beta)
        oo, oS = _wy_oracle(q, k, v, g, beta)
        e, eS = _relerr(ro, oo), _relerr(rS, oS)
        worst = max(worst, e, eS)
        print(f"      B{B} T{T} H{H} HV{HV} K{K} V{V}   o {e:.3e}   S {eS:.3e}")
    v2 = worst < 1e-12
    ok &= v2
    print(f"      {'PASS' if v2 else 'FAIL'} matches the closed form")

    RESULTS["oracle"] = "PASS" if ok else "FAIL"
    print(f"  => oracle {'VERIFIED' if ok else 'SUSPECT -- treat every number below with care'}")
    return ok


# ---------------------------------------------------------------------------
# 3. the TinyGrad port
# ---------------------------------------------------------------------------
def tiny_kda_chunk(q, k, v, g, beta, chunk_size=64, scale=None, realize=True):
    """KDA chunked form in TinyGrad. Same algorithm as the reference; the
    interesting part is that the UT transform and the inter-chunk scan are both
    sequential, and a static-shape graph has to unroll both.

    `realize=False` returns the lazy Tensors so a benchmark can time graph
    construction separately from execution. See realize_noop_probe() for why
    that distinction is not optional.
    """
    from tinygrad import Tensor, dtypes
    f32 = np.float32
    F, I = dtypes.float32, dtypes.int32
    W = lambda c, a, b: Tensor.where(c, a, b)

    q = np.asarray(q, f32); k = np.asarray(k, f32)
    v = np.asarray(v, f32); g = np.asarray(g, f32); beta = np.asarray(beta, f32)
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    G = HV // H
    C = chunk_size
    NT = T // C
    assert T % C == 0, f"T={T} not divisible by C={C}"
    if scale is None:
        scale = K ** -0.5
    q = np.repeat(q, G, axis=2) * f32(scale)
    k = np.repeat(k, G, axis=2)

    ch = lambda x: x.reshape(B, NT, C, HV, -1).transpose(0, 1, 3, 2, 4)
    chs = lambda x: x.reshape(B, NT, C, HV).transpose(0, 1, 3, 2)
    tq, tk, tv = (Tensor(ch(x)) for x in (q, k, v))
    tgc, tb = Tensor(ch(g)), Tensor(chs(beta))

    g_cum = tgc.cumsum(axis=-2)                      # [B,NT,HV,C,K]

    # decayed key-key gram, strictly lower
    strict = np.tril(np.ones((C, C), f32), -1)
    Dl = ((g_cum.reshape(B, NT, HV, C, 1, K) - g_cum.reshape(B, NT, HV, 1, C, K))
          + (Tensor(strict)[None, None, None, :, :, None] - 1.0) * 1e4).exp()
    Z = (Dl * tk.reshape(B, NT, HV, C, 1, K) * tk.reshape(B, NT, HV, 1, C, K)).sum(-1)
    Btri = Z * tb.reshape(B, NT, HV, C, 1)

    # UT transform: forward substitution, unrolled. A = I + StrictTril(Btri).
    eye_all = Tensor(np.eye(C, dtype=f32))
    A = eye_all + Btri * Tensor(strict[None, None, None])
    M = Tensor(np.zeros((B, NT, HV, C, C), dtype=f32))
    eyeT = [Tensor(np.eye(C, dtype=f32)[i]) for i in range(C)]
    for i in range(C):
        b = (Tensor(np.zeros((B, NT, HV, C), dtype=f32))
             + tb[:, :, :, i].reshape(B, NT, HV, 1) * eyeT[i])
        for r in range(1, C):
            s = (A[:, :, :, r, :r] * b[:, :, :, :r]).sum(-1)
            b = b - eye_all[r].reshape(1, 1, 1, C) * s.unsqueeze(-1)
        M = M + eye_all[i].reshape(1, 1, 1, 1, C) * b.reshape(B, NT, HV, C, 1)

    gam = g_cum.exp()
    w = M @ (gam * tk)
    u = M @ tv

    # inter-chunk scan, unrolled over NT
    tS = Tensor(np.zeros((B, HV, K, V), dtype=f32))
    g_last = g_cum[:, :, :, -1]                     # [B,NT,HV,K]
    causal = np.tril(np.ones((C, C), f32))
    Dq = ((g_cum.reshape(B, NT, HV, C, 1, K) - g_cum.reshape(B, NT, HV, 1, C, K))
          + (Tensor(causal)[None, None, None, :, :, None] - 1.0) * 1e4).exp()
    parts = []
    for n in range(NT):
        gn, kn, wn, un = g_cum[:, n], tk[:, n], w[:, n], u[:, n]
        qgn = gam[:, n] * tq[:, n]
        v_new = un - wn @ tS
        Aqk = ((Dq[:, n].reshape(B, HV, C, C, K)
               * tq[:, n].reshape(B, HV, C, 1, K)
               * kn.reshape(B, HV, 1, C, K)).sum(-1))
        parts.append(qgn @ tS + Aqk @ v_new)
        gC = g_last[:, n]
        tS = tS * gC.exp().unsqueeze(-1) + \
            ((gC.reshape(B, HV, 1, K) - gn.reshape(B, HV, C, K)).exp()
             * kn.reshape(B, HV, C, K)).permute(0, 1, 3, 2) @ v_new

    o = Tensor.stack(*parts)                        # [NT,B,HV,C,V]
    o = o.permute(1, 0, 3, 2, 4).reshape(B, T, HV, V)
    if not realize:
        return o, tS
    return o.numpy(), tS.numpy()


def realize_noop_probe():
    """Demonstrate the gotcha that makes naive tinygrad benchmarks lie.

    Tensor.realize() on an already-realized graph is a SILENT NO-OP: no kernels
    are compiled, no kernels are launched, no time passes. So timing

        t0 = now(); out.realize(); t = now() - t0        # meaningless

    reports ~0 for a real forward pass if the graph was already realized by a
    warmup iteration. Every benchmark below therefore rebuilds the graph from
    numpy inside the timed region, and counts the kernels each realize emits so
    the two cases can be told apart by more than a clock.
    """
    hr("realize() is a no-op on a realized graph (why the graph is rebuilt)")
    from tinygrad import Device, Tensor
    dev = Device[Device.DEFAULT]
    comp = getattr(dev, "compiler", None)
    orig = getattr(comp, "compile_cached", None) if comp is not None else None
    seen = []
    if orig is not None:
        def spy(src, *a, **k):
            seen.append(1)
            return orig(src, *a, **k)
        comp.compile_cached = spy
    try:
        a = (Tensor([1.0, 2.0, 3.0, 4.0]) * 2.0)
        sync()
        t0 = time.perf_counter(); a.realize(); sync(); t_first = time.perf_counter() - t0
        n_first = len(seen)
        t0 = time.perf_counter(); a.realize(); sync(); t_again = time.perf_counter() - t0
        n_again = len(seen) - n_first
        b = (Tensor([1.0, 2.0, 3.0, 4.0]) * 2.0)        # a genuinely new graph
        sync()
        t0 = time.perf_counter(); b.realize(); sync(); t_fresh = time.perf_counter() - t0
        n_fresh = len(seen) - n_first - n_again
    finally:
        if orig is not None:
            comp.compile_cached = orig

    print(f"  first realize()   {t_first*1e3:8.3f} ms   kernels compiled: {n_first}")
    print(f"  realize() again    {t_again*1e3:8.3f} ms   kernels compiled: {n_again}"
          f"   <- no-op, this is the trap")
    print(f"  fresh graph       {t_fresh*1e3:8.3f} ms   kernels compiled: {n_fresh}")
    print("  (a 'fresh graph' compiling 0 means tinygrad's on-disk cache already had")
    print("   the kernel, so the compile count is not the signal -- the TIME is:")
    print(f"   {t_fresh/max(t_again, 1e-12):.0f}x more than re-realizing, for identical work.)")
    print("  => every timing below rebuilds the graph inside the timed region.")
    RESULTS["realize_noop"] = (t_again, t_fresh)


# ---------------------------------------------------------------------------
# 4. fla / triton reference  (optional)
# ---------------------------------------------------------------------------
def _fla_chunk_kda():
    if dep("fla") is None:
        raise ImportError(dep_err("fla") or "fla unavailable")
    from fla.ops.kda import chunk_kda
    return chunk_kda


def run_fla(q, k, v, g, beta, dtype):
    """flash-linear-attention's KDA, via its public layer wrapper."""
    torch = dep("torch")
    if torch is None:
        raise ImportError(dep_err("torch"))
    chunk_kda = _fla_chunk_kda()
    t = lambda x: torch.tensor(np.ascontiguousarray(x), dtype=dtype, device="cuda")
    with torch.no_grad():
        o, _ = chunk_kda(t(q), t(k), t(v), t(g), t(beta), use_qk_l2norm_in_kernel=False)
    return o.float().cpu().numpy()


# ---------------------------------------------------------------------------
# 5. timing harness.  Every reported millisecond figure is decomposed.
# ---------------------------------------------------------------------------
def is_oom(e):
    """Recognise an OOM without importing torch.cuda.OutOfMemoryError, which
    does not exist when torch is missing -- referencing it in an `except`
    clause is itself a NameError."""
    if type(e).__name__ in ("OutOfMemoryError", "CUDAOutOfMemoryError"):
        return True
    return "out of memory" in str(e).lower()


def empty_cache():
    torch = dep("torch")
    if torch is not None:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


class Stage:
    """build / schedule+issue / kernel(waited) / copy, in ms."""
    __slots__ = ("build", "issue", "kernel", "copy", "total", "kernels")

    def __init__(self):
        self.build = self.issue = self.kernel = self.copy = float("nan")
        self.total = float("nan")
        self.kernels = None

    def fmt(self):
        k = "" if self.kernels is None else f"  [{self.kernels} kernels compiled]"
        return (f"{self.total:9.1f} = build {self.build:8.1f} + issue {self.issue:8.1f}"
                f" + kernel {self.kernel:8.1f} + copy {self.copy:6.1f}{k}")


def time_tinygrad(q, k, v, g, beta, C=64):
    """One instrumented TinyGrad call. The graph is built fresh here, inside
    the timed region -- see realize_noop_probe()."""
    s = Stage()
    sync()
    t0 = time.perf_counter()
    o, S = tiny_kda_chunk(q, k, v, g, beta, chunk_size=C, realize=False)
    t1 = time.perf_counter()
    n_before = len(KERNEL_COMPILES)
    o.realize(); S.realize()
    t2 = time.perf_counter()
    sync()
    t3 = time.perf_counter()
    out_o, out_S = o.numpy(), S.numpy()
    t4 = time.perf_counter()
    s.build = (t1 - t0) * 1e3
    s.issue = (t2 - t1) * 1e3
    s.kernel = (t3 - t2) * 1e3
    s.copy = (t4 - t3) * 1e3
    s.total = (t4 - t0) * 1e3
    s.kernels = len(KERNEL_COMPILES) - n_before
    return s, out_o, out_S


def time_fla(q, k, v, g, beta, dtype):
    torch = dep("torch")
    chunk_kda = _fla_chunk_kda()
    t = lambda x: torch.tensor(np.ascontiguousarray(x), dtype=dtype, device="cuda")
    s = Stage()
    sync(); t0 = time.perf_counter()
    tq, tk, tv, tg, tb = t(q), t(k), t(v), t(g), t(beta)
    sync(); t1 = time.perf_counter()
    with torch.no_grad():
        o, _ = chunk_kda(tq, tk, tv, tg, tb, use_qk_l2norm_in_kernel=False)
    sync(); t2 = time.perf_counter()
    out = o.float().cpu().numpy()
    t3 = time.perf_counter()
    s.build = (t1 - t0) * 1e3
    s.issue = s.kernel = (t2 - t1) * 1e3      # one fused launch; cannot separate
    s.copy = (t3 - t2) * 1e3
    s.total = (t3 - t0) * 1e3
    return s, out


def bench(fn, warmup=1, iters=3):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


# ---------------------------------------------------------------------------
# 6. correctness
# ---------------------------------------------------------------------------
def correctness():
    hr("correctness -- both against the float64 per-token recurrence")
    B, T, H, HV, K, V = 2, 64, 2, 2, 32, 32
    q, k, v, g, beta = rnd(B, T, H, HV, K, V, seed=0)
    ref, Sref = kda_recurrent(q, k, v, g, beta)

    def rel(a):
        return _relerr(a, ref)

    try:
        o, S = tiny_kda_chunk(q, k, v, g, beta, chunk_size=64)
        print(f"  tinygrad port      relerr o {rel(o):.3e}   S {_relerr(S, Sref):.3e}")
        RESULTS["tiny_err"] = rel(o)
    except Exception as e:
        print(f"  tinygrad port      FAILED {type(e).__name__}: {str(e)[:90]}")
        RESULTS["tiny_err"] = None

    if dep("torch") is None:
        print(f"  fla triton         SKIP: {dep_err('torch')}")
    elif not has_torch_cuda():
        print("  fla triton         SKIP: torch is installed but no CUDA device is visible")
    else:
        torch = dep("torch")
        for dt, name in ((torch.bfloat16, "bf16"), (torch.float16, "fp16")):
            try:
                o = run_fla(q, k, v, g, beta, dt)
                print(f"  fla triton {name}    relerr {rel(o):.3e}")
            except Exception as e:
                print(f"  fla triton {name}    FAILED {type(e).__name__}: {str(e)[:90]}")


# ---------------------------------------------------------------------------
# 7. arithmetic intensity.  Analytic, arch-independent, and the reason the
#    two implementations cannot converge.
# ---------------------------------------------------------------------------
def mib(n):
    return n / 2 ** 20


# published specs: (dense bf16/fp16 tensor-core TFLOP/s, HBM/GDDR GB/s).
# sm_75 (T4) has fp16 tensor cores but no bf16, so its ridge is the fp16 one.
RIDGE = {
    "T4":        (65e12, 320e9,  "sm_75, fp16 TC (no bf16)"),
    "P100":      (112e12, 732e9, "sm_60, fp16 TC (no bf16)"),
    "A100":      (312e12, 1555e9, "sm_80"),
    "A10G":      (125e12, 600e9, "sm_86"),
    "L4":        (121e12, 300e9, "sm_89"),
    "V100":      (125e12, 900e9, "sm_70, fp16 TC (no bf16)"),
    "H100 SXM":  (989.5e12, 3350e9, "sm_90"),
    "H100 PCIe": (756e12, 2000e9, "sm_90"),
    "H100":      (989.5e12, 3350e9, "sm_90, generic fallback"),
    "RTX 4090":  (165.2e12, 1008e9, "sm_89"),
}


def ridge_for(devname):
    for key in sorted(RIDGE, key=len, reverse=True):
        if key.lower() in (devname or "").lower():
            f, b, note = RIDGE[key]
            return f, b, f / b, f"{key} ({note})"
    return None


def cost_model(B, T, HV, K, V, C=64, esize=2):
    """FLOPs and DRAM traffic per step, for one (batch, chunk, head) block of
    the chunked KDA forward. Per-(b,n,h) FLOPs are scaled by B*NT*HV; bytes are
    for the whole batch and count one write + one read of each intermediate,
    which is the optimistic case for a fused kernel and the realistic case for
    a naive one.  esize = bytes per element in the kernel's working dtype.
    """
    NT = T // C
    BNH = B * NT * HV
    e_g = B * T * HV * K
    rows = []

    def add(step, shape, flops_bn, byts, cls, note="", ref_only=False):
        rows.append(dict(step=step, shape=shape, flops=flops_bn * BNH,
                         bytes=byts, cls=cls, note=note, ref_only=ref_only))

    add("0  gate act. -exp(A_log)*softplus(g+dt)", "[B,T,HV,K]", 0,
        3 * 2 * e_g * esize, "elementwise", "exp, softplus, 2 fma")
    add("1  g_cum = cumsum(g) over chunk", "[B,T,HV,K]", 0,
        2 * 2 * e_g * esize, "scan", "C-long axis, not T")
    add("2a D = exp(gc[i]-gc[j]) masked", "[B,NT,HV,C,C,K]", 0,
        2 * BNH * C * C * K * esize, "elementwise",
        "THE C^2K TRANSIENT. ref only; kernels fold into 2b", True)
    add("2b Z = (K . D) K^T", "[B,NT,HV,C,C]", 2 * C * C * K,
        2 * BNH * C * C * esize, "matmul", "the k-vs-k gram, tensor-coreable")
    add("3  UT: fwd substitution + I, *Diag(b)", "[B,NT,HV,C,C]", C * C * (C - 1),
        2 * BNH * C * C * esize, "SEQUENTIAL", "C-1 dependent row updates, not a matmul")
    add("4  w = M (Gamma . K)", "[B,NT,HV,C,K]", 2 * C * C * K,
        2 * BNH * C * K * esize, "matmul", "tensor-coreable")
    add("5  u = M V", "[B,NT,HV,C,V]", 2 * C * C * V,
        2 * BNH * C * V * esize, "matmul", "tensor-coreable")
    add("6a v_new = u - w S", "[B,NT,HV,C,V]", 2 * C * K * V,
        3 * B * HV * C * V * esize, "matmul", "S from the accumulator")
    add("6b Aqk = (Q . D) K^T causal", "[B,NT,HV,C,C]", 2 * C * C * K,
        2 * BNH * C * C * esize, "matmul", "tensor-coreable")
    add("6c o = (Gamma . Q) S", "[B,T,HV,V]", 2 * C * K * V,
        2 * B * T * HV * V * esize, "matmul", "tensor-coreable")
    add("6d o += Aqk v_new", "[B,T,HV,V]", 2 * C * C * V, 0, "matmul",
        "free only if fused into 6c")
    add("6e S = Diag(gam^C) S + (K . D)^T v_new", "[B,HV,K,V]", 2 * C * K * V,
        2 * B * HV * K * V * 4, "matmul+ew", "state is T-independent, registers")
    return rows, NT


def decay_ratio_block(B, T, HV, K, C=64, BC=16, esize=2):
    """The study's central finding, and the only number here that does not
    depend on which GPU you run it on.

    TinyGrad's port must materialise the decayed key-key gram as
    [B,NT,HV,C,C,K] because the graph is elementwise-then-contract over the
    full C x C block. fla's Triton kernel holds one reference row per
    sub-chunk, [BC,K], applies exp2 of the shifted difference to the matmul
    operand as it is loaded, and lets the [BC,BC] score block die in registers.
    The C^2K tensor is never named, so it is never allocated, so it costs zero
    bytes of DRAM.
    """
    NT = T // C
    NC = C // BC                      # sub-chunks per (b, chunk, head)
    mine = B * NT * HV * C * C * K     # whole batch, tinygrad
    grid = B * NT * HV * NC * BC * K   # whole grid working set, triton
    return dict(NT=NT, NC=NC, mine=mine, grid=grid, bytes=mine * esize,
                prog_tg=C * C * K, prog_fla=BC * K,
                per_prog_ratio=(C * C * K) / (BC * K),
                whole_grid_ratio=mine / grid)


def intensity_report(devname):
    hr("arithmetic intensity -- FLOPs and bytes, and the roofline")
    print("  Byte counts are analytic: one write + one read of each intermediate,")
    print("  in the kernel's working dtype. Optimistic for a fused kernel, realistic")
    print("  for a naive one. They do not depend on the GPU.\n")

    shapes = [(4, 4096, 32, 128, 128, 64, "study default"),
              (4, 1024, 8, 64, 64, 64, "throughput row 2"),
              (2, 512, 4, 64, 64, 64, "throughput row 1")]
    r = ridge_for(devname)
    for (B, T, HV, K, V, C, tag) in shapes:
        rows, NT = cost_model(B, T, HV, K, V, C)
        krows = [x for x in rows if not x["ref_only"]]
        F = sum(x["flops"] for x in rows)
        Bts = sum(x["bytes"] for x in krows)
        Bts_all = sum(x["bytes"] for x in rows)
        floor = (B * T * HV * (2 * K + V + K) + B * T * HV * V) * 2   # read q,k,v,g write o
        print(f"  --- B{B} T{T} HV{HV} K{K} V{V} C{C}  NT{NT}   [{tag}] ---")
        print(f"  {'step':<38}{'GFLOP':>8}{'MB':>9}{'FLOP/B':>9}  class")
        top = sorted(rows, key=lambda y: -y["bytes"])[:6]
        for x in top:
            ai = x["flops"] / x["bytes"] if x["bytes"] else float("inf")
            mark = "  (never built by a kernel)" if x["ref_only"] else ""
            print(f"  {x['step']:<38}{x['flops']/1e9:8.2f}{mib(x['bytes']):9.1f}"
                  f"{ai:9.1f}  {x['cls']}{mark}")
        rest = [x for x in rows if x not in top]
        if rest:
            print(f"  {'+%d smaller steps' % len(rest):<38}"
                  f"{sum(x['flops'] for x in rest)/1e9:8.2f}"
                  f"{mib(sum(x['bytes'] for x in rest)):9.1f}"
                  f"{(sum(x['flops'] for x in rest)/sum(x['bytes'] for x in rest) if sum(x['bytes'] for x in rest) else 0):9.1f}")
        print(f"  {'TOTAL (kernels only)':<38}{F/1e9:8.2f}{mib(Bts):9.1f}{F/Bts:9.1f}")
        print(f"  {'  + the C^2K transient if materialised':<38}{'':>8}"
              f"{mib(Bts_all-Bts):9.1f}{'':>9}   <-- what the port must move")
        print(f"  {'ABSOLUTE FLOOR (read q,k,v,g write o)':<38}{F/1e9:8.2f}"
              f"{mib(floor):9.1f}{F/floor:9.1f}   even perfect fusion gets here")
        if r:
            fl, bw, rp, lbl = r
            print(f"  ridge point on {lbl}: {fl/1e12:.0f} TFLOP/s / {bw/1e9:.0f} GB/s"
                  f" = {rp:.0f} FLOP/byte")
            print(f"    -> kernels-only intensity is {rp/(F/Bts):.1f}x BELOW the ridge:"
                  f" memory bound, and even the floor is {rp/(F/floor):.1f}x below it")
        else:
            ref = " / ".join(f"{k.split()[0]} {RIDGE[k][0]/RIDGE[k][1]:.0f}"
                             for k in ("H100 SXM", "A100", "T4"))
            print(f"  ridge point: unknown for '{devname}'; for reference {ref}"
                  f" FLOP/byte.")
            print(f"    -> at {F/Bts:.0f} FLOP/byte this operator is memory bound on every"
                  f" GPU on that list, by {295/(F/Bts):.0f}x or more")
        print()

    # ---- the headline, arch-independent -------------------------------
    hr("THE DECAY INTERMEDIATE -- the central finding")
    B, T, HV, K, C, BC = 4, 4096, 32, 128, 64, 16
    d = decay_ratio_block(B, T, HV, K, C, BC)
    print(f"  at B{B} T{T} HV{HV} K{K} C{C} (fla's BT={C}, BC={BC}, NC={d['NC']}), bf16:")
    print(f"    tinygrad  [B,NT,HV,C,C,K]      = {d['mine']:,} elements"
          f" = {d['bytes']/2**30:.2f} GiB, whole batch")
    print(f"    triton    [BC,K] per sub-chunk = {BC*K:,} elements, registers only")
    print(f"    triton    whole-grid set       = {d['grid']:,} elements"
          f" = {d['grid']*2/2**30:.3f} GiB, all of it L1/registers")
    print()
    print(f"    >>> RATIO  {d['per_prog_ratio']:.0f}x  <<<")
    print()
    print(f"    derivation, and it is a property of the two programs, not the GPU:")
    print(f"      tinygrad, per (b, chunk, head) : C  x C  x K = {C}x{C}x{K} = {d['prog_tg']:,}")
    print(f"      triton,  per sub-chunk         : BC x K       = {BC}x{K} = {d['prog_fla']:,}")
    print(f"      C^2 / BC = {C*C}/{BC} = {d['per_prog_ratio']:.0f}")
    print(f"      Whole grid, the triton side never exceeds B*NT*HV*NC*BC*K = B*NT*HV*C*K")
    print(f"      because exp2 of the shifted difference is applied to the operand as it")
    print(f"      is loaded, so the ratio is at least {d['whole_grid_ratio']:.0f}x even there.")
    print(f"      {d['per_prog_ratio']:.0f}x is the tight per-program figure: the one to quote.")
    print()
    print(f"    the same ratio at every shape in this run (it does not move):")
    for (B, T, HV, K, V, C, tag) in shapes:
        d2 = decay_ratio_block(B, T, HV, K, C)
        print(f"      B{B} T{T} HV{HV} K{K} C{C}  ->  {d2['per_prog_ratio']:.0f}x"
              f"   ({d2['bytes']/2**30:.2f} GiB materialised)")
    RESULTS["decay_ratio"] = d["per_prog_ratio"]
    RESULTS["decay_gib"] = d["bytes"] / 2 ** 30
    rows, NT = cost_model(4, 4096, 32, 128, 128, 64)
    F = sum(x["flops"] for x in rows)
    Bts = sum(x["bytes"] for x in rows if not x["ref_only"])
    RESULTS["study_intensity"] = F / Bts
    RESULTS["study_gflop"] = F / 1e9
    RESULTS["study_mib"] = mib(Bts)
    if r:
        RESULTS["ridge"] = r[2]


# ---------------------------------------------------------------------------
# 8. throughput
# ---------------------------------------------------------------------------
def peak_intermediate_bytes(B, T, HV, K, C=64, esize=4, live=4):
    """Bytes the port's largest transient needs, and the peak while it is
    built. The decay chain diff -> mask-add -> exp -> scale-by-k is written
    straight-line, so `live` of those [B,NT,HV,C,C,K] buffers coexist."""
    NT = T // C
    one = B * NT * HV * C * C * K * esize
    return one, one * live


def throughput():
    hr("throughput -- same GPU, same shapes, both implementations")
    dev = device_name()
    cuda = is_cuda()
    print(f"  device {dev}   backend {backend_name()}")
    print("  every ms is  total = build + issue + kernel + copy")
    print("    build  = python graph construction, no device work")
    print("    issue  = realize(): the scheduler linearises the unrolled graph, then")
    print("             launches. On CUDA the launches are async, so issue is CPU time")
    print("             and is usually dominated by tinygrad's python scheduler.")
    print("    kernel = time spent waiting for the device after that sync point")
    print("    copy   = .numpy(), the device-to-host readback")
    print("  the graph is rebuilt inside the timed region; realize() on an")
    print("  already-realized graph is a silent no-op -- see the probe above\n")

    # includes the shapes NOTES.md analyses, where the port's [B,NT,HV,C,C,K]
    # intermediate is the whole problem
    if cuda:
        shapes = [(2, 512, 4, 4, 64, 64), (4, 1024, 8, 8, 64, 64),
                  (4, 4096, 16, 16, 128, 128)]
    else:
        # no CUDA: the port's Python-side build costs ~20 s per call whatever
        # the shape (the UT transform unrolls C=64 regardless), so one small
        # shape that still exercises G>1 and K!=V, and nothing else.
        shapes = [(2, 64, 2, 4, 32, 16)]
    print(f"  {'shape':<26}{'1 interm':>10}{'peak':>9}{'fla ms':>9}{'tiny ms':>9}{'ratio':>8}")
    RESULTS["shapes"] = []
    free = free_memory_bytes()

    for (B, T, H, HV, K, V) in shapes:
        C = 64
        one, peak = peak_intermediate_bytes(B, T, HV, K, C)
        name = f"B{B} T{T} HV{HV} K{K} V{V}"
        print(f"  {name:<26}{mib(one):9.0f}M{mib(peak):8.0f}M")

        q, k, v, g, beta = rnd(B, T, H, HV, K, V, seed=0, dtype=np.float32)

        t_fla, s_fla = float("nan"), None
        if dep("torch") is None:
            print(f"      fla:    SKIP: {dep_err('torch')}")
        elif not has_torch_cuda():
            print("      fla:    SKIP: torch present but no CUDA device visible")
        else:
            try:
                dt = dep("torch").bfloat16
                s_fla, _ = time_fla(q, k, v, g, beta, dt)     # compile / autotune
                t_fla = bench(lambda: run_fla(q, k, v, g, beta, dt),
                              warmup=0, iters=3)
                empty_cache()
            except Exception as e:
                if is_oom(e):
                    print("      fla:    OUT OF MEMORY")
                    empty_cache()
                else:
                    print(f"      fla:    SKIP: {type(e).__name__}: {str(e)[:64]}")

        # a predicted OOM is a data point, not a crash. Re-query free memory
        # here rather than reusing the value from the top of the function:
        # fla's run above may have taken a chunk of it.
        free = free_memory_bytes()
        pred = ""
        if free and peak > 0.7 * free:
            pred = (f"PREDICTED OOM: peak {mib(peak):.0f} MB > 70% of free "
                    f"{mib(free):.0f} MB -- skipped, not attempted")

        t_tg, s_tg = float("nan"), None
        if pred:
            print(f"      tiny:   {pred}")
        else:
            try:
                s_tg, _, _ = time_tinygrad(q, k, v, g, beta, C)
                if cuda:
                    t_tg = bench(lambda: tiny_kda_chunk(q, k, v, g, beta, chunk_size=C),
                                 warmup=0, iters=2)
                else:
                    t_tg = s_tg.total
                empty_cache()
            except Exception as e:
                if is_oom(e):
                    print(f"      tiny:   OUT OF MEMORY (needs {mib(one):.0f} MB for the"
                          f" gram alone, {mib(peak):.0f} MB peak)")
                    empty_cache()
                else:
                    print(f"      tiny:   FAILED {type(e).__name__}: {str(e)[:64]}")

        if t_tg == t_tg:
            if s_tg:
                print(f"      tiny:   {s_tg.fmt()}   [1 instrumented run]")
                if cuda and abs(t_tg - s_tg.total) > 0.05 * t_tg:
                    print(f"              (the table row below is best-of-N at {t_tg:.1f} ms;"
                          f" the split is from this one run)")
                if not cuda:
                    print("              (non-CUDA backend: 'issue' and 'kernel' cannot be"
                          " separated, and 'kernel' still contains CPU work)")
        if t_fla == t_fla and s_fla:
            print(f"      fla:    {s_fla.fmt()}")
        r = t_tg / t_fla if t_fla == t_fla and t_tg == t_tg else float("nan")
        fs = f"{t_fla:9.1f}" if t_fla == t_fla else f"{'--':>9}"
        if r == r:
            print(f"  {'':<26}{'':>10}{'':>9}{fs}{t_tg:9.1f}{r:7.1f}x")
        else:
            print(f"  {'':<26}{'':>10}{'':>9}{fs}"
                  f"{('OOM' if pred else 'n/a'):>9}{'--':>8}")
        RESULTS["shapes"].append(dict(name=name, one=one, peak=peak, fla=t_fla,
                                     tiny=t_tg, pred=bool(pred),
                                     kernels=(s_tg.kernels if s_tg else None)))
        del q, k, v, g, beta
        empty_cache()


# ---------------------------------------------------------------------------
# 9. generated code.  CUDA only: on any other backend this is a no-op.
# ---------------------------------------------------------------------------
def codegen():
    """Dump TinyGrad's generated source for the step that decides whether KDA
    is memory- or compute-bound, to out/codegen/, so it can be diffed against
    a later run or against the Triton source by hand."""
    hr("generated code -- the decayed key-key gram")
    B, Tt, H, HV, K, V, C = 1, 64, 1, 1, 64, 64, 64
    NT = Tt // C
    d = decay_ratio_block(B, Tt, HV, K, C)

    if not is_cuda():
        print("  SKIP: backend is not CUDA -- no CUDA source is generated, nothing to")
        print("        capture. On a CUDA box this writes every kernel tinygrad emits")
        print(f"        for the gram below into {os.path.relpath(CODEGEN_DIR, HERE)}/ and")
        print("        prints the per-kernel shape, line count and MMA usage, which is")
        print("        what codegen_diff.py compares against the Triton source.")
    else:
        try:
            from tinygrad import Tensor, Device
            q, k, v, g, beta = rnd(B, Tt, H, HV, K, V, seed=0)
            gch = np.asarray(g, np.float32).reshape(B, NT, C, HV, K).transpose(0, 1, 3, 2, 4)
            kch = np.asarray(k, np.float32).reshape(B, NT, C, HV, K).transpose(0, 1, 3, 2, 4)
            gcum = Tensor(np.cumsum(gch, axis=-2))
            tk = Tensor(kch)
            strict = np.tril(np.ones((C, C), np.float32), -1)
            decay = ((gcum.reshape(B, NT, HV, C, 1, K) - gcum.reshape(B, NT, HV, 1, C, K))
                     + (Tensor(strict)[None, None, None, :, :, None] - 1.0) * 1e4).exp()
            gram = (decay * tk.reshape(B, NT, HV, C, 1, K)
                    * tk.reshape(B, NT, HV, 1, C, K)).sum(-1)

            dev = Device[Device.DEFAULT]
            comp = dev.compiler
            captured = []
            orig = comp.compile_cached
            def spy(src, *a, **k):
                captured.append(str(src))
                return orig(src, *a, **k)
            comp.compile_cached = spy
            try:
                gram.realize()
            finally:
                comp.compile_cached = orig

            os.makedirs(CODEGEN_DIR, exist_ok=True)
            tag = f"b{B}_t{Tt}_hv{HV}_k{K}_v{V}_c{C}"
            index = []
            print(f"  captured {len(captured)} kernel(s) -> "
                  f"{os.path.relpath(CODEGEN_DIR, HERE)}/")
            for i, src in enumerate(captured):
                body = re.sub(r"\x1b\[[0-9;]*m", "", src)
                nlines = len([l for l in body.split("\n") if l.strip()])
                has_mma = any(t in body for t in ("mma.sync", "wmma", "mma.", "HMMA",
                                                  "tensor_core", "dot4"))
                fn = f"{i:02d}_gram_{tag}.cu"
                with open(os.path.join(CODEGEN_DIR, fn), "w") as f:
                    f.write(f"// tinygrad generated source for the decayed k-k gram\n")
                    f.write(f"// backend: {backend_name()}\n")
                    f.write(f"// B{B} T{Tt} HV{HV} K{K} V{V} C{C} NT{NT}\n")
                    f.write(f"// kernel {i} of {len(captured)}, {nlines} non-blank lines,"
                            f" tensor-core MMA: {has_mma}\n")
                    f.write("// device: " + device_name() + "\n\n")
                    f.write(body)
                index.append((fn, nlines, has_mma, len(body)))
                print(f"    {fn:<44}{nlines:>5} lines   MMA: {has_mma}")
            with open(os.path.join(CODEGEN_DIR, "INDEX.txt"), "w") as f:
                f.write("tinygrad CUDA source for the KDA decayed key-key gram\n")
                f.write(f"device: {device_name()}\n")
                f.write(f"backend: {backend_name()}\n")
                for fn, nlines, has_mma, nbytes in index:
                    f.write(f"{fn}\t{nlines} lines\t{nbytes} bytes\tMMA={has_mma}\n")
            if not captured:
                print("    (nothing captured: the graph was already realized and the")
                print("     kernels came from tinygrad's on-disk cache -- delete")
                print("     ~/.cache/tinygrad and re-run to force codegen)")
        except Exception as e:
            print(f"  SKIP: codegen capture failed: {type(e).__name__}: {str(e)[:80]}")

    print()
    print("  hand-written Triton, same step, from fla/ops/kda/chunk_intra.py")
    print("    BT=64, BC=16, NC=4, grid=(NT, NC, B*HV)")
    print("    b_gqn  = tl.where(m, exp2(b_g1 - b_gn1[None,:]), 0)     [BC, K]")
    print("    b_kgt  = tl.trans(b_k0 * exp2(b_gn1[None,:] - b_g0))   [K, BC]")
    print("    b_Akk  = tl.dot(b_k1 * b_gqn, b_kgt, b_Akk)           [BC,BC]  MMA")
    print("    ^ one reference row per SUB-CHUNK, so the decay tensor is [16, K],")
    print("      and the [16,16] score block never leaves registers.")
    print(f"  decay tensor, per program: tinygrad {d['prog_tg']:,} vs triton"
          f" {d['prog_fla']:,} elements  ->  {d['per_prog_ratio']:.0f}x")


# ---------------------------------------------------------------------------
# 10. the 30-second summary
# ---------------------------------------------------------------------------
def summary():
    hr("SUMMARY -- 30 seconds")
    dev = device_name()
    print(f"  device            {dev}  ({backend_name()}, {compute_capability()})")
    got_fla = dep("fla") is not None and has_torch_cuda()
    print(f"  measured          tinygrad port: YES   fla triton: "
          f"{'YES' if got_fla else 'no (see the SKIP lines above)'}")
    print(f"  oracle            {RESULTS.get('oracle', '?')}  "
          f"(kda_recurrent vs 2 independent closed forms)")
    if RESULTS.get("tiny_err") is not None:
        print(f"  tinygrad vs f64   relerr {RESULTS['tiny_err']:.3e}")

    if RESULTS.get("shapes"):
        print()
        for s in RESULTS["shapes"]:
            tg = f"{s['tiny']:.1f} ms" if s["tiny"] == s["tiny"] else \
                 ("predicted OOM" if s["pred"] else "n/a")
            tf = f"{s['fla']:.1f} ms" if s["fla"] == s["fla"] else "n/a"
            print(f"    {s['name']:<26} tinygrad {tg:<14} fla {tf}")
    if KERNEL_COMPILES:
        print(f"    distinct kernels tinygrad compiled for the port: {len(KERNEL_COMPILES)}"
              f"   (0 or small here means tinygrad's on-disk cache already had them;")
        print(f"    on a fresh machine this is what the port's graph decomposes into,"
              f" against fla's handful of persistent kernels)")
    if RESULTS.get("realize_noop"):
        again, fresh = RESULTS["realize_noop"]
        print(f"    realize() on a realized graph: {again*1e3:.3f} ms vs {fresh*1e3:.3f} ms"
              f" for a fresh graph ({fresh/max(again,1e-12):.0f}x) -- so every")
        print(f"    timing here rebuilds the graph, or it would measure the no-op")

    print()
    if RESULTS.get("decay_ratio"):
        print(f"  HEADLINE  the decay intermediate: {RESULTS['decay_ratio']:.0f}x")
        print(f"            tinygrad materialises [B,NT,HV,C,C,K] = "
              f"{RESULTS.get('decay_gib', 0):.2f} GiB at the study shape;")
        print(f"            triton holds [BC,K] per sub-chunk in registers and the")
        print(f"            tensor is never named, so it costs 0 bytes of DRAM.")
        print(f"            arch-independent -- a property of the two programs.")
    else:
        print("  HEADLINE  not available: the intensity stage did not complete, so")
        print("            the decay-intermediate ratio was never computed.")
    if RESULTS.get("study_intensity"):
        print()
        print(f"  roofline  study shape (B4 T4096 HV32 K=V=128 C64): "
              f"{RESULTS['study_gflop']:.1f} GFLOP against")
        print(f"            {RESULTS['study_mib']:.0f} MB = "
              f"{RESULTS['study_intensity']:.1f} FLOP/byte.")
        if RESULTS.get("ridge"):
            print(f"            ridge on this GPU ~{RESULTS['ridge']:.0f} FLOP/byte"
                  f"  ->  ~{RESULTS['ridge']/RESULTS['study_intensity']:.0f}x memory bound,")
        else:
            lo = min(RIDGE[k][0] / RIDGE[k][1] for k in RIDGE)
            print(f"            lowest ridge of any GPU in the table {lo:.0f} FLOP/byte"
                  f"  ->  at least {lo/RESULTS['study_intensity']:.0f}x memory bound,")
        print(f"            so the gap between the two implementations is a bandwidth")
        print(f"            gap, not a FLOP gap, and no amount of better matmul codegen")
        print(f"            closes it.")
    print()
    print("  the whole comparison turns on that one ratio. Everything else -- the")
    print("  build time, the scheduler, the kernel time -- is a second-order effect")
    print("  on top of a number that was decided before either program ran.")


# ---------------------------------------------------------------------------
def main():
    print(__doc__)
    probe_deps()
    install_kernel_counter()
    stages = [
        ("environment", env_report),
        ("oracle", verify_oracle),
        ("realize probe", realize_noop_probe),
        ("correctness", correctness),
        ("intensity", lambda: intensity_report(device_name())),
        ("throughput", throughput),
        ("codegen", codegen),
    ]
    failed = []
    for name, fn in stages:
        try:
            fn()
        except Exception:
            # one bad stage must not take the summary down
            failed.append(name)
            traceback.print_exc()
    try:
        summary()
    except Exception:
        failed.append("summary")
        traceback.print_exc()
    print()
    hr("done")
    if failed:
        print(f"  stages that raised (everything else completed): {', '.join(failed)}")
    else:
        print("  all stages completed")


if __name__ == "__main__":
    main()
