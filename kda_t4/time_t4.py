#!/usr/bin/env python3
"""
time_t4.py -- the wall clock, with the scope on the same line as the number.

THE RULE THIS FILE EXISTS TO ENFORCE
------------------------------------
`notes/06-corrections-ledger.md` C-04: *"A measurement on one backend is not a
property of the compiler. C-04 exists because '27% wrong' was measured on Apple
hardware and then described as a codegen property. The description outlived the
measurement's scope because nothing recorded the scope. Record the scope with
the number."*

So `fla` rows carry a scope cell, and the scope is not a sentence afterwards.
It is assembled by `kda_t4/gate_observed.py` from objects read off this card in
this session, and this file REFUSES to print an fla number if that scope is not
in hand.  That refusal is the rule; the column is just the receipt.

WHY THE SIDE IS WHAT IT IS
--------------------------
fla is called as `chunk_kda(..., output_final_state=False)`, which returns
`(o, None)`.  So fla's work in this measurement is exactly "produce o" and
nothing else.  The TinyGrad ports return `(o, S)` as numpy, so a naive
comparison would charge TinyGrad for materialising a final state the reference
was never asked to produce.  Therefore:

  * the DEVICE number is CUDA events around realising `o` only
  * the cost of realising `S` is measured and reported separately
  * the HOST number is the unmodified end-to-end call, build + realise + read

Getting at `o` as a Tensor requires intercepting `Tensor.numpy`, because the
ports call it internally (`kda/tiny_kda.py:252`, `tiny_kda_opt.py:379`,
`tiny_kda_faithful.py:900`).  The interception is a local monkeypatch restored
immediately, it returns a zero array of the right shape so the caller cannot
tell, and it is reverted before anything else runs.  No file in `kda/` is
modified.  The same port is then called again with the patch removed to get the
real output for the agreement check, so the number being timed and the number
being checked come from the same unmodified code path.

A NOTE ON `kda/kda_vs_fla.py:640-661`
--------------------------------------
That file's own `time_tinygrad` calls `tiny_kda_chunk(..., realize=False)`.  No
port accepts a `realize` kwarg -- they all end in `return o.numpy(), tS.numpy()`
-- so that call raises `TypeError` against the ports as they stand.  Reported,
not fixed: `kda/` is read-only for this work.  It means this study has no
pre-existing working instrument for the TinyGrad device time either, which is
part of why this file exists rather than reusing one.

AND THE LEDGER'S OTHER RULE
---------------------------
`notes/06` §5b: two comparisons in this project were invalidated by porting the
wrong thing while passing every test.  So the agreement check is printed NEXT
TO the timings and explicitly NOT offered as evidence that the baseline was the
right artifact.  It is evidence about the math.  Which artifact ran is
established by `kda_t4/pin_t4.py`, from the registry, separately.

Run:  python kda_t4/time_t4.py [--largest-only] [--iters N]
"""

from __future__ import annotations

import gc
import importlib
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
REPO = os.path.dirname(HERE)
os.makedirs(OUT, exist_ok=True)

LOG: list = []
R: dict = {}


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def hr(t=""):
    say("")
    say("=" * 76)
    if t:
        say(t)
        say("=" * 76)


# --------------------------------------------------------------- the scope
def load_scope():
    p = os.path.join(OUT, "gate_observed.json")
    try:
        d = json.load(open(p))
    except BaseException as e:
        say(f"  *** NO SCOPE AVAILABLE ({type(e).__name__}: {str(e)[:80]})")
        say("  *** gate_observed.py must run first.  Refusing to print an fla")
        say("  *** number without the configuration it was measured in -- that is")
        say("  *** the C-04 rule, and this is where it is enforced.")
        return None
    lab = d.get("scope_label")
    if not lab:
        say("  *** gate_observed.py ran but produced no scope_label.")
        return None
    say("  scope, from kda_t4/out/gate_observed.json (this session):")
    say(f"    {lab}")
    sel = d.get("selected_configs")
    if sel:
        say("    autotune actually selected, on this card:")
        for s in sel:
            say(f"      {s['key']}  ->  BV={s['kwargs'].get('BV')} "
                f"num_warps={s['num_warps']} num_stages={s['num_stages']}")
    else:
        say("    (no selected-config record; see gate_observed.txt section 5)")
    mv = d.get("model_vs_observed")
    if mv:
        say(f"    MODEL vs OBSERVED: {'AGREE' if mv.get('agree') else 'DISAGREE'}")
    R["scope"] = dict(label=lab, short=d.get("scope_label_short"),
                      selected=sel, distinct=d.get("distinct"),
                      model_vs_observed=mv, fwd_finite=d.get("fwd_finite"),
                      optin=d.get("optin_distinct"), n_configs=d.get("autotune_n_configs"))
    return R["scope"]


# ------------------------------------------------------------------ capture
def capture_tensors(mod, q, k, v, g, beta, C):
    """Run the port with Tensor.numpy intercepted, to get `o` and `S` as
    unrealized Tensors.  Restores the method before returning."""
    import numpy as np
    from tinygrad import Tensor

    grabbed = []
    orig = Tensor.numpy

    def spy(self, *a, **k):
        grabbed.append(self)
        # A zero array of the right shape and dtype.  The dtype is asked for
        # through tinygrad's own converter, not through `dtype.np`: in
        # tinygrad 0.14.0 a DType carries no `.np` attribute (its public
        # surface is bitsize/const/fmt/itemsize/max/min/name/new/priority), and
        # `np.dtype(DType)` does not work either.  The port never looks at this
        # value -- it only returns it -- so shape and finiteness are all that
        # matter, and dtype float64 is the safest stand-in.
        return np.zeros(tuple(self.shape), dtype=np.float64)
    Tensor.numpy = spy
    try:
        out = mod.kda_chunk(q, k, v, g, beta, chunk_size=C)
    finally:
        Tensor.numpy = orig
    return grabbed, out


# ------------------------------------------------------------------ timing
def cuda_ms(fn, warmup=3, iters=10):
    """Median/min/max of `iters` CUDA-event-timed calls, in ms.

    Used for fla, whose `fn` is a plain torch call and always re-launches.
    """
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return dict(median=ts[len(ts) // 2], min=ts[0], max=ts[-1],
                iters=iters, warmup=warmup)


def cuda_ms_prepare(prepare, launch, warmup=2, iters=5):
    """The same, for a framework whose realise() is a NO-OP on an already
    realised tensor.

    THIS IS NOT COSMETIC.  An earlier version of this file timed
    `o_t.realize()` on a tensor that had already been realised, and
    `Tensor.realize` is `to_realize = [x for x in (self,)+lst if not
    x.uop.is_virtual and not x.uop.has_buffer_identity()]; if len(to_realize):`
    (tinygrad 0.14.0 tensor.py:420-425) -- so the second and later calls run
    NOTHING.  The result was a flat 0.28 ms across every port and every shape,
    which looked plausible and was pure no-op, and a ratio against it would
    have been a fabricated 60x.

    So: `prepare` builds a FRESH unrealized graph (host time, deliberately
    OUTSIDE the event window) and `launch` realises it (device time, inside).
    """
    import torch
    for _ in range(warmup):
        launch(prepare())
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        obj = prepare()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        launch(obj)
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return dict(median=ts[len(ts) // 2], min=ts[0], max=ts[-1],
                iters=iters, warmup=warmup)


def wall_ms(fn, warmup=1, iters=5):
    ts = []
    for i in range(warmup + iters):
        torch_sync()
        t0 = time.perf_counter()
        fn()
        torch_sync()
        if i >= warmup:
            ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return dict(median=ts[len(ts) // 2], min=ts[0], max=ts[-1], iters=iters)


def torch_sync():
    try:
        import torch
        torch.cuda.synchronize()
    except BaseException:
        pass


# ------------------------------------------------------------------- sides
def fla_side(world, q, k, v, g, beta, shp):
    import torch
    from fla.ops.kda import chunk_kda
    tq, tk, tv = (world.to_torch(x, "bfloat16") for x in (q, k, v))
    tg = world.to_torch(g, "float32")
    tb = world.to_torch(beta, "bfloat16")
    scale = world.scale_for(shp["K"])

    def call():
        with torch.inference_mode():
            o, _ = chunk_kda(tq, tk, tv, tg, tb, scale=scale,
                             use_qk_l2norm_in_kernel=False)
        return o
    return call


# --------------------------------------------------------------- agreement
def agreement(fla_out, tg_out, name):
    import numpy as np
    a = np.asarray(fla_out, dtype=np.float64).ravel()
    b = np.asarray(tg_out, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    fa, fb = np.isfinite(a), np.isfinite(b)
    finite = bool(fa.all() and fb.all())
    d = np.abs(a - b)
    rel = d / np.maximum(np.abs(a), 1e-30)
    return dict(side=name, n=int(n), both_finite=finite,
                n_nonfinite_a=int((~fa).sum()), n_nonfinite_b=int((~fb).sum()),
                max_abs=float(d.max()),
                median_rel=float(np.median(rel)),
                p99_rel=float(np.percentile(rel, 99)),
                cosine=float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)))


# -------------------------------------------------------------------- main
LADDER = [
    (1,  512,  4, 4, 128, 128, 64),
    (1, 1024,  8, 8, 128, 128, 64),
    (2, 1024,  8, 8, 128, 128, 64),
    (2, 2048,  8, 8, 128, 128, 64),
]
PORTS = (("tiny_kda (naive)", "tiny_kda"),
         ("tiny_kda_opt", "tiny_kda_opt"),
         ("tiny_kda_faithful", "tiny_kda_faithful"))


def main():
    import numpy as np
    import torch
    import world
    iters = 5
    for i, a in enumerate(sys.argv):
        if a == "--iters" and i + 1 < len(sys.argv):
            iters = int(sys.argv[i + 1])
    largest_only = "--largest-only" in sys.argv

    hr("kda_t4 TIME -- fla vs the TinyGrad ports: one card, one session, one shape")
    import triton
    import importlib.metadata as md
    say(f"  torch {torch.__version__}   triton {triton.__version__}   "
        f"tinygrad {md.version('tinygrad')}   "
        f"fla-core {md.version('fla-core')}")
    say(f"  card  {torch.cuda.get_device_name(0)}  sm_"
        f"{''.join(str(x) for x in torch.cuda.get_device_capability(0))}  "
        f"current device {torch.cuda.current_device()} of "
        f"{torch.cuda.device_count()}")
    say(f"  iters {iters} (median reported), 2 warmup")

    scope = load_scope()
    if scope is None:
        return 2
    short = scope["short"] or "fla/T4 degraded"

    # ------------------------------------------------------- which ports exist
    hr("1. which TinyGrad ports exist in this payload, right now")
    sys.path.insert(0, os.path.join(REPO, "kda"))
    present, missing = [], []
    for name, mod in PORTS:
        p = os.path.join(REPO, "kda", mod + ".py")
        ok = os.path.exists(p)
        say(f"  {name:<22} {'present' if ok else 'MISSING':<8} "
            f"kda/{mod}.py  {os.path.getsize(p) if ok else 0} B")
        (present if ok else missing).append(name)
    say("")
    say("  Ports are NOT recreated, NOT waited for, NOT substituted.  What is on")
    say("  disk at the moment of the run is what gets timed; a missing port is")
    say("  reported as missing.")
    R["ports_present"], R["ports_missing"] = present, missing
    if not present:
        say("  no ports at all")
        return 3

    rows, agree = [], []
    ladder = LADDER[-1:] if largest_only else LADDER

    for (B, T, H, HV, K, V, C) in ladder:
        shp = dict(B=B, T=T, H=H, HV=HV, K=K, V=V, C=C)
        name = f"B{B}_T{T}_H{H}_K{K}_V{V}_C{C}"
        hr(f"2. shape {name}    NT={T // C} chunks, "
           f"{B * T * H * K / 1e6:.2f} M q elements")
        q, k, v, g, beta = world.make(B, T, H, HV, K, V, seed=0)
        rec = dict(shape=shp, label=name, side={}, errors={})

        # ---------------- fla
        try:
            call = fla_side(world, q, k, v, g, beta, shp)
            t = cuda_ms(call, 3, iters)
            rec["side"]["fla"] = t
            fo = call().float().cpu().numpy()
            rec["fla_absmax"] = float(np.abs(fo).max())
            rec["fla_finite"] = bool(np.isfinite(fo).all())
            say(f"  fla                          {t['median']:9.3f} ms  device"
                f"   (min {t['min']:8.3f})   finite={rec['fla_finite']}")
            say(f"  {'':<26} scope: {short}")
        except BaseException as e:
            rec["errors"]["fla"] = f"{type(e).__name__}: {str(e)[:200]}"
            say(f"  fla   FAILED: {type(e).__name__}: {str(e)[:220]}")
            traceback.print_exc(file=sys.stdout)
        torch.cuda.empty_cache()

        # ---------------- each TinyGrad port
        for pname, modname in PORTS:
            if pname not in present:
                continue
            gc.collect()
            torch.cuda.empty_cache()
            try:
                sys.modules.pop(modname, None)
                mod = importlib.import_module(modname)
                # device: o only, matching fla's output_final_state=False
                def prep_o():
                    gr, _ = capture_tensors(mod, q, k, v, g, beta, C)
                    return gr[0]

                def prep_S():
                    gr, _ = capture_tensors(mod, q, k, v, g, beta, C)
                    return gr[1] if len(gr) > 1 else None

                o_probe = prep_o()
                o_shape = tuple(o_probe.shape)
                s_probe = prep_S()
                s_shape = tuple(s_probe.shape) if s_probe is not None else None
                del s_probe
                del o_probe
                torch.cuda.synchronize()

                n_launch = _count_launches(mod, q, k, v, g, beta, C)
                t_o = cuda_ms_prepare(prep_o, lambda t: t.realize(), 2, iters)

                rec["side"][pname] = t_o
                rec.setdefault("meta", {})[pname] = dict(
                    n_launch_o=n_launch, o_shape=o_shape, s_shape=s_shape)
                say(f"  {pname:<26} {t_o['median']:9.3f} ms  device(o only)"
                    f"   (min {t_o['min']:8.3f})   {n_launch} kernels launched")

                try:
                    t_s = cuda_ms_prepare(prep_S, lambda t: t.realize(), 1, 3)
                    rec["side"][pname + " [device, S only]"] = t_s
                    say(f"  {'':<26} {t_s['median']:9.3f} ms  device(S only)"
                        f"   -- fla returns None for this (output_final_state"
                        f"=False), so it is NOT in the ratio column")
                except BaseException as e:
                    say(f"  {'':<26} S-only timing skipped: "
                        f"{type(e).__name__}: {str(e)[:70]}")

                # host: the unmodified end-to-end call
                t_h = wall_ms(lambda: mod.kda_chunk(q, k, v, g, beta,
                                                    chunk_size=C), 1, 2)
                rec["side"][pname + " [host, end-to-end]"] = t_h
                say(f"  {'':<26} {t_h['median']:9.3f} ms  HOST, end-to-end"
                    f" (build+realise+read) -- not a device number")

                # agreement, from a clean unpatched call
                try:
                    go, _gs = mod.kda_chunk(q, k, v, g, beta, chunk_size=C)
                    if "fla" in rec["side"]:
                        a = agreement(fo, go, f"{pname} vs fla")
                        a["shape"] = name
                        agree.append(a)
                        say(f"  {'':<26} agreement: cos {a['cosine']:.8f}"
                            f"  max|d| {a['max_abs']:.3e}"
                            f"  med rel {a['median_rel']:.3e}"
                            f"  finite {a['both_finite']}")
                except BaseException as e:
                    say(f"  {'':<26} (no output to compare: "
                        f"{type(e).__name__}: {str(e)[:100]})")
            except BaseException as e:
                rec["errors"][pname] = f"{type(e).__name__}: {str(e)[:220]}"
                say(f"  {pname:<26} FAILED: {type(e).__name__}: {str(e)[:220]}")
                traceback.print_exc(file=sys.stdout)
                if "out of memory" in str(e).lower():
                    say(f"  {'':<26} (OOM: this shape is above what a "
                        f"dense-recompute port fits in "
                        f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f}"
                        f" GiB.  fla is not memory-bound here and is unaffected --"
                        f" which is a fact about the SHAPE LADDER, not about fla.)")
                torch.cuda.empty_cache()
            finally:
                torch.cuda.empty_cache()
                gc.collect()
        rows.append(rec)

    # ---------------------------------------------------------------- table
    hr("3. THE TABLE, scope in a column rather than in a footnote")
    order = [p for p, _ in PORTS if p in present]
    say("")
    say(f"  FLA_SCOPE = \"{short}\"")
    say("")
    hdr = (f"  {'shape':<20} {'side':<32} {'device ms':>10} {'min ms':>9} "
           f"{'x fla':>8}  scope")
    say(hdr)
    say("  " + "-" * (len(hdr) + 4))
    for rec in rows:
        fmed = rec["side"].get("fla", {}).get("median")
        for s in (["fla"] + order):
            t = rec["side"].get(s)
            if not t:
                continue
            ratio = "" if (s == "fla" or not fmed) else f"{t['median'] / fmed:.2f}"
            sc = short if s == "fla" else "-"
            say(f"  {rec['label']:<20} {s:<32} {t['median']:>10.3f} "
                f"{t['min']:>9.3f} {ratio:>8}  {sc}")
        say("")

    say("  'x fla' is side/fla: 1.00 means the side matched the reference on this")
    say("  card, >1 means the side was slower.  It is a ratio of two whole")
    say("  operators, same inputs, same shape, same dtype, same card, same")
    say("  process.  LEGALITY, answered before the number is read (notes/19 5.1):")
    say("  there is no per-op against a per-step anywhere above, and no")
    say("  whole-step denominator is used.")
    say("")
    say("  Every fla cell above is a millisecond of the configuration named in")
    say("  FLA_SCOPE.  It is not a millisecond of fla.")

    # ------------------------------------------------------------- agreement
    hr("4. agreement, printed separately and NOT offered as evidence about the "
       "baseline")
    say("  notes/06 5b: two comparisons here were invalidated by porting the wrong")
    say("  object while passing every test.  A green agreement check is evidence")
    say("  about the MATH.  Which artifact ran is established by kda_t4/pin_t4.py")
    say("  from the registry, independently of this table.")
    say("")
    if agree:
        say(f"  {'shape':<20} {'pair':<32} {'cosine':>12} {'max|d|':>11} "
            f"{'med rel':>11} {'finite':>7}")
        for a in agree:
            say(f"  {a['shape']:<20} {a['side']:<32} {a['cosine']:>12.8f} "
                f"{a['max_abs']:>11.3e} {a['median_rel']:>11.3e} "
                f"{str(a['both_finite']):>7}")
    else:
        say("  no pair produced comparable outputs at any shape")

    # ------------------------------------------------------ the open unknown
    hr("5. THE CLEAN-vs-DEGRADED GAP: an open unknown, not a number")
    say("  MEASURED: fla on one T4, in the configuration this session observed")
    pin = os.path.join(OUT, "pin_t4.json")
    smem = None
    try:
        pd = json.load(open(pin))
        smem = (pd.get("device") or {}).get("smem_optin")
    except BaseException:
        pass
    say(f"  the T4's {smem if smem else '?'} B of optin shared memory and its sm_75")
    say("  compute capability force on it.  That is all.")
    say("")
    say("  NOT MEASURED, and not estimated anywhere in this file: what the same")
    say("  code does on a card with >= 166,912 B optin shared memory AND cc major")
    say("  >= 8.  Scaling this number by a stage ratio, a tile ratio or a")
    say("  precision ratio would be inventing a measurement.")
    say("")
    say("  The four gap classes this project uses (notes/19 section 6):")
    say("    (i)   the DSL cannot express it")
    say("    (ii)  the DSL can express it and does not find it")
    say("    (iii) not yet implemented on the DSL side")
    say("    (iv)  does not exist for this arch")
    say("")
    say("  REACHABLE by this measurement:")
    say("    (ii)  If a port lands close to fla-on-T4, the residual is a")
    say("          scheduling question and this run can address it, because both")
    say("          sides are on the same silicon with the same numerics and the")
    say("          same degraded reference configuration.")
    say("    (i)   Reachable only negatively: a port that lands orders of")
    say("          magnitude off cannot be separated into (i) vs (ii) by a")
    say("          timing alone.  The three-port ladder is the instrument for")
    say("          that, and it is reported above.")
    say("    (iii) The ladder across the three ports measures exactly this, if")
    say("          all three are present.")
    say("")
    say("  STRUCTURALLY UNREACHABLE by this measurement, at any run quality:")
    say("    (iv)  Fixed by the silicon.  There is no bf16 MMA below sm_80 and")
    say("          IS_TF32_SUPPORTED is False, so fla's tl.dot cannot reach a")
    say("          tensor core on this card however long the run takes.  A gap")
    say("          that is really 'fla emits HMMA and TinyGrad emits FFMA' is")
    say("          (iv) here -- and is untouched by every number above, because")
    say("          the HMMA never executes on this device.")
    say("    ALSO UNREACHABLE: fla-in-its-worst-config versus fla-in-its-best.")
    say("          That needs the card this study does not have.  It stays open.")

    with open(os.path.join(OUT, "time_t4.json"), "w") as f:
        json.dump(dict(rows=rows, agreement=agree, scope=R.get("scope"),
                       ports_present=present, ports_missing=missing,
                       iters=iters, ladder=LADDER), f, indent=2, default=str)
    with open(os.path.join(OUT, "time_t4.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")
    say("")
    say("wrote kda_t4/out/time_t4.{txt,json}")
    return 0


def _count_launches(mod, q, k, v, g, beta, C):
    """How many device launches one full realise of `o` produces.

    Counted by intercepting tinygrad's own execution hook rather than by
    timing, because it is the only number that separates 'one fused kernel'
    from 'twelve thousand tiny ones' at the same wall clock, and the three
    ports differ structurally by exactly this much.
    """
    import tinygrad.tensor as _T
    n = [0]
    orig = _T.run_linear          # tensor.py:15 binds the name at import

    def spy(linear, *a, **k):
        n[0] += len(linear.src)
        return orig(linear, *a, **k)
    _T.run_linear = spy
    try:
        gr, _ = capture_tensors(mod, q, k, v, g, beta, C)
        gr[0].realize()
    finally:
        _T.run_linear = orig
    return n[0]


if __name__ == "__main__":
    sys.exit(main())
