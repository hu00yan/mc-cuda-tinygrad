"""
SIDE 1 of the TPU comparison: KDA as a direct transcription into JAX, so that
XLA compiles it.  This is the TPU analogue of kda/tiny_kda.py.

*** THIS FILE HAS NEVER BEEN EXECUTED. ***
The repo venv has no jax (verified: `import jax` -> ModuleNotFoundError), no
torch, and there is no TPU reachable from here.  It is written against the
verified JAX/XLA API and it is meant to be pasted into a Kaggle notebook cell
where `pip install -U "jax[tpu]"` has run.  Every number in kda/tpu/README.md
that would come from this file is currently a PREDICTION, not a measurement.

Read kda/tpu/README.md section 1 first: tinygrad has NO TPU backend, so S1 is
not "tinygrad on TPU" -- it is the closest available analogue, and the whole
point of S1 is to ask whether a much better whole-graph compiler fixes the
failure the CUDA study found.  We predict it does not, for a specific reason:
the [C,C,K] decay tensors are in the graph, and the graph is the semantics.

The algebra is kda/SPEC.md verbatim, and kda/tpu/test_design.py validates the
exp-folding identity in float64 against ../ref.py on CPU.
"""

# ---------------------------------------------------------------- imports
import functools

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------- config
B, T, H, HV, K, V, C = 1, 2048, 4, 4, 128, 128, 64
NT = T // C
G = HV // H
SCALE = K ** -0.5


def _prep(q, k, v, g, beta, scale=SCALE):
    """kda/ref.py::_prep, in jnp.  q,k are [B,T,H,K]; v,g [B,T,HV,?]; beta [B,T,HV].

    GVA: when HV > H, q and k are shared by G = HV//H adjacent value heads, but
    g is NOT shared -- it is per value head AND per key channel, so its head dim
    is HV.  This is the one place KDA's head layout is unusual.
    """
    q = jnp.repeat(q, G, axis=2) * scale
    k = jnp.repeat(k, G, axis=2)
    return q, k, v, g, beta


def _chunks(x):
    """[B,T,HV,X] -> [B,NT,HV,C,X]."""
    return x.reshape(B, NT, C, -1).transpose(0, 1, 3, 2, 4)


def _chunks_scalar(x):
    return x.reshape(B, NT, C, -1).transpose(0, 1, 3, 2)


def _causal_decay(g_cum, lower):
    """kda/ref.py::_causal_decay, VERBATIM.  This is the [C,C,K] tensor.

    It is here on purpose.  It is the object the whole study is about, and
    removing it is what S2 and S3 do.  Note the shape: [B,NT,HV,C,C,K], which
    at the study's target shape (B=4,T=4096,H=32,K=128,C=64) is 8.6 GB per
    tensor in bf16.  It is 64x larger than anything else in the operator.
    """
    d = g_cum[..., :, None, :] - g_cum[..., None, :, :]      # [...,C,1,K]-[...,1,C,K]
    C_ = g_cum.shape[-2]
    keep = (jnp.tril(jnp.ones((C_, C_), dtype=bool), -1) if lower
            else jnp.tril(jnp.ones((C_, C_), dtype=bool)))[..., :, None]
    return jnp.exp(jnp.where(keep, d, -jnp.inf))


def _forward_substitution(A):
    """kda/ref.py::_forward_substitution.  Unrolled C-1 times in the graph.

    This is the C-1-DEPENDENT-step sequential spine.  kda/notes/03-kda.md records
    2,016 unrolled elementary steps at C=64 in the tinygrad port, because each
    column's substitution is a sequential sum over r previous entries, so the
    whole thing is C(C-1)/2 and not C.  Expect the same explosion here.
    """
    C_ = A.shape[-1]
    a = A
    for i in range(1, C_):
        row = a[..., i, :][..., None, :] @ a[..., :, :i]     # [...,1,i] @ [...,C,i]
        a = jax.lax.dynamic_update_index_in_dim(
            a, a[..., i, :i] + row[..., 0, :], i, axis=-2)
    return a + jnp.eye(C_, dtype=a.dtype)


@functools.partial(jax.jit, static_argnames=("chunk_size",))
def kda_chunk(q, k, v, g, beta, chunk_size=C):
    """Kimi Linear 3.1, in jnp.  Shape and order follow kda/SPEC.md 3.5 exactly.

    The seven steps, and which of them we expect XLA to fuse:
      1  g_cum = cumsum(g) over the chunk axis         [B,NT,HV,C,K]
      2  Dl = _causal_decay(g_cum, lower=True)        [B,NT,HV,C,C,K]  <-- MATERIALISED
      2  Z = (Dl * K) K^T                              [B,NT,HV,C,C]
      3  Btri = StrictTril(Diag(beta) Z)
      4  M = fwdsub(-Btri) + I, * Diag(beta)          SEQUENTIAL, C-1 steps
      5  w = M (Gamma . K)                            [B,NT,HV,C,K]
      6  u = M V                                      [B,NT,HV,C,V]
      7  inter-chunk scan over NT, S carried
      7a v_new = u - w S
      7b Aqk = Tril((Q . D_causal) K^T)               <-- a SECOND [C,C,K] tensor
      7c o = (Gamma . Q) S + Aqk v_new
      7d S = Diag(g_cum[-1]) S + ((G[-1]/G) . K)^T v_new
    """
    q, k, v, g, beta = _prep(q, k, v, g, beta)
    NT_ = T // chunk_size
    qc, kc, vc = (_chunks(q), _chunks(k), _chunks(v))
    gc0 = _chunks(g)
    bc = _chunks_scalar(beta)

    g_cum = jnp.cumsum(gc0, axis=-2)                        # [B,NT,HV,C,K]
    gam = jnp.exp(g_cum)                                    # gamma^r, safe: <= 1
    kg = gam * kc                                           # Gamma (.) K

    Dl = _causal_decay(g_cum, lower=True)
    Z = jnp.einsum("bnhcpd,bnhpd->bnhcp", Dl * kc[..., None, :], kc)
    del Dl                                                  # graph node is gone; the
                                                            # BUFFER was already live
    Btri = Z * bc[..., :, None]                             # Z is already strictly lower

    Minv = _forward_substitution(-Btri)                     # (I + StrictTril)^-1
    Minv = Minv * bc[..., None, :]                           # * Diag(beta) = the paper's M

    w = Minv @ kg
    u = Minv @ vc

    # --- inter-chunk scan.  NT strictly sequential steps; S is [HV,K,V] and
    #     does NOT grow with T.  Written as a Python loop so it unrolls in the
    #     graph, because lax.fori_loop with a [HV,K,V] accumulator would force
    #     XLA to keep S in HBM across iterations.
    def scan(S, n):
        def body(S, n):
            Sd = jnp.exp(g_cum[:, n]) * S                   # Diag(gamma^C) S
            wn, un = w[:, n], u[:, n]                       # [HV,C,K], [HV,C,V]
            v_new = un - wn @ S                             # [HV,C,V]
            # Aqk = Tril((Q . D_causal) K^T); uses PLAIN q, the decay is in the exp
            Dc = _causal_decay(g_cum[:, n][..., None, :], lower=False)   # [HV,C,C,K]
            qn = qc[:, n]                                   # [HV,C,K]
            Aqk = jnp.einsum("hcpd,hpd->hcp", Dc * qn[..., None, :], kc[:, n])
            del Dc
            o_n = (gam[:, n] * qn) @ S + Aqk @ v_new
            Sn = Sd + (kg[:, n][..., :, None] * (gam[:, n] * v_new)[..., None, :]).sum(-2)
            return Sn, o_n
        return body(S, n)

    S0 = jnp.zeros((B, HV, K, V), dtype=q.dtype)
    # lax.scan over n: the state S is the carry.  kda/tiny_kda.py unrolls instead
    # (see kda/codegen_diff.py); here we do NOT, so S1 measures the *graph* and
    # not the unrolling.  A third variant that unrolls would be a fourth side.
    S_final, o_chunks = jax.lax.scan(scan, S0, jnp.arange(NT_))
    o = o_chunks.transpose(0, 1, 3, 2, 4).reshape(B, T, HV, V)
    return o, S_final


# ---------------------------------------------------------------- measurement
def measure(o_fn, *args, n=20):
    """Device-side timing, per arXiv:2607.20466 2.3.

    They are explicit that wall-clock + block_until_ready() is WRONG on TPU:
    "For short kernels (<1 ms), this overhead can constitute 10-20% of measured
    time."  We use jax.profiler.trace and parse the Perfetto trace for
    jit_*()-level device events, and report the median of 50 iterations.

    ALSO: report FLOPs/median_time / peak, i.e. MXU utilisation as a FRACTION OF
    THE ROOFLINE, never a raw ms.  That is what makes the number survive the
    v3-8 -> v5e-8 migration.  kda/tpu/roofline.py holds the per-generation peaks.
    """
    import jax.profiler
    with jax.profiler.trace("/tmp/kda_s1"):
        jax.block_until_ready(o_fn(*args))
        for _ in range(5):
            jax.block_until_ready(o_fn(*args))
        for _ in range(50):
            o = o_fn(*args)
        jax.block_until_ready(o)
    # parse /tmp/kda_s1/*.trace.json.gz for device-side jit_ events -> median ms
    raise NotImplementedError("wire the Perfetto parser from run_measure.py")


def hlo_stats(o_fn, *args):
    """XLA's own byte/FLOP accounting for the compiled program.

    lower(o_fn).compile(*args).cost_analysis() gives bytes accessed and flops per
    op, summed.  This is the arch-independent 'materialised bytes' metric: it
    counts what XLA says it will move, independent of any wall clock and
    independent of which TPU generation we are on.  It is the single most
    important number in the study and it is obtainable without running anything
    expensive.

    PREDICTION, from kda/tpu/shapes.py: at B=1,T=2048,H=4,K=128,C=64 this reports
    ~279 MB of HBM traffic for S1 against ~10.8 MB of compulsory traffic, because
    the two [C,C,K] tensors are 268 MB of it.
    """
    c = jax.jit(o_fn).lower(*args).compile()
    return c.cost_analysis()


if __name__ == "__main__":
    raise SystemExit(
        "This module is not runnable in the repo venv (no jax, no TPU).\n"
        "Copy kda_chunk / hlo_stats / measure into a Kaggle notebook with\n"
        "jax[tpu] installed.  See kda/tpu/README.md section 5 for the runbook."
    )
