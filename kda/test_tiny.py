"""test_tiny.py - the TinyGrad KDA port against the float64 reference.

The load-bearing test is the same one as in test_ref.py: the port must match
`kda_recurrent`, the per-token definition, not just the chunked numpy
implementation. Matching `kda_chunk` alone would only prove the two chunked
implementations agree with each other.

Tolerance: 1e-5 relative, in float32. ref.py runs in float64, so the gap is
purely float32 rounding. Measured worst case is ~2e-6.

Run: ../../.venv/bin/python test_tiny.py
"""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ref
import tiny_kda
import test_ref as T

TOL = 1e-5
TESTS = []


def test(t):
    TESTS.append(t)
    return t


def relerr(a, b):
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)


def approx(a, b, what, tol=TOL):
    e = relerr(a, b)
    assert e < tol, f"{what}: relerr {e:.3e} >= {tol:.1e}"
    return e


# The same shape configs test_ref.py uses, so both suites cover the same ground.
CASES = [
    # (B, T, H, HV, K, V, C)
    (2, 16, 2, 2, 4, 8, 8),
    (2, 16, 2, 2, 4, 8, 16),
    (2, 32, 2, 2, 8, 8, 8),
    (1, 32, 1, 1, 8, 8, 16),
    (3, 24, 3, 3, 6, 10, 8),
    (1, 64, 2, 2, 8, 8, 16),
    (2, 64, 2, 2, 8, 4, 32),
    (2, 16, 1, 3, 4, 4, 8),      # GVA, G = 3
    (2, 16, 1, 4, 8, 8, 16),     # GVA, G = 4
]


@test
def test_tiny_matches_recurrence():
    """THE test: tiny_kda == the per-token definition of KDA."""
    worst = 0.0
    for i, (B, Tt, H, HV, K, V, C) in enumerate(CASES):
        q, k, v, g, beta = T.rnd(B, Tt, H, HV, K, V, seed=i)
        o_ref, S_ref = ref.kda_recurrent(q, k, v, g, beta)
        o_t, S_t = tiny_kda.kda_chunk(q, k, v, g, beta, chunk_size=C)
        worst = max(worst,
                    approx(o_t, o_ref, f"case{i} C={C} o"),
                    approx(S_t, S_ref, f"case{i} C={C} S"))
    return worst


@test
def test_tiny_chunk_size_invariance():
    """The result must not depend on C. C=1 degenerates to the recurrence, so
    this is a second independent statement of the test above."""
    B, Tt, H, HV, K, V = 2, 32, 2, 2, 8, 8
    q, k, v, g, beta = T.rnd(B, Tt, H, HV, K, V, seed=11, l2n=True)
    o0, S0 = tiny_kda.kda_chunk(q, k, v, g, beta, chunk_size=1)
    worst = 0.0
    for C in (2, 4, 8, 16, 32):
        o, S = tiny_kda.kda_chunk(q, k, v, g, beta, chunk_size=C)
        worst = max(worst, approx(o, o0, f"invariance o C={C}"),
                    approx(S, S0, f"invariance S C={C}"))
    return worst


@test
def test_tiny_with_initial_state():
    """A nonzero incoming state exercises the inter-chunk scan's first
    iteration, which is the one the zero-state case never touches."""
    worst = 0.0
    for i, (B, Tt, H, HV, K, V, C) in enumerate(CASES[:5]):
        q, k, v, g, beta = T.rnd(B, Tt, H, HV, K, V, seed=100 + i)
        h0 = np.random.default_rng(200 + i).normal(size=(B, HV, K, V))
        o_ref, S_ref = ref.kda_recurrent(q, k, v, g, beta, initial_state=h0)
        o_t, S_t = tiny_kda.kda_chunk(q, k, v, g, beta, chunk_size=C,
                                      initial_state=h0)
        worst = max(worst,
                    approx(o_t, o_ref, f"h0 case{i} o"),
                    approx(S_t, S_ref, f"h0 case{i} S"))
    return worst


@test
def test_tiny_degenerate_gates():
    """g = 0 must reduce to plain DeltaNet, beta = 0 must disable writes, and
    a hard gate must wipe the state. These are the sign/order traps."""
    B, Tt, H, HV, K, V, C = 2, 32, 2, 2, 8, 8, 8
    q, k, v, _, beta = T.rnd(B, Tt, H, HV, K, V, seed=3)
    worst = 0.0
    for name, g, b in (
            ("g=0 -> DeltaNet", np.zeros((B, Tt, HV, K)), beta),
            ("beta=0 -> no writes", np.random.default_rng(1).uniform(-1, 0, (B, Tt, HV, K)),
             np.zeros((B, Tt, HV))),
            ("beta=1", np.random.default_rng(2).uniform(-1, 0, (B, Tt, HV, K)),
             np.ones((B, Tt, HV))),
            ("hard gate (g=-30)", np.full((B, Tt, HV, K), -30.0), beta)):
        o_ref, S_ref = ref.kda_recurrent(q, k, v, g, b)
        o_t, S_t = tiny_kda.kda_chunk(q, k, v, g, b, chunk_size=C)
        worst = max(worst, approx(o_t, o_ref, name), approx(S_t, S_ref, name + " S"))
    return worst


@test
def test_tiny_causality():
    """Changing a LATE token must not move an EARLY output. Catches a
    transposed time axis, which is the failure mode of a bad permute."""
    B, Tt, H, HV, K, V, C = 2, 32, 2, 2, 8, 8, 8
    q, k, v, g, beta = T.rnd(B, Tt, H, HV, K, V, seed=5)
    o1, _ = tiny_kda.kda_chunk(q, k, v, g, beta, chunk_size=C)
    k2 = k.copy()
    k2[:, 20:] += 3.0                       # perturb only t >= 20
    o2, _ = tiny_kda.kda_chunk(q, k2, v, g, beta, chunk_size=C)
    early = np.abs(o2[:, :20] - o1[:, :20]).max()
    assert early < 1e-6, f"future leaked backwards: {early:.3e}"
    late = np.abs(o2[:, 20:] - o1[:, 20:]).max()
    assert late > 1e-3, "perturbation had no effect; test is vacuous"
    return early


@test
def test_tiny_dtype_and_shapes():
    """float32 in, float32 out, and the documented shapes."""
    B, Tt, H, HV, K, V, C = 2, 32, 2, 2, 8, 8, 8
    q, k, v, g, beta = T.rnd(B, Tt, H, HV, K, V, seed=6)
    o, S = tiny_kda.kda_chunk(q, k, v, g, beta, chunk_size=C)
    assert o.shape == (B, Tt, HV, V), o.shape
    assert S.shape == (B, HV, K, V), S.shape
    assert o.dtype == np.float32, o.dtype
    return True


def main():
    print("=" * 78)
    print(f"TinyGrad KDA vs the float64 reference   (numpy {np.__version__})")
    print("=" * 78)
    passed, failed = [], []
    for fn in TESTS:
        name = fn.__name__.replace("test_tiny_", "").replace("_", " ")
        try:
            out = fn()
            extra = "" if out is True or out is None else f"   [worst relerr {out:.2e}]"
            print(f"  PASS  {name}{extra}")
            passed.append(name)
        except Exception as e:
            print(f"  FAIL  {name}")
            print(f"        {e}")
            failed.append(name)
    print("-" * 78)
    print(f"{len(passed)} passed, {len(failed)} failed")
    if failed:
        print("failed: " + ", ".join(failed))
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
