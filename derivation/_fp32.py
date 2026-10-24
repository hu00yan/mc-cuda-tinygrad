"""The float32 input recipes and reference arithmetic, shared by every divide
claim, so that `divide.py` and `divide_backends.py` cannot disagree about what
sample they measured.

THE RECIPES ARE PART OF THE CLAIM
    A rate over a distribution is a statement about a *draw*. The draw is
    therefore declared here, in one place, as an ordered list of operations on
    a named seed -- and its digest is pinned. Change a seed and every published
    rate changes, visibly, in the EXPECTED dict of every module that uses it.
    That is the point: the recipe is not a detail, it is the other half of the
    number.

    `a` uses `default_rng(0)` and three draws in the order
    `near 1.0 (LUT regime)`, `mixed +-2`, `wide +-1e3`. That is the stream
    `probes/fp32_correctness.py:42-50` consumes, so the self-division row is
    the same array that probe used and its counts are comparable to the shipped
    logs. `b` is drawn inside each law from `default_rng(7)`, after the law has
    fixed its own shape.

    Every quantity a divide claim needs is a FUNCTION of the draw, not a
    constant: the reciprocal error bound is half a float32 ULP read out of
    `np.finfo`, the cell edges are `nextafter` pairs, the 1-ULP bound is a
    measured maximum in ULPs. Nothing here is a written-down result.
"""
from __future__ import annotations

import numpy as np

# ---- the declared sample -------------------------------------------------
# n is a parameter of every rate in this study. A default, not a result.
N_DEFAULT = 1 << 21
SEED_A = 0                        # the probe's own stream
SEED_B = 7                        # the divisor stream inside each law
SEED_B_MATCHED = 20260928         # the matched-law independent draw

REGIME_ORDER = ("near 1.0 (LUT regime)", "mixed +-2", "wide +-1e3")

# The per-regime marginal law of `a`, as (width, offset) in float32. Written
# once and used by both the `a` draw and the matched-law `b` draw, which is
# what makes them "the same law, independent samples".
_MARGINALS = {
    "near 1.0 (LUT regime)": (np.float32(0.02), np.float32(0.99)),
    "mixed +-2": (np.float32(4), np.float32(-2)),
    "wide +-1e3": (np.float32(2000), np.float32(-1000)),
}


def draw(n: int, regime: str, rng: np.random.Generator) -> np.ndarray:
    """`rng.random(n)` mapped through a regime's marginal law, in float32, so
    the law is a property of the stream and not of `numpy.random.uniform`."""
    w, o = _MARGINALS[regime]
    return rng.random(n).astype(np.float32) * w + o


def regimes(n: int = N_DEFAULT) -> dict[str, np.ndarray]:
    """`a`: three magnitude regimes, one seeded stream, in REGIME_ORDER."""
    rng = np.random.default_rng(SEED_A)
    return {name: draw(n, name, rng) for name in REGIME_ORDER}


# ---- the PUBLISHED draw ---------------------------------------------------
#
# A SECOND, separately declared sample. This is the one every rate in this repo
# is quoted at, because it is the one the corrected probe publishes, and it is
# declared here for the same reason `regimes()` is: a rate over a distribution
# is a statement about a draw, so the draw is written down.
#
# It differs from `regimes()` in TWO ways, both deliberate and both load-bearing:
#
#   * n = 1,000,000 per regime, not 2**21. A rate printed to three decimals has
#     a binomial 1-sigma interval of ~0.03pp here, and that interval has to be
#     published WITH the number or the third decimal is decoration.
#   * the `near 1.0` regime is LOG-UNIFORM over two decades, `10**U(-0.3, 0.3)`,
#     not a uniform band. A uniform band of width 0.02 around 1.0 visits a
#     measure-zero slice of the mantissas a float32 near 1.0 can have; the
#     log-uniform draw is the one that actually visits two decades of them. Both
#     are called "near 1.0", which is exactly the ambiguity this removes.
#
# `a` and `b` come from two disjoint streams of the SAME marginal law, so the
# two columns differ only in dependence. The self-division column depends on the
# `a` stream alone; the general column depends on both.
N_PUBLISHED = 1_000_000
SEED_PUB_A = 0
SEED_PUB_B = 20260928

PUB_REGIME_ORDER = ("near 1.0 (log10-uniform 2 decades)",
                    "mixed +-2", "wide +-1e3")

_PUB_LINEAR = {
    "mixed +-2": (np.float32(4), np.float32(-2)),
    "wide +-1e3": (np.float32(2000), np.float32(-1000)),
}
_PUB_LOG = ("near 1.0 (log10-uniform 2 decades)",
            0.6, -0.3)                            # width, offset in log10


def draw_published(n: int, regime: str, rng: np.random.Generator) -> np.ndarray:
    """One draw from `regime`'s published marginal law. The log regime is
    `10**(width*u + offset)` with the power taken in float64 and the result
    rounded once to float32, so the law is a law rather than a chain of float32
    roundings."""
    if regime in _PUB_LINEAR:
        w, o = _PUB_LINEAR[regime]
        return rng.random(n).astype(np.float32) * w + o
    if regime != _PUB_LOG[0]:
        raise KeyError(regime)
    return (10.0 ** (rng.random(n) * _PUB_LOG[1] + _PUB_LOG[2])).astype(np.float32)


def published(n: int = N_PUBLISHED):
    """`(a, b)`: the published sample. `a` from SEED_PUB_A, `b` from
    SEED_PUB_B, the same marginal law, no shared sample."""
    ra = np.random.default_rng(SEED_PUB_A)
    rb = np.random.default_rng(SEED_PUB_B)
    a = {r: np.ascontiguousarray(draw_published(n, r, ra)) for r in PUB_REGIME_ORDER}
    b = {r: np.ascontiguousarray(draw_published(n, r, rb)) for r in PUB_REGIME_ORDER}
    return a, b


class Law:
    """A named divisor law.

    `independent` says whether `b` shares any sample with `a`. The band over
    independent laws is the citable one; a dependent law is a separate claim
    with its own name, because that is exactly the distinction the record kept
    losing. `fn(a, n, regime)` returns the divisor array.
    """
    def __init__(self, name: str, fn, independent: bool, note: str = ""):
        self.name, self.fn, self.independent, self.note = name, fn, independent, note
        self.fn.__doc__ = self.fn.__doc__ or ""

    def __call__(self, a: np.ndarray, n: int, regime: str) -> np.ndarray:
        return np.ascontiguousarray(self.fn(a, n, regime), dtype=np.float32)


def _matched(a, n, regime):
    return draw(n, regime, np.random.default_rng(SEED_B_MATCHED))


def _u(lo, hi):
    w, o = np.float32(hi - lo), np.float32(lo)
    return lambda a, n, regime: (np.random.default_rng(SEED_B).random(n).astype(np.float32) * w + o)


def _abs_normal(a, n, regime):
    r = np.random.default_rng(SEED_B)
    return (ONE / np.abs(r.standard_normal(n).astype(np.float32))).astype(np.float32)


# The six INDEPENDENT laws. These are the ones the band is over.
INDEPENDENT_LAWS = [
    Law("same law as a", _matched, True, "same marginal law, an independent draw"),
    Law("U[0.01, 2.01]", _u(0.01, 2.01), True),
    Law("U[0.5, 1.5]", _u(0.5, 1.5), True),
    Law("U[-8, 8]", _u(-8.0, 8.0), True),
    Law("U[1, 2]", _u(1.0, 2.0), True),
    Law("1/|N(0,1)|", _abs_normal, True, "heavy tailed: small divisors are common"),
]

# Correlated / dependent laws. Not in the band; each is its own claim, because
# each of them lands somewhere the band does not.
DEPENDENT_LAWS = [
    Law("a  (self-division)", lambda a, n, r: a, False,
        "quotient exactly 1.0 -- representable"),
    Law("2*a", lambda a, n, r: a * np.float32(2.0), False,
        "quotient exactly 0.5 -- representable"),
    Law("0.5*a", lambda a, n, r: a * np.float32(0.5), False,
        "quotient exactly 2.0 -- representable"),
    Law("-a", lambda a, n, r: a * np.float32(-1.0), False,
        "quotient exactly -1.0 -- representable; the sign moves, the magnitude does not"),
    Law("a+1", lambda a, n, r: a + np.float32(1.0), False,
        "quotient essentially never representable"),
    Law("a*a", lambda a, n, r: a * a, False,
        "quotient 1/a, essentially never representable"),
    Law("|a|+0.01", lambda a, n, r: np.abs(a) + np.float32(0.01), False,
        "the repo's own probe divisor, probes/fp32_correctness.py:77"),
]

REPRESENTABLE_QUOTIENT_LAWS = [l for l in DEPENDENT_LAWS
                               if "representable" in l.note and "never" not in l.note]
GENERAL_LAWS = [l for l in DEPENDENT_LAWS if "never" in l.note]


# ---- reference arithmetic ------------------------------------------------

ONE = np.float32(1.0)


def exact_f32(a, b, op: str) -> np.ndarray:
    """Correctly rounded float32 `a op b`, via float64.

    Exact in float64 for every magnitude used here: a float32 product or
    quotient needs at most 48 significand bits and float64 carries 53, so the
    single rounding down to float32 is the correctly rounded one.
    """
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    r = {"+": a64 + b64, "-": a64 - b64, "*": a64 * b64,
         "/": a64 / b64, "recip": 1.0 / b64}[op]
    return r.astype(np.float32)


def rcp(x) -> np.ndarray:
    """fl32(1/x): a correctly rounded float32 reciprocal, evaluated in float32
    so the intermediate rounding is a real float32 rounding."""
    return ONE / x


def rcp_mul(a, b) -> np.ndarray:
    """`a * fl32(1/b)`, both factors float32 and the product float32. What a
    cstyle backend's `a*(1/b)` evaluates to."""
    return (a * rcp(b)).astype(np.float32)


def delta(x) -> np.ndarray:
    """Relative error of the float32 reciprocal, in float64:
    `fl32(1/x)/(1/x) - 1`. Computed in float64 so the diagnostic is not itself
    a float32 rounding."""
    return rcp(x).astype(np.float64) / (1.0 / x.astype(np.float64)) - 1.0


def ulp_of(v) -> np.ndarray:
    """Size of one float32 ULP at `v`, in float64, measured in the direction
    UPWARD from `v`. Above a power of two this is the larger of the two
    neighbouring gaps, which is the standard convention and the one the
    "max 1 ULP" bound is stated in."""
    return np.spacing(np.abs(v).astype(np.float32)).astype(np.float64)


def ulp_in_direction(v, direction) -> np.ndarray:
    """The float32 gap at `v` on the side the error went.

    The gap immediately BELOW `v` is half the gap above `v` whenever `v` is a
    float32 power of two, and `1.0` is one. So an error of `1 - 2^-24` measured
    against `np.spacing(1.0) = 2^-23` reads as "half a ULP", which is a
    measurement artefact of a symmetric convention, not a smaller error. This
    function picks the gap on the side the error actually travelled, so a
    one-ULP error reads as one ULP in every case. `ulp_bound` uses it; a claim
    that says "1 ULP" without saying which convention is not quotable.
    """
    v32 = np.abs(v).astype(np.float32)
    up = np.spacing(v32).astype(np.float64)
    down = np.abs(v32.astype(np.float64)
                  - np.nextafter(v32, np.float32(0)).astype(np.float64))
    return np.where(direction >= 0, up, down)


def shift_ulp(x, k: int) -> np.ndarray:
    """`x` moved `k` ULPs further from zero, by moving the bit pattern. Used
    to MODEL a reciprocal that is k ULPs off. Nothing that calls this runs the
    result on a device."""
    xi = x.view(np.int32).astype(np.int64)
    mag = np.where(xi < 0, -xi, xi) + abs(int(k))
    return np.where(xi < 0, -mag, mag).astype(np.int32).view(np.float32)


# ---- the cell geometry, derived not typed --------------------------------

class Cell:
    """The float32 rounding cell that contains the exact quotient 1.0.

    The self-division failure rate is a statement about this cell's shape, so
    its edges are read out of the float32 neighbourhood of 1.0 rather than
    written down: `gap_below` and `gap_above` are the two real distances, and
    the safe window is the interval between the two midpoints.
    """
    def __init__(self, unit: float = 1.0) -> None:
        u = np.float32(unit)
        below = float(np.nextafter(u, np.float32(-np.inf)))
        above = float(np.nextafter(u, np.float32(np.inf)))
        self.unit = unit
        self.below, self.above = below, above
        self.gap_below = self.unit - below
        self.gap_above = above - self.unit
        self.edge_lo = -self.gap_below / 2.0        # midpoint below, relative
        self.edge_hi = self.gap_above / 2.0         # midpoint above, relative
        self.threshold = self.edge_lo               # delta at which 1+delta misses
        self.eps = float(np.finfo(np.float32).eps)
        self.rcp_bound = self.eps / 2.0             # |delta| of a correct reciprocal

    def fails(self, d) -> np.ndarray:
        """`a*fl32(1/a)` is exactly `1+delta(a)` in the reals, so the product
        misses 1.0 exactly when delta leaves the safe window."""
        return d < self.threshold

    def describe(self) -> str:
        return (f"float32 cell of {self.unit}: neighbour below {self.below!r} "
                f"(gap {self.gap_below:.6e}), neighbour above {self.above!r} "
                f"(gap {self.gap_above:.6e}); safe window, in delta units, "
                f"[{self.edge_lo:.6e}, {self.edge_hi:.6e}); "
                f"|delta| <= {self.rcp_bound:.6e} for a correctly rounded reciprocal")

    def failure_fraction_uniform(self) -> float:
        """Fraction of a UNIFORM perturbation on `[-rcp_bound, +rcp_bound]`
        that lands outside the safe window. This is the correct answer to the
        question `bench/HYPOTHESES.md` answered with 1.0e-7, and it is read
        off the window, not guessed."""
        below = max(0.0, self.edge_lo - (-self.rcp_bound))
        above = max(0.0, self.rcp_bound - self.edge_hi)
        return (below + above) / (2.0 * self.rcp_bound)
