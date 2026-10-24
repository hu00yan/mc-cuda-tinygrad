"""
Negative controls for factor_lint.py.

The rule: every automated check needs a deliberately-broken input that it MUST
reject, and a known-good input it MUST accept. Without both, a check that always
returns 0 is indistinguishable from a check that always passes.

Two halves, and both matter:
  * MUST-FLAG   -- `exp(a[i,..] - b[j,..])` broadcast across a contraction. This
                   is the O(C*C*K) exp site the lint exists to find. If the
                   detector stops recognising it the lint is silently useless.
  * MUST-SILENT -- a plain `.exp()` of one array, and a difference that is not
                   in a broadcast/contraction context. If the detector starts
                   firing here it would push callers to "factorise" expressions
                   that do not factorise.

Run:  .venv/bin/python tools/test_factor_lint.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import factor_lint as F

from tinygrad import Tensor

FAILS = []

def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILS.append(name)

def hits(t):
    """Every finding factor_lint reports for this tensor, via its own analyzer."""
    found = []
    for n in t.uop.toposort():
        F.analyze_exp(n, found)
    return found

# --- control 1: the KDA gram decay MUST be flagged -------------------------
# exp(g[i,d] - g[j,d]) broadcast to [C,C,K]. This is the whole reason the lint
# exists: the exp is computed C*C*K times instead of C*K.
C, K = 64, 128
g = Tensor.empty(C, K)
kda = (g[:, None, :] - g[None, :, :]).exp()
h = hits(kda)
check("flags exp of a broadcast difference", len(h) == 1, f"hits={len(h)}")
if h:
    check("reports the [C,C,K] output shape", tuple(h[0].out_shape) == (64, 64, 128),
          f"out_shape={h[0].out_shape}")
    check("reports both broadcast operands", tuple(h[0].left_shape) == (64, 1, 128)
          and tuple(h[0].right_shape) == (1, 64, 128),
          f"{h[0].left_shape} vs {h[0].right_shape}")
    check("names the exp op it found", h[0].op == "exp2", f"op={h[0].op}")

# --- control 2: detection keys on numel, not rank --------------------------
# [C,1] - [1,C] broadcasts to [C,C] with NO rank increase, so a rank-based test
# would miss it. Still a factorisable O(C*C) exp site.
bias = (g[:, 0][:, None] - g[:, 0][None, :]).exp()
check("flags a rank-preserving broadcast difference", len(hits(bias)) == 1,
      f"hits={len(hits(bias))}")

# --- control 3: a plain .exp() MUST stay silent ----------------------------
# No difference at all: exp(g) does not factorise, and rewriting it would be a
# pessimisation at best.
check("silent on plain exp", hits(g.exp()) == [], f"hits={len(hits(g.exp()))}")

# --- control 4: a same-shape difference MUST stay silent -------------------
# (g - h) broadcasts nothing, so the exp runs once per element already. There
# is no C*C blowup to remove.
other = Tensor.empty(C, K)
check("silent on a non-broadcast difference", hits((g - other).exp()) == [],
      f"hits={len(hits((g - other).exp()))}")

# --- control 5: a broadcast difference with no exp MUST stay silent --------
# The detector is about exponentials. A bare pairwise difference is just a
# subtraction; there is nothing to factorise.
check("silent on a broadcast difference without exp",
      hits(g[:, None, :] - g[None, :, :]) == [],
      f"hits={len(hits(g[:, None, :] - g[None, :, :]))}")

# --- control 6: a subtraction OUTSIDE the exp MUST stay silent -------------
# exp(a) - exp(b) has the exp on the operands, not on a difference. Flagging it
# would be a false positive.
check("silent when the subtraction is outside the exp",
      hits(g.exp() - other.exp()) == [],
      f"hits={len(hits(g.exp() - other.exp()))}")

# --- control 7: scan()'s count agrees with the per-node findings -----------
# scan() is the user-facing entry point; make sure it reports the same thing
# the controls above assert on.
mixed = {"flagged": kda.uop, "silent": g.exp().uop}
check("scan() counts only the flagged case", F.scan(mixed) == 1)

print()
if FAILS:
    print(f"{len(FAILS)} control(s) FAILED: {FAILS}")
    sys.exit(1)
print("all negative controls passed")
