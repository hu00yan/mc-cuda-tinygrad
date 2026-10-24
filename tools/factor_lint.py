#!/usr/bin/env python3
"""
Static lint for factorisable decay in generated kernels.

WHAT THIS CHECKS
----------------
A reduction of the form

    sum_d  a[i,d] * b[j,d] * exp(g[i,d] - g[j,d])

computes the exponential once per (i,j,d) triple, i.e. O(C*C*K) times. The
algebraically equal factorisation

    exp(g[i,d]) * exp(-g[j,d])     ->    (a*exp(g)) @ (b*exp(-g)).T

computes it O(C*K) times and turns the contraction into a matmul.

The two are NOT equal in IEEE float (`exp(x-y)` and `exp(x)*exp(-y)` round
differently), so a compiler is not permitted to make the change. It must be
requested. This lint finds the sites where requesting it would pay.

MEASURED PAYOFF (sm_80, via matmul/tg_backend.py, see FACTORISATION-FINDING.md):
    64 exp2  ->  32 exp2   for the C=64,K=128 KDA gram.
Not claimed: any speedup, or the jump from 32 to the 3 in notes/16. That needs
SASS on real hardware.

DETECTION
---------
Walk the UOp graph. An `EXP2` is reported when all hold:
  1. its input is (or contains) a difference whose two sides are broadcast
     copies of arrays indexed by *different* axes -- the signature of
     `g[i,..]-g[j,..]`. Note this tinygrad never emits `Ops.SUB` for tensor
     math: `a - b` lowers to `ADD(a, MUL(b, -1))`, so both spellings are
     recognised;
  2. the result's element count exceeds both operands' (the broadcast that
     creates the O(C*C*K) tensor). Numel is used rather than rank because
     `[C,1,K] - [1,C,K]` and `[C,1] - [1,C]` broadcast without a rank increase.
This is deliberately conservative: it only reports exponentials of a difference
of two broadcast terms. It will not report a legitimate `exp(x)` of one array.

USAGE
    .venv/bin/python tools/factor_lint.py            # built-in cases
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from tinygrad.uop.ops import UOp, Ops

# `.exp()` lowers to EXP2 with a log2(e) multiply; there is no Ops.EXP in this version.
_EXP = {Ops.EXP2: "exp2"}


@dataclass
class Hit:
    op: str
    out_shape: tuple
    left_shape: tuple
    right_shape: tuple

    def __str__(self):
        return (f"{self.op} of a difference, broadcast to {self.out_shape} "
                f"(operands {self.left_shape} vs {self.right_shape})")


def _is_neg_one(node: UOp) -> bool:
    return node.op is Ops.CONST and node.arg is not None and node.arg == -1


def _find_difference(node: UOp) -> tuple[UOp, UOp] | None:
    """Return the (left, right) operands of a `left - right` reachable from node."""
    for n in node.toposort():
        if n.op is Ops.SUB and len(n.src) == 2:
            return n.src[0], n.src[1]
        # `a - b` arrives as ADD(a, MUL(b, -1)): spot the negated right-hand side.
        if n.op is Ops.ADD and len(n.src) == 2:
            for i, m in enumerate(n.src):
                if m.op is Ops.MUL and len(m.src) == 2 and any(_is_neg_one(s) for s in m.src):
                    other = m.src[0] if _is_neg_one(m.src[1]) else m.src[1]
                    return n.src[1 - i], other
    return None


def analyze_exp(node: UOp, hits: list[Hit]) -> None:
    if node.op not in _EXP:
        return
    if not node.src:
        return
    diff = _find_difference(node.src[0])
    if diff is None:
        return
    left, right = diff
    # The broadcast signature: the exponential's output is larger than both
    # sides of the difference. That is what creates the O(C*C*K) intermediate.
    out_numel = node.numel()
    if out_numel > left.numel() and out_numel > right.numel():
        hits.append(Hit(_EXP[node.op], tuple(node.shape),
                        tuple(left.shape), tuple(right.shape)))


def scan(named: dict[str, UOp]) -> int:
    print(f"{'case':<26}{'finding':<16}{'out':<14}{'operands'}")
    print("-" * 84)
    total = 0
    for name, uop in named.items():
        hits: list[Hit] = []
        for n in uop.toposort():
            analyze_exp(n, hits)
        if not hits:
            print(f"{name:<26}{'(none)':<16}")
            continue
        for h in hits:
            total += 1
            print(f"{name:<26}{h.op:<16}{str(h.out_shape):<14}{h.left_shape} vs {h.right_shape}")
    print("-" * 84)
    print(f"factorisable sites: {total}")
    return total


def cases() -> dict[str, UOp]:
    from tinygrad import Tensor

    C, K = 8, 16
    k = Tensor.empty(C, K)
    g = Tensor.empty(C, K)

    out: dict[str, UOp] = {}

    # The KDA pattern: decay of a pairwise difference -> [C,C,K] intermediate.
    out["kda_gram_decay"] = (g[:, None, :] - g[None, :, :]).exp().uop

    # Legitimate: exponential of a single array, no difference -> not reported.
    out["plain_exp"] = g.exp().uop

    # A difference of two ROW vectors broadcast to [C,C] with no contracted axis.
    # Still factorisable, and still worth reporting.
    out["pairwise_bias"] = (g[:, 0][:, None] - g[:, 0][None, :]).exp().uop

    # Not a difference -> not reported.
    out["sum_exp"] = (g.exp() + g.log2()).uop

    return out


if __name__ == "__main__":
    n = scan(cases())
    sys.exit(0)
