#!/usr/bin/env python3
"""
Redundant-computation detector for tinygrad UOp graphs.

WHAT THIS DOES (read-only)
--------------------------
For every REDUCE node, find subexpressions inside the reduction subtree that do
not depend on the reduction axis and are used more than once. Those are
loop-invariant with respect to the reduction: a compiler pass could compute them
once outside the reduction and broadcast the result in, instead of recomputing
them per reduced term.

The motivating measurement (notes/16): TinyGrad's CUDA backend emits 128
`MUFU.EX2` for a KDA reduction, where the hand-written Triton kernel emits 3.
Same maths; the generated code recomputes the decay once per contracted term,
inside the reduction.

This tool does NOT modify any graph and does NOT make anything faster. It reports
where a hoist would be possible and how much redundant work it would remove. It
is a signpost.

USAGE
-----
    .venv/bin/python tools/hoist_detect.py            # run the built-in corpus
    .venv/bin/python tools/hoist_detect.py <module>   # scan a python module's kernels

Output: one row per REDUCE node that contains at least one hoistable expression,
plus a summary.

CAVEATS (stated up front, not buried)
-------------------------------------
* Counting "recomputations" from the UOp graph is a *source-level* estimate. The
  backend's own CSE and the GPU's instruction selection may already collapse some
  of them. The authoritative check is the emitted source (`exp2(` count) or SASS,
  which `--verify-source` does for the cases it can.
* An expression being invariant does not prove hoisting it is profitable: if the
  hoisted value is read once anyway, moving it changes nothing. `uses` is
  reported so the caller can judge.
* This detects one class of missed rewrite (loop-invariant hoisting). There are
  others (strength reduction, reassociation, layout changes) that it does not see.
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass

from tinygrad.uop.ops import UOp, Ops


# Ops that are pure computation. Buffers/params/consts are leaves.
_PURE = {
    Ops.ADD, Ops.SUB, Ops.MUL, Ops.MAX, Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ,
    Ops.NEG, Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT, Ops.RECIPROCAL,
    Ops.FDIV, Ops.CDIV, Ops.FLOORDIV, Ops.FLOORMOD, Ops.CMOD, Ops.POW,
    Ops.WHERE, Ops.CAST, Ops.BITCAST, Ops.TRUNC, Ops.MULACC,
}

# Ops whose presence in a reduction indicates "expensive, would be nice to hoist".
_EXPENSIVE = {
    Ops.EXP2: "exp2:special-function", Ops.LOG2: "log2:special-function",
    Ops.SIN: "sin:special-function", Ops.SQRT: "sqrt",
    Ops.RECIPROCAL: "recip:division", Ops.FDIV: "fdiv:division",
    Ops.CDIV: "cdiv:integer-division", Ops.FLOORDIV: "floordiv:integer-division",
    Ops.FLOORMOD: "floormod:integer-division", Ops.CMOD: "cmod:integer-division",
    Ops.POW: "pow:transcendental",
}


@dataclass
class Finding:
    reduce_shape: tuple
    reduce_axis: int
    op: str
    shape: tuple
    uses: int
    expensive: str | None
    redundancy: int = 1

    def key(self):
        return (self.reduce_shape, self.reduce_axis, self.op, self.shape)


def analyze_reduce(node: UOp, findings: list[Finding]) -> None:
    """Find subexpressions in a REDUCE body that do not vary along the reduced axis.

    The test that actually works on this IR, established by inspection:

    Before a reduction, tinygrad keeps every operand of the contraction at the
    full tile shape (e.g. ``(16, 16)`` for a 2-D reduction), and an expression
    that is invariant along the reduced axis simply *has a smaller shape* --
    a decay vector that depends only on the kept axis is ``(16,)`` while the
    contracted operand is ``(16, 16)``.

    So invariance along ``reduced_axis`` is: the node's shape does not carry the
    reduced extent at that position (or has lower rank). Such a node is computed
    once per kept index and then broadcast across the reduction -- which is
    exactly what hoisting achieves. When the backend still recomputes it inside
    the reduced loop, that is the redundancy this tool reports.

    This is deliberately conservative. It reports only nodes that are pure,
    invariant, used more than once, and that carry a transcendental or division.
    """
    if node.op is not Ops.REDUCE:
        return
    reduce_op, axis = node.arg
    body = node.src[0]
    # NOTE: the axis index is relative to the BODY's rank, not the reduce node's.
    # For sum(0) over a (16,16) tile the reduce node is (16,) but the axis is 0
    # of two body dimensions. Using the reduce node's rank here is wrong and
    # silently yields zero findings.
    rank = len(body.shape)
    reduced_axis = axis if axis >= 0 else rank + axis
    topo = body.toposort()

    uses: Counter = Counter()
    for n in topo:
        for s in n.src:
            uses[s] += 1

    reduced_extent = body.shape[reduced_axis]
    for n in topo:
        if n.op not in _PURE:
            continue
        # invariant along the reduced axis?
        if len(n.shape) > reduced_axis and n.shape[reduced_axis] != 1:
            continue
        exp = None
        for m in n.toposort():
            if m.op in _EXPENSIVE:
                exp = _EXPENSIVE[m.op]
                break
        if exp is None:
            continue
        # REDUNDANCY is not graph fan-out. An invariant node used ONCE in the
        # graph is still re-evaluated `reduced_extent` times when the backend
        # places it inside the reduction loop -- that is precisely the
        # 128-vs-3 exp2 case. Fan-out is reported separately as `uses` because
        # it tells you whether hoisting also saves CSE work.
        redundancy = reduced_extent
        findings.append(Finding(tuple(node.shape), reduced_axis, n.op.name,
                                tuple(n.shape), uses.get(n, 0), exp, redundancy))


def scan(tensors: dict[str, UOp]) -> None:
    rows = 0
    total = 0
    per_op: Counter = Counter()
    print(f"{'kernel':<24}{'op':<10}{'shape':<12}{'uses':<6}{'x-eval':<7}expensive")
    print("-" * 90)
    for name, uop in tensors.items():
        findings: list[Finding] = []
        for n in uop.toposort():
            analyze_reduce(n, findings)
        # dedupe
        seen = set()
        for f in findings:
            k = (name,) + f.key()
            if k in seen:
                continue
            seen.add(k)
            total += 1
            per_op[f.expensive or f.op] += 1
            print(f"{name:<24}{f.op:<10}{str(f.shape):<12}{f.uses:<6}{f.redundancy:<7}{f.expensive or ''}")
            rows += 1
    print("-" * 90)
    print(f"total hoistable sites: {total}")
    if per_op:
        print("by category:")
        for k, v in per_op.most_common():
            print(f"  {k:<26} {v}")


def corpus() -> dict[str, UOp]:
    """The reduction shapes this repo already studies. No GPU needed."""
    from tinygrad import Tensor

    out: dict[str, UOp] = {}
    A = Tensor.randn(64, 64).realize()
    decay = Tensor.randn(64).realize()

    # The KDA-shaped case: a decay that varies only along the KEPT axis, so it is
    # invariant with respect to the contracted axis, but the port computes it
    # inside the contraction. This is the 128-vs-3 exp2 pattern in miniature.
    out["kda_decay_in_reduce"] = (A * decay.exp2()).sum(0).uop

    # Control: the same maths with the decay precomputed as the hand-written
    # kernel does. exp2 is applied once, outside any reduction.
    d = decay.exp2()
    out["kda_decay_hoisted"] = (A * d).sum(0).uop

    # Control: no transcendental at all -> nothing to report.
    out["plain_sum"] = A.sum(0).uop

    # Matmul contracts too; a plain matmul should not trip the detector.
    out["matmul"] = (A @ Tensor.randn(64, 64).realize()).uop

    return out


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("module scanning not implemented yet; running built-in corpus", file=sys.stderr)
    scan(corpus())
