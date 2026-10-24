"""Batch-scan the repo's real kernel expressions with factor_lint."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'kda'))

import factor_lint as FL

CASES = []

def kda_gram():
    from tinygrad import Tensor
    C, K = 64, 128
    k = Tensor.empty(C, K); g = Tensor.empty(C, K)
    return (g[:, None, :] - g[None, :, :]).exp()

CASES.append(("kda_gram_decay [C,C,K]", kda_gram()))

def kda_full():
    from tinygrad import Tensor
    C, K = 64, 128
    k = Tensor.empty(C, K); g = Tensor.empty(C, K)
    d = (g[:, None, :] - g[None, :, :]).exp()
    return (d * k[:, None, :] * k[None, :, :]).sum(-1)

CASES.append(("kda_full_gram (sum form)", kda_full()))

def minecu_dda():
    # the voxel raymarcher pattern from notes/19: address recomputed per step
    from tinygrad import Tensor
    S, N = 32, 64
    origin = Tensor.empty(N); d = Tensor.empty(S, N)
    t = (d - origin).exp()      # pairwise difference
    return t

CASES.append(("raymarchet_tmax_diff", minecu_dda()))

def negative():
    from tinygrad import Tensor
    return Tensor.empty(64, 128).exp()

CASES.append(("control: single-array exp", negative()))

print(f"{'case':<28}{'hits':<6}{'out shape':<18}{'exp2 evals':<12}{'after factor':<14}")
print("-"*82)
for name, t in CASES:
    hits = []
    for n in t.uop.toposort():
        FL.analyze_exp(n, hits)
    if not hits:
        print(f"{name:<28}{0:<6}")
        continue
    h = hits[0]
    import math
    prod = 1
    for s in h.out_shape: prod *= s
    # factored cost ~ 2 * (numel of the invariant factor)
    inv = h.left_shape if len(h.left_shape)==len(h.out_shape) else h.right_shape
    ip = 1
    for s in inv: ip *= s
    print(f"{name:<28}{len(hits):<6}{str(h.out_shape):<18}{prod:<12}{2*ip:<14}")
