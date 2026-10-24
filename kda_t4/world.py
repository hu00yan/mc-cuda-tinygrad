#!/usr/bin/env python3
"""
world.py -- one input generator, used by every side of every measurement here.

WHY THIS FILE EXISTS AND WHY IT IMPORTS THE STUDY'S OWN
-------------------------------------------------------
A wall-clock comparison is only worth anything if both sides were handed the
same numbers.  Rather than write a second generator, this imports the study's:
`kda/kda_vs_fla.py:296` `rnd(B,T,H,HV,K,V,seed,gmax,dtype)`, with its `l2norm`
at `:255`.  That function's docstring carries a warning this file would otherwise
have to relearn the hard way:

    "q and k are L2-normalised, which is what the real KDA layer does and what
     keeps the operator well conditioned: with raw Gaussians, k^T k ~ K, the
     delta-rule matrix (I - beta k k^T) has eigenvalues near 1 - K*beta, and the
     state diverges."

An earlier version of `kda_t4/gate_observed.py` used raw Gaussians and got
`finite=False` out of fla.  That was the generator's fault, not fla's, and the
fix is to use the study's.  Recording it here because "the reference produced
NaN" is exactly the kind of thing that gets written down as a property of the
reference.

`g` is in natural-log space and negative; fla does its own chunk-local cumsum
with a 1/ln2 scale (`fla/ops/kda/chunk_fwd.py:58-64`, `chunk_local_cumsum(g,
scale=RCP_LN2)`), and the TinyGrad ports do theirs in natural log and call `exp`,
so the two sides agree on units without either of them rescaling.  `beta` is
already post-sigmoid, which is what `use_beta_sigmoid_in_kernel=False` means
(`fla/ops/kda/chunk.py:246-248`).

The dtype question, which the brief asks to be answered rather than assumed:
**bf16 is the only dtype this comparison can use.**  The FlashKDA verifier
requires `q.dtype == torch.bfloat16` (`flash_kda.py:64-88`), the study's own call
shape is bf16 (`kda/baseline_pin.py:392`), and fla's KDA builds fp32 gates and
masks around a bf16 payload, so fp16 is not a substitute that leaves the
computation alone -- it is a different payload type with a different numerics
story and a different set of compiled kernels.  It is recorded in the call-shape
block of every report so the reader can see it was a decision, not an accident.
"""

from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "kda")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                        # noqa: E402

DTYPE = "bfloat16"
GMAX = 0.7          # kda_vs_fla.py:296 default


def make(B, T, H, HV, K, V, seed=0, gmax=GMAX):
    """(q,k,v,g,beta) as float64 numpy, L2-normalised q and k. Identical on
    both sides, byte for byte, because it is the same function object."""
    from kda_vs_fla import rnd
    q, k, v, g, beta = rnd(B, T, H, HV, K, V, seed=seed, gmax=gmax,
                           dtype=np.float64)
    return q, k, v, g, beta


def to_torch(x, dtype=DTYPE, device="cuda"):
    import torch
    return torch.tensor(np.ascontiguousarray(x), dtype=getattr(torch, dtype),
                        device=device)


def scale_for(K):
    """Both sides default to 1/sqrt(K): fla says so at chunk.py:219, and
    kda/tiny_kda.py:100-101 computes `K ** -0.5`.  Passed explicitly anyway, so
    the value is a recorded input rather than a default that could drift."""
    return float(K) ** -0.5


def shapes(ladder):
    """(B,T,H,HV,K,V,chunk) for the shape ladder, as a list of dicts."""
    return [dict(B=B, T=T, H=H, HV=HV, K=K, V=V, C=C) for (B, T, H, HV, K, V, C) in ladder]
