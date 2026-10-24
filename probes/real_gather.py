"""real_gather.py - a drop-in replacement for TinyGrad's indexed load.

STATUS: works as a gather, but is NOT the fix, and this file explains why.
Kept because the negative result is the interesting part.

WHAT IT DOES
    blaze/core/obs_camera.h gathers one voxel per ray per DDA step out of a
    113x104x113 = 1,328,176-cell region tensor. TinyGrad emits an extra
    reduce-shaped kernel for that, costing 101 extra kernels per camera frame
    (37.6% of GPU time, see notes/01-obs-camera.md Finding 3).

    Tensor.custom_kernel takes a callable and passes the OUTPUT tensor as the
    first argument -- `UOp.custom_kernel(*srcs)` is called as
    `custom_kernel(self, *lst)`, and the first positional is the output. So a
    drop-in `__getitem__` is:

        def getitem(self, indices):
            if isinstance(indices, Tensor) and indices.dtype == dtypes.int32 and self.ndim == 1:
                out = Tensor.empty(indices.shape, dtype=self.dtype, device=self.device)
                return Tensor.custom_kernel(out, self, indices,
                                            fxn=lambda o, a, b: o.store(a[b]))[0]
            return orig_getitem(self, indices)

WHY IT DOES NOT COMPILE
    Two independent failures, both about how TinyGrad represents an address:

    1. Reaching a real indexed load from the Tensor API fails in
       codegen/late/coalesce.py:156 with
           AssertionError: attempting multiple stores: 6
       The index tensor has to be materialised before it can serve as an
       address, and memory_coalescing groups by (op, buf, root_src, valid) --
       so six different index values become six stores to one key, which the
       pass asserts cannot happen. It cannot happen for affine or strided
       addressing. A runtime permutation is exactly the case it cannot express.

    2. DMC=1 (which disables memory_coalescing outright) gets past that and
       then fails in the Metal compiler:
           error: invalid operands to binary expression ('device int *' and 'device int *')
       With the coalescer off, the index stays a raw pointer and the generated
       MSL does pointer arithmetic on it, which MSL does not allow for
       `device` (as opposed to `thread`) address space.

    So the capability is NOT missing from the backend: cstyle.py:50 renders
    Ops.INDEX as a native address and ops_python.py implements it. The gap is
    that TinyGrad's memory model wants addresses affine in the thread index,
    and a gather is by definition not. Fixing this means teaching
    memory_coalescing to bucket a non-affine index -- a real TinyGrad change,
    not a plugin.

    Consequence for us: the honest cost model for the DDA is "one extra kernel
    per indexed load, plus a pass over the source once it leaves cache"
    (see probes/gather_floor.py), and the port-level fixes that matter are
    reducing the NUMBER of gathers, not making each one cheaper. That is items
    1 and 2 of notes/01-obs-camera.md section 4.
"""
import numpy as np
from tinygrad import Tensor, dtypes


def _gather_real(self: Tensor, indices: Tensor) -> Tensor:
    """self[indices] with one thread per output element."""
    out = Tensor.empty(indices.shape, dtype=self.dtype, device=self.device)
    return Tensor.custom_kernel(out, self, indices,
                                fxn=lambda o, a, b: o.store(a[b]))[0]


def patch_gather(verbose: bool = False):
    """Route 1-D int32 fancy indices through custom_kernel.

    Only intercepts the case the DDA needs. Everything else is left to stock
    TinyGrad, so this cannot turn a correct graph into a wrong one.
    """
    orig = Tensor.__getitem__

    def getitem(self, indices):
        if (isinstance(indices, Tensor) and indices.dtype == dtypes.int32
                and self.ndim == 1 and self.device == indices.device):
            if verbose:
                print(f"[real_gather] {self.numel:,}-elem source, "
                      f"{indices.numel:,}-elem index")
            return _gather_real(self, indices)
        return orig(self, indices)

    Tensor.__getitem__ = getitem
    return orig


def unpatch_gather(orig):
    Tensor.__getitem__ = orig


def test_gather_matches_stock():
    """This plugin is a correct gather, just not a faster one. Prove it."""
    import os
    import subprocess
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    # The kernel needs DMC=1 to compile at all (see the docstring), so the test
    # runs in a subprocess rather than mutating this process's env.
    snip = f"""
import sys; sys.path.insert(0, {here!r})
import numpy as np
from tinygrad import Tensor, dtypes
from real_gather import _gather_real
rng = np.random.default_rng(0)
ok = True
for src_n, idx_n, oob in [(20,6,False),(1000,333,False),(65536,4096,False),
                          (300000,1000,False),(20,6,True),(5000,777,True)]:
    src = rng.integers(0,4096,src_n).astype(np.int32)
    idx = rng.integers(0,src_n,idx_n).astype(np.int32)
    if oob: idx[rng.integers(0,idx_n,5)] = src_n + 7
    stock = Tensor(src)[Tensor(idx)].numpy()
    got   = _gather_real(Tensor(src), Tensor(idx)).numpy()
    same  = np.array_equal(got, stock)
    ok &= same
    print(f"  src={{src_n:>7,}} idx={{idx_n:>5,}}{{' out-of-range' if oob else ''}}: "
          f"{{'ok' if same else 'MISMATCH'}}")
print("ALLPASS" if ok else "FAILURES")
"""
    env = dict(os.environ, DMC="1")
    p = subprocess.run([sys.executable, "-c", snip], capture_output=True,
                       text=True, env=env)
    for line in p.stdout.splitlines():
        if "src=" in line or "ALLPASS" in line or "FAILURES" in line:
            print(line)
    if "ALLPASS" not in p.stdout:
        err = [l for l in p.stderr.splitlines() if "Error" in l or "error" in l]
        for l in err[-4:]:
            print("   ", l.strip()[:140])
    return "ALLPASS" in p.stdout


if __name__ == "__main__":
    print("real_gather: can a custom_kernel replace TinyGrad's indexed load?")
    print("Answer: no. Both failure modes are documented in the docstring.\n")
    ok = test_gather_matches_stock()
    print()
    print("confirmed broken" if not ok else
          "unexpectedly compiled -- re-check the docstring, this may now be fixable")
