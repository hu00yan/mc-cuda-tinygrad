"""run_oc.py - correctness + throughput for the TinyGrad semantic camera.

Correctness is against ref_oc, the C reference compiled from the SAME
blaze/core/obs_camera.h. ref.bin holds, per (seed, pose), 2304 u16 ids,
2304 u8 depth, 2304 u8 edge -- the exact layout blaze/cuda/obs_camera.cu
prints as hex, so a match here is a match against their CUDA kernel too.
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tinygrad import Tensor, Device
from tiny_oc import (OC_NPIX, OC_W, OC_H, OC_STEPS, RG_STRIDE, RG, render,
                     sin_table, load_regions, SinTable)
from bench import bench_ms

# The 6 poses blaze/cuda/obs_camera.cu uses, at its eye.
CANON = [(180.0, 0.0), (0.0, 0.0), (90.0, 15.0), (270.0, -30.0), (45.0, 60.0),
         (200.0, -75.0)]
EYE = (8.5, 80.0, 8.5)


def read_ref(path):
    d = open(path, "rb").read()
    assert d[:7] == b"OCREF1\0", d[:8]
    npose = int.from_bytes(d[7:11], "little")
    nseed = int.from_bytes(d[11:15], "little")
    body = np.frombuffer(d[16:], dtype=np.uint8).reshape(nseed, npose, 9216)
    out = []
    for s in range(nseed):
        for p in range(npose):
            blk = body[s, p]
            out.append((blk[:4608].view(np.uint16).astype(np.float32),
                        blk[4608:6912].astype(np.float32),
                        blk[6912:].astype(np.float32)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="ref.bin")
    ap.add_argument("--seeds", default="0,3,12345")
    ap.add_argument("--steps", type=int, default=OC_STEPS)
    ap.add_argument("--bench", type=int, default=0,
                    help="if >0, also time this many envs in lockstep")
    ap.add_argument("--check-steps", action="store_true")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    sin = SinTable(sin_table())

    # ---------- correctness, one env per (seed, pose) at a time ----------
    ref = read_ref(args.ref)
    print(f"device={Device.DEFAULT}  ref entries={len(ref)}")
    tot_px = 0
    bad_id = bad_dep = bad_edge = 0
    step_hist = []
    for (seed, (yaw, pitch)), (r_id, r_dep, r_edge) in zip(
            [(s, c) for s in seeds for c in CANON], ref):
        cells = load_regions([f"cells_{seed}.bin"])
        ids, depth, edge, st = render(cells, [EYE], [yaw], [pitch], sin,
                                      steps=args.steps, want_steps=True)
        g_ids, g_dep, g_edge, g_st = (x.numpy() for x in (ids, depth, edge, st))
        d_id = int((g_ids[0] != r_id).sum())
        d_dep = int((g_dep[0] != r_dep).sum())
        d_edge = int((g_edge[0] != r_edge).sum())
        bad_id += d_id; bad_dep += d_dep; bad_edge += d_edge
        tot_px += OC_NPIX
        step_hist.append(g_st[0])

    step_hist = np.concatenate(step_hist)
    print(f"\nCORRECTNESS  {tot_px} pixels over {len(ref)} frames")
    print(f"  id   mismatched: {bad_id:6d}  ({bad_id/tot_px:.3%})")
    print(f"  depth mismatched: {bad_dep:6d}  ({bad_dep/tot_px:.3%})")
    print(f"  edge  mismatched: {bad_edge:6d}  ({bad_edge/tot_px:.3%})")

    print(f"\nDDA STEP DEPTH  (unroll={args.steps})")
    print(f"  mean {step_hist.mean():.1f}  max {step_hist.max():.0f}  "
          f"p99 {np.percentile(step_hist,99):.0f}  "
          f"unroll waste {args.steps/step_hist.mean():.1f}x")

    if args.check_steps:
        assert step_hist.max() < args.steps, "unroll bound is WRONG"
        print(f"  bound holds: max step {step_hist.max():.0f} < {args.steps}")

    # ---------- throughput: B envs, one frame each ----------
    if args.bench:
        B = args.bench
        cells = load_regions([f"cells_{i}.bin" for i in range(B)])
        rng = np.random.default_rng(0)
        yaws = rng.uniform(0, 360, B).astype(np.float32)
        pitch = rng.uniform(-89, 89, B).astype(np.float32)
        eyes = np.tile(np.array(EYE, dtype=np.float32), (B, 1))
        print(f"\nTHROUGHPUT  B={B} envs x {OC_W}x{OC_H} = {B*OC_NPIX:,} rays/frame")
        print(f"  cells buffer: {B*RG_STRIDE*4/1e6:.0f} MB int32 "
              f"(netherite's u16 would be {B*RG_STRIDE*2/1e6:.0f} MB)")
        build = lambda: render(cells, eyes, yaws, pitch, sin, steps=args.steps)[0]
        ms = bench_ms(build, reps=5, batches=3)
        rays = B * OC_NPIX
        print(f"  wall, graph rebuilt each call : {ms:9.2f} ms/frame  "
              f"{rays/ms/1e3:8.2f} M rays/s")
        render(cells, eyes, yaws, pitch, sin, steps=args.steps)[0].numpy()


if __name__ == "__main__":
    main()
