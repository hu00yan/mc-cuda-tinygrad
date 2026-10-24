"""verify_defs.py - prove the corrected report's categories are the same
categories run_oc.py counts, and re-derive the quoted numbers from the files.

The prompt handed me "41,472 pixels, 0 id mismatches, 9 depth, 3 edge" and a
"prior local run: 0/9/3".  A report that silently changes what "depth" means
is worse than one that fails, so nothing here is taken on trust: the pixel
count comes from the ref.bin header, and the three mismatch counts are
computed twice, by two independently written loops that index the same arrays
differently, and then compared.

  run:  prove_definitions_match()  -- report loop vs run_oc loop, per frame
  main: print the re-derived numbers, and diff the two per-frame vectors
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OC_DIR = os.path.dirname(HERE)
for _p in (OC_DIR, os.path.dirname(OC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tiny_oc import OC_NPIX, SinTable, sin_table, render, load_regions  # noqa: E402
from run_oc import read_ref, CANON, EYE                                 # noqa: E402
import oc_report_fixed as rep                                           # noqa: E402

SEEDS = [0, 3, 12345]
NPOSE = len(CANON)


def per_frame_report(cells_dir, ref):
    """Exactly the corrected report's loop, keeping per-frame vectors."""
    sin = SinTable(sin_table())
    out = []
    for si, s in enumerate(SEEDS):
        c = load_regions([rep.cell_path(s, cells_dir)])
        for p in range(NPOSE):
            r = ref[si * NPOSE + p]
            gi, gd, ge, _ = render(c, [EYE], [CANON[p][0]], [CANON[p][1]], sin,
                                   want_steps=True)
            gi, gd, ge = gi.numpy()[0], gd.numpy()[0], ge.numpy()[0]
            out.append((int((gi != r[0]).sum()),
                        int((gd != r[1]).sum()),
                        int((ge != r[2]).sum())))
    return np.array(out)


def per_frame_run_oc(cells_dir, ref, steps=96):
    """Exactly run_oc.py's loop, flattened differently (a generator zip) so a
    transcription slip on one side does not cancel out on the other."""
    sin = SinTable(sin_table())
    acc = []
    for (seed, (yaw, pitch)), (r_id, r_dep, r_edge) in zip(
            [(s, c) for s in SEEDS for c in CANON], ref):
        cells = load_regions([rep.cell_path(seed, cells_dir)])
        g_ids, g_dep, g_edge, _ = render(cells, [EYE], [yaw], [pitch], sin,
                                         steps=steps, want_steps=True)
        acc.append((int((g_ids.numpy()[0] != r_id).sum()),
                    int((g_dep.numpy()[0] != r_dep).sum()),
                    int((g_edge.numpy()[0] != r_edge).sum())))
    return np.array(acc)


def main():
    cells_dir = sys.argv[1] if len(sys.argv) > 1 else OC_DIR
    ref = read_ref(os.path.join(OC_DIR, "ref.bin"))

    # --- re-derive 41,472 from the file header, not from the docstring ---
    d = open(os.path.join(OC_DIR, "ref.bin"), "rb").read()
    assert d[:7] == b"OCREF1\0", d[:8]
    npose_h = int.from_bytes(d[7:11], "little")
    nseed_h = int.from_bytes(d[11:15], "little")
    body = len(d) - 16
    per_frame = 2304 * 2 + 2304 + 2304          # u16 ids + u8 depth + u8 edge
    print(f"ref.bin header      : {nseed_h} seeds x {npose_h} poses")
    print(f"body bytes          : {body} = {body/per_frame:.0f} frames "
          f"(x {per_frame} B/frame) -> {body/per_frame == npose_h*nseed_h}")
    npix = (body // per_frame) * OC_NPIX
    print(f"re-derived pixels   : {npix:,}   (report prints "
          f"{len(ref) * OC_NPIX:,}) -> {npix == len(ref)*OC_NPIX}")
    print(f"3 seeds x 6 poses x {OC_NPIX} = {3*6*OC_NPIX} -> "
          f"{npix == 3*6*OC_NPIX}")

    a = per_frame_report(cells_dir, ref)
    b = per_frame_run_oc(cells_dir, ref)
    print()
    print("per-frame mismatch vectors, report loop vs run_oc loop:")
    print(f"  frames                 : {a.shape[0]} vs {b.shape[0]}")
    print(f"  identical entry-wise   : {bool((a == b).all())}")
    for i, k in enumerate(("id", "depth", "edge")):
        fa = a[a[:, i] != 0]
        fb = b[b[:, i] != 0]
        where = np.nonzero(a[:, i])[0]
        print(f"  {k:<6} totals         : report {a[:,i].sum():3d}  run_oc "
              f"{b[:,i].sum():3d}   nonzero frames {where.tolist()} "
              f"counts {a[where, i].tolist()}")

    print()
    print("re-derived totals      : id %d  depth %d  edge %d  of %d pixels"
          % (a[:, 0].sum(), a[:, 1].sum(), a[:, 2].sum(), npix))
    print("percentages            : depth %.3f%%  edge %.3f%%"
          % (100*a[:, 1].sum()/npix, 100*a[:, 2].sum()/npix))
    if not (a == b).all():
        sys.stderr.write("MISMATCH between the two definitions\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
