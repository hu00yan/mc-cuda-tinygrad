"""Census and capture of the code TinyGrad actually generates.

Three questions, and the machinery for each:

1. **How many kernels does one render launch?** `GlobalCounters.kernel_count`,
   which is the count `DEBUG=2` reports and the same counter `perf_kda.py` used
   for KDA's 12,082.

2. **How many distinct compiled programs is that?** `len(to_program_cache)`
   deltas. Distinct programs is not launches: a loop that unrolls to the same
   shape twice reuses one program. Both numbers are reported, and the
   difference between them is stated rather than left for the reader.

3. **What is the generated CUDA C?** Captured on the real backend by wrapping
   `Device[DEFAULT].compiler.compile_cached`.

WHY `PARALLEL=0` IS MANDATORY AND NOT AN OPTIMISATION
------------------------------------------------------
`tinygrad/engine/realize.py:273`:

    pool = None if len(todo) == 1 or any(beam) else get_worker_pool()

`lower_and_compile` compiles in a **multiprocessing worker pool** whenever more
than one kernel needs compiling, and only takes the in-process `map` branch
when exactly one is outstanding. `compile_cached` runs inside whichever process
does the work, so an in-process hook on the parent sees only the `len(todo)==1`
rounds. That is why `kernels/build_notebooks.py:578` captured 1 kernel out of
12,082 for KDA (`notes/13-schedule-vs-decay.md` §4). `PARALLEL=0` makes
`get_worker_pool()` return None, so everything compiles in-process and the hook
sees the whole schedule. This module sets it and asserts it took effect.

On the CPU backend there is no `compile_cached` (the CPU runtime is not a
compiler), so the source capture is unavailable there and is reported as such
rather than silently skipped. The kernel *counts* work on every backend.
"""

from __future__ import annotations

import collections
import json
import os
from pathlib import Path


# module-level so uninstall can find what install created
CAPTURE_STATE: list[dict] = []


def install_launch_recorder() -> list[str]:
    """Record the name of every kernel launch, in order.

    This is the same census point `DEBUG=2` prints from -- the per-launch line
    in `tinygrad/engine/realize.py:track_stats` -- reached by wrapping the
    device's `runtime_t.__call__` instead of by setting `DEBUG=2`. The reason
    for preferring the hook: `DEBUG>=2` makes `run_linear` pass `wait=True`
    (`realize.py:318`), so every launch becomes a host-synchronised event. The
    *count* is the same either way, but a census that also reports timing must
    not be the instrumented one.

    Names come from the ProgramInfo attached to the compiled AST, so an entry
    like `E_32_32_64_8_4` is tinygrad's own summary of the kernel: the element
    sizes and ranks of its bounds. Two launches with the same name are the same
    compiled program reused, which is exactly the repetition the DDA loop
    prediction is about.
    """
    from tinygrad import Device

    dev = Device[Device.DEFAULT]
    base = dev.runtime_t if isinstance(dev.runtime_t, type) else type(dev.runtime_t)
    orig = base.__call__
    names: list[str] = []

    def spy(self, *a, **k):
        # Two things per launch, and the second is the one that matters:
        #   * the name -- every backend's Program carries the compiled name
        #     (CPUProgram.name, HCQ2Compiled.name). NOTE the name is derived
        #     from the kernel's *bounds*, not its body, so a run of launches
        #     that share a name can still be a run of DIFFERENT programs. That
        #     is exactly the trap this census exists to avoid.
        #   * the program object id -- `get_runtime` caches one Program per
        #     compiled AST, so within one measurement two launches with the
        #     same id are the same code, and different ids are different code.
        # The name falls back to the repr so a miss is visible, not dropped.
        names.append(f"{getattr(self, 'name', None) or repr(self)}#{id(self) & 0xffffff:06x}")
        return orig(self, *a, **k)

    base.__call__ = spy
    CAPTURE_STATE.append({"cls": base, "attr": "__call__", "orig": orig})
    return names


def uninstall(sink) -> None:
    while CAPTURE_STATE:
        h = CAPTURE_STATE.pop()
        if "cls" in h:
            h["cls"].__call__ = h["orig"]
        else:
            try:
                del h["comp"].__dict__["compile_cached"]
            except (KeyError, AttributeError):
                h["comp"].compile_cached = h["orig"]


def install_capture(capture_only: bool = False) -> tuple[list[str], str | None]:
    """Wrap `compile_cached` on the default device so every kernel's source is
    captured. Returns (sink, note). `note` explains an empty sink.

    `capture_only=True` returns an empty "binary" instead of compiling, for the
    NULL-device case where the host has no nvcc and the program is never
    executed. It changes nothing about the *generated source*, which is
    produced before the compile call, but it does mean no cubin exists for this
    particular capture: the SASS matrix in the notes comes from the T4 run and
    from the offline `archlab` route, both of which do compile.
    """
    from tinygrad import Device

    dev = Device[Device.DEFAULT]
    comp = getattr(dev, "compiler", None)
    orig = getattr(comp, "compile_cached", None) if comp is not None else None
    if orig is None:
        return [], (f"backend {Device.DEFAULT!r} has no compiler.compile_cached; "
                    "launch and program counts are still valid, source capture is not")
    sink: list[str] = []
    holder = {}

    def spy(src, *a, **k):
        sink.append(str(src))
        return b"" if capture_only else orig(src, *a, **k)

    holder["comp"], holder["orig"] = comp, orig
    comp.compile_cached = spy
    CAPTURE_STATE.append(holder)
    return sink, None


def census(render_fn, shapes, out_dir: Path | None = None,
           keep_sources: bool = True, capture_only: bool = False) -> dict:
    """Count kernels per render, for each max_steps in `shapes`.

    `render_fn(max_steps)` must build the graph. The tensor is **realized**, not
    read back: a `.numpy()` would add a device-to-host copy to the launch count,
    and that copy belongs to the harness, not to the render. On the NULL device
    a read-back is impossible at all (`ops_null.py:28`).

    Each shape reports, in order: the launch count (GlobalCounters), the number
    of distinct compiled programs, and the ordered list of launch names with
    multiplicities -- so "31 launches of one identical kernel" is visible as a
    fact rather than inferred.
    """
    from tinygrad import Device, GlobalCounters
    from tinygrad.codegen import to_program_cache

    sink, note = install_capture(capture_only)
    launches_log = install_launch_recorder()
    rows = []
    try:
        for ms in shapes:
            n0 = len(to_program_cache)
            sink.clear()
            launches_log.clear()
            GlobalCounters.reset()
            out = render_fn(ms)
            out = out[0] if isinstance(out, tuple) else out
            n_elem, n_bytes = 1, out.dtype.itemsize
            for d in out.shape:
                n_elem *= d
            out.realize()
            launches = GlobalCounters.kernel_count
            n_new = len(to_program_cache) - n0
            names = [n for n in launches_log]
            counts = collections.Counter(names)
            row = {
                "max_steps": ms,
                "kernel_launches": launches,
                "new_distinct_programs": n_new,
                "new_sources_captured": len(sink),
                "uops_total_graph": GlobalCounters.global_ops,
                "bytes_estimate": GlobalCounters.global_mem,
                "obs_shape": list(out.shape),
                "obs_bytes": n_elem * n_bytes,
                "launch_keys_ordered": names,
                "launch_counts_by_program": dict(counts),
                "distinct_programs_launched": len(counts),
                "n_repeated_program": max(counts.values(), default=0),
                "n_repeated_program_key": max(counts, key=counts.get) if counts else None,
                "n_distinct_names": len({k.split('#')[0] for k in names}),
            }
            if out_dir is not None and keep_sources and sink:
                d = out_dir / f"max_steps_{ms:04d}"
                d.mkdir(parents=True, exist_ok=True)
                for i, src in enumerate(sink):
                    (d / f"kernel{i:03d}.cu").write_text(src)
                row["sources_dir"] = str(d)
                row["source_bytes"] = [len(s) for s in sink]
            rows.append(row)
    finally:
        uninstall(sink)

    return {"device": Device.DEFAULT, "parallel": os.environ.get("PARALLEL"),
            "capture_note": note, "rows": rows}


def summarise(c: dict) -> str:
    lines = [f"device={c['device']}  PARALLEL={c['parallel']}"]
    if c["capture_note"]:
        lines.append(f"  capture: {c['capture_note']}")
    lines.append(f"{'max_steps':>10} {'launches':>9} {'distinct':>8} "
                 f"{'names':>6} {'top-rep':>8} {'uops':>12}")
    for r in c["rows"]:
        lines.append(f"{r['max_steps']:>10} {r['kernel_launches']:>9} "
                     f"{r['distinct_programs_launched']:>8} {r['n_distinct_names']:>6} "
                     f"{r['n_repeated_program']:>8} {r['uops_total_graph']:>12}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--steps", default="1,2,4,8,16,32,64")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    print(summarise(census(None, [int(x) for x in a.steps.split(",")], a.outdir)))
