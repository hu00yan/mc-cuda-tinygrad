"""provenance.py -- the record that makes a number attributable.

WHAT THIS IS FOR
    A measurement that cannot be traced to a tree state is not a measurement.
    So every run writes one JSON file next to its artefacts, containing:

      the manifest fingerprint       which tree produced this, byte for byte
      the file count and bytes       so the number can be sanity-checked
      the backend and its transport  what kind of machine this is
      the resolved device            read off the machine, never assumed
      driver and CUDA runtime       from the same read
      the exact command              argv, verbatim, not a paraphrase
      the run id, timings, exit code what happened
      the artefacts and their hashes so the link to disk is checkable

WHAT IT IS NOT
    It does not interpret the device. It records the compute capability that
    the machine reported. Whether sm_75 matters for a given kernel is the
    study's question; a transport that answered it would be making a claim
    about code it has not read.

WHY IT IS WRITTEN EVEN ON FAILURE
    Because the failures are the ones worth keeping. A run that was refused for
    a CPU-only session, or that disconnected mid-command, produced no
    measurement -- but it did produce evidence, and the next person to hit the
    same wall should not have to.
"""
from __future__ import annotations

import hashlib
import os
import platform
import sys

from . import PROVENANCE_SCHEMA, __version__
from .manifest import Manifest
from . import util


def host_facts() -> dict:
    """Facts about the machine that *planned* the run.

    Separate from the remote device on purpose: "which Mac pushed this" and
    "which GPU ran this" are different questions and conflating them is how a
    Metal number ends up attributed to a T4.
    """
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "gpush": __version__,
    }


def new_run_dir(out_root: str, backend: str, target_slug: str,
                fingerprint: str) -> str:
    """A directory named so that sorting it sorts by time and shows the tree."""
    slug = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in target_slug) or "run"
    d = os.path.join(out_root, "runs", backend, slug,
                     f"{util.stamp()}-{fingerprint[:12]}")
    os.makedirs(d, exist_ok=True)
    return d


def write(dest: str, *, backend, tree: Manifest, plan, command: list[str],
          device: dict, run_id: str, status: str, started: str, finished: str,
          exit_code, remote: dict, gates: list, artefacts: list[str],
          note: str = "") -> str:
    """Write `provenance.json` and `manifest.json` into `dest`; return the path."""
    rec = {
        "schema": PROVENANCE_SCHEMA,
        "gpush_version": __version__,
        "run_id": run_id,
        "backend": backend.name,
        "transport": backend.transport,
        "target": (remote.get("dataset") or remote.get("remote_dir")
                  or remote.get("host") or ""),
        "remote": remote,

        # the tree that produced this, and nothing else
        "fingerprint": tree.fingerprint,
        "files": len(tree),
        "bytes": tree.total_bytes,
        "manifest_source": tree.source,
        "command": list(command),
        "command_line": " ".join(command),

        # what the machine actually said about itself
        "device": device,
        # sm_75, not sm_7.5: nvidia-smi reports compute capability as a dotted
        # pair and the architecture name drops the dot.
        "compute_capability_reported": ("sm_" + str(device.get("compute_cap"))
                                        .replace(".", "")
                                        if device.get("compute_cap")
                                        not in (None, "", "unknown")
                                        else "unknown"),

        "started": started,
        "finished": finished,
        "exit_code": exit_code,
        "status": status,
        "note": note,

        "gates": [{"name": n, "ok": ok, "detail": d} for n, ok, d in gates],
        "plan": {"added": len(plan.added), "modified": len(plan.modified),
                 "deleted": len(plan.deleted), "unchanged": len(plan.unchanged),
                 "base_fingerprint": plan.base_fingerprint,
                 "new_fingerprint": plan.new_fingerprint} if plan else None,

        "host": host_facts(),
        "artefacts": [],
    }
    for a in artefacts or []:
        try:
            rec["artefacts"].append({
                "path": os.path.relpath(a, dest),
                "bytes": os.path.getsize(a),
                "sha256": _sha(a),
            })
        except OSError:
            continue
    p = os.path.join(dest, "provenance.json")
    util.write_json(p, rec)
    util.write_json(os.path.join(dest, "manifest.json"), tree.as_dict())
    return p


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: str) -> dict:
    return util.read_json(path, {}) or {}
