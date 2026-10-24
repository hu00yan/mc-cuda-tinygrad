"""tarball.py -- deterministic packaging of a manifest into one archive.

WHY A TAR AT ALL
    Two reasons, one per backend. Kaggle's dataset API converts loose tabular
    files and gives a notebook a directory listing it cannot checksum; one
    archive gives one sha256 that identifies the entire payload, which is what
    the integrity gate compares. The SSH backend would be happy with loose
    files, but a tar over one pipe is both faster and trivially atomic, and one
    code path beats two.

WHY DETERMINISTIC
    mtime, uid, gid, uname, gname and mode are pinned, and entries are emitted
    in sorted path order. An unchanged tree therefore produces byte-identical
    bytes, which buys three things:
      * the Kaggle dataset API's "this version is identical to the previous
        one" rejection becomes a meaningful signal rather than noise -- it is
        then proof that nothing changed, not a race;
      * the sha256 in a provenance record can be compared against a run from
        another machine;
      * an unchanged re-push produces an identical upload and the no-op check
        can be trusted to have seen every byte.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile

from .manifest import Manifest


def build(manifest: Manifest, dest: str, *, extra: dict[str, bytes] | None = None,
          arc_prefix: str = "payload") -> tuple[str, str, int]:
    """Pack `manifest` into `dest`.

    extra  additional members written alongside, name -> bytes. Used for the
           attestation record the kernel reads on the runner.

    Returns (path, sha256, bytes).
    """
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    n = 0

    # gzip must be constructed by hand. `tarfile.open(..., "w:gz")` builds a
    # GzipFile with the CURRENT TIME in its header, so two packs of an
    # identical tree differ in their first ten bytes and every downstream
    # sha256 is noise. Passing mtime=0 is what makes the "unchanged tree ->
    # identical bytes" claim true, and that claim is load-bearing: it is what
    # lets Kaggle's "this version is identical" rejection mean something, and
    # what makes a fingerprint comparable across machines.
    import gzip
    with open(dest, "wb") as raw:
        gz = gzip.GzipFile(filename="", mode="wb", compresslevel=9,
                           fileobj=raw, mtime=0)
        try:
            tf = tarfile.open(fileobj=gz, mode="w", format=tarfile.GNU_FORMAT)
            with tf:
                def add_bytes(name: str, blob: bytes, mode: int = 0o644) -> None:
                    ti = tarfile.TarInfo(name)
                    ti.size = len(blob)
                    ti.mtime = 0      # the whole point
                    ti.mode = mode
                    ti.uid = ti.gid = 0
                    ti.uname = ti.gname = "root"
                    tf.addfile(ti, io.BytesIO(blob))

                for e in manifest.sorted_entries():
                    with open(os.path.join(manifest.root, e.path), "rb") as f:
                        add_bytes(f"{arc_prefix}/{e.path}", f.read(), 0o644)
                    n += 1
                for name, blob in sorted((extra or {}).items()):
                    add_bytes(name, blob)
        finally:
            gz.close()

    h = hashlib.sha256()
    with open(dest, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return dest, h.hexdigest(), os.path.getsize(dest)


def attestation(manifest: Manifest, backend: str, target: str, **extra) -> bytes:
    """The record the runner reads, so a run can prove what it was given.

    This closes the loop the DECISIONS.md #18 incident left open. The lost run
    produced `payload sha256: n/a` -- the notebook had nothing to compare
    against, so an empty payload and a correct one looked the same. With this
    file inside the archive, the runner can read the fingerprint it was handed
    and compare it to the tree of files actually present, and gpush can refuse
    to believe a run that does not match.
    """
    rec = {
        "schema": "gpush/attestation@1",
        "backend": backend,
        "target": target,
        "fingerprint": manifest.fingerprint,
        "file_count": len(manifest),
        "files": {e.path: e.sha256 for e in manifest.sorted_entries()},
    }
    rec.update(extra)
    return json.dumps(rec, indent=2, sort_keys=True).encode()


def seal(manifest: Manifest, dest: str, *, extra: dict[str, bytes] | None = None,
         arc_prefix: str = "payload") -> tuple[str, str, int]:
    """Pack, then write a sidecar sums file NEXT TO the archive.

    The sidecar is a sibling, never a member. A file inside the archive cannot
    name the archive's own hash -- the bytes change the moment the name is
    written -- so putting SHA256SUMS inside would print a hash that does not
    verify, which is worse than printing none. Kaggle ships it as a second
    dataset file, exactly as kernels/pack.py does, and that is where the
    transport can verify it without a second download.

    Returns (tarball path, sha256, bytes).
    """
    path, digest, nbytes = build(manifest, dest, extra=extra, arc_prefix=arc_prefix)
    with open(sums_path(dest), "w") as f:
        f.write(f"{digest}  {os.path.basename(dest)}\n")
    return path, digest, nbytes


def sums_path(dest: str) -> str:
    d, name = os.path.split(dest)
    return os.path.join(d, os.path.splitext(name)[0] + ".sha256")


def verify_archive(path: str, manifest: Manifest, arc_prefix: str = "payload") -> dict:
    """Read an archive back and compare it to the manifest.

    Returns {"ok", "missing", "extra", "differing", "found", "checked"}. The
    comparison is by *content hash*, not by name, so a payload that carries a
    file nobody asked for is caught -- that is the stale-file half of the gate,
    and it is the half a size or timestamp comparison would miss.
    """
    import tarfile as _tf

    found: dict[str, str] = {}
    try:
        with _tf.open(path, "r:gz") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                rel = m.name
                if rel.startswith(arc_prefix + "/"):
                    rel = rel[len(arc_prefix) + 1:]
                elif rel.startswith("GPUSH-") or rel == "SHA256SUMS":
                    continue
                found[rel] = hashlib.sha256(tf.extractfile(m).read()).hexdigest()
    except (OSError, _tf.TarError) as e:
        return {"ok": False, "missing": [], "extra": [], "differing": [],
                "found": {}, "checked": 0, "error": f"unreadable archive: {e}"}

    expected = {e.path: e.sha256 for e in manifest}
    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected))
    differing = sorted(p for p in set(expected) & set(found) if expected[p] != found[p])
    return {"ok": not (missing or extra or differing), "missing": missing,
            "extra": extra, "differing": differing, "found": found,
            "checked": len(expected)}


def read_attestation(path: str, arc_prefix: str = "payload") -> dict | None:
    """Pull GPUSH-ATTEST.json back out of a packed archive, if it is there."""
    try:
        with tarfile.open(path, "r:gz") as tf:
            for name in ("GPUSH-ATTEST.json",):
                try:
                    return json.loads(tf.extractfile(name).read())
                except (KeyError, ValueError):
                    continue
    except (OSError, tarfile.TarError):
        pass
    return None
