"""manifest.py -- build the transfer manifest, fingerprint it, and diff it.

This is the part of gpush that decides *what* to transfer. It is backend-free
on purpose: the same manifest and the same fingerprint are used for a Kaggle
dataset version and for an ssh tarball, so "unchanged" means the same thing on
both transports and a plan produced offline is the plan that will execute.

WHAT IS IN THE MANIFEST
    Tracked files, plus untracked files that git does not ignore. That is
    `git ls-files` union `git ls-files --others --exclude-standard`: exactly the
    set a human means by "the working tree". Build products are gitignored in
    this repo and stay out (`.venv/`, `__pycache__/`, the 23 MB of regenerable
    probe reference data), which is also why the payload stays small.

    Outside a git repository, fall back to walking the tree with the same
    default denylist, and say so in the report -- a manifest built by a fallback
    walk is not the same contract as one built from git, and pretending
    otherwise would make "no changes" a claim rather than a fact.

THE FINGERPRINT
    sha256 over the canonical, sorted, content-addressed serialisation of the
    whole manifest, salted with the backend name, the target id and the schema
    version. It therefore changes if and only if the file *set* changes or any
    byte of any file changes. It is not a git hash: untracked files count.

    Content hashing is incremental, the way rsync's is. A file whose
    (size, mtime_ns) match the stored baseline reuses the baseline's sha256; a
    file that differs -- new, resized, or touched -- is read and hashed, which
    is "the content of every changed file" in the literal sense. `--checksum`
    drops the shortcut and re-reads everything, which is what catches an edit
    that preserved both size and mtime (a `git checkout` of an old revision,
    a `touch -r`, a copied file).

WHY NOT DIFF AGAINST GIT
    Because the interesting case is not "what did I commit". It is "what is on
    disk right now versus what the remote already has". A dirty working tree is
    the normal state of this kind of work, and it is exactly where a
    git-based diff would lie.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass, field

from . import MANIFEST_SCHEMA

CHUNK = 1 << 20


# --------------------------------------------------------------- git layer

def _git(root: str, *args: str) -> str | None:
    """Run git, returning stdout, or None if this is not usable as a git root."""
    try:
        p = subprocess.run(
            ["git", "-C", root, *args],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0:
        return None
    return p.stdout


def git_worktree(root: str) -> tuple[list[str], str] | None:
    """(paths, source) for the working tree, or None if not a git checkout.

    Tracked files and non-ignored untracked files, NUL-separated so that a
    path containing a space or a newline -- both legal, and both present in
    this repo's history of shell-quoting bugs -- survives intact.
    """
    inside = _git(root, "rev-parse", "--is-inside-work-tree")
    if inside is None or inside.strip() != "true":
        return None
    tracked = _git(root, "ls-files", "-z", "--cached")
    others = _git(root, "ls-files", "-z", "--others", "--exclude-standard")
    if tracked is None or others is None:
        return None
    paths = {p for p in tracked.split("\0") if p}
    paths |= {p for p in others.split("\0") if p}
    return sorted(paths), "git"


# The denylist for the no-git fallback. Deliberately short: a fallback that
# guesses at ignore rules produces a manifest nobody can reason about, so it
# ignores only the things that are unambiguously build output everywhere.
FALLBACK_DENY = (".git/", "__pycache__/", ".venv/", ".DS_Store", ".mypy_cache/",
                 ".pytest_cache/", ".ruff_cache/", "node_modules/")
FALLBACK_MAX_BYTES = 512 << 20


def walk_worktree(root: str) -> tuple[list[str], str]:
    out: list[str] = []
    stack = [""]
    while stack:
        rel = stack.pop()
        with os.scandir(os.path.join(root, rel) if rel else root) as it:
            for e in it:
                r = f"{rel}/{e.name}" if rel else e.name
                if e.name.startswith(".") or any(r.startswith(d) or f"/{d}" in r
                                                 for d in FALLBACK_DENY):
                    continue
                if e.is_dir(follow_symlinks=False):
                    stack.append(r)
                elif e.is_file(follow_symlinks=False):
                    try:
                        if e.stat(follow_symlinks=False).st_size <= FALLBACK_MAX_BYTES:
                            out.append(r)
                    except OSError:
                        continue
    return sorted(out), "walk"


# ---------------------------------------------------------------- the model

@dataclass(frozen=True)
class Entry:
    path: str
    size: int
    mtime_ns: int
    sha256: str

    def as_dict(self) -> dict:
        return {"path": self.path, "size": self.size, "mtime_ns": self.mtime_ns,
                "sha256": self.sha256}


@dataclass
class Manifest:
    root: str
    entries: dict[str, Entry] = field(default_factory=dict)
    source: str = "git"
    fingerprint: str = ""
    reused_hashes: int = 0
    hashed: int = 0
    total_bytes: int = 0
    candidates: int = 0

    # -- construction
    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries.values())

    def sorted_entries(self) -> list[Entry]:
        return [self.entries[p] for p in sorted(self.entries)]

    def canonical(self, backend: str, target: str) -> bytes:
        """The exact bytes the fingerprint is taken over.

        Sorted by path, one line per file, hash first so a human reading a
        diff of two fingerprints sees the file that moved. Backend and target
        are inside the hash so a payload pushed to one backend can never be
        mistaken for a payload pushed to another.
        """
        h = hashlib.sha256()
        h.update(MANIFEST_SCHEMA.encode() + b"\n")
        h.update(backend.encode() + b"\n")
        h.update(target.encode() + b"\n")
        for e in self.sorted_entries():
            h.update(f"{e.sha256}  {e.size:>12}  {e.path}\n".encode())
        return h.digest()

    def compute_fingerprint(self, backend: str, target: str) -> str:
        self.fingerprint = self.canonical(backend, target).hex()
        return self.fingerprint

    def as_dict(self) -> dict:
        return {
            "schema": MANIFEST_SCHEMA,
            "source": self.source,
            "fingerprint": self.fingerprint,
            "files": len(self.entries),
            "bytes": self.total_bytes,
            "entries": [e.as_dict() for e in self.sorted_entries()],
        }

    @staticmethod
    def from_dict(d: dict) -> "Manifest":
        m = Manifest(root="", source=d.get("source", "git"),
                     fingerprint=d.get("fingerprint", ""))
        for e in d.get("entries", []):
            m.entries[e["path"]] = Entry(e["path"], e["size"], e["mtime_ns"],
                                         e["sha256"])
            m.total_bytes += e["size"]
        return m


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def build(root: str, baseline: Manifest | None = None, *, checksum: bool = False,
          include: list[str] | None = None,
          exclude: list[str] | None = None) -> Manifest:
    """Build the manifest for `root`.

    baseline  a previous Manifest, used to skip re-hashing unchanged files.
    checksum   ignore (size, mtime_ns) and re-read every file.
    include    if given, only these path prefixes are considered at all.
    exclude    path prefixes to drop.

    An include list is not an optimisation, it is a contract: it says what the
    remote is allowed to see. A file that is in the working tree but outside
    the include list is *not* in the payload, and the diff's `uncovered` field
    exists so that this cannot pass unnoticed.
    """
    root = os.path.abspath(root)
    found = git_worktree(root)
    if found is None:
        paths, source = walk_worktree(root)
    else:
        paths, source = found
    m = Manifest(root=root, source=source)
    old = baseline.entries if baseline else {}
    m.candidates = len(paths)

    def allowed(p: str) -> bool:
        if include and not any(p == i or p.startswith(i) for i in include):
            return False
        if exclude and any(p == e or p.startswith(e) for e in exclude):
            return False
        return True

    for rel in paths:
        if not allowed(rel):
            continue
        ap = os.path.join(root, rel)
        try:
            st = os.stat(ap)
        except OSError:
            continue
        if not os.path.isfile(ap):
            continue
        prior = old.get(rel)
        if (prior is not None and not checksum
                and prior.size == st.st_size and prior.mtime_ns == st.st_mtime_ns):
            m.entries[rel] = Entry(rel, st.st_size, st.st_mtime_ns, prior.sha256)
            m.reused_hashes += 1
            m.total_bytes += st.st_size
            continue
        try:
            digest = sha256_file(ap)
        except OSError:
            continue
        m.entries[rel] = Entry(rel, st.st_size, st.st_mtime_ns, digest)
        m.hashed += 1
        m.total_bytes += st.st_size
    return m


def uncovered(root: str, manifest: Manifest, include: list[str] | None) -> list[str]:
    """Working-tree files that no include rule covers.

    Reported, never silently dropped: a file the payload cannot see is a
    measurement that cannot be attributed to the code the author had open.
    """
    if not include:
        return []
    found = git_worktree(os.path.abspath(root))
    if found is None:
        return []
    paths, _ = found
    return sorted(p for p in paths if p not in manifest.entries)


# ------------------------------------------------------------------- diff

@dataclass
class Plan:
    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    uncovered: list[str] = field(default_factory=list)
    base_fingerprint: str = ""
    new_fingerprint: str = ""
    bytes_in: int = 0

    @property
    def empty(self) -> bool:
        return not (self.added or self.modified or self.deleted)

    @property
    def change_count(self) -> int:
        return len(self.added) + len(self.modified) + len(self.deleted)

    @property
    def delete_fraction(self) -> float:
        """Fraction of the *previous* manifest that would be pruned."""
        total = len(self.added) + len(self.modified) + len(self.deleted) + len(self.unchanged)
        return (len(self.deleted) / total) if total else 0.0

    def summary(self) -> str:
        return (f"+{len(self.added)} ~{len(self.modified)} -{len(self.deleted)} "
                f"={len(self.unchanged)}")


def diff(old: Manifest | None, new: Manifest, *, include: list[str] | None = None,
         root: str | None = None) -> Plan:
    """The transfer plan: old manifest -> new manifest."""
    p = Plan()
    o = old.entries if old else {}
    for path in sorted(new.entries):
        prev = o.get(path)
        if prev is None:
            p.added.append(path)
        elif prev.sha256 != new.entries[path].sha256:
            p.modified.append(path)
        else:
            p.unchanged.append(path)
    for path in sorted(o):
        if path not in new.entries:
            p.deleted.append(path)
    p.base_fingerprint = old.fingerprint if old else ""
    p.new_fingerprint = new.fingerprint
    p.bytes_in = sum(new.entries[x].size for x in p.added) + \
        sum(new.entries[x].size for x in p.modified)
    if include and root:
        p.uncovered = uncovered(root, new, include)
    return p
