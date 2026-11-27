"""Infatoshi/mine.cu as an external dependency — not part of this package.

Resolve ``kernels.cu`` from, in order:

1. ``MINECU_KERNELS`` — explicit path to ``kernels.cu``
2. ``MINECU_MINE_CU`` — checkout root of https://github.com/Infatoshi/mine.cu
3. ``.deps/mine.cu/`` under the repo (fetched / cloned, gitignored)

Pin is the commit + sha256 this study measured against. ``ensure()`` fetches
that exact blob into ``.deps`` when nothing is configured locally.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEPS_ROOT = REPO / ".deps" / "mine.cu"

# Commit + content pin used by notes/T4 measurements (byte-identical).
PIN_COMMIT = "ec2085e3a21a8cccfc03fa330892e34c97c8e1b2"
PIN_SHA256 = "c6ed8f06ee64d40a8a8ab7ebce3b3ceea1e3ac93f9ee50cfc1e63d3f090d878f"
UPSTREAM_REPO = "https://github.com/Infatoshi/mine.cu"
RAW_URLS = (
    f"https://raw.githubusercontent.com/Infatoshi/mine.cu/{PIN_COMMIT}/src/kernels.cu",
    "https://raw.githubusercontent.com/Infatoshi/mine.cu/main/src/kernels.cu",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _candidate_paths() -> list[Path]:
    out: list[Path] = []
    env_file = os.environ.get("MINECU_KERNELS")
    if env_file:
        out.append(Path(env_file).expanduser())
    env_root = os.environ.get("MINECU_MINE_CU")
    if env_root:
        out.append(Path(env_root).expanduser() / "src" / "kernels.cu")
    out.append(DEPS_ROOT / "src" / "kernels.cu")
    return out


def find_kernels(require: bool = False) -> Path | None:
    """Return an existing kernels.cu path, or None / raise."""
    for p in _candidate_paths():
        if p.is_file():
            return p.resolve()
    if require:
        raise FileNotFoundError(
            "Infatoshi/mine.cu kernels.cu not found. Set MINECU_MINE_CU / "
            "MINECU_KERNELS, or run: PYTHONPATH=. python -m minecu.upstream"
        )
    return None


def verify(path: Path | None = None) -> tuple[Path, str]:
    """Check kernels.cu matches the study pin. Returns (path, sha256)."""
    p = path or find_kernels(require=True)
    assert p is not None
    digest = sha256(p.read_bytes())
    if digest != PIN_SHA256:
        raise SystemExit(
            f"FAIL: {p} sha256={digest}\n"
            f"  expected pin {PIN_SHA256} (Infatoshi/mine.cu@{PIN_COMMIT[:7]})\n"
            f"  refetch: PYTHONPATH=. python -m minecu.upstream --force"
        )
    return p, digest


def fetch(*, force: bool = False) -> Path:
    """Download the pinned kernels.cu into ``.deps/mine.cu/src/``."""
    dest = DEPS_ROOT / "src" / "kernels.cu"
    if dest.is_file() and not force:
        digest = sha256(dest.read_bytes())
        if digest == PIN_SHA256:
            return dest.resolve()
        # stale cache — refresh
        force = True

    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    data: bytes | None = None
    for url in RAW_URLS:
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            break
        except Exception as e:  # pragma: no cover
            last_err = e
            continue
    if data is None:
        raise RuntimeError(f"failed to fetch kernels.cu: {last_err}")

    digest = sha256(data)
    if digest != PIN_SHA256:
        # main tip may have moved; refuse to install an unpinned blob
        if url.endswith("/main/src/kernels.cu"):  # type: ignore[name-defined]
            raise RuntimeError(
                f"upstream main kernels.cu sha256={digest}, pin is {PIN_SHA256}; "
                f"refusing to install. Use commit URL or update PIN_* in upstream.py"
            )
        raise RuntimeError(f"fetched sha256={digest}, expected {PIN_SHA256}")

    dest.write_bytes(data)
    # tiny provenance marker
    (DEPS_ROOT / "PIN").write_text(
        f"repo={UPSTREAM_REPO}\ncommit={PIN_COMMIT}\nsha256={PIN_SHA256}\n"
        f"path=src/kernels.cu\n",
        encoding="utf-8",
    )
    return dest.resolve()


def ensure() -> Path:
    """Return a verified kernels.cu, fetching into ``.deps`` if needed."""
    existing = find_kernels(require=False)
    if existing is not None:
        verify(existing)
        return existing
    path = fetch(force=False)
    verify(path)
    return path


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Fetch / pin Infatoshi/mine.cu kernels.cu")
    ap.add_argument("--force", action="store_true", help="re-download into .deps")
    ap.add_argument("--print-path", action="store_true", help="print kernels.cu path only")
    args = ap.parse_args(argv)

    if args.force or find_kernels() is None:
        path = fetch(force=args.force)
    else:
        path = find_kernels(require=True)
        assert path is not None
    path, digest = verify(path)
    if args.print_path:
        print(path)
        return
    print(f"upstream kernels.cu → {path}")
    print(f"  sha256={digest}")
    print(f"  pin    ={PIN_SHA256}")
    print(f"  commit ={PIN_COMMIT} ({UPSTREAM_REPO})")
    print("PIN OK")


if __name__ == "__main__":
    main()
