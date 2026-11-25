"""Pin ``minecu_ref/src/kernels.cu`` to Infatoshi/mine.cu (byte identity)."""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCAL = HERE.parent / "minecu_ref" / "src" / "kernels.cu"
# commit used by this study's notes + current main tip (same bytes as of pin)
URLS = (
    "https://raw.githubusercontent.com/Infatoshi/mine.cu/main/src/kernels.cu",
    "https://raw.githubusercontent.com/Infatoshi/mine.cu/ec2085e3a21a8cccfc03fa330892e34c97c8e1b2/src/kernels.cu",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    local = LOCAL.read_bytes()
    local_h = sha256(local)
    print(f"local {LOCAL} {len(local)} B sha256={local_h}")
    ok = True
    vendored = HERE.parent / "_upstream" / "mine.cu" / "kernels.cu"
    if vendored.is_file():
        vh = sha256(vendored.read_bytes())
        match = vh == local_h
        print(f"{'PASS' if match else 'FAIL'} vendored {vendored}\n  sha256={vh}")
        ok = ok and match
    for url in URLS:
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                remote = r.read()
        except Exception as e:
            print(f"SKIP fetch {url}: {e}")
            continue
        h = sha256(remote)
        match = h == local_h
        print(f"{'PASS' if match else 'FAIL'} {url}\n  sha256={h}")
        ok = ok and match
    if not ok:
        raise SystemExit("FAIL: kernels.cu diverges from Infatoshi")
    print("PIN OK: kernels.cu == Infatoshi/mine.cu")


if __name__ == "__main__":
    main()
