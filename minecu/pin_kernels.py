"""Pin Infatoshi/mine.cu ``kernels.cu`` (external dep) to the study hash."""
from __future__ import annotations

from .upstream import ensure, main as upstream_main


def main() -> None:
    path = ensure()
    print(f"PIN OK: {path} == Infatoshi/mine.cu (study pin)")


if __name__ == "__main__":
    upstream_main()
