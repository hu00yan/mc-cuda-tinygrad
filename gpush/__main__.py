"""Entry point: `python -m gpush` and the `gpush` script.

There is no setup.py and nothing to install. gpush is a directory of standard
library modules; it runs from the repo root with no dependencies, no venv and
no container. That is a deliberate constraint: this machine's disk was filled
by container images once already.
"""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
