#!/usr/bin/env bash
# sync.sh -- the push loop, one command, and the one to run before every push.
#
#   ./remote/sync.sh --check     offline: manifest, diff, report.
#                              ZERO network calls, ZERO credentials. Works with
#                              no token on the machine. This is the cheap mode.
#   ./remote/sync.sh --sync      publish a new dataset version ONLY if the diff
#                              is non-empty. No kernel push, no GPU time, no cost.
#   ./remote/sync.sh --run       sync, then push the kernel, poll to a bounded
#                              timeout, pull results back, print an index.
#                              This is the single entry point for the loop.
#   ./remote/sync.sh --status    what is published right now; one API call.
#
# WHY A WRAPPER AND NOT JUST A SCRIPT. Everything that needs a working
# interpreter lives in sync.py; this file exists so the one command you type
# picks the repo's venv rather than whatever `python3` happens to be on PATH,
# and so the modes are discoverable from the shell with -h. It re-implements
# nothing: the whole driver is remote/sync.py, and the kernel side is delegated
# to remote/kaggle_run.sh with --skip-pack.
#
# COST. --check costs nothing at all: it reads files and prints. --sync costs
# one dataset version, which is a few hundred KB of upload and no accelerator
# time. --run costs one Kaggle GPU session, which the free tier meters
# (~30 GPU-hours a week) and which Kaggle caps server-side via `kernels push -t`.
#
# WHY THE INCREMENTALITY IS NOT A DELTA UPLOAD. A Kaggle dataset version is a
# full snapshot; the API has no partial-upload primitive and no delta endpoint.
# So what is incremental here is the DECISION (an unchanged tree creates no
# version at all, which is the common case and costs zero bytes), the REPORT
# (exactly which files changed, added, deleted) and the PROVENANCE (a manifest
# in the repo, diffable, that a result is attributable to). rclone would not
# help: it has no Kaggle backend, and Kaggle exposes no SFTP, WebDAV or S3
# endpoint, so it could only wrap the same API this already uses.
#
# NO LOCAL CONTAINERS. Nothing here needs a GPU, nvcc, ptxas or SASS
# disassembly. Anything that does is planned to run server-side on Kaggle, and
# its artefacts come back through `kaggle kernels output`.
#
# CREDENTIALS. `kaggle auth login`, or a token at ~/.kaggle/access_token. This
# script never looks for kaggle.json: that is the legacy path, and the current
# CLI accepts it silently and then fails with nothing useful.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PY="${SYNC_PYTHON:-$REPO/.venv/bin/python}"

if [ ! -x "$PY" ]; then
  # A system python3 is usable for --check: sync.py imports nothing outside the
  # standard library, which is why the offline mode can run on a machine that
  # has never seen kaggle. It does have to be >= 3.10 -- kda/kda_vs_fla.py:135
  # is a walrus in an unparenthesised subscript, and a 3.9 ast.parse() calls that
  # a syntax error and blames the file. sync.py checks this itself; this is the
  # early, friendly version of the same message.
  if command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
  else
    echo "no python3 found, and $REPO/.venv/bin/python does not exist" >&2
    echo "  uv venv .venv  &&  uv pip install --python .venv/bin/python kaggle" >&2
    exit 4
  fi
fi

if ! "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
  echo "the python at $PY is older than 3.10 and the payload needs 3.10+ syntax." >&2
  echo "  kda/kda_vs_fla.py:135 uses a walrus in an unparenthesised subscript;" >&2
  echo "  macOS's /usr/bin/python3 is 3.9.  Set SYNC_PYTHON=<path>." >&2
  exit 4
fi

exec "$PY" "$HERE/sync.py" "$@"
