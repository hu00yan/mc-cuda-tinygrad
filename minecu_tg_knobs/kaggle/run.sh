#!/usr/bin/env bash
# run.sh -- arm B on a real T4, one push, one run, both arms in one session.
#
#   ./minecu_tg_knobs/kaggle/run.sh [--dry-run] [--session MIN]
#
# WHAT THIS COSTS. `run` SPENDS GPU QUOTA on Kaggle's free tier. `plan` costs
# nothing. The tool prints which one it is before it does it, and this script
# prints a summary of that before it starts.
#
# WHY gpush AND NOT remote/kaggle_run.sh
# ---------------------------------------
# `remote/kaggle_run.sh` is the older recipe and it works, but gpush ships the
# payload as a VERSIONED DATASET with a SHA256 attestation the notebook verifies
# before it runs anything, so "which bytes produced this number" is answerable
# from the run record. This arm's whole argument is attribution -- a knob's
# effect has to be attributable to that knob -- and a measurement whose inputs
# cannot be named does not support an attribution.
#
# THE SHAPE. batch 1024, res 32, max_steps 32, world 32^3 -- the same shape as
# every other number in this study, including notes/19 §5.2's reference column.
# Render against render, at equal batch, both sides in the same run.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
GP="$REPO/gpush/gpush"
KAGGLE_CLI="$REPO/.venv/bin/kaggle"

DATASET="${GPUSH_KNOB_DATASET:-tenghuzhang/mc-minecu-knobs-payload}"
KERNEL="${GPUSH_KNOB_KERNEL:-tenghuzhang/mc-minecu-knobs-t4}"
SESSION=25
QUEUE_WAIT=30
POLL=20
DRY=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY="--dry-run"; shift ;;
    --session) SESSION="$2"; shift 2 ;;
    --queue-wait) QUEUE_WAIT="$2"; shift 2 ;;
    --poll) POLL="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 64 ;;
  esac
done

# Only this lane's files go over the wire. `minecu_tg/`, `minecu_tg_opt/`,
# `minecu_ref/src/` and the other directories belong to other work in this repo
# and are large; shipping them would be slow and would put another agent's
# in-flight edits into a measurement this arm is responsible for.
# gpush's `--include` is a PREFIX match (`gpush/manifest.py:227`:
# `p == i or p.startswith(i)`), so `--include minecu_tg` also ships
# `minecu_tg_faithful/`, `minecu_tg_opt/` and `minecu_tg_knobs/` -- other agents'
# in-flight work, in a payload whose whole purpose is that the bytes can be
# named. Three of those runs shipped a `minecu_tg_faithful/` tree that this arm
# never read. The trailing slash makes each entry an exact directory.
INCLUDE=(
  --include minecu_tg_knobs/     # this lane, in full
  --include minecu_tg/           # read-only from this lane: blocks.py, world.py,
                                 # render.py (arm A, the control), cpu_reference.py
  --include minecu_ref/bench/    # dda_model.py: the third transcription
  --include tools/archlab/       # inspect_ptx.py: the shared CATEGORIES
  --include derivation/          # minecu_gap.py, cited for the decomposition
  --exclude .agent-scratch
  --exclude out
  --exclude .venv
)

# `--full-resync` is not optional here, and the reason is worth recording
# because `plan` will otherwise cheerfully report "no changes".
#
# gpush's manifest is content-addressed and its baseline is a per-file
# (size, mtime, sha256) table. Deleting files from the working tree is
# indistinguishable, to that table, from files that were never there: the run-7
# payload contained `armb.py`, `census.py`, `knobs.py` and five other files
# this lane had **moved out of the directory on 2026-09-29**, and
# `gpush kaggle plan` reported `change +0 ~0 -0 =48, no changes`. The stale
# files were shipped because the previous run had put them in the remote
# manifest and a deletion is not a content change.
#
# A payload that contains a superseded module of the same name is worse than one
# that fails: on the card, `battery.py` was picked up from that stale tree and
# failed for a reason the fixed file no longer has. `--full-resync` resets the
# remote workdir and re-seeds from the tree as it is now, which is the only
# thing that makes a deletion mean a deletion.
RESYNC="--full-resync"

echo "== 1/3  plan (offline: no network, no credentials, costs nothing)"
"$GP" kaggle plan --dataset "$DATASET" --kernel "$KERNEL" \
  --kaggle-cli "$KAGGLE_CLI" "${INCLUDE[@]}"

echo ""
echo "== 2/3  run  (THIS SPENDS GPU QUOTA)"
echo "   dataset $DATASET"
echo "   kernel  $KERNEL"
echo "   session cap ${SESSION}m, queue wait ${QUEUE_WAIT}m"

# The command is `python3`, NOT this host's `$PY`. The remote is a Kaggle image
# with its own interpreter at a different absolute path, and gpush runs the
# command string verbatim on the far side: an absolute host path resolves
# nowhere and the notebook cell dies with
#
#   FileNotFoundError: [Errno 2] ... '/Users/.../.venv/bin/python'
#
# which is what the first run of this script did, after the payload had already
# been pushed, verified byte-for-byte on the card, and the device reported.
# A host path is the one thing about a remote command that cannot be right.
"$GP" kaggle run \
  --dataset "$DATASET" --kernel "$KERNEL" --kaggle-cli "$KAGGLE_CLI" \
  --session "$SESSION" --queue-wait "$QUEUE_WAIT" --poll "$POLL" \
  "${INCLUDE[@]}" $RESYNC $DRY \
  -- python3 -m minecu_tg_knobs.battery \
       --out /kaggle/working/out/minecu_tg_knobs \
       --ports naive,lean,lean-no-flat,lean-no-unsigned
