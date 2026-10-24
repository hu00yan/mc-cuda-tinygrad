"""gates.py -- the checks that must pass before any GPU time is spent.

Two incidents in this repo's history motivate this whole file. Both cost a GPU
session, and neither raised an error: they returned success.

  * DECISIONS.md #18 -- a dataset version was published four seconds before
    the kernel was pushed, had not finished being created, did not mount, so
    the kernel ran against an EMPTY /kaggle/input, and all 15 battery items
    recorded `skip`. The run was `COMPLETE`. A session of quota was spent
    producing the string "skip" fifteen times.
  * the stale-file incident -- a payload carried a file that was no longer the
    file the working tree held, so a measurement was attributed to code that
    was not the code that produced the number.

A third, cheaper-to-hit failure has the same shape: deleting 90% of the tree
(a stray `rm -r`, or a branch switch) and pushing the result as a normal sync.

EVERY GATE HERE RUNS BEFORE QUOTA IS SPENT, and every one of them can only
ever *refuse* to proceed. None of them auto-repairs a remote, because a tool
that silently fixes a payload is a tool that can silently ship the wrong bytes.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Gate:
    name: str
    ok: bool
    detail: str = ""
    #: The action that stops everything when this gate fails. Phrased as an
    #: instruction because that is what the user has to do.
    remedy: str = ""

    def line(self) -> str:
        mark = "pass" if self.ok else "REFUSE"
        return f"  [{mark:<6}] {self.name:<34} {self.detail}"


@dataclass
class GateResult:
    gates: list[Gate] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "", remedy: str = "") -> Gate:
        g = Gate(name, ok, detail, remedy)
        self.gates.append(g)
        return g

    @property
    def ok(self) -> bool:
        return all(g.ok for g in self.gates)

    @property
    def failed(self) -> list[Gate]:
        return [g for g in self.gates if not g.ok]

    def report(self, title: str) -> str:
        out = [title]
        out += [g.line() for g in self.gates]
        if not self.ok:
            out.append("")
            for g in self.failed:
                out.append(f"  REFUSED by '{g.name}': {g.detail}")
                if g.remedy:
                    out.append(f"    fix: {g.remedy}")
        return "\n".join(out)


# ---------------------------------------------------------------- the guards


def mass_deletion(plan, threshold: float, allowed: bool) -> Gate:
    """Refuse a delete-sync that would prune a suspiciously large share.

    Delete-sync is a feature (a stale file on the remote is a measurement
    attributed to the wrong code) and a hazard (one `rm -r` away from deleting
    the remote tree). The threshold is a fraction of the *previous* manifest,
    not of the new one, so shrinking the tree to nothing reads as 100%.

    `--full-resync` and `--allow-mass-delete` both set `allowed`, and the gate
    then says so in its detail rather than quietly passing -- an override that
    prints nothing is indistinguishable from a guard that did not fire.
    """
    frac = plan.delete_fraction
    names = ", ".join(plan.deleted[:4]) + (" ..." if len(plan.deleted) > 4 else "")
    detail = (f"would prune {len(plan.deleted)}/{plan.change_count + len(plan.unchanged)} "
              f"= {frac * 100:.1f}% of the remote manifest (limit {threshold * 100:.0f}%)")
    if plan.deleted:
        detail += f"; first: {names}"
    if allowed:
        return Gate("mass-deletion guard", True,
                    detail + "  [OVERRIDDEN on the command line]", "")
    ok = frac <= threshold
    return Gate(
        "mass-deletion guard", ok, detail,
        "if this is intended, re-run with --allow-mass-delete "
        "(or --full-resync to reset the remote workdir and re-seed)")


def payload_completeness(expected: dict[str, str], actual: dict[str, str],
                         context: str) -> tuple[Gate, list[str], list[str]]:
    """The stale-file and empty-payload gate, in one comparison.

    expected  path -> sha256, from the manifest that was packed
    actual    path -> sha256, as read back from where the payload actually is

    Three ways to fail, all of them incidents that have already happened here:
      * missing   -- the payload is empty or partial. This is the DECISIONS.md
        #18 shape, and it must be caught before the run, not after it.
      * extra     -- a file is on the remote that is not in the manifest. This
        is the stale-file shape. It is not cosmetic: the remote will happily
        execute the stale copy, and the number it produces will be attributed
        to the current tree.
      * differing -- the same path with different bytes. Rare, and it means the
        transfer is not what we think it is.

    Returns (gate, missing, extra) so the caller can print the offenders.
    """
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    differing = sorted(p for p in set(expected) & set(actual) if expected[p] != actual[p])
    ok = not (missing or extra or differing)
    bits = [f"{len(expected)} expected"]
    if missing:
        bits.append(f"MISSING {len(missing)}: " + ", ".join(missing[:4])
                    + (" ..." if len(missing) > 4 else ""))
    if extra:
        bits.append(f"EXTRA {len(extra)}: " + ", ".join(extra[:4])
                    + (" ..." if len(extra) > 4 else ""))
    if differing:
        bits.append(f"DIFFERING {len(differing)}: " + ", ".join(differing[:4])
                    + (" ..." if len(differing) > 4 else ""))
    if ok:
        detail = f"{context}: all {len(expected)} files present with matching hashes"
    else:
        detail = "; ".join(bits[1:]) or "mismatch"
    remedy = ""
    if not ok:
        remedy = ("this is the exact condition that lost a GPU session in "
                  "DECISIONS.md #18 (empty payload) and the one that attributes a "
                  "number to stale code (extra file). Re-run with --full-resync to "
                  "reset the remote workdir and re-seed from scratch")
    return Gate(f"payload integrity ({context})", ok,
                f"{context}: " + detail if not ok else detail, remedy), missing, extra


def device_is_gpu(device: dict, require_gpu: bool) -> Gate:
    """Refuse to call a CPU box a GPU run.

    Colab free sessions are frequently CPU-only, and Kaggle can queue a TPU
    kernel on a machine with no NVIDIA device at all. Either way the honest
    response is to report what the machine says. The tool surfaces the compute
    capability and does not interpret it -- sm_75 is stated, not judged, because
    what sm_75 means for a given kernel is the study's question, not the
    transport's.

    This gate therefore checks only presence, never capability.
    """
    name = device.get("name") or ""
    has = bool(device.get("gpu")) and bool(name)
    if has:
        return Gate("device is a GPU", True,
                    f"{name}  {device.get('memory_mb', '?')} MB  "
                    f"compute capability {device.get('compute_cap', 'unknown')}")
    if not require_gpu:
        return Gate("device is a GPU", True,
                    f"NO GPU ({device.get('reason', 'none reported')}); "
                    "continuing because --allow-cpu was given", "")
    return Gate(
        "device is a GPU", False,
        f"this machine reports no GPU ({device.get('reason', 'none reported')})",
        "a run here would not be a GPU measurement. Re-run with --allow-cpu if "
        "you want a CPU result recorded as a CPU result")


def no_changes(skip: bool, plan_summary: str, reason: str) -> str:
    """The one-line report for the no-op decision. Returns text, not a Gate.

    Deliberately not a Gate. A Gate carries a verdict, and this decision is not
    a verdict: skipping is a success, and threading it through GateResult.ok
    would mean either "skipping fails the check" (which would refuse the push)
    or "skipping passes the check" (which would hide the one time it did not).
    Control flow belongs to the caller, which already has `skip`; this only has
    to say so out loud.

    Silence at this point reads as "did nothing" or as "hung", and the
    difference matters when the reason is a stale pending baseline.
    """
    if skip:
        return reason
    return f"{plan_summary}: {reason}"


def coverage(plan) -> Gate:
    """Files in the working tree that no include rule reaches.

    Not fatal by itself -- most trees ship a subset on purpose -- but never
    silent. A file the payload cannot see is a measurement that cannot be
    attributed to the code the author had open.
    """
    if not plan.uncovered:
        return Gate("working-tree coverage", True, "every working-tree file is in the payload")
    return Gate("working-tree coverage", True,
                f"{len(plan.uncovered)} working-tree files are NOT in the payload "
                f"(e.g. {', '.join(plan.uncovered[:3])}); they cannot be measured remotely",
                "")
