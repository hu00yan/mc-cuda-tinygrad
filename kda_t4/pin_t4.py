#!/usr/bin/env python3
"""
pin_t4.py -- run `kda/baseline_pin.py` on the card, and record what it said.

`kda/baseline_pin.py` exists because a Triton path can silently be a vendor
CUTLASS kernel, or a naive fallback, and "we compared against fla" can be true
of three different objects. Its own docstring (`:14-15`) calls a comparison
against the wrong object "a measurement of the wrong object". This file does
not reimplement it. It runs it, as a subprocess, exactly as its docstring
prescribes, and keeps the three things a subprocess can tell you that an
in-process call cannot:

  * the exit code, under `--strict` (0 all gates VALID, 1 at least one INVALID,
    2 at least one UNKNOWN -- and UNKNOWN is not valid, it is undecided)
  * the `--json` document, verbatim, as a separate artefact
  * the full text report, verbatim

The reasons those have to be subprocess artifacts rather than a dict on the way
out:

  - `assert_pinned()` raises. A raised exception is indistinguishable from a
    crash unless you see the message, and the message is the report.
  - the report prints `>78-column` tables of file:line citations. A JSON blob
    does not carry them, and the citations are the part a reader has to be
    able to check.
  - `probe_deps()` in baseline_pin *imports* fla, and fla at import writes
    `TRITON_F32_DEFAULT` into `os.environ` (fla/utils/_device.py:149-152). An
    in-process pin would therefore perturb the very process that then does the
    measuring. Subprocess isolation keeps the pin from contaminating the
    measurement; that is the whole reason for running it this way and it is
    worth stating because the alternative looks equivalent.

It then re-derives, independently, the two facts the scope label rests on:
which backend the dispatch loop would select for the study's call shape, and
what the shared-memory number is. If those disagree with baseline_pin's own
reading, both are printed and the disagreement is a finding.

Run:  python kda_t4/pin_t4.py       (writes kda_t4/out/pin_t4.{txt,json})
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
REPO = os.path.dirname(HERE)
PIN = os.path.join(REPO, "kda", "baseline_pin.py")
os.makedirs(OUT, exist_ok=True)

LOG: list = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def hr(t=""):
    say("")
    say("=" * 76)
    if t:
        say(t)
        say("=" * 76)


def run_pin(flag):
    """Run baseline_pin.py with one flag. Returns (rc, stdout, stderr)."""
    cmd = [sys.executable, PIN, flag]
    say(f"  $ python kda/baseline_pin.py {flag}")
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=900)
    return p.returncode, p.stdout, p.stderr


def main():
    R = {}
    hr("kda_t4 PIN -- kda/baseline_pin.py, run on this card as a subprocess")
    say(f"  pin file     {PIN}")
    try:
        say(f"  pin sha256   {__import__('hashlib').sha256(open(PIN,'rb').read()).hexdigest()}")
        R["pin_sha256"] = __import__('hashlib').sha256(open(PIN, 'rb').read()).hexdigest()
    except BaseException as e:
        say(f"  pin sha256   <{type(e).__name__}>")

    # ---------------------------------------------------------------- 1. --json
    hr("1. baseline_pin.py --json")
    rc_json, out_json, err_json = run_pin("--json")
    say(f"  exit {rc_json}")
    with open(os.path.join(OUT, "pin_t4.baseline_pin.json"), "w") as f:
        f.write(out_json)
    doc = None
    try:
        doc = json.loads(out_json)
        R["json_ok"] = True
        R["json_type"] = type(doc).__name__
        say(f"  parsed as JSON, and the top-level value is a {type(doc).__name__}")
        if isinstance(doc, dict):
            say("  top-level keys:")
            for k in doc:
                v = doc[k]
                n = f" ({len(v)} entries)" if isinstance(v, (list, dict)) else ""
                say(f"    {k}{n}")
        else:
            # FINDING, not a crash.  baseline_pin.py:1440 does
            #     json.dumps(p, indent=2, default=str)
            # on a @dataclass instance.  dataclasses are not JSON-serialisable,
            # so `default=str` fires on the WHOLE object and the document
            # degenerates to one string containing repr(p).  It is valid JSON
            # and completely unparseable as data.  See notes/30 section 2.
            say("  *** FINDING: --json did NOT emit an object.  It emitted a single")
            say("      string.  baseline_pin.py:1440 calls json.dumps(p, default=str)")
            say("      on a @dataclass; dataclasses are not serialisable, so the")
            say("      `default` hook is applied to the entire Pin and the document")
            say("      becomes repr(p) as a lone string.  Valid JSON, zero fields.")
            say(f"      length of that string: {len(doc)} chars")
            say(f"      first 240 chars: {doc[:240]!r}")
            R["json_is_bare_string"] = True
    except BaseException as e:
        R["json_ok"] = False
        say(f"  NOT VALID JSON: {type(e).__name__}: {str(e)[:120]}")
        say("  --- raw stdout (first 3000 chars) ---")
        say(out_json[:3000])
    if err_json.strip():
        say("  --- stderr ---")
        say(err_json[-3000:])
    R["rc_json"] = rc_json

    # -------------------------------------------------------------- 2. --strict
    hr("2. baseline_pin.py --strict")
    rc_strict, out_strict, err_strict = run_pin("--strict")
    with open(os.path.join(OUT, "pin_t4.baseline_pin.strict.txt"), "w") as f:
        f.write(out_strict + ("\n--- stderr ---\n" + err_strict if err_strict.strip() else ""))
    say(f"  exit {rc_strict}   "
        f"({'all gates VALID' if rc_strict == 0 else 'at least one INVALID' if rc_strict == 1 else 'at least one UNKNOWN'})")
    R["rc_strict"] = rc_strict

    # -------------------------------------------------------- 3. the full report
    hr("3. baseline_pin.py  (full report, verbatim)")
    rc_rep, out_rep, _ = run_pin("")
    with open(os.path.join(OUT, "pin_t4.baseline_pin.report.txt"), "w") as f:
        f.write(out_rep)
    say(out_rep.rstrip())
    R["rc_report"] = rc_rep

    # ------------------------------------------- 4. independent re-derivation
    hr("4. independent re-derivation of the two facts the scope label rests on")
    try:
        sys.path.insert(0, os.path.join(REPO, "kda"))
        import baseline_pin as bp            # noqa: E402
        bp.probe_deps()
        say("  4a. which backend would the study's call shape select?")
        for label, call in (("study harness", bp.STUDY_CALL), ("fla layer shape", bp.LAYER_CALL)):
            sel, dec, err = bp.simulate_dispatch(call, idx=0)
            say(f"      {label:<16} -> {sel!r}   err={err!r}")
            for d in dec:
                say(f"         {d['backend']:<26} {d['outcome']:<9} {d['predicate']}")
                if d.get("reason"):
                    say(f"           {d['reason']}")
            R.setdefault("dispatch", {})[label] = dict(selected=sel, error=err, decisions=dec)

        say("")
        say("  4b. shared memory, read four independent ways")
        d = bp.device_report(0)
        say(f"      device                 {d.get('name')}")
        say(f"      cc                     {d.get('cc')}  {d.get('arch')}")
        say(f"      torch .shared_memory_per_block_optin   {d.get('smem_optin')}  ({d.get('smem_src')})")
        say(f"      triton max_shared_mem                  {d.get('smem_triton')}")
        say(f"      fla get_all_max_shared_mem             {d.get('smem_fla')}")
        say(f"      all agree                              {d.get('smem_agree')}")
        say(f"      verdict                                {d.get('smem_verdict')}")
        say(f"      why                                    {d.get('smem_why')}")
        R["device"] = d
        for n in d.get("note", []):
            say(f"      note: {n}")
    except BaseException as e:
        import traceback
        traceback.print_exc()
        R["rederive_error"] = f"{type(e).__name__}: {str(e)[:200]}"

    # -------------------------------------------------------- 5. the fleet
    hr("5. fleet, and the device-0 rule")
    try:
        fl = bp.fleet_report()
        say(f"  CUDA devices visible: {fl.get('count')}   heterogeneous: {fl.get('heterogeneous')}")
        for dev in fl.get("devices", []):
            say(f"    {dev['index']}: {dev.get('name'):<24} {dev.get('arch'):<8} "
                f"smem={dev.get('smem_optin')}  {dev.get('smem_verdict')}")
        for n in fl.get("notes", []):
            say(f"    note: {n}")
        R["fleet"] = fl
    except BaseException as e:
        say(f"  fleet read failed: {type(e).__name__}: {str(e)[:120]}")

    # ------------------------------- 6. a machine-readable pin, built here
    # NOT the --json output (see section 1): this is the same build_pin() call,
    # serialised field by field, so the gates survive as data.  Labelled as
    # derived so nobody mistakes it for the tool's own --json.
    hr("6. the same pin, as a real object (built here, not by --json)")
    try:
        import dataclasses
        p = bp.build_pin(idx=0, simulate=True)
        obj = dataclasses.asdict(p)
        with open(os.path.join(OUT, "pin_t4.build_pin.json"), "w") as f:
            json.dump(obj, f, indent=2, default=str)
        say(f"  VERDICT  {p.verdict()}")
        say(f"  problems ({len(p.problems)}):")
        for m in p.problems:
            say(f"    FAIL   {m}")
        say(f"  unknowns ({len(p.unknowns)}):")
        for m in p.unknowns:
            say(f"    UNKNOWN {m}")
        say("")
        say("  and the two selection questions, answered:")
        say(f"    study call  -> {p.study_call.get('sim_selected')}")
        say(f"    layer call  -> {p.layer_call.get('sim_selected')}")
        say(f"    registry    -> {len(p.registry_live or [])} live backends, "
            f"{len(p.drift)} drift")
        say(f"  wrote {OUT}/pin_t4.build_pin.json")
        R["build_pin"] = dict(verdict=p.verdict(), problems=p.problems,
                              unknowns=p.unknowns, drift=p.drift)
    except BaseException as e:
        import traceback
        traceback.print_exc()
        say(f"  build_pin serialisation failed: {type(e).__name__}: {str(e)[:150]}")

    with open(os.path.join(OUT, "pin_t4.json"), "w") as f:
        json.dump(R, f, indent=2, default=str)
    with open(os.path.join(OUT, "pin_t4.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")
    hr("")
    say(f"exit codes: --json {R.get('rc_json')}   --strict {R.get('rc_strict')}   "
        f"report {R.get('rc_report')}")
    say("wrote kda_t4/out/pin_t4.{txt,json}, pin_t4.baseline_pin.{json,strict.txt,report.txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
