# Rule audit — does each rule have a real event behind it?

Class A = a concrete measured event (command run, number changed, bug found)
Class B = a real observation, but from discussion rather than a measurement
Class C = no event at all — written from plausibility

| rule | class | the event |
|---|---|---|
| 0 negative control | **A** | `sass_lint` returned 0 forever; reverting bug 4 turned 3 controls red |
| 1 never a bare multiple | **A** | "43×" was read as speed when it was instruction count |
| 2 name the statistic | **A** | `n` was 7.4 / ~30 / 12.88 — three statistics, one word |
| 2b expiry on world-claims | **B** | no event; derived from 2 on Tim's per-period data |
| 3 magnitude excludes | **B** | reasoned, not measured |
| 4 independence is a degree | **A** | two regexes sharing `{4}` agreed on a wrong answer |
| 5 residuals must not be named | **B** | the −0.62pp remainder was *not* mis-named in the end |
| 6 probe width = claim width | **A** | `hoist_detect` measured the IR it inspected |
| 7 suspect your parser first | **A** | 3,842 vs 16,952 — 77% dropped |
| 8 say "unknown" | **B** | right by construction; no counter-event |
| 9 commit the output | **A** | day's whole output untracked; controls claimed but unautomated |
| 9a message carries WHY | **A** | my own commit was an inventory, minutes after Tim named it |
| 9b provenance unrecoverable | **B** | argued from the `.agent-scratch` case, but no test |
| 10 cannot-go-red | **A** | gate broken → verify exited 1; then healthy → 0 |
| 10a read ≠ tested | **A** | I called a guarded file a "gap" after reading its guard |
| 10b recursion stops | **B** | no event; a boundary claim |
| 10c measure exit code | **A** | `cmd \| tail; echo $?` read "OK" off a broken gate |
| 11 files stay untracked | **A** | staged then failed commit left the index dirty |
| 11a failure mode not label | **A** | parser raises (not guesses) — opposite of the guess |
| 11b checklist from blanks | **B** | real but from one conversation, no measurement |
| 12 "can't be done" needs evidence | **A** | 0/25 commits had the trailer I claimed existed |
| 12a restatement | **A** | same event; also `verify_gate.py` earlier that day |
| 13 delivery is a position | **A** | my files in Tim's tree stalled his gate run |
| 13a grep by name | **A** | 9 hits were his `test_pit_leakage_canary.py` |
| 13b residue vs design | **B** | distinction holds, but both cases were the same day |
| 14 feature ≠ identity | **A** | 4 instances: HMMA=0, exclude, canary, .pyc |
| 14a nothing-wrong isn't evidence | **B** | meta-claim; correctly about itself |
| 14b tool prints candidates | **A** | 10 vs 5 files; `.git/` and `.venv/` in my version |

**A: 18   B: 10   C: 0**

No rule is class C. Ten are class B — real thinking without a measurement.
The audit itself is class B: it classifies by my own reading.
