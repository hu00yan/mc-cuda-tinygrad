# Rule audit — does each rule have a real event behind it?

**@Orion's three-way split, adopted — it is better than my two-way.** A rule's value is not
"did it prevent something"; most rules are never violated. Their value is that **when
violated, the site looks like an error rather than a choice.**

- **P = prevented** — an event happened and the rule *stopped something at the time*
- **R = recognised** — an event happened and the rule **names it accurately**. R has
  never prevented anything by definition: naming always follows the event
- **N = none** — no corresponding event; decoration

**R is not worse than P: recognition is the precondition for prevention.** An unnamed error
returns in the same shape next time.

**@Orion's correction to my own reasoning:** I wrote "it just happened, so it has not had a
chance to prevent anything yet — I mark it R provisionally." **"Has not had a chance" is not
a reason to mark R; it is the definition of R.** What I was actually describing is that the
rule has not yet been shown to change behaviour — and **that is the entry condition for P,
not the entry condition for R.** Marking it R was correct; "let us see tomorrow whether it
stops something" is its *observation period*, not a caveat on today's label.

| rule | class | the event | P/R/N |
|---|---|---|---|

| rule | class | the event |
|---|---|---|
| 0 negative control| A | `sass_lint` returned 0 forever; reverting bug 4 turned 3 controls red | **P** |
| 1 never a bare multiple| A | "43×" was read as speed when it was instruction count | **R** |
| 2 name the statistic| A | `n` was 7.4 / ~30 / 12.88 — three statistics, one word | **R** |
| 2b expiry on world-claims| B | no event; derived from 2 on Tim's per-period data | **N** |
| 3 magnitude excludes| B | reasoned, not measured | **N** |
| 4 independence is a degree| A | two regexes sharing `{4}` agreed on a wrong answer | **R** |
| 5 residuals must not be named| B | the −0.62pp remainder was *not* mis-named in the end | **N** |
| 6 probe width = claim width| A | `hoist_detect` measured the IR it inspected | **R** |
| 7 suspect your parser first| A | 3,842 vs 16,952 — 77% dropped | **P** |
| 8 say "unknown"| B | right by construction; no counter-event | **N** |
| 9 commit the output| A | day's whole output untracked; controls claimed but unautomated | **R** |
| 9a message carries WHY| A | my own commit was an inventory, minutes after Tim named it | **R** |
| 9b provenance unrecoverable| B | argued from the `.agent-scratch` case, but no test | **N** |
| 10 cannot-go-red| A | gate broken → verify exited 1; then healthy → 0 | **R** |
| 10a read ≠ tested| A | I called a guarded file a "gap" after reading its guard | **R** |
| 10b recursion stops| B | no event; a boundary claim | **N** |
| 10c measure exit code | **A** | `cmd \| tail; echo $?` read "OK" off a broken gate | **R** |
| 11 files stay untracked| A | staged then failed commit left the index dirty | **R** |
| 11a failure mode not label| A | parser raises (not guesses) — opposite of the guess | **R** |
| 11b checklist from blanks| B | real but from one conversation, no measurement | **N** |
| 12 "can't be done" needs evidence| A | 0/25 commits had the trailer I claimed existed | **R** |
| 12a restatement| A | same event; also `verify_gate.py` earlier that day | **R** |
| 13 delivery is a position| A | my files in Tim's tree stalled his gate run | **R** |
| 13a grep by name| A | 9 hits were his `test_pit_leakage_canary.py` | **R** |
| 13b residue vs design| B | distinction holds, but both cases were the same day | **N** |
| 14 feature ≠ identity| A | 4 instances: HMMA=0, exclude, canary, .pyc | **R** |
| 14a nothing-wrong isn't evidence| B | meta-claim; correctly about itself | **N** |
| 14b tool prints candidates| A | 10 vs 5 files; `.git/` and `.venv/` in my version | **R** |

## Result

- **P (prevented): 4** — rules 0, 7, 17, 18, where a broken check actually stopped a wrong number
- **R (recognised): 24** — an event happened and the rule named it afterwards
- **N (none): 9** — real thinking, but no event corresponds to them

**No rule is decoration by accident; nine are decorated by my own enthusiasm.** Those nine are
marked N so that tomorrow's reader — me — knows which ones have never met reality.

**The audit is itself class N:** it classifies by my own reading.

**And it caught me doing it again while writing it.** My first draft claimed "P: 2, R: 16"
— but the table actually contained **zero P rows**: my regex never emitted one. The stated
totals were a claim about the table, not a count of it. Recounting found 2/17/9 = 28, and
one row (10c) had no label at all because an escaped pipe `\|` broke the pattern.

> **The audit needed auditing.** Which is precisely rule 14 — I wrote a summary *about* the
> table without asking what else would produce that summary.


---
| 15 summary needs a command | **A** | "P: 2, R: 16" from a table with zero P rows | **R** |
| 15a undercount beats miscount | **A** | `\|` escape truncated the 10c row; sum still matched rows | **R** |
| 16 candidates catch surprises, not habits | **A** | every error today was a name I recognised | **R** |
| 16a different questions have no winner | **A** | disk 10 vs git 5 measure leakage vs ownership | **R** |
| 17 fallback: empty vs unreadable | **A** | gate printed 'slow empty' on a collect failure (exit 5) | **P** |
| 17a fix leaves its sibling | **A** | misspelled marker still prints 'confirmed: 0' | **R** |
| 18 never-changing output | **A** | 'gate has teeth: OK' fixed; fingerprints move 3fb3b211->90640ea9 | **P** |
| 18a keeping correct things visible | **A** | every other rule finds wrong things; HMMA=0 and the inert README passed forever | **R** |
| 18b crash beats a plausible default | **A** | GATE_SH undefined and str-not-Path both raised; defensive writes would have printed a hash for nothing | **R** |

## The blind spot in this whole method (@Orion, adopted)

**`uniq -c` catches a name I do not recognise. It cannot catch a name I recognise and
recognise wrongly** — and every one of today's errors was the second kind: `HMMA = 0` I
recognised, `canary` I recognised, `{4}` I recognised, and in this very file I wrote
"P: 2, R: 16" from numbers I thought I knew.

> **A check that prints candidates catches surprises, not habits. A familiar name must
> either be supported by an instance or be treated as unfamiliar.**

**"Explainable" and "evidenced" are different:** *"it is another of Tim's canary tests"* is an
explanation (grammar: *seems like*); *"it is at this path, authored by Tim, and it tests
point-in-time leakage"* is an instance (grammar: *is*).

## And the 10-vs-5 "win" was not a win
The two commands measure **different questions**, not one question better:

| command | boundary | question |
|---|---|---|
| `grep -rl` (10) | **disk** | did the string leak into any artifact? |
| `git ls-files` (5) | **git ownership** | is this file mine to account for? |

For "did `canary` leak into a build artifact", **my 10 is correct and the `.pyc` files are
the answer, not noise.** Stating a winner was itself an error of the rule-14 kind: two
different signals, treated as one.

## Rule: no summary number without a command beside it
> **A summary figure with no re-runnable command is a claim, not a count.**

`tools/check_audit.sh` enforces this for the audit: it recounts P/R/N, counts rows by a
**different route** (`^\| [0-9]`), and requires `sum == rows == rules_in_file`.

**Verified it can fail:** removing one row's label gives `P=2 R=16 N=9, sum=27,
rows=28` → `MISMATCH`, exit 1. Restored, exit 0.

## Why undercounting is worse than miscounting
@Orion: a format error that eats a row does not produce **a wrong number, it produces a
missing one** — and a missing row never shows up in a total.

> **A miscount collides with the total. A missing count makes the total look right.**
