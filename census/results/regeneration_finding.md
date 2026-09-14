# Part 3, Priority 1: a transcription error in a committed artifact

**Found by end-to-end regeneration** (rerunning the sweep from current code and
diffing against the committed artifact), 2026-09-13. This is exactly the class
of error that checking numbers *against* artifacts cannot find.

## What happened

`results/onset_curves.csv` held three wrong rate values at `B = 8,000`:

| `a` | committed | `onset_law.log` (the original run) | regenerated (current code) |
|---:|---:|---:|---:|
| 1.30 | 0.900 | 0.900 | 0.900 |
| 1.25 | 0.875 | 0.875 | 0.875 |
| 1.22 | 0.875 | 0.875 | 0.875 |
| **1.20** | **0.875** | **0.775** | **0.775** |
| **1.18** | **0.850** | **0.600** | **0.600** |
| **1.16** | **0.475** | **0.200** | **0.200** |

**The log and the regeneration agree exactly. The committed artifact was wrong.**

## Cause

`src/onset_resume.py` was written to resume a sweep killed mid-run by a stray
`pkill`. The completed `2k/8k/32k` cells were **not recomputed** — they were
**hardcoded as literals transcribed by hand from `onset_law.log`** (see its
docstring: "recovered from the killed run's log"). Three of the six `B = 8,000`
values were mistyped in that transcription. The script then wrote those
literals into `onset_curves.csv` as if they were measurements.

Every other cell it transcribed (2k, 32k, and three of six at 8k) is correct,
so this was manual-copy error, not a systematic offset.

## Impact on reported numbers: none, and the correction slightly helps

**The onset at `B = 8,000` is 1.18 either way**, bracketed by `a = 1.16` in
both versions (0.475 and 0.200 are both below 50%). So:

- onset exponent **−0.7340** — unchanged
- `alpha` **1.1172** — unchanged (different artifact entirely)
- through-origin slope **1.0984** — unchanged
- ledger verifier: **0 discrepancies**; figure audit: **0 findings**

Two derived quantities that *consume the rate values* move, and both move
**toward** the rest of the evidence:

| quantity | before | after |
|---|---:|---:|
| collapse minimising `theta` | 0.7250 | **0.7500** (onset-derived: 0.7340) |
| 25% threshold exponent | −0.5037 (3 of 6 bracketed) | **−0.6581 (4 of 6 bracketed)** |
| 50% / 75% | −0.7325 / −0.7325 | **−0.7367 / −0.7487** |

The 25% level, previously the outlier in the threshold-independence test, is
now both better bracketed and closer to the 50% value. The corrected data makes
the threshold-independence claim *stronger*, not weaker.

## Action taken

`onset_curves.csv` corrected to the logged and regenerated values. The three
wrong cells are replaced, not deleted; this note records the original values.

## What this says about the reproducibility claim

The Reproducibility Statement should say that **artifacts were regenerated from
current code and diffed**, that **one transcription error was found and
corrected this way**, and that **no reported number moved**. That is a stronger
and more honest statement than "all numbers verify against artifacts", which
was true while this error was present.

**Lesson, generalising**: any script that hardcodes previously-measured values
rather than recomputing them is outside every verifier. `onset_resume.py` is
the only such script in the repository (checked); it is now annotated.
