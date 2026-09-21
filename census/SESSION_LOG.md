# Session log — unattended, 2026-09-21

Append-only. Timestamps are local.

---

## Read first

Updated as outcomes change. Three most important results so far.

1. **CORRECTION (see first entry below).** The training-free smoke test at
   `a = 1.30` was reported as "bracketing" the measured 0.233. **It does not.**
   The conditional minimizer's gap is already positive at `|w2| = 5.00`
   (`R = 0.2145`), so its switch lies in `R in (0.172, 0.215)` and the measured
   crossing at 0.2332 sits **at least 8% above** that interval. The offset is in
   the direction a bifurcation picture predicts — Block B tests it.
2. *(across-`a` outcome pending Block A)*
3. *(bifurcation outcome pending Block B)*

---

## 2026-09-21 — correction recorded before any new work

**What was said**: "The switch brackets the measured 0.233 with no training at
all", citing the smoke-test table at `a = 1.30`.

**What is true**: the smoke test evaluated `|w2|` at 2.00, 4.00, 5.00, 5.50,
6.00, 8.00. The gap is already **positive at 5.00** (`R = 0.2145`, gap
`+0.00901`) and negative at 4.00 (`R = 0.1716`, gap `-0.03271`). So the switch
lies in `R in (0.1716, 0.2145)`. The measured crossing is **0.2332**, which is
**above the whole interval**, by at least `0.2332/0.2145 - 1 = 8.7%`.

The finer grid run afterwards puts the switch at `|w2|* = 4.75`,
`R_cross = 0.2037`, giving a **12.6%** offset. Either way the prediction is an
**under-estimate**, not a bracket.

**Why this matters and is not merely a wording fix**: a systematic
under-estimate is the signature of a run following a *branch* past the point
where the global minimizer switches, and crossing only when that branch
disappears. That is the upper-spinodal picture, and it predicts
`R_measured ≈ R_spin > R_glob`. Block B computes both.

Restated in `results/phase2b_results.md` and in the Block B registration.
