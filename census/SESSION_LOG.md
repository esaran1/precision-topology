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
2. **Block A FINAL, all six `a`: O1 — geometry carries the threshold.**
   Per-`a` median crossing `R` = **0.2330 / 0.2375 / 0.2418 / 0.2456 / 0.2489 /
   0.2563** while median `|w2|` falls **5.43 / 4.45 / 3.76 / 3.25 / 2.85 /
   2.30**. **CV(R) = 0.0311** [0.013, 0.041] against **CV(|w2|) = 0.2821**
   [0.121, 0.364] — 9x apart, registered O1 satisfied
   (`CV(R) <= 0.15` and `CV(R) < CV(|w2|)/2`). Within-cell CV is ~0.065, so `R`
   varies little more across `a` than within one cell.
   **Drift check**: measured `R` drifts **+10.0%** over the range; `R_glob`
   drifts **+6.5%** and `R_spin` **+16.9%**, both the **same sign**, with
   `corr(measured, R_glob) = +0.980` and `corr(measured, R_spin) = +0.947`.
   The predicted switch drifts the same way the measurement does.
3. **Block B at `a = 1.30`: NO HYSTERESIS WINDOW — a registered falsifier.**
   `R_glob = R_spin = 0.2145` exactly. Following the `G <= 0` branch upward by
   continuation, the gap rises **smoothly through zero** (d(gap) per 0.05 step
   in `|w2|` is 0.00185, 0.00181, 0.00177, 0.00173, 0.00169, 0.00166, ...,
   monotone and continuous). There is **one branch, not two**: the crossing is
   transcritical/continuous, not a saddle-node, so there is no bistability and
   nothing to be hysteretic about.
   **Consequences**: the "loss-landscape bifurcation with hysteresis" framing is
   dropped. Block E runs the **reduced arm set** (control, null, freeze-low,
   freeze-high, jump) at the single switch point, per the registration. The
   freeze-mid and down-hold-mid arms are not defined and are not run.
   Confirmed a second way: the `G > 0` branch followed **downward** from a
   known good placement is lost at `|w2| = 4.75` (`R = 0.2037`), one grid step
   below the switch. (The automated scan's `R_fold = 0.1072` was **spurious** —
   its downward seeding fell through to an unrelated basin; fixed and rerun.)
   **Block C explains the offset**: measured crossing 0.2332 against switch
   0.2145 is **+8.7%**, and runs track the conditional branch to **0.8%** in the
   400 steps before crossing and **0.17%** after, having relaxed onto it from
   random initialisation (distance 1.09 → 0.83 → 0.39 → 0.15 → 0.008). The
   offset is **relaxation lag**: a residual gap deficit of 0.0144 on a branch of
   slope 0.033 per unit `|w2|`. Small, one-signed, consistent.
   **The derived threshold stands**: `R = 0.2145` at `a = 1.30` from the loss
   landscape with no training run in it, against a measured 0.2332.

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

## 2026-09-21 — VALIDITY GATE (rule 7): Block B inner optimiser under-converged

**Caught before any switch point was recorded.** The conditional minimiser ran 900 inner steps. At `a = 1.30`, `|w2| = 6.0`, that gives gap **-0.003070**; at 3,000 steps the same point gives **+0.028333**, and the loss falls from 0.328331 to 0.317968. The sign of `G` at the minimiser — the entire quantity Block B measures — was wrong.

Convergence swept at `|w2|` = 4.0, 5.0, 6.0 over 900/2,000/3,000/5,000/8,000 steps: every point is **still moving at 2,000 and settled by 3,000**, unchanged through 8,000. `STEPS` raised 900 -> 3,000.

**Nothing was committed from the under-converged code**; the earlier `phase2b_conditional.py` run (600 inner steps) is therefore also suspect and its R_cross/R_solve numbers are **superseded by Block B** rather than trusted. Those were the 0.2037 / 0.3600 figures; they will be recomputed.

## 2026-09-21 — VALIDITY GATE 2 (rule 7): the log-2 constant predictor

The two-stage screening initially returned **loss 0.693147 = log 2** as the conditional minimiser at `|w2| = 6.0`, with `gap` exactly 0. All six top cheap candidates converged there.

That is the **constant predictor** (`w1 -> 0`): a genuine stationary point of the loss with **no placement content**, which wins the cheap screen at large `|w2|` because driving `w1` to zero is an easy way to reduce the loss from a bad start. Taking it as 'the minimiser' would have set `gap = 0` at every large `|w2|` and corrupted all four switch points.

Excluded by `is_degenerate` (`|loss - log 2| < 1e-4` or `|w1| < 1e-3`) at both screening and convergence stages. With it excluded the gap is cleanly monotone at `a = 1.30`: **-0.0401 (4.0), +0.0014 (5.0), +0.0283 (6.0), +0.0606 (8.0)**.

Also in this pass: the scan cost was 21 s per grid point at 50 restarts x 3,000 steps, which is ~0.5 h per `a` for the global scan alone. Restructured to **coarse-bracket then refine**, plus **screen at 600 steps and converge only the best 8** -- 5.8 s per grid point, same answers.

## 2026-09-21 — Block B: `R_fold` from the automated scan is SPURIOUS

The scan reported `R_fold = 0.1072` at `a = 1.30`, which would be a hysteresis window of width 0.107 and would contradict the falsifier already recorded. **Checked directly and it does not hold.**

Seeding the downward continuation from the known good placement at `|w2| = 6.0` (`gap +0.0283`) and stepping down:

| `|w2|` | 5.50 | 5.25 | **5.00** | 4.75 |
|---|---:|---:|---:|---:|
| gap | +0.01618 | +0.00916 | **+0.00139** | **−0.00726 LOST** |

The `G > 0` branch is lost at `|w2| = 4.75`, `R = 0.2037` — **one 0.25 grid step below** `R_glob = R_spin = 0.2145`. That is grid resolution, not a window.

**Why the scan was wrong**: its downward pass seeds by screening at the top of the range, and at `|w2| = 9.0` *every* non-degenerate screened candidate is absent — the log-2 constant predictor dominates there, so the seeding fell through to a lower `|w2|` and picked up an unrelated basin. Fixed by seeding the downward branch from `best_conditional` at a mid-range `|w2|` where the good placement is the actual minimiser.

**The falsifier stands: there is no hysteresis window.** `R_fold ≈ R_glob ≈ R_spin ≈ 0.21` to within a grid step. Block E keeps the reduced arm set.
