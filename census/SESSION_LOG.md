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
4. **The scaling limit DERIVES why `R` is the right variable.** Near the fold,
   `f_a(pi+s) - pi = eps^{3/2}[-sigma + (1+eps)sigma^3/6] + O(eps^{5/2})` exactly,
   with `sigma = s/sqrt(eps)`, so `|w2|` and `eps` enter only through
   `W = w2 eps^{3/2}` -- and `R = |w2| Ghat/2` with `Ghat = K eps^{3/2}` is the
   scale-free form of `W`. Constants obtained with **no training and no fit**:
   `K = 0.579454926`, **`kappa_0 = 0.307302` against measured `kappa(1.02) =
   0.306122` (0.38%, registered tolerance 3%)**, and, from Block B run directly on
   `h(sigma)`, **`R_glob^inf = R_spin^inf = 0.19991`** -- **no hysteresis window in
   the limit either**, reproducing the `a = 1.30` falsifier from the limit
   activation alone -- and **`R_solve^inf = 0.30711`**.
   **Registered S-3 splits.** For `R_glob` it passes: deviations +7.28% to
   **+14.29%** over `a = 1.30..1.60`, all positive, strictly monotone,
   `corr(eps,dev) = +0.9866`, log-log slope 0.947, and
   `R_glob(a) = 0.19991(1 + 0.231 eps)` fits all six to max residual 0.00133,
   inside the grid spread. For `R_solve` **it fails**: all six deviations are
   negative (mean **-0.29%**). The failure is explained, not repaired -- `R_solve`
   is *already at* its limit value at every `a`, which is why its across-`a` CV was
   0.0020; a constant sequence cannot show the predicted approach.
   **Step 5 (exact loss invariance) fails at the `eps` used**: the neglected term
   is 6.3% of the leading one at `a = 1.30` and 16.4% at `a = 1.60`. Asymptotic
   only. **`c1 = 0.231` is fitted, not derived** (the `K(eps)` argument predicts
   0.49, a factor 2.4 off, because `R_glob` tracks the conditional *minimiser*,
   which trades gap against loss rather than maximising gap).
   The measured **+10.0%** drift in crossing `R` across `a` is then the same
   `O(eps)` correction that moves `R_glob` by +7 to +14%, same direction.

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

## 2026-09-21 — TIMESTAMP AUDIT: the prospective set was 3 values, not 4

The Block B registration named `a` = 1.40, 1.45, 1.50, 1.60 prospective. **`a = 1.40` is not.** Commit timeline:

| time | commit | event |
|---|---|---|
| 18:38:04 | `9e30027` | Block B registered |
| **18:41:03** | `f00ce39` | Block A interim — **read crossing medians for 1.30, 1.35, 1.40** |
| 18:46:23 | `87ee918` | log-2 exclusion — procedure changed |
| **18:56:15** | `3497f7f` | **seeding fix + frozen sha `9f1b10741d8bf48c`** |

The seeding fix is what creates the `R_fold`/`R_spin` separation at `a >= 1.35`; before it no window existed anywhere. So the `R_spin` values at those `a` come from a procedure finalised **after** I had seen the 1.40 crossing.

**Genuinely prospective: `a` = 1.45, 1.50, 1.60.** Result on that set is 3 of 3, all within 15%. Corrected in `blockB_results_final.md`.

Also recorded there: **the 4/4 (now 3/3) test does not discriminate.** `R_spin >= R_glob` always and every measured crossing exceeds `R_spin`, so 'closer to `R_spin`' follows from the ordering alone and holds equally under a pure-lag account with `R_glob` as the true switch.

## 2026-09-21 — VALIDITY GATE 4 (rule 7): the scaling-limit grid was 7x too coarse

Block B on `h(sigma)` uses `W = w2 eps^{3/2}`, so a 0.05 step in `W` is **0.0145
in `R_inf`** against 0.0021 at finite `a` -- seven times coarser. The coarse scan
returned `R_solve^inf = 0.3183`, which sits 3.8% **above** every finite-`a`
value and would have been recorded as a clean S-3 failure with the wrong
explanation. **Refined to step 0.005**: `R_solve^inf = 0.30711`, `R_glob^inf =
R_spin^inf = 0.19991`. The refined `R_solve^inf` is **0.29%** from the six-`a`
mean. Caught before the result was written up; both grids are reported in
`scaling_limit_results.md`.

## 2026-09-21 — the secondary item: `a = 1.60` window at step 0.01

Running. Registered threshold: a freeze-mid arm is added only if the refined
`R_spin - R_fold` window is **at least five grid steps of 0.01 in `|w2|`** wide.
Reversibility remains the primary Block E result either way.

## 2026-09-21 — Block E: the causal test passes, and two registered arms fail

**Primary result (prospective).** Runs trained to placement, held 200 steps, then
pinned to one side of `R_glob = 0.21446` and held there:

| arm | held at | kept placement | CP95 |
|---|---|---|---|
| `hold_low` | 0.85 R_glob | **0 / 37** | [0.0000, 0.0949] |
| `hold_high` | 1.15 R_glob | **33 / 37** | [0.7458, 0.9697] |
| `noise_floor` | x(1+1e-6) | 37 / 37 | [0.9051, 1.0000] |

**Fisher exact p = 1.2e-16.** Placement is lost within **50 steps** (the first
checkpoint) in all 37 `hold_low` runs -- median = min = max = 50.

**`hold_high` separates the two thresholds without being designed to**: places
33/40, solves **0/40**, sitting 15% above `R_glob` and 19.6% **below**
`R_solve = 0.3067`. That is the placed-but-unsolved regime T57 predicts.

**Two registered arms FAILED.**
- `cold_high` (registered "placement achieved"): **2 of 40**; a pilot at
  1.50 R_glob over 20,000 steps gives **0 of 9**.
- `jump` (registered "places well before control"): **falsified in the opposite
  direction**, 9/40 vs control 37/40, Fisher **p = 1.3e-10**. Pre-supplying
  `|w2|` STALLS the run: `R` moves 0.225 -> 0.236 in 19,500 steps while control
  passes the same `R` near step 4,000 and reaches 1.80. Dose-response over 5
  seeds: control 4/5, `w2` alone 2/5 (~5x later), `w2`+`b2` 0/5.

**Net**: `R > R_glob` is **necessary for placement to persist, not sufficient for
it to be found**. Growth and placement are coupled and ordered -- `|w2|` must
grow *while* `(w1,b1)` are shaped. The paper must not claim `R` predicts when
gradient descent succeeds; it predicts when success is stable.

**Instrument note**: the arm named `null` round-tripped through CSV as NaN
(`pandas` default NA). Renamed **`noise_floor`** in code and in the committed
artifact; no data changed, verified by re-reading with `keep_default_na=False`.

## 2026-09-21 — the a >= 1.35 window audit at step 0.01

Running. `a = 1.60` already refuted its own window (0.0279 at step 0.05 -> 0.00111
at 0.01, with `R_glob = R_spin` exactly). If 1.35-1.50 follow, every `R_spin`
above `a = 1.30` in `blockB_switches.csv` is a grid-limited upper bound and D-1's
two models coincide identically.
