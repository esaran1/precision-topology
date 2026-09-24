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
3. **`R_spin` DROPPED: a single continuous switch at `R_glob`.** On a 0.01 grid
   `R_glob = R_spin` to 1e-15 at **all six `a`**, and every apparent hysteresis
   window is **exactly one grid step** (0.00053 to 0.00111). The 0.05-grid
   windows were resolution. Confirmed independently in the scaling limit
   (`R_glob^inf = R_spin^inf = 0.19991`). **Withdrawn**: the `R_spin` drift of
   +16.9%, the "crossings track the upper spinodal" test and its 3-of-3
   prospective result, and **D-1, which is now vacuous** (its two models are
   identical). **Replaces it**: crossings sit above `R_glob` by 1.087-1.144
   (mean 1.115, sd 0.020), one-signed; drift measured +10.0% vs `R_glob` +4.5%,
   **r = +0.9965** (the refinement *improves* the correlation from +0.980).
   Original falsifier entry, still accurate for `a = 1.30`, follows:
   **Block B at `a = 1.30`: NO HYSTERESIS WINDOW — a registered falsifier.**
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

**`hold_high` is a CAUSAL CONFIRMATION of the second threshold**: the held
`R = 0.24663` sits **strictly between** `R_glob = 0.21446` (+15.0%) and
`R_solve = 0.30667` (-19.6%), and in that interval the conditional landscape
predicts placement achievable and solving NOT achievable. Observed: **33/37
placed, 0/40 solved.** Checked against the conditional minimiser at exactly the
held `|w2| = 5.750`: gap `+0.022543`, **does not solve** -- so solving is
*impossible* at that `R`, not merely unattained. Block E therefore tests **both**
switch points: below `R_glob` placement is destroyed (0/37), between the two it is
stable but unsolvable (33/37, 0/40), free it grows past `R_solve` and solves
(37/40).

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

## 2026-09-21 — four-item follow-up: what changed

**Item 1 — `hold_high` restated.** Its held `R = 0.24663` lies **strictly
between** `R_glob = 0.21446` and `R_solve = 0.30667`, so 33/37 placed and 0/40
solved is the landscape's prediction met, not a shortfall. Verified against the
conditional minimiser at exactly `|w2| = 5.750`: gap `+0.022543`, **does not
solve** — solving is *impossible* there. Block E therefore tests **both**
switch points.

**Item 2 — the stall is SATURATION, not trapping.** Registered expectation was
trapping; it is wrong. Gradients fall to **0.07% of control's** and a 5x budget
recovers placement **2/15 -> 7/15** (S-A satisfied); the loss sits **+0.34**
above the conditional minimum (S-B's signature) but that is explained by the
endpoint: an **ill-conditioned flat valley** near the constant predictor
(`w1 ~ -0.003`, loss ~ log 2, Hessian eigenvalues ~1e-6 against ~25-50,
condition number >1e7). **Not a stationary point** — at `w1 = 0` the gradient is
0.18-0.36, because the sampled classes' first moments differ (+0.063 vs +0.148).
**Consequence: the annealing framing is NOT adopted.**

**Item 3 — Block H's validity gate FAILED.** `m = 1` places 20/40 against the
unconstrained control's 37/40: **42.5 pp, p = 4.3e-5**. Cause: the natural `|w2|`
trajectory is **non-monotone** (seed 0 falls 0.371 -> 0.162 before rising to
9.45), so a linear schedule forces growth through the phase where the run
naturally shrinks `|w2|` while re-orienting `(w1,b1)`. The dose-response is
present and strictly monotone (**0.550 / 0.500 / 0.375 / 0.000**, endpoint
p = 8.4e-09) but is **not a valid rate test**. `m=0.5` vs `m=1` is **not
separated** (p = 0.823), so there is no "slower is better", only a penalty for
speed. The valid design — replay each seed's own trajectory time-warped — is
recorded, not run.

**Item 4 — done, see Read first #3.**

**Block G — the machinery transfers.** Four unseen task windows, nothing
refitted. `|w2|` at crossing spans **5.8x** while `CV(R) < CV(|w2|)/2` at both
`a` (**3.36x, 3.01x**); each window's `kappa_0` from the cubic alone matches its
measured `kappa(1.02)` within **0.81%, 5 of 5**, across an **8.1x** range of
`kappa_0`. G-1 fails 6/10 on a flat 15% tolerance, but the offset is a property
of `a` not the window (sd 0.014/0.016; all ten within **2.72%** after
normalising by the base task's own offset).

## 2026-09-22 — push: `main` carries the full history, the 2.7 GB CSV stripped

The first push of `master` **failed**: `results/phase2b_checkpoints.csv` is
**2.7 GB** in the current tree and ~9.7 GiB across the **seven** versions it had
been committed in — far over GitHub's 100 MiB hard limit, and sitting in the
middle of the 28 unpushed commits, so no incremental push could succeed. (The
attempt appeared to succeed because `git push ... | tail` returns `tail`'s exit
status, not git's; the real error was `RPC failed; HTTP 500`. Exit status is
checked directly now.)

An intermediate single-commit snapshot branch (`clean-master`) was pushed and
then **deleted**: it discarded the per-commit registration trail, which is the
evidence for which predictions were registered before the data that tested them.

**What is on the remote now: branch `main`, 318 commits.**

The 289 already-published commits were **never touched** — `origin/master` was
verified to contain no oversized blob and to be a clean ancestor of the new work,
so only the 28 new commits needed rewriting. Those 28 were **replayed** onto it
with the CSV removed from each tree, preserving every message, author and **both
timestamps**; one further commit adds the `.gitignore` entry.

**The audit chain is intact and verifiable from the remote**, which is the point:

| time | commit | event |
|---|---|---|
| 18:38:04 | `c05c03e` | Block B registered |
| 18:41:03 | `94121eb` | Block A interim — crossing medians read for 1.30/1.35/1.40 |
| 18:56:15 | `b7734c2` | seeding fix + frozen sha `9f1b10741d8bf48c` |

Same times as the originals (`9e30027`, `f00ce39`, `3497f7f`); only the SHAs
differ, because the trees differ by the removed CSV. The timestamp audit of
2026-09-21 — which downgraded the prospective set from 4 to 3 on exactly these
three commits — remains reproducible from `origin/main` alone.

Verified: the `main` tree differs from the original `master` **only** by the
removed CSV and the `.gitignore` line; no blob over 50 MiB anywhere in the new
commits; `verify_ledger.py` passes from it with 0 findings.

**Branches**: remote default is now `main`. `origin/master` is kept as a frozen
pointer to the old 289-commit state and is fully contained in `main`. Locally,
`master` and `master-full-local-20260922` both still hold the original 28 commits
**with** the CSV, as backups.

`phase2b_checkpoints.csv` is a raw per-step training log — an input, not a
result. Regenerate with `src/phase2b_ordering.py`. Every number derived from it
(Blocks A and C) is committed in `blockA_crossings.csv`, `blockA_per_a.csv` and
`blockC_adiabatic.csv`.

## 2026-09-22 — priority run: K, F, and the kappa certification

**Block K — family B suspension resolved.** `Ĝ_norm(α) = 0.4|α|` (max gap at
`|w₁| = 1`) is **forced by homogeneity**, not chosen. The committed box-limited
`Ĝ` was exactly **8.000× Ĝ_norm** at every α, confirming the diagnosis.
- **`β_B = 1` RESTORED** — exponent 1.000000. The suspended 0.9935 was
  *accidentally right*: homogeneity forces linearity at any box size.
- **Overshoot RESTORED at 11.8×–145×** (was 922–1,160×, inflated exactly 8×).
  *My registered range "115–145×" was arithmetically wrong — I omitted the
  2,000-budget row at 11.8×. The 8× factor and the >10× floor both hold.*
- **AUC comparison REMOVED permanently.** Margin over `|w₂|` is **0.0089** vs a
  registered ≥ 0.10. Two fatal reasons: within fixed α, `R_B` and `|w₁w₂|` have
  **identical AUC by construction**; and 4 of 5 α cells are perfectly separable
  (α = −1: unsolved max **0.295** vs solved min **7.26**, **2 runs between**)
  against T54's own ≥20 band-run requirement. **The box was a second defect —
  this claim was never supportable at any `Ĝ`.**
- Onset exponent and the counterexample status unchanged (never used `Ĝ`).

**Block F — the relaxation-lag account is a LABEL, not a mechanism.** Instrumented
crossing `R` fresh, since no artifact had per-optimiser *crossing* `R` (only
terminal `|w₂|`). `R_glob(1.25) = 0.21066`, new.
- Adam **0.23094** / AdamW **0.22980** / SGD **0.22888**, with a **3.9× spread**
  in growth rate.
- **F-1's ordering holds exactly but has no power**: bootstrap CIs overlap almost
  completely, **P(full ordering) = 0.290** vs 0.167 by chance.
- **F-2 FAILS**: SGD's predicted offset **2.46%** against **8.65%** measured, a
  **6.19 pp** miss. The account says SGD's offset should be a quarter of Adam's;
  it is **90%** of it. Offsets span **0.98 pp while rate spans 3.9×**.
- **F-3 passes (Spearman +0.783, restated against `R_glob` per T60) but is
  confounded**: rate and offset both rise with `a` while rates vary only 9%.
- Per the registration's own falsifier, **dropped**. **What survives**: the offset
  is robust and one-signed (8.6–9.6% across optimisers, 8.6–14.4% across `a`),
  which is what makes the training-free prediction useful; Block C's
  relaxation-then-tracking is unaffected. **Second failed rate account**, after
  Block H.

**The κ certification** (registered separately — *not* Block H, which is the
already-run dose-response). `Ĝ` is a **supremum**, so a grid can only
**under-estimate** it; T50's interval had no error bar.
- **C-1 PASSES**: certificate widths **0.022–0.086%** at ten `a` (registered
  < 0.2%), via `Ĝ_grid ≤ Ĝ ≤ Ĝ_grid + Lh/2` with `L = (1+a)·max(1,max|x|)`.
- **C-2 splits**: magnitude passes (endpoints move **+0.755%**, **+0.693%**), but
  **containment FAILS** — certified `[0.307747, 0.317616]` excludes T50's lower
  endpoint 0.30544. The grid was low at **every** `a` (0.13–0.68%). **T50's
  substance survives**: the *width* the theorem chain uses is **3.207% certified
  vs 3.271% reported**. The interval's **location** is corrected upward.
- **C-3 PASSES**: boundary-shell gap **non-positive**, so the domain cut is sound.
- **C-4 PASSES**: `K ∈ [0.579454977, 0.579950977]` (width 0.086%), with the
  reported `K` and `κ₀` at the **lower endpoints** to 7 and 6 digits.
- **Consequence for T58**: certified `κ(1.02)` and `κ₀` are **disjoint**, closest
  approach **0.145%**. The reported **0.061%** was a coincidence of two
  under-estimates. S-1 unaffected (3% tolerance), but quote **0.15% certified**.

**Naming note**: the priority list called the κ certification "Block H". Block H
is the growth-rate dose-response (registered `3c1c0f4`, run, reported). The
certification is recorded under its own name to keep the register unambiguous.

---

**Machine crash, 2026-09-24 (about 00:40 EDT).** The laptop crashed under memory pressure from long certified
computations.
- **Jobs lost**, none of which had finished:
  - the annulus certificate (10.5 h);
  - the solve competitor check (7 h);
  - the c₁ test's registered finite run (a = 1.02–1.04 held their results in memory; a = 1.01 was paused in its Ĝ
    step);
  - the per-window A* (queued, restarted).
- **No registered result was affected.** Every scored result had already been committed. `/tmp` logs were lost.
- **Every lost job reruns** from its registered or amended procedure:
  - the c₁ test: as registered at a = 1.02–1.04, and under amendment 2 at a = 1.01;
  - the annulus and the solve competitor check: redesigned before rerunning, because the original design could not
    close (math_note_v2);
  - the per-window A*: unchanged.
- **Since then**: long jobs checkpoint each finished unit to disk; at most 3 workers at low priority; a memory
  watchdog pauses any job above 3 GB.
