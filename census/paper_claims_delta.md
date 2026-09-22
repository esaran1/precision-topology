# Claims delta for the writing side

One row per claim that changed. **Old claim** is what the current draft says or
what an earlier note said; **new claim** is what the artifacts support now.
Every new claim has a ledger ID and passes `verify_ledger.py` unless marked
pending.

Updated during the unattended session of 2026-09-21, and again after the
four-item follow-up of the same date. Append-only.

---

## Corrections carried in from earlier work

| # | old claim | new claim | ledger | section |
|---|---|---|---|---|
| 1 | "200 training points" | **400 training points** — `N_PER_CLASS = 200` is *per class*, and both classes are used every step. Full batch, no minibatching. | 0b | setup |
| 2 | success = 0 errors on the evaluation sample | **success = `solves()`, the dense regional check** on 4,001 inner and 4,000 outer points. The sample clause never binds on its own: 0 runs are solved-with-errors, while **4 runs have 0 sample errors and fail the dense check**. | 0b, T57 | setup |
| 3 | "float32 and float64 differ materially; `a = 1.5` seed 0 solves in one and not the other" | **float32 and float64 agree run-for-run given a shared initialisation** — 180 paired runs, 0 flips, relative parameter distance 0.0000. The apparent effect was an RNG artifact: `torch.empty(4, dtype=d).uniform_()` consumes the stream differently per dtype, so the two arms were **different networks**, not the same network computed twice. | 0b | reproducibility |
| 4 | SGD results reported without qualification | **SGD learning rate 0.3 is the only one that solves at all** (recorded in `sgd_onsets.LR`); the SGD arm is not a generic-SGD claim. | 0b | optimisers |
| 5 | family B: `β = 0.9935` measured, 900–1,200× overshoot, AUC comparison | **RESOLVED by Block K (T62), 2026-09-22.** Under the forced normalisation `Ĝ_norm(α) = 0.4\|α\|` (max gap at `\|w₁\| = 1`, fixed by homogeneity, not chosen): **`β_B = 1` RESTORED** (fitted exponent 1.000000 — the old 0.9935 was accidentally right, since homogeneity forces linearity at any box size); **overshoot RESTORED at 11.8×–145×**, the old figures inflated **exactly 8×** by the box (`Ĝ_box = 3.2\|α\| = 8·Ĝ_norm`); **AUC comparison REMOVED permanently** — margin over raw `\|w₂\|` is **0.0089** against a registered ≥ 0.10, and it was never supportable at any `Ĝ` because within fixed α `R_B` and `\|w₁w₂\|` have **identical AUC by construction** while 4 of 5 α cells have **~2 transition-band runs** against T54's own ≥ 20 requirement. **Onset exponent +0.25 and the counterexample status UNCHANGED** — never touched `Ĝ`. | T62 | family B |
| 6 | pooled R dataset "2,910 runs, 0 of 2,160 below, 402 of 403 above" | **3,150 runs, 0 of 2,285 below `R = 0.30`, 460 of 461 above `R = 0.50`.** Same dataset with AdamW's 240 runs added; the old figures are stale, not a second population. Now a committed artifact (`r_pooled_union.csv`) and checked by the verifier. | T55 | §5 |
| 7 | the through-origin slope / `R` described as "governs" | **"predicts"** throughout. `\|w2\|` does not appear in the placement condition; its predictive power operates through the dynamics. | T57 | §5 |

## New claims from this session

| # | old claim | new claim | ledger | section |
|---|---|---|---|---|
| 8 | *(none — new)* | **Sign correctness decomposes exactly into placement and bias.** A network is sign-correct iff `G(w1,b1) > 0` **and** `b2` lies in an interval of width `\|w2\|·G`. **`\|w2\|` does not appear in the placement condition.** The identity holds run-for-run: **0 disagreements with `solves()` in 4,800 runs**. | T57 | §4 |
| 9 | *(none — new)* | **84.5% of failures are placement failures**, 15.5% bias. Sign-balanced (84.0% vs 85.0% for `w2 > 0`/`w2 < 0`, 1.0 pp apart). | T57 | §4 |
| 10 | *(none — new)* | **Within each cell, `P(G > 0)` rises monotonically with terminal `\|w2\|`** (correlation +0.78 to +0.95); the lowest `\|w2\|` quartile has `P(G>0) = 0.000` in **every** cell. Correlational. | T57 | §4 |
| 11 | *(none — new)* | **All 870 solved runs are certified sign-correct region-wide**, not merely at grid points: 866 at `h = 4.0e-4` via `\|N'\| ≤ \|w2\|\|w1\|(1+a)`, the remaining 4 at `h = 4.0e-5`. The criterion is **certified, not sampled**. | T57 | §4 |
| 12 | *(none — new)* | **The theorem's first inequality verified at each network's own placement** — previously only the uniform form was checked. **0 violations of 66 solvers**, 22 of them with `w2 < 0` so the mirrored orientation is exercised; median slack **1.2488**, min **1.0070**. | T56 | §3 |
| 13 | *(none — new)* | **Placement flips at a reproducible value of `R`, not at a reproducible time.** At `a = 1.30`, median `R` at crossing **0.2332** (within-cell CV 0.067) while the crossing step ranges **1,860 to 6,839**. | pending | §5 |
| 14 | *(none — new)* | **Geometry carries the threshold (registered outcome O1).** Per-`a` median crossing `R` is 0.2330 / 0.2375 / 0.2418 at `a` = 1.30/1.35/1.40 while median `\|w2\|` falls 5.43 / 4.45 / 3.76. **CV(R) = 0.0152 against CV(\|w2\|) = 0.1508.** | pending | §5 |
| 15 | "the reverse mechanism (placement enables `\|w2\|` growth) explains the correlation" | **Rejected.** `\|w2\|` growth rate is **identical before and after** the crossing: 0.00237 vs 0.00239 per step at `a=1.30 B=4k`, 0.00238 vs 0.00240 at `B=8k`; post/pre ratio **1.003**, stable across window sizes 50–800. There is no post-crossing acceleration. | pending | §5 |

## Claims to REMOVE

| # | claim to remove | reason |
|---|---|---|
| R1 | "the training-free switch **brackets** the measured 0.233" | **False.** The conditional minimiser's gap is already positive at `R = 0.2145`, so its switch lies in `R ∈ (0.172, 0.215)` and the measured 0.2332 sits **at least 8.7% above** the whole interval. It is a systematic **under-estimate**. |
| R2 | any conditional-landscape number from the 600-step or 900-step runs (`R_cross = 0.2037`, `R_solve = 0.3600`) | **Under-converged.** At `a=1.30`, `\|w2\|=6.0`, 900 steps gives gap `−0.003070` and 3,000 gives `+0.028333` — the *sign* was wrong. Superseded by Block B at 3,000 steps. |
| R3 | "the precision effect shows transition-band outcomes are irreproducible" | Retracted: it was the per-dtype RNG bug, not arithmetic. |

## Second session: the scaling limit, the causal test, and transfer

| # | old claim | new claim | ledger | section |
|---|---|---|---|---|
| 16 | `R` is the right variable, on the evidence that `CV(R) = 0.0311` vs `CV(\|w₂\|) = 0.2821` | **Derived, not just observed.** Near the fold `f_a(π+s) − π = ε^{3/2}[−σ + (1+ε)σ³/6] + O(ε^{5/2})` with `σ = s/√ε`, so `\|w₂\|` and `ε` enter only through `w₂ε^{3/2}`, whose scale-free form is `R = \|w₂\|Ĝ/2` with `Ĝ = K·ε^{3/2}`. **`R` is the only combination the limit problem sees.** | T58 | §5 |
| 17 | `κ ≈ 0.31`, "fixed by the task windows", measured | **`κ₀ = K/(4√2/3) = 0.307302`, computed from the cubic `h(σ) = −σ + σ³/6` alone**, against measured `κ(1.02) = 0.307483` — **0.061%**. `K = 0.579454926`. The dip depth `4√2/3 = 1.885618083` is exact. | T58 | §3 |
| 18 | *(none — new)* | **κ's near-constancy across `a` is a cancellation, not a coincidence**: `Ĝ/ε^{3/2}` and `D/ε^{3/2}` each drift ~16% over ε = 0.02–0.60 and the corrections cancel in their ratio, leaving κ within 2.7%. | T58 | §3 |
| 19 | *(none — new)* | **Training-free switch points exist in the scaling limit**: Block B run directly on `h(σ)` gives `R_glob^∞ = R_spin^∞ = 0.19991` and `R_solve^∞ = 0.30711`. Finite-`a` `R_glob` converges to it as `R_glob(a) = 0.19991(1 + 0.231ε)`, six points, max residual **0.00133** (inside the 0.002 grid spread), log-log slope **0.947**. `R_solve` is **already at** its limit value at every `a` (mean deviation **−0.29%**), which is why its across-`a` CV was 0.0020. | T58 | §5 |
| 20 | "the loss at fixed `\|w₂\|` depends on `w₂` only through `w₂ε^{3/2}`" | **False at the ε this paper uses.** The neglected term is **6.3% of the leading one at a = 1.30 and 16.4% at a = 1.60**; holding `w₂ε^{3/2}` fixed drifts the conditional gap +0.001388 → −0.055321. The invariance is **asymptotic only**, because the task windows are fixed in `x` while the fold narrows as `√ε`. | T58 | §5 |
| 21 | four switch points `R_fold < R_glob < R_spin`, a hysteresis window | **`R_spin` does not exist as a distinct switch.** On a 0.01 grid, `R_glob = R_spin` to 1e-15 at **all six `a`**, and every apparent window is **exactly one grid step**. The conditional landscape has a **single continuous switch at `R_glob(a)`**. The `+16.9%` `R_spin` drift is **withdrawn** (coarse-grid artifact; its a = 1.60 endpoint moves 0.25077 → 0.22402). | T60 | §5 |
| 22 | measured crossings track the upper spinodal (3 of 3 prospective) | **Withdrawn as meaningless** — there is no upper spinodal distinct from the global switch. Replaced by: **measured crossings sit above `R_glob` by 1.087–1.144 (mean 1.115, sd 0.020), one-signed at every `a`**, which is relaxation lag. Drift: measured **+10.0%**, `R_glob` **+4.5%**, **r = +0.9965**. | T60 | §5 |
| 23 | *(none — new)* | **Causal test of the threshold.** Same run, trained to placement, then `\|w₂\|` pinned to one side of `R_glob` and held: placement kept **0/37** below (0.85 R_glob) and **33/37** above (1.15 R_glob), **Fisher p = 1.2e-16**; a ×(1+1e-6) noise-floor arm keeps 37/37. Placement is lost **within 50 steps** in all 37. | T59 | §6 |
| 24 | *(none — new)* | **The second threshold confirmed causally too.** `hold_high`'s held `R = 0.24663` lies **strictly between** `R_glob = 0.21446` and `R_solve = 0.30667`, and gives **33/37 placed, 0/40 solved** — the regime the decomposition predicts. Verified against the landscape: the conditional minimiser at exactly that `\|w₂\| = 5.750` has gap **+0.022543** and **does not solve**, so solving is *impossible* there, not merely unattained. | T59 | §6 |
| 25 | *(none — new)* | **`R > R_glob` is necessary for placement to persist, NOT sufficient for it to be found.** Cold starts above the threshold place **2/40** (0/9 at 1.50 R_glob over 20,000 steps); pre-supplying `\|w₂\|` and releasing gives **9/40 against control's 37/40, Fisher p = 1.3e-10**. Growth and placement are coupled and ordered. | T59 | §6 |
| 26 | *(none — new)* | **Why the jump arm fails: saturation and conditioning, not a wrong basin.** Gradients fall to **0.07% of control's**; a 5× budget recovers placement **2/15 → 7/15**; the endpoint is a near-constant-predictor region (`w₁ ≈ −0.003`, loss ≈ log 2) with Hessian condition number above 10⁷ (eigenvalues ~1e-6 against ~25–50). **Not a stationary point** — `w₁ = 0` has gradient 0.18–0.36 because the sampled classes' first moments differ. | pending | §6 |
| 27 | *(none — new)* | **The machinery transfers to four unseen task windows.** Across five windows at fixed `a`, `\|w₂\|` at crossing spans **2.34–13.65 (5.8×)** while `CV(R) < CV(\|w₂\|)/2` at both `a` (**3.36× and 3.01×**). Each window's `κ₀`, computed from `h(σ)` alone, matches its measured `κ(1.02)` within **0.81%, 5 of 5**, across κ₀ values spanning **8.1×**. Nothing was refitted per window. | T61 | §7 |
| 28 | *(none — new)* | **The per-`a` lag is a property of `a`, not of the task.** Measured/predicted ratios have sd **0.014** at a = 1.30 and **0.016** at a = 1.50 across five windows; normalised by the base task's own offset at the same `a`, all ten land within **2.72%** of 1.0. | T61 | §7 |

## Claims to REMOVE (second session)

| # | claim to remove | reason |
|---|---|---|
| R4 | `R_spin`, the upper spinodal, and any number derived from it (drift +16.9%, "crossings track `R_spin`", the 3-of-3 prospective result) | **`R_spin = R_glob` at all six `a`** on a 0.01 grid; the apparent windows were the 0.05 grid. D-1 (`blockF_lag_prediction.md`) becomes **vacuous** — its two models are identical. |
| R5 | any suggestion that reaching `R > R_glob` *produces* sign-correct placement | Falsified by intervention: cold starts above the threshold place 2/40, jump-started runs 9/40 vs control 37/40. The threshold governs **stability**, not discovery. |
| R7 | **the family B AUC comparison** (`R` vs `\|w₂\|` vs `Ĝ` as separators) | **Removed entirely, not caveated.** Within fixed α, `R_B` is `\|w₁w₂\|` times a constant so their AUCs are identical by construction; and 4 of 5 α cells are perfectly separable (α = −1: largest unsolved `\|w₁w₂\|` **0.295** vs smallest solved **7.26**, **2 runs between**), so every monotone variable scores 1.0000. T54's own criterion required ≥ 20 band runs. The box-limited `Ĝ` was a *second* defect, not the only one. |
| R6 | the registered `G3_far_outer ≥ base` sub-claim | Backwards by a provable inequality: `O' ⊃ O` ⟹ `Ĝ(O',I) ≤ Ĝ(O,I)`. Corrected before any Block G crossing was measured. |

## Registered predictions that FAILED, with their explanations

| prediction | outcome |
|---|---|
| P1-b (median solved ρ in [0.5,0.95]) | median **0.9674**; the inference from the theorem's slack was invalid |
| `R·ρ` governs the second stage | `R` has CV 0.0830 vs `R·ρ` 0.1933 |
| precision effect | retracted — an RNG artifact of per-dtype `uniform_()` |
| S-3 for `R_solve` (positive, increasing in ε) | all six **negative**, mean −0.29% — `R_solve` is *already at* its limit value, so a constant sequence cannot show the predicted approach |
| Block E `cold_high` ("placement achieved") | **2/40** |
| Block E `jump` ("places well before control") | **falsified in the opposite direction**, 9/40 vs 37/40, p = 1.3e-10 |
| G-1 (crossing within 15% of `R_glob`, all windows) | **6/10** on the letter; the four misses are the per-`a` lag, which varies **under 3%** between windows |
| G-3 sub-claim on `G3_far_outer` | direction backwards by a provable inequality; corrected before measuring |
| Block E stall: registered expectation **trapping** | **saturation** — gradients 0.07% of control, and a 5× budget recovers 2/15 → 7/15 |
| K-2's stated range "roughly 115–145×" | **arithmetically wrong in my own registration** — I omitted the 2,000-budget row at 11.8×; the correct range is **11.8×–145×**. The substantive predictions (exactly 8×, floor above 10×) both hold. |
| K-3 (AUC margin ≥ 0.10 over `\|w₂\|`) | **0.0089** — and the claim is removed rather than reported, per the standing instruction |
| Block H H-3 (validity gate) | **FAILED**, 42.5 pp, p = 4.3e-5 — a linear schedule is not a null because the natural `\|w₂\|` trajectory is non-monotone |

## Framings considered and NOT adopted

- **Loss-landscape bifurcation with hysteresis** — dropped on its own registered falsifier at a = 1.30 and now excluded at all six `a` (claim 21).
- **Output scale as an annealing parameter** — the registration made it conditional on the stall being **trapping**; it is **saturation**, and Block H's validity gate failed independently. Not adopted.

## Open / pending

- **Block F** (three-optimiser crossing `R`, F-1/F-2/F-3) — the remaining
  registered test. D-1 is now **vacuous** (T60) and is not run.
- **Block H, valid version**: a control-matched arm must replay each seed's own
  `|w₂|(t)` time-warped by `m`, so `m = 1` is the natural trajectory exactly.
  Not run; recorded as the correct design in `blockH_rate_results.md`.
- **Family B** (`Ĝ`-derived claims) — still suspended pending Block K.
- Claim 26 (saturation/conditioning of the stall) has no ledger ID yet.
