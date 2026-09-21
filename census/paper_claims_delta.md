# Claims delta for the writing side

One row per claim that changed. **Old claim** is what the current draft says or
what an earlier note said; **new claim** is what the artifacts support now.
Every new claim has a ledger ID and passes `verify_ledger.py` unless marked
pending.

Updated during the unattended session of 2026-09-21. Append-only.

---

## Corrections carried in from earlier work

| # | old claim | new claim | ledger | section |
|---|---|---|---|---|
| 1 | "200 training points" | **400 training points** — `N_PER_CLASS = 200` is *per class*, and both classes are used every step. Full batch, no minibatching. | 0b | setup |
| 2 | success = 0 errors on the evaluation sample | **success = `solves()`, the dense regional check** on 4,001 inner and 4,000 outer points. The sample clause never binds on its own: 0 runs are solved-with-errors, while **4 runs have 0 sample errors and fail the dense check**. | 0b, T57 | setup |
| 3 | "float32 and float64 differ materially; `a = 1.5` seed 0 solves in one and not the other" | **float32 and float64 agree run-for-run given a shared initialisation** — 180 paired runs, 0 flips, relative parameter distance 0.0000. The apparent effect was an RNG artifact: `torch.empty(4, dtype=d).uniform_()` consumes the stream differently per dtype, so the two arms were **different networks**, not the same network computed twice. | 0b | reproducibility |
| 4 | SGD results reported without qualification | **SGD learning rate 0.3 is the only one that solves at all** (recorded in `sgd_onsets.LR`); the SGD arm is not a generic-SGD claim. | 0b | optimisers |
| 5 | family B: `β = 0.9935` measured, 900–1,200× overshoot, AUC comparison | **All `Ĝ`-derived family B claims SUSPENDED** pending Block K. `f(t) = max(t, αt)` is positively homogeneous, so `G(c·w1, c·b1) = c·G(w1,b1)` and the unrestricted supremum is **infinite**; `Ĝ` scales exactly with the search box (0.2/0.4/0.8/1.6/3.2 for `w1 ≤ 1/2/4/8/16` at `α = −0.5`), and the committed values are exactly `3.2\|α\|`. **Not suspended**: the onset exponent **+0.25**, which comes from solve rates and never used `Ĝ`. | T28, T54 | family B |
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

## Open / pending

- Block B switch points (`R_glob`, `R_spin`, `R_fold`, `R_solve`) and whether
  measured crossings track the upper spinodal — **decides the paper's mechanism
  claim**.
- Block D: `R` vs `R·ρ` at the solve step. **Interim result contradicts the
  registered prediction** — `R` has CV 0.0775 against `R·ρ` at 0.1935, so `R`
  governs the second stage, not `R·ρ`. Logged as a failed registered prediction.
- Blocks E, G: intervention and out-of-distribution windows.
