# Registration: optimizer equivalence of the R₅₀ threshold (TOST)

**Written before any run of this block.** Date: 2026-09-22.

Theory review item 12: overlapping intervals and null Fisher tests show that no
difference was *detected*, not that the optimizers are *equivalent*. This block
tests equivalence directly with two one-sided tests at a registered margin.

## Margin

**δ = 0.024** in `R` for every pairwise difference `R₅₀(o₁) − R₅₀(o₂)`.

Justification: the whole solve transition spans a band of width **0.048** in
`R` (T52). A shift of half that width is the largest that leaves the midpoint
of one optimizer's curve inside the central half of the other's. `δ = 0.024`
also coincides with the materiality threshold `0.5·W` already used for the T52
stratum tests (0.0241 before the certified `Ĝ`, 0.0224 after), so equivalence
here means "below the threshold this paper already treats as immaterial".

## Setting, identical to the existing three-optimizer comparison

1D fold task at `a = 1.25`, `make_data(200, seed)` (400 points, full batch),
`θ ~ U(−1,1)⁴` drawn once in float32 (`torch.manual_seed(seed)`), float32
training:

| arm | optimizer |
|---|---|
| Adam | `torch.optim.Adam`, lr 1e-2 (`src/budget_law.py::run_full`) |
| AdamW | `torch.optim.AdamW`, lr 1e-2, weight decay 0.01 |
| SGD | `torch.optim.SGD`, lr 0.3 (the only rate that solves) |

Success is `solves()` (dense 4,001-point regional check). `R = |w₂|·Ĝ(1.25)/2`
with the **certified** `Ĝ(1.25) = 0.0665056` (T65). "In band" means
`0.30 ≤ R ≤ 0.50`.

## Estimator and test

- **R₅₀**: maximum-likelihood two-parameter logistic `P(solve) = sigmoid(b₀ + b₁R)`
  fitted to individual runs, `R₅₀ = −b₀/b₁`.
- **Cells** are `(optimizer, budget)`. **Cell-level bootstrap**, 4,000
  resamples: within each optimizer, resample its budget cells with replacement,
  refit, and take the pairwise difference.
- **Verdict per pair**, from the **90%** percentile interval of the difference
  (TOST at α = 0.05 each side):
  - **equivalent** if the interval lies inside `(−δ, +δ)`;
  - **not equivalent** if the interval lies entirely outside `[−δ, +δ]`;
  - **inconclusive** otherwise.
- Secondary, reported but not a verdict: the 95% interval, and whether it
  excludes 0.

**Only the new runs of this block enter the verdicts.** The existing 600 runs
used untargeted budgets, where most cells sit far outside the band and cell
resampling mostly measures where the budgets happened to land. They are reported
alongside for context.

## Sample size

Calibration from the existing data, computed before registering: the
cell-bootstrap SE of R₅₀ scales as `c/√n_band`, with `c = 0.131` for AdamW, the
only existing arm with a band-targeted design (57 in-band runs). Adam and SGD give
0.218 and 0.043 on untargeted designs; the targeted value is used for planning.

TOST at α = 0.05 with 80% power when the true difference is 0 needs
`SE_diff ≤ δ/(z₀.₉₅ + z₀.₉₀) = 0.024/2.927 = 0.0082`, so a per-arm
`SE ≤ 0.0058`, so **`n_band ≥ (0.131/0.0058)² ≈ 510` in-band runs per
optimizer**. The target is **≥ 500 in-band runs per optimizer**.

## Design: placement pilot, then targeted cells

1. **Pilot** (the method that gave AdamW 57 in-band runs): 8 seeds (900–907) at
   each candidate budget. It measures **where runs land in `R`, not whether they
   solve**. Candidates: Adam and AdamW 3k–12k in 1k steps; SGD 8k–48k in 4k steps.
2. **Select** every candidate budget whose pilot median `R` lies in `[0.30, 0.50]`.
   If fewer than 6 qualify, add midpoints between qualifying and adjacent budgets
   until 6 do, or report that it could not be done.
3. **Main runs**: at up to 12 selected budgets per optimizer, seeds **100 onward**,
   disjoint from the pilot and from every existing run. Seeds are added in blocks of
   60 per cell until each optimizer has ≥ 500 in-band runs.
4. Pilot runs are reported and **excluded** from the verdicts.

## Registered expectations

From the existing point estimates (Adam 0.3364, AdamW 0.3479, SGD 0.3723 under
the MLE and certified `Ĝ`):

- **Adam vs AdamW: equivalent.**
- **Adam vs SGD** and **AdamW vs SGD: not equivalent or inconclusive.** Their
  existing differences, 0.036 and 0.024, are at or beyond `δ`.

These are expectations; the verdict rule above decides.
