# Registration: does the conditional threshold predict SGD crossings? (Track 4)

**Written and committed before any registered run.** Date: 2026-09-25. Code: `src/sgd_own.py`. Tests:
`tests/test_sgd_own.py`, which exercise the budget rule and both scorers on constructed pass, fail and unresolved
cases, and check that the initialisation is identical to the Adam runs'.

**Seen before this registration:**
- the Adam own-sample results (`own_threshold_scores.csv`: Spearman 0.877 at a = 1.30 and 0.881 at a = 1.50);
- Block F's SGD result at a = 1.25 (43/60 crossed in 20,000 steps; offset 8.6%);
- the budget pilot below, which records |w₂| only.

No SGD crossing on seeds 0–39 has been computed.

## Why this test

The conditional threshold is a property of the loss and the training set, not of the optimiser. Own-sample global
thresholds already exist for the phase-2b training sets (`own_threshold_crossing.csv`: `fold1d.make_data(200, seed)`,
seeds 0–39, a = 1.30 and 1.50, base window, n = 400; all 80 present). Training SGD on exactly those sets tests the
prediction across optimisers with no new threshold computation.

## Protocol

- SGD, lr 0.3 (`blockF_optimisers.SGD_LR`, the only rate that solves in this setting), full batch, float64, on
  `fold1d.make_data(200, seed)`.
- **The same initialisation as the Adam runs of each seed**: `torch.manual_seed(seed)`, then U(−1, 1)⁴ drawn in
  float32 and cast, exactly as `phase2b_ordering.run`. This is tested.
- **Every-step crossing detection** with the Adam test's rule, `phase2b_ordering.state`: the first step at which the
  gap in w₂'s orientation is > 0 on the dense 4,001-point windows. The crossing |w₂| is recorded.
- **Budget**, from a pilot that records only |w₂| growth, on calibration seeds 700,000–700,019 (disjoint from
  0–39), with no placement evaluated:
  - B = the smallest of {16k, 32k, 64k, 128k} at which at least 95% of calibration runs have |w₂| ≥ 1.5 × the
    population |w₂|_glob, at every a; otherwise 128k.
  - The chosen B is written to `sgd_own_budget.json` and committed before any registered run.

## Predictions (scored separately at a = 1.30 and a = 1.50, on the runs that cross)

- **G1 (primary).** Each run's own global threshold predicts its SGD crossing better than the population threshold.
  - Per-run absolute log errors are e_own = |log(w₂,cross/w₂,own)| and e_pop = |log(w₂,cross/w₂,pop)|.
  - PASS iff the paired run-level bootstrap 95% interval (10,000 resamples) of mean(e_own − e_pop) lies entirely
    below 0.
  - This mirrors the Adam own-sample test.
- **G2 (primary).** Spearman(w₂,cross, w₂,own) ≥ 0.6.
- **G3 (descriptive, no criterion).** The median SGD residual w₂,cross/w₂,own − 1, and against the population.
  Any residual against the branch threshold at crossing will be labelled post hoc.
- **UNRESOLVED** at an a with fewer than 30 crossings. The crossing fraction is reported.

**Competing outcome.** Under SGD the crossing is governed by optimiser dynamics, not by the sample's threshold. Then
G1 fails, since the own threshold is no better than the population's, and G2 fails.

**Extension (Track 3).** A prediction of the median SGD residual from the Adam-fitted timescale relationship may be
added **only** if Track 3's post hoc analysis finds Spearman ≥ 0.6 across both a and the learning-rate arms. It must
be committed as an amendment before the SGD runs are scored. If not, it is not registered.

## Reporting

Runs go to `results/sgd_own_runs.csv` and scores to `results/sgd_own_scores.csv` (`python -m src.sgd_own score`).
Nothing is re-run to change a verdict. The cutoff is 01:00 EDT: whatever is incomplete then is reported as not done.

## Amendment 1 (2026-09-25 13:13 EDT): the extension is registered, before any SGD run is scored

**State at registration:**
- All 80 SGD runs have finished (`sgd_own_runs.csv`). They have **not been scored or inspected**: no crossing |w₂|,
  residual or crossing count has been read.
- Track 3's post hoc analysis is complete (`residual_timescale_summary.json`). The run-level Spearman between
  residual and timescale ratio, pooled over both a and the four learning-rate arms (n = 384), is **0.634**, with
  bootstrap 95% interval [0.542, 0.721]. That meets the author's rule (≥ 0.6), so this extension is registered.
- Caveat, stated now: within each a the correlation is weaker (0.454 at a = 1.30, 0.507 at a = 1.50). Much of the
  pooled correlation is the difference between the two a values.

**EXT (prospective, on a different optimiser).**
- **Fitted relationship (Adam, post hoc, frozen now in `residual_timescale_fit.json`).** Residual = α + β·ratio,
  fitted by OLS over the 384 Adam runs: α = 0.0157178, β = 2.65842. The fitted ratio range is [0.00137, 0.0229].
- **Each SGD crossing run's own ratio at its crossing:**
  - replay the run deterministically to its recorded crossing step (the replay must reproduce the crossing |w₂|
    exactly);
  - growth = d log|w₂|/dt over the last min(100, step − 1) steps;
  - relax = 0.3·λ_min(H), where H is the Hessian of the joint loss in (w₁, b₁, b₂), w₂ fixed, at the branch
    (damped Newton from the crossing state). SGD's preconditioner is the identity;
  - ratio = growth/relax (`sgd_own.ratio_at_crossing`).
- **Prediction at each a:** the median over crossing runs of α + β·ratioᵢ.
- **Observed:** the median SGD residual against the own global threshold, w₂,cross/w₂,own − 1.
- **PASS** iff |observed − predicted| ≤ max(0.01, 0.25·|predicted|).
- UNRESOLVED with fewer than 30 runs whose ratio is finite and positive (a non-positive relaxation rate is unusable).
- Tests: `tests/test_sgd_own.py::test_extension_scorer_cases`, on constructed pass, fail and unresolved cases.
- The SGD ratios may lie outside the fitted Adam range. The prediction is then an extrapolation, which is stated and
  not a reason to exempt it.
- **Competing outcome:** the SGD residual is not governed by this timescale ratio, and the prediction misses the
  tolerance.
