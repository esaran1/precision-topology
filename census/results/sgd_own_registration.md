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
