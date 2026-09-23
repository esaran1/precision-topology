# Registration: sample-size test of the own-seed account

**Written before any training set with n ≠ 400 has been generated, trained on or searched.** Date: 2026-09-23 15:29 EDT.
Producer: `src/sample_size.py`, committed with this file.

**Timing check disclosed**: n = 400, a = 1.30, seed 0 (already seen). It reproduces phase 2b's seed-0
crossing exactly (step 4,016, R = 0.211725). The own-seed bracket [4.7375, 4.74375] overlaps 1d's certified
[4.738, 4.742].

## Design

- **Cells**: a ∈ {1.30, 1.50} × n ∈ {400, 1,600, 6,400} total training points
  (`fold1d.make_data(n/2, seed)`).
- **Seeds**: fresh seeds 300,000–300,049, used in every cell. Within a cell, both arms use the same
  training set.
- **Population threshold, the same at every n**: the certified quadrature bracket midpoint
  (`cond_certified_brackets.csv`, kind glob). R_pop = w₂,pop·Ĝ_cert(a)/2, with the certified Ĝ
  (`ghat_certified_all.csv`). No quantity in it depends on n.
- **Own-seed arm (50 seeds per cell)**: the own-threshold method with grid step 0.05 and 16 local
  refinements, identical at every n. It brackets from w₂,pop in steps of 0.1 and bisects to 0.01 in |w₂|.
  own/pop = w₂,own/w₂,pop.
- **Free-training arm**: the phase 2b protocol.
  - float64, full batch, Adam lr 1e−2, torch.manual_seed(seed) then U(−1, 1)⁴, budget 32,000.
  - Crossing is the first step at which the dense-grid oriented gap is > 0. R_cross = |w₂|·Ĝ_cert/2
    there.
  - Seeds: the same 50. If fewer than 40 cross in a cell, 20 more seeds are added at a time
    (free-training arm only), up to 110.
- **Quantities per cell**:
  - own excess = median(own/pop) − 1;
  - free offset = median(R_cross/R_pop) − 1.
- **Uncertainty**: bootstrap 95% intervals for differences of medians (10,000 resamples, seed 0).

## Registered predictions (per a; scored separately at a = 1.30 and 1.50)

- **N1**: the own excess decreases as n grows.
  - excess(400) > excess(1,600) > excess(6,400) (point estimates), **and**
  - the bootstrap interval of excess(400) − excess(6,400) lies above 0.
- **N2**: the free-training offset decreases correspondingly, with the same two conditions on the
  offsets.
- **N3**: at each n, the own excess is at least half the free offset: excess(n) ≥ offset(n)/2. It passes
  at a given a iff this holds at all three n. Per-cell verdicts are reported.
- **Competing**: the offset is not a finite-sample effect. That is, the bootstrap interval of
  offset(400) − offset(6,400) contains 0 **and** offset(6,400) ≥ offset(400)/2. Reported alongside N2.

## Validation (reported with the scores)

1. **At n = 400, a = 1.30**: the method on 1d's 50 seeds (0–49) against 1d's certified brackets. Reported:
   the number of overlapping brackets and the largest midpoint difference.
2. **Certified branch and bound** (`conditional_certified.evaluate`) at both bracket ends, on seeds
   300,000 and 300,001 in every cell.
- **Stop and report** if any certified status contradicts a bracket end, or if fewer than 45 of 50
  brackets overlap in (1).

## Order

This test runs after the registered c₁ test (`first_order_prediction.md`). The sequence is: `validate400`,
`own`, `free`, `certify`, then `score`.

## Reporting addendum (2026-09-23 15:32 EDT; before any run; no verdict changes)

Alongside the registered verdicts, the scorer reports:
- the value in every cell with its bootstrap 95% interval;
- every pairwise difference (400 − 1,600, 400 − 6,400, 1,600 − 6,400) with its interval.
  (`sample_size_detail_cells.csv`, `sample_size_detail_pairs.csv`)

If a strict-ordering step fails while both values' intervals contain 0, the registered failure is recorded
as a failure. Separately, the offsets are stated to be **indistinguishable at those sizes**. That outcome
differs from an offset that does not shrink (the competing outcome).
