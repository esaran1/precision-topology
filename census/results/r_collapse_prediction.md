# Registration: is R the controlling variable?

**Written before any R-collapse quantity was computed.** The only R numbers in
hand at registration time are Item 5's (`mechanism_checks.md`): pooled over
Adam runs at 6 values of `a`, `R < 0.30` gave 0/191 and `R > 0.50` gave 91/91.
Nothing below was adjusted after seeing a stratified fit.

Date: 2026-09-12.

---

## The quantity

For a run with terminal output weight `|w2|` on activation `f_a`,

    R = |w2| * G*(a) / 2

where `G*(a) = max_{w1,b1} G(w1,b1)` is the maximum achievable class gap. By
the theorem, a solution with logit margin `m` requires `|w2| >= 2m/G*`, i.e.
**`R >= m`**: R is the largest margin the run's weight scale could support
under optimal placement.

## Hypothesis

**P(solve | R) is a single curve.** Conditioning on `a`, budget, activation
family, or optimizer does not shift it materially.

## "Materially", fixed numerically now

Let `R50` be the R value at which a fitted curve crosses 50%, and let `W` be
the pooled transition width, defined as `R(75%) - R(25%)` on the pooled fit.

> **A stratum shifts materially if its `R50` differs from the pooled `R50` by
> more than `0.5 * W`.**

Half the transition width is the natural unit: a shift that size moves a
stratum's median from the pooled centre to the pooled quartile, which is
visible in any plot and is not attributable to bin placement.

Secondary criterion, reported alongside: the **RMS residual of stratified fits
against the pooled fit**, compared against binomial sampling noise at the cell
counts. Collapse is claimed only if that ratio is **below 2**, the same
standard applied (and failed) by the `u`-collapse in `collapse_results.md`.

## Failure conditions — every one is checked and reported

1. `R50` shifts systematically with **`a`** beyond `0.5 * W`.
2. `R50` shifts systematically with **budget** beyond `0.5 * W`.
3. `R50` differs between **optimizers** beyond `0.5 * W`.
4. `R50` differs across **activation families** beyond `0.5 * W`.

**Any one of these means R is a correlate rather than the controlling
variable**, and the paper keeps its current structure. That outcome is
reported immediately and the remaining parts are not run.

## The optimizer test (1d), registered specifically

Adam and SGD reach different terminal `|w2|` at the same budget — measured
medians at `a = 1.25` differ by up to 5x at 40k steps. The hypothesis predicts
they nonetheless lie on **the same P(solve | R) curve**, with the optimizer
entering only through *where on the R axis its runs land*.

- **If it holds**: the Adam-only scope limitation **dissolves for the R
  claim** — the optimizer sets the distribution of R, and R sets solvability.
- **If it fails**: R is Adam-specific, the scope limitation stands unchanged,
  and we say so.

This is the test that changes the scope claim, and it is registered as
decisive either way.

## Part 2 checks, registered

- **2b.** R uses `G*`, the maximum over placements, so it upper-bounds the
  achieved margin. We compute the **achieved** `G` for each run and test
  whether `R_achieved = |w2| * G / 2` collapses *tighter* than R. If it does
  substantially (residual ratio improving by more than 25%), then R as defined
  is a proxy and the achieved quantity is the right variable. Reported either
  way.
- **2c.** The theorem gives `R >= m`. An empirical threshold near `R ~ 0.4`
  would assert that solving requires achievable margin above roughly 0.4. We
  check this against the measured margin distribution (0.014-1.97, median
  0.68) and report whether the threshold has a margin interpretation or is a
  bare empirical cut.

## Part 3, registered

If R controls solvability, the onset at each budget is **predictable without
fitting the onset**: from the terminal `|w2|` distribution at that budget,
`G*(a)`, and the R threshold. Predicted onsets are compared against measured
onsets. **If the prediction works, the budget law is a consequence of the R
threshold rather than an independently fitted law**, and the exponent drift
becomes a statement about `|w2|` growth.

Registered in advance: this is a *prediction*, not a fit. No parameter is
tuned to the onsets.

## Coverage, to be reported

The pooled claim's strength depends on how many runs carry a usable terminal
`|w2|`. We report recoverable versus lost counts before any collapse number.
