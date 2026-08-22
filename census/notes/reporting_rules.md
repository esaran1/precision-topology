# Standing rules for reporting minima, gaps, and comparisons

Three artifacts of the same family have appeared in this project, each caught
only after being stated as a result:

1. **The gap/monotonic-minimum identity.** The gap is
   `monotonic_min − GELU_min`, and GELU's minimum is 0 wherever it separates
   exactly, so the gap *is* the monotonic minimum. Reporting them as two
   quantities invited reading an identity as a relationship.
2. **The r = 0.9997 correlation.** The gap was correlated against the monotonic
   minimum across configurations. Same identity, so the correlation was a
   quantity against itself. Against proxies that do not enter the definition the
   relationship was +0.45 and +0.21 on six points.
3. **The tanh minimum across protocols.** 26 against 42 looked like a protocol
   effect. The full distribution moved the other way — theirs better at every
   percentile — and the 26 was one run, 23 points clear of the next-best, with a
   resampled probability of 0.0000.

All three came from summarising with a minimum or a gap over modest `n`. These
are extreme-value statistics: they move with `n`, with luck, and with any
identity hiding in their definition.

## The rules

**Rule 1. Any minimum or gap reported anywhere must appear with its
distribution.** Median and quantiles at minimum; a full histogram where the
shape matters. A minimum quoted alone is not a reportable summary.

**Rule 2. Any comparison of two minima across conditions must be
resampling-checked before it is stated as an effect.** Different `n` between
conditions makes the comparison invalid on its face; equal `n` still leaves
luck. The check is: resample the larger condition down to the smaller, many
times, and ask how often the observed difference arises by chance.

**Rule 3. Before correlating two derived quantities, check whether one is
defined in terms of the other.** If it is, the correlation is an identity and is
not evidence. Correlate against a proxy that does not enter the definition.

**Rule 4. Prefer counts of exactly zero over minima where available.** "No
monotonic run separated in 600 runs" is not an extreme-value statistic — it is a
count over the whole population, and it does not move with `n` in the way a
minimum does. Claims of this form are the ones to lean on.

## Applying rule 4: what this project should lean on

The load-bearing claim is **the width-3 monotonic zero**: 0 separations across
1,440 runs under our protocol and 360 under theirs, spanning twelve
parametrizations, four depths, three monotonic activations, and 20–30 seeds per
cell.

It is a count, not an extreme. Increasing `n` can only overturn it by producing
a separation, which is the honest way for it to fail. Every other quantitative
claim in this project is subordinate to it and should be reported as such.
