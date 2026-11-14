# Figure: scope

**Message.** Every registered causal pass outside the width-one sine network is in the slow-growth or branch-start regime; in standard training the lag law's regime is rare at width two and the output-layer lever has no effect in a standard MLP.

**Population and n.** (a) Registered strictly causal forecasts of the crossing step: fraction of scored runs within the registered tolerance at the registered cutoff 0.95 (filled; 0.95 was chosen after the pilot) and at 0.90 (open; scored descriptively, not computed for the width-4 and tanh tests). n = 79, 101, 120, 112, 119, 22 (top to bottom; the tanh row counts 22 rates, each a deterministic run, and its tolerance is 15% of the forecast delay). Dotted: the registered 80% bar. (b) Standard training, descriptive: fraction of crossings with predicted lag |κχ| ≤ 0.1, n = 408, 207, 114, 705, 75. (c) Registered test in a ReLU MLP (6144-256-256-1) on MNIST-CIFAR dominoes, predicted to show no effect: median paired change, output-layer learning rate ÷16 against the baseline arm, at three matched training losses, n = 40 seeds.

**Uncertainty.** (a, b) Clopper–Pearson 95% intervals. (c) 95% percentile bootstrap intervals of the median paired difference (10,000 resamples, the registered procedure).
