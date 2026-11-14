# Figure: lag causal

**Message.** Strictly causal per-run forecasts of the lag track the observed lag within the registered ratio bands in the two registered tests; in the post hoc re-score at a = 1.65 the observed lag exceeds the forecast by a median 22% (SGD) and 16% (Adam).

**Population and n.** Width-one sine network. Every forecast reads the run only before its cutoff, the first step where the output scale reaches 0.90 of the run's causally estimated switch scale (f = 0.90). (a) SGD: registered test at a = 1.58 (n = 84 scored, 0 without a forecast); post hoc re-score at a = 1.65 (n = 63, 0 without a forecast). (b) Adam: registered test at a = 1.77 (n = 98, 3 without a forecast, not plotted); post hoc re-score at a = 1.65 (n = 74, 8 without a forecast, not plotted). Lag: relative excess of the crossing scale over the switch scale. Dashed: identity. 1 run(s) off the axes, drawn as a triangle at the edge (a = 1.77 Adam: forecast -0.53, observed 0.10).

**Uncertainty.** Each point is one run; no interval is drawn. Median observed/forecast ratio: 1.169 (a = 1.58), 1.216 (a = 1.65 SGD), 0.940 (a = 1.77), 1.158 (a = 1.65 Adam).
