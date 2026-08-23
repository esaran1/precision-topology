# Dense-verified separation rate as a function of corrugation strength

New analysis prompted by the Part 2c pattern that corrugated dense survivors
span only mild corrugation. Data: `corrugation_sweep.parquet` joined with
`dense_check.csv`; GELU, width 3, 30 runs per configuration.

## The table, ordered by severity

| Configuration | reading | amplitude | frequency | noise | sample-level sep | **dense-verified** |
|---|---|---:|---:|---:|---:|---:|
| flat | — | 0 | 0 | 0 | 2 | **1** |
| A_embedded_a0.001 | A | 0.001 | 100 | 0 | 1 | 0 |
| A_a0.05 | A | 0.05 | 100 | 0 | 3 | **1** |
| B_a0.05 | B | 0.05 | 100 | 0 | 2 | 0 |
| A_a0.15 | A | 0.15 | 100 | 0 | 3 | 0 |
| B_a0.15 | B | 0.15 | 100 | 0 | 2 | 0 |
| A_embedded_f0.5 | A | 0.30 | 0.5 | 0 | 2 | **2** |
| A_f10 / B_f10 | A/B | 0.30 | 10 | 0 | 1 / 2 | 0 / 0 |
| A_f50 | A | 0.30 | 50 | 0 | 2 | **1** |
| B_f50 | B | 0.30 | 50 | 0 | 1 | 0 |
| A_paper / B_paper | A/B | 0.30 | 100 | 0 | 2 / 2 | 0 / 0 |
| A_n0.005 / A_n0.02 | A | 0.30 | 100 | >0 | 2 / 2 | 0 / 0 |
| A_f200 | A | 0.30 | 200 | 0 | 1 | 0 |
| A_a0.5 / B_a0.5 | A/B | 0.50 | 100 | 0 | 2 / 2 | 0 / 0 |

Along the amplitude axis at frequency 100: dense-verified separations occur
at amplitude ≤ 0.05 (2 of 11 sample-level separations) and **never at
amplitude ≥ 0.15** (0 of 15). Along the frequency axis at the published
amplitude 0.3: survivors at frequency ≤ 50 (3, concentrated at 0.5), none
at ≥ 100. **No noise configuration and no Reading-B configuration has any
dense survivor** — Reading B is 0 for 11 sample-level separations across
all nine of its configurations, including its mildest.

## Is it systematic or attrition?

Two ways to ask it, answered separately:

**Conditional on separating, do survivors concentrate in mild
configurations?** Splitting at the survivors' observed boundary (amplitude
≤ 0.05, or frequency ≤ 50, or the embedded arms, no noise): 5 of 16
mild-group separations survive against 0 of 18 strong-group.
Hypergeometric probability that all 5 survivors land in the mild group by
chance: **p = 0.016**. The split was drawn at the observed boundary, so
this p-value is post-hoc and should be read as descriptive strength, not a
registered test — but the monotone decline along the amplitude axis was
not chosen, and points the same way.

**Is the strong-group regional rate distinguishable from zero-vs-mild?**
Pooling the strong group: 0 dense-verified separations in 360 runs bounds
its regional-separation rate below **0.83%** (one-sided exact 95%). The
mild group's observed rate is 5/270 = **1.9%** (exact 95% interval
0.6–4.3%). The strong-group bound sits below the mild point estimate but
inside the mild interval, so **the data are consistent with regional
separation going to zero above mild corrugation, and also cannot exclude a
small surviving rate (~0.6–0.8%) that our n is underpowered to detect.**
Per-configuration (n = 30) the zero bounds are 9.5% each — far too weak
alone; the pooled bound is the informative one.

## What this displaces and what it opens

If the threshold reading is right, **corrugation above a modest strength
makes the task regionally unsolvable at width 3 for GELU within this
protocol** — a stronger and more interesting statement than the fold-layer
negative it displaces, because it is about the task rather than about the
optimizer's fold placement. The registered corrugation Prediction 1 asked
whether corrugation moves the fold later; the answer now available is that
above mild corrugation there is no dense-verified fold to place. Deciding
between "unsolvable" and "rate below our power" needs either many more
seeds in the strong group (a 0.5% rate needs ~600 runs per condition for
expected count 3) or wider networks, where rates are higher and a
corrugation-strength gradient would be measurable with existing budgets.

The Reading-B zero (0 dense survivors in 11 separations, all nine
configurations) was not predicted and is left as an observation; Reading B
modulates sampled radius rather than displacing the core, and why that
should be uniformly harder regionally is not explained by anything above.
One Reading-B failure (B_f10 d12 s4, 1 error) is a flip case that passes
0/0 on two extra samples; counts here follow the primary protocol, and the
Reading-B zero would become 1/11 under a best-of-three convention.

---

> **Reading-B correction (2026-08-23).** The Reading-B anomaly flagged
> above is resolved — and removed from the corrugation-strength story —
> in `reading_b_anomaly.md`: Reading B's amplitude parameter is ignored
> (its four zero-noise f=100 labels are one condition, so B has no mild
> arm), and its sampler concentrates ~20% of points exactly on the tube
> surface, making dense verification a surface probe (83% of one failed
> run's dense errors lie exactly on the surface, vs 20% base rate).
> Deduplicated, Reading B has 5 distinct separations, 0 dense survivors.
> The strength analysis above should be read on Reading A + flat, where
> the amplitude axis is real.
