# Width-4 correction and training-protocol comparison

## The width-4 claim was overstated

The author reports that ReLU could not be reliably trained to perfect accuracy
at width 4. **Our own data agrees with him.** The error was in our wording, not
in the measurement.

Perfect-run counts on held-out evaluation accuracy, n = 80 per cell (4 depths ×
20 seeds):

| Width | Activation | Perfect | % perfect | At chance | % at chance | Median | Max |
|---:|---|---:|---:|---:|---:|---:|---:|
| **3** | GELU | 6 | **7.50** | 0 | 0.00 | 0.9615 | 1.0000 |
| 3 | tanh | 0 | 0.00 | 0 | 0.00 | 0.9608 | 0.9870 |
| 3 | leaky-ReLU | 0 | 0.00 | 8 | 10.00 | 0.9585 | 0.9700 |
| 3 | ReLU | 0 | 0.00 | 49 | 61.25 | 0.5000 | 0.9735 |
| **4** | GELU | 54 | **67.50** | 0 | 0.00 | 1.0000 | 1.0000 |
| 4 | tanh | 41 | **51.25** | 0 | 0.00 | 1.0000 | 1.0000 |
| 4 | leaky-ReLU | 5 | **6.25** | 1 | 1.25 | 0.9665 | 1.0000 |
| **4** | **ReLU** | **2** | **2.50** | **36** | **45.00** | 0.9468 | 1.0000 |
| **5** | GELU | 61 | 76.25 | 0 | 0.00 | 1.0000 | 1.0000 |
| 5 | tanh | 65 | 81.25 | 0 | 0.00 | 1.0000 | 1.0000 |
| 5 | leaky-ReLU | 31 | 38.75 | 1 | 1.25 | 0.9980 | 1.0000 |
| 5 | ReLU | 19 | 23.75 | 22 | 27.50 | 0.9640 | 1.0000 |
| **6** | GELU | 77 | 96.25 | 0 | 0.00 | 1.0000 | 1.0000 |
| 6 | tanh | 76 | 95.00 | 0 | 0.00 | 1.0000 | 1.0000 |
| 6 | leaky-ReLU | 51 | 63.75 | 0 | 0.00 | 1.0000 | 1.0000 |
| 6 | ReLU | 25 | 31.25 | 16 | 20.00 | 0.9670 | 1.0000 |

### ReLU at width 4 in full

Of 80 runs: **36 (45.0%) at exactly chance, 2 (2.5%) perfect, 42 (52.5%) in
between.** The two perfect runs are depth 3 seed 5 and depth 12 seed 5. By
depth, the dying-ReLU fraction rises from 2 of 20 at depth 3 to 15 of 20 at
depth 12, and the median accuracy is exactly 0.5000 at depths 8 and 12.

The same holds in the 12-configuration parametrization grid: at width 4, of 480
runs each, GELU separates 315 times, tanh 209, leaky-ReLU 55, and **ReLU 10
(2.08%)**, with ReLU at chance in 244 runs.

### What was wrong and what was fixed

"At width 4 every activation reaches 1.0000" is true only in the sense of *at
least one run*. It is not "every activation succeeds at width 4", and reading it
that way overstates the result. ReLU reaching 1.0000 in 2 of 80 runs while
sitting at chance in 45% is not reliable training to perfect accuracy — it is
occasional success against a background of optimisation failure.

Three instances were corrected: `width_sweep.md` (one), and
`parametrization_sensitivity.md` (two). Each now gives the per-activation rates
inline. A fourth match in `linking_projected.md` refers to projected linking
values reaching 0, not to accuracy, and is correct as written.

No measured value changed.

## Training protocol: ours versus theirs

Their protocol is stated in Appendix G.1–G.2 of arXiv:2606.31856v1 and the code
is at `github.com/7pocheR/low_dimensional_topology`, which is public and
reachable.

| | **Ours** | **Theirs** |
|---|---|---|
| Optimizer | Adam | Adam (FFN), AdamW (ResNet) |
| Learning rate | `1e-2` | `1e-3` |
| Schedule | none, fixed | none stated; early stopping |
| Batch size | **full-batch (2,000)** | **128** |
| Steps / epochs | 2,000 steps, fixed | up to 800 epochs, early stopping with patience 100–200 |
| Loss | cross-entropy | cross-entropy |
| Initialization | PyTorch `nn.Linear` default: `kaiming_uniform_(a=sqrt(5))`, bias `U(±1/sqrt(fan_in))` | **not stated in the paper** |
| Data | 2,000 train + 2,000 eval, independent seeds | 6,000 points (3,000/class), 80/20 train/validation split |
| Success criterion | 100% train **and** ≥99% eval | **not stated**; accuracy reported as mean/max over seeds |
| Seeds | 20 per cell | 30 per cell (Tables 2, 3), 15 (Table 7) |

### Differences that plausibly matter

1. **Learning rate, 10× higher.** Ours is `1e-2` against their `1e-3`. Combined
   with ReLU's dying-unit failure mode, a larger step is a candidate explanation
   for our 45% at-chance rate at width 4.
2. **Full-batch versus batch size 128.** We take 2,000 deterministic full-batch
   steps; they take many small stochastic steps per epoch. Minibatch noise is a
   known escape route from the dead-ReLU basin, so this difference cuts in the
   direction of our seeing more collapse than they do.
3. **No early stopping.** We report the final step unconditionally; they stop on
   a validation criterion with patience 100–200. Ours cannot recover a better
   earlier checkpoint.
4. **Held-out data is independently generated** for us (separate seed, disjoint
   sample) rather than an 80/20 split of one draw.

### Their data generation includes corrugation, ours does not

Appendix G.1 states the thickening procedure: points are sampled as
`γ(t) + ε·n(t)` with `ε ~ U(0, r)`, `r = 0.15`, **and high-frequency
oscillations `0.3 sin(100t)` are added to preserve topology**.

Our `linked_tori` has no such oscillation. This is the corrugation the author
describes in his follow-up, and it is already part of their published
parametrization — so the difference is not a new suggestion but a respect in
which our replication has been incomplete from the start. It is implemented and
swept in Task 2.

### Cannot be determined

- Their initialization scheme (not stated in the paper; not centrally
  documented in the repository README).
- Their acceptance criterion for a "successful" run; Tables 2 and 3 report mean,
  standard deviation, and maximum accuracy over seeds rather than a pass rate,
  so there may be no gate equivalent to ours.
- Their learning-rate schedule beyond the stated constant `1e-3` and early
  stopping.
- Per-experiment hyperparameters live in individual scripts in the repository
  rather than the README; the values above are those stated in the paper, which
  is the authoritative published source. Where the repository README lists
  differing values (for example batch size 32 for a smoke test), those are
  noted as smoke-test settings rather than the reported protocol.

## Result: the width-3 monotonic zero survives their protocol

1,920 runs at 30 seeds per cell under Appendix G.2 settings, parametrization
held fixed so protocol is the only variable.

**Zero monotonic separations in 360 width-3 runs under their protocol**, joining
0 of 240 under ours. Best monotonic evaluation accuracy anywhere is 0.9790.

| Activation | Ours (n=80) | Theirs (n=120) |
|---|---|---|
| tanh | 0 perfect, max 0.9870 | **0 perfect**, max 0.9780 |
| ReLU | 0 perfect, max 0.9735 | **0 perfect**, max 0.9790 |
| leaky-ReLU | 0 perfect, max 0.9700 | **0 perfect**, max 0.9775 |
| GELU | 6 perfect | 2 perfect |

### The gap did not respond to protocol; the monotonic minimum did

Stating this carefully, because the two are the same quantity here. GELU's
minimum error count is **0 under both protocols**, so the "gap"
(`monotonic_min − GELU_min`) simply *is* the monotonic minimum. Reporting that
the gap widened from 26 to 42 would present an identity as a response.

What actually happened:

| Protocol | GELU min errors | Monotonic min errors |
|---|---:|---:|
| Ours | **0** (floored) | 26 |
| Theirs | **0** (floored) | 42 |

Per activation, minimum errors at width 3:

| Activation | Ours | Theirs |
|---|---:|---:|
| tanh | 26 | 44 |
| ReLU | 53 | 42 |
| leaky-ReLU | 60 | 45 |

**GELU stayed floored while the monotonic activations got further from
separation.** The best monotonic run went from 26 misclassified points to 42.

This is a stronger statement than the gap framing allowed. Under a protocol that
**measurably makes GELU separate less often** (see below), the monotonic
activations still never separate, and their closest approach gets worse. Pure
reachability struggles to account for that: if the difference were only about
how easily each activation reaches a solution that exists for all of them, a
protocol that hinders GELU should help or at least not hurt the others relative
to it. Instead GELU remains at the floor and monotonic degrades.

### GELU separates less often under their protocol, and the comparison is limited

| Width | Ours % perfect | Theirs % perfect |
|---:|---:|---:|
| 3 | 7.5% | **1.7%** |
| 4 | 67.5% | **40.0%** |
| 5 | 76.2% | **60.0%** |
| 6 | 96.2% | **76.7%** |

The mechanism is early stopping. At width 3, **75% of GELU runs stopped early,
at a median epoch of 488 with the best checkpoint at median epoch 338**, against
a budget of 800. Those runs were **cut off, not converged**.

This is a limitation of the comparison, not merely an explanation of it. **Under
their protocol we do not know what GELU's separation rate would be with the full
budget**, because patience terminated most runs before it was spent. Two
framings are both defensible and neither is privileged:

- **Ours is the fairer test of what GELU can do.** Fixed 2,000 steps with no
  stopping criterion lets every run use its whole budget, so the separation rate
  reflects what the architecture achieves when optimisation is allowed to run.
- **Theirs is the fairer test of what GELU does under their stated conditions.**
  It is the published protocol, and their reported numbers were produced under
  it, so it is the right basis for comparing against their tables.

The monotonic zero is unaffected by this ambiguity: it holds under both, and
early stopping cannot explain a rate of exactly zero across 600 runs in total.

### The dead-ReLU prediction failed, which strengthens the numbers

Predicted: a substantial drop in the at-chance fraction under their protocol.
Observed: roughly five points.

| Width | Ours at chance | Theirs at chance | Change |
|---:|---:|---:|---:|
| 3 | 61.3% | 55.8% | −5.5 pts |
| 4 | 45.0% | 40.8% | −4.2 pts |
| 5 | 27.5% | 20.0% | −7.5 pts |
| 6 | 20.0% | 12.5% | −7.5 pts |

Minibatch noise, a learning rate ten times smaller, and early stopping with
best-checkpoint restoration together barely move it. Had the rate collapsed, our
reported dead-ReLU figures would have been an optimiser artifact needing
correction wherever quoted. Instead they survive three simultaneous changes,
each independently working against the collapse, so they are closer to a
property of narrow ReLU networks on this problem than to our training choices.
The figures stand as measured; what is new is that they are not
protocol-specific.

Width-4 ReLU reaches 1.0000 in 1.7% of runs under their protocol against 2.5%
under ours, so the author's observation that ReLU is not reliably trainable to
perfect accuracy at width 4 holds under both.
