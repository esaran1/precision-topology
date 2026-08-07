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
