DRAFT, 2026-10-09: the author's decisions are applied; this is NOT a registration. The author reviews this page
before anything is registered. Inputs: the exploration in `trackL_explore/` (README, `L_tables.md`), run on seeds
2,991,000–2,991,099, which are never to be registered (b08843d). ‡ = set after exploratory data.

# Track L: the output-layer lever on MNIST-CIFAR dominoes

**Predicted outcome, stated up front: R FAILS and N PASSES.** These are predictions from the exploration: R has
power 0.00 at every seed count, and N has power 0.98 at n = 40 under SGD. The criteria are the ones drafted before
this prediction. The track registers this predicted null.

**Data.**
- On-disk caches only (`src/mnist_data.py`, `src/cifar_data.py`); nothing is downloaded.
- The recorded SHA-256 values equal the published ones: MNIST train-images 440fcabf, train-labels 3552534a,
  t10k-images 8d422c7b, t10k-labels f7ae60f9; CIFAR-10 tarball 6d958be0.
- The full values, and the hashes of the `.npy` files actually read, are in `trackL_explore/L_data_check.log`
  (checked by `L_data.verify`).

**Construction** (follows the author's summary of Shah et al.'s released code; the code was not re-read here because
there is no network access).
- Class 0 is MNIST 0 with a CIFAR automobile. Class 1 is MNIST 1 with a CIFAR truck.
- MNIST is padded to 32×32, repeated to 3 channels and placed on top: 3×64×32, values in [0, 1], no normalisation.
- Images are paired one-to-one at random within each class, and each class is truncated to the smallest count. This
  gives **10,000 train and 1,960 test** images.
- **Discrepancy:** Shah et al.'s text says 50,000 / 10,000.
- **One fixed dataset** is built, with construction seed 20261009.
- **p = 0.8‡** (following Kirichenko et al. 2023, who use 100/99/95%):
  - 80% of each training class carries the matching digit and 20% carries the other digit. CIFAR is always correct.
  - So MNIST separates the class means but not every point: its BCE floor is 0.500. CIFAR separates every point.
- **Why p = 0.8:** the L2 gate passed with the largest margin there.
  - Standard training's randomised accuracy rose from 0.487 to +0.198 (5/5 seeds).
  - At 0.9 the rise was +0.109, a narrow pass. At 0.95 it was +0.062, a fail.
  - At 1.0 CIFAR was never used.
- **Tests:**
  - *orig*: the true digit.
  - *rand*: the MNIST halves under one fixed permutation, so MNIST is independent of the label. An MNIST-only
    classifier scores 0.479.
  - *rev*: each class is paired with the other class's digits.

**Network and optimiser.**
- MLP 6144 → 256 → 256 → 1, ReLU, PyTorch default initialisation drawn inside `fork_rng`, BCE loss.
- SGD with momentum 0.9, lr 0.01‡, batch 128, no weight decay. A run stops at train BCE ≤ 0.002.
- Full-train BCE is checked every 5 steps up to step 500, then every 2% of the step count.

**Arms.**
- **Arm 1, standard:** lr 0.01 everywhere.
- **Arm 2, output:** lr 0.01/16 on the output **weight**; the output bias stays at 0.01.
- **Arm 3, global:** lr 0.01/16 everywhere.
- **F = 16 is fixed a priori**, as in 2B. It replaces the proposed pilot rule (arm-2 step cost ≥ 2), which no F up to
  1024 met in the exploration, so that rule would have stopped the track.
- **Arm 3cm is dropped:** with arm 2's step cost ≈ 1, the cost-matched factor G_cm ≈ 1.

**Matching.**
- Each matched point is the first check with train BCE ≤ ℓ, for ℓ ∈ {0.6, 0.3, 0.03}‡.
- The three levels are the MNIST stage, below the MNIST floor, and late. Arm 1's *rand* accuracy there was 0.487, 0.649
  and 0.683.
- Matched steps and convergence are reported descriptively, with step costs. **No endpoint advantage is claimed.**

**Measures.**
- Primary: *rand* accuracy.
- Secondary: *rev* accuracy.
- Descriptive: *orig* accuracy, and train accuracy on the flipped 20%.

**Criteria.**
- Δ = arm − arm 1 on the same seed.
- Each Δ gets a 95% percentile bootstrap interval of the median, from 10,000 resamples.
- **R (arm 2):** the lower end is > 0 at all three levels.
- **N (arm 3):** the upper end is < δ = 0.02‡ at all three levels.

**Validity: UNRESOLVED if** fewer than 90% of seeds reach a level in both arms, or a run in an arm the criterion uses
is non-finite (2B's D5). Arm 2's realised step cost is reported.

**Exploration (`L_tables.md`).**
- Arm 2's |median Δ*rand*| was ≤ 0.020 everywhere, with no consistent sign. This held for factors 4–1024, SGD and
  Adam, ReLU and tanh, widths 256 and 16, and an output initialisation scaled by 0.01.
- ÷16 costs ×1.33, ×1.34 and ×0.98 the standard arm's steps to the three levels; 2B's arm cost ×7–16.
- The readout ends smaller under ÷16 (‖v‖₂ 1.44 against 3.88), but the hidden layers carry the scale.
- Arm 3 costs ×3.3–7, with Δ*rand* between −0.005 and +0.001.

**Seeds and power.**
- **40 registered seeds:** 2,992,000–2,992,039. The seed scan is clean.
- **Pilot seeds 2,993,000–2,993,009 are unused** unless a machine check needs them; any such use would be stated.
- Power, from resampling the 5 exploration seeds (crude): R 0.00 and N 0.98 at n = 40.

**Compute.**
- About 20 s per arm-1 or arm-2 run and 75 s per arm-3 run.
- About 2 min per seed, so **about 1.3 h** for all 40 seeds on one worker.
- Peak RSS 1.1 GB.

**Outcomes.**
- **R FAIL, N PASS (predicted):** in this MLP the readout's learning rate does not decide which feature is used.
- **R PASS:** the lever transfers, against the prediction.
- **N FAIL:** global slowing helps.
- **Arm 2 below arm 1:** the lever hurts.

**Strict causal rule.** No criterion is a forecast. All criteria compare observed outcomes after every run has ended,
so no cutoff applies.

**OTS guard.** The registered `run` refuses to start until the registration's OpenTimestamps proof file exists.

**Decided (author, 2026-10-09).**
1. Register the predicted null, with the criteria unchanged.
2. p = 0.8.
3. SGD; Adam is dropped.
4. Arm 2 slows the output weight only.
5. F = 16, fixed a priori.
6. Arm 3cm is dropped; N applies to arm 3 only.
7. Levels {0.6, 0.3, 0.03} and δ = 0.02.
8. One fixed dataset.
9. The CNN is dropped.
10. 40 seeds.

**Still open:**
- Whether the secondary *rev* measure gets its own criterion. Proposed: descriptive only.
- The registration text will carry the full hashes and tables.
