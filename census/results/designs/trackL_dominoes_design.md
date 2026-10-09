DRAFT for the author, 2026-10-09 — NOT a registration and not approved. Inputs: exploration
`trackL_explore/` (README, `L_tables.md`; seeds 2,991,000–2,991,099, never to be registered). ‡ = set after
exploratory data. **The L2 gate passed, but the exploration predicts R FAILS (power 0.00): see decision 1 first.**

# Track L: the output-layer lever on MNIST-CIFAR dominoes

**Data (on-disk copies only; no download).** `src/mnist_data.py`, `src/cifar_data.py` caches. Recorded SHA-256
(`data/*/SHA256SUMS`) equal the published values:
- MNIST train-images `440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609`
- MNIST train-labels `3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c`
- MNIST t10k-images `8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6`
- MNIST t10k-labels `f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6`
- CIFAR-10 tarball `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`

The archives are not on disk, so the `.npy` files actually read are hashed too (`L_data.verify`; e.g. mnist/train_images c5b45806…,
cifar10/train_images 304a769a…).

**Construction** (Shah et al. 2020 released code, as summarised by the author; not re-read here, no network). Class 0
= MNIST 0 + CIFAR automobile, class 1 = MNIST 1 + CIFAR truck; MNIST padded to 32×32, three channels, on top →
3×64×32 in [0, 1], no normalisation; random one-to-one pairing within class, truncated to the smallest count: **10,000
train, 1,960 test**. The paper's text says 50,000 / 10,000; the code gives these counts. Fixed construction seed
20261009 (one dataset for all seeds‡).
- **p = 0.8‡** of each training class has the matching digit; the other 20% the other digit (Kirichenko et al. 2023 use
  100/99/95%). The CIFAR half is always correct. MNIST then separates the class means but only 80% of points (its Bayes
  BCE floor is 0.500); CIFAR separates every point.
- **Why 0.8:** the gate passed with the largest margin (randomised accuracy 0.487 → +0.198, 5/5 seeds). It passed
  narrowly at 0.9 (+0.109), failed at 0.95 (+0.062), and at 1.0 CIFAR was never used (0.478 throughout, as Shah).
- **Tests:** *orig* (true digit); *rand* (the 1,960 MNIST halves under one fixed permutation; MNIST-only score 0.479);
  *rev* (each class paired with the other class's test digits).

**Network, optimiser.** MLP 6144 → 256 → 256 → 1, ReLU, PyTorch default init (drawn inside `fork_rng`). BCE. SGD,
momentum 0.9, lr 0.01‡, batch 128, no weight decay. Runs stop at train BCE ≤ 0.002. Full-train BCE is checked every 5
steps to 500, then every 2%. No CNN was explored.

**Arms.**

| arm | learning rates |
|---|---|
| 1 standard | 0.01 everywhere |
| 2 output | 0.01/F on the output weight (bias at 0.01) |
| 3 global | 0.01/F everywhere |
| 3cm | 0.01/G_cm everywhere, cost-matched to arm 2's steps to 0.03 (2B's G^a rule) |

**Pilot rule for F (proposed).** F is the smallest of 2, 4, …, 1024 whose median arm-2 step cost to ℓ₁ is ≥ 2 on 10
pilot seeds. If no F qualifies, the lever has no handle and the track stops. In the exploration no F qualified: ÷1024
cost only ×1.33.

**Matching.**
- Matched loss is the first check with train BCE ≤ ℓ, for ℓ ∈ {0.6, 0.3, 0.03}‡: the MNIST stage, then below the
  MNIST floor, then late. Arm 1's randomised accuracy at these levels was 0.487, 0.649 and 0.683.
- Also reported descriptively: matched steps and convergence, with step costs. **No endpoint advantage is claimed.**

**Measures.** Primary: *rand* accuracy. Secondary: *rev*. Descriptive: *orig*, and train accuracy on the flipped 20%.

**Criteria.** Δ = arm − arm 1, same seed. Each cell uses a 95% percentile bootstrap of the median Δ (10,000
resamples).
- **R:** arm 2's lower end is > 0 at all three levels.
- **N:** for arms 3 and 3cm, each upper end is < δ = 0.02‡ at all three levels.

**Validity.**
- UNRESOLVED if fewer than 90% of seeds reach a level in both arms, or if a run of an arm used is non-finite (2B's D5).
- Arm 2's realised cost at ℓ₁ is reported.

**Exploration (5 seeds unless noted; `L_tables.md`).**
- Arm 2 had no effect anywhere: |median Δrand| ≤ 0.020, with no consistent sign. This held for factors 4–1024, SGD and
  Adam, ReLU and tanh, widths 256 and 16, and an output init ×0.01 (2–5 seeds each).
- ÷16 cost ×1.33, 1.34 and 0.98 at the three levels (2B: ×7–16). The readout ends smaller (‖v‖₂ 1.44 against 3.88)
  while the hidden layers carry the scale. ‖v‖₁ ≈ 8 at default init; 2B started at ≤ 0.5 s\*.
- Arm 3 ÷16 cost ×3.3–7, with Δrand −0.005 to +0.001. Under Adam it was +0.010 to +0.014 (5/0).
- G_cm degenerates to about 1, because arm 2 costs about 1.

**Seeds, power** (exploration's paired differences resampled; 300 simulations, inner bootstrap 1,000; crude, since
they rest on 5 seeds):

| criterion | n = 20 | 40 | 80 |
|---|---|---|---|
| R (÷16) | 0.00 | 0.00 | 0.00 |
| N, arm 3 (SGD) | 0.89 | 0.98 | 1.00 |
| N, arm 3 (Adam) | 0.15 | 0.31 | 0.56 |

Proposed: 40 registered seeds, 2,992,000–2,992,039, and pilot seeds 2,993,000–2,993,009 (the scan of 2,992,000–099
and 2,993,000–019 is clean).

**Compute.** About 20 s per arm-1 or arm-2 run and 75 s per arm-3 run, at a peak RSS of 1.1 GB. That is about 2.4 min
per seed: 1.6 h for 40 seeds, 3.2 h for 80. This is far below 24 h on one worker.

**Outcomes.**
- R PASS, N PASS: the lever transfers.
- R FAIL, N PASS (expected): in a wide ReLU MLP the readout's rate does not set feature use.
- N FAIL: global slowing helps.
- Arm 2 below arm 1: the lever hurts.

**Strict causal rule.** No criterion is a forecast. Every criterion compares observed outcomes after all runs end, so
no cutoff applies.

**OTS guard.** The registered `run` refuses to start until the registration's OpenTimestamps proof file exists.

**Needs your decision.**
1. **Whether to register at all.** The exploration gives R a power of 0.00 at every n, and the proposed pilot rule
   would stop the track. The options are:
   - (a) Register as a predicted negative: the boundary of the lever.
   - (b) Change the network so that the readout must grow, for example a small output init. That needs more
     exploration; ×0.01 was null on 3 seeds.
   - (c) Report the null descriptively and stop Track L.
2. p = 0.8 rather than 0.9.
3. SGD rather than Adam. Both were null, and Adam threatens N.
4. Arm 2 slows the weight only, as 2B's primary arm did.
5. The pilot rule for F, or F = 16 fixed a priori.
6. Arm 3cm: drop it, or clamp G_cm ≥ 1 (it equals arm 1 here).
7. The levels {0.6, 0.3, 0.03} and δ = 0.02.
8. One fixed dataset, or a per-seed pairing.
9. The CNN: drop it, or explore it (not timed).
10. The seed count.
