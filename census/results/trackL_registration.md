# Track L registration: the output-layer lever on MNIST-CIFAR dominoes

**NOT YET REGISTERED: STOPPED BEFORE THE REGISTRATION COMMIT.** This text implements the approved page rule by rule.
Eight details that the page leaves open touch a criterion, a validity condition or the scoring set (§14, D1–D8). Each
has a proposed resolution, implemented in `src/trackL.py` and tested in `tests/test_trackL.py`. **They need the
author's decision before the registration commit.** The proposals carry over 2B's approved D1–D4 and D6–D8 (2B
registration, §11) unchanged in substance.
- **Not yet done:** no registered seed (2,992,000–2,992,039) and no pilot seed (2,993,000–2,993,009) has been drawn,
  initialised or trained.
- **At registration:** this file, the approved page, `src/trackL.py`, `tests/test_trackL.py`, the frozen inputs
  (`results/trackL/seed_scan.json`, `results/trackL/frozen.json`), the data checksums and the exploration files they
  read are hashed into `results/trackL/registration.sha256` and committed in one commit. Then
  `results/trackL/registration_stamp.txt` (the registration commit hash and the SHA-256 of this file and the manifest) is
  committed, pushed and OpenTimestamped.
- **`run` refuses to start until `results/trackL/registration_stamp.txt.ots` exists** (§17; tested).

## 1. Design reference

- **Approved page:** `results/designs/trackL_dominoes_design.md` (commit 9d7e77c). The author approved it on
  2026-10-09: "approve the page as drafted". Everything on it is binding here. Exploration: `trackL_explore/` (b08843d),
  seeds 2,991,000–2,991,099, never registered.
- **The author's decisions (2026-10-09), from the page:**
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
- **The page's two open items, as approved with it:**
  - The secondary *rev* measure has no criterion of its own. It is **descriptive only** (the page's proposal).
  - This text carries the full hashes (§3) and the tables (§13, §15).
- **The author's instruction with the approval:** the α follow-up (`trackL_alpha_explore/`, df2b365) is reported
  **descriptively beside the Track L null**, labelled as motivated by it (§12). It adds no runs and no criterion.

## 2. Predicted outcome, stated up front

**R FAILS and N PASSES.** This is the exploration's prediction: R has power 0.00 at every seed count, and N has power
0.98 at n = 40 under SGD (`trackL_explore/power_sgd.log`). The criteria were drafted before this prediction and are
unchanged. The track registers this predicted null. A PASS of R would be against the prediction.

## 3. Data (`trackL.verify_data`, `trackL.build`)

**On-disk caches only. Nothing is downloaded.**
- `src/trackL.py` reads the eight `.npy` caches with `np.load`. It does not import `src/mnist_data.py` or
  `src/cifar_data.py`, which would fetch a missing file. It refuses to run if a cache is missing (tested). torchvision is
  not installed and is not used.
- Before every build, the recorded `data/*/SHA256SUMS` must equal the published SHA-256 values, and every `.npy` cache
  must hash to the registered value. Any mismatch stops the build (tested).

| file (published download) | SHA-256 |
|---|---|
| MNIST `train-images-idx3-ubyte.gz` | `440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609` |
| MNIST `train-labels-idx1-ubyte.gz` | `3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c` |
| MNIST `t10k-images-idx3-ubyte.gz` | `8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6` |
| MNIST `t10k-labels-idx1-ubyte.gz` | `f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6` |
| CIFAR-10 `cifar-10-python.tar.gz` | `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce` |

| cache actually read | SHA-256 |
|---|---|
| `data/mnist/train_images.npy` | `c5b45806d970e809a5f376f9b8461b7ca3b3e0a321282d72d06d00ab6c6c418f` |
| `data/mnist/train_labels.npy` | `5dd4d822cab3e20099239bc9d433d587ae3ce00e084d191079dd30b38380b336` |
| `data/mnist/test_images.npy` | `4acfa5c2911a2f95015eda9a9b825fbd6bec0f6a6f66942979b1473d33943a11` |
| `data/mnist/test_labels.npy` | `ff7e84b144c037e7215dfa787d6773550c5db83029d9a4e7bae6e90f605f081d` |
| `data/cifar10/train_images.npy` | `304a769ab0682c43bbdc4f766303066aef2073d37441d820ecc2666ccb3432b3` |
| `data/cifar10/train_labels.npy` | `9e2ee9261f6a7c6509b35aaa7161bf15d6d99673bbcac04130702d12a6da88d3` |
| `data/cifar10/test_images.npy` | `efbfa3b24c4f91c6febe8a57d87b6448f2c1226d8b2aa24d14db1eea85ad46c6` |
| `data/cifar10/test_labels.npy` | `fc48d9ecfdbeacce2dacf004498170f2df12e75e3485475017d2663b587a92f3` |

These equal `trackL_explore/L_data_check.log`.

**Construction.** It follows the author's summary of Shah et al.'s released code. The code itself was not re-read
(there is no network access).
- **Classes:** class 0 is MNIST 0 with a CIFAR-10 automobile (CIFAR label 1). Class 1 is MNIST 1 with a CIFAR-10 truck
  (CIFAR label 9).
- **Image:** MNIST 28×28 is zero-padded by 2 on every side to 32×32, repeated to 3 channels and placed **on top** of the
  CIFAR image: 3×64×32, values in [0, 1] (÷255, no normalisation).
- **Pairing:** one-to-one at random within each class. Each class is truncated to the smallest count. This gives
  **10,000 train** (5,000 per class) and **1,960 test** (980 per class) images.
- **Discrepancy:** Shah et al.'s text says 50,000 / 10,000. The released code, as summarised, gives 10,000 / 1,960,
  and that is what is built.
- **One fixed dataset**, from `numpy.random.default_rng(20261009)` (the construction seed). The seeds vary only the
  initialisation and the minibatch order.
- **p = 0.8‡:** exactly 1,000 of each class's 5,000 training images (20%) carry a digit of the other class. CIFAR is
  always correct. Each digit is used exactly 5,000 times. MNIST separates the class means but not every point (its BCE
  floor is H(0.8) = 0.500); CIFAR separates every point.
- **Tests** (CIFAR halves fixed):
  - *orig*: the true digit.
  - *rand*: the 1,960 test MNIST halves under one fixed permutation, so MNIST is independent of the label. The
    permuted digit agrees with the label on 938 of 1,960 points, so an MNIST-only classifier scores **0.479**.
  - *rev*: each class is paired with the other class's digits.

**Dataset hash.** `trackL.dataset_sha256` hashes the seven arrays (`Xtr`, `ytr`, `flip`, `Xorig`, `Xrand`, `Xrev`,
`yte`: name, dtype, shape and bytes). The registered build gives
`6867910ba670acfced6154d198fdf509edeb350afa5e936735ea5fbe082b332f`. The exploration did not record a dataset hash, so
this value was computed from the committed exploration code (`trackL_explore/L_data.py`, b08843d,
`L_data.build(0.8)`). The test suite rebuilds it with both codes and checks both against it. `run` refuses a dataset
with any other hash.

Tested: the counts; 1,000 flips per class; flipped images carry the other class's digit; CIFAR is always the true
class; *orig*, *rand* and *rev* carry the right digits; the 0.479 agreement; the padding and channel layout.

## 4. Network, optimiser, stopping (`trackL.make_net`, `run_one`)

- **Network:** MLP 6144 → 256 → 256 → 1, ReLU. PyTorch default initialisation after `torch.manual_seed(seed)`, drawn
  inside `torch.random.fork_rng`, so the global torch RNG state is restored (tested).
- **Loss:** BCE with logits (mean).
- **Optimiser:** SGD with momentum 0.9, lr 0.01‡, batch 128, no weight decay. The minibatch order comes from
  `numpy.random.default_rng(seed)`, reshuffled every epoch. One torch thread inside the run (restored after).
- **Checks** (no interpolation): step 0, every 5 steps up to step 500, then every max(5, ⌊0.02·t⌋) steps. At each check:
  full-train BCE, train accuracy on the predictive 80% and the flipped 20%, test accuracy on *orig*, *rand* and *rev*,
  ‖output weight‖₂ and ‖W₂‖_F.
- **Stop:** at the first check with train BCE ≤ 0.002. **The step cap is 200,000 steps** (the exploration's cap). In the
  exploration no run came near it: arm 3 stopped at 41,854–42,691 steps. A run also stops at a non-finite check (D5).
- **Machine stop:** the process exits (code 3, the run is not recorded) if its max RSS exceeds 1.5 GB.

## 5. Arms and seeds

| arm | name | learning rates | role |
|---|---|---|---|
| 1 | `std` | 0.01 everywhere | reference |
| 2 | `out16` | 0.01/16 on the output layer's **weight**; its bias and the hidden layers at 0.01 | R |
| 3 | `glob16` | 0.01/16 everywhere | N |

- **F = 16 is fixed a priori**, as in 2B. It replaces the proposed pilot rule (arm-2 step cost ≥ 2), which no F up to
  1024 met in the exploration, so that rule would have stopped the track.
- **Arm 3cm is dropped:** with arm 2's step cost ≈ 1, the cost-matched factor G_cm ≈ 1.
- **Seeds:** 40 registered seeds, **2,992,000–2,992,039**, shared by the three arms (120 runs). Per seed the arms run
  in the order 1, 2, 3. `run` is resumable per (seed, arm).
- **Pilot seeds 2,993,000–2,993,009 are unused.** No machine check needed them: the freeze reproduces committed
  exploration runs on exploration seed 2,991,000 instead (§15).

## 6. Matching and measures

- **Matched point:** for each level ℓ ∈ {0.6, 0.3, 0.03}‡, the first check with train BCE ≤ ℓ.
- The three levels are the MNIST stage, below the MNIST floor (0.500), and late. In the exploration, arm 1's *rand*
  accuracy there was 0.487, 0.649 and 0.683 (5 seeds).
- **Primary measure:** *rand* accuracy at the matched points.
- **Secondary:** *rev* accuracy, descriptive only (§1).
- **Descriptive:** *orig* accuracy, train accuracy on the flipped 20%, matched steps, step costs and convergence (§11).

## 7. Criteria (`trackL.criterion_R`, `criterion_N`, `score_tables`)

- **Paired differences:** Δ = arm − arm 1 on the same seed, in *rand* accuracy at each level.
- **Interval:** a 95% percentile bootstrap interval of the median Δ, from 10,000 resamples (details D1–D3, §14).

| criterion | arm | PASS |
|---|---|---|
| **R** | 2 | the lower end is > 0 at all three levels |
| **N** | 3 | the upper end is < δ = 0.02‡ at all three levels |

- **FAIL:** a criterion that does not pass and is not UNRESOLVED (§8).
- **Strict inequalities:** a lower end of exactly 0 fails R; an upper end of exactly 0.02 fails N (tested).
- **N is one-sided:** an arm 3 that is worse than arm 1 passes N (tested).

## 8. Validity: UNRESOLVED

A criterion is **UNRESOLVED**, overriding PASS and FAIL (D4), if either holds:
- **Reach:** fewer than 90% of the seeds reach a level in both arms, i.e. fewer than 36 of 40 at any of the three
  levels (D6).
- **Non-finite:** a run in an arm the criterion uses is non-finite (2B's D5, on the page). R uses arms 2 and 1, N uses
  arms 3 and 1. So a non-finite arm-2 run voids R only, a non-finite arm-3 run voids N only, and a non-finite arm-1 run
  voids both. What counts as non-finite: D5.

Arm 2's realised step cost is reported (§11).

Constructed cases, all tested: R and N PASS and FAIL at each level, at the strict boundaries, and N one-sided; reach at
36/40 (resolved) and 35/40 (UNRESOLVED), counted in both arms; reach overriding a FAIL; non-finite runs in each arm with
the scope above; incomplete or duplicated sets of runs refused.

## 9. Outcome rule (`trackL.outcome`)

| verdicts | outcome | statement (the page's) |
|---|---|---|
| R FAIL, N PASS | R FAIL, N PASS (**predicted**) | "In this MLP the readout's learning rate does not decide which feature is used." |
| R PASS | R PASS | "The lever transfers, against the prediction." |
| N FAIL | N FAIL | "Global slowing helps." |
| any UNRESOLVED | UNRESOLVED | lists the unresolved criteria, and the other verdict |
| arm 2 below arm 1 (D8) | added to the above | "The lever hurts." |

`matches_prediction` is true only for R FAIL and N PASS.

## 10. Strict causal rule

**No criterion is a forecast.**
- Every criterion compares observed outcomes after every run has ended, so no cutoff applies. `score_tables` refuses
  anything but the complete set of runs, every arm × every seed exactly once.
- `src/trackL.py` imports no other `src` module, so it reaches no forecaster (its import closure is itself; tested).
- A test asserts that no criterion function mentions a forecast, and that forecast keys added to every run row and
  matched point change no verdict.

## 11. Descriptive, registered as such (`trackL.descriptive`; never a verdict)

- Values at each matched point and at the end of training, per arm: *rand*, *rev*, *orig*, flipped-20% and
  predictive-80% train accuracy, ‖v‖₂, ‖W₂‖_F.
- Paired arm − arm 1 medians (seeds up/down) for each of these. For *rev* (secondary) and *orig*, the same bootstrap
  interval as the criteria, labelled descriptive.
- **Matched steps and step costs:** steps to each level, and the per-seed ratio to arm 1 (arm 2's realised step cost).
- **Convergence:** steps to the stop, and stop reasons (loss, cap, non-finite), with the ratio to arm 1.
- **No endpoint advantage is claimed.**

## 12. The α follow-up: a descriptive section of the results page (planned; no new runs)

The results page will carry a section titled **"The α follow-up (EXPLORATORY; motivated by the Track L null)"**,
placed beside the Track L verdicts. It will:
- cite df2b365 and `results/designs/trackL_alpha_explore/` (README, `A_tables.md`, `A_summary.json`; their hashes are
  in the manifest);
- say that the follow-up was run **after** the L2 exploration showed the null and **before** this registration, on
  exploration seeds 2,994,000–2,994,007 (never registered), and that it is motivated by the null;
- report, descriptively, f = α·g on the same data, network and optimiser: Δ*rand* vs α = 1 at the three levels for
  fix α ∈ {4, 0.5, 0.25, 0.1, 0.05, 0.01} and fs α ∈ {4, 0.5, 0.25}, with step costs; the slope on log₁₀ α; the
  analogue of N; and the estimated power of an R-type criterion, 0.00 at n = 20, 40 and 80 for every α arm;
- state its limits (4–8 seeds per cell, one p, one width, ReLU and SGD; the fs mode collapses at α ≤ 0.1).

It runs nothing and computes no new number from new runs. No criterion, gate or validity rule reads it. It changes no
Track L verdict.

## 13. Disclosures

- **The predicted null.** The track registers an outcome the exploration predicts (R FAIL, N PASS). The criteria were
  drafted before the prediction and are unchanged (decision 1).
- **F = 16 was fixed a priori** (decision 5). It replaced the proposed pilot rule (arm-2 step cost ≥ 2). No F up to
  1024 met that rule in the exploration, so the rule would have stopped the track.
- **Set from exploratory data (‡ on the page)**, all on exploration seeds 2,991,000–2,991,004 (5 seeds):
  - **p = 0.8.** The L2 gate passed with the largest margin there: arm 1's *rand* rose from 0.487 to +0.198 (5/5). At
    0.9 the rise was +0.109 (a narrow pass), at 0.95 +0.062 (a fail). At 1.0 CIFAR was never used.
  - **lr = 0.01** (SGD).
  - **Levels {0.6, 0.3, 0.03}:** the MNIST stage, below the MNIST floor, and late.
  - **δ = 0.02.**
- **What the exploration showed** (`L_tables.md`; 5 seeds, SGD, width 256):
  - Arm 2's median Δ*rand*: +0.0046 (3/1), −0.0015 (1/4), +0.0041 (3/2). Step cost ×1.33, ×1.34, ×0.98.
  - Arm 3's median Δ*rand*: +0.0005 (3/2), −0.0051 (2/3), −0.0051 (1/4). Step cost ×7.0, ×5.3, ×3.3.
  - Across every explored setting (factors 4–1024, SGD and Adam, ReLU and tanh, widths 256 and 16, output
    initialisation ×0.01), the largest |median Δ*rand*| of an output-slowed arm was 0.020, with no consistent sign.
  - The readout is not rate-limiting: at BCE 0.03, ‖v‖₂ is 1.44 under ÷16 against 3.88 standard, but the hidden layers
    carry the scale.
- **Power** (resampling the 5 exploration seeds; 300 simulations, inner bootstrap 1,000; crude): R 0.00 and N 0.98 at
  n = 40 (`power_sgd.log`).
- **The construction follows the author's summary** of Shah et al.'s released code, not a re-reading of it, and
  departs from their text's 50,000 / 10,000 (§3).
- **Dropped:** Adam, the CNN, and arm 3cm (decisions 3, 6, 9).
- **Exact reproduction of the exploration.** `run_one` reproduces committed exploration runs bit for bit, at the
  exploration's recorded precision (every value rounded to 5 decimals, every check step):
  - the test suite: arm 2 on seed 2,991,000 (8,110 steps);
  - the freeze: all three arms on seed 2,991,000 (§15).
- **The registered scoring on the exploration seeds** (§15) is a disclosure, not a verdict.
- **The α follow-up** (§12) was run after the null was seen. It is descriptive and motivated by the null.

## 14. The details the approved page left open

### Touching a criterion, a validity condition or the scoring set: PROPOSED, pending the author

Each is implemented and tested as proposed. Each carries over 2B's approved resolution (2B registration §11, decided
2026-10-04/05) unless stated.

| # | detail | proposed resolution | source |
|---|---|---|---|
| D1 | The bootstrap generator | A **fresh** `default_rng(20261009)` for each compared arm (2 and 3). One 10,000 × 40 index matrix is drawn from it and used for all three levels of that arm. The seed is the approval date, as 2B used its own. | 2B D1 (seed changed) |
| D2 | Seeds that do not reach a level in both arms | Excluded from that level only: in each resample the median is over the resampled seeds that reach it (`nanmedian`), and the point median likewise. Resampling stays over all 40 seeds. | 2B D2 |
| D3 | The percentile rule | numpy's default (linear interpolation) at 2.5 and 97.5. `boot_ci_median` is 2B's registered function with the seed changed; a test checks equality with `src/phase2b.boot_ci_median`. | 2B D3 |
| D4 | UNRESOLVED precedence | UNRESOLVED overrides PASS as well as FAIL. | 2B D4; page: "UNRESOLVED if" |
| D5 | What "non-finite" means | At a check, the train BCE or any parameter is not finite. The run stops there and is recorded with `finite = false`. (Checks fall at least every 2% of the steps; a non-finite parameter between checks propagates to the next check.) | 2B D6, adapted to checks |
| D6 | Reach | A seed reaches a level if a check with train BCE ≤ ℓ occurs before the run stops: at train BCE ≤ 0.002, at the 200,000-step cap, or at a non-finite check. 90% of 40 = 36. | 2B D7; cap from the exploration |
| D7 | The accuracy decision at a tie | Class 1 iff the logit > 0, so a logit of exactly 0 is class 0. | as explored |
| D8 | "Arm 2 below arm 1: the lever hurts" | The page lists this outcome without a rule. Proposed: stated (descriptively, never a verdict) when arm 2's upper end is < 0 at all three levels, the mirror of R. | — |

### Implementation only (recorded; no decision needed)

- The run order: per seed, arms 1, 2, 3.
- Resumability per (seed, arm), with one JSON line per run (appended, flushed, fsynced).
- One dataset build per `run` process, checked against the registered hash.
- The memory gate before every run; the 1.5 GB RSS machine stop.
- The full check trajectory is stored per run, unrounded. The exploration rounded to 5 decimals.
- Parameter hashes at initialisation and at the end of each run.
- The file formats.

## 15. Seed scan and freeze

**Seed scan** (`trackL.scan` → `results/trackL/seed_scan.json`): to be filled.

**Freeze** (`trackL.freeze` → `results/trackL/frozen.json`; no registered or pilot seed): to be filled.

## 16. Frozen files and hashes

At the registration commit, `results/trackL/registration.sha256` (`trackL.manifest`) lists the SHA-256 of:
`src/trackL.py` (its whole import closure), `tests/test_trackL.py`, this file, the approved page,
`results/trackL/seed_scan.json`, `results/trackL/frozen.json`, `data/mnist/SHA256SUMS`, `data/cifar10/SHA256SUMS`,
the exploration files (`trackL_explore/`: README, `L_data.py`, `L_data_check.log`, `L_explore.py`, `L_summary.py`,
`L_power.py`, `L_tables.py`, `L_tables.md`, `L_seedscan.py`, `L_seedscan.json`, `power_sgd.log`,
`explore_gate.jsonl`, `explore_lever.jsonl`) and the α follow-up files (`trackL_alpha_explore/`: README,
`A_explore.py`, `A_summary.py`, `A_summary.json`, `A_tables.md`, `A_seedscan.json`).

## 17. Compute, machine rules and the OpenTimestamps guard

- **Compute** (exploration): about 20 s per arm-1 or arm-2 run and 75 s per arm-3 run, about 2 min per seed, so
  **about 1.3 h** for 40 seeds on one worker. Peak RSS about 1.1 GB.
- **Machine rules:** one process, `nice -n 15`, one thread (`OMP_NUM_THREADS=1` etc.). A memory gate is logged to
  `results/trackL/memory_gate.log` before every job and every run: `memory_pressure -Q` free ≥ 25% and swap free
  ≥ 500 MB (else wait and re-check every 60 s), disk free ≥ 20 GB (else STOP). The process stops above 1.5 GB RSS.
- **The OpenTimestamps guard:** `run` (and `score`) call `assert_registration()` first. It refuses (SystemExit) unless
  `results/trackL/registration_stamp.txt.ots` exists, before any other check, any data build or any run. Then every
  manifest hash must match, and the manifest and the stamp must be committed and unmodified. Tested: without the proof,
  `run` and `score` refuse and nothing is built or trained; with it, `run` proceeds and resumes per (seed, arm).

## 18. Reproduce

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
        nice -n 15 .venv/bin/python -m src.trackL scan | freeze | manifest | stamp | run | score
