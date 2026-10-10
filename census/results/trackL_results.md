# Track L results: the output-layer lever on MNIST-CIFAR dominoes

**Registered outcome: R FAIL, N PASS. This matches the registered prediction (R FAIL, N PASS).**

> Registered statement (`scores.json` → `outcome.statements`): "R FAIL, N PASS (predicted): in this MLP the readout's
> learning rate does not decide which feature is used."

- **Registration:** `results/trackL_registration.md`, commit a8ab6db. Stamp commit 0a340f4. OpenTimestamps proof
  commit 36bd16f, made and pushed before any registered run.
- **Runs and scores:** `results/trackL/runs.jsonl` and `results/trackL/scores.json`, commit a36c702. They were produced
  by `python -m src.trackL run` and `python -m src.trackL score`, with the registered code unchanged.
- **Scope:** 40 registered seeds (2,992,000–2,992,039) × 3 arms = 120 runs. Every run is finite. Every run stopped on
  the loss rule (train BCE ≤ 0.002). No run hit the 200,000-step cap.
- **What this page adds:** it writes no new runs and computes no new statistics. Every number is copied from a committed
  producer output (§4). Numbers are rounded **for display only**: accuracies and their differences to 4 decimals, step
  ratios to 2 decimals. The verdicts were computed by the producer on unrounded values.

## 1. Registered verdicts (from `scores.json`)

**Producer:** `src/trackL.py` `score` → `results/trackL/scores.json`. The keys are `arms.<arm>.cells_rand`,
`arms.<arm>.criterion`, `verdicts`, `outcome` and `nonfinite_by_arm`.

**Measure.** Δ = (arm − arm 1) on the same seed, in *rand* test accuracy at the first check with train BCE ≤ ℓ. The
interval is the registered 95% percentile bootstrap interval of the median Δ (10,000 resamples, D1–D3). δ = 0.02.

| arm | level ℓ | median Δ*rand* | 95% interval | seeds up/down | n_both |
|---|---|---|---|---|---|
| 2 `out16` (R) | 0.6 | +0.0000 | [−0.0015, +0.0005] | 17/19 | 40/40 |
| 2 `out16` (R) | 0.3 | +0.0028 | [−0.0051, +0.0099] | 22/18 | 40/40 |
| 2 `out16` (R) | 0.03 | −0.0036 | [−0.0099, +0.0008] | 16/24 | 40/40 |
| 3 `glob16` (N) | 0.6 | −0.0018 | [−0.0036, −0.0003] | 12/26 | 40/40 |
| 3 `glob16` (N) | 0.3 | −0.0038 | [−0.0112, +0.0005] | 14/24 | 40/40 |
| 3 `glob16` (N) | 0.03 | −0.0079 | [−0.0143, −0.0003] | 13/26 | 40/40 |

The two criteria:

| criterion | rule | ends used | reach | non-finite in arms used | verdict |
|---|---|---|---|---|---|
| **R** (arm 2) | lower end > 0 at all three levels | lower ends −0.0015, −0.0051, −0.0099: none > 0 | `reach_ok` true (40/40 at every level ≥ 36) | false | **FAIL** |
| **N** (arm 3) | upper end < δ = 0.02 at all three levels | upper ends −0.0003, +0.0005, −0.0003: all < 0.02 | `reach_ok` true (40/40 at every level ≥ 36) | false | **PASS** |

- **Validity:** no criterion is UNRESOLVED.
  - Reach is 40 of 40 seeds in both arms at every level. The threshold is 36.
  - `nonfinite_by_arm` is false for `std`, `out16` and `glob16`, and `nonfinite_runs` is empty.
- **`outcome`:** "R FAIL, N PASS", with `matches_prediction` = true.
- **The §15 risk did not materialise.** Before the run, the registered scoring on the 5 exploration seeds gave N an
  upper end of 0.0219 at BCE 0.3, which is ≥ δ (`results/trackL/frozen.json`; registration §15). That was a
  disclosure, not a verdict. At n = 40, N's upper end at 0.3 is +0.0005, and N's largest upper end at any level is
  +0.0005. This is consistent with the registered power estimate for N at n = 40 (0.98, `trackL_explore/power_sgd.log`).
- **D8 ("the lever hurts") is not triggered.** D8 would state it, descriptively, only if arm 2's upper end were < 0 at
  all three levels. Arm 2's upper ends are +0.0005, +0.0099 and +0.0008, so none is below 0. The interval contains 0 at
  every level. The producer's `outcome.statements` carries no D8 statement.
- **Arm 3 below arm 1 (not a verdict).** Arm 3's upper end is below 0 at 0.6 (−0.0003) and at 0.03 (−0.0003). N is
  one-sided, so this does not change N, and the registration has no rule for it. This page draws no conclusion from it.

## 2. DESCRIPTIVE (registered as descriptive, §11; never a verdict; no claims)

`scores.json` labels this block "DESCRIPTIVE (never a verdict)", with `endpoint_no_advantage_claimed` = true.
**No endpoint advantage is claimed for any arm.**

**Producer:** `src/trackL.py` `descriptive` (called by `score`) → `results/trackL/scores.json`, key `DESCRIPTIVE.arms`.

### 2.1 Matched steps, step costs and convergence

**Source:** `steps_to`, `step_cost_vs_arm1` and `convergence`.
- Steps are shown as median [min–max] over 40 seeds.
- The step cost is the per-seed ratio to arm 1 on the same seed, shown as median [q25, q75].
- The "stop" row gives the steps to the stop. Its cost is the per-seed ratio of total steps.

| level | arm 1 `std` steps | arm 2 `out16` steps | arm 3 `glob16` steps | arm 2 step cost | arm 3 step cost |
|---|---|---|---|---|---|
| 0.6 | 15 [15–20] | 20 [15–20] | 110 [95–130] | ×1.33 [1.00, 1.33] | ×7.00 [6.33, 7.33] |
| 0.3 | 1,751 [1,619–1,969] | 2,349 [2,129–2,442] | 9,313 [8,777–9,881] | ×1.32 [1.26, 1.37] | ×5.32 [5.11, 5.42] |
| 0.03 | 6,032 [5,151–7,205] | 6,152 [5,686–6,528] | 20,530 [19,734–21,358] | ×1.00 [0.98, 1.09] | ×3.40 [3.26, 3.56] |
| stop (BCE ≤ 0.002) | 7,422 [6,658–8,272] | 8,437 [8,110–9,499] | 42,691 [41,854–44,414] | ×1.15 [1.10, 1.19] | ×5.81 [5.48, 5.93] |

- **Stop reasons** (`convergence.stop`): loss 40, cap 0 and non-finite 0, in each of the three arms.
- **Arm 2's realised step cost** is ×1.00 to ×1.33 at the levels (medians), as in the exploration (×1.33, ×1.34,
  ×0.98; registration §13).
- **Checks are discrete.** At 0.6 the checks fall every 5 steps, so 15 against 20 steps is one check apart.

### 2.2 Secondary *rev* and descriptive *orig*: paired Δ vs arm 1 with the registered bootstrap

**Source:** `bootstrap_rev_DESCRIPTIVE` and `bootstrap_orig_DESCRIPTIVE`. These use the same bootstrap as the criteria.
They are labelled descriptive and carry no criterion.

| arm | test | level | median Δ | 95% interval | seeds up/down |
|---|---|---|---|---|---|
| 2 `out16` | rev | 0.6 | −0.0003 | [−0.0082, +0.0008] | 19/20 |
| 2 `out16` | rev | 0.3 | +0.0082 | [−0.0041, +0.0240] | 26/14 |
| 2 `out16` | rev | 0.03 | −0.0191 | [−0.0250, −0.0089] | 11/29 |
| 3 `glob16` | rev | 0.6 | −0.0051 | [−0.0135, +0.0026] | 16/24 |
| 3 `glob16` | rev | 0.3 | −0.0061 | [−0.0270, +0.0069] | 17/22 |
| 3 `glob16` | rev | 0.03 | −0.0232 | [−0.0367, −0.0138] | 8/32 |
| 2 `out16` | orig | 0.6 | +0.0010 | [−0.0010, +0.0046] | 21/17 |
| 2 `out16` | orig | 0.3 | −0.0010 | [−0.0031, +0.0031] | 16/23 |
| 2 `out16` | orig | 0.03 | +0.0071 | [+0.0028, +0.0110] | 30/9 |
| 3 `glob16` | orig | 0.6 | +0.0010 | [−0.0033, +0.0056] | 22/17 |
| 3 `glob16` | orig | 0.3 | −0.0026 | [−0.0082, +0.0015] | 16/24 |
| 3 `glob16` | orig | 0.03 | +0.0112 | [+0.0064, +0.0151] | 31/9 |

n_both is 40 in every cell.

### 2.3 Values per arm at the matched points and at the end of training

**Source:** `values`. Each cell is the median [q25, q75] over 40 seeds.
- *rand*, *rev* and *orig* are test accuracies.
- "flip 20%" and "pred 80%" are train accuracy on the flipped and the predictive training points.
- ‖v‖₂ is the output weight norm, and ‖W₂‖_F is the second hidden layer's Frobenius norm.

| measure | level | arm 1 `std` | arm 2 `out16` | arm 3 `glob16` |
|---|---|---|---|---|
| rand | 0.6 | 0.4908 [0.4886, 0.4969] | 0.4893 [0.4876, 0.4923] | 0.4895 [0.4872, 0.4918] |
| rand | 0.3 | 0.6469 [0.6342, 0.6602] | 0.6500 [0.6423, 0.6577] | 0.6449 [0.6397, 0.6503] |
| rand | 0.03 | 0.6849 [0.6770, 0.6925] | 0.6816 [0.6744, 0.6853] | 0.6773 [0.6722, 0.6802] |
| rand | end | 0.6860 [0.6820, 0.6904] | 0.6844 [0.6791, 0.6878] | 0.6783 [0.6735, 0.6819] |
| rev | 0.6 | 0.0355 [0.0268, 0.0546] | 0.0293 [0.0199, 0.0449] | 0.0337 [0.0249, 0.0409] |
| rev | 0.3 | 0.3704 [0.3438, 0.4054] | 0.3765 [0.3524, 0.3994] | 0.3617 [0.3495, 0.3786] |
| rev | 0.03 | 0.4796 [0.4584, 0.4987] | 0.4658 [0.4457, 0.4820] | 0.4526 [0.4443, 0.4628] |
| rev | end | 0.4740 [0.4680, 0.4777] | 0.4681 [0.4635, 0.4726] | 0.4533 [0.4495, 0.4593] |
| orig | 0.6 | 0.9816 [0.9717, 0.9872] | 0.9844 [0.9784, 0.9898] | 0.9819 [0.9754, 0.9852] |
| orig | 0.3 | 0.9390 [0.9304, 0.9440] | 0.9383 [0.9323, 0.9455] | 0.9355 [0.9305, 0.9393] |
| orig | 0.03 | 0.9033 [0.8944, 0.9092] | 0.9099 [0.9032, 0.9165] | 0.9138 [0.9097, 0.9175] |
| orig | end | 0.9097 [0.9050, 0.9122] | 0.9122 [0.9097, 0.9153] | 0.9138 [0.9096, 0.9158] |
| flip 20% | 0.6 | 0.0410 [0.0335, 0.0546] | 0.0355 [0.0294, 0.0476] | 0.0390 [0.0311, 0.0435] |
| flip 20% | 0.3 | 0.5225 [0.4958, 0.5651] | 0.5263 [0.4943, 0.5549] | 0.5277 [0.5097, 0.5390] |
| flip 20% | 0.03 | 0.9823 [0.9744, 0.9904] | 0.9855 [0.9815, 0.9921] | 0.9975 [0.9959, 0.9985] |
| flip 20% | end | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] |
| pred 80% | 0.6 | 0.9796 [0.9683, 0.9846] | 0.9826 [0.9757, 0.9883] | 0.9791 [0.9749, 0.9833] |
| pred 80% | 0.3 | 0.9663 [0.9574, 0.9703] | 0.9661 [0.9587, 0.9703] | 0.9653 [0.9624, 0.9689] |
| pred 80% | 0.03 | 0.9971 [0.9960, 0.9986] | 0.9989 [0.9981, 0.9995] | 0.9999 [0.9999, 1.0000] |
| pred 80% | end | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] |
| ‖v‖₂ | 0.6 | 0.6130 [0.5986, 0.6281] | 0.5802 [0.5679, 0.5873] | 0.6126 [0.6005, 0.6192] |
| ‖v‖₂ | 0.3 | 1.7280 [1.6950, 1.7776] | 0.8167 [0.8067, 0.8257] | 1.7934 [1.7832, 1.8116] |
| ‖v‖₂ | 0.03 | 3.8904 [3.8282, 3.9802] | 1.4369 [1.4215, 1.4461] | 3.5425 [3.5240, 3.5626] |
| ‖v‖₂ | end | 4.2579 [4.1696, 4.3316] | 1.6081 [1.6011, 1.6239] | 4.3724 [4.3655, 4.3780] |
| ‖W₂‖_F | 0.6 | 9.2395 [9.2264, 9.2523] | 9.2416 [9.2285, 9.2533] | 9.2398 [9.2268, 9.2522] |
| ‖W₂‖_F | 0.3 | 9.4065 [9.3993, 9.4210] | 9.5631 [9.5506, 9.5803] | 9.3963 [9.3822, 9.4066] |
| ‖W₂‖_F | 0.03 | 10.2333 [10.1944, 10.2759] | 10.7265 [10.6926, 10.7587] | 9.8831 [9.8671, 9.8966] |
| ‖W₂‖_F | end | 10.4153 [10.3762, 10.4514] | 11.1156 [11.0968, 11.1476] | 10.2076 [10.1962, 10.2194] |

### 2.4 Paired Δ vs arm 1 for every measure (medians, seeds up/down)

**Source:** `paired_vs_arm1`. n = 40 in every cell. The level rows for *rand* repeat §1's point medians.

| measure | level | arm 2 − arm 1 | arm 3 − arm 1 |
|---|---|---|---|
| rand | 0.6 | +0.0000 (17/19) | −0.0018 (12/26) |
| rand | 0.3 | +0.0028 (22/18) | −0.0038 (14/24) |
| rand | 0.03 | −0.0036 (16/24) | −0.0079 (13/26) |
| rand | end | −0.0031 (15/24) | −0.0089 (4/35) |
| rev | 0.6 | −0.0003 (19/20) | −0.0051 (16/24) |
| rev | 0.3 | +0.0082 (26/14) | −0.0061 (17/22) |
| rev | 0.03 | −0.0191 (11/29) | −0.0232 (8/32) |
| rev | end | −0.0061 (15/25) | −0.0196 (2/38) |
| orig | 0.6 | +0.0010 (21/17) | +0.0010 (22/17) |
| orig | 0.3 | −0.0010 (16/23) | −0.0026 (16/24) |
| orig | 0.03 | +0.0071 (30/9) | +0.0112 (31/9) |
| orig | end | +0.0036 (29/10) | +0.0051 (30/9) |
| flip 20% | 0.6 | −0.0005 (17/21) | −0.0050 (16/24) |
| flip 20% | 0.3 | +0.0147 (24/16) | +0.0002 (20/19) |
| flip 20% | 0.03 | +0.0045 (24/15) | +0.0152 (39/1) |
| flip 20% | end | +0.0000 (7/0) | +0.0000 (7/0) |
| pred 80% | 0.6 | −0.0001 (19/20) | +0.0017 (21/19) |
| pred 80% | 0.3 | −0.0017 (17/23) | −0.0004 (19/21) |
| pred 80% | 0.03 | +0.0017 (27/11) | +0.0027 (40/0) |
| pred 80% | end | +0.0000 (0/0) | +0.0000 (0/0) |
| ‖v‖₂ | 0.6 | −0.0282 (0/40) | +0.0039 (27/13) |
| ‖v‖₂ | 0.3 | −0.9081 (0/40) | +0.0568 (34/6) |
| ‖v‖₂ | 0.03 | −2.4677 (0/40) | −0.3594 (1/39) |
| ‖v‖₂ | end | −2.6484 (0/40) | +0.1164 (33/7) |
| ‖W₂‖_F | 0.6 | +0.0023 (22/18) | +0.0003 (27/13) |
| ‖W₂‖_F | 0.3 | +0.1559 (40/0) | −0.0136 (6/34) |
| ‖W₂‖_F | 0.03 | +0.4923 (40/0) | −0.3453 (0/40) |
| ‖W₂‖_F | end | +0.7088 (40/0) | −0.2007 (0/40) |

**Read descriptively.**
- Arm 2's output weight is smaller at every level (0/40 seeds up). Its ‖W₂‖_F is larger from BCE 0.3 on (40/0).
- This repeats the exploration's observation (registration §13) that the hidden layers carry the scale when the
  readout is slowed.
- None of this is a verdict, and none of it is an endpoint claim.

## 3. The output-multiplier (α) follow-up (EXPLORATORY; motivated by the Track L null)

This section is the one planned in registration §12 (there titled "The α follow-up").
- **Source:** commit df2b365, `results/designs/trackL_alpha_explore/` (`README.md`, `A_tables.md`, `A_summary.json`).
  These files are hashed in the Track L manifest.
- **Producer:** `results/designs/trackL_alpha_explore/A_summary.py` → `A_tables.md` and `A_summary.json`. That script
  read `explore_main.jsonl` and `explore_pilot.jsonl` (produced by `A_explore.py`).
- **No new runs** were made for this page, and no number here is recomputed.

**Status, plainly.** The follow-up was run **after** the L2 exploration (`trackL_explore/`, b08843d) showed the null,
and **before** the Track L registration.
- It was not registered, and it carries **no verdict**.
- No Track L criterion, gate or validity rule reads it, and it changes no Track L verdict.
- Its seeds are exploration seeds **2,994,000–2,994,007**, which were never registered.

**What it varied.** It used f = α·g, where g is the same MLP on the same data with the same optimiser.
- **fix:** lr 0.01 at every α.
- **fs:** lr 0.01/α².
- **glob16:** the fix lr divided by 16 everywhere. This is the analogue of N.

**Result: no effect.** A smaller α does not increase *rand* accuracy at matched training loss. The estimated power of an
R-type criterion is **0.00** at n = 20, 40 and 80 for every α arm.

Paired Δ*rand* vs α = 1 on the same seed, shown as median (seeds up/down). **Source:** `A_tables.md`, "Paired
differences vs alpha 1", which equals `A_summary.json` `vs_alpha1` (4 decimals as written by the producer).

| mode, α | n | Δrand 0.6 | Δrand 0.3 | Δrand 0.03 | step cost 0.6 / 0.3 / 0.03 |
|---|---|---|---|---|---|
| fix, 4 (contrast) | 8 | +0.0166 (7/1) | −0.0107 (3/5) | −0.0054 (3/5) | ×0.333 / ×0.847 / ×1.137 |
| fix, 0.5 | 8 | −0.0013 (2/6) | +0.0043 (4/4) | +0.0010 (4/4) | ×2.0 / ×1.181 / ×1.0 |
| fix, 0.25 | 8 | −0.0023 (1/5) | −0.0028 (2/6) | −0.0079 (3/5) | ×4.0 / ×1.537 / ×1.104 |
| fix, 0.1 | 8 | −0.0036 (1/7) | −0.0041 (3/5) | −0.0071 (3/5) | ×9.667 / ×2.588 / ×1.514 |
| fix, 0.05 | 8 | −0.0046 (1/7) | −0.0097 (3/5) | −0.0074 (2/6) | ×19.333 / ×3.957 / ×2.338 |
| fix, 0.01 | 4 | −0.0059 (0/4) | −0.0043 (0/3) | −0.0036 (0/4) | ×104.833 / ×11.852 / ×6.803 |
| fs, 4 | 8 | +0.0074 (6/2) | −0.0074 (3/5) | −0.0043 (3/5) | ×1.333 / ×1.834 / ×1.345 |
| fs, 0.5 | 8 | −0.0003 (3/4) | +0.0018 (4/4) | −0.0079 (3/5) | ×1.0 / ×1.061 / ×1.218 |
| fs, 0.25 | 8 | +0.0005 (4/4) | +0.0020 (4/4) | −0.0196 (1/7) | ×0.667 / ×1.228 / ×1.957 |

**Other blocks from the same producer:**
- **Slope on log₁₀ α** (fix mode, std; `A_summary.json` `monotone`; `A_tables.md`): the median per-seed slope per
  decade is +0.0068, +0.0034 and +0.0021 at 0.6, 0.3 and 0.03. The seeds negative/positive are 0/8, 2/6 and 1/7. An
  account in which a smaller α helps predicts a negative slope.
- **Analogue of N** (glob16 vs std at the same α; `glob_vs_std`): Δ*rand* at 0.6 / 0.3 / 0.03 was:

  | α | Δ*rand* 0.6 / 0.3 / 0.03 | step cost |
  |---|---|---|
  | 1 | −0.0036 / −0.0056 / −0.0061 | ×7.0 / ×4.916 / ×3.404 |
  | 0.25 | −0.0005 / −0.0031 / −0.0128 | ×10.883 / ×9.679 / ×8.293 |
  | 0.1 | −0.0015 / +0.0005 / −0.0117 | ×12.955 / ×11.834 / ×11.623 |

  The seeds were 5 per α.
- **Power of R** (`power`): P(R) is 0.00 at n = 20, 40 and 80 for fix α 0.5, 0.25, 0.1, 0.05 and 0.01, and for fs α 0.5
  and 0.25.

**Limits** (from its README):
- It used 4–8 seeds per main cell (5 for glob16, and 1–3 for fs α ≤ 0.1).
- It covers one p (0.8), one width (256), ReLU and SGD.
- The fs mode collapses to a constant output at α ≤ 0.1 (lr ≥ 1), so the fs comparison exists only for α ≥ 0.25.

It is EXPLORATORY and descriptive only.

## 4. Provenance

| table / statement on this page | producer | committed output (commit) |
|---|---|---|
| §1 verdicts, intervals, n_both, reach, validity, outcome, statements | `src/trackL.py` `score` (`score_tables`, `criterion_R`, `criterion_N`, `outcome`) | `results/trackL/scores.json` (a36c702) |
| §1 the 120 runs, finiteness, stop reasons | `src/trackL.py` `run` | `results/trackL/runs.jsonl` (a36c702); counts as summarised in `scores.json` |
| §1 the §15 exploration-seed upper end 0.0219 | `src/trackL.py` `freeze` | `results/trackL/frozen.json` (fc31e04; in the manifest at a8ab6db) |
| §1 N power 0.98 at n = 40 | `trackL_explore/L_power.py` | `results/designs/trackL_explore/power_sgd.log` (b08843d) |
| §2.1–2.4 descriptive tables | `src/trackL.py` `descriptive` (called by `score`) | `results/trackL/scores.json` key `DESCRIPTIVE` (a36c702) |
| §2.1 exploration step costs ×1.33 / ×1.34 / ×0.98, §2.4 reference | — (quoted) | `results/trackL_registration.md` §13, from `trackL_explore/L_tables.md` (b08843d) |
| §3 all numbers | `results/designs/trackL_alpha_explore/A_summary.py` | `A_tables.md`, `A_summary.json`, README (df2b365) |

**Display rounding.**
- §1–2 values were rounded from the unrounded floats in `scores.json`: 4 decimals for accuracies, differences and
  norms; 2 decimals for step ratios; whole numbers for steps.
- The interval ends in the criteria table are the same rounded values, and the verdicts use the unrounded ones.
- §3 values are as written by `A_summary.py`: 4 decimals for differences, 3 for step costs.

**What this page did not change.**
- No criterion, registered file, run, score or manifest entry was changed.
- Nothing under `paper/` was edited.
- Nothing in this page is post hoc except the labelled §3, which the registration planned (§12) and which predates the
  registration.
