# Track L follow-up: a fixed output multiplier α on MNIST-CIFAR dominoes (EXPLORATORY, NOT a registration; MOTIVATED BY THE TRACK L NULL)

Started 2026-10-09 after the L2 exploration (`../trackL_explore/`, commit b08843d) found that slowing the output
layer's learning rate had no effect on feature reliance, because the hidden layers carry the output scale. This
follow-up tests the account's actual variable, the output scale, directly: f(x) = α·g(x), g the same MLP. It is
exploration only. There is no design page and no registration, and none of these seeds may be registered.

## Plain answer

**No. A smaller α does not increase MNIST-randomised accuracy at matched training loss.** At α = 0.5 the paired
differences are within ±0.004 (4/4 seeds up/down at 0.3 and 0.03). At α ≤ 0.25 they are slightly NEGATIVE at all
three levels: medians −0.002 to −0.010 under a fixed lr, with 5–7 of 8 seeds down at the first level. In this
exploration, smaller α means very slightly MORE reliance on MNIST (less use of CIFAR), not less. Every effect is
below 0.01 except one: lr/α² at α = 0.25 gives −0.020 at BCE 0.03 (1/7 seeds up/down). The estimated power of an
R-type criterion (smaller α > α = 1 at all three levels) is **0.00 at n = 20, 40 and 80** for every α arm. The L2
gate (MNIST early, CIFAR later) **passes in every α arm that trains**. Slowing every learning rate by 16 at α < 1
(the analogue of N) does not help either: Δrand ranges from −0.013 to +0.001.

Why there is no effect: at a matched loss the output |f| is fixed by the loss, whatever α is. A small α only makes
the network grow g = f/α, and it does this mostly through the readout and the second layer. At BCE 0.03 (fixed lr),
going from α = 1 to α = 0.01 raises ‖v‖ from 3.9 to 17.1 and ‖W2‖ from 10.2 to 19.5, while mean |f| goes from 9.4 to
6.3. The extra scale is paid in steps (×6.8 to BCE 0.03 at α = 0.01), and the network ends at the same solution.

## What was held fixed

These are the same as in L2: dominoes p = 0.8, construction seed 20261009 (`../trackL_explore/L_data.py`, on-disk
caches only, SHA-256 verified by `L_data.verify` on every build, no download); MLP 6144 → 256 → 256 → 1, ReLU, PyTorch
default init (g at init is the same function for every α on a seed, because the multiplier has no parameters and
the draw happens before it); SGD momentum 0.9, no weight decay, batch 128, the same minibatch order per seed; BCE on
f; the same check schedule; the stop at train BCE ≤ 0.002 or the step cap; matched train-BCE levels {0.6, 0.3, 0.03}
(first check at or below the level, no interpolation). Measures: MNIST-randomised test accuracy (primary), reversed
(secondary), original (descriptive).

The learning-rate modes vary only α and the lr:
- **fix**: lr = 0.01 for every α. The parameter-space step is held fixed, and the function-space step at init scales
  as α². We report the step cost rather than correcting for it.
- **fs**: lr = 0.01/α². The first-order function-space step at a given parameter point is held equal to α = 1's.
- **glob16**: the fix mode's lr divided by 16 on every parameter (the analogue of N), compared with the same α on the
  same seed.

The reference is α = 1 with lr 0.01, which is the same run in both modes. The α = 1 reference reproduces L2 on fresh
seeds: rand at ℓ_A 0.491, end 0.686 (L2: 0.487, 0.688).

## Seeds

The fresh exploration seeds are **2,994,000–2,994,099**, and those used were 2,994,000–2,994,007. `A_seedscan.py` →
`A_seedscan.json` (clean) checked them with L2's method, on a text scan of src, tests, results, paper, notes,
independent, data and dist, the range literals and the parquet seed columns. They are disjoint from L2's
2,991,000–099, the proposed registered 2,992,000–039 and the proposed pilot 2,993,000–019.

## What was run (2026-10-09, 17:29–19:23)

| file | content |
|---|---|
| `explore_pilot.jsonl`, `pilot.log` | seed 2,994,000, cap 20,000: fix α ∈ {4, 1, 0.5, 0.25, 0.1, 0.05, 0.01}; fs α ∈ {4, 0.5, 0.25, 0.1, 0.05, 0.01} |
| `explore_main.jsonl`, `main.log` | 8 seeds: fix α ∈ {4, 1, 0.5, 0.25, 0.1} (cap 40,000) and 0.05 (cap 60,000); fs α ∈ {4, 0.5, 0.25} (cap 40,000). fs α 0.1 × 2 more seeds (cap 3,000). glob16 at α ∈ {1, 0.25, 0.1} × 5 seeds (cap 150,000/150,000/200,000). fix α 0.01 × 4 seeds (cap 120,000) |
| `main_failed_launch.log` | first launch: zsh `seq` printed the seed as 2.994e+06 and the script refused it at the argument parse, so no training ran; relaunched with `{a..b}` |
| `A_summary.py` → `A_tables.md`, `A_summary.json`, `A_summary.log` | all tables, the monotonicity check and the power |
| `memory_gate.log` | the memory gate, logged as a separate step before every run (108 OK, 0 waits) |
| `suite.sh` → `A_suite_main.log`, `A_suite_certs.log` | the gated two-process test suite before the commit |

The run took 93 runs in main plus 13 in the pilot, 105 minutes of run time. It used one process, nice 15 and one
thread. The driver waits when another python process is over 1 GB RSS (it never did). Peak RSS was 1.08 GB. Seed
2,994,000 was run in both files, and the shared prefix of the records is bit-identical (determinism check). The main
batch's row is the one used.

## Findings (numbers from `A_tables.md`)

1. **Paired Δ MNIST-randomised accuracy vs α = 1 (same seed), at BCE 0.6 / 0.3 / 0.03, median (up/down):**

   | arm | n | 0.6 | 0.3 | 0.03 | step cost 0.6 / 0.3 / 0.03 |
   |---|---|---|---|---|---|
   | fix α 4 (contrast) | 8 | +0.017 (7/1) | −0.011 (3/5) | −0.005 (3/5) | ×0.33 / ×0.85 / ×1.14 |
   | fix α 0.5 | 8 | −0.001 (2/6) | +0.004 (4/4) | +0.001 (4/4) | ×2.0 / ×1.18 / ×1.00 |
   | fix α 0.25 | 8 | −0.002 (1/5) | −0.003 (2/6) | −0.008 (3/5) | ×4.0 / ×1.54 / ×1.10 |
   | fix α 0.1 | 8 | −0.004 (1/7) | −0.004 (3/5) | −0.007 (3/5) | ×9.7 / ×2.59 / ×1.51 |
   | fix α 0.05 | 8 | −0.005 (1/7) | −0.010 (3/5) | −0.007 (2/6) | ×19.3 / ×3.96 / ×2.34 |
   | fix α 0.01 | 4 | −0.006 (0/4) | −0.004 (0/3) | −0.004 (0/4) | ×105 / ×11.9 / ×6.8 |
   | fs α 4 (lr 0.000625) | 8 | +0.007 (6/2) | −0.007 (3/5) | −0.004 (3/5) | ×1.33 / ×1.83 / ×1.35 |
   | fs α 0.5 (lr 0.04) | 8 | −0.000 (3/4) | +0.002 (4/4) | −0.008 (3/5) | ×1.0 / ×1.06 / ×1.22 |
   | fs α 0.25 (lr 0.16) | 8 | +0.001 (4/4) | +0.002 (4/4) | −0.020 (1/7) | ×0.67 / ×1.23 / ×1.96 |

   The reversed-test differences at 0.03 are negative and larger for every α < 1 arm, ranging from −0.005 to −0.037
   (they are mixed in sign at 0.3). Original-test accuracy at 0.03 moves by +0.001 to +0.008 under fix and by −0.002 to −0.011 under fs.

2. **The fs mode cannot be taken below α ≈ 0.25.** At lr/α² ≥ 1 (α = 0.1, 0.05, 0.01) SGD with momentum 0.9 first
   trains (α = 0.1 reaches BCE 0.6 on 3/3 seeds) and then collapses within 3,000 steps to a constant output: train BCE
   ln 2 = 0.693, every test accuracy 0.500, finite values. No α ≤ 0.1 fs run reaches BCE 0.3. The function-space-matched
   comparison therefore exists only for α ≥ 0.25.

3. **Monotonicity (fix, std; per-seed least-squares slope of rand on log10 α).** The slope is positive at every level
   (median per decade +0.007 / +0.003 / +0.002; seeds negative/positive 0/8, 2/6, 1/7). Larger α goes with slightly
   higher rand, which is the opposite of the account's direction. The medians at 0.03 are, by ascending α: 0.669,
   0.677, 0.682, 0.683, 0.684, 0.687 at α = 0.01 … 1, then 0.678 at α = 4. So the relation is not monotone in α, and
   α = 1 is the highest. The α = 4 point at 0.6 (+0.017, 7/1) is reached within 5–15 steps (the level is crossed almost
   at init), so it reflects the features at initialisation.

4. **The L2 gate holds in every α arm that trains**, including fix α = 0.01, all glob16 arms and fs α ≥ 0.25. The
   median rand at ℓ_A is 0.486–0.509, the median increase to ℓ_B is +0.168 to +0.197, and 100% of seeds go up.

5. **End of training (descriptive; stop at BCE 0.002).** The median end rand is 0.671–0.686 for every α arm that
   trains (α = 1: 0.686; fix α 0.01: 0.671; fs α 0.25: 0.672). The median end original accuracy is 0.897–0.920.
   glob16 α = 0.1 ends at its 200,000-step cap at BCE 0.0025 (0/5 reach 0.002, and every level is reached).

6. **Analogue of N (glob16 vs std, same α, same seed).** At α = 1 the differences are −0.004 (2/3) / −0.006 (1/3) /
   −0.006 (0/5) at ×7.0 / ×4.9 / ×3.4 steps, which reproduces L2's null. At α = 0.25 they are −0.001 / −0.003 / −0.013
   (1/4) at ×10.9 / ×9.7 / ×8.3. At α = 0.1 they are −0.002 / +0.001 / −0.012 at ×13.0 / ×11.8 / ×11.6. Slowing every
   rate never increases rand by more than 0.001.

7. **Power of R** (smaller α > α = 1 at all three levels, with the lower 95% bootstrap end of the median above 0;
   `../trackL_explore/L_power.py` method, resampling the per-seed paired differences): **0.00 at n = 20 / 40 / 80 for
   every arm** (fix α 0.5, 0.25, 0.1, 0.05, 0.01; fs α 0.5, 0.25). This follows from the median differences being
   ≤ 0 at one or more levels in every arm. The exploration seeds (n = 4–8) give no basis for an R-type criterion on α.

## Limits

These are exploration seeds only (8 per main cell; 4 for fix α 0.01; 5 for glob16; 1–3 for fs α ≤ 0.1). The results
cover one p (0.8), one width (256), ReLU and SGD. The step caps never bind before BCE 0.03. The fs failure at α ≤ 0.1
is a property of SGD with momentum 0.9 at lr ≥ 1, and other lr schedules were not tried. Everything here is
EXPLORATORY and motivated by the Track L null, and nothing in it is a registration.

## Suite before the commit

`suite.sh` ran with the memory gate first (logged), as two sequential processes with no outer nice, 19:24–19:33.
The main suite gave 1336 passed, exit 0 (`A_suite_main.log`). `tests/test_verify_certificates.py` gave 33 passed,
exit 0 (`A_suite_certs.log`).
