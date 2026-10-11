# Track LN: Track L with a LayerNorm (no affine) before the readout (EXPLORATORY, NOT a registration, no design page)

Requested by the author 2026-10-10. Exploration only: no registered or pilot seed exists, and none of the seeds
below may ever be registered. Whether to register is the author's decision, later, and only if both gates below hold.

## Setting (the author's request)

Identical to Track L (`src/trackL.py`, registration a8ab6db; imported READ-ONLY for the data construction, the data
check, the check schedule and the matching rule; nothing registered is modified) except for ONE change:

    MLP 6144 -> 256 -> ReLU -> 256 -> ReLU -> LayerNorm(256, elementwise_affine=False, default eps) -> Linear(256, 1)

so that the readout alone governs the output scale (the LayerNorm output has ‖h‖₂ ≤ √256 = 16, hence
|f| ≤ 16‖v‖₂ + |b|). Everything else as Track L: dominoes p = 0.8, construction seed 20261009, on-disk caches only
(SHA-256 verified by `trackL.verify_data`; the dataset must hash to 6867910b…332f); PyTorch default init drawn inside
`torch.random.fork_rng` after `torch.manual_seed(seed)`; BCE with logits; SGD momentum 0.9, lr 0.01, batch 128, order
from numpy `default_rng(seed)`; arms 1 std / 2 lr ÷16 on the output WEIGHT only (bias at full lr) / 3 lr ÷16 everywhere;
matched train-BCE levels {0.6, 0.3, 0.03} (first check at or below, no interpolation); stop at train BCE ≤ 0.002 or
the cap 200,000; checks at step 0, every 5 steps to 500, then every max(5, ⌊0.02 t⌋).

## Gate rules, written BEFORE any training (2026-10-10)

### Gate (2): the setting still relies on MNIST early and CIFAR later — Track L's L2 gate rule, verbatim, on arm 1

(Copied from `../trackL_explore/README.md`, applied to the arm-1 (std) runs at p = 0.8; H(0.8) = 0.5004.)

The author's L2 gate asks whether standard training relies mainly on the MNIST feature early and on the CIFAR feature
increasingly later. Operationalised on arm-1 (std) runs at a given p, with H(p) = −p ln p − (1−p) ln(1−p) the
MNIST-only Bayes floor of the training BCE (0 at p = 1):
- early point ℓ_A = the first check with train BCE ≤ H(p) + 0.10; late point ℓ_B = the first check with train BCE ≤ 0.02;
- **PASS** if the median MNIST-randomised accuracy at ℓ_A is ≤ 0.60 **and** the median paired increase from ℓ_A to ℓ_B
  is ≥ 0.10 with ≥ 80% of seeds increasing;
- **never uses CIFAR**: median randomised accuracy at the end of training ≤ 0.60;
- **uses CIFAR from the start**: median randomised accuracy at ℓ_A ≥ 0.70.

### Gate (1): the readout now controls the output scale — PROPOSED operationalisation, FOR THE AUTHOR TO APPROVE OR CHANGE

The author's wording: the step cost of output-only slowing (arm 2) rises WELL ABOVE Track L's 1.33×.

**PROPOSED rule.** Step cost of a seed at a level = (steps of arm 2 to the level) / (steps of arm 1 to the same level),
same seed (matched checks as above). **Gate (1) holds if the median step cost is ≥ 2.0 at EACH of the three levels
{0.6, 0.3, 0.03}** (a seed that does not reach a level in both arms is left out of that level and counted in the report).

Reasoning:
- Track L's registered arm-2 step costs (`results/trackL/scores.json`, 40 seeds) are medians ×1.33 / ×1.32 / ×1.00 at
  0.6 / 0.3 / 0.03, with per-seed maxima ×1.33 / ×1.51 / ×1.19. A median of 2.0 is above EVERY Track L seed at every
  level, i.e. outside Track L's whole distribution, and 1.5× its largest median; it is far below the ÷16 ceiling
  (×16 if the readout were the only path to scale) and below Phase 2B's ×7–16, so it asks for "well above", not "fully
  rate-limited".
- All three levels, not one: Track L's ×1.33 at 0.6 is partly the check grid (arm 1 reaches 0.6 at 15–20 steps and
  checks are 5 steps apart, so a ratio there moves in steps of about 1/3); requiring 0.3 and 0.03 too means the
  readout must govern the scale over the whole run, which is where the lever would have to act.
- Step cost is necessary, not sufficient: Track L's exploration finding 3 (output weight ×0.01 at init: ÷16 cost ×3.25
  at 0.6, yet Δrand null). So no Δrand threshold is part of gate (1); Δrand (arm 2 − arm 1, same seed, rand accuracy)
  is reported alongside, DESCRIPTIVELY, with its per-seed signs, as is the readout's ‖v‖₂ and the mean |f|.
- Readings the author may prefer instead (not used here unless approved): a per-seed version (≥ 80% of seeds with cost
  ≥ 2.0 at each level), or a threshold at 0.3 and 0.03 only. Both are computed and shown in the report for comparison;
  the verdict on gate (1) uses the rule in bold above.

Everything else (arm 3's cost and Δrand, rev / orig accuracy, power) is DESCRIPTIVE.

## Seeds

Exploration range **2,995,000–2,995,099** only (scanned before use: `LN_seedscan.py` → `LN_seedscan.json`; text in
src, tests, results, paper, notes, independent, data, dist; range literals in src/ and tests/; parquet and JSONL seed
columns; disjoint from 2,991,000–099, 2,992,000–039, 2,993,000–009 and 2,994,000–007). Used: 2,995,000–007,
8 seeds × 3 arms = 24 runs, after one verification run (std, seed 2,995,000: timing and RSS).

## Machine rules

One process, nice 15, one thread (OMP/MKL/OPENBLAS/VECLIB = 1, `torch.set_num_threads(1)`), memory gate before every
run inside the driver (memory_pressure free ≥ 25% and swap free ≥ 500 MB, else wait; disk ≥ 20 GB, else stop; logged
to `memory_gate.log`), stop at 1.5 GB RSS, detached, resumable per (seed, arm). No global numpy/torch RNG state; no
download.

## What was run (2026-10-10, 19:54–20:20; everything below was added AFTER training, the rules above are unchanged)

The README above (gate rules), the seed scan, `LN_explore.py`, `LN_summary.py` and the driver were committed BEFORE any
training (a485d0f; gated: memory gate, suite 1411 passed + 1 skipped exit 0, certificates 33 passed exit 0, ledger 0
findings: `LN_suite_main_pre.log`, `LN_suite_certs_pre.log`, `LN_ledger_pre.log`).

| file | content |
|---|---|
| `LN_seedscan.py` → `LN_seedscan.json`, `LN_seedscan.log` | range 2,995,000–099 clean: 0 text hits, 715 range literals with no overlap, 72 parquet seed columns and 102 JSONL files with seed keys (105 checked) with 0 values in range, disjoint from the four named ranges |
| `verify.log` | verification run, std seed 2,995,000: dataset hash 6867910b…332f reproduced, 9,131 steps, 89 s, max RSS 1.13 GB |
| `main.log` | the 24 runs (seeds 2,995,000–007 × arms std/out16/glob16; the verification row was reused, 23 new runs), one process, 19:55–20:20 |
| `runs.jsonl` | one row per (seed, arm), full check records |
| `memory_gate.log` | memory gate as a separate step before the scan, each suite, the job and every run (33 checks, all OK, 0 waits) |
| `LN_summary.py` → `LN_tables.md`, `LN_summary.json`, `LN_summary.log` | all tables below |
| `suite.sh` → `LN_suite_*_final.log`, `LN_ledger_final.log` | the gated suite before the exploration commit |

All 24 runs stopped at train BCE ≤ 0.002, all finite, none near the cap; 26 min of run time (median 64 s std, 69 s
out16, 48 s glob16); peak RSS 1.13 GB.

## Findings (8 exploration seeds; numbers from `LN_tables.md`)

1. **Gate (1) (PROPOSED rule) does NOT hold.** Median arm-2/arm-1 step cost ×1.00 / ×1.00 / ×1.09 at BCE 0.6 / 0.3 /
   0.03 (seeds ≥ 2.0: 1/8, 0/8, 0/8); every alternative reading fails too. Track L had ×1.33 / ×1.32 / ×1.00: the
   LayerNorm did not raise the cost of output-only slowing.
2. **Gate (2) PASSES** (Track L's L2 rule on arm 1): median rand at ℓ_A (BCE ≤ 0.6004, 5–25 steps) 0.515; median
   increase ℓ_A → ℓ_B +0.168, 8/8 seeds increasing; end 0.682 (Track L exploration: 0.487 → +0.198, end 0.688).
3. **Δrand, arm 2 − arm 1 (DESCRIPTIVE):** +0.012 (6/2), +0.012 (6/2), +0.0005 (4/4) at 0.6 / 0.3 / 0.03; descriptive
   bootstrap intervals all contain 0 ([−0.024, +0.022], [−0.009, +0.026], [−0.016, +0.019]); per-seed SD 0.023–0.027.
4. **Arm 3 (÷16 everywhere) is FASTER than arm 1** in this network: ×0.50 / ×0.52 / ×0.64 (Track L: ×7.0 / ×5.3 / ×3.4);
   Δrand −0.008 (4/4), +0.011 (5/3), +0.007 (6/2).
5. **Output scale (DESCRIPTIVE).** The readout norm ‖v‖₂ at BCE 0.03 is 2.08 (std) and 1.00 (out16), against Track L's
   3.89 and 1.44. Mean |f| there is 8.5 (std) and 6.5 (out16), well under the LayerNorm bound 16‖v‖₂ (33 and 16). So the
   hidden layers still change |f|, by aligning the normalised features with v, and the readout norm is not the
   bottleneck. This is an observation from 8 seeds, not a tested mechanism.
6. **Power (DESCRIPTIVE, from 8 exploration seeds, Track L's L_power.py method, δ = 0.02):** R (arm 2) P = 0.04 / 0.04
   / 0.05 at n = 20 / 40 / 80; N (arm 3) P = 0.21 / 0.31 / 0.60 (Track L's exploration: R 0.00, N 0.89/0.98/1.00).

Track LN is not registered and no registration is proposed here. Gate (1) is the precondition the author named, and
by the proposed rule it fails; whether to register, and whether the gate-(1) rule itself stands, is the author's call.

## Suite before the exploration commit

`suite.sh final`: memory gate first (logged); main 1411 passed, 1 skipped, exit 0 (`LN_suite_main_final.log`);
certificates 33 passed, exit 0 (`LN_suite_certs_final.log`); `src.verify_ledger` 0 findings, exit 0
(`LN_ledger_final.log`). Exploration files under results/designs/ have no PRODUCERS entries (as for Track L's
explorations). Not pushed.
