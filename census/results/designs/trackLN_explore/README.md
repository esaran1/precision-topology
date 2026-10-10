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
