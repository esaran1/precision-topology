# Track L, step L2: exploration of the output-layer lever on MNIST-CIFAR dominoes (EXPLORATORY, NOT a registration)

Started 2026-10-09. Exploration seeds **2,991,000–2,991,099** only (never to be registered; scan `L_seedscan.json`).
Files: `L_data.py` (construction), `L_explore.py` (one run → one JSON line), `run.sh` (detached driver, memory gate
before every run), `gate.sh` (memory gate; log `memory_gate.log`), `L_seedscan.py` (seed-range scan).

## Gate rule, written BEFORE any exploration training (2026-10-09)

The author's L2 gate asks whether standard training relies mainly on the MNIST feature early and on the CIFAR feature
increasingly later. Operationalised on arm-1 (std) runs at a given p, with H(p) = −p ln p − (1−p) ln(1−p) the
MNIST-only Bayes floor of the training BCE (0 at p = 1):
- early point ℓ_A = the first check with train BCE ≤ H(p) + 0.10; late point ℓ_B = the first check with train BCE ≤ 0.02;
- **PASS** if the median MNIST-randomised accuracy at ℓ_A is ≤ 0.60 **and** the median paired increase from ℓ_A to ℓ_B
  is ≥ 0.10 with ≥ 80% of seeds increasing;
- **never uses CIFAR**: median randomised accuracy at the end of training ≤ 0.60;
- **uses CIFAR from the start**: median randomised accuracy at ℓ_A ≥ 0.70.

## What was run (2026-10-09, 14:41–16:32)

107 runs (69 min of run time; one process, nice 15, one thread; memory gate logged before every run in
`memory_gate.log`, 3 one-minute waits at 15:05–15:07 for swap; peak RSS 1.39 GB before the in-place construction of
`L_data.build` (15:13; identical runs, `explore_recheck.jsonl`), 1.09 GB after). All on p = 0.8 unless stated;
SGD (momentum 0.9, lr 0.01, batch 128) or Adam (torch defaults, lr 1e-3); MLP 6144 → W → W → 1, PyTorch default init;
every run stops at train BCE ≤ 0.002 or at its step cap. One tanh/Adam batch was stopped by hand at 16:10 (cap 400,000
too slow; relaunched with cap 60,000; the stopped run wrote nothing).

| file | content |
|---|---|
| `explore_timing.jsonl`, `timing.log` | first timed run (p 0.95, std, seed 2,991,000: 18 s) |
| `explore_gate.jsonl`, `gate.log` | std, ReLU, W 256, SGD: p ∈ {1.0, 0.95, 0.9, 0.8} × 5 seeds; p = 0.5 × 2 (MNIST uninformative: CIFAR-only reference) |
| `explore_lever.jsonl`, `lever1.log` | ReLU W 256 SGD: p 0.8 out4/out16/out64/glob16 × 5 seeds; p 0.9 out16/glob16 × 3 |
| `explore_lever_adam.jsonl`, `lever_adam.log` | ReLU W 256 Adam: p 0.8 std/out16/out64/glob16 × 5 |
| `explore_probe.jsonl`, `probe.log` | SGD out1024 × 3; width 16 (SGD std/out16, Adam std/out16/glob16) × 3 |
| `explore_tanh.jsonl`, `tanh.log`, `tanh_sgd.log` | tanh W 256: Adam std/out16/glob16 × 2 (cap 60,000); SGD std/out16 × 2 |
| `explore_vsmall.jsonl`, `vsmall.log` | ReLU W 256 SGD, output weight ×0.01 at init: std/out16/glob16 × 3 |
| `L_summary_*.log`, `L_tables.py` → `L_tables.md` | gate statistics and paired differences |
| `L_power.py` → `power_sgd.log`, `power_adam.log` | power of R and N (resampling the paired differences) |
| `suite.sh` | gated two-process test suite before the commit |

## Findings

1. **Gate (rule above): PASS at p = 0.9 and 0.8; not at 0.95 or 1.0.** Std, ReLU W 256, SGD, 5 seeds each:
   median MNIST-randomised accuracy at ℓ_A → paired increase to ℓ_B (seeds up): p 1.0 0.478 → +0.000 (1/5; end 0.480,
   never uses CIFAR, as Shah et al.); p 0.95 0.483 → +0.062 (5/5; below the 0.10 rule); p 0.9 0.483 → +0.109 (5/5);
   p 0.8 0.487 → +0.198 (5/5; end 0.688). Adam p 0.8: 0.499 → +0.164 (5/5). CIFAR-only reference (p 0.5): 0.76.
   The fixed MNIST-randomised permutation agrees with the label on 47.9% of test points, so a pure MNIST classifier
   scores 0.479 there (observed 0.478–0.480 at p = 1).
2. **The output-layer lever has no measurable effect in any explored setting** (`L_tables.md`): the largest
   |median Δrand| of any output-slowed arm at the levels {0.6, 0.3, 0.03} is 0.020, with no consistent sign
   (factors 4–1024; SGD and Adam; ReLU and tanh; width 256 and 16; default and ×0.01 output init).
3. **The readout is not rate-limiting.** Arm 2 costs ×0.9–1.5 steps to each level (÷16 SGD: ×1.33/1.34/0.98; ÷1024:
   ×1.33/1.48/1.27), against ×7–16 in Phase 2B. At BCE 0.03 the readout's ‖v‖₂ is 1.44 (÷16) vs 3.88 (std), yet the
   steps and the CIFAR use are the same: the hidden layers carry the scale. At PyTorch default init ‖v‖₁ ≈ 8.0 already
   (2B started at ≤ 0.5 s\*). With the output weight ×0.01 at init, ÷16 costs ×3.25 at 0.6 but Δrand stays null.
4. **Global arm.** ÷16 everywhere costs ×3.3–7 (SGD); Δrand −0.005 to +0.001 (SGD), but +0.010 to +0.014 at 0.3 and
   0.03 under Adam (5/0 seeds). The 2B-style cost-matched factor degenerates (arm 2's cost ≈ 1 → G_cm ≈ 1).
5. **Power** (`power_sgd.log`, `power_adam.log`; δ = 0.02, levels {0.6, 0.3, 0.03}): R for ÷16 is 0.00 at n = 20, 40
   and 80 (SGD and Adam); N for ÷16 global is 0.89/0.98/1.00 (SGD) and 0.15/0.31/0.56 (Adam). The ÷4 SGD arm shows
   P(R) 0.68–1.00 from medians of +0.0005 to +0.007 on 5 seeds: an artefact of resampling 5 seeds (a median of
   +0.0005 is one test image) and one of several factors tried; not a basis for choosing ÷4.

## Suite before the commit

`suite.sh` (memory gate first, logged; two processes; no outer nice): main 1336 passed, exit 0 (`L_suite_main.log`);
`tests/test_verify_certificates.py` 33 passed, exit 0 (`L_suite_certs.log`). A first attempt ran pytest under
`nice -n 15`: 3 failures, all `PermissionError` from registered code calling `os.nice(15)` (band_rd, ramp2), exit 1
(`L_suite_main_nice15.log`; certificates 33 passed, `L_suite_certs_nice15.log`). Design page:
`results/designs/trackL_dominoes_design.md`.

**Addendum (2026-10-09, author's decisions applied to the page; still a DRAFT).** The page was trimmed; the full
SHA-256 values are in `L_data_check.log`. Suite rerun before that commit (memory gate first, two processes, no outer
nice): main 1336 passed, exit 0 (`L_suite_main.log`); certificates 33 passed, exit 0 (`L_suite_certs.log`). The
b08843d suite logs are kept as `L_suite_main_b08843d.log` and `L_suite_certs_b08843d.log`.
