# Phase 2B exploration, addendum: global control, matched loss, converged budget (EXPLORATORY, NOT a registration)

Run 2026-10-02 for the design page `results/designs/phase2b_lever_design.md`. Same non-registered exploration seeds
**2,953,000–2,953,019** as `README.md` (range 2,953,000–2,953,099 never to be registered). No registered or pilot seed
drawn. Existing files in this folder are unchanged; this addendum adds `p2b_lever.py`, `run_lever.sh`,
`p2b_lever_power.py`, `runs_lever/`, `p2b_lever.json`, `p2b_lever_tables.md`, `p2b_lever_power_out16_glob.json` and
the logs `p2b_lever.log`, `p2b_lever_power.log`.

**Runs.** Adam, v3 init and protocol, every run a fixed 100,000 steps (no early stop); ρ₂ at every step; BCE (data loss,
the objective minus (λ/2)(‖W‖² + ‖c‖²)) at every step.
- `std` lr 0.01 everywhere; `out16` lr 0.01/16 on v, 0.01 on W, c, b; `out16b` lr 0.01/16 on v and b;
  `glob` lr 0.01/G everywhere, **G = 12.18** = median over the 20 seeds of steps-to-3 s\* (adam_r16 ÷ adam_r1) from the
  committed `runs/` (12.177).
- `std` and `out16` reproduce the committed `adam_r1` / `adam_r16` exactly (steps to 3 s\* and ρ₂ at 3 s\*: 0 difference).
- Not completed: `globb` (lr 0.01/13.0, the v+b analogue; 6/20, stopped when the session stalled) and `glob16`
  (lr 0.01/16 everywhere; stopped before any run finished). Neither enters any number below.
- Matched points are first passages (no interpolation); max relative overshoot at a matched scale 2.3% (std), ≤ 0.3%
  (others).

**Findings (median paired difference vs std, seeds up/down; full tables in `p2b_lever_tables.md`).**

| point | out16 Δρ₂ | out16 Δshuffled / Δreversed | glob Δρ₂ | glob Δshuffled / Δreversed |
|---|---|---|---|---|
| s = 1.25 s\* | +0.27 (20/0) | +0.12 (19/1) / +0.15 (19/1) | +0.007 (14/6) | +0.006 / +0.008 |
| s = 2 s\* | +0.25 (20/0) | +0.15 (20/0) / +0.25 (20/0) | −0.002 (9/11) | −0.002 / −0.004 |
| s = 3 s\* | +0.31 (20/0) | +0.16 (20/0) / +0.32 (20/0) | −0.043 (2/18) | −0.028 / −0.056 |
| BCE = 0.1 | +0.12 (20/0) | +0.039 (18/2) / +0.071 (18/2) | −0.005 (8/12) | −0.000 / −0.001 |
| BCE = 0.03 | +0.06 (20/0) | +0.025 (18/2) / +0.050 (18/2) | −0.018 (2/18) | −0.012 / −0.024 |
| BCE = 0.01 | +0.19 (20/0) | +0.068 (20/0) / +0.137 (20/0) | −0.008 (5/15) | −0.016 / −0.031 |
| arm 1's step to 3 s\* | −0.24 (0/20) | −0.17 / −0.28 (0/20) | −0.32 (0/20) | −0.20 / −0.30 |
| step 40,000 | −0.047 (3/17) | −0.016 / −0.033 (3/17) | −0.087 (3/17) | −0.015 / −0.030 |

1. **Global slowing gives no gain** at matched scale or matched loss: |median Δρ₂| ≤ 0.008 up to 2 s\* and at BCE 0.1,
   0.01; at 3 s\* and BCE 0.03 it is slightly *lower* than standard (−0.043, −0.018; 2/18 seeds). So the step count is
   not the lever; the output/hidden rate ratio is. Exact invariance (two-sided) does not hold at 3 s\*.
2. **Global cost:** lr ÷ 12.18 costs ×9.45 [9.07–9.72] steps to 3 s\* (not ×12.18: Adam's s-growth per step is not
   proportional to lr), against ×12.18 [11.18–13.41] for out16.
3. **Output-only slowing at matched loss:** smaller than at matched scale but positive in every cell (ρ₂ 20/0 at the
   three levels; shuffled 18/2–20/0). At a given loss the slowed run sits at a smaller scale (s at BCE 0.03: 7.4 vs
   10.4).
4. **out16b (v and b slowed):** similar at matched scale (Δρ₂ +0.18 to +0.24, 20/0) but weak at matched loss on the
   shifted accuracies (shuffled at BCE 0.05 / 0.03: 11/9, 11/9). Cost ×13.0.
5. **Converged budget:** at 40,000 and 100,000 steps every arm is near the slab (ρ₂ ≈ 0.74–0.84, shuffled ≥ 0.98);
   standard is *higher* in 17/20 seeds. The lever is a matched-scale / matched-loss effect, not an end-state one.
6. **Lasting onset** (ρ₂ ≥ q at every step from t_on to the end of the run): out16 18/20 above s\* (median 1.75 s\*;
   the two below: 0.61 and 0.88 s\*), identical whether the run ends at 3 s\*, 40,000 or 100,000 steps; out16b 20/20
   (1.89 s\*, 1 in [1, 1.25] s\*); std 20/20 at 40,000 (2.99 s\*), but with a 100,000-step end 10/20 standard runs
   lose and regain ρ₂ ≥ q at large s (onset median 60.9 s\*), so "lasting" depends on the end of the run.

**Compute.** One process, nice 15, one thread; memory gate (`gate.sh`) logged before every job (one 60 s wait at
08:32, swap free 310 MB). 86 runs 08:31–09:02 (16 s per run unloaded, up to 371 s under other load); peak RSS 0.38 GB.

**Proposed criteria on these seeds, and power** (`p2b_lever_power.py`, `p2b_lever_power80.py`; local generators only).
- Cells: matched scale {1.25, 2, 3}·s\* and BCE {0.1, 0.03, 0.01} × (ρ₂, shuffled, reversed). Statistic: 95% percentile
  bootstrap of the median paired difference (seeds resampled jointly; 10,000 resamples, `default_rng(20261002)`).
- On the 20 exploration seeds, all pass: R (out16, every lower end > 0; smallest lower end 0.016, shuffled at BCE 0.1),
  N (glob, every upper end < δ = 0.03 / 0.02 / 0.03 for ρ₂ / shuffled / reversed; binding cell ρ₂ at 1.25 s\*, upper
  end 0.028), O (out16 lasting onset above s\* in 18/20 ≥ 75%).
- Power (seeds resampled from the 20 paired differences; 300 simulations, inner bootstrap 1,000):

| scenario | n = 30 | 40 | 60 | 80 |
|---|---|---|---|---|
| as explored: P(all pass) | 0.63 | 0.78 | 0.90 | 0.96 |
| arm-2 effect halved, spread doubled: P(R) | 0.04 | 0.06 | 0.09 | — |
| arm-3 spread doubled: P(N) | 0.15 | 0.26 | 0.39 | — |
| arm-3 shifted to the margin: P(N) (should fail) | 0.00 | 0.00 | 0.00 | — |

- O alone, exact binomial P(≥ 75%): 0.9998 at n = 60 at the exploration rate 0.90; 0.99 at 0.85; 0.24 at its Wilson
  95% lower bound 0.70.

**Test suite** (before the commit; gated, two processes, nice 15, one thread; no src/ or tests/ change): main suite
917 passed, exit 0; `test_verify_certificates` 33 passed, exit 0 (logs `p2b_lever_suite_main.log`,
`p2b_lever_suite_certs.log`, kept in the session scratch directory).

## Second addendum (2026-10-02, after the author's decisions on the 2B page): cost-matched global arm, per-arm power

**Cost-matched factor, fixed from the committed runs before the confirmation** (`p2b_lever_cm.py`).
- r_g = median over the 20 seeds of t_glob(3 s\*)/t_std(3 s\*) = 9.4541 (lr ÷ 12.18); r_o = the same for out16 = 12.1773.
- Model: steps-to-3 s\* ratio = G^a through (1, 1) and (12.18, r_g), so a = ln r_g/ln 12.18 = 0.8985.
- G_cm = 12.18^(ln r_o/ln r_g) = 16.1426 → **16.14**. (The linear rescaling 12.18·r_o/r_g would give 15.69.)
- Confirmation, `globcm` = lr 0.01/16.14 everywhere, 40,000 steps, 20 exploration seeds (`runs_lever/globcm_*`,
  `p2b_lever_cm.json`, log `p2b_lever_cm.log`): cost to 3 s\* ×11.78 [10.95–12.22] vs std; per seed 0.97 [0.92–1.01]
  of out16's. Paired Δρ₂ vs std: +0.004 (1.25 s\*, 15/5), −0.012 (2 s\*), −0.053 (3 s\*, 1/19); at BCE 0.1/0.03/0.01:
  −0.013 / −0.022 / −0.010. No gain anywhere; like glob, slightly lower at 3 s\*.

**Per-arm power** (`p2b_lever_power2.py` → `p2b_lever_power2.json`; seeds resampled from the 20 paired differences,
300 simulations, inner bootstrap 1,000). Output arms: R_s (9 scale cells), R_ℓ-ρ₂ (3), R_ℓ-acc (shuffled and reversed
at the 3 BCE levels, 6), O. Global arms: N (18 cells).

| arm | on the 20 exploration seeds | P(all pass), n = 60 | n = 80 |
|---|---|---|---|
| out16 (v) | all pass | 1.00 | 1.00 |
| out16b (v, b) | R_ℓ-acc FAILS (lower ends at BCE 0.1, 0.03 < 0: −0.003, −0.010 shuffled; −0.010, −0.019 reversed) | 0.16 | 0.18 |
| glob (÷12.18) | N passes | 0.91 | 0.96 |
| globcm (÷16.14) | N passes | 0.96 | 0.99 |

**Test suite (second addendum, before its commit):** main 917 passed, exit 0; `test_verify_certificates` 33 passed, exit 0 (gated, two processes, nice 15; logs `p2b_rev_suite_*.log` in the session scratch directory).
