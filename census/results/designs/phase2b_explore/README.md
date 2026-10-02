# Phase 2B exploration: does slowing the output make training reach the slab? (EXPLORATORY, NOT a registration)

**Everything here is EXPLORATORY.** Nothing is registered or frozen, and no number here is a test outcome. It was run
to inform the author's decision on Phase 2B ("the lever").
- Seeds: non-registered exploration seeds **2,953,000–2,953,019**, with 2,953,099 used for the implementation check.
  The range 2,953,000–2,953,099 is reserved for this exploration and **must never be registered**.
  - Before use, a text search of src/, tests/, paper/, results/, notes/, independent/, data/ and dist/ found no
    occurrence (as 2953xxx, 2_953_xxx or 2,953,xxx).
  - No registered or pilot seed range was drawn.
- Only small per-run summaries are saved: one JSON per run in `runs/`, with no parameter paths.

## Setup

**Benchmark: v3, unchanged.**
- v2's 800 points, tanh width 4, BCE (mean) + (λ/2)(‖W‖² + ‖c‖²) with λ = 1e−4, float64, full batch.
- Constants: s = ‖v‖₁, q = 0.3914, s\* = 3.5914, and M's fold s_F = 4.7677 (Track 1A).

**Initialisation: v3's own `init_net`.**
- PyTorch default init after `torch.manual_seed(seed)`, cast to float64, output weights capped at 0.5·s\*.
- It runs inside `torch.random.fork_rng`, so the global RNG state is restored afterwards.
- Idle units are not zeroed. No other global state is touched.

**Training is in numpy.**
- The Adam update reproduces `torch.optim.Adam`. In `p2b_check.json`, the v3 torch loop and this code agree to 9e−16
  (relative, parameters) over 1,500 steps.
- The grid ρ₂ and G₊ equal `src.simplicity_bias_v2.rho2_batch` and `src.simplicity_bias.gplus` to 8e−16.

| Condition | Rule |
|---|---|
| `adam_r1` | Adam lr 0.01 on all parameters (the v3 protocol) |
| `adam_r4`, `adam_r16`, `adam_r64` | Adam, lr 0.01·r on all of v (r = 1/4, 1/16, 1/64); W, c and the output **bias b** stay at lr 0.01 |
| `gd_rho1` | full-batch GD, η = 1 on everything (reference for the GD family; added) |
| `gd_scale_r6`, `gd_scale_r9` | GD, η = 1 on (W, c, b); v ← v − η[(I − ââᵀ) + ρââᵀ]∇_vL with â = sign(v)/‖sign(v)‖ (**2A's rule verbatim**); ρ = 2⁻⁶, 2⁻⁹ |
| `gd_plain_r6`, `gd_plain_r9` | GD, η on (W, c, b), ρη on all of v (the mechanism contrast) |
| `gd_wn_r6`, `gd_wn_r9` | **added:** ℓ₁ weight-normalised output v = g·u/‖u‖₁, so ‖v‖₁ = g exactly. GD at η on (W, c, b, u) and ρη on the gain g |
| `adam_wn_r64` | **added:** the same reparametrisation under Adam: lr 0.01 on (W, c, b, u) and 0.01/64 on g |

**Why the weight-normalised conditions were added.**
- From a random init, 2A's rule turned out not to slow the scale. The reason is given in the findings below.
- The `wn` conditions are a leak-free way to slow *only* the scale, so that the scale-only idea itself could be compared.
- They were added after the pilot seed 2,953,000 had shown the leak.

**Stopping rule.** A run stops at the first step with s ≥ 3·s\* **and** t ≥ the matched step budget, or at its step
budget.
- The matched step budget is the same seed's `adam_r1` step count to 3·s\* (median 576).
- The step budget is 1M for `gd_wn_r9` and at least 1.6M for every other condition; no other condition hit it.

**Measures.**
- **ρ₂** is v2/v3's S-randomisation share, evaluated at **every step**.
- **G₊** (the "gap") is ½(min_{y=1}φ − max_{y=0}φ) of the unit-ℓ₁ function.
- The accuracies are taken at the first step reaching each matched scale and at the matched step:
  - **Train accuracy.**
  - **Shuffled accuracy:** x₁ is replaced by every x₁ value of the 800 points. This is the exact expectation over a
    uniform random permutation of x₁ across examples, so no RNG is involved.
  - **Reversed accuracy:** x₁ → −x₁ for every point. The generator gives class 1 x₁ = a and class 0 x₁ = −a with the
    same level set a (`make_data`), so this swaps the classes' x₁ distributions exactly. The x₁–label correlation goes
    from +0.80 to −0.80, and x₂ (the slab) is unchanged.
  - Reference values: a pure-x₁ rule scores 0.50 shuffled and 0.10 reversed; a pure-slab rule scores 1.00 on both.
- **Onset of slab use:**
  - the **first upward passage** of q (v3's event) and the literal first ρ₂ ≥ q;
  - **last upward passage**: the start of the final stretch with ρ₂ ≥ q, counted only for runs that end with ρ₂ ≥ q.
  - The last upward passage is the meaningful onset here. In 13 of 20 seeds the random init already has ρ₂(0) ≥ q,
    so many first passages are re-crossings after an early dip at s < s\*.
- **Fixed-scale preference** (`p2b_reference.json`, from the committed branch continuation in `results/sb_fold/`):
  - the global minimiser is M at 0.5 s\* (ρ₂ 0.08), S at s\*, 1.25 s\*, s_F and 2 s\* (ρ₂ 0.447, 0.538, 0.559, 0.665),
    and S2 at 3 s\* (ρ₂ 0.741);
  - L0, the pure-x₁ unit, has ρ₂ = 0 at every scale.

## Headline table

20 seeds per condition, all conditions on the same seeds. Cells are median [IQR] across seeds; the full tables are in
`p2b_tables.md`.

| Condition | Steps to 3 s\* (× adam_r1, per seed) | ρ₂ at s\* | ρ₂ at s_F | ρ₂ at 3 s\* | Shuffled / reversed accuracy at 3 s\* | Onset: last up s/s\* (runs ending ≥ q) | At matched step: s, ρ₂, shuffled accuracy |
|---|---|---|---|---|---|---|---|
| fixed-scale minimiser | — | 0.447 | 0.559 | 0.741 | — | (s\* by construction) | — |
| adam_r1 | 576 (1.0) | 0.03 [0.00–0.10] | 0.07 [0.02–0.12] | 0.39 [0.37–0.42] | 0.72 / 0.43 | 2.86 [2.59–2.97] (11/20) | 10.8, 0.39, 0.72 |
| adam_r4 | 2,642 (4.6) | 0.16 [0.07–0.21] | 0.25 [0.17–0.28] | 0.59 [0.58–0.61] | 0.83 / 0.66 | 2.03 [1.84–2.10] (20/20) | 5.2, 0.28, 0.63 |
| adam_r16 | 7,082 (12.2) | 0.25 [0.18–0.30] | 0.33 [0.30–0.37] | 0.70 [0.69–0.70] | 0.88 / 0.76 | 1.75 [1.44–1.86] (20/20) | 2.4, 0.15, 0.54 |
| adam_r64 | 19,352 (32.9) | 0.26 [0.23–0.31] | 0.36 [0.34–0.37] | 0.70 [0.70–0.71] | 0.88 / 0.77 | 1.73 [1.54–1.84] (20/20) | 1.4, 0.13, 0.53 |
| gd_rho1 | 236 (0.4) | 0.06 [0.02–0.08] | 0.02 [0.01–0.03] | 0.24 [0.22–0.25] | 0.64 / 0.29 | 4.04 [3.90–4.12] (17/20) | 15.1, 0.42, 0.75 |
| gd_scale_r6 (2A rule) | 1,376 (2.0) | 0.00 [0.00–0.01] | 0.00 [0.00–0.01] | 0.50 [0.49–0.54] | 0.80 / 0.59 | 2.46 [2.41–2.54] (20/20) | 7.7, 0.29, 0.66 |
| gd_scale_r9 (2A rule) | 1,393 (2.1) | 0.00 [0.00–0.01] | 0.00 [0.00–0.01] | 0.51 [0.49–0.54] | 0.80 / 0.59 | 2.46 [2.41–2.53] (20/20) | 7.6, 0.29, 0.66 |
| gd_plain_r6 | 13,546 (23.3) | 0.00 [0.00–0.00] | 0.17 [0.16–0.18] | 0.61 [0.61–0.61] | 0.84 / 0.67 | 2.03 [2.02–2.06] (20/20) | 3.3, 0.00, 0.50 |
| gd_plain_r9 | 97,601 (168) | 0.15 [0.14–0.17] | 0.28 [0.28–0.29] | 0.70 [0.70–0.70] | 0.88 / 0.77 | 1.71 [1.67–1.74] (20/20) | 1.6, 0.00, 0.50 |
| gd_wn_r6 | 149,891 (204) | 0.00 [0.00–0.00] | 0.00 [0.00–0.26] | 0.74 [0.74–0.74] | 0.90 / 0.80 | 2.06 [1.51–2.12] (20/20) | 2.1, 0.00, 0.50 |
| gd_wn_r9 | 3 of 20 reach 3 s\* | 0.00 [0.00–0.00] | 0.00 [0.00–0.00] | 0.74 (n = 3) | 0.90 / 0.80 (n = 3) | 1.72 (3/20) | 1.2, 0.00, 0.50 |
| adam_wn_r64 | 69,140 (117) | 0.00 [0.00–0.00] | 0.00 [0.00–0.00] | 0.00 [0.00–0.00] | 0.50 / 0.10 | — (0/20) | 1.1, 0.00, 0.50 |

**Paired differences against adam_r1** (same seed; "a/b" = number of seeds above/below; `p2b_tables.md` F):

| Condition | Δρ₂ at s_F | Δρ₂ at 3 s\* | Δshuffled accuracy at 3 s\* | Δreversed accuracy at 3 s\* | Δρ₂ at the matched step |
|---|---|---|---|---|---|
| adam_r4 | +0.17 (16/4) | +0.21 (20/0) | +0.12 (20/0) | +0.24 (20/0) | −0.13 (2/18) |
| adam_r16 | +0.28 (20/0) | +0.31 (20/0) | +0.16 (20/0) | +0.32 (20/0) | −0.24 (0/20) |
| adam_r64 | +0.31 (20/0) | +0.31 (20/0) | +0.17 (20/0) | +0.33 (20/0) | −0.25 (0/20) |

**Gap G₊.**
- G₊ follows ρ₂.
- At s_F it is > 0 (median) only for adam_r16 (0.043), adam_r64 (0.061) and gd_plain_r9 (0.023).
- At 3 s\* it is 0.055 for adam_r1 and 0.18–0.21 for the slowed Adam runs.

## Findings

**1. Plain output-LR reduction under Adam (the practitioner's knob) changes what is learned at matched output scale.**
- At every matched scale from 1.25 s\* up, slab use and shifted-test accuracy are higher than standard.
  - At 3 s\* this holds in 20 of 20 paired seeds for every r.
  - At s_F it holds in 16/20 (r = 1/4) and 20/20 (r = 1/16, 1/64).
  - At 3 s\*, ρ₂ goes 0.39 → 0.59 → 0.70 → 0.70, shuffled accuracy 0.72 → 0.83 → 0.88 → 0.88, and reversed accuracy
    0.43 → 0.66 → 0.76 → 0.77.
  - The effect **saturates by r ≈ 1/16**.
  - Every slowed run ends with ρ₂ ≥ q, against 11 of 20 standard runs.
- **It does not make slab use start at s\*.**
  - Persistent onset moves from ≈ 2.9 s\* to ≈ 1.7–2.0 s\*.
  - With r = 1/64: 6/20 runs fall in the s_F band [1, 1.25]·s_F, 12 above it, 2 below s\*, and **none in v3's band
    [1, 1.25]·s\***.
  - ρ₂ rises smoothly with s and stays below the fixed-scale minimiser at every matched scale: 0.26 vs 0.447 at s\*,
    0.36 vs 0.559 at s_F, 0.43 vs 0.665 at 2 s\*.
  - So output-LR reduction **reduces the lag** behind the preferred feature; it does not make training track it.
- **At matched steps the sign reverses.**
  - At the standard run's step budget, the slowed runs are at s = 5.2 / 2.4 / 1.4.
  - They use the slab **less**: ρ₂ 0.28 / 0.15 / 0.13 against 0.39, with 18/20, 20/20 and 20/20 seeds below.
  - Their shuffled accuracy is lower: 0.63 / 0.54 / 0.53 against 0.72.
- **The cost.** Per seed, slowed runs take ×4.6 / ×12 / ×33 the steps to 3 s\* and ×3.6 / ×15 / ×53 the steps to s_F.

**2. Plain GD shows the same pattern.**
- With ρη on all of v, ρ₂ at 3 s\* goes 0.24 (ρ = 1) → 0.61 (2⁻⁶) → 0.70 (2⁻⁹).
- At ρ = 2⁻⁹ the result matches Adam r = 1/64 at 3 s\* (0.70; shuffled 0.88, reversed 0.77).
- The cost is ×57 and ×414 the steps of gd_rho1.

**3. 2A's scale-only rule, applied verbatim from a random init, does not slow the scale.**
- It is **independent of ρ**: 2⁻⁶ and 2⁻⁹ give 1,376 vs 1,393 steps to 3 s\* and identical ρ₂.
- The scale grows through **output-weight sign changes**, about 1,500–1,600 per run.
  - The units being pruned (v_k → 0) chatter across zero at η = 1.
  - Each crossing adds about 2|v_k| to ‖v‖₁, because the projector removes the â-component only within one sign
    orthant.
  - In `p2b_diag.json` (3 seeds), 9.5–10.2 of the ≈ 10 units of s growth occur on sign-change steps.
- 2A did not see this because it released from M with the idle unit exactly 0 and the active signs fixed.
- Its outcome lies between GD and slowed GD: 0.50 at 3 s\*, but ≈ 0 up to s_F.
- **As written, the rule needs a hold or warm start; it is not usable from a standard init.**

**4. Leak-free scale-only slowing locks onto the linear feature.**
This is the weight-normalised gain at ρη, with the shares fast.
- At s\*, 1.25 s\* and s_F, nearly every run is the pure-x₁ function: ρ₂ 0.00, shuffled accuracy 0.50, reversed 0.10.
- `adam_wn_r64` reaches 3 s\* in 20/20 runs **still at ρ₂ = 0** (G₊ < 0, shifted accuracies 0.50 / 0.10).
- `gd_wn_r9`:
  - 17/20 runs **stall at s = 6.049**. The value is the same in every run; ρ₂ = 0, the other three units are dead,
    and (c, b) sit in a period-2 GD oscillation at η = 1. They stay there for the rest of the 1M steps; the same state
    persists to 6M steps on seed 2,953,000 (`p2b_timing.log`).
  - 3/20 escape and end at ρ₂ 0.74.
- `gd_wn_r6`:
  - all 20 runs escape eventually, and at 3 s\* have the best values of any condition: ρ₂ 0.741 (the S2 minimiser's
    value), accuracies 0.90 / 0.80;
  - but at s_F, 13 of 20 are still linear (ρ₂ < 0.01; median 0.00, IQR 0–0.26);
  - the cost is ×204 steps.
- **Mechanism** (diagnostics on 3 seeds; an interpretation):
  1. With fast shares and a slow scale, the shares settle at small s, where the fixed-scale minimiser is L0, one
     pure-x₁ unit (global below s = 1.49).
  2. The other units' output shares go to ≈ 0.
  3. Weight decay then removes their hidden weights. Adam's normalised steps do this within ~10⁴ steps; GD at ηλ = 1e−4
     takes ~10⁵ steps.
  4. What is left is a width-1 network on L0, which has no fold up to s = 37.5 (Track 1A).
- Whether the slab is ever reached becomes a race between a unit's recruitment and its decay. Slowing the scale more
  makes recruitment lose.

**5. A possible reason the plain output-LR knob works and scale-only slowing does not** (an interpretation, not tested
separately).
- Slowing all of v also slows the shares, so no unit is pruned before the scale grows: the Adam-r runs had a median of 0
  output sign changes and at most 1 per run.
- The idle units therefore stay available when the slab starts to pay.
- This matches 2A's finding that ρ on all of v leaves χ ≈ 3.5–4.3: these runs are not quasi-static. They help at
  matched scale by reducing the lag, not by tracking the minimiser.

## Plain answer to the author's question

- **Does a plain output-LR reduction change what training learns?** Yes, at matched output scale, consistently across
  the 20 seeds. It gives more slab use and markedly better accuracy with x₁ shuffled or reversed, and the effect
  saturates near r = 1/16.
  - It does not move the onset of slab use to s\*. The onset sits at ≈ 1.7–2.0 s\*, in the s_F band or above it.
  - At a matched **step** budget it is worse than standard, because the slowed run has not yet grown its scale.
  - The benefit costs 4.6–33× the steps to 3 s\*.
- **How does it compare with scale-only slowing?**
  - From a standard init, 2A's scale-only rule leaks through sign chatter and does not depend on ρ.
  - A leak-free scale-only slowing pushes training onto the **linear** feature, because the extra units die before the
    scale grows. It does reach the best end state when ρ is mild (2⁻⁶), but only after ×200 steps and with nothing at s_F.
  - For a lever applied from a standard init, plain output-LR reduction is the one that moves training toward the
    preferred feature in this benchmark. Scale-only slowing would need 2A's hold, or protection of the idle units.

## Caveats

- 20 seeds and one fixed point set: the benchmark has no sampling variation, so the seeds vary only the initialisation.
  The IQRs are across those 20 initialisations.
- GD uses η = 1, as in 2A. At s ≈ 6 on L0, η = 1 is past the (c, b) stability threshold: a period-2 oscillation was
  observed. Sharpness was not checked in the other GD conditions.
- The Adam-r conditions keep the output **bias** at lr 0.01, following "all of v". A practitioner's layer-wise learning
  rate would usually slow the bias too.
- Matched quantities are taken at the first step with s ≥ the scale. Under GD at η = 1, s can overshoot within a step.
- No χ, branch labels or fold forecasts were computed for these runs.

## Compute

- **One process at a time**, nice 15, one thread (OMP/VECLIB = 1). A memory gate (`gate.sh`: free ≥ 25%, swap free
  ≥ 500 MB) ran as a separate logged step before every job. It held once, for 60 s.
- Peak RSS was 0.48 GB per run.
- Wall time, 00:13–01:38, about 85 minutes:
  - 240 runs from 00:24 to 01:33, of which 66.5 min was training and measurement;
  - the check (2 s), the s-only timing pilot (~7 min, including one 6M-step run) and the diagnostics (~4 min).
- The budget allowed 20 seeds instead of the suggested 8–10.

**Test suite.** It ran as two processes, each gated, under nice 15, with no src/ or tests/ change in this work.

| Log | Run | Result |
|---|---|---|
| `p2b_suite_certs.log` | test_verify_certificates | 33 passed, exit 0 |
| `p2b_suite_main.log` | main suite, first run, detached in the background | 3 failed, 914 passed, exit 1 |
| `p2b_suite_rerun3.log` | the 3 failing tests in the foreground | 3 passed, exit 0 (they also pass in the foreground under nice 15) |
| `p2b_suite_main_fg.log` | main suite, rerun in the foreground | **917 passed, exit 0** |

- The 3 failures were `PermissionError` from an unguarded `os.nice(15)` in `src/band_rd.py:629` and `src/ramp2.py:116`.
- They are environmental: setpriority was refused in the detached background context of this session.

## Files

| File | Contents |
|---|---|
| `p2b_explore.py` | producer: `check`, `run COND SEED`, `summarise` |
| `p2b_report.py` | writes `p2b_tables.md` |
| `p2b_reference.py` | writes `p2b_reference.json` (branch ρ₂ at the matched scales) |
| `p2b_diag.py` | writes `p2b_diag.json` (per-unit states, sign-change s increments) |
| `p2b_timing.py` | s-only timing pilot |
| `run.sh`, `queue.sh`, `gate.sh` | driver, seed-major queue, memory gate |
| `runs/COND_SEED.json` | per-run summaries: matched-point measures, events, a 250-point log-spaced (t, s, ρ₂) trace |
| `p2b_summary.json` | per-condition medians, IQRs and paired differences |
| `*.log` | logs: memory-gate lines and one JSON line per job |

In `p2b_diag.json`, the split of the s increment into sign-change and other steps is meaningful only for the 2A rule.
In the `wn` conditions, s = g exactly.
