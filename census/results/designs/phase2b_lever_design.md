Draft, revised with the author's decisions of 2026-10-02 (first draft dbb7cdb). **Not a registration**: no registered
or pilot seed drawn, no pilot run. **2B registers only after Phase 2A's result, with its OpenTimestamps proof before any
run.** Inputs: exploration e7c53fa, 68bdf76, a8f88c9, bbf5406 and `phase2b_explore/README_lever.md` (seeds
2,953,000–2,953,019; never registered). ‡ = set from exploratory data.

# Design 2B: the lever (output learning rate) on the simplicity-bias benchmark

**Setting.** v3 unchanged: data, tanh width 4, λ = 1e−4, `init_net` (cap 0.5·s\*, inside `fork_rng`; no warm start,
no idle-unit zeroing), Adam (0.9, 0.999, 1e−8), full batch, float64; s = ‖v‖₁, q = 0.3914, s\* = 3.5914. Every run
trains **T_C = 40,000 steps** (v3's cap); ρ₂ and BCE at every step.

| arm | learning rates | verdict |
|---|---|---|
| 1 standard | 0.01 on everything | reference |
| 2 output, **primary** | 0.01/16 on v (4 output weights); 0.01 on W, c, b | R_s, R_ℓ-ρ₂, R_ℓ-acc, O |
| 2b output, secondary | 0.01/16 on v and b; 0.01 on W, c | own verdict, same criteria |
| 3 global | 0.01/12.18 on everything‡ | N |
| 3cm global, cost-matched | 0.01/16.14 on everything‡ | N |

**Global factors‡** (from committed exploration runs). 12.18 = median over seeds of arm-2 ÷ arm-1 steps to 3 s\*
(12.177). It realises only ×9.454 (r_g). Cost-matched: model the cost ratio as G^a through (1, 1) and (12.18, r_g);
G_cm = 12.18^(ln 12.177/ln 9.454) = **16.14**. Confirmed on the exploration seeds: ×11.78, i.e. 0.97 [0.92–1.01] of
arm 2's steps to 3 s\* per seed.

**Basis.** Corollary L3 (§15.4): for fixed-P descent the lag is η-invariant, so the global arms should match arm 1 at
matched s. It is not proved for Adam, whose preconditioner moves (§15.5). R4 (§13.6) is registered Adam evidence in the
sine family only. Arm 2 changes the output/hidden rate ratio, which L3 does not fix.

**Matching** (first passage, no interpolation; overshoot reported).
- Scale: the first step with s ≥ s_m, s_m ∈ {1.25, 2, 3}·s\*.
- Loss: the first step with BCE (mean data loss) ≤ ℓ, ℓ ∈ {0.1, 0.03, 0.01}‡.
- Shuffled accuracy: x₁ replaced by every x₁ of the 800 points (the exact permutation expectation).
- Reversed accuracy: x₁ → −x₁, which swaps the classes' x₁ distributions exactly (x₂ unchanged).

**Criteria.** Δ = arm − arm 1 on the same seed. Each cell uses a 95% percentile bootstrap interval of the median Δ
(10,000 resamples, seeds jointly, `default_rng(20261002)`).
- **R_s**: ρ₂, shuffled and reversed accuracy at the 3 scales (9 cells). PASS if every lower end > 0.
- **R_ℓ-ρ₂**: ρ₂ at the 3 BCE levels. PASS if every lower end > 0.
- **R_ℓ-acc**: shuffled and reversed accuracy at the 3 BCE levels (6 cells). PASS if every lower end > 0.
- **O**: t_on is the first step with ρ₂ ≥ q at every step to T_C. PASS if ≥ 75%‡ of runs have s(t_on) > s\*; a run
  with no such step counts against.
- **N**, each global arm: all 18 cells. PASS if every upper end < δ, with δ = 0.03 for ρ₂ and reversed accuracy and
  0.02 for shuffled accuracy‡.
- Any criterion that does not pass is FAIL. It is UNRESOLVED if fewer than 90% of seeds reach a cell in both arms, or
  if any run is non-finite.

**Strict causal rule.** No criterion is a forecast. All compare observed outcomes after every run ends, so no cutoff
applies.

**Descriptive, registered as such.** The end-of-training comparison at T_C (ρ₂, both accuracies, s). **No endpoint
advantage is claimed.** Also reported: the matched step (arm 1's step to 3 s\*), steps to each matched point, onset
location (s/s\*, s/s_F; not tested against the fold) and two-sided equivalence for the global arms.

**Exploration.**
- Arm 2: Δρ₂ +0.25 to +0.31 at matched scale and +0.06 to +0.19 at matched loss (20/0). Matched-loss accuracy
  +0.025 to +0.14.
- Global arms: |Δρ₂| ≤ 0.013 up to 2 s\*, slightly lower at 3 s\*.
- Arm 1 is better at the matched step (Δρ₂ −0.24 for arm 2), and higher at T_C in 17/20 seeds.

**Seeds, power.** P(all pass) by resampling the exploration's paired differences, at n = 60 / 80:

| arm | 60 | 80 |
|---|---|---|
| 2 | 1.00 | 1.00 |
| 2b | 0.16 | 0.18 |
| 3 | 0.91 | 0.96 |
| 3cm | 0.96 | 0.99 |

- Arm 2b fails R_ℓ-acc on the exploration seeds: its lower ends are below 0 at BCE 0.1 and 0.03.
- O for arm 2 at n = 60: 0.9998 at the exploration rate 0.90, and 0.24 at its Wilson low 0.70.
- **80 seeds, 2,962,000–2,962,079**, shared by all arms. Verified unused in src/, tests/, results/, paper/, notes/,
  independent/, data/ and dist/.

**Compute.** 400 runs × 40,000 steps at about 160 µs/step: about 45 min unloaded, up to about 3 h under other load.
One worker, nice 15, one thread, < 0.5 GB, with a memory gate before each job.

**Outcomes.**
- All pass: the lever is the rate ratio, not the step count, and slab use still starts above s\*.
- N fails: global slowing helps too.
- Only R_ℓ fails: a scale-only effect.
- O fails: the onset moves to or below s\*.

**Decided (author, 2026-10-02).**
1. Arm 2 is primary; arm 2b is secondary.
2. Both global arms, with the factors above.
3. The matched scales and BCE levels.
4. N one-sided, with the δ above.
5. O at 75%, "lasting" to T_C.
6. The median bootstrap.
7. 80 seeds.
8. R_ℓ-acc is a named criterion.
9. The T_C comparison is descriptive.

10. Arm 2b keeps R_ℓ-acc as a registered criterion (author, 2026-10-02): its expected failure (power 0.18) is informative.

**Still open.** None on this page. 2B registers only after Phase 2A's result.
