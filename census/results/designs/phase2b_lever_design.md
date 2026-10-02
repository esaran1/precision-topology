Draft for the author's approval (2026-10-02). **Not a registration**: no registered or pilot seed drawn, no pilot run.
**2B is registered only after Phase 2A's result.** Inputs: exploration e7c53fa, 68bdf76, a8f88c9 and the addendum
`phase2b_explore/README_lever.md` (seeds 2,953,000–2,953,019; never registered). ‡ = chosen after seeing exploratory data.

# Design 2B: the lever (output learning rate) on the simplicity-bias benchmark

**Setting.** v3 unchanged: data, tanh width 4, λ = 1e−4, `init_net` (cap 0.5·s\*, inside `fork_rng`; no warm start,
no idle-unit zeroing), Adam (0.9, 0.999, 1e−8), full batch, float64; s = ‖v‖₁, q = 0.3914, s\* = 3.5914. Every run
trains **T_C = 40,000 steps** (v3's cap); ρ₂ and BCE at every step.

| arm | learning rates |
|---|---|
| 1 standard | 0.01 on everything |
| 2 output-only | 0.01/16 on v (the 4 output weights); 0.01 on W, c and the output bias b‡ |
| 3 global | 0.01/G on everything, **G = 12.18**‡ |

G = median over the 20 exploration seeds of arm-2 ÷ arm-1 steps to 3 s\* (committed `runs/`: 12.177), fixed now; no
pilot. Realised cost to 3 s\*: arm 3 ×9.45, arm 2 ×12.18 (exploratory).

**Basis.** Corollary L3 (§15.4): for fixed-P descent the lag is η-invariant, so arm 3 should match arm 1 at matched
s. It is not proved for Adam, whose preconditioner moves (§15.5); R4 (§13.6) is registered Adam evidence in the sine
family only. Arm 2 changes the output/hidden rate ratio, which L3 does not fix.

**Matching** (first passage, no interpolation; overshoot reported). Scale: first step with s ≥ s_m, s_m ∈ {1.25, 2,
3}·s\*‡. Loss: first step with BCE (mean data loss) ≤ ℓ, ℓ ∈ {0.1, 0.03, 0.01}‡. Quantities: ρ₂; shuffled accuracy
(x₁ replaced by every x₁ of the 800 points: the exact permutation expectation); reversed accuracy (x₁ → −x₁, which
swaps the classes' x₁ distributions exactly; x₂ unchanged).

**Criteria.** Δ = arm − arm 1 on the same seed; 95% percentile bootstrap interval of the median Δ (10,000 resamples,
seeds jointly, `default_rng(20261002)`).
- **R_s / R_ℓ**: arm 2, all 9 scale / 9 loss cells: lower end > 0.
- **N** (no gain): arm 3, all 18 cells: upper end < δ = 0.03 (ρ₂, reversed), 0.02 (shuffled)‡. Fails if global
  slowing gains ≥ δ anywhere.
- **O**: lasting onset t_on = first step with ρ₂ ≥ q at every step to T_C. PASS iff ≥ 75%‡ of arm-2 runs have
  s(t_on) > s\* (runs without one count against).
- Otherwise FAIL; UNRESOLVED if validity fails: < 90% of seeds reach a cell in both arms, or any non-finite run.

**Strict causal rule.** No criterion is a forecast: all compare observed outcomes after every run ends; no cutoff.

**Reported only.** Matched step (arm 1's step to 3 s\*) and T_C comparisons; steps to each matched point and ratios;
onset location (s/s\*, s/s_F; not tested against the fold); two-sided equivalence for arm 3.

**Exploration.** All four pass. Arm 2 Δρ₂ +0.25 to +0.31 at scale, +0.06 to +0.19 at loss (20/0). Arm 3 |Δρ₂| ≤ 0.008,
except lower at 3 s\* (−0.04) and BCE 0.03 (−0.02). At the matched step, arm 2 is lower (−0.24, 0/20); at T_C arm 1 is
higher in 17/20.

**Seeds, power.** Resampling the exploration's paired differences gives P(all pass) = 0.63 / 0.78 / 0.90 / 0.96 at
n = 30 / 40 / 60 / 80. N binds: ρ₂ at 1.25 s\*, arm-3 median +0.007. O: 0.9998 at rate 0.90 (n = 60), 0.24 at its
Wilson low 0.70. With the arm-2 effect halved and spread doubled, P(R) < 0.1. **80 seeds, 2,962,000–2,962,079**,
verified unused in src/, tests/, results/, paper/, notes/, independent/, data/, dist/.

**Compute.** 240 runs × 40,000 steps at ≈ 160 µs/step: ≈ 26 min unloaded, ≤ 1.5 h under load. One worker, nice 15, one
thread, < 0.5 GB, memory gate per job.

**Outcomes.** All pass: the lever is the rate ratio, not the step count, and slab use still starts above s\*. N
fails: global slowing helps too. Only R_ℓ fails: a scale-only effect. O fails: the onset moves to or below s\*.

**Needs your decision.**
1. Output layer: v only (as explored), or v and b‡ (G = 13.0; shuffled accuracy at matched loss 11/9).
2. G by the step-ratio rule, or cost-matched (≈ 16; not run).
3. The matched scales, the BCE levels‡, and BCE versus the full objective.
4. N one-sided with these δ‡, or two-sided equivalence (fails at 3 s\* in exploration).
5. O: the 75% threshold and "lasting" to T_C‡ (to 100,000 steps, 10/20 standard runs dip below q again).
6. The median bootstrap, or a sign test.
7. 80 seeds.
