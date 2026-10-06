DRAFT, not a registration. It applies the author's decisions of 2026-10-06. No registered or pilot seed has been
drawn and no pilot has been run. Exploratory data: `phase3_explore/` (the population and seeds 7,410,000–059).
‡ marks anything set after exploratory data.

# Design P3: the lag law at width 4 on the asymmetric windows

**Finding (exploratory): random holds collapse.**
- At s₀, random holds (U(−1, 1)⁹) reach a genuine four-unit branch 0/400 times.
- All 170 of 200 accepted landings are coinciding-unit points: 3+1, 2+2 or the duplicate. These are width-2 or width-1
  functions.
- The width-4 conditional minimiser is the width-2 function: its loss matches to 10 digits, and its switch
  [0.43805, 0.43839] lies inside T2-1's.
- Genuine three- and four-function copies appear only from wide holds (U(−3, 3)⁹), at most 4/150 each.
- So **Q is a MECHANISM test from a branch point.**

**Setting.**
- W2-A's, with N = Σ₁⁴ v_i f(α_i x + β_i) + b.
- Held output v₀ = s₀·(0.1, 0.2, 0.3, 0.4)‡, with s₀ = 0.5·s_pop2 = 0.2225397‡.
- Release by SGD: η on z, ρη on v.

**Symmetries.**
- Sign flips: fixed by v > 0.
- Permutations: not symmetries at distinct shares.
- Windings k ∈ ℤ⁴: κ_k = κ₀ + a·k exactly. The table is ‖k‖₁ ≤ 3 (129 configurations), fixed at release.
- Canonical form: β ∈ (−π, π]. A group of m coinciding units excludes 2(m − 1) split directions from λ.

| arm | copy | start | η, ρ‡ | seeds | gate (P at Wilson low) |
|---|---|---|---|---|---|
| **Q** headline | four-function; switch 0.482, κ₀ +0.017 | branch point | 0.03, 2⁻¹³ | 120 | ≥ 96/120, no hold G > 0 (0.98) |
| **T4** control | W2-A T embedded (3+1); switch 0.762 | random hold | 0.03, 2⁻¹⁰ | 710 | ≥ 60 (0.95; landed 53/400) |
| **S** sign test | three-function; switch 0.440, κ₀ −0.012 | branch point | 0.03, 2⁻¹² | 120 | ≥ 96/120, no hold G > 0 (≈ 1) |

**Own copies (60 seeds).**
- Valid copies: Q 57, S 59, T4 60.
- κ₀ < 0: Q 7/57, S 57/59, T4 0/60.
- Branch-point holds stay on their copy.

**Activity test** (Phase 3 only; added after the 2A-PS post hoc diagnosis, 2f0fb33).
- Unit i is active iff |v_i| ≥ θ·s, with **θ = 1e−8**‡.
- Why θ: 2A-PS's dead unit (|v| = 1.2e−93) passed the rule v ≠ 0 and leaked the scale direction â. The smallest
  genuine share in this exploration is 0.036.
- A run with an inactive unit is counted as "neither". â uses active units only.
- Tested now (`p3_activity.py`, 8 constructed cases, all passed): the 1e−93 unit is inactive; small genuine units are
  active; the threshold is inclusive and scale-free; it changes â and Δs at step 1 (0.1028 → 9.4e−6).

**Release classification.** W2-A's Newton rule, plus the same partition and all units active.

**Budgets‡.**
- B = min(⌈1.5·t\*⌉ + 3000, 10⁶), where t\* = (1/ρη)∫ ds/D along the frozen adiabatic path, D = −sign(v)·∇_vL.
  Exploration: t_obs/t\* = 0.998–1.015.
- Hard per-run caps: **10⁶ steps** and **RSS 1 GB**. A seed whose budget would exceed the step cap is counted and not
  trained.

**Frozen before training** (one commit, with OpenTimestamps).
- Per seed and copy: θ\*_own(s₀), λ, W, the validated switch, κ₀ and a, t\*, B, and the follow point.
- Also: ρ, f, θ, the tolerances, and the code.

**Strictly causal forecasts.**
- t_c = the first step with s ≥ 0.95·(the frozen switch).
- `causal_forecast`'s quad rule, with shares extrapolated per unit‡.
- R4 and the own-path switch run along the visible and extrapolated path.
- Checks: the GuardedArray test, the last row read, and a NaN recomputation.
- Forecasts are committed before observation.

**Criteria (1C).**
- C1, C2: |t_fc − t_obs| ≤ τ_cross and |lag_fc − lag_obs| ≤ τ_lag, each in ≥ 80% of runs.
- τ‡ = 1.5 × the pilot q90, rounded up to a multiple of 5.
- C3: median r_obs/r_fc ∈ [0.9, 1.1].
- C4: the paired bootstrap interval of D has its upper end < 0.
- S (arm S): both lags < 0 in ≥ 80% of runs.
- C3 and S use runs with |lag_fc| ≥ 6 only.
- Descriptive only: S on Q, and L1–L5.
- Validity: V1–V7, the follow check (partition kept), and t_c before the crossing in ≥ 90% of runs.
- Pilot: 10 seeds per arm; ρ is halved until the q90 χ ≤ 0.1.

**Exploratory runs** (8 seeds per arm):

| | lags | q90 crossing error (τ) | lag error | q90 χ | t_obs | C4 |
|---|---|---|---|---|---|---|
| Q | 19–213; −61, −192 | 33 (50) | ≤ 4 | 0.108 | 25k–90k | [−113, −6] |
| S | −9 to −69 | 17 (30) | ≤ 1 | 0.133 | 10k–32k | [−57, −25] |
| T4 | 109–180 | 7.6 (15) | ≤ 1 | 0.075 | 8k–21k | [−153, −126] |

**Seeds** (to be verified): Q 7,430,000–119; S 7,431,000–119; T4 7,432,000–709; pilot 7,439,000–009.

**Compute.** One worker, nice 15, with a memory gate before every job.
- **≈ 13 h** in jobs of at most 4 h: freeze 2, Q 6, S 2.5, T4 1.5, pilot 1.
- About 1 GB of paths; stop below 20 GB free disk.

**Outcomes.**
- Q PASS: a four-unit branch's signed lag is forecast.
- Q FAIL with T4 PASS: the law holds only on embedded width-2 branches.
- S PASS: the sign is forecast.
- C4 FAIL: the forecast is no better than no lag.
