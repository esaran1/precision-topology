DRAFT for approval, 2026-10-05. Not a registration: no registered or pilot seed drawn, no pilot. Exploratory:
`phase3_explore/` (population; unused seeds 7,410,000–059). ‡ = set after exploratory data.

# Design P3: the lag law at width 4 on the asymmetric windows

**Setting.** W2-A's, with N = Σ₁⁴ v_i f(α_i x + β_i) + b and shares (0.1, 0.2, 0.3, 0.4)‡.
**s₀ = 0.5·s_pop2 = 0.2225397‡.** The width-4 conditional minimiser is the width-2 function (units coincide): its loss
matches to 10 digits, and its switch bracket [0.43805, 0.43839] lies inside T2-1's.

**Branches at v₀.** Random holds (U(−1, 1)⁹) reach only two-function points (unit partitions 3+1, 2+2) or the
duplicate (170/200 accepted; the rest are idle-unit saddles). Wider holds (U(−3, 3)⁹) reach three-function points
(~25 copies, ≤ 4/150 each) and four-function points (3 copies, 1/150 each).

**Symmetries.** Sign flips (2⁴) are fixed by v > 0. Permutations (4!) are not symmetries at distinct shares: each
relabelling is its own copy. Windings k ∈ ℤ⁴ give κ_k = κ₀ + a·k exactly (5 frozen numbers per copy); the table is
‖k‖₁ ≤ 3 (129 configurations), with k fixed at release. Canonical form: β ∈ (−π, π] with b shifted. A group of m
coinciding units excludes 2(m − 1) split directions from λ.

| arm | population copy | start | η, ρ‡ | seeds | gate‡ (Wilson-low P) |
|---|---|---|---|---|---|
| **Q** headline | four-function; switch 0.482, κ₀ +0.017 | branch point | 0.03, 2⁻¹³ | 120 | ≥ 96/120, no hold G > 0 (0.98; 108/120: 0.15) |
| **T4** control | W2-A T's function embedded (3+1); switch 0.762 (width 2: 0.482) | random hold, as T | 0.03, 2⁻¹⁰ | 710 | ≥ 60 on T4 (0.95) |
| **S** sign test | three-function; switch 0.440, κ₀ −0.012 | branch point | 0.03, 2⁻¹² | 120 | ≥ 96/120, no hold G > 0 (≈ 1) |

**Own samples.** Valid copies: Q 57/60, S 59/60, T4 60/60. κ₀ < 0 in Q 7/57 (κ/λ −4.1 to 7.0), S 57/59, T4 0/60.
Branch-point holds stay on every valid copy. Random-hold landing: T4 53/400 (Wilson low 0.103), Q 0/400. Split blocks
stay positive definite up to every switch.

**Release classification.** W2-A's Newton rule plus the same unit partition. Others are counted, never replaced.

**Frozen before training** (one commit, OpenTimestamps). Per seed and copy: θ\*_own(s₀), λ, W = max(4000, ⌈25/λ⌉)
(Q up to 68,064), the validated adiabatic switch (plus a positive-definite split block), and κ₀ and a (total derivative,
reduced λ). Also the follow point, ρ, f, the tolerances, and the code including the own-path switch.

**Strictly causal forecasts.** t_c = the first step with s ≥ 0.95·(frozen switch). `causal_forecast`'s quad rule
(5% window, horizon 1.6), with shares extrapolated per unit and renormalised‡. R4 and the own-path switch run along the
visible and extrapolated path. Checks: the GuardedArray test, the last row read, and an identical NaN recomputation.
Forecasts are committed before observation.

**Criteria (1C).**
- C1, C2: |t_fc − t_obs| ≤ τ_cross and |lag_fc − lag_obs| ≤ τ_lag, each in ≥ 80% of runs; τ‡ = 1.5 × the pilot
  q90, rounded up to a multiple of 5.
- C3: median r_obs/r_fc ∈ [1 ± 0.10]. C4: the paired bootstrap interval of D has its upper end < 0.
- S: the forecast and observed lags are both negative in ≥ 80% of runs.
- C3 and S use runs with |lag_fc| ≥ 6 only; UNRESOLVED if fewer than half (or 2) remain. L1–L5 are descriptive.
- Validity: V1–V7, the follow check (partition kept), t_c before the crossing in ≥ 90%, no forecast = a miss.
- Pilot: 10 seeds per arm, to t_sw; ρ halved (≤ 3 times) until the q90 window χ ≤ 0.1.

**Exploratory runs** (8 seeds per arm, f = 0.95):

| | lags (steps) | q90 \|t_fc − t_obs\| (τ by the rule) | max \|lag error\| | q90 window χ | t_obs | C4 interval |
|---|---|---|---|---|---|---|
| Q | 19–213; −61, −192 | 33 (50) | 4 | 0.108 (ρ → 2⁻¹⁴) | 25k–90k | [−113, −6] |
| S | −9 to −69 | 17 (30) | 1 | 0.133 (ρ → 2⁻¹³) | 10k–32k | [−57, −25] |
| T4 | 109–180 | 7.6 (15) | 1 | 0.075 | 8k–21k | [−153, −126] |

The lag signs agree in 24/24 runs.

**Seeds** (proposed; to be verified unused at registration): Q 7,430,000–119; S 7,431,000–119; T4 7,432,000–709;
pilot 7,439,000–009.

**Compute.** One worker, nice 15, one thread, < 0.5 GB, a memory gate before every job. Freeze 2 h, Q 6 h, S 2.5 h,
T4 1.5 h, pilot 1 h: **≈ 13 h**, in jobs of ≤ 4 h. About 1 GB of untracked paths; stop below 20 GB free disk.

**Outcomes.** Q PASS: a four-unit branch's signed lag is forecast before it happens. Q FAIL with T4 PASS: the law
holds only on embedded width-2 branches. S PASS: the sign is forecast. C4 FAIL: no better than no lag. Gate FAIL: the
branch is not reached or not kept.

**Needs your decision.**
1. **Is width 4 feasible?** W2-A-style random holds never reach a genuine four-unit branch. Accept Q from a branch
   point, with its random landing (0/400) reported? Otherwise P3 tests only width-2 branches embedded in width 4 (T4).
2. Should s₀ = 0.5·s_pop2 and the shares (0.1, 0.2, 0.3, 0.4) stand?
3. T4: 710 random seeds, or 120 branch-point seeds, or score every two-function T-type copy (179/400; 200 seeds)?
4. Gates of 96/120 (rather than W2-A's 108/120)?
5. Apply |lag_fc| ≥ 6 to C3 and S? Should S on Q's mixed signs be descriptive?
6. ‖k‖₁ ≤ 3 and the per-unit share forecast. Smaller option: Q and S with 80 seeds each (P(Q ≥ 60) = 0.998), ≈ 10.5 h.
