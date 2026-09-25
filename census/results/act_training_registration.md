# Registration: training test of the GELU / SiLU / Mish switches (Track 3A, step 4)

**Written and committed before any run on the registered seeds.** Date: 2026-09-25 (see `git log` for the time).
Code: `src/act_general.py` (`run_one`, `train`, `train_ext`, `score`, `score_runs`). Tests: `tests/test_act_general.py`
(T-a, T-b and UNRESOLVED exercised on constructed pass and fail cases, including the 90% edge and the tolerance floor;
crossing detection with these activations; fixed-column run records). All pass at this commit.

## Which activations

Every activation with a VALIDATED bracket (step 3, `results/act_general/bracket_*.json`):

| act | criterion verdict (registered, step 2) | validated bracket [s_lo, s_hi] | s_glob = √(s_lo·s_hi) | s* (tracked branch) |
|---|---|---|---|---|
| GELU | switch predicted | [6.611690, 6.671427] | 6.641492 | 6.645633 |
| SiLU | undetermined (s = 0.1 status undecided) | [3.619043, 3.651741] | 3.635355 | 3.639193 |
| Mish | undetermined (s = 0.1 status undecided) | [3.190849, 3.219678] | 3.205231 | 3.214656 |

SiLU and Mish are included because their brackets are validated; their criterion verdict stays "undetermined".

## κ, frozen before any registered run (Track 1's method, k = 0: no winding for a non-periodic activation)

κ = λ_min(P^{1/2}HP^{1/2})·[∇G·(PH)⁻¹θ*′]/[∇G·θ*′] at s* on the population landscape; the branch is the retained
minimiser at the bracket's lower end (σ = +1, mirror-canonical w₁ > 0), tracked by damped Newton; G changes sign on it
inside the bracket; H positive definite; ∇G by central differences (one-sided agreement ≤ 2e−6 relative). P = the
median normalised Adam preconditioner diag(1/(√v̂ + ε)) at the crossing of the **calibration seeds 851,000–851,009**
(never registered seeds; crossing runs not placed at initialisation).

| act | κ_Adam (frozen) | calibration IQR of per-run κ | κ_SGD (info) | frozen file SHA-256 |
|---|---|---|---|---|
| GELU | +0.147280 | [0.1314, 0.1517] | +0.1321 | 9a4848d27bbfb10e1ecf031f01abe8fec5973d75d4dd974f15e81ee3ac1db779 |
| SiLU | −0.144403 | [−0.1514, −0.1275] | −0.1571 | 5869d772dda839856c1215b3525d5879a4fef2ad5b38856fe8e730c659f7c088 |
| Mish | −0.153877 | [−0.1587, −0.1349] | −0.1739 | 58359defe125a95f68f5eb5a00d3e327959bf0c6135c2565cfed3086f195bf64 |

Files: `results/act_general/kappa_{gelu,silu,mish}_frozen.json` (+ `.sha256`); `score` stops if a hash changes.

**Disclosure (calibration, seen before this registration).** On the 10 calibration seeds, 5 of 10 runs per activation
cross (not placed at init); the rest stall near a trivial point (|w₂| < 1, loss ≈ log 2). GELU: one run placed at
initialisation, one crossing at s = 0.24 (a genuine placement far below the switch); crossings 5.9–7.3. SiLU crossings
3.30–4.34; Mish 2.95–3.90. χ at crossing ≈ 0.004–0.015 (GELU) and 0.05–0.58 (SiLU, Mish). So, before any registered
run, coverage below 30 crossings (UNRESOLVED) and a T-a failure are both plausible; the predictions below are the ones
the task specifies, unchanged.

## Protocol (registered seeds)

- Training sets `fold1d.make_data(200, seed)`, **seeds 850,000–850,039** (40 fresh seeds; a search of every file in
  `results/` and of `src/`, `tests/` for 850,000–851,999 found none).
- Standard protocol as `fold1d` / `phase2b_ordering.run`: θ = (w₁, b₁, w₂, b₂) ~ U(−1, 1)⁴ drawn in float32
  (`torch.manual_seed(seed)`) then cast to double; Adam, lr 0.01, full batch; logits w₂·u(w₁x + b₁) + b₂ with u the
  torch activation (GELU exact erf form, SiLU, Mish); budget 32,000 steps. One worker, nice 15.
- Every-step crossing detection with `phase2b_ordering.state` (dense windows, G > 0 in w₂'s orientation) with the
  activation. **Crossing runs** = runs with a crossing that were not placed at initialisation (placed-at-init runs are
  counted and reported, not scored). s_cross = |w₂| at the crossing step.
- χ at the crossing, exactly as `ts_test` / Track 3: growth = log(|w₂|(t)/|w₂|(t − w))/w, w = min(100, t − 1); relax =
  lr·λ_min(D^{−1/2}HD^{−1/2}), H the joint Hessian in (w₁, b₁, b₂) at the branch reached by damped Newton from the
  crossing state on the run's training set, D = diag(√v̂ + ε); χ = growth/relax.

## Predictions (scored separately per activation, on the crossing runs)

- **T-a.** At least 90% of the crossing runs cross at or above the bracket's lower end: s_cross ≥ s_lo.
- **T-b.** The median residual r = s_cross/s_glob − 1 lies within max(0.01, 0.25·|pred|) of pred = the median over the
  crossing runs of κ·χᵢ (runs with finite, positive χ).
- **UNRESOLVED** (both) with fewer than 30 crossing runs; T-b also UNRESOLVED with fewer than 30 usable χ.

## Registered, reported beside the primary verdicts (they do not replace them)

- **Sensitivity (crossing definition):** the same scoring with the crossing taken at the first step with dense
  G ≥ 1e−8 (guards against tail-level placements, cf. step 2).
- **Secondary arm (coverage):** because the calibration crossing rate (~50%) makes UNRESOLVED likely at 40 seeds, 160
  further fresh seeds **850,040–850,199** are run after the 40 primary seeds under the same protocol, and T-a / T-b are
  scored on the pooled 200 seeds as a separate, registered secondary verdict. The primary verdict is the one on the 40
  seeds the task specifies.
