# Track T v3: registered test at the attained switch (frozen before any training)

**Registration, 2026-09-26 ~03:00 EDT, committed before any registered training.**
- One registered test, approved by the author; the author's text is binding.
- Producer: `src/simplicity_bias_v3.py`. Tests: `tests/test_simplicity_bias_v3.py`, which cover the decision rules on
  constructed PASS, FAIL and UNRESOLVED cases, the initialisation rule, χ and the training recorder.
- Frozen inputs: `results/simplicity_bias_v3/frozen_inputs.json`, with SHA-256 in `frozen_inputs.sha256`.
- **Deadline:** scored by 04:15 EDT, or stopped and committed.

## Frozen inputs

- **s_switch = 3.5914**, exactly 3.5913755424683727.
  - This is the committed v2 s_q (`simplicity_bias_v2/frozen.json`, SHA-256 deb8853e…).
  - The attained, weight-decayed path jumps across it: ρ₂ goes from 0.182 to 0.448, and G₊ from −0.153 to +0.071.
- **λ = 1e−4.**
- **q = 0.3914**, exactly 0.3914103370353161.
- The same data as v1 and v2: p = 0.2, 400 points per class.
- The same ρ₂ definition as the path: the deterministic S-randomisation share, via `sb.feature_usage` and
  `v2.rho2_batch`, which are tested equal.
- No κ.

## Prediction

Training acquires the slab feature just above the switch. Its crossing scale lies at or above s_switch, within
[1.00, 1.25]·s_switch.

## Protocol

- **Seeds:** 2,720,000–2,720,039 (40 fresh seeds). Integer search of src/, results/ and tests/ found no use of them.
- **Initialisation rule, identical for every seed:**
  1. `torch.manual_seed(seed)`.
  2. PyTorch default initialisation of Linear(2, 4) and Linear(4, 1), float64.
  3. If the output scale ‖w₂‖₁ exceeds **0.5·s_switch = 1.7957**, multiply the output weights by 1.7957/‖w₂‖₁.
     The output bias is unchanged.
  - The raw scale is recorded.
- **Optimisation:**
  - Full batch on the 800 points.
  - Adam with lr 0.01 (the v2 value), betas (0.9, 0.999), eps 1e−8.
  - Loss: BCE-with-logits (mean) + (λ/2)(‖W‖² + ‖c‖²), the same λ and the same decayed parameters as the landscape.
- **Budget:** stop at the first step with s ≥ 3·s_switch = 10.77, a rule on s only, or at 40,000 steps.
- **Every-step detection.** Every step's parameters are recorded, and ρ₂ is evaluated at **every** recorded step after
  training by `v2.rho2_batch`. This is exact, with no cadence.
- **Event (crossing)** = the first **upward passage** of q by ρ₂: the first step t with ρ₂(t) ≥ q after an earlier
  step with ρ₂ < q. This is the v2 definition.
  - "First reaching q" is read as reaching it from below, because a random hidden layer can start with ρ₂ ≥ q, which is
    not acquisition.
  - A run with no upward passage within the budget is not a crossing run.
  - Also reported, not scored: the literal first step with ρ₂ ≥ q, and the number of runs with ρ₂(0) ≥ q.
- **Crossing scale:** s = ‖w₂‖₁ at the crossing step.
- **χ at crossing** (a validity quantity only):
  - χ = (d log s/dt) / (lr·λ_min(P^{1/2}HP^{1/2})).
  - d log s/dt = (log s(t_c) − log s(t_c − w))/w, with w = min(100, t_c).
  - H is the **committed** Hessian of the weight-decayed fixed-scale loss at the landscape minimiser at the switch, in
    coordinates (W, c, b), from v2's `frozen.json`.
  - P is the run's Adam 1/(√v̂ + ε) at the crossing, as within-block medians (`v2.block_p`).

## Criteria (`decide`, tested)

- **C1:** the fraction of crossing runs with s_cross ≥ 3.5914 is at least 0.90.
- **C2:** median(s_cross/3.5914) lies in [1.00, 1.25].
- **Validity:** at least 30 crossings of 40 **and** median χ at crossing ≤ 0.06.
  - Otherwise **both** criteria are UNRESOLVED.
  - The would-be verdicts are reported, labelled as such.

## Note recorded before training

- The committed H has smallest eigenvalue 1.0e−4, which equals λ.
- That direction belongs to an idle hidden unit held only by the decay (v2 freeze record).
- λ_min(P^{1/2}HP^{1/2}) may therefore be small, and χ large, so the χ validity condition may leave the test UNRESOLVED.
- This follows from the author's χ definition and the committed H. It is not changed.
