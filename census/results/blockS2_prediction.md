# Registration: does the budget law extend to the Ren–Lim S²⊔S² setting?

**Written before any S²⊔S² run.** Date: 2026-09-22. Compute estimated first:
**~23 minutes** (3.4M steps at 0.40 ms/step), inside the 9 am 24th cutoff.

Marked **EXPLORATORY** in advance: this is a new setting registered on the last
experiment day, not a planned block.

## What is being tested

The budget law (T44, `budget_law_results.md`) is established **only for the 1D
fold task**: the solve rate at `a = 1.25` goes 0/200 at 2,000 steps to 100% at
80,000. Every link-setting artifact in this project (`linking_width3.csv`,
`linking_projected.csv`, 5,120 runs) is at a **single budget**, so the law has
never been tested where the topology claim actually lives.

## Construction

Ren–Lim's own, from `github.com/7pocheR/low_dimensional_topology`
(`exp_4_higher_dim_r5_multicopy/generate_linked_spheres_dataset.py`): the verified
embeddings `tildeA`, `tildeB` mapping `S² → ℝ⁵`, thickened by a uniform offset in
the normal space of radius `ρ`.

| setting | value |
|---|---|
| architecture | **width 5, depth 5**, single logit |
| activations | **ReLU** and **GELU** |
| `ρ` | **0.5** |
| `k` (points per sphere) | **10** per the brief — interpreted as `k = 10` *hundred*, i.e. **1,000 per class**, since 10 points cannot populate a thickened `S²`. Stated here because it is an interpretation, not a reading. |
| budgets | **1,000 / 4,000 / 16,000 / 64,000** |
| seeds | **20** per cell |
| optimiser | Adam, lr 1e-3, full batch |

Pilot check: between-class minimum distance **0.1753** at `ρ = 0.5`, so the
two-class problem is well-posed.

## Registered prediction

**S2-1 (the budget law extends).** The monotonic/non-monotonic gap —
`acc(GELU) − acc(ReLU)`, GELU being the non-monotonic one — **changes with
budget**, and specifically **narrows** as budget grows, because the 1D result says
budget buys what the activation otherwise supplies.

> Registered: **the gap at 64,000 steps is smaller than the gap at 1,000 steps**,
> and the difference between those two gaps exceeds **2 percentage points**.

**Falsifiers, both reported as such:**

- **Gap flat in budget** (change ≤ 2 pp): the budget law does **not** extend to
  the link setting, and the paper says the law is a 1D-fold result, as currently
  planned.
- **Gap widens** with budget: the opposite of the 1D behaviour, which would be a
  substantive negative result and must be reported as one.

**S2-2 (both activations improve).** Accuracy is non-decreasing in budget for
each activation separately. If not, the runs are not converging and S2-1 is
uninterpretable — a validity gate, not a result.

## Reporting

Whatever completes by the cutoff is reported and **labelled exploratory**.
Registered on the last experiment day with no prior pilot beyond the timing test
above, so it carries less weight than the pre-registered blocks and the paper
should present it as a first look, not as a tested claim.
