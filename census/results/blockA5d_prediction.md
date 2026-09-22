# Registration: the expressivity threshold versus the capability onset in five dimensions

**Written before any run of this block.** Date: 2026-09-22. Everything below —
construction, network, protocol, success criterion, grid, onset definition,
two-stage seeding rule and predictions — is fixed here and not revised.

---

## Hypotheses of Ren & Lim Theorem 4.7, checked before running

The theorem: a width-`d` feedforward network with affine maps and continuous
coordinate-wise non-decreasing activations cannot linearly separate `F(M)` and
`F(N)` when `M^m, N^n ⊂ ℝ^d` are disjoint closed oriented submanifolds with
`m + n + 1 = d` and `link(M, N) ≠ 0`.

| hypothesis | here | holds? |
|---|---|---|
| complementary dimension | `M = N = S²`, `m + n + 1 = 2 + 2 + 1 = 5 = d` | yes |
| width `d` | every hidden layer is `Linear(5, 5)` | yes |
| activation continuous, coordinate-wise non-decreasing | `f_a(x) = x + a sin x`, `f_a'(x) = 1 + a cos x ≥ 1 − |a| ≥ 0` for `|a| ≤ 1` | yes, for `a ∈ {0.9, 1.0}` |
| linear separation | the readout is one linear map on the width-5 features | yes |
| linked, disjoint cores | Ren–Lim's `tildeA`, `tildeB`; core separation `2(√2 − 1) = 0.828427`, measured `0.828459` | yes (the linking number is their verified claim) |

For `a > 1`, `f_a' = 1 + a cos x` changes sign, so `f_a` has strict local
extrema on open intervals (at `x = π ∓ arccos(1/a)`), which is the hypothesis of
their Appendix F.2. **The expressivity threshold is therefore exactly `a = 1`.**

**Scope.** The theorem concerns the core spheres. We measure a thickened,
held-out sample, so zero perfect runs for `a ≤ 1` is **consistent with** the
barrier rather than an instantiation of it.

## Construction — their released generator, not the T69 one

Ren & Lim's `exp_4_higher_dim_r5_multicopy/train_width_scaling_v7.py::
generate_multi_copy_dataset`, reimplemented line for line:

- `n = 2`, `d = 5`, **`k = 10` copies** of the linked pair, centred at the first
  10 points of the L1-ordered integer grid in `ℝ⁵` scaled by **spacing 10**
- **targeted thickening** (their default): A is thickened in its Y-subspace
  (last 2 coordinates), B in its X-subspace (first 2), each by a uniform draw
  from the 2-ball of radius **`ρ = 0.5`**
- classes balanced, points split evenly across copies

**Well-posedness, checked**: under targeted thickening the two solids of one
copy coincide only if `max(|X_A|, |Y_B|) ≤ ρ` at a shared `Z`, whose minimum
over `Z` is **1.000000**. So the solids are disjoint for every `ρ < 1`, and
`ρ = 0.5` is safe. Copies are 10 apart against a per-copy extent below 2.5.

**This supersedes the T69 construction for any comparison.** T69 used a single
copy with **ambient** 5-ball thickening at `ρ = 0.5`, where disjointness needs
`ρ < 0.4142` — so T69's two class regions **overlap**. That is recorded as a
correction to T69 in `results/blockS2_overlap_correction.md`.

## Network and training

- **their depth convention**: five hidden blocks `Linear(5,5) → act`, then a
  linear readout
- readout: **one logit, BCE**. Their code uses two logits with cross-entropy,
  which depends only on the logit difference, so function class and loss are
  identical; the single readout vector is the analogue of `w₂`
- activations: `f_a` coordinatewise for `a ∈ {0.9, 1.0, 1.05, 1.1, 1.2, 1.35,
  1.5, 2.0, 3.0}`; controls **ReLU** and **GELU**
- **protocol matching the T69 sweep**: full-batch Adam, lr 1e-3, constant, float32,
  PyTorch default initialisation, one CPU thread per run
- **training set**: 10,000 points (1,000 per copy, 500 per class per copy). Their
  script uses 10,000 per copy (100,000 at `k = 10`); full batch at that size is
  not feasible here, and this is recorded as a deviation.
- **budgets 1,000 / 4,000 / 16,000 / 64,000 steps**, taken as checkpoints of one
  64,000-step run. With constant-lr full-batch Adam on one CPU thread the
  trajectory is deterministic, so the checkpoint at step `B` *is* the `B`-step run.

## Success criterion, fixed before running

**A run is perfect at budget `B` iff it makes zero errors on a held-out sample of
20,000 points** (2,000 per copy, balanced), drawn from the same generator with a
seed disjoint from training. Held-out accuracy is reported alongside.

Justification: zero errors in `n = 20,000` bounds the true error rate below
`3/n = 1.5 × 10⁻⁴` at 95% (rule of three). Every imperfect run in T69 had error
≥ 5 × 10⁻³, and a network with true error 10⁻³ would be detected with
probability `1 − (1 − 10⁻³)^{20000} > 1 − 10⁻⁸`. A sample smaller than a few
thousand could not tell a network that misclassifies one link's entanglement
region from a perfect one.

## Grid and seeding

- 9 values of `a` + ReLU + GELU, **20 seeds** each.
- **Two-stage rule, fixed now**: after stage 1, for each budget, the two grid values
  of `a` that bracket the onset each get **20 more seeds (40 total)**, and the onset
  is recomputed on stage-2 counts. If no onset brackets at a budget, no seeds are
  added there.

**Onset**: `a_on(B)` = the smallest grid `a` at which at least half the runs are
perfect, **bracketed** only if a strictly smaller grid `a` has fewer than half.

## Registered predictions

- **A1.** For `a ≤ 1` (`a ∈ {0.9, 1.0}`): **zero perfect runs at every budget.**
- **A2.** At each budget a **bracketed onset `a_on(B) > 1`** exists.
- **A3.** `a_on(B)` is **non-increasing in `B`**, and strictly lower at 64,000 than
  at 1,000.
- **A4 (positive control).** At `a = 3.0` and `B = 64,000`, **at least one**
  perfect run.
- **A5 (exploratory, direction only).** With `ε_on = a_on − 1`, fit
  `ε_on ∝ B^{−γ}` over the bracketed budgets. The readout-scale proxy is the
  **Frobenius norm of the final linear layer** (the analogue of `w₂`), chosen now:
  `α_L` = slope of log median `‖W_out‖_F` against log `B`, over the `a > 1` runs in
  the onset region. Registered **direction only**: `γ` and `α_L/β` with
  `β = 3/2` have the same sign, both positive.

## Protocol comparison (private, not paper-facing)

One added cell: **GELU at `k = 10` under their Appendix G.2 protocol**, as already
implemented in `src/author_protocol.py` (Adam 1e-3, batch 128, up to 800 epochs,
patience 150 on validation accuracy, best checkpoint restored), on **6,000 points
split 80/20**, 20 seeds, scored by the same 20,000-point held-out criterion. It is
compared against GELU at our `B = 64,000`. Their Table 4 reports 91.1% best for
GELU at `k = 10`. **This concerns their published paper, so the result goes only
in a private note for the co-author, not in any paper-facing file.**

## Stop and report if

- **any perfect run occurs at `a ≤ 1`** — this contradicts the theorem and almost
  certainly means a bug or a held-out sample too small to detect errors;
- **no onset brackets at any budget**.
