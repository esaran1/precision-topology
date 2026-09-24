# Registration: the small-scale limit of the conditional minimiser, widths 1 and 2

**Written and committed before any Δμ-maximiser is computed.** Theory: `math_note_v2.md` §10 (Lemma 1,
Corollaries 1–2). Producer: `src/scale_limits.py`, committed with this file.

## What is computed

- **D(α)**: D(α) = mean over class-1 (outer) points of cos(αx) minus the mean over class-0 (inner) points. It is
  computed on the population objective (800 points), on a grid α ∈ (0, 50] with step 0.001, then refined with
  golden section. α* = argmax |D(α)|.
- **The Δμ-maximiser, selected as Lemma 1 prescribes.**
  - Among parameters within 1e−9 of the maximal Δμ, take the one with the smallest Var(φ) (Corollary 1(ii)).
  - **This second-order selection is my addition**, derived in the lemma. It is flagged for the author's review.
  - The G of every type of maximiser is also reported (see below), so the outcome can be read without the
    selection.
- **Width 1** (a = 1.30, 1.35, 1.40, 1.45, 1.50, 1.60): maximise Δμ over (w₁, b₁) with s > 0.
  - Search: multistart BFGS, 2,000 restarts over w₁ ∈ [−10, 10] and b₁ ∈ [0, 2π), every restart retained. Then
    the Var selection among maximisers.
  - Report G (continuous windows, exact extrema) and G_n (data points).
- **Width 2** (a = 1.30, 1.50): maximise Δμ over (α₁, β₁, α₂, β₂, t, σ) with ‖ṽ‖₁ = 1.
  - Validation as in W0:
    - 2,000 restarts, every one retained;
    - a restart ladder 500 → 2,000 (the maximum unchanged to 1e−9);
    - an independent search in the alternative parametrisation (v unconstrained, normalised; CMA-ES) that must
      not exceed the retained maximum by more than 1e−9.
  - Then the Var selection among maximisers.
  - Report G and G_n of the selected maximiser, and of each maximiser type found: single unit (|t| = 1); two units
    with c ≠ 0; the cancelling pair (c = 0).
- **Numerical check** (the author's): the linear part of f_a contributes nothing to Δμ, so Δμ computed from φ must
  equal a·Σᵢ ṽᵢ·[mean_O sin(αᵢx + βᵢ) − mean_I sin(αᵢx + βᵢ)] to 1e−12 at every retained candidate.

## Registered outcomes

- **Width 2 (the prediction)**:
  - **If the selected Δμ-maximiser has G > 0**: width 2 has no placement threshold for f_a (Corollary 2). Both f_a
    arms are recorded as **not applicable**, like tanh, and **W1 and W4 are not run**. The W0 scan's low end is not
    needed for W1.
  - **If G ≤ 0**: a threshold exists, and W0 continues as designed.
- **Width 1 (the consistency check)**: the expected result is G ≤ 0 at all six a, because certified thresholds exist
  there. **If any a gives G > 0, that contradicts Corollary 2 against the certified thresholds: stop and report.**

## Disclosed prior exposure (this registration is not blind)

- The exploratory width-2 pilot (`width2_design.md`, revision 2) already found placed conditional minimisers at
  small scale. At R₂ = 0.03 with 200 restarts it retained the equal-weight cosine pair (α ≈ 1.79).
- The structure derived in §10, before this computation, shows that on the maximiser set the selected member is
  φ = const + a·sign(D(α*))·cos(α*x). The pilot's α ≈ 1.79 suggests that this member is placed.
- So the width-2 outcome is **largely predictable**. The value of registering is that the selection rule, the
  validation and the consequences are fixed before the number exists.
- **Without the second-order selection, the maximiser set M contains both unplaced members (single units, whose
  linear part makes G < 0) and, by the structure, placed members.** In that case the outcome would be reported as
  undetermined by Δμ alone.

## Result — computed 2026-09-24 with the committed producer (`python -m src.scale_limits`)

- **α* = 1.7913244** and D(α*) = −1.5710673 on the population data (`scale_limits_D.csv`).
- **Width 1 (the consistency check): passed.** The selected Δμ-maximiser has G ≤ 0 at all six a, as expected, since
  certified thresholds exist there (`scale_limits_width1.csv`).
  - G = −3.662, −3.610, −3.557, −3.505, −3.453 and −3.349 at a = 1.30, 1.35, 1.40, 1.45, 1.50 and 1.60.
  - The maximiser is w₁ = ±α*, b₁ = 3π/2.
  - The sine-only Δμ matches to ≤ 5e−12.
- **Width 2 (the prediction)** (`scale_limits_width2.csv`, `scale_limits_width2_maximisers_a*.csv`):
  - **Validation passed.**
    - The search maximum of Δμ equals the bound a·|D(α*)| to 2e−13.
    - The restart ladder (500 → 2,000) is unchanged to 1e−12.
    - The independent CMA-ES search does not exceed it.
    - The sine-only check holds to 1e−14.
  - **The maximiser set** has 902 retained maximisers at a = 1.30 and 904 at a = 1.50. By type:

    | maximiser type | number (1.30 / 1.50) | G range at a = 1.30 | G range at a = 1.50 |
    |---|---|---|---|
    | single unit | 546 / 547 | −3.66 (all) | −3.45 (all) |
    | pair with c ≠ 0 | 356 / 357 | −3.66 to +0.036 | −3.45 to +0.929 |

  - **The selected (Var-minimising) maximiser is the cancelling cosine pair (c = 0).**
    - It attains the maximal Δμ.
    - Its Var(φ) is at most that of every retained maximiser.
    - The random search does not land exactly on it: c = 0 is a measure-zero slice of the maximiser set.
  - **Its G is +0.8896 at a = 1.30 and +1.0265 at a = 1.50**, the same value for the data gap G_n.
- **Registered verdict: width 2 has NO placement threshold for f_a.** Both f_a arms are recorded as **not
  applicable**, like tanh. **W1 and W4 are not run.**
- **Caveat, stated with the verdict.** The verdict rests on the second-order (Var) selection of Corollary 1(ii). I
  added that rule; it is flagged for the author. The maximiser set itself contains unplaced single units and placed
  pairs, so **Δμ alone does not decide the outcome**; the second-order term does.
  - The same tie explains the pilot's occasional unplaced retained minimisers at very small scale. At first order
    the single unit and the pair tie, and at small s their loss difference is (s²/8)·ΔVar, which a finite restart
    budget can miss.
