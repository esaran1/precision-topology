# Registration: Block K — family B under a scale-invariant normalisation

**Written before any normalised quantity has been recomputed.** Date: 2026-09-22.

Three family B claims were suspended on 2026-09-21 because every `Ĝ`-derived
number used a **box-limited** search: `f_α(t) = max(t, αt)` is positively
homogeneous of degree 1, so `G(c·w₁, c·b₁) = c·G(w₁, b₁)` and the unrestricted
supremum is **infinite**. The committed values were exactly `3.2|α|` — the box
size in disguise, an implicit normalisation nowhere stated.

Suspended: **`β = 0.9935`**, the **900–1,200× overshoot**, the **AUC comparison**.
Not suspended: the **onset exponent +0.25**, which came from solve rates and never
touched `Ĝ`.

## The normalisation, fixed by homogeneity rather than chosen

Homogeneity gives `w₂ · f(w₁x + b₁) = (w₁w₂) · f(x + b₁/w₁)` for `w₁ > 0`, so the
network depends on the weights only through the scale-invariant direction `b₁/w₁`
and the **product `|w₁w₂|`**. The amplification variable is therefore the product,
never `|w₂|` alone. `family_b_correction.md` already establishes, independently of
any box:

    max class gap at |w₁| = 1  =  0.4·|α|  exactly
    (fitted exponent 1.000000, ratio constant at 0.4 over three decades)

So the scale-invariant replacement for `Ĝ` is

    Ĝ_norm(α) := sup over placements of G(w₁, b₁) subject to |w₁| = 1  =  0.4·|α|

and the margin capacity becomes

    R_B := |w₁w₂| · Ĝ_norm(α) / 2  =  |w₁w₂| · 0.4|α| / 2

This is the unique normalisation that (i) is finite, (ii) is invariant under the
homogeneity rescaling `(w₁,b₁,w₂) → (c·w₁, c·b₁, w₂/c)`, and (iii) reduces to the
family A definition when the activation is not homogeneous. **It is forced, not
fitted**, which is why it can be registered before computing anything.

## Registered predictions

**K-1 (β survives).** Recomputed under `Ĝ_norm`, family B's depth exponent is

> **`β_B = 1` exactly**, by homogeneity — `Ĝ_norm ∝ |α|` is linear in `|α|` with
> fitted exponent **within 0.01 of 1.000**.

The suspended `β = 0.9935` was measured under the box and is expected to be
*confirmed*, not overturned: homogeneity forces linearity regardless of box size,
so this one number was accidentally right. **Registered: it survives.**

**K-2 (the overshoot is a real effect, but its SIZE changes).** The 900–1,200×
figures divided a reached `|w₁w₂|` by a requirement computed from the box-limited
`Ĝ`. Under `Ĝ_norm` the requirement changes by the constant factor
`3.2|α| / 0.4|α| = 8`, so:

> the overshoot factors should fall by **exactly 8×**, to roughly **115–145×**,
> and remain **> 10×** at every `α`.

> **Falsified** if the recomputed overshoot is not `old/8` to within 1%, which
> would mean the old numbers were not simply the box factor; or if any overshoot
> falls below 10×, which would remove the "training massively overshoots the
> requirement" claim.

**K-3 (the AUC comparison, which is the one that may not survive).** The AUC
compared `R` against `|w₂|` and `Ĝ` as separators of solved/unsolved runs. Under
homogeneity `|w₂|` alone is **meaningless** for this family, so:

> `R_B` should separate solved from unsolved with **AUC ≥ 0.90**, and should beat
> raw `|w₂|` by a margin of **≥ 0.10**.

> **Falsified** if `AUC(R_B) < 0.90`, or if the margin over `|w₂|` is under 0.10.
> In that case the family B AUC claim is **removed from the paper**, per the
> standing instruction, rather than reported with a caveat.

**K-4 (the law's status is unchanged).** `family_b_correction.md` records that
family B is a **counterexample** to `ε_onset ~ B^(−α/β)`, not its limiting case:
`β_B = 1` predicts an onset exponent of **−1.1172** while the measured value is
**0.0000**. Block K must not disturb this.

> Registered: the recomputation **leaves the counterexample standing**. The onset
> exponent is a solve-rate quantity and never used `Ĝ`, so it cannot move.
> **Falsified** if the normalisation changes the measured onset exponent at all.

## Decision rule, stated in advance

Per the standing instruction: **any claim that does not survive under `Ĝ_norm`
comes out of the paper entirely**, not in with a caveat. K-3 is the one at genuine
risk. K-1 and K-4 are near-certain by homogeneity and are registered mainly so
that a surprise would be visible.
