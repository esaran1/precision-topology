# Block K: family B under the scale-invariant normalisation

Scored against `results/blockK_prediction.md` (registered `04a3843`, before any
normalised quantity was recomputed). Artifacts: `blockK_family_b_normalised.csv`,
`blockK_auc.csv`.

**Verdict: two claims survive, one is REMOVED — and it was never supportable,
for a reason independent of the box.**

---

## The normalisation, and confirmation that the box was the whole story

`f_α(t) = max(t, αt)` is positively homogeneous, so the network depends on the
weights only through `b₁/w₁` and the product `|w₁w₂|`. The forced normalisation is
`Ĝ_norm(α) = 0.4|α|` (max class gap at `|w₁| = 1`), giving
`R_B = |w₁w₂|·Ĝ_norm/2`.

The committed box-limited values were exactly `3.2|α|`:

| `α` | committed `Ĝ` | `3.2\|α\|` | `Ĝ_norm = 0.4\|α\|` | ratio |
|---|---:|---:|---:|---:|
| −1.00 | 3.119598 | 3.200000 | 0.40 | 7.799 |
| −0.50 | 1.600000 | 1.600000 | 0.20 | 8.000 |
| −0.25 | 0.800000 | 0.800000 | 0.10 | 8.000 |
| −0.10 | 0.320000 | 0.320000 | 0.04 | 8.000 |
| −0.05 | 0.160000 | 0.160000 | 0.02 | 8.000 |

**Exactly 8× at every `α`** (the −1.00 row is 7.799 because its box-limited search
hit the box corner rather than the analytic value). The suspension diagnosis was
correct: the committed `Ĝ` was the box size in disguise.

## K-1: `β_B = 1` — PASSES, and the old number was accidentally right

`Ĝ_norm ∝ |α|` with fitted exponent **1.000000**, against a registered tolerance of
0.01. Homogeneity forces linearity regardless of box size, so the suspended
`β = 0.9935` was **confirmed, not overturned** — it was right by accident, since
any box gives a linear relation.

## K-2: the overshoot — substance PASSES, my registered range was wrong

| `α` | budget | reached `\|w₁w₂\|` | old overshoot | `req_norm` | **new overshoot** | ratio |
|---|---:|---:|---:|---:|---:|---:|
| −0.050 | 2,000 | 94.1 | 94× | 8.00 | **11.8×** | 7.991 |
| −0.050 | 32,000 | 1,159.7 | 1,160× | 8.00 | **145.0×** | 8.002 |
| −0.010 | 32,000 | 5,742.3 | 1,148× | 40.00 | **143.6×** | 7.997 |
| −0.005 | 32,000 | 11,141.5 | 1,114× | 80.00 | **139.3×** | 7.999 |
| −0.002 | 32,000 | 23,051.5 | 922× | 200.00 | **115.3×** | 7.999 |

Registered: falls by exactly 8× (**holds, 7.99–8.00 in every row**) and stays
above 10× (**holds, minimum 11.8×**).

**My registered range "roughly 115–145×" was arithmetically wrong.** I divided
only the large overshoots and omitted the 2,000-budget row, which is 11.8×. The
correct statement is **11.8× to 145×**. The two substantive predictions — the
exact 8× factor and the >10× floor — both pass; the stated range does not, and the
error was mine in the registration, not in the data.

## K-3: the AUC comparison — REMOVED, and it was never supportable

Registered: `AUC(R_B) ≥ 0.90` **and** a margin of `≥ 0.10` over raw `|w₂|`.

| `α` | solved | `AUC(R_B)` | `AUC(\|w₁w₂\|)` | `AUC(\|w₂\|)` | `AUC(\|w₁\|)` | band runs |
|---|---:|---:|---:|---:|---:|---:|
| −1.00 | 113/200 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **2** |
| −0.50 | 111/200 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **2** |
| −0.25 | 116/200 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **2** |
| −0.10 | 119/200 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **2** |
| −0.05 | 119/200 | 0.9830 | 0.9830 | 0.9778 | 0.9874 | 69 |

**The AUC criterion passes and the margin criterion fails**: pooled,
`AUC(R_B) = 0.9990` against `AUC(|w₂|) = 0.9901` — a margin of **0.0089**, an
order of magnitude short of the registered 0.10. Within `α`, the product beats
`|w₂|` in **1 of 5** cells.

**Two independent reasons the claim cannot be made, both fatal:**

1. **Within fixed `α`, `Ĝ_norm` is a constant**, so `R_B` is `|w₁w₂|` times a
   positive number and their AUCs are **identical by construction** (confirmed:
   every row above). There is no sense in which `R_B` "beats" the product; the
   only meaningful comparison is the product against `|w₂|` alone.
2. **Four of five cells are perfectly separable**, with enormous gaps — at
   `α = −1` the largest unsolved `|w₁w₂|` is **0.295** and the smallest solved is
   **7.26**, a 25-fold gap with **2 runs in between**. Every monotone variable
   scores 1.0000. **T54's own registered criterion required ≥ 20 transition-band
   runs to score an AUC comparison**, and family B has 2 in four of five cells.

So the box was a *second* defect. **The family B AUC claim was never supportable
even with a correct `Ĝ`**, because the design has almost no runs where the
comparison could discriminate. Per the standing instruction, it **comes out of the
paper entirely** rather than in with a caveat.

The one cell with real contrast (`α = −0.05`, 69 band runs) gives product 0.9830
against `|w₂|` 0.9778 — a margin of 0.0052, and `|w₁|` alone scores **higher**
(0.9874) than either. That single cell cannot carry the claim, and it points the
wrong way.

## K-4: the law's status is unchanged — PASSES

The onset exponent is computed from solve rates against budget
(`budget_alpha.csv`: `a, budget, w2_med, w2_p25, w2_p75, stall_frac, rate, n`).
**No `Ĝ`, gap, or `w₁` column is read**, so the normalisation cannot move it. The
counterexample recorded in `family_b_correction.md` stands unchanged: `β_B = 1`
predicts an onset exponent of **−1.1172** while the measured value is **0.0000**.
Family B remains a **counterexample** to `ε_onset ~ B^(−α/β)`, not its limiting
case.

## Net effect on the paper

| claim | status |
|---|---|
| `β_B = 1` (was `0.9935`) | **RESTORED** — confirmed under the forced normalisation |
| overshoot, training exceeds the requirement | **RESTORED** at **11.8× to 145×** (was 922–1,160×, inflated 8× by the box) |
| family B AUC comparison | **REMOVED** — not supportable at any `Ĝ`; the design lacks transition-band runs |
| onset exponent +0.25, and the counterexample | **UNCHANGED** — never depended on `Ĝ` |

The suspension is lifted for the first two and made permanent for the third.
