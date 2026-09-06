# Part 2a: the margin gap. Option 1 unavailable, option 3 succeeds.

**Provenance: raised in external expert review.** The theorem is stated at
fixed margin m; the experiments count sign correctness, which is the m -> 0
limit where |w2| >= 2m/(kappa*D(a)) degenerates to |w2| >= 0 and says nothing.

## Option 1 -- does the finite sample force a minimum margin? NO.

`fold1d.solves()` evaluates on `torch.linspace` grids over the **continuous
intervals** [-0.8, 0.8] and +-[1.2, 2.0]. It is a dense approximation to the
exact-region condition, not a test on a fixed finite point set, so there is no
inter-point spacing from which to bound a margin away from zero. Refining the
grid can only remove solutions, never establish a floor.

**Confirmed empirically -- no floor exists.** Margins of solutions the
experiments count as solved:

| a | n | min margin | p05 | median |
|---|---|---|---|---|
| 1.35 | **1** | 0.083 | -- | 0.083 |
| 1.50 | 48 | **0.0038** | 0.019 | 0.112 |
| 2.00 | 85 | 0.167 | 0.428 | 0.774 |
| 3.00 | 80 | 0.689 | 1.308 | 1.609 |

**The a = 1.35 row is n = 1 and is not a distribution.** Only a >= 1.5 is
informative; a = 1.5 is the straddling case.

What this buys, stated precisely: the criterion **is** m -> 0, the theorem
**does** degenerate there, and empirically the found solutions sit at margins
where the bound is informative -- **except near the onset, which is exactly
where section 6 invokes it** (at a = 1.5 the margin tail reaches 0.0038).
That supports using the theorem **descriptively**, relying on an observed
regularity rather than a proved one. It is not a connection between the
theorem and the criterion.

## Option 3 -- redefine separation at fixed m > 0. SURVIVES.

Survival band registered **before** the data, derived from the m = 0 fit's own
uncertainty rather than chosen: statistical 95% CI +-0.0725 (SE 0.0370, 6
bracketed cells, dof 4) combined in quadrature with the grid-resolution
component +-0.1140 (mean ln-step 0.474 over ln-span 4.16) = **+-0.1351** about
-0.7340, i.e. **[-0.8690, -0.5989]**.

| criterion | onsets (B = 2k / 8k / 32k) | exponent | inside band |
|---|---|---|---|
| m = 0 (sign) | 1.60 / 1.18 / 1.06 | **-0.8305** | yes |
| m = 0.01 | 1.60 / 1.18 / 1.06 | **-0.8305** | yes |
| m = 0.05 | 1.70 / 1.20 / 1.08 | **-0.7823** | yes |

**Verdict under the registered rule: CLEAN SURVIVAL.** Not the mixed case the
registration anticipated and pre-emptively refused to score as success.

- **m = 0.01 is bit-identical to m = 0** in every cell: at this margin the
  criterion change costs nothing at all.
- **m = 0.05 shifts every onset up by exactly one grid step** (1.60->1.70,
  1.18->1.20, 1.06->1.08), a uniform translation that leaves the exponent
  within 0.05 of the sign-correctness value.

**So the empirical criterion can be restated at fixed margin, and section 6's
use of the theorem is repaired rather than lost.** The theorem applies to the
experiments when separation is defined at m > 0, and the budget law is
unchanged by that redefinition.

## Bookkeeping

This 3-cell m = 0 fit gives -0.8305 against the committed 6-cell value
-0.7340. The difference is the **cell subset** (2k/8k/32k here versus six
budgets 2k-128k), not the margin criterion -- the m = 0 and m = 0.01 fits are
identical to four decimals. The committed -0.7340 remains the better-supported
estimate; the three-cell fits are used here only to compare criteria on
matched cells.
