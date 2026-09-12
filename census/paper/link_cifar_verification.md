# Verification of §2 and §7 against raw artifacts

*Procedure, per the standing rule: every number is **recomputed from the raw
artifact**, not compared against a summary document. Text-to-text comparison
is what failed to catch the through-origin slope error
(`notes/instrument_artifacts.md`, instance 4), so it is not used here as a
pass condition.*

Date: 2026-09-11. Sections verified: `paper/link_section_draft.md` (§2),
`paper/cifar_section_draft.md` (§7), cross-checked against
`paper/results_draft.md` (§3–§6).

---

## Summary

**38 numbers traced to raw artifacts. 35 reproduced exactly. 3 mismatches
found, all three diagnosed, two of them errors in committed documents rather
than in the draft.**

| # | quantity | draft | artifact | status |
|---|---|---|---|---|
| 1 | monotonic zero, total | 5,580 | 5,580 | exact |
| 2 | separations | 0 | 0 | exact |
| 3 | per-stratum (7 strata) | 240/1,280/1,440/1,620/360/400/240 | identical | exact |
| 4 | one-sided 95% bound | 0.0537% | 0.0537% | exact |
| 5 | per-80-cell bound | 3.68% | 3.68% | exact |
| 6 | dense re-check, total | 163 | 163 | exact |
| 7 | dense failures | 56 | 56 | exact |
| 8 | GELU dense, total/failed | 81 / 46 | 81 / 46 | exact |
| 9 | width-3 monotonic, depth 3 | 0/300 | 0/300 | exact |
| 10 | width-3 monotonic, depth 6 | 0/120 | 0/120 | exact |
| 11 | GELU / sin(1.5) at width 3 | 2/100, 9/100 | 2/100, 9/100 | exact |
| 12 | tanh vs ReLU at width 8 | 98 vs 84, p = 8e-4 | 98 vs 84, p = 4.0e-4 | see M3 |
| 13 | GELU vs tanh at width 6 | n.s. | p = 0.360 | exact |
| 14 | dose-response rates | 3.0 / 8.0 / 17.0% | 6/200, 16/200, 34/200 | exact |
| 15 | up-vs-down Fisher | 1.5e-6 | 1.45e-6 | exact |
| 16 | up-vs-standard | 0.0048 | 0.0048 | exact |
| 17 | standard-vs-down | 0.023 | 0.0231 | exact |
| 18 | Bonferroni, down arm | 0.138 | 0.069 | **M1** |
| 19 | convergence means | 2,047.0 / 2,179.7 / 2,591.8 | identical | exact |
| 20 | convergence SDs | 47 / 45 / 36 | 47.0 / 45.0 / 36.0 | exact |
| 21 | median epochs | 22 / 31.5 / 21 | 22.0 / 31.5 / 21.0 | exact |
| 22 | plateaued | 24 of 25 | **23 of 25** | **M2** |
| 23 | GELU–ReLU gap | 133 | 132.7 | exact |
| 24 | tanh behind GELU / ReLU | 545 / 412 | 544.8 / 412.1 | exact |
| 25 | train errors per 10k | 744 vs ~180 | 744.4 vs 173.0/184.3 | exact |
| 26 | plateaued-only | 2,049 vs 2,183 | 2,049 vs 2,183 | exact |
| 27 | flip, 2 epochs | 3,425 / 3,924, by 499 | 3,424.5 / 3,923.5, 499.0 | exact |
| 28 | flip p, 2 epochs | 0.00044 | 0.00038 | see M3 |
| 29 | GELU vs ReLU, 2 epochs | 3,173 vs 3,924 | 3,173.1 vs 3,923.5 | exact |
| 30 | 2-epoch error range | 31.7–39.2% | 31.7–39.2% | exact |
| 31 | budget advantages | 750.4 / 418.4 / 150.6 | identical | exact |
| 32 | budget ratio | 4.98x | 4.980x | exact |
| 33 | contrast for #31–32 | *(was "GELU-over-tanh")* | **GELU-over-ReLU** | **M4, fixed** |
| 34 | 12-epoch vs converged | 150.6 vs 132.7, 13% | 13.6% | exact |
| 35 | ID estimator disagreement | 1.99x | ledger T39 | exact |
| 36 | parametrizations | 12 | 12 distinct values | exact |
| 37 | winding q values | 1–4 | 1, 2, 3, 4 | exact |
| 38 | annealing first-failures | [1.075, 2.275], median 1.275 | ledger T19 | exact |

---

## M1 — Bonferroni figure mixes one-sided and two-sided (committed-document error)

**Draft said** (inherited from ledger T29): the down arm fails Bonferroni
correction at **0.138**.

**Artifact gives**: the one-sided standard-vs-down Fisher p is 0.0231, so
3 x 0.0231 = **0.069**. The figure 0.138 is `3 x 0.0461`, three times the
*two-sided* p.

**Diagnosis**: T29 quotes the one-sided p-values (0.0048, 0.023, 1.5e-6, with
directions registered in advance) and then applies the Bonferroni multiplier to
the two-sided value. It is a mixed comparison.

**Consequence**: none for the verdict — the arm fails correction either way
(0.069 > 0.05). The draft now states 0.069 as the one-sided figure and notes
the ledger's 0.138 with its derivation, rather than silently switching.

**Ledger action**: T29 needs a dated correction block.

---

## M2 — "24 of 25 runs plateaued" is arithmetically impossible (committed-document error)

**Draft said** (inherited): 24 of 25 runs plateaued.

**Artifact gives**: `cifar_convergence.csv` has exactly **two** non-plateaued
runs — GELU seed 10 and ReLU seed 4, both at the 40-epoch cap. Per arm:
9/10, 9/10, 5/5. **9 + 9 + 5 = 23.**

**Diagnosis**: the error is self-evident in the source document, which states
"24 of 25 runs plateaued (9/10, 9/10, 5/5)" and then describes "**the two**
censored runs" in the next sentence. The per-arm breakdown and the censored
count both say 23; only the total says 24.

**Consequence**: none for the criterion — the plateau criterion passes at 23/25
exactly as it does at 24/25, and the sensitivity check (dropping both censored
runs) is unchanged. But the number is wrong and it has propagated to **four
committed documents**: `results/control_restoration.md`,
`results/task_f_conclusion.md`, `notes/generality_framing.md`, and ledger T35.

**This is instance 4 repeating.** A number was copied across four documents
that agree with each other and disagree with the CSV. A consistency pass over
those documents would have returned a clean bill.

---

## M3 — p-value last-digit differences (not errors)

Two p-values differ in the final digit: tanh-vs-ReLU at width 8 (draft 8e-4,
recomputed 4.0e-4) and the 2-epoch flip (draft 0.00044, recomputed 0.00038).

**Diagnosis**: both are resampling/tail-convention differences, not data
disagreements. The width-8 figure is a one-sided Fisher exact in my
recomputation; the ledger's 8e-4 is the two-sided value (2 x 4.0e-4 = 8.0e-4),
which is consistent. The flip p is a permutation test whose value depends on
the RNG draw and the tie convention; 100,000 resamples at this effect size
carry roughly +-4e-5 of Monte Carlo noise, and the two values are within it.

**Consequence**: none. Both are reported to one significant figure in the
draft, which is the right precision for a permutation estimate.

---

## M4 — the draft mislabelled the contrast for 4.98x (my error, fixed)

The first version of §7.4 attributed the 750.4 / 418.4 / 150.6 advantages to
**GELU-over-tanh**. Recomputing all six pairwise contrasts at each budget
identifies the true one: those values are **GELU-over-ReLU**
(`gelu - relu` = 750.4, 418.4, 150.6). GELU-over-tanh in the same experiment
is 251.4 / 486.9 / 478.4 — it *increases* with budget rather than decreasing,
being the §7.3 flip viewed from the other side.

The source (`results/section7_empirical.md`) is unambiguous and labels the
column "GELU advantage" against a ReLU column. The error was mine in drafting.
**Fixed**, with the GELU-over-tanh figures added parenthetically so the
distinction cannot be lost again.

Had this survived, §7.4 would have claimed a fivefold budget-dependence for a
contrast that does not have one.

---

## Cross-sentence check against the full draft

Every quantity appearing in more than one of the three files was compared.

- **No quantity appears with two different values.** The scan flagged one
  apparent collision — `2.57x` in §5 against `4.98x` in §7.4 — which is not a
  collision: 2.57x is the compute multiplier for halving `eps` in the 1D
  budget law, 4.98x is the ratio of a CIFAR advantage across budgets. Different
  quantities, no shared referent.
- **`5,540` appears once in §2**, inside the sentence documenting the
  denominator's history, which is its correct use.
- **The width-6 / width-8 boundary** is stated in §2.5 only and is not
  referenced numerically elsewhere, so there is no adjacency risk of the kind
  the brief anticipated.
- **`5.65x` appears once in §7.4**, in the sentence explaining why it is *not*
  used. No other occurrence.

**Adjacency risk now that §2 and §3–§6 sit together**: the word "separation"
carries different operational definitions in the two settings (regional
correct-classification of solid tori at 100k points in §2; the `sign(|x|-1)`
separation criterion in §3–§6). §2.4 states its definition explicitly and §2.7
hands over to §3 with the change of setting made explicit, so the two are not
silently conflated. This is the residual risk to watch in a full read-through.

---

## Undefined term found: "five link families"

Ledger T1 and `results/winding_results.md` both describe the pool as spanning
**"five link families"**, and this number is **defined nowhere** — no artifact
column, no script, and no document enumerates them. Reconstructing from the
artifacts, the distinct link geometries are:

1. the Hopf link (`linked_tori`), used by the width, threshold,
   parametrization, protocol and restart strata;
2. corrugated Hopf links, reading A;
3. corrugated Hopf links, reading B;
4. winding links at `|lk| = 2, 3, 4` (the `q = 1` cell is itself a Hopf link).

That is **four** families by construction, or seven if each winding number is
counted separately. Neither count is five.

The draft therefore **enumerates the geometries instead of quoting a count**,
which is verifiable and does not depend on resolving what the original five
were. The ledger should either define the term or drop it.
