# Provenance: artifacts that backed reported numbers without a producer

Found 2026-09-22. The verifier's provenance check (added for `paper-submitted`)
passed if **any** committed script **mentioned** an artifact's name. Scripts that
only *read* an artifact therefore counted as its producer, and artifacts read
through a variable (`for fn in ("r_pooled.csv", "r_adamw.csv")`) were never
checked at all. The check reported "every verified artifact has a producing
script". **That statement was false.**

## The replacement check

`src/verify_ledger.py::PRODUCERS` maps every artifact that backs a reported
number (the 69 data artifacts in the supplementary plus this block's outputs) to
a `module.function`. `provenance_check()` confirms from the source that the
module exists, references the artifact (by name, by stem, or by an f-string
prefix) and that the function performs a write. An artifact whose producer
regenerates only some columns is recorded as **partial** and reported as a finding.

## What it found

| artifact | backs | before | now |
|---|---|---|---|
| `r_pooled.csv` | T52, T55, T65 pooled counts (0 of 2,281; 461 of 463) | **no producer** | `provenance_rebuild.r_pooled`: a join of four committed artifacts (`fold1d_sweep`, `fold1d_refine`, `termination` at budget 2,000 without the `seed = −1` sentinel, `alpha_composition`), `gstar = maximum_gap(a, 600)`. **All 2,910 rows reproduce exactly**: keys and flags identical, numeric differences ≤ 4.4e-16 |
| `r_adamw.csv` | T52 third optimizer, T65 | **no producer** | `provenance_rebuild.r_adamw`: retraining with AdamW lr 1e-2, wd 0.01. **Bit-identical** on the rows checked (one last printed digit differs by CSV round-trip) |
| `r_families.csv` | T52 condition 4: family R₅₀ 0.247 (q1), 0.295 (q2) | **no producer** | **partial.** `w2` and `solved` regenerate bit-identically from the committed `family_onsets` training path. **The `gstar` column does not**: it came from a search that is not in the repository. It is an under-estimate — a centred search already finds a gap 89% larger at `q = 1, a = 1.1` |
| `r_family_b.csv` | T62 inputs; Fig 3(c), Fig 7(b) | no producer | **no longer backs any number.** Its primary columns reproduce exactly from `fold1d_sweep.csv`, which Block K now reads directly (Block K outputs unchanged to 1.4e-14). Its `gstar`/`R`/`bin` columns are the superseded box values |
| `ghat_certified_all.csv`, `kappa_certified_exact.csv`, `kappa_certified_full_range.csv`, `kappa_certified_smalleps.csv`, `exact_all_solved.csv`, `exact_vs_grid.csv` | T64, T65 | **no producer** — generated in inline scripts during the 2026-09-22 session and never committed | `session_artifacts.SESSION_PRODUCERS`, written by `write_session_producers()` |

Figures corrected at the same time, because they read the superseded columns:

- **Fig 3(c)** plotted an AUC bar for family B, the comparison T62 removed.
  The bar is gone, and family A now uses the certified `Ĝ`.
- **Fig 7(b)** used the box-limited `gstar` as its x-axis. It now uses
  `Ĝ_norm = 0.4|α|`; the exponent and ratio are unchanged because the box value
  is exactly 8× this.

## Open

**`r_families.csv`'s `gstar` column has no producer, and the committed values
under-estimate the supremum.** The family R₅₀ values in T52 condition 4 rest on
it. This is a finding in `verify_ledger` until resolved.
