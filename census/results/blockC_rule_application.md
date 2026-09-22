# Block C: how the registered cell-selection rule was applied

Written after the placement pilot and **before any main run**. The registration
(`blockC_equivalence_prediction.md`) is unchanged.

## Pilot (8 seeds per budget; measures where runs land in R, not whether they solve)

| budget | Adam median R | AdamW median R | SGD median R |
|---:|---:|---:|---:|
| 3,000 | 0.185 | 0.165 | |
| 4,000 | 0.261 | 0.225 | |
| 5,000 | **0.342** | 0.285 | |
| 6,000 | **0.426** | **0.344** | |
| 7,000 | 0.511 | **0.398** | |
| 8,000 | 0.596 | **0.449** | 0.241 |
| 9,000 | 0.681 | **0.496** | |
| 12,000 | 0.934 | 0.617 | **0.316** |
| 16,000 | | | **0.380** |
| 20,000 | | | **0.438** |
| 24,000 | | | **0.489** |
| 28,000 | | | 0.535 |

Pilot budgets with median R in [0.30, 0.50]: Adam 2, AdamW 4, SGD 4 — all fewer
than the required 6, so the registered fallback applies: *"add midpoints between
qualifying and adjacent budgets until 6 do"*.

## Application

Midpoints have not been piloted, so whether one "qualifies" is judged by the
pilot median **linearly interpolated** between piloted budgets. The spacing is
halved repeatedly until at least 6 budgets qualify (`selected_budgets()` in
`src/blockC_equivalence.py`).

| optimizer | cells | budgets | interpolated pilot median R |
|---|---:|---|---|
| Adam | 10 | 4,500–6,750 in steps of 250 | 0.302 → 0.490 |
| AdamW | 8 | 5,500–9,000 in steps of 500 | 0.315 → 0.496 |
| SGD | 7 | 12,000–24,000 in steps of 2,000 | 0.316 → 0.489 |

The fallback's wording is followed except that the check on each midpoint is by
interpolation, not a fresh pilot. That is the only reading under which the rule
can be applied without new pilot runs.
