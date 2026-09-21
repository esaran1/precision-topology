# Registration: Phase 1, what determines sign correctness

**Written before the decomposition was computed.** The rerun
(`src/phase1_relog.py`) was launched first because it only logs parameters;
no placement or bias quantity existed when this was written.

Date: 2026-09-21.

---

## The decomposition, and why `|w2|` is not in it

For `w2 > 0`, with `M_I = max_{x in I} f_a(w1 x + b1)` and
`m_O = min_{x in O} f_a(w1 x + b1)`, sign correctness on the windows holds iff

    w2 M_I + b2 < 0 < w2 m_O + b2

which is exactly two conditions together:

- **placement**: `G(w1, b1) = m_O - M_I > 0`
- **bias**: `b2` in `(-w2 m_O, -w2 M_I)`, an interval of width `w2 G(w1, b1)`

Mirror for `w2 < 0`. **`|w2|` does not appear in the placement condition.** It
only scales the admissible bias interval. So any predictive power of `|w2|`
must act through the dynamics that produce placement and bias, not through the
criterion itself.

Placement quality is `rho = G(w1, b1) / Ĝ(a)`, and the run's own attainable
margin is `R * rho`.

## Registered predictions

**P1-a (primary).** Failures are **predominantly placement failures**
(`G <= 0`), not bias failures. Registered because the fold must be *traversed*
before any bias can separate the classes, and traversal is a property of
`(w1, b1)` alone.

**P1-b.** Among solved runs, `rho` is **well below 1** — solved placements are
not optimal. The theorem-verification median slack of **1.274** implies a
typical `rho` near **1/1.274 = 0.785**; registered as: median solved `rho` in
**[0.5, 0.95]**, not concentrated near 1.

**P1-c (the competing hypothesis, registered so it cannot be dismissed after
the fact).** If instead bias failures dominate, or if small-`|w2|` runs with
good placement fail only on bias, then **`R` tracks the bias-interval width
rather than placement**, and the paper's framing changes from a statement about
fold traversal to one about the output bias.

## Stop and report

Either of these goes out immediately, not in a summary:

1. **Failures are predominantly bias failures.** The story becomes one about
   the output bias.
2. **Placement success is common at small `|w2|` and failure there is due to
   bias.** That would mean `R` tracks the bias-interval width, not placement.

## Validity condition, fixed now

The placement/bias classification must agree with `solves()` **exactly**, run
for run. `solves()` is the dense 4,001-point regional check and is the success
criterion throughout Phases 1-3. **Any disagreement is a bug in the
decomposition and must be found before any Phase 1 number is reported** — the
algebra above is an identity, so disagreement means the implementation is
wrong, not the theory.

## Precision

The rerun records **both float32 and float64 on the same seeds**, because the
pooled `R` dataset and every headline number are float32 while new theory work
is float64. The per-cell solve disagreement rate is reported before any
decomposition result. If it is material, Phase 1 describes a different
population from the headlines and must say so.
