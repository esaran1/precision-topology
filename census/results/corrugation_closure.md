# Closing the corrugation question (Part B)

Data: `corrugation_width.csv` (1,350 new runs: GELU widths 3/4/6, tanh
widths 4/6, nine Reading-A + flat configurations, 30 seeds, depth 5, dense
verification built in), plus the B1/B2 analyses on existing data.

## B1: the question does not dissolve

Dense-verification failure does not track tube self-overlap: embedded
configurations 3/16 separations survive against non-embedded 2/18, and
within Reading A — where embeddedness is constant — survival still varies
with amplitude. The two-class problem is well-posed throughout
(between-class sample distance verified positive). The ill-posedness
reading is rejected.

## B2: the gradient was already in the existing data

Continuous outcome (eval-error count) over existing width-3 runs: GELU
degrades monotonically with amplitude (Spearman +0.47 [+0.33, +0.60]) and
frequency (+0.47 [+0.33, +0.60]) — and **monotonic activations degrade at
least as much** (+0.61 [+0.53, +0.68]). The trend needed no new runs; what
it could not supply was width-specificity, hence B3/B4.

## B3: the gradient at widths where rates are measurable

Dense-verified separations out of 30, depth 5:

| | flat | a=0.05 | a=0.15 | a=0.3 (paper) | **a=0.5** | f=0.5 | f=10 | f=50 | f=200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GELU w4 | 19 | 15 | 18 | 16 | **0** | 17 | 15 | 12 | 15 |
| GELU w6 | 30 | 30 | 30 | 25 | **13** | 26 | 27 | 27 | 29 |
| tanh w4 | 13 | 11 | 17 | 11 | **0** | 15 | 13 | 14 | 13 |
| tanh w6 | 29 | 29 | 30 | 28 | **6** | 28 | 30 | 30 | 28 |

Two sharp facts:

1. **The amplitude axis is flat until 0.3 and cliffs at 0.5.** Rates at
   the published amplitude are barely below flat; amplitude 0.5 sends
   width 4 to zero (both activations) and halves-or-worse width 6.
2. **The frequency axis barely matters**: at amplitude 0.3, frequency 200
   costs a few runs at most, at every width. The "high-frequency
   oscillation" is not what makes corrugation hard; amplitude is.

## B4: the control — generic difficulty, width-amplified, not topology-specific

The a=0.5 cliff hits **every width including 6**, and hits **monotonic
tanh at least as hard as GELU** (w6: tanh 29→6, GELU 30→13). tanh at
width ≥ 4 faces no topological obstruction, so a corrugation effect that
punishes it equally is about task difficulty, not about interaction with
the linking obstruction. Severity is width-modulated — the same amplitude
that halves width 6 zeroes width 4 — but the signature of a
threshold-specific interaction (harm near the width threshold, none above
it) is absent.

## The width-3 revision this sweep forces

The new width-3 GELU arm (30 seeds, depth 5) found **dense-verified
separations at strong corrugation**: A_paper 1/30, A_a0.15 1/30, A_f10
2/30, A_f50 1/30, A_f200 1/30 (plus A_embedded_f0.5 5/30). The earlier
threshold reading — "regional separation collapses above mild
corrugation" (`corrugation_dense_rates.md`, T21) — **was an underpowered
zero, exactly as its own power note warned**: the strong-group bound was
0.83% against a true rate that is evidently of order 3% at depth 5.
Superseded; T21 revised.

**And the reopened fold-layer question closes with dense-verified
support**: all 11 new dense-verified width-3 runs — including at the
published amplitude and at frequency 200 — fold at **layer 1** (input lk
−1 → 0 at layer 1, 8,192-point float64 traces, healthy distances). The
strong-corrugation fold-layer negative that dense verification had
demoted to "open" is restored: corrugation does not move the fold later,
now on regional separations.

## B5: the conclusion

**Corrugation makes the task uniformly harder at all widths — a modest,
generic-difficulty finding — with severity that grows toward lower width,
driven almost entirely by amplitude and hardly at all by frequency.** The
width-3 "regional unsolvability" threshold dissolved under adequate
seeds; the fold stays at layer 1 under strong corrugation on
dense-verified runs; and no width-specific interaction with the
topological obstruction is present (the monotonic comparator is hit
equally). The question is closed: outcome 2 of the acceptable list, with
the width-3 special case now consistent rather than exceptional.
