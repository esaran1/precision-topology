# Task E Parts 1b/1c: width dependence of the activation advantage (d = 3)

Registered predictions: `width_prediction.md` (P-W1..P-W4 + amendment).
Data: `width_effect.csv` (5,300 runs, seeds 100–199, all out-of-sample
relative to `width_sweep.csv`). Analysis: `src/width_effect_report.py`.

**Provenance flag, kept prominent on purpose:** the coarse decay shape
was partly postdiction — `width_sweep.md` (committed 2026-08) already
showed a sample-level advantage vanishing by width ~6–8 at 20 seeds/cell.
What was out-of-sample here: widths 16/24/32, fresh seeds, dense-verified
rates as primary outcome, sin(1.5), convergence speed, and the depth-6
slice. A reader who finds `width_sweep.md` independently should know we
did too, first.

## Headline: dense-verified separations per 100 (depth 3)

| width | gelu | sin(1.5) | tanh | leaky-relu | relu |
|---:|---:|---:|---:|---:|---:|
| 3 | 2 | 9 | **0** | **0** | **0** |
| 4 | 72 | 93 | 64 | 18 | 8 |
| 5 | 94 | 99 | 91 | 52 | 33 |
| 6 | 97 | 100 | 95 | 76 | 56 |
| 8 | 99 | 100 | 98 | 93 | 84 |
| 12 | 100 | 100 | 100 | 99 | 99 |
| 16–32 | 100 | 100 | 100 | 100 | 100 |

Width 3: monotonic 0/300 (per-cell zero-rate bound 2.95%; adds to the
project's exact-zero body, now 0 in 5,840+ monotonic runs); GELU 2/100
(CP95 [0.2%, 7.0%]), sin(1.5) 9/100 (CP95 [4.2%, 16.4%]). Error
distributions at width 3 are bimodal exactly as the categorical account
requires: monotonic minima never approach zero (best 25–53 errors of
2,000), non-monotonic runs have the same bulk plus an atom at exactly 0.
Dense verification was not decorative: 7 eval-clean runs at widths 3–4
failed the 100k check and are counted as failures.

## P-W1 verdict: holds from width 12 up; **fails as registered at width 8**

Reported unsoftened. The registered criterion was Fisher-n.s. at *every*
width ≥ 8 against each monotonic activation.

- **Versus tanh** — the comparison the fold account most directly owns —
  the advantage is gone far *earlier* than registered: GELU-vs-tanh is
  n.s. from width 4 on (p = 0.29 at 4); sin-vs-tanh is significant at 4
  (93 vs 64, p = 6.5e-7) and 5 (p = 0.018), n.s. from 6 on.
- **Versus the ReLU family the test fails at width 8**: GELU-vs-relu
  p = 1.5e-4, sin-vs-relu p = 1.6e-5, sin-vs-leaky p = 0.014. From width
  12 up every pairwise comparison is n.s. (p = 1 at 16–32, all cells
  100/100).
- The dead-unit qualifier does **not** rescue it: leaky-ReLU cannot die
  and still fails 7/100 at width 8 with inactive fractions ≤ 0.125, and
  ReLU's width-8 failures have dead fractions (median ~0.21) overlapping
  its solvers' (mean 0.16).
- What the width-4–8 residual actually is: a **ReLU-family deficit, not
  a monotonicity effect** — monotonic tanh beats ReLU at width 8 with
  the same significance (98 vs 84, p = 8e-4) and beats both piecewise
  activations at 4–6 at p ≤ 2e-4. The pairs that stay significant past
  width 6 are all "anything smooth vs ReLU-family", including
  monotonic-vs-monotonic.

So the decay of the *monotonicity-specific* advantage is: categorical at
width 3, sin-only remnant through width 5, statistically zero from width
6 on. The registered test named width 8 against all three monotonic
activations and failed there; the failing component is an optimization
property of the ReLU family shared by comparisons the fold account does
not speak to.

## P-W2 (shape): borne out

Categorical at width d (0/300 vs positive); findability remnant from
d + 1; gone above, with the measured transition at width 6 (vs tanh) /
width 12 (vs ReLU family). Decay is monotone in width for every
non-monotonic/monotonic pair (gelu−relu gap: 64, 61, 41, 15, 1, 0, 0, 0
across widths 4–32; sin−tanh: 29, 8, 5, 2, 0, 0, 0, 0). No
reappearance, no growth anywhere above width 4. None of the registered
falsifiers occurred (nothing significant at ≥ 12, no growth, no
non-monotone decay, no monotonic width-3 solve).

## P-W3 (large-width ceiling): borne out exactly

All five activations at 100/100 dense-verified at widths 16, 24, 32 —
including ReLU at depth 3 (chance collapses: 12 at width 3, 0 from
width 6 up).

## 1c: separating the fold mechanism from optimization effects

**Convergence speed shows no residual non-monotonic advantage at large
width.** Median steps to train criterion at widths 16–32: ReLU and
leaky-ReLU 50, sin(1.5) 50–100, GELU 50–100, tanh 100 (probe resolution
50 steps). GELU is, if anything, *slower* than the piecewise monotonic
activations once everyone separates. Speed ordering at large width
(piecewise < sin ≈ gelu < tanh) tracks activation smoothness/curvature,
not monotonicity. Under the registered P-W4 interpretation rule there is
therefore nothing at large width to attribute to the fold mechanism —
and nothing that needs explaining away: **both** the separation
advantage and any speed pattern consistent with folding are confined to
widths ≤ ~2d.

The two mechanisms the brief asked us to separate come apart cleanly:

1. **Fold capacity (monotonicity-specific):** categorical at width 3,
   remnant to width 5, zero from 6. Visible in dense-verified
   separation only.
2. **ReLU-family optimization deficit (not monotonicity):** widths 4–8,
   shared by tanh-vs-ReLU comparisons, not explained by dead units
   (leaky-ReLU shows it while immune to dying), gone by width 12.

This is the in-setting preview of Part 2b's crux: above width d, tanh
tracks GELU (smooth pair, n.s. everywhere ≥ 4–6) while ReLU lags both —
i.e. in the wide regime the operative distinction is ReLU-family vs
smooth, and only at width ≈ d does the monotonic/non-monotonic
distinction appear at all.

## Depth-6 slice: shape unchanged, no interaction detected

| width | gelu | sin(1.5) | tanh | leaky-relu | relu | (of 40) |
|---:|---:|---:|---:|---:|---:|---|
| 3 | 3 | 5 | 0 | 0 | 0 | categorical |
| 6 | 38 | 40 | 37 | 28 | 16 | remnant (vs ReLU family only) |
| 16 | 37 | 40 | 40 | 40 | 39 | n.s. everywhere |
| 32 | 36 | 40 | 40 | 39 | 40 | n.s. everywhere |

Same two-phase shape. The remnant is not detectably wider or narrower at
depth 6 (significant pairs at width 6 are the same ReLU-family ones as
depth 3). Two depth-6 notes, neither significant at n = 40: GELU shows
a few large-width failures (36–37/40, p ≥ 0.12 vs monotonic — worth a
look only if it recurs elsewhere), and ReLU's width-6 rate drops with
depth (40% at depth 6 vs 56% at depth 3), consistent with the
depth-driven dying documented in `width_sweep.md`. No evidence that
depth substitutes for width in the fold-relevant regime: width-3
categorical structure is identical at both depths.

## Verdict for the Task E gate

The width-dependence prediction **holds where the account owns it and
the bridge stands**: the monotonicity-specific advantage exists only
near width = d and is statistically zero well before width 2d, with no
residual separation or speed advantage at widths 16–32. The registered
P-W1 boundary ("every width ≥ 8") failed at width 8 for ReLU-family
comparisons — recorded as a failed registered detail, with the
diagnosis (a monotonicity-irrelevant ReLU-family optimization deficit)
supported by monotonic-vs-monotonic controls inside the same data.
Part 2 is worth running.
