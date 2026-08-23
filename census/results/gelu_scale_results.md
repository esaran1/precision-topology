# Task A: the weight-scale mechanism reaches the standard setting

Predictions registered in `gelu_scale_prediction.md` after the standard-arm
measurement and before the scaled arms ran. Data:
`gelu_scale_standard.csv`, `gelu_scale_scaled_up.csv`,
`gelu_scale_scaled_down.csv`. All arms: GELU, width 3, depth 5, baseline
link, 200 seeds, standard Adam, dense verification of every eval-0 run;
the arms differ **only in the initialization scale**.

## The result: a two-sided manipulation of GELU's rate

| Arm | dense-verified rate | 95% CI | vs standard (Fisher, one-sided) |
|---|---:|---|---|
| scaled down (0.3× per layer) | **6/200 = 3.0%** | [1.1%, 6.4%] | p = 0.023 |
| standard Kaiming | 16/200 = 8.0% | [4.6%, 12.7%] | — |
| scaled up (found-separator pattern) | **34/200 = 17.0%** | [12.1%, 22.9%] | p = 0.0048 |

Scaled-up versus scaled-down: p = 1.5e−6. The dose–response is monotone
across the three arms (3% → 8% → 17%): raising initialization scale to
the found-separator pattern **doubles** GELU's separation rate, and
lowering it to 0.3× **halves-and-more** it. **The weight-scale mechanism
reaches the standard setting, in both directions.**

## How this stands against the registration

The outcome exceeded the registered R2 expectation (5–13%, "unchanged
within noise") and lands at the registered resolvability boundary with a
clean test. **R2's scope restriction was too conservative and R1's
direction was right** — and the interesting part is *why* R2's premise
failed:

The standard-arm measurement showed separators' **final** spectral-norm
products are not in the distribution's tail (percentile ranks 0.34–0.86)
— aggregate endpoint scale does not distinguish separating runs. Yet
moving the **initial** scale moved the rate strongly in both directions.
The mechanism therefore operates through **where training starts and
what scale region it traverses**, not through the endpoint scale the
2b-tail comparison used for family A. For sin(1.02) the demand-vs-reach
framing worked because demand exceeded even the endpoint scale; for GELU
— demand (~19×) far below endpoint reach (~2,000×) — the binding factor
is evidently the *early-training* scale environment, which
initialization sets directly. One consistent reading across both
settings: **findability tracks the scale at which training operates when
the fold must form, and initialization is the lever that sets it.** The
endpoint-scale measure was a proxy that happened to work in the extreme
regime and fails in the moderate one; the intervention is the
measurement that matters.

## Ancillary observations

- Scaled-up initialization also raises sample-level-only separations
  (51 eval-0, 34 dense-verified: 33% dense attrition, against 11% in the
  standard arm) — larger scale finds more true solutions *and* more
  sample-level artifacts; dense verification remains mandatory.
- The scaled-down arm's rate (3.0%) confirms the low side is not
  saturated: standard initialization is not at the floor of the scale
  response.
- Substantive difference from the sin intervention, as required: the sin
  pattern came from a construction; GELU's comes from **found** solutions
  (its construction failed and no exhibit exists there). This is a
  weaker, more diffuse intervention — which makes the observed 2.1×
  effect more notable, not less.

## Scope statement for the paper

The findability gap responds to initialization scale in a
continuously-parametrized synthetic family at its analytic threshold
(17/40 vs 0/400 at `sin(1.02)`) **and** in the standard activation people
actually use (17.0% vs 8.0% vs 3.0% across a 3-arm dose–response,
n = 200 each, at GELU width 3). The mechanism is not an artifact of the
synthetic family.
