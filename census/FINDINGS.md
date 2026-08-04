# Findings

The measured findings will be written after the census has run.

## Specification corrections

Thresholds use the exact identity
`atanh(1 - delta) = 0.5 * (ln(2 - delta) - ln(delta))` throughout. The
asymptotic form `0.5 * ln(2 / delta)` was rejected as the primary calculation.
Its largest deviation in the configured table is 0.0159 at fixed-4. Direct
`atanh` and the asymptotic form are retained only as documented cross-checks.

The initially specified pooled scalar collision rate was rejected because its
numerator is bounded by format cardinality while its denominator pooled every
input-unit pair. It measured a pigeonhole artifact. Collision measurements use
within-unit distributions and complete-vector duplication instead, with the
untrained network on identical inputs as the baseline.

## Reasons this result might be wrong

- Pre-activation scale may depend more strongly on weight initialization than on
  training or precision. Main runs retain PyTorch's default `nn.Linear`
  initialization (Kaiming uniform with `a=sqrt(5)`, gain `1/sqrt(3)`, and default
  uniform bias). This is a first-order confound, not a neutral implementation
  detail; the present experiment does not compare initialization schemes.
