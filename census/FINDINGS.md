# Findings

In the preregistered pilot, the trained final tanh layer retained 1,225 of 2,000
distinct evaluation vectors at bfloat16 (38.75% vector collision), versus 2,000
of 2,000 at float32 and at bfloat16 initialization; all 248 collision groups
were class-pure. Full-sweep findings will replace the pilot-only interpretation.

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

Simulated fixed-point collision metrics are reported only for tanh. ReLU and
leaky-ReLU outputs are unbounded, so fixed-point quantization would require a
free per-tensor scale; those rows are omitted rather than made incomparable by
an improvised scale.

## Paper proxy versus exact rounding

In the pilot's final hidden layer, the paper threshold counted 10.355% of
pre-activations while the exact-rounding threshold counted 1.105%, a nearly
tenfold gap. Both definitions remain primary because this ambiguity is
load-bearing rather than cosmetic.

## Reasons this result might be wrong

- Pre-activation scale may depend more strongly on weight initialization than on
  training or precision. Main runs retain PyTorch's default `nn.Linear`
  initialization (Kaiming uniform with `a=sqrt(5)`, gain `1/sqrt(3)`, and default
  uniform bias). This is a first-order confound, not a neutral implementation
  detail; the present experiment does not compare initialization schemes.
