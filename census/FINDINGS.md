# Findings

The measured findings will be written after the census has run.

## Specification corrections

Thresholds use the exact identity
`atanh(1 - delta) = 0.5 * (ln(2 - delta) - ln(delta))` throughout. The
asymptotic form `0.5 * ln(2 / delta)` was rejected as the primary calculation.
Its largest deviation in the configured table is 0.0159 at fixed-4. Direct
`atanh` and the asymptotic form are retained only as documented cross-checks.
