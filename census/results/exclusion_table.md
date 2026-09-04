# The exclusion table: four standard explanations, each ruled out by measurement

Data: `exclusion_table.csv`. Every column is a direct measurement on the same
objects, and every one could have come out the other way. Rows marked
**zero-basin** are solutions that exist, solve the task exactly, and are never
reached (0/200 SGD runs at a <= 1.30; exact basin fraction 0 through a = 1.25).

| a | population | \|w2\| | MEP barrier | lambda_max*eta | distance from init | margin |
|---|---|---|---|---|---|---|
| 1.02 | **zero-basin** | 1.00 | **0.000** | 0.027 | **4.67** | 0.00027 |
| 1.10 | **zero-basin** | 1.00 | **0.000** | 0.027 | **4.82** | 0.0031 |
| 1.25 | **zero-basin** | 1.00 | **0.000** | 0.028 | **4.94** | 0.0129 |
| 1.35 | **zero-basin** | 1.00 | **0.000** | 0.028 | **5.00** | 0.0219 |
| 1.45 | **zero-basin** | 1.00 | **0.000** | 0.029 | **5.05** | 0.0325 |
| 1.45 | found | 5.47 | 0.000 | 0.0036 | 17.69 | 0.1035 |
| 1.50 | **zero-basin** | 1.00 | **0.000** | 0.029 | **5.07** | 0.0384 |
| 1.50 | found | 4.39 | 0.000 | 0.0028 | 14.20 | 0.0771 |
| 2.00 | **zero-basin** | 1.00 | **0.000** | 0.030 | **5.26** | 0.1149 |
| 2.00 | found | 4.24 | 0.000 | 0.0015 | 13.44 | 0.8982 |
| 3.00 | **zero-basin** | 1.00 | **0.000** | 0.025 | **5.52** | 0.3329 |
| 3.00 | found | 3.47 | 0.000 | 0.0005 | 10.23 | 1.6436 |

## What each column excludes, and how

**Energy barrier — excluded by measurement.** Minimum-energy-path barrier is
**exactly 0.000** for every zero-basin solution at every a, from 5
initializations each (string method, `barrier.csv`). The solutions are
downhill-connected to typical initializations. Nothing is being escaped.
*This is a measurement.* It could have come out positive; the linear-path
proxy, which was the original evidence, does show positive values, and the
first version of this project believed them (`instrument_artifacts.md`
instance 3).

**Sharpness / edge of stability — excluded by measurement.** `lambda_max *
eta_eff` is 0.025-0.030 for zero-basin solutions against the Ahn-Zhang-Sra
threshold of **2**: 70x inside the stable regime. Found solutions are further
inside still (0.0005-0.0036). Ahn-Zhang-Sra Theorem 1 requires
lambda_max > 2/eta at the stationary point to apply; it does not apply here
and is not violated. *This is a measurement* (exact 4x4 Hessians), and the
registered prediction was that it would separate the populations at 2. It
did not.

**Distance — excluded by measurement, with the sign reversed.** Zero-basin
solutions sit **4.7-5.5** from typical initialization; found solutions sit
**10.2-17.7** away, 2-3x further. Training routinely travels past the
unreachable solutions to reach the reachable ones. *This is a measurement*,
and it is the cleanest reversal in the table: the never-found solutions are
the nearer ones.

**Margin — excluded by measurement, with population overlap.** At a = 1.5 the
zero-basin solution's margin is 0.0384, and **5 of 20 found solutions have a
smaller margin than that** (min 0.0038). So margin does not separate the
populations even as an ordering. *This is a measurement.* (The theorem in
`fold1d_theorem.md` does predict these solutions must have small margin,
m = |w2|*G/2 -- that part is derivation; the overlap with found solutions is
measurement.)

## What is inference rather than measurement

- That **no other standard account applies** is an inference from having
  checked the four above, not a proof of exhaustiveness.
- That the mechanism is **thin-target capture** is currently supported by
  `reach` (T40) and by the closed-form thickness |w2|*G(a), and is tested
  quantitatively in `capture_cross_section.csv`. Where that test is
  correlational rather than derived, it is labelled as such there.
- The **MEP barrier of exactly 0** is a numerical result from a string method
  with finite images (41) and finite steps (400); it is "0 to the resolution
  of the method", not a theorem.

## Why this table is the central artifact

Every existing account of gradient descent failing to reach a minimum is
about instability (sharpness beyond 2/eta), energy (a barrier to cross), or
distance (too far to travel). This table measures all three on the same
objects and rules out all three, plus margin. What remains is a solution set
that is stable, downhill-connected, nearby, and adequately-margined -- and
never found.
