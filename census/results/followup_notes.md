# Follow-up analysis notes

All uncertainties are sample standard deviations across seed-level values.
No training was rerun. A1 recreates deterministic step-zero models only so
failed trained runs do not remove their initialization baselines.

## A1. Initialization baseline by width

| Width | All-init baseline | Trained collision (accepted) | Matched baseline (accepted) | Matched excess | Excess seed n |
|---:|---:|---:|---:|---:|---:|
| 5 | 56.0400% ± 8.4149% | 80.7583% ± 2.2480% | 55.5854% ± 9.6393% | 25.1729% ± 11.1080% | 4 |
| 15 | 12.2000% ± 8.6410% | 60.1625% ± 3.5662% | 12.2000% ± 8.6410% | 47.9625% ± 11.9408% | 5 |
| 30 | 0.0700% ± 0.0727% | 50.6625% ± 3.2947% | 0.0700% ± 0.0727% | 50.5925% ± 3.2979% | 5 |
| 50 | 0.0050% ± 0.0112% | 43.4225% ± 2.8252% | 0.0050% ± 0.0112% | 43.4175% ± 2.8248% | 5 |

The initialization baseline falls systematically with width. The pooled
13.8630% ± 5.9025% baseline is therefore retained only as a secondary
accepted-run summary. Width-5 excess has four seed-level estimates because
seed 2 failed the training gate at every depth; no value is imputed.

## A2. Collision trajectory after the accuracy plateau

| Step | Train accuracy | Eval accuracy | Vector collision | Paper saturation |
|---:|---:|---:|---:|---:|
| 0 | 50.5300% ± 1.1851% | 50.6100% ± 1.3640% | 0.0000% ± 0.0000% | 0.0000% ± 0.0000% |
| 200 | 100.0000% ± 0.0000% | 100.0000% ± 0.0000% | 26.7400% ± 10.6906% | 15.1067% ± 15.2022% |
| 500 | 100.0000% ± 0.0000% | 100.0000% ± 0.0000% | 31.8300% ± 11.6901% | 18.9670% ± 17.4745% |
| 1000 | 100.0000% ± 0.0000% | 100.0000% ± 0.0000% | 37.2900% ± 11.6644% | 22.6503% ± 18.5668% |
| 2000 | 100.0000% ± 0.0000% | 100.0000% ± 0.0000% | 43.4400% ± 10.8292% | 27.4227% ± 19.1404% |

The first observed all-seed accuracy plateau is step 200. Vector collision is 26.7400% ± 10.6906% there and 43.4400% ± 10.8292% at the final step. 40.5176% ± 11.5939% of each seed's total observed collision increase occurs after the plateau.

The trajectory supports continued post-plateau growth but not an onset time:
the only checkpoints before or at the plateau are steps 0 and 200. Resolving
onset would require checkpoints about every 20 steps or finer through step 200.

## A3. Paired blob-minus-tori gaps

Pairing is by seed and matched accepted architecture-layer rows. Error bars
are sample SDs of the five paired seed-level differences, not unpaired error
propagation.

| Comparison | Blobs | Linked tori | Blob − tori gap | Clears SD ≤ half-mean rule? |
|---|---:|---:|---:|:---:|
| bfloat16: all-layer paper saturation | 23.5296% ± 3.4718% | 3.0363% ± 0.5482% | 20.4932% ± 3.7752% | yes |
| bfloat16: final-layer vector collision excess | 82.4909% ± 6.5347% | 43.0156% ± 3.0951% | 39.4753% ± 5.6362% | yes |
| float32: final-layer vector collision | 54.1285% ± 2.0079% | 0.3455% ± 0.3488% | 53.7830% ± 2.0083% | yes |

All three paired gaps clear seed variance under the stated rule. This does
not contradict the current numerical results, but it sharpens their framing:
the blob–tori difference is large relative to seed variation, and the width
dependence makes the pooled initialization baseline unsuitable as the primary
excess summary.
