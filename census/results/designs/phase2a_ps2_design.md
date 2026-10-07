Registered after 2A-PS's UNRESOLVED (validity) outcome. 2A-PS's registered verdict STAYS UNRESOLVED (validity); this is a
new registration, not a re-scoring. DRAFT (decisions applied 2026-10-07): NOT a registration; no registered or pilot seed
drawn; no pilot. Exploratory numbers: `phase2a_ps2_explore/` (seeds 2,930,112–2,930,199 only). ‡ = set after seeing
2A-PS data: **every change below is ‡**, justified from the POST HOC diagnosis (2f0fb33).

# Design 2A-PS2: the per-seed fold test, with the pipeline repaired

**Unchanged from 2A-PS.** Samples, branch identity, s_F/s\*/Λ_F (0.05 window; agreement descriptive), exact-M release
at s₀, scale-only rule η = 1, rates 2⁻¹⁶ (scored) and 2⁻¹⁴, f = 0.95 leading-order forecast, criteria F, H, E_seed, C3,
C4, P (C1, C2 secondary), validity, gate.

**(a) Activity test ‡ (Phase 3's).** Unit i is active iff |v_i| ≥ 10⁻⁸·s. 2A-PS counted v₂ = 1.2e−93 active
(2,975,052), so â leaked, a sign flipped at step 1 and there was no forecast (also 2,975,127). **Handling:** status
"inactive unit at release" (after "release Newton failed"): counted, never trained or scored. Exploration: 3 of 54
releases (shares 2e−89 to 2e−83; smallest live share 0.129), 2 of them clean scoreable.

**(b) Per-seed budget ‡ (Phase 3's).**
- t\* = (1/ρ)∫ds/(3·(−dL_M/ds)), s₀ → 0.9999·s_F on the frozen M ("stall", counted, if dL_M/ds ≥ 0).
- B = min(⌈k·t\*⌉ + 3000, ⌈192/ρ⌉); forecast budget B − t_c. **k = 1.5 for clean seeds and 2.5 for none and mixed
  seeds**, whose observation must reach 1.25·s_F. This is DISCLOSED as set from 2A-PS's committed observations: 6 of
  27 none runs ended after 1.5·t_F (max 1.94); clean runs ended by 1.15·t_F.
- Calibration (2⁻¹²): t_F/t\* = 1.002–1.011 (n 34). ρ·t\* median 23.8, max 90.6.
- Over cap: counted, not trained (slow, large-s_F seeds; disclosed). Exploration: 1/36.
- **Hard per-run caps:** ⌈192/ρ⌉ steps (3.1M at 2⁻¹⁴, 12.6M at 2⁻¹⁶) and 1 GB RSS (aborted, counted). 2A-PS peak:
  0.73 GB.
- **Pre-registration check:** STOP and report if more than 5 scoreable clean seeds, or more than 5 scoreable none seeds,
  are over cap.

**(c) Class rule for H: the REGISTERED 30-minimum rule, unchanged.** It is the fraction of 30 perturbed minima at
1.01·s_F with ρ₂ ≥ q (clean ≥ 27, none 0, mixed 1–26). It measures WHETHER SLAB BASINS LIE NEAR THE FOLD. **H re-tests
the same rule that failed in 2A-PS** (20/30), now with the pipeline repairs (activity test, per-seed budgets).
- 2A-PS: 7/30 none seeds crossed; 3 more ended at the budget below 1.25·s_F.
- Exploration (2⁻¹², not a registered rate): none 1/13 crossed, mixed 7/7, clean 14/14.
- H's power at n = 30: 0.93 if 87.5% truly do not cross, 0.61 at 80%, 0.16 at 70%. 2A-PS's 23/30 (0.77) sits near the
  threshold.

**Wider search ‡: DESCRIPTIVE ONLY, not a criterion.** 360 minima (s ∈ {1.01, 1.02, 1.05, 1.10}·s_F × perturbations ×1,
×2, ×4 × 30), frozen per scoreable seed. Reported as crossing rates by wider-search class. It did not separate 2A-PS's
crossers (it reclassified 6/7 vs 15/23 of the none seeds). In exploration it left 0/36 none, and wide-mixed seeds
crossed 8/20. A none class defined by S's ρ₂ < q was empty (0/36), so it was not adopted.

**Seeds and gate ‡ (accepted).**
- Registered 2,987,000–2,987,599 (600); pilot 2,988,000–2,988,059.
- Unused (`p2_seedscan`): no text match in the eight trees, no overlap with 663 range literals, no parquet hit.
- Sized from 2A-PS (per 400 seeds: scoreable 173, clean 74, none 67). Untraceable 35.5% (exploration 38.6%), kept as a
  headline count.
- Gate: ≥ 24 clean scoreable (P ≈ 1). Fewer than 10 trained none seeds make H UNRESOLVED (about 100 expected).
- Caps 40/30/20. 600 seeds were sized when a wider-rule none class was considered, so they are now generous.

**Compute (one worker, nice 15; jobs ≤ 4 h).** About 15–17 h:
- freeze ≈ 2.6 h (with the wider search);
- pilot ≈ 0.5 h;
- runs and observation ≈ 12–14 h (2A-PS: 3.5 + 7.6 h; the slow seeds now run longer).

**Competing outcomes**
- **PASS:** 2A-PS's failures were the pipeline (release, budget).
- **C3 FAIL (known risk):** in 2A-PS the ratio was > 1 on only 32/36 forecasting seeds (0.89).
- **H FAIL:** with the pipeline repaired, slab basins near the fold do not predict crossing.
- **H UNRESOLVED:** fewer than 10 none seeds (unlikely).
- **F or C4 FAIL:** the no-forecast seeds were not only a budget artefact.
- **UNRESOLVED (validity):** a sign or release problem survives the activity test.

**Decided (author, 2026-10-07)**
1. An inactive-unit release is counted, not trained.
2. k = 1.5 clean, 2.5 none and mixed ‡: set from 2A-PS's COMMITTED observations (6 of 27 none runs ended after
   1.5·t_F, max 1.94; clean ≤ 1.15·t_F). Cap ⌈192/ρ⌉ steps; RSS 1 GB; STOP if more than 5 scoreable clean or more
   than 5 scoreable none seeds are over cap.
3. H on the registered 30-minimum class; the wider search is descriptive only.
4. Seeds as above.
5. Gate as above.

**Still open:** the implementation details, which go to the registration text for approval (e.g. t\*'s grid, and
where "stall" and "over cap" sit in the status order).
