Registered after 2A-PS's UNRESOLVED (validity) outcome. 2A-PS's registered verdict STAYS UNRESOLVED (validity); this is a
new registration, not a re-scoring. DRAFT for the author's approval: NOT a registration; no registered or pilot seed
drawn; no pilot. Exploratory numbers: `phase2a_ps2_explore/` (seeds 2,930,112–2,930,199 only). ‡ = set after seeing
2A-PS data: **every change below is ‡**, justified from the POST HOC diagnosis (2f0fb33).

# Design 2A-PS2: the per-seed fold test, with three repairs

**Unchanged from 2A-PS.** Samples, branch identity, s_F/s\*/Λ_F (0.05 window; agreement descriptive), exact-M release
at s₀, scale-only rule η = 1, rates 2⁻¹⁶ (scored) and 2⁻¹⁴, f = 0.95 leading-order forecast, criteria F, H, E_seed, C3,
C4, P (C1, C2 secondary), validity, gate.

**(a) Activity test ‡ (Phase 3's).** Unit i is active iff |v_i| ≥ 10⁻⁸·s. 2A-PS counted v₂ = 1.2e−93 active
(2,975,052), so â leaked, a sign flipped at step 1 and there was no forecast (also 2,975,127). **Handling:** status
"inactive unit at release" (after "release Newton failed"): counted, never trained or scored. Exploration: 3 of 54
releases (shares 2e−89 to 2e−83; smallest live share 0.129), 2 of them clean scoreable.

**(b) Per-seed budget ‡ (Phase 3's).**
- t\* = (1/ρ)∫ds/(3·(−dL_M/ds)), s₀ → 0.9999·s_F on the frozen M ("stall", counted, if dL_M/ds ≥ 0).
- B = min(⌈k·t\*⌉ + 3000, ⌈192/ρ⌉); forecast budget B − t_c. **k = 1.5 for clean seeds; 2.5 proposed for none and
  mixed seeds**, whose observation must reach 1.25·s_F. In 2A-PS, 6 of 27 none runs ended after 1.5·t_F (max 1.94);
  clean runs ended by 1.15·t_F.
- Calibration (2⁻¹²): t_F/t\* = 1.002–1.011 (n 34). ρ·t\* median 23.8, max 90.6.
- Over cap: counted, not trained (slow, large-s_F seeds; disclosed). Exploration: 1/36.
- **Hard per-run caps:** ⌈192/ρ⌉ steps (3.1M at 2⁻¹⁴, 12.6M at 2⁻¹⁶) and 1 GB RSS (aborted, counted). 2A-PS peak:
  0.73 GB.
- **Pre-registration check:** STOP and report if more than 5 scoreable clean seeds, or more than 5 scoreable none seeds,
  are over cap.

**(c) Wider class search ‡.** The diagnosis's search: s ∈ {1.01, 1.02, 1.05, 1.10}·s_F × perturbations ×1, ×2, ×4 ×
30 = 360 minima. **Class:** clean if n(ρ₂ ≥ q) ≥ 324, none if 0, mixed otherwise. **H** predicts that wide-none seeds
do not cross by 1.25·s_F at ≥ 80% (≥ 10 needed).

**Is H then informative? Probably not.**
- The search did not separate 2A-PS's crossers from non-crossers (reclassified 6/7 vs 15/23).
- In exploration it empties the none class (0/36 scoreable; all 13 registered-none become mixed).
- At 2⁻¹² (not a registered rate), the REGISTERED class predicted crossing: none 1/13, mixed 7/7, clean 14/14.
  Wide-mixed crossed 8/20.
- None defined by S's ρ₂ < q at 1.01·s_F ‡ is empty too: 0/36 in exploration, 4/30 in 2A-PS (two crossed exactly where
  S reached q).
- Hence decision 3.

**Seeds and gate ‡.**
- Registered 2,987,000–2,987,599 (600); pilot 2,988,000–2,988,059.
- Unused (`p2_seedscan`): no text match in the eight trees, no overlap with 663 range literals, no parquet hit.
- Sized from 2A-PS (per 400 seeds: scoreable 173, clean 74). Untraceable 35.5% (exploration 38.6%), kept as a
  headline count.
- Gate: ≥ 24 clean scoreable (P ≈ 1). Caps 40/30/20.
- 600 seeds give P(≥ 10 wide-none) = 0.99 at a 3% rate. Exploration (0/88) allows ≈ 0.
- H's power at n = 20: 0.91 if 87.5% truly do not cross; 0.63 at 80%.

**Compute (one worker, nice 15; jobs ≤ 4 h).** About 15–17 h:
- freeze ≈ 2.6 h (with the wider search);
- pilot ≈ 0.5 h;
- runs and observation ≈ 12–14 h (2A-PS: 3.5 + 7.6 h; the slow seeds now run longer).

**Competing outcomes**
- **PASS:** 2A-PS's failures were the pipeline (release, budget) and the class rule.
- **C3 FAIL (known risk):** in 2A-PS the ratio was > 1 on only 32/36 forecasting seeds (0.89).
- **H FAIL:** the static basin census does not predict crossing.
- **H UNRESOLVED:** no none class (likely under (c) as written).
- **F or C4 FAIL:** the no-forecast seeds were not only a budget artefact.
- **UNRESOLVED (validity):** a sign or release problem survives the activity test.

**Needs your decision**
1. **Inactive unit:** count it and do not train (recommended), or drop the unit and run 2 units (changes the setting).
2. **Budget:** k = 1.5 for every class (as specified) or 2.5 for none and mixed (recommended); cap ⌈192/ρ⌉; RSS 1 GB;
   X = 5.
3. **Class rule for H:**
   - (i) wider, as written (likely uninformative);
   - (ii) **recommended:** H on the registered 30-minimum class, with the wider class frozen as a secondary H_w;
   - (iii) S-based none (empty).
4. **Seeds:** 600 + 60 pilot, as above.
5. **Gate:** ≥ 24 clean only; a none shortfall makes H UNRESOLVED.
