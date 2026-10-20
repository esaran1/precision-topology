# Track U, step U2: design, and a sample of leaf timings (checkpoint 2; NOT a certificate)

**Status.** This page is a design plus timings from an exploratory prototype, `src/u2_sample.py`.
- The prototype works to the producer standard of §20: float arithmetic with a stated rounding allowance, a 10⁻⁹ allowance per
  decision and τ = 10⁻⁷.
- Nothing here has been checked independently. Nothing here is cited as proved.
- No full run has been made, and none will be without the author's approval.
- Sample outputs are in `results/uniform_eps/u2_sample/` (under 0.5 MB in total). `STATUS.md` there records the hold of
  2026-10-16.

## 0. The author's decisions (2026-10-16) on flags F1–F6, and what each requires

**F1. A-window: APPROVED; the claim is WINDOWED (author's decision, 2026-10-16 evening; the earlier all-scale condition is
withdrawn).** For every ε ∈ [0, 0.6] and every A in the window J(ε), the global conditional minimiser is unplaced (G < 0) for
A < A_ε and placed (G > 0) for A > A_ε. Here A_ε is the branch switch, s_br(ε) = A_ε·ε^{−3/2}, and s = A·ε^{−3/2}.

- **The window at ε = 0.6.**
  - **A:** J ≈ [0.8958, 0.9637]. **s:** J ≈ [1.9275, 2.0735], with ε^{−3/2} = 2.15166 at ε = 0.6.
  - These come from the float anchor A_ε^float = 0.93036, computed at ε = 0.5995 (`probes1.jsonl`). That anchor lies inside the
    certified a = 1.60 bracket A ∈ [0.92952, 0.93533], s ∈ [2.0000, 2.0125].
  - The anchor at ε = 0.6 itself moves by about 10⁻⁴ in A. The full run's switch step records the exact window endpoints for
    the revision to quote.
  - At ε = 0 the window is [0.66, 0.71] in A. In s it is unbounded, because s = A·ε^{−3/2} → ∞.

- **The window contains the certified thresholds.** J(ε) is A_ε^float × [0.96287, 1.03582]. The margins below assume the
  worst case: the anchor A_ε^float may lie anywhere inside each certified bracket. (Arithmetic on the committed CSVs only.)

| threshold | A bracket | margin below (bracket lo − window lo) | margin above (window hi − bracket hi) |
|---|---|---|---|
| limit, A\* ∈ (0.68125, 0.6875] (window [0.66, 0.71] at ε = 0) | [0.68125, 0.6875] | 0.02125 | 0.02250 |
| a = 1.01 (`first_order_finite.csv`) | [0.68973, 0.69016] | 0.02519 | 0.02428 |
| a = 1.02 | [0.69403, 0.69490] | 0.02493 | 0.02400 |
| a = 1.03 | [0.69879, 0.69922] | 0.02552 | 0.02460 |
| a = 1.04 | [0.70272, 0.70447] | 0.02440 | **0.02343** (smallest finite-a margin) |
| a = 1.08 (`c1_followup_finite.csv`) | [0.72041, 0.72221] | 0.02501 | 0.02401 |
| a = 1.09 | [0.72536, 0.72581] | 0.02649 | 0.02553 |
| a = 1.10 | [0.72989, 0.73034] | 0.02666 | 0.02569 |
| a = 1.11 | [0.73398, 0.73489] | 0.02637 | 0.02538 |
| a = 1.12 | [0.73856, 0.73902] | 0.02697 | 0.02600 |
| a = 1.30 (`cond_certified_brackets.csv`, glob; A = s·ε^{3/2}) | [0.81337, 0.81542] | 0.02822 | 0.02708 |
| a = 1.35 | [0.83343, 0.83602] | 0.02845 | 0.02727 |
| a = 1.40 | [0.85381, 0.85698] | 0.02865 | 0.02742 |
| a = 1.45 | [0.87165, 0.87542] | 0.02872 | 0.02745 |
| a = 1.50 | [0.89272, 0.89714] | 0.02888 | 0.02756 |
| a = 1.60 | [0.92952, 0.93533] | 0.02891 | 0.02749 |

  Every certified threshold lies inside its window, with margins of at least 0.0212 (limit) and 0.0234 (finite a). The full
  run will also check that every certified switch box Z(ε) lies inside J(ε).

- **OPEN, and not built (author's decision): the global minimiser outside the window.** There is no status branch and
  bound outside J(ε), and no s → 0 or s → ∞ lemma. So U2 says nothing about global sign changes of the gap at scales outside
  the window.
  - **Why the small-scale limit does not carry over:** on the exact 800-point population, §21.5 (Lemma K4) shows that
    |D| = 2 forces |α| ≥ π/q. The small-scale limit is therefore forced onto the alias, and every admissible W is at least
    623,614. A uniform-in-ε "unplaced at small scale" statement would have to be proved for that aliased limit; the
    continuous-window and float-quadrature results of §21.3–§21.4 do not transfer to this objective.
  - The large-scale side, an explicit s_large(ε) with the configuration placed, is likewise not written.

**F2. Continuity across slabs: APPROVED (author's decision)**, provided that:
- every leaf has a STRICT certified uniqueness margin. Outer leaves have D > τ = 10⁻⁷ on top of the 10⁻⁹ allowance; ring boxes
  have a gradient margin > 0 after the allowance; PD sub-boxes have λ_lower > 0; Krawczyk images lie strictly inside their
  boxes;
- slabs are CLOSED intervals [e_k, e_{k+1}] sharing endpoints.

The producer and the checker both test closedness and contiguity: e_{k+1} of one slab equals e_k of the next, exactly, as
floats. Then at a shared endpoint both slabs certify the same unique global minimiser.

**F3. Coordinates: APPROVED (author's decision).** The covered domains contain the localisation set at every ε:
- **Regime R** (slabs inside [0, 0.0462]). Lemma G2 holds uniformly for ε ≤ 0.0462 (§20.2). Every global minimiser lies in
  K(24) relative to its fold centre π + 2πk. The windings b₁ ↦ b₁ + 2πk and the reflection p ↦ −p (exact x-symmetry of
  the data) reduce this to K(24) ∩ {p ≥ 0}, which is the regime-R domain, for every ε in the slab.
- **Regime O** (slabs inside [0.0462, 0.6]). The §5.0 lemma holds for every a: every global minimiser has |w₁| ≤ W(s, a), and
  W increases as s falls and as a rises. So for every (ε, A) in the unit, W(s, a) ≤ W(s_min, a_max), with s_min = A_lo·ε_hi^{−3/2}
  and a_max = 1 + ε_hi.
  - In rescaled coordinates, p = w₁/√ε ≤ W(s_min, a_max)/√ε_lo, which is the domain's p_max.
  - One b₁-period is q-length 2π/√ε ≤ 2π/√ε_lo, so q ∈ [−π/√ε_lo, π/√ε_lo] contains a full period for every ε in the slab.
  - The periodic images of the branch box lie outside that q-window while ε_hi < 4ε_lo, which holds for every planned slab.
  - p ≥ 0 by the reflection.
- **Together.** The slabs cover [0, 0.6] as closed intervals (F2). So at every ε, one of the two lemmas applies and its
  domain is searched.

**F4. Start at ε = 0: APPROVED AND PREFERRED (author's decision)**, with U1 kept as a cross-check, provided:
- **The rescaled objective extends analytically to ε = 0.** In the kernel form φ_ε = −σ + (1+ε)σ³Gt(u) (§2):
  - every kernel (Gt, J, sinc, K1, JJ, K2, K3, SS) is an even entire function of u, so a function of u² = εσ²;
  - φ_ε, φ_σ, φ_σσ and their ε-derivatives are therefore entire in (σ, ε), with φ_0 = h;
  - β/√ε = arccos(1/(1+ε))/√ε is analytic at 0 with value √2. The prototype now uses its series below ε = 10⁻⁶;
  - the second fold (2π − β)/√ε tends to +∞, so its condition is vacuous at ε = 0.
- **check_u2_unit encloses it correctly on intervals containing 0.** Its activation evaluates the kernels as series in
  v = u² = E·S² (with E the ε-ball), never via √E. The closed forms are used only when v is certainly ≥ 1, and the series are
  used, with their tail ball, otherwise.
  - Tests: balls with E ∋ 0 contain the mpmath values at ε = 0 (h, h′, h″) and at random small ε. They also contain U1's
    second-order expansion h + εr ± ε²m₂.
  - The prototype's regime R already accepts e_lo = 0, since every √ε enters only inside u.

**F5. R-form: nothing to build** (the author answers this).

**F6. Checker: APPROVED (author's decision).**
- **Re-search is the PRIMARY route:** check_u2_unit runs its own Arb branch and bound per unit.
- Replay of the exported leaves is used only for units the re-search fails.
- The producer still exports the integer quadtree indices and the ring/PD/switch box lists for every unit, so any failing
  unit can be replayed. The exports are small (§4).

## 1. The target, and what touches its definitions (proposals; the author's decisions are in §0)

**Target (author's U2 text).**
- For every ε ∈ [ε₀, ε_max], with ε₀ = 4.1·10⁻⁶, ε_max = 0.6, a = 1 + ε, and the 800-point quadrature objective:
  - the continued branch is the global conditional minimiser at its switch;
  - so s_glob(ε) = s_br(ε).

**What each slab must establish.** For every ε in the slab and every A in the slab's A-tube J(ε), using rescaled A = s·ε^{3/2}:
- **(i)** the global minimiser of L\*_ε(·; A) is unique, up to p ↦ −p and the windings b₁ ↦ b₁ + 2πk. It lies in the 0.05 box U
  of that A-sub-interval, and it is the unique critical point there.
- **(ii)** the branch's gap changes sign exactly once in J(ε), at A_ε inside the slab's switch box Z.

That gives s_glob(ε) = s_br(ε) = A_ε·ε^{−3/2} for every ε in the slab.

**Flags.** Each of these touches a definition or criterion of the target. Each is a proposal.

- **F1. The A-window (tube).**
  - Theorem U1 states its conclusion for A ∈ [0.66, 0.71]. Its "s_glob" is the global sign change *within that window*. A
    sign change of the global gap far from the branch switch is not excluded by U1, and would not be by U2.
  - At finite ε the switch moves: A_ε rises from 0.6854 at ε = 0 to about 0.930 at ε = 0.6. A fixed window therefore cannot
    serve.
  - **Proposal:** J(ε) = A_ε^float × [0.66/A\*, 0.71/A\*] = A_ε^float × [0.9629, 1.0358], i.e. the same relative window as U1. It is
    split into 40 A-sub-intervals, as U1 was. Here A_ε^float is a float Newton point used only to place the window; the
    certified switch box Z must lie inside it.
  - Cost scales linearly with the window width. The author may prefer a narrower or wider window.
  - **This choice sets what "s_glob" means in U2. It needs the author's decision.**
- **F2. "The continued branch" across slabs.** In each slab the branch is identified only as the unique global minimiser.
  Continuity across slab boundaries, and across the step from the rescaled domain to the original-coordinate domain, then
  follows from that uniqueness: at a shared ε, both slabs certify the same unique global minimiser. No separate continuation
  certificate is proposed.
- **F3. Coordinates.**
  - The author asked for "rescaled coordinates near ε₀ and original coordinates further out". The prototype uses the
    rescaled parametrisation (p, q, A) with the *exact* φ_ε (no expansion) everywhere. That is an exact linear change of
    variables, w₁ = √ε p, b₁ = π + √ε q, s = A ε^{−3/2}.
  - The original coordinates enter only through the domain and localisation lemma:
    - **Regime R** (ε ≤ 0.0462, Lemma G2): K(24) ∩ {p ≥ 0}, with U1's exact isotonic exclusions;
    - **Regime O** (any a; the §5.0 lemma): {0 ≤ w₁ ≤ W(s_min, a_max)} × one b₁-period, mapped to (p, q).
  - **Why:** in fixed (w₁, b₁) the branch moves with ε (dw₁/dε = p/(2√ε)), and ∂_ε L\* is of order 1/ε. In (p, q) the branch
    moves only slowly, and ∂_ε L\* near it is about 0.08 (U1's L₁), so slabs can be about 20 times wider.
  - This choice does not change the statement. The author may still prefer literal (w₁, b₁) cells for regime O, as in the
    finite-a certificates. They would be costlier in ε.
- **F4. Overlap with U1.** U1 covers (0, ε₀], but its outer exclusion is not yet checked in Arb (that is U3). U2's first
  slab can start at ε = 0 at no extra cost: the rescaled regime is valid down to 0, and [4.1·10⁻⁶, 5·10⁻³] already certifies
  in the sample. Doing so would make one certificate cover (0, 0.6]. **Proposal for the author:** start at 0, and keep U1 as
  the separate, sharper small-ε statement.
- **F5. The R-form.** The R-form, R_glob = sĜ/2, is not covered. As in §16.5, it also needs Ĝ(a) to be attained at the
  tied corner for every ε. This is unchanged.
- **F6. The rigour route.** The producer's bounds on the kernels (§3) are grid envelopes, not proved. **Choice: the checker
  route (§5).** The independent Arb checker computes its own enclosures of φ_ε and its derivatives directly, and relies on no
  producer bound. The certificate is then "producer float + independent Arb check", as in Tracks 4 and 5. No kernel lemma is
  needed in the math note.
  - **Why this route:**
    - Arb evaluates sin and cos of balls rigorously, so each kernel's range over a u-ball is a direct ball computation. It
      needs no envelope and no monotonicity claim.
    - A hand-written proof of 7 envelope functions (sup over |u| ≥ U of each kernel) would be a long new lemma with its own
      error risk, and the checker would have to trust it.
    - The checker must re-derive every leaf anyway, so routing the kernel rigour through it costs nothing extra.

## 2. Decomposition

**Slabs.** [ε₀, 0.6] is split into ε-slabs [e_lo, e_hi]. Each slab carries its own A-tube J (F1), split into 40
A-sub-intervals. A **unit** is one (slab × A-sub-interval). Per unit:

1. **Branch side.**
   - θ_c is the float branch point at (A_c, ε_c).
   - The upper bound is BU = L\*(θ_c) + |∂_A L\*|h_A + |∂_ε L\*|h_ε + ½·(an upper bound of the (A, ε) Hessian of L\*(θ_c; ·)).
   - That Hessian bound is ¼·mean(centred (∇_{A,ε} z)²) + mean|z_{Aε}, z_{εε} terms|.
2. **Outer exclusion.** Every θ outside the 0.15 box, in the regime's domain, satisfies L\*(θ; A, ε) > L\*(θ_c; A, ε). This
   is checked on a quadtree over (p, q) from 0.25 cells, with (A, ε) as cell dimensions. A cell is excluded by any of:
   - **(iso), (full):** R only. Exact and ε-uniform (Lemma G2), as in U1.
   - **(pert):** the coupled 4-D Taylor bound on the DIFFERENCE,
     D ≥ (L − L_b) − |g_p|h_p − |g_q|h_q − |g_A − g_A,b|h_A − |g_ε − g_ε,b|h_ε − ½·mean_i T_i − curv_b − 2·10⁻⁹ > τ.
     - T_i bounds Σ_ab |z_ab|h_a h_b over the cell. It uses the Gauss–Newton/Schur argument: the profiled Hessian is at least
       mean((σ − y)∇²z) in all four variables.
     - The ε-terms of T_i are **centred by a per-cell shift c(ε) = φ_ε(q_c)**, which the profiled bias absorbs. That is,
       |φ_ε(σ_i) − φ_ε(q_c)| ≤ sup|φ_σε|·|σ_i − q_c|, and likewise for φ_εε via φ_σεε.
     - **Without this centring, regime O failed at Δε = 0.01.** It got stuck at p ≈ 0, q at the b₁-period edge, where every
       point has the same large |σ| ≈ 14.6, and one unit ran 4.7 h. With it, Δε = 0.03 certifies (§3).
   - **(fav):** the favourable-logit profile. φ_ε's exact range over each σ-interval at ε_c, using its critical points
     u = ±β + 2πk, is widened by h_ε·sup|φ_ε|.
3. **Ring.** On 0.05 ≤ |θ − θ_c|∞ ≤ 0.15, every box shows |∂_p L\*| or |∂_q L\*| > Σ_a M_ka h_a + 10⁻⁹.
   - M_ka = ¼√(Q_k Q_a) + mean sup|z_ka| bounds the profiled Hessian entries in (p, q, A, ε).
   - Boxes start at side 0.0125 and split adaptively to depth 5. The working set is now capped at 256 boxes × 800 points (§4).
4. **PD.** On the 0.05 box U × A-sub-interval × slab, the (p, q, b) Hessian of L at b ∈ b\*'s bracket is enclosed in
   intervals, from mean-value enclosures of φ, φ_σ, φ_σσ. The test is λ_min(mid) − ‖rad‖_F > 0, on adaptive sub-boxes.
5. **Switch (per slab).**
   - **Box Z:** a 4-variable Krawczyk test on (∇_{p,q,b} L, G) in (p, q, b, A), with ε as an interval parameter. The active
     pair (inner x = 0.8, outer x = −1.2) is checked against φ_ε's critical points β/√ε and (2π − β)/√ε over the slab.
   - **Branch boxes:** 3-variable parametric Krawczyk boxes over the A-tube (geometric A-edges, as in U1), with the gap's
     sign certified left and right of Z.
   - Each box must lie in its sub-interval's U.
   - It hit a shape bug in the sample, now fixed. **It has not been timed yet.**

**Leaf claim exported per outer leaf:** (p, q, h_p, h_q, test, D). The unit's header records ε-slab, A-sub-interval, θ_c,
BU and its parts, regime, domain, τ, the 10⁻⁹ allowance, and the population SHA-256. Ring and PD export their box lists in
the same way.

**Time cap and slab splitting** (`certify_split`). Each unit runs under a wall-clock cap, checked inside rounds, not only
between them. A unit that times out or fails has its ε-slab halved, and both halves are retried, recursively, down to a
stated minimum width. A slab that still fails at the minimum width is reported as failed. Tested on constructed cases
(`tests/test_u2_gate.py`); **those tests have not run yet** (§6).

## 3. Sample timings (one process, nice 15, one thread)

**Outer exclusion, middle A-sub-interval k = 20 of 40** (`probes3.jsonl`, `probes4.jsonl`). These are the runs after the
centring fix; "fail" means the unit stopped at its cap or round limit.

| regime | ε-slab | Δε | certified | leaves | seconds |
|---|---|---|---|---|---|
| R | [4.1e-6, 1.0e-3] | 1e-3 | yes | 39,171 | 2.5 |
| R | [4.1e-6, 3.0e-3] | 3e-3 | yes | 39,180 | 2.4 |
| R | [4.1e-6, 5.0e-3] | 5e-3 | yes | 39,183 | 3.5 |
| R | [4.1e-6, 1.0e-2] | 1e-2 | **no** (band p ≈ 18, q ≈ −39) | 40,729 | 167 |
| R | [1e-3, 4e-3] | 3e-3 | yes | 39,180 | 2.4 |
| R | [0.010, 0.015] | 5e-3 | yes | 39,178 | 4.1 |
| R | [0.010, 0.020] | 1e-2 | **no** | 41,041 | 159 |
| R | [0.0262, 0.0362] | 1e-2 | yes | 39,180 | 5.3 |
| R | [0.0352, 0.0452] | 1e-2 | yes | 39,171 | 2.2 |
| O | [0.001, 0.002] | 1e-3 | **no** (domain to q = ±99) | 97,470 | 132 |
| O | [0.010, 0.013] | 3e-3 | yes | 19,696 | 5.6 |
| O | [0.010, 0.020] | 1e-2 | **no** | 32,885 | 365 |
| O | [0.0462, 0.0762] | 0.03 | yes | 6,699 | 2.5 |
| O | [0.0462, 0.1462] | 0.1 | **no** | 4,501 | 143 |
| O | [0.15, 0.18] | 0.03 | yes | 1,762 | 0.7 |
| O | [0.15, 0.25] | 0.1 | **no** | 7,943 | 224 |
| O | [0.30, 0.33] | 0.03 | yes | 1,205 | 0.3 |
| O | [0.30, 0.40] | 0.1 | **no** (live cell next to the box) | 4,625 | 66 |
| O | [0.50, 0.51] | 0.01 | yes | 986 | 0.2 |
| O | [0.50, 0.60] | 0.1 | **no** (live cell next to the box) | 2,876 | 2.8 |

**Full sample unit,** ε ∈ [4.1·10⁻⁶, 3·10⁻³], R, three A-sub-intervals k = 0, 20 and 39. All certified:

| k | outer | ring | PD |
|---|---|---|---|
| 0 | 39,179 leaves in 53.0 s (an outlier; probably contention with another job, not reproduced) | 5,252 boxes, 0.78 s, margin ≥ 2.78·10⁻⁶ | 80 boxes, 0.09 s, λ ≥ 0.0453 |
| 20 | 39,180 leaves in 2.6 s | 5,424 boxes, 1.0 s, margin ≥ 1.17·10⁻⁵ | 80 boxes, 0.13 s, λ ≥ 0.0412 |
| 39 | 39,172 leaves in 3.7 s | 5,568 boxes, 1.76 s, margin ≥ 1.84·10⁻⁶ | 80 boxes, 0.19 s, λ ≥ 0.0373 |

For comparison, U1's first-order bounds stop PD at ε = 5.17·10⁻⁶. The ring and PD links no longer bind.

**Exports:** 3 npz files of 101–109 KB each, **2.6–2.8 bytes per leaf** (compressed), with SHA-256s in `probes4.jsonl`.

**Soundness spot checks** (`validate.json`, before the centring fix):
- 4 slabs, 1,200 random (θ, A, ε) per slab in pert leaves, plus fav leaves: no bound violated; the smallest slack is
  1.2·10⁻³.
- Shifting θ_c by 0.3 makes every case fail, as it should.

**Peak RSS:**
- Outer units: ≤ 1.6 GB, which includes the envelope tables and numpy.
- **Spec 11 (ε ∈ [0.0352, 0.0452], R, k = 0, with ring and PD) reached 3.73 GB and was SIGSTOPped by the watchdog.** It was
  later killed by the coordinator.
- **Diagnosis:** ring_unit (and pd_unit) evaluated *all* pending boxes at once, i.e. (boxes × 800) arrays, about 30 of them.
  With a 10⁻² slab the Lipschitz terms grow, so more boxes split, and 20k+ boxes give about 3–4 GB. The outer unit is
  already chunked (2,000 × 800).
- **Cap:** at most 256 boxes per evaluation (`BOX_CHUNK`), which bounds the ring/PD working set to about 50 MB.
- **Still to do:** confirm which stage it was, and measure the peak after the cap on the same spec. `python -m src.u2_sample
  stages 0.0352 0.0452 R 0 40` reports peak RSS after each stage. It is blocked by the disk hold.

## 4. Preliminary projection (the author asked for it; incomplete; switch and checker not yet timed)

**Slabs.**
- **R:** 5·10⁻³ slabs up to about 0.026, then 10⁻², reaching 0.0462. That is **about 8 slabs**.
- **O:** 0.03 slabs over [0.0462, 0.6]. That is **about 19 slabs**.
- In total **about 27 slabs × 40 = about 1,080 units**. Allow up to 2× for splits at harder A-sub-intervals: only k = 0, 20
  and 39 were sampled, and only on one slab.

**Producer, one worker.**

| part | per unit | units | total |
|---|---|---|---|
| R outer | 2.5–5 s (one 53 s outlier) | about 320 | 0.2–0.5 h |
| O outer | 0.2–2.5 s | about 760 | ≤ 0.5 h |
| ring + PD | about 1–2 s | about 1,080 | ≤ 0.6 h |
| switch | not timed | 27 slabs | U1 took 452 s for 110 boxes in mpmath; the numpy interval version should be faster |

- **Total:** order 1.5–4 h at one worker, including the 2× allowance and taking the switch at U1's cost. At the 3-worker cap,
  about 0.5–1.5 h.
- **Peak memory:** about 1.6 GB per worker, after the cap (to be confirmed).

**Exports.**

| part | leaves or boxes | size at 2.7 B per leaf |
|---|---|---|
| R outer | about 39k per unit, about 12.5M in total | about 34 MB |
| O outer | about 1–7k per unit, about 2.5M in total | about 7 MB (bytes per leaf in O not measured) |
| ring | about 5.5k boxes per unit, about 6M in total | order 20 MB |
| PD | small | small |
| headers | | small |

- **Total on disk:** about 0.1 GB, against free disk of 16.7–23 GB (it has fluctuated). Disk is not the constraint.

**Checker.** Not timed; this is an extrapolation from Track 4.
- Track 4's Arb outer check cost about 19 ms per cell evaluation (82,686 evaluations in 1,577 s).
- Replaying every producer leaf (15M) would take **about 80 h at one process**, about 27 h at 3.
- A fresh Arb branch and bound per unit, as Track 4 did (about 2k evaluations per unit), would take **about 11 h** at one
  process.
- **Recommendation:** use a fresh branch and bound for the outer link, and replay the ring/PD boxes and the switch boxes.
- The checker, not the producer, is the cost driver.

**Feasibility.** Nothing in [ε₀, 0.6] looks infeasible on this sample.
- The failures are all slabs that are too wide, and the splitting rule handles them.
- The regime boundary is the one real constraint: regime O fails below about 0.004 (domain to q = ±99), so the rescaled
  regime R must cover [ε₀, 0.0462].
- The open risks are the switch box over a 0.03 slab and the edge A-sub-intervals.

## 5. Specification of the new ε-interval check function in the independent checker (not written; independent of the producer)

**Name:** `check_u2_unit(name)`, in `src/verify_certificates.py`. It is written fresh: it shares no code with
`src/u2_sample.py` and imports nothing from `src/`.

**Inputs (data only, hash-checked):**
- the population npz;
- the unit header (JSON): ε-slab, A-sub-interval, regime, θ_c, the published claims (τ, the 0.15 / 0.05 radii);
- the leaf arrays.

**Activation.** A new class `PhiEpsBall(E)` with E an Arb ball [e_lo, e_hi]. It gives balls for φ, φ_σ and φ_σσ for a σ-ball
S, valid for every ε in E. It is computed by the checker itself:
- u = √E·S as a ball, then φ = −S + (1 + E)·S³·Gt(u) and φ_σ = −1 + (1 + E)·S²·J(u).
- Gt and J are evaluated on the u-ball as follows:
  - for |u| ≤ 1, by the alternating Taylor series with an explicit Lagrange-type tail ball. The tail is bounded by its first
    omitted term, which holds once the terms decrease. This avoids the cancellation of (u − sin u)/u³ for small balls;
  - for |u| > 1, by the closed forms in Arb.
- φ_σσ = (1 + E)·S·sinc(u).
- Every result's range must be finite. If any intermediate is not finite, the bound becomes ±∞, as with HAct (the NaN rule).
- **No envelope, no monotonicity claim, and no producer constant is used.**

**Per-cell objective.**
- The logit is A·(φ − c) + b with c = the ball φ(q_c, E) at the cell centre; this is the HShift idea of Track 4. A and ε are
  both balls.
- The lower bound of L\* over the cell is the checker's own: the mean-value form and the direct form, as in
  `Objective.lower_bound`, but with the activation above. b\* is bracketed in Arb (tail form).
- Its own subdivision is allowed (in θ and in ε) up to a stated depth. A leaf passes if its rigorous bound exceeds
  U = max over the (A, ε) corners and an Arb ball bound of L\*(θ_c; A, ε).
- Convexity in A is used, as stated in the math note. In ε no convexity is assumed: the checker bounds over the ε-ball
  directly.

**Checks:**
1. **Hashes.**
2. **Domain:**
   - R: K(24) and the B(P) table recomputed by exact PAVA (existing code);
   - O: W(s_min, a_max) from `lemma_W`, mapped to (p, q) with one full b₁-period for every ε in E.
3. **Exact tiling** of the domain minus the closed 0.15 box. The leaves' integer quadtree indices are exported for this.
4. **Every leaf:** the rigorous L\* lower bound > U, or (iso/full) the exact-PAVA conditions.
5. **Ring:** on each exported box, the Arb gradient ball excludes 0 in ∂_p or ∂_q, for all (A, ε) balls.
6. **PD:** Sylvester on the Arb (p, q, b) Hessian over box × bracket.
7. **Switch:** Arb Krawczyk on Z and on every branch box with ε as a ball; active-pair validity with β = acos(1/(1+E))
   as a ball; gap signs.

**Tests (constructed cases):**
- pass: a real exported unit;
- fail: θ_c shifted by 0.3, with a witness;
- fail: a leaf bound edited above the truth;
- unresolved: an ε-ball too wide within the depth cap;
- the activation's balls contain mpmath 60-digit values at 1,000 random (σ, ε);
- NaN propagation.

**Export additions this needs from the producer:**
- the integer quadtree indices (level, i_p, i_q) per leaf, as in the finite-a exports;
- the ring and PD box lists.

### 5.1 As written (2026-10-16; not yet run)

The functions are added to `src/verify_certificates.py`. No existing function was changed, and nothing is imported from
`src/`. Tests are in `tests/test_verify_u2.py`.

**`U2Act(E, c)`: the activation.** It gives φ − c, φ_σ, φ_σσ, φ_ε and φ_εε for every ε in the ball E.
- The kernels Gt, J, S, K1 and K2 are evaluated as functions of v = E·σ², with σ² formed with a nonnegative lower end.
  No √ε is ever taken.
- The series is used when v < 400 (|u| < 20), with the geometric tail bound 2|c_M|V^M. It applies once
  3V/((2M+2)(2M+3)) ≤ ½, since |c_{m+1}/c_m| ≤ 3/((2m+2)(2m+3)) for all five kernels; M is then extended until the tail
  is below 2⁻¹⁰⁰.
- The closed form in u = √v is used when v ≥ 400. A v-ball that straddles 400 is split into the two parts.
- The first draft switched to closed forms at v = 1, and the 60-digit tests caught it: (u − sin u)/u³ on a u-ball near 1
  loses most of its width to dependency. Horner on the v-ball does not.
- Powers are repeated products, never `**`.
- A non-finite intermediate gives a ±∞ range.

**`u2_outer_arb`: the re-search, the primary route.** It runs dyadic cells over the regime's domain:
- R: K(24), with Lemma G2's iso/full tests against B(P) and B_full from the checker's own `localisation_bounds`.
- O: the `lemma_W` domain, with one b₁-period for every ε in the slab, asserted in Arb.

A cell passes by either of two tests:
- **(direct):** `Objective.lower_bound`, with ε as a ball and A as a ball, exceeds the branch's ε-ball upper bound, taken
  as the max over the two A endpoints (convexity in A).
- **(taylor):** at the point ε_c,
  D(ε_c) − h_ε·|∂_ε D(ε_c)| − ½h_ε²·(sup|∂²_ε L\*(θ)| + sup|∂²_ε L\*(θ_c)|) > target.
  - ∂_ε L\* = A·mean((σ(z) − y)φ_ε), enclosed with b\* bracketed.
  - The bound |∂²_ε L\*| ≤ ¼A²·mean((φ_ε − c)²) + A·mean|φ_εε| holds for any constant c.

If both tests fail, the cell is split in θ. The ε-interval can be split n_eps ways. A witness search reports 'fail' when a
point θ outside the box has an Arb upper bound below the branch's Arb lower bound.

**The other parts.**
- **`u2_ring_arb`:** gradient balls over box × A-ball × ε-ball, with b\* bracketed; boxes split up to a depth cap.
- **`u2_pd_arb` and `u2_pd_box_arb`:** Sylvester's criterion on the (p, q, b) Hessian enclosure.
- **`u2_switch_arb`:** replays the producer's published Z box and branch boxes with `krawczyk_arb` (existing,
  unchanged), with ε as a ball.
  - The active pair is checked without β. It needs φ_σσ > 0 on the inner window (convex, so the max is at an end),
    φ_σσ < 0 on the negative outer window, φ_σ > 0 on the positive outer window, and the active ends strictly
    dominating the others.
  - The gap signs are checked left and right of Z, every inside box must lie within Z's θ-range, and the A-tube must be
    tiled exactly.
- **`u2_slabs_closed_ok`:** checks that the slabs are closed and share their endpoints exactly (F2).

**Where it departs from §5.**
- **Switch:** §5 planned to check the active pair through β = arccos(1/(1+E)). The convexity criterion replaces that, so
  no series for β/√ε is needed at ε = 0.
- **Outer exclusion:** the Taylor-in-ε test is added because a pure ε-ball evaluation loses the cancellation in
  mean((σ(z) − y)·∂_εφ).

## 6. Status of the author's items (2026-10-16)

1. **Cap the 3 GB stage.**
   - Ring/PD box evaluation is now chunked, at most 256 boxes per evaluation, with box-count caps. The stage-by-stage RSS
     command is written, and the 3 GB watchdog is kept.
   - **Not yet measured:** the disk hold blocks it (free disk was 16.7 GB at the last gate, against the 20 GB minimum).
2. **Launcher gate.**
   - `src/u2_gate.py`: a separate `python -m src.u2_gate gate "<label>" [--disk-gb 25]` step writes a JSON entry with pid and
     time.
   - `src.u2_sample`'s `__main__` calls `require_gate()` first. It refuses (exit 3) unless the newest entry is:
     - OK;
     - written by a different process before this process started;
     - younger than 10 min;
     - newer than the last recorded launch.
   - Tests are in `tests/test_u2_gate.py`. **Not yet run** (gate FAIL on disk).
3. **Time cap and splitting.**
   - Done in code: `certify_split`, and a cap checked inside each round.
   - Constructed-case tests are written, including a real too-wide unit that must stop within its cap. **Not yet run.**
   - The failing A-sub-intervals at ε-slab width 0.1 (`python -m src.u2_sample ascan 0.1 0.2 O 40 30 ascan_w0.1`: no export,
     about 10 KB of output) **are deferred until the disk hold lifts**.
4. **Kernel bounds:** the checker route was chosen (F6, §5). The specification is written; the checker itself is not.
5. **Ring margin against the rounding allowance:**
   - The ring test subtracts one absolute allowance of 10⁻⁹ per box (`SAFETY`) from |g_k| − Σ M_ka h_a.
   - The recorded margins are net of it: min over boxes 2.78·10⁻⁶ (k = 0), 1.17·10⁻⁵ (k = 20), 1.84·10⁻⁶ (k = 39).
   - So the ratio of margin to allowance, minimised over every box of the three units, is **1.84·10⁻⁶ / 10⁻⁹ ≈ 1,840**.
   - The allowance covers float rounding in the centre gradient. That is about 800 terms of order ≤ 10, so ≤ 10⁻¹²
     relative to the terms, plus the b\* residual ≤ 10⁻¹⁴.
   - The Lipschitz sums are upper bounds computed in floats, with no separate allowance.
   - The thin margin, about 2·10⁻⁶ against gradients of about 10⁻², means the Arb checker's ring boxes will need their own
     subdivision.
6. **This page.**

## 7. Restart results (2026-10-18; sample only, NO full run)

All runs below are one process, nice 15, one thread. Each was preceded by its own gate step; the disk thresholds were set
by the author for this phase (see STATUS.md). Unit results are in `u2_sample/units/*`, the logs in `u2_sample/*.log`.

### 7.1 Memory (spec 11: ε ∈ [0.0352, 0.0452], R, k = 0)

| stage | result | time |
|---|---|---|
| outer | certified | 2.8 s |
| ring (Lipschitz) | fails at this width, quickly | 1,536 boxes, 0.22 s |
| PD | passes | 80 boxes, 0.10 s |
| switch | the first version's switch needs ≤ 3.9·10⁻⁵ sub-slabs | stopped at its 1,800 s cap |

- **Peak RSS: 1.00 GB for the whole process, imports included**, against 3.73 GB before the 256-box cap.
- Peak RSS over every later sample run: 1.15 GB (all-40 runs) and 1.51 GB (widening units).

### 7.2 Item 3: diagnosis of the switch and ring failures

**Switch box Z alone** (4-variable Krawczyk at the slab's float switch point):

| ε-slab | plain | first-order (mv) | sheared |
|---|---|---|---|
| [0, 10⁻³] | pass | pass | pass |
| [0, 10⁻²] | fail | fail | fail |

**The whole switch part** (Z plus the branch rows that sign the gap):
- The first version, and the mv form, need sub-slabs ≤ 3·10⁻⁵ at every ε tested (0 to 0.59). The cause is in the branch
  rows: the row boxes must contain the branch for every ε in the slab, so their gap enclosure is too wide near Z.
- **The sheared form** (`switch_unit_shear`) follows the branch along its predicted ε-dependence. Z's A-range is treated
  as moving, and the band rows are decided ε-piece by ε-piece. Widths that pass:

| ε | 0 | 0.035 | 0.1 | 0.3 | 0.59 |
|---|---|---|---|---|---|
| widest sub-slab that passes | 3·10⁻³ | 3·10⁻³ | 5·10⁻³ | 5·10⁻³ | 5·10⁻³ |
| seconds per sub-slab | about 2 | 2 | 2–3 | 3 | 3 |

  On [0, 10⁻²] it **fails** (Z does not contract).
- **Neither form passes on [0, 10⁻²].** The sheared form is implemented and in use. The switch slab widths needed are:
  3·10⁻³ for ε < 0.1 and 5·10⁻³ for ε ≥ 0.1, which is about 134 switch sub-slabs over [0, 0.6].

**Ring:**

| ε-slab | Lipschitz | first-order 3-D in (p, q, b) | sheared 3-D |
|---|---|---|---|
| [0, 10⁻³] | pass | pass | pass |
| [0, 10⁻²] | fail | pass | pass |
| [0.0352, 0.0452] (spec 11) | fail | pass | pass |

- The first-order 3-D form takes about 11 s per unit.

**Implemented on both sides, with no shared code** (tests: `tests/test_u2_shear.py` and `tests/test_verify_u2.py`, all on
ε-intervals that contain 0):
- **Producer:** `switch_unit_shear`, now the default in `switch_split`; `ring3d` (mv), used as the fallback when the
  Lipschitz ring fails; `gap_sheared`; `krawczyk_modes`.
- **Checker:** `u2_ring3d_arb` (first-order form in ε); `u2_switch_shear_arb` (sheared Krawczyk in Arb with the
  producer's shear vector taken as data; any vector gives a valid test); `U2Act.dse` and the JJ kernel.
- 42 of the new tests pass.

### 7.3 The ε-slab width 0.1 scan (slab [0.1, 0.2], original-coordinate domain, 40 A-sub-intervals, 30 s cap)

**All 40 fail**, each at the time cap. The live cell is the same in all of them: the domain corner p ≈ 6.26 ≈ p_max,
q ≈ −9.92 ≈ −π/√ε_lo, with lower bound D ≈ −19 to −20. So at width 0.1 the obstruction is the edge of the large-|σ|
domain, not the branch.

### 7.4 All 40 A-sub-intervals on four slabs (producer; `probes6.jsonl`)

| slab | outer | outer time (total / median) | leaves | ring | PD | switch | total |
|---|---|---|---|---|---|---|---|
| R [0, 0.005] | 40/40 | 138 s / 3.5 s | 1.57M | 40/40 (mv3d), 488 s | 40/40, 5 s, λ ≥ 0.026 | 2 sub-slabs, 7 s | 640 s |
| R [0.0362, 0.0462] | 40/40 | 115 s | 1.57M | 40/40, 377 s | 40/40, 5 s, λ ≥ 1.6·10⁻⁵ | 4 sub-slabs, 13 s | 509 s |
| O [0.30, 0.33] | 40/40 | 11 s | 49k | 40/40, 331 s | 40/40, 88 s (2 sub-slabs) | 16 sub-slabs, 94 s | 524 s |
| O [0.57, 0.60] | 40/40 | 13 s | 41k | 40/40, 459 s | 40/40, 313 s, **λ ≥ 4.7·10⁻⁷** | 16 sub-slabs, 93 s | 877 s |

- **The PD margin is thin at large ε** (λ ≥ 4.7·10⁻⁷ on [0.57, 0.60]).
- Every ring unit needed the mv3d form; the Lipschitz form failed at these widths.

### 7.5 Checker (check_u2_unit itself; Arb, 80 bits)

**Outer re-search** (the primary route): 0.29–0.47 s per cell evaluation.
- On the O unit [0.30, 0.33], k = 20: **unresolved after 12,000 evaluations** (≈ 1.5 h). 6,303 cells passed by the direct
  ε-ball bound, 0 by the Taylor test, and 5,394 ε-splits were made.
- The first version, which split ε only by depth, did no better (unresolved after 12,000 evaluations).
- On the R unit: unresolved after 3,000 evaluations (1,000 s).
- **Cause:** the checker's cell bound is the existing first-order `Objective.lower_bound` with A and ε as balls. Near the
  box, the ε-ball loses the cancellation that the producer's coupled second-order bound keeps.
- **As written, the re-search is not feasible at the producer's slab widths.**

**Ring, first-order 3-D in Arb:** 0.51–0.85 s per tile, so 512 tiles is 4–7 min per unit. All 16 sampled tiles per unit
certified.

**Thin-margin ring boxes:** the 10 thinnest producer boxes of two units (producer margins 1.8·10⁻⁶ to 6.8·10⁻⁵) were
all certified by the checker:

| form | certified | seconds per box |
|---|---|---|
| ball form (`u2_ring_arb`) | 10/10 | 0.024 |
| first-order 3-D | 10/10 | 0.47 |

**PD:** 0.7 s (R) to 21 s (O, 320 boxes) per unit.

**Switch replay** (`u2_switch_shear_arb`, on the first sub-slab of each export):
- Z's Krawczyk test passes on both. On the O sub-slab, the active pair is not certified on Z's hull.
- **Most rows fail**: 109 of 188 (R) and 165 of 176 (O), at 22–35 s per sub-slab.
- It passes on the narrow-tube test slab [0, 3·10⁻⁴]. The replay uses the producer's radii as they are; it needs its own
  radius inflation, or narrower sub-slabs.

### 7.6 Projection (NOT a full run; for the author's decision)

**Slabs:**
- R: 5·10⁻³ slabs on [0, 0.03] (6), then 10⁻² on [0.03, 0.0462] (2). That is 8 slabs.
- O: 0.03 slabs on [0.0462, 0.6]. That is 19 slabs.
- In total 27 slabs × 40 A-sub-intervals = 1,080 units, plus about 134 switch sub-slabs.

**Producer:**
- Measured cost per slab: 509–877 s, all links included. That gives about 8 × 600 s + 19 × 700 s ≈ **5 h at one worker**.
  With up to 1.5× for splits, ≤ 7.5 h.
- At the 3-worker cap: ≈ 1.7–2.5 h.

**Memory:**
- Peak RSS ≤ 1.51 GB per worker (measured).
- Each worker stays well under the 3 GB rule. Three workers need about 4.5 GB, which fits when the memory gate passes
  (≥ 25 % of 16 GB free). **3 workers fit; the 3-worker cap binds.**

**Exports:**

| part | size |
|---|---|
| R outer leaves (~105 KB per unit × 320) | ≈ 34 MB |
| O outer leaves (~5 KB per unit × 760) | ≈ 4 MB |
| switch sub-slabs (~50 KB × 134) | ≈ 7 MB |
| Lipschitz ring boxes where used (~52 KB per unit) | ≤ 56 MB |
| mv3d ring box lists, if exported for replay (~100 KB per unit) | ≈ 110 MB |
| **total** | **≲ 0.2 GB on disk**, against 15–19 GB free |

**Checker (measured unit costs):**
- **Outer: ≥ 1.5 h per unit and still unresolved**, so ≥ 1,600 h for 1,080 units. **Infeasible as written.**
- Ring: 4–7 min per unit, ≈ 100 h.
- PD: ≤ 21 s per unit, ≈ 3 h.
- Switch: currently fails most rows.
- **What would make it feasible:**
  1. the producer's coupled second-order cell bound (difference to the branch, gradients in θ, A and ε at the cell centre,
     curvature from Arb sup-balls), derived independently in Arb;
  2. the checker's own Krawczyk radius inflation in the switch replay;
  3. the ball-form ring (0.024 s per box) as the first try, with the 3-D form only as the fallback.

  Each of these is code still to write; none has been timed.

**Widening the window to [0.5 A_ε, 2 A_ε]** (outer exclusion only; sampled at r = 0.5, 0.75, 1.5, 2):
- **Above the window:** cheap. R: 2.3–2.4 s per unit. O: 0.2 s per unit.
- **Below the window, O:** cheap, 0.2–0.3 s per unit.
- **Below the window, R (r = 0.5, 0.75 at ε ∈ [0, 0.005]): fails at the 300 s cap.** The live cell is the K(24) corner
  (p ≈ 24, q ≈ −50.8): at lower A the branch bound rises above the B(P) table's exclusion. That region would need a new
  bound.
- **Unit count:** the window is ≈ 20× wider, so about 800 A-sub-intervals per slab.
- **Projected producer cost of the widening:**
  - O ≈ 3–4 min per slab, about 1 h for 19 slabs;
  - R above the window ≈ 25 min per slab, ≈ 3.5 h for 8 slabs;
  - R below the window: infeasible without a new bound.

## 8. The producer run over [0, 0.6] (run1), the three checker pieces, and the checker re-projection (2026-10-18/19)

**Everything in §8.1 is PRODUCER-ONLY and has NOT been independently checked.** It is float arithmetic at the producer
standard, not a certificate, and is labelled that way in every record and export. Nothing in it is reported as certified.

### 8.1 run1: the producer over ε ∈ [0, 0.6] at 3 workers (`run1_specs.json`; `units/run1/`; `run1_w*.jsonl`)

**Layout:** 27 closed ε-slabs sharing their endpoints, × 40 A-sub-intervals = 1,080 units, plus the sheared switch for
each slab.
- R: 6 slabs of width 0.005 on [0, 0.03], then [0.03, 0.04] and [0.04, 0.0462].
- O: slabs of width 0.03 on [0.0462, 0.5862], then [0.5862, 0.6].

**Run:**
- Workers launched at 22:58, each after its own gate.
- They were restarted at 23:08 with the outer ε-split rule, because slab [0.005, 0.01] failed on its unsplit units.
- They finished at 00:29. Wall time was about 1.5 h; the summed per-slab time is 12,200 s after the restart.
- The watchdog never fired.

**Producer result** (not independently checked): every unit and every slab's switch passed the producer's tests, and no
unit failed after splitting.

| link | result |
|---|---|
| outer exclusion | 1,080 / 1,080 units |
| outer ε-splits | 13 units of slab [0.005, 0.01] needed one halving into two 0.0025 halves |
| ring | 1,080 / 1,080 (first-order 3-D form wherever the Lipschitz form failed) |
| PD | 1,080 / 1,080 |
| switch | 27 / 27 slabs, 269 sheared sub-slabs, widths 0.0019–0.0069 |

**Thin PD margins at large ε** (producer λ_lower):

| slab | λ_lower ≥ |
|---|---|
| [0.0762, 0.1062] | 9.6·10⁻⁷ |
| [0.4362, 0.4662] | 3.7·10⁻⁷ |
| [0.4662, 0.4962] | 9.5·10⁻⁸ |
| [0.5562, 0.5862] | **1.3·10⁻⁸** |

**Peak RSS:** 1.62 GB per worker.

**Exports** (each file labelled PRODUCER-ONLY):

| export | size |
|---|---|
| outer leaves, 1,070 npz (the split units export none) | 65 MB, including the ring boxes |
| Lipschitz ring boxes (200 units) | 21 MB |
| switch sub-slab files | 13 MB |
| unit records | 35 MB |
| **total** | ≈ 113 MB |

### 8.2 The three checker pieces (independent of the producer: `src/verify_certificates.py`, additions only)

- **(a) The coupled second-order cell bound:** `u2_coupled_lower`, math note §23 (Proposition 23.3, every step labelled).
  Its remainders are Arb sup-balls of the kernel functions: no grid envelopes. It is the first test in `u2_outer_arb`.
- **(b) The checker's own box radii in the switch replay:** `u2_krawczyk_shear_inflate`. It starts Z from the §16.3
  method radii and the rows from 10⁻⁷, and grows them by 1.5 × the Krawczyk image. The producer's radii are ignored (a test
  zeroes them).
- **(c) The ring:** `u2_ring_fast_arb` runs the direct gradient-ball test first and the first-order 3-D form only for the
  tiles it leaves open.
- **Fixes found on the way, in the switch replay:**
  - The active-pair test checked convexity of the whole inner window, which fails once the window reaches σ < 0.3. It now
    checks "φ decreasing on [σ_lo, 0.3], convex on [0.3, σ_hi]", piecewise on 16 sub-intervals.
  - The gap at ε_c is now enclosed in centred (mean-value) form.
- **Tests:** `tests/test_verify_u2.py` and `tests/test_u2_entrypoints.py` (with the earlier U2 tests, 52 passed). They
  include pass, fail and undecided cases on ε-intervals that contain 0, and a sampled soundness check of (a).

### 8.3 Checker timing on run1's real exports (3 processes, each gated; Arb at 80 bits)

| unit | outer re-search | ring (fast first) | PD |
|---|---|---|---|
| R [0, 0.005], k = 20 | **pass**, 1,672 evaluations (1,325 coupled), 738 s | 512/512 tiles by the fast test, 13 s | 16 boxes, 0.7 s |
| O [0.1962, 0.2262], k = 20 | **pass**, 1,465 evaluations (1,091 coupled), 914 s | 512/512 fast, 19 s | 268 boxes, 17.5 s |
| O [0.5862, 0.6], k = 0 | **pass**, 7,477 evaluations (5,317 coupled), 6,081 s | 512/512 fast, 19 s | 80 boxes, 5.5 s |
| R [0.03, 0.04], k = 20 | still running when stopped: **≥ 2.9 h** | — | — |
| O [0.5562, 0.5862], k = 20 | still running when stopped: **≥ 2.9 h** | — | — |
| O [0.5862, 0.6], k = 39 | still running when stopped: **≥ 1.2 h** | — | — |

- **Peak RSS per checker process:** 0.04–0.05 GB.
- **Switch replay** (first sub-slab of R [0, 0.005] and O [0.1962, 0.2262]): Z's Krawczyk test passes and so does the
  active pair (after the fix), but **most rows fail**.
  - On [0, 0.0025]: 111 of 188 rows fail. The reasons are 73–95 failed sign checks and 16–18 failed Z-containment checks.
  - Letting the checker halve failing rows six more times did not help: 5,194 failing pieces after 5,162 splits, 40 min.
    So the cause is not the row width; it is the checker's gap and containment enclosures.
  - **The switch replay is UNRESOLVED.**
- **PD on every unit of [0.5562, 0.6]:** see §8.4.

### 8.4 PD on [0.5562, 0.6] (the producer's thinnest PD margins)

**The checker's PD test (Sylvester's criterion on Arb Hessian enclosures) passes on all 80 units of [0.5562, 0.5862] and
[0.5862, 0.6]** (`chkPD.json`).
- These include the producer's thinnest PD margin, λ ≥ 1.3·10⁻⁸.
- Cost: 16,640 boxes in 1,077 s in total, at most 22 s per unit.
- The sampled ring tiles there (8 per unit) all passed the fast test.

### 8.5 Checker re-projection: ABOVE 48 h at 3 workers (stop and report, as instructed)

- **Measured outer re-search cost per unit:** 12–101 min on the three units that finished, and at least 72–174 min on the
  three that were stopped. Ring and PD add ≤ 0.5 min.
- **Lower bound** from the cheapest unit (12.6 min): 1,080 × 12.6 min ≈ 227 h at 1 worker, ≈ **76 h at 3 workers**.
- **From the six sampled units** (mean ≥ 92 min, counting the unfinished units at their elapsed times): ≈ 1,650 h at 1
  worker, ≈ **550 h at 3 workers**.
- **The switch replay is not yet feasible:** its rows do not certify.
- **Peak memory per checker worker:** ≤ 0.05 GB. Memory does not bind; 3 workers fit easily.
- **The checker was not run** over the producer output. Per the author's rule, it goes to a cluster.

## 9. The checker after the author's four steps (2026-10-19). The 1,080-unit result is still PRODUCER ONLY, NOT INDEPENDENTLY CHECKED.

### 9.1 Step 1: soundness first. No producer problem

**What was checked** (`u2_sample/soundness_switch.json`, `src/u2_soundness.py`):
- 20 switch-replay rows that had failed, spread over six sub-slabs from ε = 0 to ε ≈ 0.56.
- For each row, the active gap was evaluated at 60 digits, at the centre and four corners of each sheared box and at
  three ε per piece. That is 570 points in all.
- The 60-digit values come from mpmath and the closed form of φ_ε (h at ε = 0), a code path shared with neither the
  producer nor the checker.

**Result:**
- **Every point lies inside the producer's enclosure.**
- On the 270 points where the producer's enclosure has a definite sign, **every sign agrees**.
- The 61 "wrong-sign" points all lie on pieces where the producer's enclosure contains 0, so the producer claims no
  sign there.

**Cause: a checker bug.** It cut each row's ε-range at the crossings with Z's moving A-range, then classified each piece
with ball comparisons taken over the whole piece. At a crossing, the comparison is an equality, which the ball cannot
decide. So the checker demanded a sign exactly on pieces where the switch passes through the row.

**Fix:**
- Each crossing gets a sliver [t − δ, t + δ] of its own, with δ = 10⁻⁶·Δε, large enough for the 30-bit Arb radii. In
  the sliver the row must lie in Z AND carry the side's sign.
- The pieces between slivers then classify strictly.
- After the fix the replay **passes** on the first sub-slab of [0, 0.005], [0.03, 0.04], [0.1962, 0.2262] and
  [0.5562, 0.5862].

### 9.2 Step 2: where the width came from, term by term (`u2_sample/terms_outer*.json`)

On 40 outer cells from four units, three of which had not finished before:
- the centre value D(w_c) and all four first-order terms are equal in producer and checker;
- **only the checker's remainder M_comp blew up**, by up to 250× at large |σ|.

**Cause** (math note §23.5, Lemma 23.4): Horner on a wide v-ball. The series was used up to v = 400, and the width of a
ball Horner evaluation grows like e^{√V}·w, while the kernel decays.

**Fix:** series only for v < 16; closed forms above.

**Second checker bug found on the way:**
- **What:** a midpoint-radius power u⁵ of a wide positive ball reached 0, a division gave NaN, and `min`/`max` dropped
  the NaN half of a split ball. That is an unsound, too-narrow kernel enclosure.
- **Fix:** powers of positive balls are taken from their endpoints; any non-finite half makes the result non-finite;
  `_absmax` of a non-finite ball is +∞. A regression test covers it.
- **Consequence: every checker number produced before this fix is VOID, and has been redone.** That covers the §8.3–8.4
  timings, the PD pass on [0.5562, 0.6] and the earlier switch-replay results.

**After both fixes:**
- the checker's M_comp is within 3.3× of the producer's on every one of the 40 cells, and smaller near the box;
- the checker's lower bound of D is positive on every cell.

### 9.3 Step 3: outer exclusion, replay with refinement (`u2_outer_replay`, `u2_leaf_cells`, `u2_leaf_coverage`)

**The producer's leaves are used as GEOMETRY ONLY. The checker trusts none of the producer's bounds.**
- **Indices:** each leaf (p, q, h_p, h_q) is mapped to integer quadtree indices of a dyadic grid.
- **Tiling:** the checker verifies, in exact integers and exact rationals, that the leaves tile its own grid rectangle
  except for cells inside the closed 0.15 box.
- **Domain:** the rectangle is compared with the regime's domain, rounded outward in Arb. Any rounding-level strip
  between the two becomes extra cells.
- **Cells:** every cell is checked by the checker's own Arb bounds (coupled, direct, iso, full), subdivided in θ and
  then ε on failure.
- **Time caps:** a unit gets a time cap with ε-splitting, then the re-search as a fallback.

**Exports.** run1 has valid leaf exports for 1,067 of the 1,080 units. Missing are the **13 units of slab
[0.005, 0.01] that were split in ε** (s01 k000–k012). The files present for k000–k002 are stale, from their failed first
attempt. Regenerating all 13 is about 1.3 MB.

**Measured: replay is slower than re-search here.**
- A rescaled-regime unit has about 39k producer leaves, at about 0.4 s each. Started at leaf level, that is hours per
  unit, so the replay timing runs were stopped at 30 min.
- The re-search starts from coarse cells and certifies large regions in one evaluation.
- So **re-search is the primary route, and replay is kept as the code path for units whose re-search fails.**

### 9.4 Step 4: re-timed on run1's real exports after all fixes (3 gated processes, Arb 80 bits)

| unit | outer re-search | ring (fast first) | PD | switch sub-slab replay |
|---|---|---|---|---|
| R [0, 0.005], k = 20 | pass, 1,602 evaluations, 1,138 s | 512/512 fast, 20 s | 1.0 s | [0, 0.0025]: **pass**, 73 s |
| R [0.03, 0.04], k = 20 (was ≥ 2.9 h) | pass, 1,925 evaluations, 1,342 s | 512/512, 20 s | 4.2 s | first: **pass**, 92 s |
| O [0.1962, 0.2262], k = 20 | pass, 887 evaluations, 889 s | 512/512, 31 s | 27 s | first: **pass**, 109 s |
| O [0.5562, 0.5862], k = 20 (was ≥ 2.9 h) | pass, 683 evaluations, 690 s | 512/512, 41 s | 39 s | first: **pass**, 130 s |
| O [0.5862, 0.6], k = 0 (was 6,081 s) | pass, 585 evaluations, 615 s | 512/512, 31 s | 9.5 s | see below |
| O [0.5862, 0.6], k = 39 (was ≥ 1.2 h) | pass, 601 evaluations, 624 s | 512/512, 31 s | 9.4 s | see below |

- **Switch on [0.5862, 0.6]:** its two sub-slabs (width 0.0069) need the checker's own ε-split (`u2_switch_shear_split`).
  **Both pass**: each is split into two halves and takes about 1,100 s.
- **Peak RSS:** ≤ 0.05 GB per checker process.

### 9.5 Re-projection (core-hours and wall time). Above 48 h at 3 workers: STOP and report for a cluster allocation

**Outer re-search, measured per unit:** 615–1,342 s.
- Rescaled-regime units: 1,240 s on average, × 320 units ≈ 110 core-h.
- Original-coordinate units: 705 s on average, × 760 units ≈ 149 core-h.

**The other links:**
- Ring (fast test first) and PD: about 45 s per unit, × 1,080 ≈ 13.5 core-h.
- Switch: 269 producer sub-slabs at about 70 s to 1,100 s each. Assuming about 200 s on average, ≈ 15 core-h.

**Total ≈ 290 core-hours.**
- **≈ 96 h wall at the 3-worker cap**, which is above the author's 48 h limit. **The checker is NOT run locally.**
- Peak memory ≤ 0.05 GB per process, so a cluster node can run many workers.
- At 32 cores: ≈ 9 h wall.

**Inputs a cluster run needs:**
- the population certificate file (hash-checked);
- run1's unit headers and switch exports (≈ 13 MB);
- the 1,067 leaf files, needed only for the replay fallback (≈ 65 MB);
- the 13 units of s01 k000–k012, which go by re-search (no leaves needed).

## 10. NaN audit, the regenerated leaves, and the cluster package (2026-10-19)

**NaN audit** (`results/certificate_audit.md`, "NaN audit"):
- The affected helpers (`_pow`, `u2_kernel`'s split union, `_absmax`) first appear in bd1083e and 3ce551d
  (2026-10-18/19), and only U2 code calls them.
- `git diff 69af96f 0f3b252` on the checker is one insertion hunk: every older function is unchanged.
- Every earlier check, from the paper-submitted-v5 checker through Track 4, used only older functions. Their range
  functions are NaN-guarded since 12c79ea, or cannot produce NaN.
- **No earlier verdict is affected; none needed a re-run.** The U2 checker results from before the fix were already
  voided and redone.

**Leaves for the 13 ε-split units** (s01 k000–k012): regenerated per sub-slab, as `run1_s01_kNNN_e{0,1}.npz` plus meta
(26 files, about 39k leaves each). The stale files from the failed first attempts are renamed `*.stale`. All are
PRODUCER-ONLY.

**Cluster package** (`results/uniform_eps/u2_cluster/`):
- `jobs.json`: 1,080 units and 269 switch sub-slabs.
- `MANIFEST.sha256`: every input export, the checker source, the run script, the verify-back script, the gate and store
  helpers, the version pins, jobs.json and the reference verdicts.
- `BUNDLE_FILES.txt`: the inputs that are not in git.
- `requirements-u2-cluster.txt`: numpy 2.4.6 and python-flint 0.9.0. The run script refuses on any other version, and
  the wheel hashes are generated on the VM, since wheels are platform-specific.
- `src/u2_cluster_run.py`, in this order:
  - verifies every manifest hash first and the pinned versions; refuses without a separate gate entry;
  - phase reference: the 5 units must match the laptop's verdicts exactly, else it stops with exit 4;
  - phase full: refuses unless the reference phase matched against this manifest;
  - each job writes one atomic result file; resumable.
- `src/u2_cluster_verify.py`: the local verify-back.
- `VM_COMMANDS.md`: the author's command sequence.
