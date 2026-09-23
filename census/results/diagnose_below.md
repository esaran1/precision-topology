# The Block 4 placements at 0.9× (EXPLORATORY)

Characterisation, not a test. Producers: `src/diagnose_below.py` (`main`, `endpoints`). Artifacts:
`diagnose_below.csv` and `diagnose_below_endpoints.csv`.

**Runs**: seven checkpoints from three runs (seeds 200082, 200087, 200127), all "preserved" replays placed at
R/R_glob = 0.9 after 4,000 steps. Each is analysed on its **own** 400-point training set, at the held scale
|w₂| = 0.9 × the population |w₂|,glob = 4.4606.

- **Own threshold.** The certified own R_glob is 1.18, 1.23 and 1.22 × the population value. So the held
  scale is well below each run's own switch. The own-seed threshold does not explain these placements.
- **Persistence (Q2).** All seven remain placed at 16,000 and at 64,000 steps. One further replay (seed
  200135, checkpoint 1,776) becomes placed between 16,000 and 64,000. It is not characterised here.
- **The endpoint is a strict local minimum.**
  - At the 64,000-step endpoint the profiled-loss gradient norm is ≤ 8e−4, and the Hessian eigenvalues
    are 1.54–2.36 (positive definite).
  - Descent from the endpoint stays in the same minimum.
  - The endpoint's gap is G = +0.012 to +0.018 (placed).
- **Its loss sits above the global minimum.** The certified global minimum of the run's own objective at
  this scale is in the G ≤ 0 region (G = −0.048 to −0.059). The endpoint's loss exceeds it by
  **0.008 (200082), 0.025 (200087) and 0.064 (200127)**.
- **So these are metastable correct-placement basins.**
- **Which basin**: the mirror of the global branch. In every case the endpoint's w₁ has the sign
  opposite to the global argmin's, with |w₁| and b₁ nearly equal (e.g. +0.836 / 3.819 against
  −0.846 / 3.876).
  - On the population objective, which is x-symmetric, the branch and its mirror are exact images
    with equal loss.
  - On a finite training set that symmetry is broken. The two basins then have different losses and
    different gaps, and at this scale the higher-loss mirror is the placed one.
- **Barrier to the global branch: exactly log 2.**
  - Any path from w₁ > 0 to w₁ < 0 crosses w₁ = 0. There the network is constant, and the profiled loss
    equals the label entropy, log 2 (balanced classes). So the barrier is at least log 2.
  - The sublevel-set computation on the 0.004 grid connects the two basins at 0.693143–0.693145
    (log 2 = 0.693147). The barrier is therefore log 2, 0.30–0.32 above the local minimum.
  - b₂ is profiled and w₂ is frozen in these replays. So the barrier computed on the profiled (w₁, b₁)
    landscape equals the minimal barrier in the full (w₁, b₁, b₂) space at that scale. It is exact for
    the replay's problem, up to the 0.004 level grid; here the value is fixed analytically at log 2.
- **Scope of the single-branch certificate.**
  - The certified single-branch result at a = 1.30 (`cond_scan_certified_a130.csv`) is for the
    **population** objective. Its competitor exclusion is stated modulo the x-symmetry (balls around
    the minimiser and its mirror).
  - These seeds' own objectives **do have a second basin** at this scale: the mirror, a strict local
    minimum with G > 0, separated from the global branch by a barrier of log 2.
- **Bearing on S1.** The seven replays count against S1's per-replay agreement as registered. They are
  misclassified by the own-seed rule, which tracks the global branch's switch and not the mirror basin's.
  Whether mirror-basin occupancy accounts for more of S1's shortfall is not examined here.
