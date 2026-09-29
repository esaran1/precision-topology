# Design 2A: Adam per-run ordering at an unseen activation value (for approval; not registered)

**Setting**
- Width 1, f_a(t) = t + a sin t, at a = 1.85.
- Why 1.85: a scan of every results CSV found it only in a landscape table (`corner_tracking.csv`) and a search log
  (`search_anneal.csv`). It has no training or crossing data. (1.35 and 1.70 have onset data; 1.65 was used by Track A.)
- 80 fresh seeds, free Adam training (lr 0.01), each on its own 400-point sample, with every-step crossing detection.

**Prediction.** It is the Track A pipeline unchanged, except for one rule: when the preconditioner is read.
- The rule point is the last upward passage of 0.5·s\*_frozen. At the rule point, identify the occupied branch and its
  switch s\*_run.
- **P is frozen at t_sw**, the first step at which |w₂| reaches s\*_run. That is the candidate rule found post hoc at
  a = 1.65, where it gave Spearman 0.95 and 91% within 10%. It is read from the output-scale trajectory in real time,
  before the crossing.
- Primary prediction: the trajectory-integrated lag with P = P(t_sw). Secondary: the closed form κ_k χ with the same P.

**Derivation.** The lag law r = κχ (math note §13) needs P at the time the lag is set. The rule point (0.5·s\*) is about
1,000 steps before the crossing, longer than Adam's v̂ memory. t_sw is within one relaxation time of it.

**Frozen before any training** (one hashed commit, plus an OpenTimestamps proof):
- the validated switch and κ at a = 1.85, by continuation from the certified a = 1.60 switch plus the conditional
  search, as done for a = 1.65;
- the winding rule (canonical b₁ ∈ (−π, π]);
- each seed's own threshold and branch switch;
- the pipeline code and its hash.

**Criteria, registered** (matching the a = 1.65 test):
- A1: per-run Spearman(pred, obs) ≥ 0.5.
- A2: fraction of runs within 10% ≥ 0.6. **Needs your number:** "fraction within 10%" had no threshold in your text,
  and 0.6 is my proposal.
- A3: median obs/pred in [0.9, 1.1].

**Validity**
- At least 60 of 80 runs cross.
- t_sw precedes the crossing in at least 90% of crossing runs. The others are unscored and reported.
- The rule point precedes the crossing in every scored run.

**Competing outcomes**
- PASS on all three: the a = 1.65 failure was a measurement-point artifact, now shown prospectively.
- A1 fails: per-run variation is set by more than P at t_sw.
- A3 fails: the magnitude is off at this a.

**Compute.** About 80 Adam runs to about 3·s\* at one worker, plus prediction replays: under 1 hour.
