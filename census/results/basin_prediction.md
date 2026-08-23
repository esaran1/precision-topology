# Registered predictions: basin geometry of the findability gap (Parts 1a–1c)

Written before the basin experiments run. The asset: dense-verified
solutions now exist at parameter values where search fails — including
**pure `sin_family(1.02)` depth-4 networks** (all activations f_a; the
construction occupies layers 1–2, the continuation layers 3–4; 4/40
continuation seeds dense-verify). Solutions at findable values come from
the recorded sweeps.

## Protocol, fixed in advance

- **Perturbation**: every parameter tensor gets Gaussian noise with
  standard deviation `ε · RMS(tensor)` (relative units). ε sweeps
  {0.003, 0.01, 0.03, 0.1, 0.3, 1.0}, 20 noise seeds per level.
- **Recovery**: full Adam training (lr 1e-2, 2,000 steps, all parameters
  trainable) from the perturbed point; success = eval-0 **and** 100k
  dense verification. ε₅₀ = largest ε with recovery ≥ 50%.
- **Solutions**: constructed at a = 1.02, 1.05, and 1.10; found (SGD)
  at a = 1.09, 1.10, 1.25, 2.0, 3.0. At 1.10 both kinds exist — the
  calibration point for whether constructed and found solutions have
  comparable basins at the same `a`. Depths vary across found solutions
  and are reported alongside; this is a known confound.
- **Initialization distance (1b)**: 200 Kaiming initializations of each
  architecture; L2 distance to the solution reported raw and normalized
  by the solution's norm. Registered expectation: the normalized number
  is the meaningful one (the constructed solutions carry ~600×
  amplification weights, so raw distance is dominated by the solution's
  own scale); both reported, and a disagreement is itself data.

## Predictions

**P-basin.** ε₅₀ grows with `a`: smallest at 1.02, growing through the
onset, largest at 3.0.

**P-onset.** At a = 1.02 the basin radius is far smaller than the
typical initialization distance (normalized units); near a ≈ 1.09 the
two become comparable. This is the mechanistic claim: findability turns
on where basins grow to the scale of initialization distance.

**P-calibration.** At a = 1.10, constructed and found solutions have
basin radii of the same order. If they differ wildly, constructed-basin
numbers do not speak for SGD's solutions and P-onset cannot be tested
with constructed basins alone — the analysis then rests on found
solutions only.

**P-barrier (1c).** Linear interpolation from random initializations to
the solution (Goodfellow-style loss profiles — standard methodology,
applied here to an analytic-threshold gap): a loss barrier exists at
a = 1.02 and its height falls with `a`, becoming negligible near the
onset. A flat barrier profile across `a` rules this account out.

## Named failure conditions

- ε₅₀ flat in `a` while findability jumps → the basin-size account is
  wrong; report as failure and fall back to 2a/2b (amplification).
- ε₅₀ non-monotone or dominated by depth rather than `a` → the depth
  confound wins; rerun at matched depth before concluding anything.
- Constructed basins ≪ found basins at 1.10 → construction places
  solutions in atypically narrow basins; restrict claims to found ones.
