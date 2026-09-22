# Item #9: MEP method details for the appendix

Junyu's theory review item **#9**. Source: `src/barrier.py`. Data: `barrier.csv`,
reported in `barrier_results.md`. This note records the definitions and numerical
parameters behind the reported barrier heights, in the form the appendix needs.

---

## Energy

The **training loss** — binary cross-entropy with logits on the task sample,
`N_PER_CLASS = 200` per class, so **400 points**, drawn with `DATA_SEED = 0` and
held fixed across every path and every `a`. No minibatching; the energy is the
full-batch loss.

    E(θ) = BCE_with_logits(w₂·f_a(w₁x + b₁) + b₂,  y)

## Barrier definition

    ΔE  =  max over the path of E  −  max(E(θ_start), E(θ_end))

the **Goodfellow convention** (barrier measured above the *higher* endpoint), the
same one already used in `basin_profiles.csv`. Consequences worth stating
explicitly:

- ΔE ≥ 0 always, and **ΔE = 0 means the path is monotonically non-increasing in
  energy from the higher endpoint** — a downhill connection, not a small barrier.
- Subtracting the **higher** endpoint (not the start, and not the mean) is what
  makes the measure symmetric under reversing the path.

## Endpoint convention

**Start**: a random initialisation, `θ ~ U(−1, 1)^4`, from a **separate RNG
stream** — `torch.manual_seed(100_000 + seed)` — so path initialisations are
independent of the training seeds used to find solutions.

**End**: two endpoint types are reported, and they are **not** interchangeable:

| type | construction | why both |
|---|---|---|
| `constructed` | the analytic `\|w₂\| = 1` solution (fold the data window into the local **minimum** `m₀ = π + arccos(1/a)`, centre `b₂` in the resulting gap) | exists at **every** `a`, so `a` can be varied with the endpoint held fixed in scale |
| `found` | `θ` from an actual Adam run (lr 1e-2, 2,000 steps) that **passes `solves()`** | exists only where findability > 0, and carries the run's own large `\|w₂\|` |

The distinction matters for the reported result: holding the endpoint type fixed
isolates `a`, and the two types give barriers differing by an **order of
magnitude** (0.021 vs 0.320 at `a = 1.5`) because the height is dominated by the
endpoint's own weight scale.

## Path discretisation

| estimator | images | what is optimised |
|---|---:|---|
| **linear** | `PATH_POINTS = 101`, uniform in `α ∈ [0,1]` on the straight segment `(1−α)θ_start + αθ_end` | nothing; a chord |
| **MEP (string)** | `STRING_IMAGES = 41` | the 39 interior images |

**String method**, per iteration: (i) one gradient-descent step on each interior
image, `lr = STRING_LR = 5e-3`, on the **sum** of the interior energies (so the
images are independent, no spring term); (ii) **re-space by cumulative arclength**
— compute `‖θᵢ₊₁ − θᵢ‖`, cumulatively sum, normalise to [0,1], and linearly
interpolate each of the 4 coordinates onto a uniform grid. Endpoints are **pinned**
throughout. If the total arclength collapses to 0 the loop breaks.

This is the **simplified** string method: reparametrisation-by-arclength with no
tangent projection and no nudging force. It converges to a path lying along the
descent flow, which is sufficient for a *zero* barrier to be meaningful (a
monotonically descending discretised path is exhibited) but does **not** certify a
true minimum-energy path in the sense of a converged saddle search.

## Tolerance — and the honest statement

**There is no convergence tolerance.** The code runs a **fixed
`STRING_STEPS = 400`** iterations. Because the appendix should not present a fixed
iteration count as if it were a converged result, the iteration-dependence was
measured:

| iterations | 50 | 100 | 200 | **400** | 800 | 1,600 | 3,200 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ΔE_MEP, `a = 1.50` | 0.000000 | 0.000000 | 0.000000 | **0.000000** | 0.000000 | 0.000000 | 0.000000 |
| ΔE_MEP, `a = 2.00` | 0.000000 | 0.000000 | 0.000000 | **0.000000** | 0.000000 | 0.000000 | 0.000000 |
| ‖grad‖ on interior, `a = 1.50` | 5.50 | 4.21 | 2.54 | **1.55** | 1.24 | 1.16 | 1.13 |
| ‖grad‖ on interior, `a = 2.00` | 5.42 | 4.11 | 2.64 | **1.86** | 1.55 | 1.47 | 1.45 |

**The barrier is 0.000000 at every iteration count from 50 to 3,200.** The
reported value does not depend on the fixed choice of 400, which is what the
result needs.

**But the path is not converged in the strict sense**: the interior gradient norm
plateaus at ≈1.13 (`a = 1.50`) and ≈1.45 (`a = 2.00`) rather than approaching
zero. The images are sliding along a descent direction that the arclength
re-spacing keeps re-populating; the plateau is the residual tangential component
the simplified method does not project out.

## Appendix wording — what to state, and what not to

**State all of the following:**

1. **Barrier definition.** `ΔE = max_path E − max(E_start, E_end)` — measured
   relative to the **higher endpoint**, so `ΔE ≥ 0` and `ΔE = 0` means the path
   is monotonically non-increasing from the higher endpoint. The convention is
   symmetric under reversing the path.
2. **Discretisation.** **41 images**, endpoints pinned. (The linear estimator
   uses 101 points on the straight chord.)
3. **Iterations.** A **fixed 400**, with **no convergence tolerance**.
4. **Independence from that choice.** The barrier is **0.000000 at every
   iteration count from 50 to 3,200**, so the reported value does not depend on
   the fixed 400.
5. **The path is not a converged MEP.** Interior gradient norms **plateau near
   1.1 to 1.5** rather than approaching zero; the simplified string method does
   not project out the tangential component, and the arclength re-spacing keeps
   re-populating the images along a descent direction.

**Claim only this:** *a path exists, at this discretisation, along which the
training loss never exceeds the higher endpoint.* Equivalently, the measurement
**rules out a positive static barrier along that path**.

**Do not claim** that the path is the minimum-energy path, that a saddle search
converged, or that no barrier exists anywhere between the endpoints — none of
those follows from a non-converged string at 41 images.

**Do not use the word "Arrhenius" in the paper.** State the result directly: no
positive static barrier along the exhibited path, so a barrier-crossing account
of findability has nothing to rest on here. The rate-theory framing adds no
information and invites argument about whether that premise was ever the right
one.

## Parameter summary for the appendix table

| parameter | value |
|---|---|
| energy | full-batch BCE-with-logits, 400 points, `DATA_SEED = 0` |
| barrier | `max_path E − max(E_start, E_end)` (Goodfellow) |
| start | `U(−1,1)^4`, RNG stream `100_000 + seed` |
| end | `constructed` (`\|w₂\| = 1`, analytic) or `found` (Adam lr 1e-2, 2,000 steps, must pass `solves()`) |
| linear path points | 101 |
| string images | 41, endpoints pinned |
| string step size | 5e-3, plain gradient descent on the summed interior energy |
| string iterations | **400, fixed — no tolerance** |
| re-spacing | cumulative arclength, linear interpolation per coordinate, every iteration |
| precision | float64 |
| coverage | 13 values of `a`, 20 initialisations per cell, MEP on 5 per cell |
| convergence evidence | ΔE = 0.000000 invariant over 50–3,200 iterations; interior ‖grad‖ plateaus at 1.1–1.5, **not** zero |
