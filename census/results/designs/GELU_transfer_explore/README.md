# Exploratory feasibility scripts for the GELU transfer design (NOT registered)

Producers of the numbers labelled "exploratory" in `../GELU_transfer_design.md`. They print their results (no
artifacts); they used only the population, or the landscape of already-used Track 3A seeds 850,000–850,011, and trained
no candidate registered seed. Every number they produced will be recomputed and frozen by the registered pipeline if
the design is approved. Run each with `nice -n 15 .venv/bin/python <script>` from `census/`.

- `gelu_explore_a.py`: population branch continued from s_pop down to s = 0.3 and up to 2·s_pop.
- `gelu_explore_b.py`: free SGD after release (population): κ_SGD, χ, closed-form obs/pred vs η and output-rate ρ.
- `gelu_explore_c.py`: Adam after release / free Adam (population): ηλ_min(P_tsw), lag in steps.
- `gelu_explore_d.py`: own-sample landscape (seeds 850,000–850,011): both copies' switches, own κ_SGD, vs act_fold.
- `gelu_explore_e.py`: random hidden start at fixed s₀ (population): branch reached (54/60 target or mirror).
- `gelu_explore_f.py` (added after approval, before registration): placed fraction of random (w₁, b₁) at w₂ = +s₀; G during the hold for 150 random hidden starts; population t_sw at η = 0.03.
