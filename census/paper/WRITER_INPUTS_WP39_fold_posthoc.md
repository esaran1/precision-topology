# WP-39. Branch tracking and folds (Track 1A, POST HOC; discussion phase)

This is a separate file; the main patch is unchanged. Everything here is POST HOC on existing runs. The registered gate
to a prospective fold test failed (the v3 late crossings do not match the fold), so no fold test was registered.

- **Producers:** `src/sb_fold.py` → `results/sb_fold/` (linear-plus-slab, v3), `src/width2_fold.py` →
  `results/width2_fold/` (T2-3), `src/act_fold.py` → `results/act_fold/` (GELU non-early crossers).
  Reports: `results/sb_fold_report.md`, `results/width2_fold_report.md`, `results/act_fold_report.md`.
- **Numbers used below (ledger block `discussion_phase_checks`):**
  - GELU: 114 non-early crossers, 0 with a fold, median crossing/switch 1.0017.
  - Width 2: 79 runs, 0 with a fold, median crossing/switch 2.79.
  - Linear-plus-slab (v3): the linear branch folds at 1.3275 s\*; the late crossers leave its basin at a median
    0.66 s_fold and cross at a median 2.17 s_fold, and no run is ever on a branch.

**Paper sentence (one sentence, labelled post hoc):**
> "Post hoc, locating each run's branch by validated continuation shows that slowly driven training (GELU) crosses at
> its branch's ordinary switch (median ratio 1.0017), whereas fast training (width 2, and the linear-plus-slab task)
> leaves its branch well before the switch or fold and tracks no branch, crossing a median 2.79 (width 2) and 2.17
> (linear-plus-slab, relative to the linear branch's fold) times later."

**Do not say:**
- that training follows its branch to a fold, or that any crossing is a delayed exit at a fold;
- that the fold-delay law (§14) is confirmed by training: in the linear-plus-slab setting ε_F = 5.6, outside its regime;
- that this was a registered or prospective test;
- anything about SiLU or Mish (partial / not run).
