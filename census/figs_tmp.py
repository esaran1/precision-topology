import pandas as pd, torch
from pathlib import Path
from src.data import linked_core_circles
from src.linking_sweep import load_sweep, select, reconstruct
from src.projection_sweep import propagate
from src.projection_figures import figure_for_projections, pca_bases, random_bases
from src.width_sweep import WidthSweepConfig

out = Path('results/figures/projections'); out.mkdir(parents=True, exist_ok=True)
s = load_sweep(Path('results'))
a, b = linked_core_circles(512)
A = torch.tensor(a, dtype=torch.float64); B = torch.tensor(b, dtype=torch.float64)

# 1. the self-inconsistent cell, named in the writeup
sel = select(s, width=15, activation='leaky_relu')
sel = sel[(sel.depth == 3) & (sel.seed == 0)]
m = reconstruct(sel.iloc[0], WidthSweepConfig()).model
L, R = propagate(m, A, B, 3)
bases = pca_bases(L, R, [(0,1,2),(1,2,3),(1,3,4)]) + random_bases(15, [1,2,3])
figure_for_projections(L, R, bases,
  'leaky_relu w15 d3 s0 final layer: the SAME representation gives link 0, -1, +1',
  out / 'self_inconsistent_cell.png')
print('wrote self_inconsistent_cell.png', flush=True)

# 2. a representative set across activations at width 8
for act in ['tanh','relu','leaky_relu','gelu']:
    sel = select(s, width=8, activation=act)
    sel = sel[(sel.depth == 5) & (sel.seed == 0)]
    if not len(sel): continue
    m = reconstruct(sel.iloc[0], WidthSweepConfig()).model
    L, R = propagate(m, A, B, 5)
    bases = pca_bases(L, R, [(0,1,2),(0,1,3),(0,2,3)]) + random_bases(8, [1,2,3])
    figure_for_projections(L, R, bases, f'{act} w8 d5 s0, final layer', out / f'width8_{act}.png')
    print('wrote', act, flush=True)

# 3. the input configuration, as a reference of what a clean Hopf link looks like
figure_for_projections(A, B, [('R^3 (no projection)', None)] + random_bases(3, []),
  'input configuration: unprojected Hopf link, link = -1', out / 'input_reference.png')
print('done', flush=True)
