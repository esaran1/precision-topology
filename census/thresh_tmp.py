import pandas as pd, torch, numpy as np
from pathlib import Path
from src.data import linked_core_circles
from src.linking_sweep import load_sweep, select, reconstruct
from src.projection_sweep import propagate, pca_triple_distribution, random_projection_distribution
from src.width_sweep import WidthSweepConfig

d = pd.read_csv('results/projection_distribution.csv')
res = d[(d.reportable_native) & (d.final_layer) &
        (d.pca_any_nonzero_reportable | d.rand_any_nonzero_reportable)]
print('cells with a surviving nonzero at 0.02:', len(res), flush=True)

s = load_sweep(Path('results'))
a, b = linked_core_circles(512)
A = torch.tensor(a, dtype=torch.float64); B = torch.tensor(b, dtype=torch.float64)
rows = []
for _, r in res.iterrows():
    sel = select(s, width=int(r.width), activation=str(r.activation))
    sel = sel[(sel.depth == int(r.depth)) & (sel.seed == int(r.seed))]
    if not len(sel): continue
    m = reconstruct(sel.iloc[0], WidthSweepConfig()).model
    L, R = propagate(m, A, B, int(r.layer))
    for fam, dist in (('pca', pca_triple_distribution(L, R, n_components=min(6, int(r.width)))),
                      ('rand', random_projection_distribution(L, R, n_projections=48, seed_base=int(r.seed)*97+int(r.layer)))):
        for val, pdist in zip(dist.values, dist.projected_distances):
            rows.append(dict(activation=r.activation, width=int(r.width), depth=int(r.depth),
                             seed=int(r.seed), layer=int(r.layer), family=fam,
                             value=val, proj_dist=pdist))
f = pd.DataFrame(rows)
f.to_csv('results/projection_triples.csv', index=False)
print('triple rows:', len(f), flush=True)
