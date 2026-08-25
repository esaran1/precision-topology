import pandas as pd, torch, numpy as np
from pathlib import Path
from src.data import linked_core_circles
from src.linking_sweep import load_sweep, select, reconstruct
from src.projection_sweep import (propagate, distortion,
    random_projection_distribution, pca_triple_distribution, reportable)
from src.width_sweep import WidthSweepConfig
s=load_sweep(Path('results')); rows=select(s,seeds=range(3)); rows=rows[rows.width>3]
a,b=linked_core_circles(512)
A=torch.tensor(a,dtype=torch.float64); B=torch.tensor(b,dtype=torch.float64)
out=[]
for i,(_,r) in enumerate(rows.iterrows(),1):
    m=reconstruct(r,WidthSweepConfig()).model
    for layer in range(1,int(r.depth)+1):
        L,R=propagate(m,A,B,layer)
        rp=random_projection_distribution(L,R,n_projections=48,seed_base=abs(hash((int(r.width),int(r.depth),int(r.seed),layer)))%100000)
        pc=pca_triple_distribution(L,R,n_components=min(6,int(r.width)))
        def summarize(d,pre):
            rv=np.array(d.reportable_values,dtype=int); pd_=np.array(d.projected_distances)
            return {pre+'_n':d.n_projections, pre+'_n_reportable':len(rv),
                pre+'_frac_zero_all':d.fraction_zero,
                pre+'_frac_zero_reportable':float((rv==0).mean()) if len(rv) else float('nan'),
                pre+'_any_nonzero_reportable':bool(len(rv) and (rv!=0).any()),
                pre+'_distinct_all':str(d.distinct_values),
                pre+'_distinct_reportable':str(tuple(sorted(set(rv.tolist())))),
                pre+'_min_proj_dist':float(pd_.min()) if len(pd_) else float('nan'),
                pre+'_median_proj_dist':float(np.median(pd_)) if len(pd_) else float('nan')}
        row=dict(activation=r.activation,width=int(r.width),depth=int(r.depth),seed=int(r.seed),
            layer=layer,final_layer=(layer==int(r.depth)),distortion=distortion(m,layer),
            native_min_distance=rp.min_distance,reportable_native=reportable(rp))
        row.update(summarize(rp,'rand')); row.update(summarize(pc,'pca'))
        out.append(row)
    if i%25==0:
        print(f'{i}/{len(rows)}',flush=True); pd.DataFrame(out).to_csv('results/projection_distribution.csv',index=False)
f=pd.DataFrame(out); f.to_csv('results/projection_distribution.csv',index=False); f.to_parquet('results/projection_distribution.parquet',index=False)
print('rows:',len(f),flush=True)
