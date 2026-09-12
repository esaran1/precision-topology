# quotation/theorem verification, re-runnable
import re, os, sys
os.chdir(os.path.join(os.path.dirname(__file__) if False else '.', 'census/paper/sources'))
def norm(s):
    s=s.replace('­','').replace('ﬁ','fi').replace('ﬂ','fl').replace('ﬀ','ff')
    s=s.replace('’',"'").replace('“','"').replace('”','"').replace('−','-').replace('–','-')
    return re.sub(r'\s+','',s)
src={f[:-4]: norm(open(f).read()) for f in os.listdir('.') if f.endswith('.txt')}
Q=[("ren_lim","perfect classification is impossible"),("ren_lim","under our training protocol"),
   ("ren_lim","the ambient-dimension d= 2 instance of our linking framework"),
   ("ren_lim","GELU mean accuracy stays at 89-91% for depths 3-12"),
   ("hanin_sellke_width","din + 1≤wmin(din,d out)≤din +dout"),
   ("cohen_edge_of_stability","the sharpness does not flatline at any value during SGD"),
   ("ahn_zhang_sra_uphill","λmin(∇2f (p)) < 0 or λmax(∇2f (p)) > 2 η"),
   ("soudry_implicit_bias","very slow, and only logarithmic in the convergence of the los")]
bad=[q for f,q in Q if norm(q) not in src[f]]
print(f"{len(Q)-len(bad)}/{len(Q)} quotations verified")
sys.exit(1 if bad else 0)
