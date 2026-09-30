"""Step 5: Experiment 2 - simulation of 8 sequencing policies on 3,000 learners (Table II)."""
import numpy as np, pandas as pd, pickle, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import sim
from bkt import CANONICAL
from scipy.stats import wilcoxon
EMP=dict(CANONICAL,learn=0.11)
POL={'Random':lambda r:sim.Random(r),'Fixed curriculum':lambda r:sim.Fixed(),'Counter rules':lambda r:sim.Counter(),
'Flat BKT+rules':lambda r:sim.BKTRules(CANONICAL),
'KT-IDEM-style BKT+rules':lambda r:sim.BKTDiff(CANONICAL,gate=0),'Level-BKT+rules':lambda r:sim.LevelBKT(CANONICAL),
'Level-BKT+rules (cal.)':lambda r:sim.LevelBKT(EMP),'Oracle':None}
N=1000; out={}; stats={}
os.makedirs('results/exp2_cache',exist_ok=True)  # resumable: each (population, policy) is cached
for pop,mu in [('weak',-2),('typical',-1),('strong',0)]:
    th0,eta=sim.make_learners(N,np.random.default_rng(1000+int(mu*10)),mu0=mu)
    for name,f in POL.items():
        cf=f"results/exp2_cache/{pop}_{name.replace(' ','_').replace('(','').replace(')','').replace('.','')}.pkl"
        if os.path.exists(cf): out[(pop,name)]=pd.read_pickle(cf); continue
        out[(pop,name)]=pd.DataFrame(sim.run(f,th0,eta,np.random.default_rng(7)))
        out[(pop,name)].to_pickle(cf)
    print(pop,'done',flush=True)
pickle.dump(out,open('results/exp2.pkl','wb'))
rows=[]
for (pop,name),R in out.items():
    prec=R.correct_decl.sum()/max(R.n_decl.sum(),1) if R.n_decl.sum()>0 else np.nan
    rows.append(dict(pop=pop,policy=name,mastery=R.mastery.mean(),mastery_sd=R.mastery.std(),challenge=R.challenge.mean(),
        waste=R.waste.mean(),gain=R.gain.mean(),prec=prec,t_all=R.t_all.median(),all_rate=R.t_all.notna().mean()))
T=pd.DataFrame(rows); T.to_csv('results/exp2_table.csv',index=False); print(T.round(3).to_string())
def comp(pop,a,b,m='mastery'):
    x=out[(pop,a)][m].values; y=out[(pop,b)][m].values; d=x-y
    bs=[np.random.default_rng(i).choice(d,len(d)).mean() for i in range(2000)]
    nz=d[d!=0]; rb=(np.sum(nz>0)-np.sum(nz<0))/len(nz) if len(nz) else 0
    return dict(pop=pop,a=a,b=b,metric=m,diff=d.mean(),ci=np.percentile(bs,[2.5,97.5]).tolist(),p=wilcoxon(x,y).pvalue if len(nz) else 1.0,sign_r=rb)
C=[]
for pop in ['weak','typical','strong']:
    for a,b in [('Level-BKT+rules','Flat BKT+rules'),('Level-BKT+rules','KT-IDEM-style BKT+rules'),('KT-IDEM-style BKT+rules','Flat BKT+rules'),('Level-BKT+rules','Fixed curriculum'),('Level-BKT+rules (cal.)','Level-BKT+rules'),('Level-BKT+rules','Counter rules')]:
        for m in ['mastery','challenge']:
            C.append(comp(pop,a,b,m))
C=pd.DataFrame(C); C.to_csv('results/exp2_tests.csv',index=False); print(C.round(4).to_string())
