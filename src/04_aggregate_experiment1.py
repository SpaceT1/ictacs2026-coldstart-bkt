"""Step 4: aggregate Experiment 1 (Table I, zone agreement, Wilcoxon tests)."""
import numpy as np, pandas as pd, pickle, glob, json
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from scipy.stats import wilcoxon
ys=[];P={};rows=[];prm=[];zc=[];zf=[];mc=[];mf=[]
for fn in sorted(glob.glob('results/e1_cache/*.pkl')):
    D=pickle.load(open(fn,'rb'))
    for F in D['folds']:
        y=F['y']; ys.append(y)
        P.setdefault('Mean',[]).append(np.full(len(y),F['mean']))
        for k,v in F['pred'].items(): P.setdefault(k,[]).append(v)
        prm.append(dict(skill=D['skill'],**F['params']['Fitted-all']))
        zc.append(np.digitize(F['L']['Canonical'],[.4,.7])); zf.append(np.digitize(F['L']['Fitted-all'],[.4,.7]))
        mc.append(F['L']['Canonical']>=.95); mf.append(F['L']['Fitted-all']>=.95)
    # per-skill pooled over folds
    y=np.concatenate([F['y'] for F in D['folds']])
    r=dict(skill=D['skill'],students=D['n_students'],resp=len(y))
    for k in ['Canonical','Fitted-all']+[f'Fitted-{n}' for n in [10,25,50,100,200]]:
        if all(k in F['pred'] for F in D['folds']):
            p=np.concatenate([F['pred'][k] for F in D['folds']])
            r[k+'_auc']=roc_auc_score(y,p); r[k+'_rmse']=np.sqrt(((y-p)**2).mean())
    m=np.concatenate([np.full(len(F['y']),F['mean']) for F in D['folds']]); r['Mean_rmse']=np.sqrt(((y-m)**2).mean())
    rows.append(r)
y=np.concatenate(ys); S={}
for k,v in P.items():
    p=np.concatenate(v)
    if len(p)!=len(y): continue
    S[k]=dict(auc=0.5 if k=='Mean' else roc_auc_score(y,p),rmse=float(np.sqrt(((y-p)**2).mean())))
R=pd.DataFrame(rows); R.to_csv('results/exp1_perskill.csv',index=False)
Pm=pd.DataFrame(prm); Pm.to_csv('results/exp1_params.csv',index=False)
zc=np.concatenate(zc); zf=np.concatenate(zf); mc=np.concatenate(mc); mf=np.concatenate(mf)
S['zone']=dict(agree=float((zc==zf).mean()),kappa_lin=float(cohen_kappa_score(zc,zf,weights='linear')),
  adjacent=float((abs(zc-zf)<=1).mean()),share_c=np.bincount(zc,minlength=3).tolist(),share_f=np.bincount(zf,minlength=3).tolist(),
  conf=pd.crosstab(zc,zf).values.tolist(),
  mastery_c=float(mc.mean()),mastery_f=float(mf.mean()),mastery_agree=float((mc==mf).mean()),
  c_mast_not_f=float((mc&~mf).mean()),f_mast_not_c=float((mf&~mc).mean()))
S['wilcoxon_auc']=wilcoxon(R['Fitted-all_auc'],R['Canonical_auc']).pvalue
S['wilcoxon_rmse']=wilcoxon(R['Fitted-all_rmse'],R['Canonical_rmse']).pvalue
S['perskill']=dict(canon_auc_med=R.Canonical_auc.median(),fit_auc_med=R['Fitted-all_auc'].median(),
  canon_auc_iqr=R.Canonical_auc.quantile([.25,.75]).tolist(),fit_auc_iqr=R['Fitted-all_auc'].quantile([.25,.75]).tolist(),
  canon_better_rmse_than_mean=int((R.Canonical_rmse<R.Mean_rmse).sum()),fit_better_rmse_than_mean=int((R['Fitted-all_rmse']<R.Mean_rmse).sum()),
  n_skills=len(R),students=int(R.students.sum()))
S['params']=Pm[['prior','learn','guess','slip']].describe().round(3).to_dict()
S['n_resp']=int(len(y))
json.dump(S,open('results/exp1_summary.json','w'),indent=1,default=float)
print(json.dumps(S,indent=1,default=float))
