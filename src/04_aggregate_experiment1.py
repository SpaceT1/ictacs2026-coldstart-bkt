"""Step 4: aggregate Experiment 1 (Table II of the paper, Fig. 1, supplement tables).

* pooled AUC / RMSE over all test predictions with 95% CIs from a learner-level cluster
  bootstrap (1,000 resamples of learners);
* per-skill AUC / RMSE and paired Wilcoxon tests with bootstrap CIs of the mean difference;
* pilot calibration: three independent pilot draws per skill and fold, three variants
  (all four parameters / prior fixed at .10 / only the prior fitted);
* agreement of difficulty zones and mastery states (canonical vs fitted parameters).
"""
import numpy as np, pandas as pd, pickle, glob, json
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from scipy.stats import wilcoxon, spearmanr

NS = [10, 25, 50, 100, 200]; REPS = 3; B = 1000
files = sorted(glob.glob('results/e1v2_cache/*.pkl'))
assert len(files) == 48, len(files)

ys, uids, P = [], [], {}
rows, prm, zc, zf, mc, mf = [], [], [], [], [], []
for fn in files:
    D = pickle.load(open(fn, 'rb'))
    for F in D['folds']:
        ys.append(F['y']); uids.append(F['uid'])
        for k, v in F['pred'].items(): P.setdefault(k, []).append(v)
        prm.append(dict(skill=D['skill'], fold=F['fold'], **F['params']['Fitted-all']))
        zc.append(np.digitize(F['L']['Canonical'], [.4, .7])); zf.append(np.digitize(F['L']['Fitted-all'], [.4, .7]))
        mc.append(F['L']['Canonical'] >= .95); mf.append(F['L']['Fitted-all'] >= .95)
    y = np.concatenate([F['y'] for F in D['folds']]).astype(float)
    nseq = D['n_students']
    r = dict(skill=D['skill'], students=nseq, resp=len(y), resp_per_learner=len(y) / nseq, acc=y.mean())
    for k in D['folds'][0]['pred']:
        if all(k in F['pred'] for F in D['folds']):
            p = np.concatenate([F['pred'][k] for F in D['folds']]).astype(float)
            r[k + '_auc'] = roc_auc_score(y, p) if k != 'Constant' else 0.5
            r[k + '_rmse'] = np.sqrt(((y - p) ** 2).mean())
    rows.append(r)

y = np.concatenate(ys).astype(float); uid = np.concatenate(uids)
models = [k for k, v in P.items() if sum(len(a) for a in v) == len(y)]
pred = {k: np.concatenate(P[k]).astype(float) for k in models}

# ---------------- learner-level cluster bootstrap for pooled AUC / RMSE
_, ui = np.unique(uid, return_inverse=True); nU = ui.max() + 1
rng = np.random.default_rng(0)
W = np.stack([np.bincount(rng.integers(0, nU, nU), minlength=nU) for _ in range(B)])   # (B, nU)


def auc_fast(p, w):
    """Weighted AUC with ties, for many weight vectors at once. w: (B, n)."""
    vals, inv = np.unique(p, return_inverse=True)
    pos = np.stack([np.bincount(inv, weights=wi * y, minlength=len(vals)) for wi in w])
    neg = np.stack([np.bincount(inv, weights=wi * (1 - y), minlength=len(vals)) for wi in w])
    cneg = np.cumsum(neg, 1) - neg
    return ((pos * (cneg + 0.5 * neg)).sum(1)) / (pos.sum(1) * neg.sum(1))


S = {}
cnt_u = np.bincount(ui, minlength=nU).astype(float)
for k in models:
    p = pred[k]
    auc = 0.5 if k == 'Constant' else roc_auc_score(y, p)
    rmse = float(np.sqrt(((y - p) ** 2).mean()))
    sse_u = np.bincount(ui, weights=(y - p) ** 2, minlength=nU)
    rm_b = np.sqrt((W @ sse_u) / (W @ cnt_u))
    entry = dict(auc=auc, rmse=rmse, rmse_ci=np.percentile(rm_b, [2.5, 97.5]).tolist())
    if k in ('Canonical', 'Fitted-all', 'PFA', 'Learner mean', 'Full-noprior', 'Full-prioronly', 'P50-all-r0'):
        a_b = np.concatenate([auc_fast(p, W[i:i + 50][:, ui].astype(float)) for i in range(0, B, 50)])
        entry['auc_ci'] = np.percentile(a_b, [2.5, 97.5]).tolist()
        entry['_auc_b'] = a_b
    S[k] = entry
# CI for the canonical-vs-fitted AUC difference (paired bootstrap)
d_b = S['Fitted-all']['_auc_b'] - S['Canonical']['_auc_b']
S['diff_fit_minus_canon_auc'] = dict(est=S['Fitted-all']['auc'] - S['Canonical']['auc'],
                                     ci=np.percentile(d_b, [2.5, 97.5]).tolist())
for k in list(S):
    if isinstance(S[k], dict): S[k].pop('_auc_b', None)

# ---------------- pilot calibration: mean and range over the three independent draws
gap = S['Fitted-all']['auc'] - S['Canonical']['auc']
pilot = {}
for v in ('all', 'noprior', 'prioronly'):
    for n in NS:
        aucs = [S[f'P{n}-{v}-r{r}']['auc'] for r in range(REPS) if f'P{n}-{v}-r{r}' in S]
        rmses = [S[f'P{n}-{v}-r{r}']['rmse'] for r in range(REPS) if f'P{n}-{v}-r{r}' in S]
        pilot[f'{v}-{n}'] = dict(auc_mean=float(np.mean(aucs)), auc_min=float(min(aucs)), auc_max=float(max(aucs)),
                                 rmse_mean=float(np.mean(rmses)),
                                 recovered=float((np.mean(aucs) - S['Canonical']['auc']) / gap),
                                 recovered_min=float((min(aucs) - S['Canonical']['auc']) / gap),
                                 recovered_max=float((max(aucs) - S['Canonical']['auc']) / gap))
S['pilot'] = pilot
S['full_variants'] = {k: dict(auc=S[k]['auc'], rmse=S[k]['rmse'],
                              recovered=(S[k]['auc'] - S['Canonical']['auc']) / gap)
                      for k in ('Full-noprior', 'Full-prioronly')}

# ---------------- per-skill comparisons
R = pd.DataFrame(rows)
for n in NS:
    cols = [f'P{n}-all-r{r}_auc' for r in range(REPS) if f'P{n}-all-r{r}_auc' in R]
    R[f'P{n}-all_auc_mean'] = R[cols].mean(axis=1)
R.to_csv('results/exp1_perskill.csv', index=False)


def paired(a, b):
    d = (R[a] - R[b]).values
    bs = [np.random.default_rng(i).choice(d, len(d)).mean() for i in range(2000)]
    return dict(mean_diff=float(d.mean()), ci=np.percentile(bs, [2.5, 97.5]).tolist(),
                n_pos=int((d > 0).sum()), p_wilcoxon=float(wilcoxon(R[a], R[b]).pvalue))


S['perskill'] = dict(
    fit_vs_canon_auc=paired('Fitted-all_auc', 'Canonical_auc'),
    fit_vs_canon_rmse=paired('Canonical_rmse', 'Fitted-all_rmse'),
    canon_rmse_worse_than_constant=int((R.Canonical_rmse > R.Constant_rmse).sum()),
    fit_rmse_better_than_constant=int((R['Fitted-all_rmse'] < R.Constant_rmse).sum()),
    pfa_vs_fit_auc=paired('PFA_auc', 'Fitted-all_auc'),
    lmean_vs_fit_auc=paired('Learner mean_auc', 'Fitted-all_auc'),
    pfa_vs_canon_auc=paired('PFA_auc', 'Canonical_auc'),
    n_skills=len(R), students=int(R.students.sum()))
g = (R['Fitted-all_auc'] - R['Canonical_auc'])
rec50 = ((R['P50-all_auc_mean'] - R['Canonical_auc']) / g)[g > 0.02]
S['perskill']['recovered50'] = dict(n_skills=int(len(rec50)), median=float(rec50.median()),
                                    iqr=rec50.quantile([.25, .75]).tolist(),
                                    share_below_half=float((rec50 < 0.5).mean()),
                                    spearman_resp_per_learner=float(spearmanr(rec50, R.loc[g > 0.02, 'resp_per_learner'])[0]),
                                    spearman_students=float(spearmanr(rec50, R.loc[g > 0.02, 'students'])[0]))

# ---------------- zone and mastery agreement
Pm = pd.DataFrame(prm); Pm.to_csv('results/exp1_params.csv', index=False)
zc = np.concatenate(zc); zf = np.concatenate(zf); mc = np.concatenate(mc); mf = np.concatenate(mf)
S['zone'] = dict(agree=float((zc == zf).mean()), kappa_lin=float(cohen_kappa_score(zc, zf, weights='linear')),
                 two_levels_apart=float((abs(zc - zf) == 2).mean()),
                 share_c=np.bincount(zc, minlength=3).tolist(), share_f=np.bincount(zf, minlength=3).tolist(),
                 mastery_agree=float((mc == mf).mean()))
S['params'] = Pm[['prior', 'learn', 'guess', 'slip']].describe().round(3).to_dict()
S['learner_mean_k'] = None
S['n_resp'] = int(len(y))
json.dump(S, open('results/exp1_summary.json', 'w'), indent=1, default=float)
show = {k: S[k] for k in ('Constant', 'Learner mean', 'PFA', 'Canonical', 'Fitted-all', 'Full-noprior',
                          'Full-prioronly', 'diff_fit_minus_canon_auc', 'full_variants', 'perskill', 'zone')}
print(json.dumps(show, indent=1, default=float))
print(json.dumps(S['pilot'], indent=1, default=float))
