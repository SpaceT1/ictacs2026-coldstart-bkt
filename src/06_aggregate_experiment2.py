"""Step 6: aggregate Experiment 2 (Table III of the paper and supplement tables).

* per policy and population: share of truly mastered skills, challenge share, share of
  learners mastering all four skills, declaration precision - each with a 95% learner-level
  bootstrap CI (2,000 resamples);
* primary paired comparisons: mean difference with bootstrap CI, Wilcoxon signed-rank test
  and matched-pairs rank-biserial correlation (continuous outcomes), McNemar test (all four
  skills mastered); Holm correction over the whole family of primary tests;
* policy-parameter sensitivity, simulator sensitivity, tuning results.
"""
import numpy as np, pandas as pd, json, os, sys
from scipy.stats import wilcoxon, rankdata
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
J = json.load(open('results/exp2_jobs.json'))
CACHE = 'results/exp2_cache'
import importlib; e2 = importlib.import_module('05_experiment2')


def load(jobs):
    return {(j[0], j[1], j[3], j[4]): pd.read_pickle(f"{CACHE}/{e2.key(tuple(j))}.pkl")[1] for j in jobs}


def tuplify(j):
    # json turned tuples into lists; rebuild the exact job tuple used for the cache key
    st, cond, kw, pop, name, spec = j
    kind, skw = spec
    skw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in skw.items()}
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in kw.items()}
    return (st, cond, kw, pop, name, (kind, skw))


M = load([tuplify(j) for j in J['main']]); PV = load([tuplify(j) for j in J['policy']])
SV = load([tuplify(j) for j in J['simulator']])
POPS = ['weak', 'typical', 'strong']
rng = np.random.default_rng(0); NB = 2000


def boot_idx(n): return [rng.integers(0, n, n) for _ in range(NB)]


IDX = {}


def summarize(R, n_key):
    if n_key not in IDX: IDX[n_key] = boot_idx(len(R))
    ix = IDX[n_key]
    m = R.mastery.values; a4 = R.t_all.notna().values.astype(float)
    nd, cd = R.n_decl.values, R.correct_decl.values
    prec = cd.sum() / nd.sum() if nd.sum() > 0 else np.nan
    pb = [cd[i].sum() / nd[i].sum() for i in ix] if nd.sum() > 0 else [np.nan]
    return dict(mastery=m.mean(), mastery_ci=np.percentile([m[i].mean() for i in ix], [2.5, 97.5]).tolist(),
                challenge=R.challenge.mean(), too_hard=R.too_hard.mean(), too_easy=R.too_easy.mean(),
                all4=a4.mean(), all4_ci=np.percentile([a4[i].mean() for i in ix], [2.5, 97.5]).tolist(),
                precision=prec, precision_ci=np.nanpercentile(pb, [2.5, 97.5]).tolist(),
                n_decl=int(nd.sum()), correct_decl=int(cd.sum()))


rows = []
for (st, cond, pop, name), R in M.items():
    rows.append(dict(pop=pop, policy=name, **summarize(R, ('main', pop))))
T = pd.DataFrame(rows)
# pooled precision over populations
pp = T.groupby('policy')[['n_decl', 'correct_decl']].sum()
T['precision_pooled'] = T.policy.map((pp.correct_decl / pp.n_decl).where(pp.n_decl > 0))
T.to_csv('results/exp2_table.csv', index=False)


def rb(d):
    """Matched-pairs rank-biserial correlation (Kerby, 2014)."""
    d = d[d != 0]
    if len(d) == 0: return 0.0
    r = rankdata(np.abs(d)); return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def comp(pop, a, b, metric, src=M, st='main'):
    A = src[(st, 'base', pop, a)]; Bq = src[(st, 'base', pop, b)]
    ix = IDX[('main', pop)]
    if metric == 'all4':
        x = A.t_all.notna().values; z = Bq.t_all.notna().values
        tab = [[np.sum(x & z), np.sum(x & ~z)], [np.sum(~x & z), np.sum(~x & ~z)]]
        d = x.astype(float) - z.astype(float)
        p = mcnemar(tab, exact=False).pvalue; r = np.nan; test = 'McNemar'
    else:
        x = A[metric].values; z = Bq[metric].values; d = x - z
        p = wilcoxon(x, z).pvalue if np.any(d != 0) else 1.0; r = rb(d); test = 'Wilcoxon'
    ci = np.percentile([d[i].mean() for i in ix], [2.5, 97.5]).tolist()
    return dict(pop=pop, a=a, b=b, metric=metric, diff=float(d.mean()), ci_low=ci[0], ci_high=ci[1],
                p=float(p), test=test, rank_biserial=r)


PRIMARY = [('Level-BKT+rules', 'Flat BKT+rules', 'mastery'),
           ('Level-BKT+rules', 'Flat BKT+rules', 'challenge'),
           ('Level-BKT+rules', 'KT-IDEM-style BKT+rules', 'mastery'),
           ('Level-BKT+rules', 'KT-IDEM-style BKT+rules (tuned)', 'mastery'),
           ('KT-IDEM-style BKT+rules', 'Flat BKT+rules', 'mastery'),
           ('Level-BKT+rules', 'Counter rules', 'mastery'),
           ('Level-BKT+rules', 'Fixed curriculum', 'mastery'),
           ('Level-BKT+rules', 'Fixed curriculum', 'all4'),
           ('Level-BKT+rules', 'Fixed curriculum (tuned)', 'mastery'),
           ('Level-BKT+rules', 'Fixed curriculum (tuned)', 'all4'),
           ('Level-BKT+rules (cal.)', 'Level-BKT+rules', 'mastery'),
           ('Level-BKT+rules (cal.)', 'Level-BKT+rules', 'challenge')]
C = pd.DataFrame([comp(pop, a, b, m) for pop in POPS for a, b, m in PRIMARY])
C['p_holm'] = multipletests(C.p, method='holm')[1]
C.to_csv('results/exp2_tests.csv', index=False)

# ---------------- policy-parameter sensitivity (main learners)
prow = []
for (st, cond, pop, name), R in PV.items():
    prow.append(dict(variant=name, pop=pop, **summarize(R, ('main', pop))))
PT = pd.DataFrame(prow)
pp = PT.groupby('variant')[['n_decl', 'correct_decl']].sum()
PT['precision_pooled'] = PT.variant.map(pp.correct_decl / pp.n_decl)
PT.to_csv('results/exp2_policy_sensitivity.csv', index=False)

# ---------------- simulator sensitivity
srow = []
for (st, cond, pop, name), R in SV.items():
    s = summarize(R, ('sim', pop)); s.pop('mastery_ci'); s.pop('all4_ci'); s.pop('precision_ci')
    srow.append(dict(cond=cond, pop=pop, policy=name, **s))
ST = pd.DataFrame(srow); ST.to_csv('results/exp2_simulator_sensitivity.csv', index=False)
piv = ST.pivot_table(index=['cond', 'pop'], columns='policy', values='mastery')
L, Fl, KT, FX, FH = ('Level-BKT+rules', 'Flat BKT+rules', 'KT-IDEM-style BKT+rules (tuned)',
                     'Fixed curriculum (tuned)', 'Fixed curriculum')
a4 = ST.pivot_table(index=['cond', 'pop'], columns='policy', values='all4')
pr = ST.groupby(['cond', 'policy'])[['n_decl', 'correct_decl']].sum()
pr = (pr.correct_decl / pr.n_decl).unstack()
robust = dict(
    n_cells=int(len(piv)),
    level_gt_flat=int((piv[L] > piv[Fl]).sum()), level_gt_ktidem_tuned=int((piv[L] > piv[KT]).sum()),
    level_minus_flat_range=[float((piv[L] - piv[Fl]).min()), float((piv[L] - piv[Fl]).max())],
    level_minus_kt_range=[float((piv[L] - piv[KT]).min()), float((piv[L] - piv[KT]).max())],
    level_vs_fixed_tuned_mastery_range=[float((piv[L] - piv[FX]).min()), float((piv[L] - piv[FX]).max())],
    level_gt_fixed_tuned_mastery=int((piv[L] > piv[FX]).sum()),
    level_gt_fixed_tuned_all4=int((a4[L] > a4[FX]).sum()),
    precision_level_range=[float(pr[L].min()), float(pr[L].max())],
    precision_flat_max=float(pr[Fl].max()), precision_kt_max=float(pr[KT].max()))
by_cond = {c: dict(level_minus_flat=float((piv.loc[c][L] - piv.loc[c][Fl]).mean()),
                   level_minus_kt=float((piv.loc[c][L] - piv.loc[c][KT]).mean()),
                   level_minus_fixed_tuned=float((piv.loc[c][L] - piv.loc[c][FX]).mean()),
                   all4_level_minus_fixed_tuned=float((a4.loc[c][L] - a4.loc[c][FX]).mean()),
                   prec_level=float(pr.loc[c][L]), prec_flat=float(pr.loc[c][Fl]), prec_kt=float(pr.loc[c][KT]))
           for c in ST.cond.unique()}
json.dump(dict(robust=robust, by_cond=by_cond), open('results/exp2_sensitivity_summary.json', 'w'), indent=1)

pd.set_option('display.width', 250)
show = T.copy(); show[['mastery', 'challenge', 'all4', 'precision_pooled']] *= 100
print(show.pivot_table(index='policy', columns='pop', values=['mastery', 'challenge', 'all4']).round(1))
print(show.groupby('policy').precision_pooled.first().round(1))
print(C.round(4).to_string())
print(json.dumps(robust, indent=1)); print(pd.DataFrame(by_cond).T.round(3))
x = PT.copy(); x[['mastery', 'all4', 'precision_pooled']] *= 100
print(x.pivot_table(index='variant', columns='pop', values=['mastery', 'all4']).round(1))
print(x.groupby('variant').precision_pooled.first().round(1))
