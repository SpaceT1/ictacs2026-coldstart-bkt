"""Step 7: validation of the BKT implementation (bkt.py).

(a) Parameter recovery on synthetic BKT learners across 36 parameter regimes
    (prior x learn x guess x slip) and three data sizes, using the same EM settings as
    Experiment 1 (3 restarts, at most 60 iterations).
(b) Independent check on the 48 real skills (all learners): the EM solution is compared with
    (i) an independently written forward-algorithm log-likelihood maximised by L-BFGS-B from
    several starts, and (ii) the reference library pyBKT on every fourth skill (12 skills), if it is installed
    (pyBKT 1.4 needs a one-line NumPy-2 fix in pyBKT/fit/EM_fit.py, see README).
Outputs: results/validation_recovery.csv, results/validation_real.csv, results/validation.json
"""
import numpy as np, pandas as pd, os, sys, json, itertools, time
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bkt import to_matrix, em_fit, simulate, loglik

NAMES = ('prior', 'learn', 'guess', 'slip')


def recovery():
    rows = []
    regimes = list(itertools.product((0.1, 0.4, 0.7), (0.05, 0.15, 0.30), (0.10, 0.25), (0.05, 0.15)))
    for (n, T) in ((50, 10), (200, 10), (800, 20)):
        for ri, (pr, le, gu, sl) in enumerate(regimes):
            true = dict(prior=pr, learn=le, guess=gu, slip=sl)
            X, M = to_matrix(simulate(n, T, true, np.random.default_rng(100 + ri)))
            fit, _ = em_fit(X, M, n_restarts=3, n_iter=60, seed=ri)
            rows.append(dict(n=n, T=T, **{f'true_{k}': true[k] for k in NAMES},
                             **{f'fit_{k}': fit[k] for k in NAMES},
                             **{f'err_{k}': abs(fit[k] - true[k]) for k in NAMES}))
    return pd.DataFrame(rows)


def indep_loglik(theta, seqs):
    """Independent implementation: forward recursion on P(L) per sequence, written separately
    from bkt.py (predict-update form). theta = (prior, learn, guess, slip)."""
    pr, le, gu, sl = theta
    ll = 0.0
    T = max(len(s) for s in seqs)
    X = np.full((len(seqs), T), -1)
    for i, s in enumerate(seqs): X[i, :len(s)] = s
    L = np.full(len(seqs), pr)
    for t in range(T):
        x = X[:, t]; on = x >= 0
        pc = L * (1 - sl) + (1 - L) * gu
        ll += np.sum(np.log(np.where(x[on] == 1, pc[on], 1 - pc[on])))
        post = np.where(x == 1, L * (1 - sl) / pc, L * sl / (1 - pc))
        L = np.where(on, post + (1 - post) * le, L)
    return ll


def real_check(df):
    try:
        from pyBKT.models import Model
        have_pybkt = True
    except Exception:
        have_pybkt = False
    rows = []
    bounds = [(1e-4, 0.999)] * 2 + [(1e-4, 0.5)] * 2
    for si, (sk, d) in enumerate(df.groupby('skill_name')):
        seqs = [g.correct.values[:100] for _, g in d.groupby('user_id')]
        X, M = to_matrix(seqs)
        em, _ = em_fit(X, M, n_restarts=3, n_iter=60, seed=0)
        th_em = np.array([em[k] for k in NAMES])
        ll_em_indep = indep_loglik(th_em, seqs)
        ll_em_own = loglik(X, M, em)
        best_ll, best_th = -np.inf, None
        for start in (th_em, np.array([0.1, 0.25, 0.2, 0.1])):
            r = minimize(lambda th: -indep_loglik(th, seqs), start, method='L-BFGS-B', bounds=bounds)
            if -r.fun > best_ll: best_ll, best_th = -r.fun, r.x
        row = dict(skill=sk, n_seq=len(seqs), n_resp=int(M.sum()),
                   ll_em=ll_em_indep, ll_own_minus_indep=ll_em_own - ll_em_indep,
                   ll_lbfgs=best_ll, ll_gain_lbfgs=best_ll - ll_em_indep,
                   **{f'em_{k}': em[k] for k in NAMES},
                   **{f'lbfgs_{k}': v for k, v in zip(NAMES, best_th)})
        if have_pybkt and si % 4 == 0:
            dd = d.copy(); dd['order_id'] = dd['order_id'].astype(int)
            dd = dd.groupby('user_id').head(100)
            m = Model(seed=0, num_fits=3, parallel=False)
            m.fit(data=dd[['order_id', 'user_id', 'skill_name', 'correct']])
            pp = m.params().reset_index()
            get = lambda name: float(pp[pp.param == name].value.iloc[0])
            th_py = np.array([get('prior'), get('learns'), get('guesses'), get('slips')])
            row.update({f'pybkt_{k}': v for k, v in zip(NAMES, th_py)},
                       ll_pybkt=indep_loglik(th_py, seqs))
        rows.append(row)
        print(sk, f"EM {row['ll_em']:.1f}  L-BFGS {row['ll_lbfgs']:.1f}" +
              (f"  pyBKT {row['ll_pybkt']:.1f}" if 'll_pybkt' in row else ''), flush=True)
    return pd.DataFrame(rows), have_pybkt


if __name__ == '__main__':
    t0 = time.time()
    R = recovery(); R.to_csv('results/validation_recovery.csv', index=False)
    summ = {}
    for (n, T), g in R.groupby(['n', 'T']):
        summ[f'n={n},T={T}'] = {k: dict(median=float(g[f'err_{k}'].median()),
                                         p90=float(g[f'err_{k}'].quantile(.9)),
                                         max=float(g[f'err_{k}'].max())) for k in NAMES}
        summ[f'n={n},T={T}']['share_all_within_0.05'] = float((g[[f'err_{k}' for k in NAMES]].max(1) < 0.05).mean())
    print(json.dumps(summ, indent=1), flush=True)
    df = pd.read_csv('data/as_clean.csv')
    V, have = real_check(df); V.to_csv('results/validation_real.csv', index=False)
    summ['real'] = dict(max_abs_ll_own_minus_indep=float(V.ll_own_minus_indep.abs().max()),
                        median_ll_gain_lbfgs=float(V.ll_gain_lbfgs.median()),
                        max_ll_gain_lbfgs=float(V.ll_gain_lbfgs.max()),
                        median_ll_gain_lbfgs_per_resp=float((V.ll_gain_lbfgs / V.n_resp).median()),
                        max_param_diff_lbfgs=float(max((V[f'em_{k}'] - V[f'lbfgs_{k}']).abs().max() for k in NAMES)))
    if have and 'll_pybkt' in V:
        W = V.dropna(subset=['ll_pybkt'])
        summ['real'].update(
            n_skills_pybkt=int(len(W)),
            median_ll_em_minus_pybkt_per_resp=float(((W.ll_em - W.ll_pybkt) / W.n_resp).median()),
            min_ll_em_minus_pybkt_per_resp=float(((W.ll_em - W.ll_pybkt) / W.n_resp).min()),
            share_em_ll_ge_pybkt=float((W.ll_em >= W.ll_pybkt - 1e-6).mean()),
            median_param_diff_pybkt={k: float((W[f'em_{k}'] - W[f'pybkt_{k}']).abs().median()) for k in NAMES})
    json.dump(summ, open('results/validation.json', 'w'), indent=1)
    print(json.dumps(summ['real'], indent=1), f'{time.time() - t0:.0f}s')
