"""Step 3: Experiment 1 - canonical vs fitted BKT, 5-fold learner-level CV.
Resumable: results per skill are cached in results/e1_cache/. Runtime ~10 min."""
import numpy as np, pandas as pd, pickle, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from bkt import CANONICAL, to_matrix, predict, em_fit
MAXLEN = 100; NS = [10, 25, 50, 100, 200]
os.makedirs("results/e1_cache", exist_ok=True)
df = pd.read_csv("data/as_clean.csv")
users = np.sort(df.user_id.unique()); rng = np.random.default_rng(42); rng.shuffle(users)
fold_of = {u: i % 5 for i, u in enumerate(users)}
for si, sk in enumerate(sorted(df.skill_name.unique())):
    fn = f"results/e1_cache/{si:02d}.pkl"
    if os.path.exists(fn): continue
    d = df[df.skill_name == sk]
    seqs = {u: g.correct.values[:MAXLEN] for u, g in d.groupby("user_id")}
    res = []
    for f in range(5):
        tr = [seqs[u] for u in seqs if fold_of[u] != f]; te = [seqs[u] for u in seqs if fold_of[u] == f]
        Xtr, Mtr = to_matrix(tr); Xte, Mte = to_matrix(te)
        models = {"Canonical": CANONICAL}
        models["Fitted-all"], _ = em_fit(Xtr, Mtr, n_restarts=3, n_iter=60, seed=f)
        idx = np.random.default_rng(f).permutation(len(tr))
        for n in NS:
            if n < len(tr):
                Xs, Ms = to_matrix([tr[i] for i in idx[:n]])
                models[f"Fitted-{n}"], _ = em_fit(Xs, Ms, n_restarts=3, n_iter=60, seed=f)
        out = dict(fold=f, y=Xte[Mte], mean=Xtr[Mtr].mean(), params=models, pred={}, L={})
        for name, p in models.items():
            P, L = predict(Xte, Mte, p, return_L=True); out["pred"][name] = P[Mte]
            if name in ("Canonical", "Fitted-all"): out["L"][name] = L[Mte]
        res.append(out)
    pickle.dump(dict(skill=sk, n_students=len(seqs), folds=res), open(fn, "wb"))
    print(si, sk, flush=True)
print("done")
