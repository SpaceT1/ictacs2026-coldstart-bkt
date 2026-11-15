"""Step 3: Experiment 1 - canonical vs fitted BKT, 5-fold learner-level cross-validation.

For every skill and fold (learners are assigned to folds once, at random, seed 42):
  * Constant      : training-fold proportion correct for the skill
  * Learner mean  : smoothed running mean of the learner's own earlier answers on the skill,
                    p = (s + k*m) / (n + k); k chosen from a grid on the training fold
  * PFA           : Performance Factors Analysis - logistic regression on the numbers of
                    earlier successes and failures (fitted on the training fold)
  * Canonical     : BKT with prior .10, learn .25, guess .20, slip .10 (no data used)
  * Fitted-all    : BKT, all four parameters fitted by EM on the whole training fold
  * Full-noprior  : BKT, prior fixed at .10, learn/guess/slip fitted on the training fold
  * Full-prioronly: BKT, only the prior fitted; learn/guess/slip canonical
  * P{n}-{v}-r{r} : pilot calibration - the same three variants v (all / noprior / prioronly)
                    fitted on a random subset of n training learners (n = 10..200),
                    three independent pilot draws r = 0,1,2 per skill and fold.
All predictions are online: each response is predicted from the learner's earlier responses only.
Resumable: results per skill are cached in results/e1v2_cache/. Runtime ~1 h on 2 cores.
"""
import numpy as np, pandas as pd, pickle, os, sys, time
from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bkt import CANONICAL, to_matrix, predict, em_fit

MAXLEN = 100; NS = [10, 25, 50, 100, 200]; REPS = 3
VARIANTS = {"all": None, "noprior": {"prior": CANONICAL["prior"]},
            "prioronly": {k: CANONICAL[k] for k in ("learn", "guess", "slip")}}
CACHE = "results/e1v2_cache"


def counts(seqs):
    """Earlier successes / attempts before every step, flattened in the same order as X[M]."""
    s, n = [], []
    for q in seqs:
        c = np.concatenate([[0], np.cumsum(q)[:-1]])
        s.append(c); n.append(np.arange(len(q)))
    return np.concatenate(s), np.concatenate(n)


def flat(seqs):
    """Responses in row-major order of (learner, step) - identical to X[M] of to_matrix."""
    return np.concatenate(seqs).astype(float)


def run_skill(args):
    si, sk, d, fold_of = args
    fn = f"{CACHE}/{si:02d}.pkl"
    if os.path.exists(fn): return si
    t0 = time.time()
    seqs = {u: g.correct.values[:MAXLEN] for u, g in d.groupby("user_id")}
    res = []
    for f in range(5):
        tr_u = [u for u in seqs if fold_of[u] != f]; te_u = [u for u in seqs if fold_of[u] == f]
        tr = [seqs[u] for u in tr_u]; te = [seqs[u] for u in te_u]
        Xtr, Mtr = to_matrix(tr); Xte, Mte = to_matrix(te)
        y = Xte[Mte]
        assert np.array_equal(y, flat(te))
        uid = np.concatenate([np.full(len(seqs[u]), u) for u in te_u])
        out = dict(fold=f, y=y.astype(np.int8), uid=uid, pred={}, L={}, params={})
        # ---- lightweight baselines
        m = Xtr[Mtr].mean()
        out["pred"]["Constant"] = np.full(len(y), m, np.float32)
        s_tr, n_tr = counts(tr); y_tr = flat(tr)
        best = min([0.5, 1, 2, 4, 8, 16],
                   key=lambda k: np.mean((y_tr - (s_tr + k * m) / (n_tr + k)) ** 2))
        s_te, n_te = counts(te)
        out["pred"]["Learner mean"] = ((s_te + best * m) / (n_te + best)).astype(np.float32)
        out["params"]["Learner mean"] = {"k": best}
        lr = LogisticRegression(C=1e4, max_iter=500)
        lr.fit(np.c_[s_tr, n_tr - s_tr], y_tr)
        out["pred"]["PFA"] = lr.predict_proba(np.c_[s_te, n_te - s_te])[:, 1].astype(np.float32)
        # ---- BKT
        models = {"Canonical": CANONICAL}
        models["Fitted-all"], _ = em_fit(Xtr, Mtr, n_restarts=3, n_iter=60, seed=f)
        models["Full-noprior"], _ = em_fit(Xtr, Mtr, n_restarts=3, n_iter=60, seed=f, fixed=VARIANTS["noprior"])
        models["Full-prioronly"], _ = em_fit(Xtr, Mtr, n_restarts=3, n_iter=60, seed=f, fixed=VARIANTS["prioronly"])
        for r in range(REPS):
            idx = np.random.default_rng(f if r == 0 else 1000 * r + f).permutation(len(tr))
            for n in NS:
                if n >= len(tr): continue
                Xs, Ms = to_matrix([tr[i] for i in idx[:n]])
                for v, fx in VARIANTS.items():
                    models[f"P{n}-{v}-r{r}"], _ = em_fit(Xs, Ms, n_restarts=3, n_iter=60, seed=f, fixed=fx)
        for name, p in models.items():
            P, L = predict(Xte, Mte, p, return_L=True)
            out["pred"][name] = P[Mte].astype(np.float32)
            if name in ("Canonical", "Fitted-all"): out["L"][name] = L[Mte].astype(np.float32)
            out["params"][name] = p
        res.append(out)
    pickle.dump(dict(skill=sk, n_students=len(seqs), folds=res), open(fn, "wb"))
    print(si, sk, f"{time.time() - t0:.0f}s", flush=True)
    return si


if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    df = pd.read_csv("data/as_clean.csv")
    users = np.sort(df.user_id.unique()); rng = np.random.default_rng(42); rng.shuffle(users)
    fold_of = {u: i % 5 for i, u in enumerate(users)}
    skills = sorted(df.skill_name.unique())
    only = [int(a) for a in sys.argv[1:]]
    jobs = [(si, sk, df[df.skill_name == sk], fold_of) for si, sk in enumerate(skills)
            if (not only or si in only) and not os.path.exists(f"{CACHE}/{si:02d}.pkl")]
    with Pool(int(os.environ.get("NPROC", os.cpu_count()))) as pool:
        for _ in pool.imap_unordered(run_skill, jobs): pass
    print("done", len(os.listdir(CACHE)), "/", len(skills))
