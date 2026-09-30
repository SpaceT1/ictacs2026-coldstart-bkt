"""Minimal, vectorised Bayesian Knowledge Tracing (Corbett & Anderson, 1995).
Two latent states (unlearned U / learned L), no forgetting.
Params: p = dict(prior, learn, guess, slip).
"""
import numpy as np

CANONICAL = dict(prior=0.10, learn=0.25, guess=0.20, slip=0.10)


def to_matrix(seqs):
    """List of 0/1 arrays -> (N,T) obs matrix and mask."""
    T = max(len(s) for s in seqs)
    X = np.zeros((len(seqs), T))
    M = np.zeros((len(seqs), T), dtype=bool)
    for i, s in enumerate(seqs):
        X[i, :len(s)] = s
        M[i, :len(s)] = True
    return X, M


def predict(X, M, p, return_L=False):
    """Online predictions P(correct_t | obs_<t) for every step (forward filter)."""
    N, T = X.shape
    L = np.full(N, p['prior'])
    P = np.zeros((N, T)); LL = np.zeros((N, T))
    g, s, t = p['guess'], p['slip'], p['learn']
    for k in range(T):
        pc = L * (1 - s) + (1 - L) * g
        P[:, k] = pc; LL[:, k] = L
        x = X[:, k]
        post = np.where(x == 1, L * (1 - s) / pc, L * s / (1 - pc))
        newL = post + (1 - post) * t
        L = np.where(M[:, k], newL, L)
    return (P, LL) if return_L else P


def em_fit(X, M, n_restarts=5, n_iter=100, tol=1e-5, seed=0, max_gs=0.5):
    rng = np.random.default_rng(seed)
    best, best_ll = None, -np.inf
    for r in range(n_restarts):
        p = dict(prior=rng.uniform(0.05, 0.6), learn=rng.uniform(0.05, 0.4),
                 guess=rng.uniform(0.05, 0.35), slip=rng.uniform(0.05, 0.25))
        prev = -np.inf
        for it in range(n_iter):
            p, ll = _em_step(X, M, p, max_gs)
            if ll - prev < tol:
                break
            prev = ll
        if ll > best_ll:
            best, best_ll = p, ll
    return best, best_ll


def _em_step(X, M, p, max_gs):
    N, T = X.shape
    pi = np.array([1 - p['prior'], p['prior']])            # [U, L]
    A = np.array([[1 - p['learn'], p['learn']], [0.0, 1.0]])
    B1 = np.array([p['guess'], 1 - p['slip']])              # P(correct | state)
    # emission probs (N,T,2); masked steps -> 1
    E = np.where(X[..., None] == 1, B1, 1 - B1)
    E = np.where(M[..., None], E, 1.0)
    # forward (scaled)
    alpha = np.zeros((N, T, 2)); c = np.zeros((N, T))
    a = pi * E[:, 0]
    c[:, 0] = a.sum(1); alpha[:, 0] = a / c[:, [0]]
    for k in range(1, T):
        a = (alpha[:, k - 1] @ A) * E[:, k]
        a = np.where(M[:, [k]], a, alpha[:, k - 1])        # carry through padding
        c[:, k] = a.sum(1); alpha[:, k] = a / c[:, [k]]
    ll = np.log(np.where(M, c, 1.0)).sum()
    # backward
    beta = np.ones((N, T, 2))
    for k in range(T - 2, -1, -1):
        b = (A @ (E[:, k + 1] * beta[:, k + 1]).T).T / c[:, [k + 1]]
        beta[:, k] = np.where(M[:, [k + 1]], b, beta[:, k + 1])
    gamma = alpha * beta
    gamma /= gamma.sum(2, keepdims=True)
    # xi for U->L transitions
    xi_UL = (alpha[:, :-1, 0] * A[0, 1] * E[:, 1:, 1] * beta[:, 1:, 1]) / c[:, 1:]
    valid = M[:, 1:]
    gU_prev = np.where(valid, gamma[:, :-1, 0], 0).sum()
    learn = np.where(valid, xi_UL, 0).sum() / max(gU_prev, 1e-12)
    gm = np.where(M[..., None], gamma, 0)
    prior = gamma[:, 0, 1].mean()
    guess = (gm[..., 0] * X).sum() / max(gm[..., 0].sum(), 1e-12)
    slip = (gm[..., 1] * (1 - X)).sum() / max(gm[..., 1].sum(), 1e-12)
    clip = lambda v, hi=0.999: float(np.clip(v, 1e-4, hi))
    return dict(prior=clip(prior), learn=clip(learn), guess=clip(guess, max_gs),
                slip=clip(slip, max_gs)), ll


def simulate(n, T, p, rng):
    seqs = []
    for _ in range(n):
        L = rng.random() < p['prior']; out = []
        for _ in range(T):
            out.append(int(rng.random() < ((1 - p['slip']) if L else p['guess'])))
            if not L and rng.random() < p['learn']:
                L = True
        seqs.append(np.array(out))
    return seqs


if __name__ == '__main__':  # parameter-recovery check
    rng = np.random.default_rng(1)
    for true in [dict(prior=0.3, learn=0.15, guess=0.2, slip=0.1),
                 dict(prior=0.5, learn=0.08, guess=0.25, slip=0.15)]:
        seqs = simulate(800, 12, true, rng)
        X, M = to_matrix(seqs)
        fit, ll = em_fit(X, M, seed=2)
        print('true', true); print('fit ', {k: round(v, 3) for k, v in fit.items()})
