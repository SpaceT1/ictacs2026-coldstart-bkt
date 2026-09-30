"""Experiment 2: simulated learners, comparison of task-sequencing policies.
Generative learner model (deliberately NOT a BKT model, so the BKT policy is misspecified):
  P(correct | theta, d) = g + (1-g-s) * sigmoid(a*(theta - b_d))
  after each attempted task: theta += eta * 4*q*(1-q),  q = sigmoid(a*(theta-b_d))
  (learning is largest for tasks of matched difficulty - zone of proximal development).
"""
import numpy as np

K = 4                       # skills: interpretation, analysis, evaluation, inference
B = np.array([-1.5, 0.0, 1.5])   # item difficulty per level 1..3
A, G, S = 1.7, 0.20, 0.05
BUDGET = 100
Q_MASTER = 0.85             # true mastery: P(correct at level 3) >= 0.85
CH_LO, CH_HI = 0.50, 0.90   # appropriate-challenge band for P(correct)

sig = lambda z: 1 / (1 + np.exp(-z))
def p_correct(theta, d): return G + (1 - G - S) * sig(A * (theta - B[d]))
THETA_M = B[2] + np.log((Q_MASTER - G) / (1 - G - S) / (1 - (Q_MASTER - G) / (1 - G - S))) / A


def make_learners(n, rng, mu0=-1.0, sd_person=1.0, sd_skill=0.5, eta_med=0.25, eta_sd=0.5):
    theta0 = rng.normal(mu0, sd_person, (n, 1)) + rng.normal(0, sd_skill, (n, K))
    eta = eta_med * np.exp(rng.normal(0, eta_sd, (n, K)))
    return theta0, eta


# ---------------- policies ----------------
class Fixed:
    """Non-adaptive linear curriculum: skills in fixed rotation, difficulty by schedule."""
    name = 'Fixed'
    def __init__(self): self.t = 0
    def next(self):
        k = self.t % K; per = self.t // K            # tasks already given per skill
        d = 0 if per < 8 else (1 if per < 16 else 2)
        self.t += 1
        return k, d
    def update(self, k, d, c): pass
    def done(self): return False


class Random:
    name = 'Random'
    def __init__(self, rng): self.rng = rng
    def next(self): return int(self.rng.integers(K)), int(self.rng.integers(3))
    def update(self, k, d, c): pass
    def done(self): return False


class Counter:
    """Pure rule-based (heuristic counters, no learner model)."""
    name = 'Counter-rules'
    def __init__(self):
        self.lvl = np.zeros(K, int); self.cs = np.zeros(K, int); self.ws = np.zeros(K, int)
        self.mastered = np.zeros(K, bool); self.i = 0
    def next(self):
        act = np.where(~self.mastered)[0]
        if len(act) == 0:
            k = self.i % K; self.i += 1; return k, 2
        k = act[self.i % len(act)]; self.i += 1
        return k, self.lvl[k]
    def update(self, k, d, c):
        if c:
            self.cs[k] += 1; self.ws[k] = 0
            if self.cs[k] >= 3:
                if self.lvl[k] == 2: self.mastered[k] = True
                self.lvl[k] = min(2, self.lvl[k] + 1); self.cs[k] = 0
        else:
            self.ws[k] += 1; self.cs[k] = 0
            if self.ws[k] >= 2: self.lvl[k] = max(0, self.lvl[k] - 1); self.ws[k] = 0
    def done(self): return False


class BKTRules:
    """Proposed: BKT estimation layer + rule groups R1-R3 (Chapter 2)."""
    def __init__(self, p, name='BKT+rules'):
        self.p = p; self.name = name
        self.L = np.full(K, p['prior']); self.cs = np.zeros(K, int); self.ws = np.zeros(K, int)
        self.override = np.full(K, -1)       # difficulty override from R3.x
        self.last_k, self.run = -1, 0; self.n = 0; self.review_due = False
    def mastered(self): return self.L >= 0.95
    def next(self):
        act = np.where(~self.mastered())[0]
        if len(act) == 0:                                    # all declared: consolidation at L3
            return int(np.argmin(self.L)), 2
        # R1.3 spaced review of a mastered skill every 15 tasks
        if self.review_due and self.mastered().any():
            self.review_due = False
            m = np.where(self.mastered())[0]
            return int(m[self.n % len(m)]), 2
        order = act[np.argsort(self.L[act])]                 # R1.1 weakest first
        k = order[0]
        if k == self.last_k and self.run >= 3 and len(order) > 1:   # R1.2 interleaving
            k = order[1]
        if self.override[k] >= 0: d = self.override[k]
        else: d = int(np.digitize(self.L[k], [0.40, 0.70]))   # R2.1-R2.3
        return int(k), d
    def update(self, k, d, c):
        p = self.p; L = self.L[k]
        pc = L * (1 - p['slip']) + (1 - L) * p['guess']
        post = L * (1 - p['slip']) / pc if c else L * p['slip'] / (1 - pc)
        was_m = self.L[k] >= 0.95
        self.L[k] = post + (1 - post) * p['learn']
        if was_m and not c: self.L[k] = min(self.L[k], 0.94)  # failed review -> active again
        self.run = self.run + 1 if k == self.last_k else 1; self.last_k = k
        self.n += 1
        if self.n % 15 == 0: self.review_due = True
        if c: self.cs[k] += 1; self.ws[k] = 0
        else: self.ws[k] += 1; self.cs[k] = 0
        zone = int(np.digitize(self.L[k], [0.40, 0.70]))
        if self.ws[k] >= 2:  self.override[k] = max(0, d - 1); self.ws[k] = 0      # R3.1
        elif self.cs[k] >= 3: self.override[k] = min(2, d + 1); self.cs[k] = 0     # R3.2
        elif self.override[k] >= 0 and self.override[k] == zone: self.override[k] = -1
    def done(self): return False


class Oracle:
    """Upper bound: knows true abilities; weakest non-mastered skill, gain-maximising level."""
    name = 'Oracle'
    def __init__(self, th): self.th = th
    def next(self):
        nm = np.where(self.th < THETA_M)[0]
        k = int(nm[np.argmin(self.th[nm])]) if len(nm) else int(np.argmin(self.th))
        q = sig(A * (self.th[k] - B)); return k, int(np.argmax(q * (1 - q)))
    def update(self, k, d, c): pass
    def done(self): return False


def run(policy_factory, theta0, eta, rng):
    n = len(theta0); out = []
    for i in range(n):
        th = theta0[i].copy(); pol = policy_factory(rng) if policy_factory is not None else Oracle(th)
        ch = 0; waste = 0; t = 0; t_all = None; decl = {}; curve = np.zeros(BUDGET)
        while t < BUDGET and not pol.done():
            k, d = pol.next()
            pc = p_correct(th[k], d)
            ch += CH_LO <= pc <= CH_HI
            waste += th[k] >= THETA_M
            c = rng.random() < pc
            q = sig(A * (th[k] - B[d]))
            th[k] += eta[i, k] * 4 * q * (1 - q)
            pol.update(k, d, c); t += 1
            curve[t - 1] = (th >= THETA_M).mean()
            if t_all is None and (th >= THETA_M).all(): t_all = t
            dm = pol.mastered() if isinstance(pol, (BKTRules, LevelBKT)) else (pol.mastered if isinstance(pol, Counter) else None)
            if dm is not None:
                for j in np.where(dm)[0]:
                    decl.setdefault(j, bool(th[j] >= THETA_M))
        final_decl = dm.sum() if dm is not None else 0
        out.append(dict(tasks=t, mastery=(th >= THETA_M).mean(), challenge=ch / t,
                        waste=waste / t, gain=(th - theta0[i]).mean(),
                        t_all=t_all if t_all else np.nan,
                        n_decl=len(decl), premature=sum(1 for v in decl.values() if not v),
                        correct_decl=sum(1 for v in decl.values() if v), curve=curve))
    return out


class BKTDiff(BKTRules):
    """KT-IDEM-style flat BKT: one latent state per skill, but guess/slip depend on the task level
    (cf. Pardos & Heffernan, 2011). Values are set a priori (cold start: nothing to fit them on).
    gate > 0 additionally requires `gate` consecutive correct level-3 answers before mastery
    (not used in the paper; gate=0 is the KT-IDEM-style baseline of Table II)."""
    GUESS = [0.55, 0.35, 0.20]
    SLIP = [0.05, 0.08, 0.10]
    def __init__(self, p, name='BKT-D+rules', gate=2):
        super().__init__(p, name); self.gate = gate
        self.s3 = np.zeros(K, int)         # current streak of correct level-3 responses
    def mastered(self): return (self.L >= 0.95) & (self.s3 >= self.gate)
    def update(self, k, d, c):
        g, s = self.GUESS[d], self.SLIP[d]
        self.p = dict(self.p, guess=g, slip=s)
        super().update(k, d, c)
        self.s3[k] = self.s3[k] + 1 if (c and d == 2) else (self.s3[k] if (c and d < 2) else 0)


class LevelBKT:
    """Proposed refinement: one BKT estimate per (skill, level); a learner advances to level d+1
    only after level d is mastered; a skill is mastered when its level-3 estimate reaches 0.95.
    Rule layer as in Chapter 2: weakest-skill-first (R1.1), interleaving cap (R1.2),
    step-down after 2 errors (R3.1)."""
    def __init__(self, p, name='Level-BKT+rules', thr=0.95):
        self.p = p; self.name = name; self.thr = thr
        self.L = np.full((K, 3), p['prior']); self.lvl = np.zeros(K, int)
        self.ws = np.zeros(K, int); self.drop = np.zeros(K, bool)
        self.last_k, self.run = -1, 0
    def mastered(self): return self.L[:, 2] >= self.thr
    def progress(self): return self.lvl + self.L[np.arange(K), self.lvl]   # 0..3 scale
    def next(self):
        act = np.where(~self.mastered())[0]
        if len(act) == 0:
            return int(np.argmin(self.L[:, 2])), 2
        order = act[np.argsort(self.progress()[act])]
        k = order[0]
        if k == self.last_k and self.run >= 3 and len(order) > 1: k = order[1]
        d = self.lvl[k] - 1 if (self.drop[k] and self.lvl[k] > 0) else self.lvl[k]
        return int(k), int(d)
    def update(self, k, d, c):
        p = self.p; L = self.L[k, d]
        pc = L * (1 - p['slip']) + (1 - L) * p['guess']
        post = L * (1 - p['slip']) / pc if c else L * p['slip'] / (1 - pc)
        self.L[k, d] = post + (1 - post) * p['learn']
        if c and d < 2: self.L[k, :d] = np.maximum(self.L[k, :d], self.L[k, d])  # easier levels implied
        self.run = self.run + 1 if k == self.last_k else 1; self.last_k = k
        self.drop[k] = False
        if c: self.ws[k] = 0
        else:
            self.ws[k] += 1
            if self.ws[k] >= 2: self.drop[k] = True; self.ws[k] = 0          # R3.1 temporary step-down
        while self.lvl[k] < 2 and self.L[k, self.lvl[k]] >= self.thr: self.lvl[k] += 1
    def done(self): return False
