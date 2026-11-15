"""Experiment 2: simulated learners and task-sequencing policies.

Generative learner model (deliberately NOT a BKT model, so every BKT policy is misspecified):
  P(correct | theta, d) = g + (1-g-s) * sigmoid(a*(theta - b_d))            (3PL IRT)
  after each attempted task: theta += eta * 4*q*(1-q),  q = sigmoid(a*(theta-b_d))
  (learning is largest for tasks of matched difficulty - zone of proximal development).
Optional variants used in the sensitivity analysis (see configure() and run()):
  forgetting (abilities of skills not practised decay towards their initial value) and
  outcome-dependent growth (gains scaled up/down after correct / incorrect answers).
"""
import numpy as np

K = 4                              # skills: interpretation, analysis, evaluation, inference
sig = lambda z: 1 / (1 + np.exp(-z))

# ---------------- simulator configuration (module globals; change only via configure) -------
CFG = dict(b=(-1.5, 0.0, 1.5), a=1.7, g=0.20, s=0.05, budget=100, q_master=0.85,
           forget=0.0, growth='zpd')
CH_LO, CH_HI = 0.50, 0.90          # appropriate-challenge band for P(correct)


def configure(**kw):
    """Set simulator parameters and recompute derived constants."""
    global B, A, G, S, BUDGET, Q_MASTER, THETA_M, FORGET, GROWTH
    CFG.update(kw)
    B = np.array(CFG['b'], float); A, G, S = CFG['a'], CFG['g'], CFG['s']
    BUDGET, Q_MASTER = CFG['budget'], CFG['q_master']
    FORGET, GROWTH = CFG['forget'], CFG['growth']
    z = (Q_MASTER - G) / (1 - G - S)
    THETA_M = B[2] + np.log(z / (1 - z)) / A   # true mastery: P(correct at level 3) >= q_master


configure()
DEFAULTS = dict(CFG)


def p_correct(theta, d): return G + (1 - G - S) * sig(A * (theta - B[d]))


def make_learners(n, rng, mu0=-1.0, sd_person=1.0, sd_skill=0.5, eta_med=0.25, eta_sd=0.5):
    theta0 = rng.normal(mu0, sd_person, (n, 1)) + rng.normal(0, sd_skill, (n, K))
    eta = eta_med * np.exp(rng.normal(0, eta_sd, (n, K)))
    return theta0, eta


# ---------------- non-adaptive policies ----------------
class Fixed:
    """Non-adaptive curriculum. Level 1 for the first `a` tasks of each skill, level 2 until task
    `b`, level 3 afterwards. mode='spiral': skills in fixed rotation (k = t mod 4);
    mode='blocked': one skill at a time, BUDGET/4 tasks per skill.
    The paper's original hand-set schedule is spiral with a=8, b=16."""
    def __init__(self, a=8, b=16, mode='spiral', name='Fixed'):
        self.t = 0; self.a, self.b, self.mode, self.name = a, b, mode, name
    def next(self):
        if self.mode == 'spiral':
            k = self.t % K; per = self.t // K               # tasks already given on skill k
        else:
            blk = max(1, BUDGET // K); k = min(self.t // blk, K - 1); per = self.t - k * blk
        d = 0 if per < self.a else (1 if per < self.b else 2)
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
    """Pure rule-based (heuristic counters, no learner model): 3 correct in a row -> level up,
    2 errors in a row -> level down, 3 correct in a row at level 3 -> skill mastered."""
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


# ---------------- BKT policies ----------------
class BKTRules:
    """Flat BKT + rules (initial design, Chapter 2): one P(L) per skill; difficulty from zones of
    P(L) (`zones`), weakest-first, max 3 consecutive tasks per skill, review every 15 tasks,
    2 errors -> level down, 3 correct -> level up; mastery when P(L) >= thr."""
    def __init__(self, p, name='BKT+rules', thr=0.95, zones=(0.40, 0.70)):
        self.p = p; self.name = name; self.thr = thr; self.zones = list(zones)
        self.L = np.full(K, p['prior']); self.cs = np.zeros(K, int); self.ws = np.zeros(K, int)
        self.override = np.full(K, -1)       # difficulty override from R3.x
        self.last_k, self.run = -1, 0; self.n = 0; self.review_due = False
    def mastered(self): return self.L >= self.thr
    def next(self):
        act = np.where(~self.mastered())[0]
        if len(act) == 0:                                    # all declared: consolidation at L3
            return int(np.argmin(self.L)), 2
        if self.review_due and self.mastered().any():        # R1.3 spaced review
            self.review_due = False
            m = np.where(self.mastered())[0]
            return int(m[self.n % len(m)]), 2
        order = act[np.argsort(self.L[act])]                 # R1.1 weakest first
        k = order[0]
        if k == self.last_k and self.run >= 3 and len(order) > 1:   # R1.2 interleaving
            k = order[1]
        if self.override[k] >= 0: d = self.override[k]
        else: d = int(np.digitize(self.L[k], self.zones))     # R2.1-R2.3
        return int(k), d
    def update(self, k, d, c):
        p = self.p; L = self.L[k]
        pc = L * (1 - p['slip']) + (1 - L) * p['guess']
        post = L * (1 - p['slip']) / pc if c else L * p['slip'] / (1 - pc)
        was_m = self.L[k] >= self.thr
        self.L[k] = post + (1 - post) * p['learn']
        if was_m and not c: self.L[k] = min(self.L[k], self.thr - 0.01)  # failed review -> active
        self.run = self.run + 1 if k == self.last_k else 1; self.last_k = k
        self.n += 1
        if self.n % 15 == 0: self.review_due = True
        if c: self.cs[k] += 1; self.ws[k] = 0
        else: self.ws[k] += 1; self.cs[k] = 0
        zone = int(np.digitize(self.L[k], self.zones))
        if self.ws[k] >= 2:  self.override[k] = max(0, d - 1); self.ws[k] = 0      # R3.1
        elif self.cs[k] >= 3: self.override[k] = min(2, d + 1); self.cs[k] = 0     # R3.2
        elif self.override[k] >= 0 and self.override[k] == zone: self.override[k] = -1
    def done(self): return False


class BKTDiff(BKTRules):
    """KT-IDEM-style flat BKT: one latent state per skill, but guess/slip depend on the task level
    (cf. Pardos & Heffernan, 2011). Under cold start they cannot be fitted, so they are set a
    priori (default) or chosen by a grid search against the simulator (sensitivity analysis).
    gate > 0 additionally requires `gate` consecutive correct level-3 answers before mastery."""
    def __init__(self, p, name='BKT-D+rules', gate=2, guess=(0.55, 0.35, 0.20),
                 slip=(0.05, 0.08, 0.10), **kw):
        super().__init__(p, name, **kw); self.gate = gate
        self.GUESS, self.SLIP = list(guess), list(slip)
        self.s3 = np.zeros(K, int)         # current streak of correct level-3 responses
    def mastered(self): return (self.L >= self.thr) & (self.s3 >= self.gate)
    def update(self, k, d, c):
        g, s = self.GUESS[d], self.SLIP[d]
        self.p = dict(self.p, guess=g, slip=s)
        super().update(k, d, c)
        self.s3[k] = self.s3[k] + 1 if (c and d == 2) else (self.s3[k] if (c and d < 2) else 0)


class LevelBKT:
    """Level-BKT + rules (proposed refinement). Full specification:
      state      P(L_{k,d}) for skill k and level d (index 0..2), all initialised to p['prior'];
                 current level d_k initialised to 0 (level 1).
      update     after an answer at level d only P(L_{k,d}) is updated, with the standard BKT
                 equations and the same four parameters for every level.
      levels     d_k increases while P(L_{k,d_k}) >= thr; it never decreases. At d_k = 2
                 (level 3) there is no further level: the skill is declared mastered when
                 P(L_{k,2}) >= thr (optionally also requiring `confirm` consecutive correct
                 level-3 answers). If later level-3 errors push P(L_{k,2}) below thr the skill
                 becomes active again.
      selection  among non-mastered skills the one with the lowest progress index
                 pi_k = d_k + P(L_{k,d_k}) in [0,3] (order='weakest'), or plain rotation
                 (order='roundrobin'); a skill is not given more than 3 times in a row.
      step-down  after 2 consecutive errors on skill k the next task of k is one level lower
                 (temporary, a single task; not possible at level 1).
      done       when every skill is declared mastered, level-3 consolidation tasks are given
                 on the skill with the lowest P(L_{k,2}).
    `propagate=True` reproduces the max-propagation step of the submitted version (a correct
    answer at level 2 raised the level-1 estimate). Because d_k never decreases and decisions
    only use P(L_{k,d_k}) and P(L_{k,2}), the step cannot change any decision (verified)."""
    def __init__(self, p, name='Level-BKT+rules', thr=0.95, confirm=0, order='weakest',
                 propagate=False):
        self.p = p; self.name = name; self.thr = thr; self.confirm = confirm
        self.order = order; self.propagate = propagate
        self.L = np.full((K, 3), p['prior']); self.lvl = np.zeros(K, int)
        self.ws = np.zeros(K, int); self.drop = np.zeros(K, bool); self.s3 = np.zeros(K, int)
        self.last_k, self.run, self.i = -1, 0, 0
    def mastered(self): return (self.L[:, 2] >= self.thr) & (self.s3 >= self.confirm)
    def progress(self): return self.lvl + self.L[np.arange(K), self.lvl]   # 0..3 scale
    def next(self):
        act = np.where(~self.mastered())[0]
        if len(act) == 0:
            return int(np.argmin(self.L[:, 2])), 2
        if self.order == 'weakest':
            order = act[np.argsort(self.progress()[act])]
        else:
            order = np.roll(act, -(self.i % len(act))); self.i += 1
        k = order[0]
        if k == self.last_k and self.run >= 3 and len(order) > 1: k = order[1]
        d = self.lvl[k] - 1 if (self.drop[k] and self.lvl[k] > 0) else self.lvl[k]
        return int(k), int(d)
    def update(self, k, d, c):
        p = self.p; L = self.L[k, d]
        pc = L * (1 - p['slip']) + (1 - L) * p['guess']
        post = L * (1 - p['slip']) / pc if c else L * p['slip'] / (1 - pc)
        self.L[k, d] = post + (1 - post) * p['learn']
        if self.propagate and c and d < 2:
            self.L[k, :d] = np.maximum(self.L[k, :d], self.L[k, d])
        if d == 2: self.s3[k] = self.s3[k] + 1 if c else 0
        self.run = self.run + 1 if k == self.last_k else 1; self.last_k = k
        self.drop[k] = False
        if c: self.ws[k] = 0
        else:
            self.ws[k] += 1
            if self.ws[k] >= 2: self.drop[k] = True; self.ws[k] = 0          # R3.1 temporary step-down
        while self.lvl[k] < 2 and self.L[k, self.lvl[k]] >= self.thr: self.lvl[k] += 1
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


def run(policy_factory, theta0, eta, rng, keep_curve=True):
    """Simulate every learner for BUDGET tasks under one policy.
    Pairing: all policies are run on the same learners (theta0, eta); the generator `rng`
    that draws the responses is created with the same seed for every policy (common random
    numbers), so differences between policies are paired by learner."""
    n = len(theta0); out = []
    for i in range(n):
        th = theta0[i].copy(); pol = policy_factory(rng) if policy_factory is not None else Oracle(th)
        ch = hard = easy = 0; waste = 0; t = 0; t_all = None; decl = {}; curve = np.zeros(BUDGET)
        dm = None
        while t < BUDGET and not pol.done():
            k, d = pol.next()
            pc = p_correct(th[k], d)
            ch += CH_LO <= pc <= CH_HI; hard += pc < CH_LO; easy += pc > CH_HI
            waste += th[k] >= THETA_M
            c = rng.random() < pc
            q = sig(A * (th[k] - B[d]))
            gain = eta[i, k] * 4 * q * (1 - q)
            if GROWTH == 'success': gain *= 1.5 if c else 0.5
            elif GROWTH == 'error': gain *= 0.5 if c else 1.5
            if FORGET > 0:
                keep = th[k]
                th += -FORGET * (th - theta0[i])          # decay of every skill ...
                th[k] = keep                              # ... except the one practised
            th[k] += gain
            pol.update(k, d, c); t += 1
            curve[t - 1] = (th >= THETA_M).mean()
            if t_all is None and (th >= THETA_M).all(): t_all = t
            dm = pol.mastered() if isinstance(pol, (BKTRules, LevelBKT)) else (pol.mastered if isinstance(pol, Counter) else None)
            if dm is not None:
                for j in np.where(dm)[0]:
                    decl.setdefault(j, bool(th[j] >= THETA_M))
        rec = dict(tasks=t, mastery=(th >= THETA_M).mean(), challenge=ch / t, too_hard=hard / t,
                   too_easy=easy / t, waste=waste / t, gain=(th - theta0[i]).mean(),
                   t_all=t_all if t_all else np.nan,
                   n_decl=len(decl), premature=sum(1 for v in decl.values() if not v),
                   correct_decl=sum(1 for v in decl.values() if v))
        if keep_curve: rec['curve'] = curve
        out.append(rec)
    return out
