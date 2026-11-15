"""Step 5: Experiment 2 - simulated sequencing policies (Table III of the paper) plus the
robustness analyses requested in review. Every (stage, condition, population, policy) run is
cached in results/exp2_cache/, so the script can be stopped and restarted.

Stage 1  tuning   : grid search on a separate tuning sample (300 typical learners) for
                    (a) the switch points of the spiral and blocked fixed curricula and
                    (b) the level-specific guess/slip/learn values of the KT-IDEM-style policy.
                    Objective: mean share of truly mastered skills after the task budget.
Stage 2  main     : all policies, 1,000 learners per population (same learners as the paper).
Stage 3  policy   : policy-parameter sensitivity / ablations on the main learners.
Stage 4  simulator: simulator sensitivity (guess rate, difficulty spacing, learning rate,
                    forgetting, outcome-dependent growth, task budget), 500 new learners per
                    population and condition.
Runtime ~40 min on 2 cores.
"""
import numpy as np, pandas as pd, os, sys, json, itertools, hashlib
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim
from bkt import CANONICAL

CACHE = 'results/exp2_cache'
CAL = dict(CANONICAL, learn=0.11)                       # P(T) recalibrated from Experiment 1
POPS = {'weak': -2, 'typical': -1, 'strong': 0}


def make_policy(spec):
    kind, kw = spec
    kw = dict(kw)
    if kind == 'Random': return lambda r: sim.Random(r)
    if kind == 'Oracle': return None
    if kind == 'Fixed': return lambda r: sim.Fixed(**kw)
    if kind == 'Counter': return lambda r: sim.Counter()
    p = kw.pop('p', CANONICAL)
    if kind == 'Flat': return lambda r: sim.BKTRules(p, **kw)
    if kind == 'KTIDEM': return lambda r: sim.BKTDiff(p, gate=0, **kw)
    if kind == 'Level': return lambda r: sim.LevelBKT(p, **kw)
    raise ValueError(kind)


def learners(stage, pop, cond_kw):
    mu = POPS[pop]
    if stage == 'tuning':
        return sim.make_learners(300, np.random.default_rng(5000 + int(mu * 10)), mu0=mu), 11
    if stage == 'simulator':
        eta = cond_kw.get('eta_med', 0.25)
        return sim.make_learners(500, np.random.default_rng(3000 + int(mu * 10)), mu0=mu, eta_med=eta), 13
    return sim.make_learners(1000, np.random.default_rng(1000 + int(mu * 10)), mu0=mu), 7


def key(job):
    return hashlib.md5(json.dumps(job, sort_keys=True, default=str).encode()).hexdigest()[:16]


def work(job):
    fn = f"{CACHE}/{key(job)}.pkl"
    if os.path.exists(fn): return fn
    stage, cond, cond_kw, pop, name, spec = job
    sim.configure(**sim.DEFAULTS)
    sim.configure(**{k: v for k, v in cond_kw.items() if k != 'eta_med'})
    (th0, eta), rseed = learners(stage, pop, cond_kw)
    R = pd.DataFrame(sim.run(make_policy(spec), th0, eta, np.random.default_rng(rseed),
                             keep_curve=(stage == 'main')))
    R.attrs = dict(stage=stage, cond=cond, pop=pop, policy=name)
    pd.to_pickle((dict(stage=stage, cond=cond, pop=pop, policy=name, spec=spec), R), fn)
    return fn


def run_jobs(jobs, nproc):
    todo = [j for j in jobs if not os.path.exists(f"{CACHE}/{key(j)}.pkl")]
    print(f"{len(jobs)} jobs, {len(todo)} to run", flush=True)
    with Pool(nproc) as pool:
        for i, _ in enumerate(pool.imap_unordered(work, todo)):
            if i % 20 == 0: print(f"  {i + 1}/{len(todo)}", flush=True)


def load(jobs):
    return {(j[0], j[1], j[3], j[4]): pd.read_pickle(f"{CACHE}/{key(j)}.pkl")[1] for j in jobs}


# ------------------------------------------------------------------ policy definitions
BASE = {
    'Random': ('Random', {}),
    'Fixed curriculum': ('Fixed', {'a': 8, 'b': 16, 'mode': 'spiral'}),
    'Counter rules': ('Counter', {}),
    'Flat BKT+rules': ('Flat', {}),
    'KT-IDEM-style BKT+rules': ('KTIDEM', {}),
    'Level-BKT+rules': ('Level', {}),
    'Level-BKT+rules (cal.)': ('Level', {'p': CAL}),
    'Oracle': ('Oracle', {}),
}


def fixed_grid(mode):
    per = 25
    return [(a, b) for a in range(0, 17, 2) for b in range(a, per + 1, 2)]


KT_GRID = [dict(guess=(g1, g2, g3), slip=sl, p=dict(CANONICAL, learn=lt))
           for g1, g2, g3, sl, lt in itertools.product(
               (0.35, 0.55, 0.75), (0.25, 0.35, 0.50), (0.15, 0.20),
               ((0.05, 0.08, 0.10), (0.10, 0.10, 0.10)), (0.11, 0.25))]


if __name__ == '__main__':
    os.makedirs(CACHE, exist_ok=True)
    NP = int(os.environ.get('NPROC', os.cpu_count()))
    stages = sys.argv[1:] or ['tuning', 'main', 'policy', 'simulator']

    # ---------------- Stage 1: tuning
    tjobs = []
    for pop in POPS:
        for mode in ('spiral', 'blocked'):
            for a, b in fixed_grid(mode):
                tjobs.append(('tuning', 'base', {}, pop, f'Fixed-{mode}-{a}-{b}',
                              ('Fixed', {'a': a, 'b': b, 'mode': mode})))
    for i, kw in enumerate(KT_GRID):
        tjobs.append(('tuning', 'base', {}, 'typical', f'KTIDEM-{i}', ('KTIDEM', kw)))
    run_jobs(tjobs, NP)
    T = load(tjobs)
    tune = {}
    for pop in POPS:
        for mode in ('spiral', 'blocked'):
            sc = {(a, b): T[('tuning', 'base', pop, f'Fixed-{mode}-{a}-{b}')].mastery.mean()
                  for a, b in fixed_grid(mode)}
            best = max(sc, key=sc.get)
            tune[f'{mode}-{pop}'] = dict(a=best[0], b=best[1], score=sc[best],
                                         hand=sc.get((8, 16)) if mode == 'spiral' else None)
    sc = {i: T[('tuning', 'base', 'typical', f'KTIDEM-{i}')].mastery.mean() for i in range(len(KT_GRID))}
    prec = {}
    for i in range(len(KT_GRID)):
        R = T[('tuning', 'base', 'typical', f'KTIDEM-{i}')]
        prec[i] = R.correct_decl.sum() / max(R.n_decl.sum(), 1)
    bi = max(sc, key=sc.get)
    tune['ktidem'] = dict(index=bi, guess=KT_GRID[bi]['guess'], slip=KT_GRID[bi]['slip'],
                          learn=KT_GRID[bi]['p']['learn'], score=sc[bi], precision=prec[bi],
                          score_range=[min(sc.values()), max(sc.values())],
                          precision_range=[min(prec.values()), max(prec.values())])
    pd.DataFrame([dict(i=i, guess=KT_GRID[i]['guess'], slip=KT_GRID[i]['slip'],
                       learn=KT_GRID[i]['p']['learn'], mastery=sc[i], precision=prec[i])
                  for i in sc]).to_csv('results/exp2_tuning_ktidem.csv', index=False)
    rows = []
    for pop in POPS:
        for mode in ('spiral', 'blocked'):
            for a, b in fixed_grid(mode):
                rows.append(dict(pop=pop, mode=mode, a=a, b=b,
                                 mastery=T[('tuning', 'base', pop, f'Fixed-{mode}-{a}-{b}')].mastery.mean()))
    pd.DataFrame(rows).to_csv('results/exp2_tuning_fixed.csv', index=False)
    json.dump(tune, open('results/exp2_tuning.json', 'w'), indent=1, default=float)
    print(json.dumps(tune, indent=1, default=float))
    if stages == ['tuning']: sys.exit()

    sp, bl, kt = tune['spiral-typical'], tune['blocked-typical'], KT_GRID[tune['ktidem']['index']]
    MAIN = dict(BASE)
    MAIN['Fixed curriculum (tuned)'] = ('Fixed', {'a': sp['a'], 'b': sp['b'], 'mode': 'spiral'})
    # In the base simulator skills are independent and nothing is forgotten, so a blocked
    # schedule gives exactly the same outcomes as the spiral one with the same switch points;
    # the blocked curriculum is therefore evaluated only in the simulator-sensitivity stage.
    BLOCKED = ('Fixed', {'a': bl['a'], 'b': bl['b'], 'mode': 'blocked'})
    MAIN['KT-IDEM-style BKT+rules (tuned)'] = ('KTIDEM', kt)

    # ---------------- Stage 2: main
    mjobs = [('main', 'base', {}, pop, n, s) for pop in POPS for n, s in MAIN.items()]
    # population-specific tuned curricula (upper bound for non-adaptive sequencing; supplement)
    for pop in POPS:
        t = tune[f'spiral-{pop}']
        mjobs.append(('main', 'base', {}, pop, 'Fixed curriculum (tuned per population)',
                      ('Fixed', {'a': t['a'], 'b': t['b'], 'mode': 'spiral'})))
    # ---------------- Stage 3: policy-parameter sensitivity and ablations
    PVAR = {
        'Level thr=0.90': ('Level', {'thr': 0.90}),
        'Level thr=0.98': ('Level', {'thr': 0.98}),
        'Level thr=0.99': ('Level', {'thr': 0.99}),
        'Level confirm=2': ('Level', {'confirm': 2}),
        'Level confirm=3': ('Level', {'confirm': 3}),
        'Level cal. confirm=2': ('Level', {'p': CAL, 'confirm': 2}),
        'Level P(T)=0.05': ('Level', {'p': dict(CANONICAL, learn=0.05)}),
        'Level guess=.25 slip=.15': ('Level', {'p': dict(CANONICAL, guess=0.25, slip=0.15)}),
        'Level round-robin order': ('Level', {'order': 'roundrobin'}),
        'Level with max-propagation': ('Level', {'propagate': True}),
        'Flat thr=0.98': ('Flat', {'thr': 0.98}),
        'Flat thr=0.99': ('Flat', {'thr': 0.99}),
        'Flat cal. P(T)=0.11': ('Flat', {'p': CAL}),
        'Flat zones .30/.60': ('Flat', {'zones': (0.30, 0.60)}),
        'Flat zones .50/.80': ('Flat', {'zones': (0.50, 0.80)}),
    }
    pjobs = [('policy', 'base', {}, pop, n, s) for pop in POPS for n, s in PVAR.items()]
    # ---------------- Stage 4: simulator sensitivity
    CONDS = {
        'base': {},
        'guess g=0.10': {'g': 0.10}, 'guess g=0.30': {'g': 0.30},
        'spacing b=+-1.0': {'b': (-1.0, 0.0, 1.0)}, 'spacing b=+-2.0': {'b': (-2.0, 0.0, 2.0)},
        'learning rate x0.6': {'eta_med': 0.15}, 'learning rate x1.4': {'eta_med': 0.35},
        'forgetting 0.5%/task': {'forget': 0.005}, 'forgetting 2%/task': {'forget': 0.02},
        'success-driven growth': {'growth': 'success'}, 'error-driven growth': {'growth': 'error'},
        'budget 50': {'budget': 50}, 'budget 150': {'budget': 150},
    }
    SPOL = ['Fixed curriculum', 'Fixed curriculum (tuned)', 'Counter rules', 'Flat BKT+rules',
            'KT-IDEM-style BKT+rules (tuned)', 'Level-BKT+rules', 'Level-BKT+rules (cal.)', 'Oracle']
    sjobs = [('simulator', c, kw, pop, n, MAIN[n]) for c, kw in CONDS.items() for pop in POPS for n in SPOL]
    sjobs += [('simulator', c, kw, pop, 'Blocked curriculum (tuned)', BLOCKED) for c, kw in CONDS.items() for pop in POPS]
    jobs = {'main': mjobs, 'policy': pjobs, 'simulator': sjobs}
    for st in stages:
        if st in jobs: run_jobs(jobs[st], NP)
    json.dump(dict(main=[list(j) for j in mjobs], policy=[list(j) for j in pjobs],
                   simulator=[list(j) for j in sjobs]),
              open('results/exp2_jobs.json', 'w'), default=str)
    print('done')
