# Cold-Start Knowledge Tracing for an Adaptive Critical-Thinking Learning Game

Code, results and supplementary material for the paper:

> A. Tileubay, Z. Sadirmekova, "Cold-Start Knowledge Tracing for an Adaptive Critical-Thinking Learning Game: Evidence from Real Learner Data and Policy Simulation," 5th International Conference on Technological Advancements in Computational Sciences (ICTACS-2026), Tashkent, 2026.

The repository contains a minimal Bayesian Knowledge Tracing (BKT) implementation, the learner simulator, the scripts that reproduce every number, table and figure of the paper, and the supplementary analyses requested in review (**[SUPPLEMENT.md](SUPPLEMENT.md)**).

## Structure

```
src/
  bkt.py                         BKT: forward filtering, EM (Baum-Welch) fitting, simulation of BKT learners
  sim.py                         simulated learners (3PL IRT + ZPD learning, optional forgetting and
                                 outcome-dependent growth) and all sequencing policies
  01_download_data.py            downloads ASSISTments 2009-2010 (public)
  02_prepare_data.py             cleaning: original problems, skills with >= 300 learners
  03_experiment1.py              Experiment 1: baselines, canonical / fitted / pilot-fitted BKT, 5-fold learner CV
  04_aggregate_experiment1.py    Table II, Fig. 1 data, bootstrap CIs, per-skill tests, zone agreement
  05_experiment2.py              Experiment 2: tuning, main runs, policy and simulator sensitivity
  06_aggregate_experiment2.py    Table III, CIs, paired tests with Holm correction, sensitivity summaries
  07_validate_bkt.py             validation of bkt.py (parameter recovery, independent likelihood, pyBKT)
  08_make_figures.py             Fig. 1, Fig. 2 and Fig. S1
  09_make_supplement.py          writes SUPPLEMENT.md from the result files
results/                         result files used in the paper (CSV/JSON)
figures/                         figures (PDF/PNG)
SUPPLEMENT.md                    supplementary tables S1-S8
```

## Reproduce

Requires Python 3.10+. Run all commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/01_download_data.py     # ~83 MB
python src/02_prepare_data.py
python src/03_experiment1.py       # ~1 h on 2 cores, resumable (cache in results/e1v2_cache)
python src/04_aggregate_experiment1.py
python src/05_experiment2.py       # ~40 min on 2 cores, resumable (cache in results/exp2_cache)
python src/06_aggregate_experiment2.py
python src/07_validate_bkt.py      # ~30 min; the pyBKT part runs only if pyBKT is installed
python src/08_make_figures.py
python src/09_make_supplement.py
```

All random seeds are fixed (listed in SUPPLEMENT.md, S8), so the scripts reproduce the published values exactly. Set `NPROC` to change the number of worker processes.

**pyBKT (optional, for the independent check in step 7).** `pip install pyBKT`. pyBKT 1.4 fails under NumPy 2 in `pyBKT/fit/EM_fit.py`; replace
`log_likelihoods[i][0] = result['total_loglike']` by
`log_likelihoods[i][0] = np.asarray(result['total_loglike']).ravel()[0]`.

## Changes after review

- Level-BKT is specified completely (see `LevelBKT` in `sim.py`). The max-propagation step of the submitted version is switched off by default (`propagate=False`); it cannot change any decision, and the ablation `Level with max-propagation` reproduces identical results.
- Experiment 1: learner running mean and PFA baselines, three independent pilot draws, pilot variants with the prior fixed or fitted alone, learner-level bootstrap CIs, mean per-skill AUC.
- Experiment 2: tuned fixed curricula and a tuned KT-IDEM-style policy (grid search on a separate tuning sample), policy-parameter sensitivity, 13 simulator variants (guess rate, level spacing, learning rate, forgetting, outcome-dependent growth, task budget), Holm-corrected paired tests with effect sizes.
- Validation of the BKT implementation across 36 synthetic regimes and against an independent likelihood and pyBKT.

## Data

ASSISTments 2009-2010 skill-builder data (Feng, Heffernan & Koedinger, 2009), downloaded from the public pyBKT-examples repository. The data are not redistributed here.

## License

MIT (code). The dataset is subject to its own terms of use.
