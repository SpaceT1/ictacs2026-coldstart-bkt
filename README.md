# Cold-Start Knowledge Tracing for an Adaptive Critical-Thinking Learning Game

Code and results for the paper:

> A. Tileubay, Z. Sadirmekova, "Cold-Start Knowledge Tracing for an Adaptive Critical-Thinking Learning Game: Evidence from Real Learner Data and Policy Simulation," submitted to ICTACS-2026.

The repository contains a minimal Bayesian Knowledge Tracing (BKT) implementation, the learner simulator, and the scripts that reproduce every number, table and figure of the paper.

## Structure

```
src/
  bkt.py                       BKT: forward filtering, EM (Baum-Welch) fitting, parameter-recovery check
  sim.py                       simulated learners (3PL IRT + ZPD learning) and 8 sequencing policies
  01_download_data.py          downloads ASSISTments 2009-2010 (public)
  02_prepare_data.py           cleaning: original problems, skills with >= 300 learners
  03_experiment1.py            Experiment 1: canonical vs fitted BKT, 5-fold learner-level CV
  04_aggregate_experiment1.py  Table I, zone agreement, Wilcoxon tests
  05_experiment2.py            Experiment 2: policy simulation (Table II, statistical tests)
  06_make_figures.py           Fig. 1 and Fig. 2
results/                       result files used in the paper (CSV/JSON)
figures/                       figures used in the paper (PDF)
```

## Reproduce

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/01_download_data.py     # ~83 MB
python src/02_prepare_data.py
python src/03_experiment1.py       # ~10 min, resumable
python src/04_aggregate_experiment1.py
python src/05_experiment2.py       # ~5 min
python src/06_make_figures.py
```

All random seeds are fixed, so the scripts reproduce the published values exactly. Run all commands from the repository root.

Check the BKT implementation (parameter recovery on synthetic learners):

```bash
python src/bkt.py
```

## Data

ASSISTments 2009-2010 skill-builder data (Feng, Heffernan & Koedinger, 2009), downloaded from the public pyBKT-examples repository. The data are not redistributed here.

## License

MIT (code). The dataset is subject to its own terms of use.
