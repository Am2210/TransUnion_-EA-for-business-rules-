# Coevolution GA — Further Analysis

## Overview

This document accompanies `Coevolution_further_analysis.ipynb`, which extends the
coevolution-based fraud rule-learning pipeline with additional robustness checks
and a feature-sensitivity study. The goal is to ensure that the rules we evolve
are **generalizable** — that they capture genuine fraud patterns rather than
artifacts of the particular dataset we trained on.

## Motivation

The coevolution GA produces interpretable rule-sets of the form:

```
R1: V14 < -1.2 AND V12 <= -0.5 AND V10_bin_code < 2
R2: V3 <= -2.1 AND V9 < -2.2 AND V22 <= 0.3
```

Two questions to focus on:

1. **Stability** — If we re-run the GA with different random seeds, do we get
   similar operating points (precision, recall, alert rate)?  Or is a single
   run a lucky outlier?

2. **Feature robustness** — Are any rules relying on features that won't exist
   (or won't behave the same way) in a production environment?

This notebook addresses both.

### Why stability matters

A genetic algorithm is stochastic — it uses random number generation at every
stage: initialising the population, choosing parents, deciding where to mutate,
picking crossover points.  Two runs with different starting randomness can
converge to different solutions, even on the same data.

If the GA only finds a good rule-set under one specific sequence of random
decisions, the result is fragile and hard to trust.  Conversely, if five
independent runs (each with a different random seed) all arrive at similar
precision, recall, and alert-rate numbers, we can be confident the patterns are
real and robust.

### What is a random seed?

A **random seed** is an integer that initialises the random number generator to
a known starting state.  Setting `seed = 42` before a run means every random
choice (population init, mutation, crossover selection, etc.) will follow the
exact same sequence every time — making the run **deterministic and
reproducible**.

Changing the seed (e.g. to 11, 22, 33, 44, 55) produces a completely different
sequence of random decisions while keeping every other parameter identical.
This lets us isolate the effect of randomness from the effect of configuration.

### How we use seeds in this analysis

We run the full coevolution pipeline **five times**, once per seed:

| Seed | Purpose |
|---|---|
| 11 | First independent run |
| 22 | Second independent run |
| 33 | Third independent run |
| 44 | Fourth independent run |
| 55 | Fifth independent run |

For each run we record the Pareto front and the three selected operating points
(conservative, balanced, aggressive).  We then aggregate across all five runs:

- **Overlap** — How many unique rule-sets were selected?  If all five seeds
  pick the same (or very similar) rules, the search space has a clear optimum.
  If every seed picks a different rule-set, the landscape is flat or noisy.

- **Mean ± standard deviation** — For each operating point we compute the
  average and spread of precision, recall, and alert rate across the five
  seeds.  A low standard deviation means the GA reliably finds solutions of
  similar quality.

This is analogous to running a machine-learning model with different random
initialisations and reporting the mean ± std of accuracy — a standard practice
to demonstrate that results are not a fluke.

## What the notebook does

| Section | Purpose |
|---|---|
| Setup & Configuration | Detects repo root, imports modules, defines all GA parameters in one place |
| Single-seed demo | Runs one coevolution cycle to verify the pipeline works end-to-end |
| Multi-seed stability (baseline) | Loads or runs the GA across 5 seeds (11, 22, 33, 44, 55) with all features |
| Feature-sensitivity run | Re-runs the same 5 seeds with a curated feature exclusion list |
| Step-3 analysis | Selects 3 operating points, evaluates on the held-out TEST split, generates Pareto plots |
| Verification | Scans every output file to confirm excluded features do not appear in any evolved rule |
| Comparison | Side-by-side table of baseline vs. feature-restricted metrics, with an automated materiality check |

## About the `Time` feature

The original credit card dataset includes a `Time` column that records the
**number of seconds elapsed since the first transaction in the dataset**.  During
preprocessing, two derived columns were also created:

- `Time_scaled` — standardised version of `Time`
- `Time_days` — `Time` converted to fractional days

These features are inherently tied to the specific two-day collection window of
the dataset.  A rule such as `Time_days < 0.8` effectively means *"transactions
that occurred in roughly the first 19 hours of recording"*, which has no
meaning in a live fraud-detection system where transactions arrive continuously.

To test whether any evolved rules were leaning on this collection artifact, the
notebook re-runs the full pipeline with `Time`, `Time_scaled`, and `Time_days`
removed from the GA search space.  All other settings (population sizes,
constraint budgets, seeds, selection criteria) remain identical so the
comparison is fair.

## How the coevolution GA works

### What is a genetic algorithm?

A genetic algorithm (GA) borrows ideas from biological evolution to search for
good solutions.  Instead of testing every possible rule combination (which would
be astronomically large), the GA maintains a *population* of candidate
solutions and iteratively improves them through:

- **Selection** — better-performing candidates are more likely to survive to the
  next generation.
- **Crossover** — two parent candidates swap parts of their structure to create
  offspring, combining good traits from both.
- **Mutation** — small random changes (tweaking a threshold, swapping a feature,
  adding or removing a condition) introduce diversity and prevent the search
  from getting stuck.

Over many generations, the population converges toward high-quality solutions.

### Two-population coevolution

Our GA co-evolves **two populations** that help each other improve:

- **Population A — Paths** (60 individuals).  Each path is a single AND-rule
  with up to 5 conditions, e.g. `V14 < -1.2 AND V12 <= -0.5`.  Paths are
  evaluated individually using the **F2 score** (a recall-weighted metric) on
  the validation split, subject to an alert-rate cap and a minimum
  true-positive count.  Selection uses tournament selection (pick 3 at random,
  keep the best).

- **Population B — Rule-sets** (40 individuals).  Each rule-set is an OR of
  multiple paths, e.g. `R1 | R2 | R3`.  A transaction is flagged if *any*
  path fires.  Rule-sets are evaluated on three objectives simultaneously:
  maximize recall, maximize precision, minimize alert rate.  Selection uses
  **NSGA-II**, which maintains a diverse front of non-dominated trade-off
  solutions (the Pareto front).

### Migration between populations

After each evolution cycle, the two populations exchange genetic material:

1. **Paths → Rule-sets**: the top 5 paths are injected into random rule-sets,
   giving rule-sets access to newly discovered high-quality building blocks.
2. **Rule-sets → Paths**: paths extracted from the top 3 rule-sets are promoted
   back into the path population, giving the path search new starting points
   informed by what works well in combination.

This co-evolutionary loop runs for 10 cycles (each containing 5 inner
generations per population), totalling roughly 100 generations of refinement.

### Key constraints (sponsor-aligned)

| Constraint | Default |
|---|---|
| Max conditions per path | 5 |
| Max paths per rule-set | 10 |
| Total condition budget | 20 |
| Alert-rate cap (α) | 1 % |
| Min TP on validation | 5 |

### V10 binning

`V10` (a PCA component) shows a strong non-linear fraud signal concentrated in
its lowest decile.  To let the GA exploit this cleanly, V10 is discretised into
10 ordinal bins (`V10_bin_code` = 0–9):

- Bin edges are fit on **TRAIN only** (`pd.qcut`, `q=10`).
- Edges are extended to `[-inf, inf]` and applied to val/test with `pd.cut`.
- The GA treats `V10_bin_code` as categorical, allowing the equality operator
  (`=`) alongside the usual comparisons.

## Operating-point selection criteria

### What is the Pareto front?

Because rule-set fitness is multi-objective (recall vs. precision vs. alert
rate), there is no single "best" solution — improving one metric typically
worsens another.  The **Pareto front** is the set of rule-sets where no other
solution is better on *all three* objectives simultaneously.  Every point on
this front represents a different trade-off.

The Pareto front is saved as `pareto_front.csv`, and the three scatter plots
visualise the trade-off surface from different angles.

### Selecting operating points

From the Pareto front we pick **three operating points** that correspond to
different business risk appetites:

| Operating point | Objective | Alert-rate cap | Support / quality floor |
|---|---|---|---|
| Conservative | Maximise precision | ≤ 0.05 % | Val TP ≥ 5 |
| Balanced | Maximise F2 score | ≤ 0.20 % | Val TP ≥ 10 |
| Aggressive | Maximise recall | ≤ 1.00 % | Val precision ≥ 0.10 |

- **Conservative** is for teams that can only investigate a handful of alerts
  per day — it fires rarely but is almost always right.
- **Balanced** trades off both directions — a good default for most fraud
  operations teams.
- **Aggressive** catches as many fraudulent transactions as possible, accepting
  more false alerts in return.

Selection is performed on **validation** metrics.  The held-out **TEST** split
is used only for final reporting — never for tuning or selection — to give an
honest estimate of real-world performance.

## Outputs

All artifacts are saved under `outputs_no_time/`:

```
outputs_no_time/
├── feature_list_no_time.json        # Filtered feature list
├── thresholds_train_no_time.json    # Filtered threshold bank
├── pareto_front.csv                 # Pareto-optimal rule-sets
├── selected_operating_points.json   # 3 selected points (TEST metrics)
├── report.md                        # Step-3 summary report
├── stability_report.md              # Cross-seed stability narrative
├── stability_metrics.csv            # Stability stats table
├── comparison_summary.md            # Baseline vs restricted comparison
├── comparison_table.csv             # Same, in CSV
├── plots/
│   ├── recall_vs_precision.png
│   ├── alert_rate_vs_precision.png
│   └── alert_rate_vs_recall.png
└── stability_seeds/
    ├── seed_11/ … seed_55/          # Per-seed pareto, operating points, logs
    ├── stability_report.md
    └── stability_metrics.csv
```

## How to reproduce

### From the notebook

Open `Coevolution_further_analysis.ipynb` and **Run All**.  Configuration
is in Section 2; adjust parameters there before running if needed.

### From the command line

```bash
# Full feature-sensitivity run (5 seeds + step-3 + comparison)
python run_no_time.py

# Single coevolution run
python coevolution_ga.py --seed 42 --exclude_features "Time,Time_scaled,Time_days"

# Multi-seed stability
python run_coevolution_stability.py --seeds 11,22,33,44,55 --exclude_features "Time,Time_scaled,Time_days"
```

## Interpreting the comparison

The notebook automatically flags whether removing a feature group caused
**material changes** (defined as a > 5 percentage-point shift in precision or
recall for any operating point).

- **Material change detected** — the excluded features were influencing rule
  selection.  Review the comparison table to decide whether those rules were
  capturing real signal or dataset artifacts.

- **No material change** — the excluded features were not driving rule quality.
  They can be safely dropped to keep the rule-set deployment-ready.

## Dependencies

| Package | Purpose |
|---|---|
| `deap` | Genetic algorithm framework (NSGA-II, tournament selection) |
| `pandas`, `numpy` | Data handling and metrics |
| `matplotlib` | Pareto front visualisation |

Install with:

```bash
pip install deap pandas numpy matplotlib
```
