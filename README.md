# Evolutionary Algorithms for Adaptive Business Rule Optimization
## TransUnion × UIC MSBA — IDS 560 Capstone | Group 1

> **File:** `IDS560_Combined_GA_Pipeline_Colab.ipynb`  
> **Results:** `results_summary.json`  
> **Split mode:** Stratified (70 / 15 / 15)  
> **Time features excluded:** Yes (confirmed standard configuration)

---

## What This Project Does

This project automatically generates and evolves **interpretable fraud detection rules** using Genetic Algorithms, as an alternative to ML-based approaches. Each rule follows the structure:

```
IF (V16 <= -1.05  AND  V4 > -1.35  AND  V14 < -0.42)  →  Flag as Fraud
```

Rules are AND-joined within a single path, and multiple paths are OR-joined across a rule set — directly matching TransUnion's production rule engine structure. The system evolves these rules over generations, optimizing for precision, recall, and alert rate simultaneously.

---

## Results Summary

### Top Features Selected (12 of 30)

Selected via Mutual Information + fraud-rate spread, fitted on training data only.

| Rank | Feature | Notes |
|------|---------|-------|
| 1 | V14 | Strongest continuous fraud signal |
| 2 | V4 | |
| 3 | **V10_bin_code** | Engineered: V10 binned into deciles (bin 0 = ~60× baseline fraud rate) |
| 4 | V12 | |
| 5 | V11 | |
| 6 | V17 | |
| 7 | V3 | |
| 8 | V16 | |
| 9 | V7 | |
| 10 | V2 | |
| 11 | V9 | |
| 12 | V21 | |

---

### Algorithm 1 — Baseline GA (Single Rule)

Evolves one AND-rule of up to 5 conditions. Serves as the benchmark.

**Best Rule Found:**
```
V16 <= -1.0568  AND  V4 > -1.3564  AND  V14 < -0.4262
AND  V10_bin_code < 2  AND  V12 <= -0.7819
```

| Split | Precision | Recall | F1 | Alert Rate |
|-------|-----------|--------|----|------------|
| Validation | 0.818 | 0.761 | 0.788 | 0.155% |
| **Test** | **0.845** | **0.690** | **0.760** | **0.136%** |

---

### Algorithm 2 — Coevolution GA (Rule Set)

Evolves a full OR-of-AND rule set using two co-evolving populations (path pool + rule set pool). The best result uses 2 paths of 5 conditions each.

**Best Rule Set Found:**
```
Rule 1: V16 <= -0.1230  AND  V14 <= -1.4372  AND  V12 <= -1.1373
        AND  V10_bin_code <= 0  AND  V4 >= -0.6978

Rule 2: V17 < -0.9812   AND  V4 > -0.0194   AND  V3 < -0.6104
        AND  V2 > 1.1081  AND  V14 < -0.0355
```

| Split | Precision | Recall | F1 | Alert Rate |
|-------|-----------|--------|----|------------|
| Validation | 0.817 | 0.817 | 0.817 | 0.167% |
| **Test** | **0.803** | **0.690** | **0.742** | **0.143%** |

---

### Operating Points (Coevolution — Test Set)

Three operating points represent different business stances on how aggressively to flag fraud.

| Operating Point | Objective | Precision | Recall | F1 | Alert Rate |
|-----------------|-----------|-----------|--------|----|------------|
| Conservative | Max precision | — | — | — | — |
| **Balanced** | Max F2 score | **0.803** | **0.690** | **0.742** | **0.143%** |
| **Aggressive** | Max recall | **0.803** | **0.690** | **0.742** | **0.143%** |

> **Note — Conservative point missing:** The Pareto front from this run did not produce any rule set with alert rate ≤ 0.05% (the conservative threshold), so no conservative point was selected. This is a dataset limitation — with only ~71 fraud cases in the validation set, the GA converges to solutions that flag slightly more transactions than the conservative cap allows. This will be revisited with synthetic multimodal data.

> **Note — Balanced = Aggressive:** Both operating points resolve to the same rule set because the Pareto front from a single-seed, single-dataset run contains limited diversity. This is the primary motivation for the synthetic data experiments planned for Week 9.

---

### Baseline vs Coevolution Comparison

| Method | Precision | Recall | F1 | Alert Rate | Rules | Conditions |
|--------|-----------|--------|----|------------|-------|------------|
| Baseline GA (single rule) | 0.845 | 0.690 | 0.760 | 0.136% | 1 | 5 |
| Coevolution GA (best rule set) | 0.803 | 0.690 | 0.742 | 0.143% | 2 | 10 |

The baseline achieves slightly higher precision on this dataset; the coevolution achieves slightly higher recall on validation (0.817 vs 0.761). The coevolution architecture's real advantage — discovering complementary rules for different fraud subtypes — requires a dataset with multiple fraud clusters to demonstrate. Both results exceed TransUnion's typical production precision target of 20–50%.

---

## How to Run

### Google Colab (Recommended)

1. Upload `IDS560_Combined_GA_Pipeline_Colab.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
3. In **Section 0**, review parameters (split mode, seeds, run flags)
4. Run all cells top to bottom — DEAP installs automatically
5. When prompted in **Section 2**, upload `creditcard.csv`
6. Outputs save to `/content/outputs_combined/`

### Quick Demo (< 5 minutes)

Set these in Section 0 before running:

```python
COEVO_CYCLES     = 3
INNER_PATH_GENS  = 2
INNER_RS_GENS    = 2
RUN_STABILITY    = False
```

### Full Run (recommended)

Default settings. Estimated runtime: ~10 min for a single coevolution run, ~40–50 min with stability analysis across all 5 seeds.

---

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SPLIT_MODE` | `"stratified"` | `"stratified"` or `"time_ordered"` (in-time/out-of-time) |
| `EXCLUDE_TIME_FEATURES` | `True` | Remove Time/Time_scaled/Time_days — confirmed standard |
| `TOP_K` | `12` | Features retained after mutual information screening |
| `RANDOM_SEED` | `42` | Primary seed |
| `STABILITY_SEEDS` | `[11,22,33,44,55]` | Seeds for multi-seed stability runs |
| `RUN_BASELINE_GA` | `True` | Run single-rule baseline (Algorithm v1) |
| `RUN_COEVOLUTION` | `True` | Run dual-population coevolution (Algorithm v2) |
| `RUN_STABILITY` | `True` | Run stability analysis across 5 seeds |
| `MAX_NODES_PER_PATH` | `5` | Max AND-conditions per rule |
| `MAX_PATHS_PER_RS` | `10` | Max OR-paths per rule set |
| `COEVO_CYCLES` | `12` | Coevolution outer cycles |
| `ALPHA` | `0.01` | Max alert rate cap (1%) for operating point selection |

---

## Output Files

All outputs are written to `outputs_combined/` (or `/content/outputs_combined/` on Colab).

| File | Description |
|------|-------------|
| `results_summary.json` | Full config + best rules + all test metrics |
| `pareto_front.csv` | All Pareto-front rule sets with precision, recall, alert rate, complexity |
| `selected_operating_points.json` | Conservative / balanced / aggressive operating points |
| `stability_metrics.csv` | Mean ± std of metrics across 5 seeds per operating point |
| `method_comparison.csv` | Side-by-side test metrics for all methods |
| `baseline_convergence.png` | Best and avg validation F1 per generation (baseline GA) |
| `coevo_convergence.png` | F1 and alert rate per cycle (coevolution GA) |
| `pareto_plots.png` | Three scatter plots: recall vs precision, alert rate vs precision, alert rate vs recall |
| `method_comparison.png` | Grouped bar chart comparing all methods |

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **No Time features** | Timestamp-based rules are dataset-specific and don't generalize to new production data. Performance is equivalent without them. |
| **Full operator set** (>, >=, <, <=, =) | Vendor constraint formally relaxed Feb 25, 2026. Restricting to `>` only caused measurable F1 drop. |
| **Rule sets, not single rules** | OR-of-AND structure matches TransUnion's production rule engine. |
| **No-duplicate-feature constraint** | Each feature appears at most once per rule for interpretability. |
| **F2 fitness for balanced point** | Missing fraud (false negative) is costlier than reviewing a false alarm (false positive). |
| **Elitism + Hall of Fame** | Best solution found at any generation is never lost. |
| **Leakage-safe engineering** | All thresholds and bin edges fitted on training data only. |

---

## Bug Fix Log

| Version | Issue | Fix |
|---------|-------|-----|
| v1 | Baseline GA crossover (`cx_base`) could swap a condition into a rule that already contained the same feature, producing rules with duplicate features (e.g., `V10_bin_code` appearing twice) | Added `deduplicate_base()` function; called on both children after every crossover. Also added inline comments to all 5 mutation operators clarifying which ops are duplicate-safe. |

---

## Next Steps

- [ ] Synthetic multimodal data generation (scikit-learn `make_classification`) with multiple fraud clusters to stress-test the coevolution architecture
- [ ] Time-ordered (in-time / out-of-time) split experiment on Kaggle dataset
- [ ] Investigate the jump in recall at ~0.13% alert rate in the Pareto front
- [ ] Parsimony penalty sweep to explore the precision/recall tradeoff space more fully
- [ ] Final presentation: completed results + vision/roadmap for future work

---

## Team

**UIC MSBA Group 1 — Spring 2026**  
Adrian Garces · Debangana Sanyal · Sam Chyu · Anand Mathur · Siddhi Jain

**TransUnion Sponsors**  
Paul Williams · Jonah Henry
