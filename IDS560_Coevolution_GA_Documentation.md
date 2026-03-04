# Cooperative Coevolutionary Genetic Algorithm for Fraud Detection Rules
## IDS 560 — TransUnion Capstone Documentation

---

# 1. Motivation and Problem Framing

### Why Not a Standard ML Model?

- **Interpretability requirement.** Fraud detection in a business context requires rules that can be audited, explained to regulators, and acted on by analysts. A neural network or gradient-boosted tree produces a score — not an explanation.
- **Actionability.** A rule like `V14 <= -2.1 AND V4 > 3.0` can be handed directly to an operations team as a filter. A model weight vector cannot.
- **Evolutionary search advantage.** The space of valid business rules is combinatorially large but structured. Genetic algorithms navigate it efficiently without requiring gradients, making them well-suited to this discrete, interpretable search space.

### Why Not a Single Rule?

- A single AND-rule (one root-to-leaf path in a decision tree) targets one fraud pattern. Fraud is diverse — different fraud types leave different signals in different features.
- A **rule set** — multiple paths combined by OR — can cover multiple fraud subtypes simultaneously, improving recall without sacrificing per-path precision.
- The challenge is that searching over a full rule set at once is computationally intractable. Coevolution decomposes the problem.

### Why Coevolution?

- Evolving a rule set with up to 16 paths × 10 conditions = 160 simultaneous variables as a monolithic individual leads to an explosion in search space.
- **Cooperative coevolution** splits the problem into two smaller, interacting populations: one that finds good individual paths, one that finds good combinations of paths.
- The two populations evolve in alternating cycles and share building blocks through **migration**, creating a feedback loop where improvements in one population accelerate improvements in the other.

---

# 2. Business Rule Constraints

### What a Business Rule Must Look Like

These constraints are non-negotiable — they define what makes an output interpretable and deployable.

- **Condition form.** Every condition is a simple comparison between one feature and one threshold value. No compound expressions, no arithmetic between features.
- **Allowed operators.** Five and only five: `>`, `>=`, `=`, `<`, `<=`. Equality (`=`) is only meaningful for categorical/ordinal features like `V10_bin_code`; for continuous features it is excluded.
- **AND logic within a path.** A path is a conjunction — ALL conditions must be satisfied for a transaction to fire the path. This is equivalent to one root-to-leaf path in a decision tree.
- **OR logic across paths.** The rule set is a disjunction — ANY path firing flags the transaction as fraud. This is equivalent to a UNION of SQL filter statements.
- **Maximum path length.** Each path may contain at most 10 conditions. Longer paths become opaque and harder to validate.
- **Maximum rule set size.** Each rule set may contain at most 16 paths. Larger sets become unwieldy to audit and deploy.
- **No feature repetition within a path.** Each feature may appear at most once per path. Repeating a feature produces a condition that is always subsumed by or redundant with the earlier one on the same feature, wasting a node slot.

### Formal Structure

```
CONDITION  =  feature  operator  threshold
               e.g.  V14 <= -2.1
               e.g.  V10_bin_code = 0

PATH       =  condition_1  AND  condition_2  AND  ...  AND  condition_k
               k ∈ {1, ..., 10}

RULE SET   =  path_1  OR  path_2  OR  ...  OR  path_n
               n ∈ {1, ..., 16}
               Equivalent to n UNION statements in SQL
```

### Feature Types and Their Operator Pools

- **Continuous features** (V1–V28 except V10, Amount, Time after screening).
  Thresholds drawn from 19 quantile points computed on the training set. Operators: `>`, `>=`, `<`, `<=`.

- **Categorical / ordinal features** (`V10_bin_code`).
  V10 is replaced after splitting by its decile bin code — an integer 0–9 where bin 0 corresponds to the lowest decile of V10 values in the training set. This bin carries the strongest fraud signal (bin 0 has ~60× the baseline fraud rate). Operators: `>`, `>=`, `=`, `<`, `<=`. Equality is meaningful here: `V10_bin_code = 0` precisely targets the highest-risk decile.

---

# 3. Data Engineering

### V10 Binning

- **Why V10?** V10 ranks 3rd by mutual information with the fraud label and 3rd by equal-width bin fraud-rate spread in the EDA. Its fraud signal is highly non-linear — almost entirely concentrated in the bottom decile.
- **Why bin after splitting?** Binning on the full dataset before splitting would leak validation and test quantile information into the training-derived bin boundaries, contaminating the evaluation. The correct approach computes bin boundaries from training data only, then applies those fixed boundaries unchanged to all splits.
- **Step 1 — Fit on training only.** `pd.qcut(X_train["V10"], q=10, duplicates="drop", retbins=True)` computes 10 equal-frequency decile boundaries from the training set and returns the raw `bin_edges` array. The outer edges are then extended: `bin_edges[0] = -np.inf`, `bin_edges[-1] = np.inf`. This ensures that any V10 value in val or test that falls outside the training range is absorbed into the first or last bin rather than producing NaN.
- **Step 2 — Apply to all splits.** `pd.cut(X["V10"], bins=bin_edges, labels=False, include_lowest=True)` converts raw V10 values to integer codes 0–9 using the same fixed boundaries for all three splits. `pd.cut` (not `pd.qcut`) is used deliberately: `pd.qcut` recomputes its own quantile boundaries from whatever data is passed to it, so applying it independently to val or test would produce different cut points than training. `pd.cut` takes pre-computed fixed edges and simply assigns each value to a bin — no boundary recomputation occurs.
- **Consistency guarantee.** `V10_bin_code = 0` means exactly the same raw V10 interval `(−∞, q10_train]` in training, validation, and test. The bin boundaries are determined once at training time and never revisited.

### Feature Screening

- After the train/val/test split, features are ranked by **mutual information** between their decile-binned form and the fraud label, computed on the training set only.
- The top 12 features are retained for the GA. This keeps the threshold bank manageable and focuses the search on the most discriminative features.
- Importantly, `V10_bin_code` is included in this screening. Its MI score on the binned representation is high, and it now enters the GA as a categorical feature rather than a continuous one.

---

# 4. Representation

### Conditions

- The atomic unit of every rule. A triple `(feature, operator, threshold)`.
- **Threshold bank.** Rather than searching over all real values, each continuous feature has a discrete bank of 19 quantile thresholds computed from the training set. The GA picks from this bank. This makes the search space finite and interpretable — every threshold in the output corresponds to a real observed data quantile.
- For `V10_bin_code` the bank is simply `{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}`.

### Paths

- An ordered list of conditions combined by AND.
- **Length.** Between 1 and 10 conditions.
- **No duplicate features.** Each feature appears at most once. Enforced at construction time (`random.sample` without replacement), during mutation (exclude already-used features from the draw pool), and after crossover (deduplication pass).
- Analogous to a single root-to-leaf path in a decision tree.

### Rule Sets

- An ordered list of paths combined by OR.
- **Size.** Between 1 and 16 paths.
- A transaction is flagged as fraud if **any** path fires.
- Analogous to 1–16 UNION statements, each with its own WHERE clause.

---

# 5. Two-Population Architecture

### Population 1 — Path Pool

- **Individual type.** A single path (list of conditions, ≤10 nodes).
- **Fitness signal.** Standalone F1 on the validation set, treating the path as a one-path rule set.
- **What this rewards.** Paths that are individually strong fraud detectors — high-precision, high-recall single-branch rules.
- **Population size.** 60 paths (default).

### Population 2 — Rule Set Pool

- **Individual type.** A list of paths (≤16 paths, each ≤10 nodes).
- **Fitness signal.** Multi-component score: base F1 + diversity bonus − parsimony penalty (see Section 7).
- **What this rewards.** Combinations of paths that together catch more fraud than any single path could, while staying parsimonious and diverse.
- **Population size.** 40 rule sets (default).

### Why Two Separate Populations?

- A path that scores poorly in standalone F1 may be an excellent building block in combination — it catches a rare fraud subtype that other paths miss. If we evolved only rule sets, these paths would never be discovered because they are never competitive alone.
- Conversely, a path that is a strong standalone detector may dominate a rule set, leaving all other slots redundant. The path pool creates selection pressure for individual quality without forcing the rule set pool to rely on it exclusively.

---

# 6. Initialisation

### Path Pool — Random with No-Duplicate Constraint

- Each path is initialised by sampling `k` features without replacement (`random.sample`) where `k` is drawn uniformly from `{1, ..., 10}`.
- For each sampled feature, a random operator and threshold are drawn from the appropriate pool and bank.
- This guarantees the no-duplicate constraint from the first generation.

### Rule Set Pool — Coverage-Greedy (70%) + Random (30%)

#### Coverage-Greedy Initialisation
Rather than filling all path slots randomly, each slot is filled by selecting the best of 10 random candidates, where "best" means maximising the number of fraud training cases **not already covered** by previously selected paths in this rule set. This seeds diversity structurally — each path in a greedy rule set aims at a different subset of fraud cases.

#### Why 30% Random?
A fully greedy population would start too homogeneous — all rule sets would have similar path compositions because greedy selection always picks from the same small pool of effective paths. The 30% random fraction preserves population diversity and ensures the evolutionary search has varied starting material.

#### Why Use Training Fraud Rows Only?
The greedy evaluation uses only the ~350 fraud rows from the training set (pre-extracted once as `_fraud_X_train`). This is fast (350 rows vs. 198,000), leakage-free (training set only), and sufficient — we only need to distinguish which fraud patterns each candidate path covers, not measure precise F1.

---

# 7. Fitness Functions

### Path Fitness

- Straightforward validation-set F1, treating the path as a single-rule classifier.
- F1 is used rather than accuracy because of the severe class imbalance (~0.17% fraud). Accuracy would be >99% for a rule that flags nothing.
- F1 balances precision (of flagged transactions, how many are fraud) and recall (of all fraud, how many are caught).

### Rule Set Fitness — Three Components

```
score = base_F1  +  λ × diversity_bonus  −  μ × parsimony_penalty
```

#### Base F1
The OR-combination of all paths is applied to the validation set and F1 is computed. This is the primary signal — a rule set with no fraud-catching ability cannot compensate with high diversity or low complexity.

#### Diversity Bonus (λ = 0.05)
For each path in the rule set, count the fraud cases it catches that **no other path in the same rule set** catches. Sum these unique catches across all paths and normalise by `(n_fraud_val × n_paths)` to produce a value in [0, 1].

A rule set where every path catches the same 50 obvious fraud cases scores 0. A rule set where each path exclusively covers its own distinct fraud subtype scores 1. The bonus discourages convergence on a narrow fraud-dense region and incentivises complementary coverage.

#### Parsimony Penalty (μ = 0.02)
Average of `(n_paths / MAX_PATHS)` and `(avg_nodes_per_path / MAX_NODES)`. Both terms are in [0, 1], so their average is also in [0, 1].

At μ = 0.02 the maximum possible penalty is 0.02 — small enough that a genuinely better F1 always wins, but sufficient to break ties in favour of the simpler, more interpretable rule set. This reflects a real business preference: a 4-path rule with F1 = 0.78 is preferable to a 14-path rule with F1 = 0.78.

---

# 8. Genetic Operators

### Path-Level Mutations (5 types)

These operate inside a single path, modifying individual conditions. They are the fine-grained search mechanism.

#### `thr` — Threshold Shift
Replace the threshold of one condition with a different value from the same feature's bank. The feature and operator are unchanged. This slides the decision boundary along the feature's value range without altering the rule's structure.

#### `flip` — Operator Change
Switch the operator of one condition to a different operator from the same feature's allowed pool. For continuous features, cycles among `{>, >=, <, <=}` (4 choices). For categorical features, cycles among `{>, >=, =, <, <=}` (5 choices). Always picks a different operator, so the condition is guaranteed to change. Corrects cases where the GA has found the right threshold and feature but the wrong direction.

#### `replace` — Full Condition Replacement
Replace one condition entirely with a new random condition drawn from a feature not already in the path (respecting the no-duplicate constraint). This is the most disruptive path-level mutation — it discards a feature/direction/threshold triple and draws a completely fresh one. Useful when a condition is occupying a slot unproductively and preventing the path from escaping a local optimum.

#### `add_node` — Specialise the Path
Append one new condition to the path, drawn from features not already present. The path becomes longer (more specific), narrowing the set of transactions it fires on. Increases precision at the cost of recall. Only fires if `len(path) < MAX_NODES (10)`.

#### `drop_node` — Generalise the Path
Remove one condition at random from the path. The path becomes shorter (more general), broadening its coverage. Increases recall at the cost of precision. Only fires if `len(path) > MIN_NODES (1)`. Together with `add_node`, gives the GA full control over path length — it can discover that a 3-node path outperforms a 7-node path and converge on the right depth.

### Path-Level Crossover

- Swap one randomly selected condition between two paths.
- After the swap, both paths are passed through `deduplicate_path()`, which removes any condition whose feature already appears earlier in the path. This restores the no-duplicate constraint that a swap can violate.
- Fine-grained recombination: shares individual split points between paths rather than whole branches.

### Rule Set-Level Mutations (3 types)

These operate on the structure of the rule set — adding, removing, or modifying entire paths. They are the coarse-grained search mechanism.

#### `add_path` — Expand Coverage
Append a new random path to the rule set. Because OR logic means any path firing constitutes a fraud flag, this expands the rule set's reach — it can now catch a fraud pattern previously missed. Tends to increase recall. Only fires if `len(ruleset) < MAX_PATHS (16)`.

#### `drop_path` — Reduce Noise
Remove one path at random. If a path was generating false positives or overlapping entirely with another path, dropping it improves precision and reduces complexity. Only fires if `len(ruleset) > MIN_PATHS (1)`.

#### `mutate_node` — Internal Fine-Tuning
Pick one path inside the rule set at random and apply one of the five path-level mutations to it. This allows the rule set population to refine individual paths internally, without depending on the path pool to discover improvements first. It is the bridge between coarse (rule set) and fine (path) granularity.

### Rule Set-Level Crossover

- Swap one entire path between two rule sets.
- No deduplication needed at the rule set level — paths are the atomic units here, not conditions.
- Coarse-grained recombination: entire decision branches are exchanged between rule sets.

---

# 9. Migration — The Coevolution Link

Migration is what distinguishes cooperative coevolution from two independent GAs running in parallel.

### Direction 1: Paths → Rule Sets (`seed_rulesets_with_best_paths`)

#### What it does
After each path evolution block, the top `n_migrate` paths (by standalone F1) are injected into randomly chosen rule sets. If the target rule set has a free slot, the path is appended. If full, it replaces a random existing path. The rule set's fitness is invalidated and it will be re-evaluated.

#### Why it matters
Proven individual paths become available as building blocks in the rule set population immediately. Rule sets do not have to rediscover good paths from scratch through their own mutation — they can inherit them directly.

### Direction 2: Rule Sets → Paths (`promote_paths_from_rulesets`)

#### What it does
After each rule set evolution block, every individual path inside the top `n_best_rs` rule sets is extracted, wrapped as a Path individual, evaluated for standalone F1, and injected into the path pool. The path pool is then trimmed to its cap by keeping the highest-fitness paths.

#### Why it matters
A path may score poorly in standalone F1 — it catches only a rare fraud subtype that few transactions match — but be highly valuable in a rule set context where other paths cover the common cases. Without this migration direction, such paths would never survive in the path pool. With it, paths discovered through collaborative context get promoted and can seed future rule sets.

---

# 10. Advanced Mechanisms

### Elitism

#### Definition
Elitism is the practice of carrying the best `n_elite` individuals from the current generation directly into the next generation, bypassing selection, crossover, and mutation.

#### Implementation
At the start of each generation, `tools.selBest(pop, n_elite)` extracts the top individuals. They are deep-copied and prepended to the offspring list. The remaining `pop_size − n_elite` slots are filled normally.

#### Why it matters
Without elitism, even the best individual in a generation can be lost — tournament selection is stochastic and a high-fitness individual may simply not be selected. A single bad mutation can destroy weeks of evolved structure. With `n_elite = 1`, the best solution found so far is guaranteed to survive to the next generation. This prevents regression without meaningfully reducing selection pressure on the rest of the population.

---

### Hall of Fame

#### Definition
The Hall of Fame (`tools.HallOfFame(k)`) is a data structure that maintains the top `k` individuals ever evaluated across all generations and all coevolution cycles — not just the survivors in the final population.

#### Implementation
After each inner evolution block (both path and rule set), `hof.update(pop)` is called. The HoF compares the current population's fitnesses against its stored individuals and replaces lower-fitness entries. The final answer (`best_path`, `best_ruleset`) is taken from `hof[0]` rather than from the end-of-run population.

#### Why it matters
Evolution is not monotone — a great individual discovered early can be displaced by selection pressure favouring a different region of search space. Without a HoF, a rule set that achieved F1 = 0.82 in cycle 3 but drifted away by cycle 10 would be lost. The HoF guarantees that the globally best solution found at any point during the entire run is returned.

#### Separation of concerns
Two independent HoFs are maintained: `path_hof` for standalone paths and `rs_hof` for rule sets. The best path by standalone F1 and the best rule set by composite score are tracked independently. The top-paths inspection cell reports both, and also shows the **true F1** of each HoF rule set separately from its composite fitness score (which includes the diversity bonus and parsimony penalty).

---

### No-Duplicate Feature Constraint

#### Definition
Each feature may appear at most once in any given path. A path cannot contain both `V14 <= -2.1` and `V14 > -1.5` as separate conditions.

#### Why this is redundant
If both `V14 <= -2.1` and `V14 <= -1.5` appear in the same AND-path, the more restrictive condition always subsumes the other. The second condition never adds information — it either further restricts an already-restricted set (redundant) or contradicts the first (impossible to satisfy). Either way, the node slot is wasted.

#### Enforcement points
- `random_path()`: uses `random.sample(FEATURES, k)` — sampling without replacement ensures uniqueness at construction.
- `random_condition(exclude_features=)`: the optional argument accepts a set of already-used features and excludes them from the draw pool.
- `mutate_path` `add_node`: passes `{c for c, _, _ in path}` as `exclude_features` before appending.
- `mutate_path` `replace`: excludes all features currently in the path except the one being replaced.
- `cx_paths()`: calls `deduplicate_path()` on both children after the condition swap, removing any condition whose feature already appeared earlier in the path.

---

### Coverage-Greedy Rule Set Initialisation

#### Definition
A greedy heuristic for constructing diverse rule sets at initialisation time. Each path slot is filled by selecting the candidate (from a pool of 10 random paths) that maximises the number of fraud training cases **not already covered** by previously selected paths in this rule set.

#### Why not fully greedy?
30% of rule sets are still initialised fully randomly. A fully greedy population would be homogeneous — all rule sets would begin with similar path compositions because the greedy criterion always draws from the same finite pool of effective paths. The random fraction ensures the rule set population starts with varied structure.

#### Runtime cost
Greedy evaluation uses only `_fraud_X_train` — the ~350 fraud rows pre-extracted from the training set. Each candidate path is evaluated against these 350 rows, not the full 198,000-row training set. This makes greedy init comparable in speed to random init while producing meaningfully more diverse starting rule sets.

---

### Coverage-Aware Rule Set Fitness

#### Definition
A component of the rule set fitness function that rewards rule sets where each path catches fraud cases that no other path in the same rule set catches.

#### Formula
For each path `i`, compute the boolean array of fraud-validation-row activations: `fires[i]`. Count the entries where `fires[i]` is true but **all other paths** are false — these are the unique catches of path `i`. Sum across all paths and normalise:

```
diversity_bonus = Σ_i |fires[i] AND NOT(OR_{j≠i} fires[j])| / (n_fraud_val × n_paths)
```

The result is in [0, 1]. A rule set where every path fires on exactly the same fraud cases scores 0. A rule set where every path exclusively covers its own distinct fraud subtype scores 1.

#### Why this is necessary
Without diversity pressure, the rule set population converges on monocultures — 16 near-identical paths all targeting the same obvious high-fraud-density region. The OR combination adds negligible recall over a single path. The diversity bonus directly penalises this behaviour and rewards complementary coverage, which is the entire value proposition of a multi-path rule set.

---

### Parsimony Penalty

#### Definition
A small penalty term in the rule set fitness function that mildly penalises complexity, preferring simpler rule sets when fraud detection performance is otherwise equal.

#### Formula
```
parsimony_penalty = (n_paths / MAX_PATHS  +  avg_nodes_per_path / MAX_NODES) / 2
```

Both terms are normalised to [0, 1]. Their average is also in [0, 1]. Multiplied by μ = 0.02, the maximum possible penalty is 0.02 — smaller than any realistic difference in F1 between a good and a mediocre rule set.

#### Business motivation
Two rule sets with identical F1 are not equally valuable. A 3-path rule with 4 conditions per path is auditable in minutes. A 16-path rule with 9 conditions per path takes hours to validate and is difficult to explain to regulators. The parsimony penalty encodes this preference without compromising on fraud detection quality.

#### Calibration note
μ should be kept small enough that the penalty never overrides a genuine F1 improvement. At μ = 0.02, a rule set would need to be 10 F1 points worse to be preferred over a maximally complex alternative on parsimony grounds alone. Increase μ only if the output rule sets are consistently more complex than the business needs.

---

# 11. Coevolution Cycle Summary

```
INITIALISE
  path_pop    ← 60 random paths (no-duplicate constraint)
  ruleset_pop ← 40 rule sets (70% greedy-coverage, 30% random)
  path_hof, rs_hof ← empty HallOfFame(5) objects
  evaluate all individuals

FOR each coevolution cycle (default 12):

  EVOLVE path_pop for 5 generations
    each generation: elite(1) preserved + select + crossover + mutate
  UPDATE path_hof

  MIGRATE paths → rule sets
    top 5 standalone paths injected into random rule sets

  EVOLVE ruleset_pop for 5 generations
    each generation: elite(1) preserved + select + crossover + mutate
  UPDATE rs_hof

  MIGRATE rule sets → paths
    all paths inside top 3 rule sets extracted, evaluated standalone,
    injected into path_pop; path_pop trimmed to 60

RETURN path_hof[0], rs_hof[0]
```

---

# 12. Output and Reporting

### Best Individual Path
The highest-fitness path in `path_hof` by standalone F1. Reported with its node count, full condition listing, and val/test metrics (precision, recall, F1, confusion matrix).

### Best Rule Set
The highest-fitness rule set in `rs_hof` by composite score. Reported with:
- The composite fitness score (including diversity bonus and parsimony penalty)
- The **true F1** separately, so the contribution of the bonus/penalty terms is visible
- Full listing of all paths in the rule set
- Val and test metrics

### Hall of Fame Inspection
Both the top-5 paths and top-5 rule sets are displayed, allowing comparison of:
- How much the all-time-best differs from the end-of-run best (measures how much genetic drift occurred)
- Whether top rule set paths appear in the path HoF (measures how well standalone and collaborative fitness align)

### JSON Export
Results are saved to `coevo_ga_results.json` with full condition listings, operator strings, thresholds (integers for categorical, 4dp floats for continuous), and all metrics.

---

*End of documentation.*
