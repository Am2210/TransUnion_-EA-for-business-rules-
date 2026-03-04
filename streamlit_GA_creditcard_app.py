
from pdb import run

import streamlit as st
import pandas as pd
import numpy as np
import random
import copy
import time
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mutual_info_score, roc_auc_score, average_precision_score
from deap import base, creator, tools
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="GA Fraud Detection", layout="wide")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def evaluate_metrics(y_true, y_pred):
    """Compute fraud-sensitive evaluation metrics"""
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

def confusion(y_true, y_pred):
    """Compute confusion matrix values"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "TP": int(((y_pred==1)&(y_true==1)).sum()),
        "FP": int(((y_pred==1)&(y_true==0)).sum()),
        "FN": int(((y_pred==0)&(y_true==1)).sum()),
        "TN": int(((y_pred==0)&(y_true==0)).sum()),
    }

def feature_screen(Xtr, ytr, max_bins=10):
    """Screen features by Mutual Information and fraud spread"""
    rows = []
    for col in Xtr.columns:
        uniq = Xtr[col].nunique()
        if uniq < 3:
            continue
        try:
            b = pd.qcut(Xtr[col], q=min(max_bins, uniq), duplicates="drop")
        except:
            continue

        tmp = pd.DataFrame({"bin": b, "Class": ytr}).dropna()
        if tmp["bin"].nunique() < 2:
            continue

        spread = tmp.groupby("bin", observed=True)["Class"].mean()
        spread = float(spread.max() - spread.min())

        mi = float(mutual_info_score(tmp["bin"].cat.codes, tmp["Class"]))

        rows.append({"feature": col, "Mutual Information": mi, "Spread": spread})

    return pd.DataFrame(rows).sort_values(["Mutual Information","Spread"], ascending=False)

def random_condition(features, threshold_bank, categorical_features=None):
    """Generate a random condition for a rule with operator"""
    if categorical_features is None:
        categorical_features = []
    
    col = random.choice(features)
    
    # Use "==" ONLY for categorical features, comparison operators for continuous
    if col in categorical_features:
        direction = "=="
    else:
        direction = random.choice([">", "<", ">=", "<="])
    
    thr = float(random.choice(threshold_bank[col]))
    return (col, direction, thr)

def random_rule(features, threshold_bank, min_k, max_k, categorical_features=None):
    """Generate a random rule"""
    k = random.randint(min_k, max_k)
    return [random_condition(features, threshold_bank, categorical_features) for _ in range(k)]

def apply_rule(rule, Xdf):
    """Apply a rule to data and return binary predictions"""
    mask = np.ones(len(Xdf), dtype=bool)
    for col, direction, thr in rule:
        if direction == ">":
            mask &= (Xdf[col] > thr)
        elif direction == "<":
            mask &= (Xdf[col] < thr)
        elif direction == ">=":
            mask &= (Xdf[col] >= thr)
        elif direction == "<=":
            mask &= (Xdf[col] <= thr)
        elif direction == "==":
            mask &= (Xdf[col] == int(thr))
    return mask.astype(int)

def pretty_rule(rule):
    """Pretty print a rule"""
    parts = []
    for col, direction, thr in rule:
        parts.append(f"{col} {direction} {thr:.6f}")
    return " AND ".join(parts) if parts else "(empty rule)"

def setup_ga(features, threshold_bank, categorical_features=None):
    """Setup DEAP genetic algorithm components"""
    if categorical_features is None:
        categorical_features = []
    
    if "FitnessMax" in creator.__dict__:
        del creator.FitnessMax
    if "Individual" in creator.__dict__:
        del creator.Individual
    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    
    def make_individual():
        return random_rule(features, threshold_bank, 1, 5, categorical_features)
    
    toolbox.register("individual", tools.initIterate, creator.Individual, make_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("clone", copy.deepcopy)
    
    return toolbox

def mutate(ind, features, threshold_bank, categorical_features=None, min_k=1, max_k=5):
    """Mutation operator with operators and thresholds"""
    if categorical_features is None:
        categorical_features = []
    
    op = random.choice(["thr", "direction", "replace", "add", "drop"])
    if op == "thr" and len(ind) > 0:
        i = random.randrange(len(ind))
        col, direction, _ = ind[i]
        ind[i] = (col, direction, float(random.choice(threshold_bank[col])))
    elif op == "direction" and len(ind) > 0:
        i = random.randrange(len(ind))
        col, _, thr = ind[i]
        # Choose operator appropriate for feature type
        if col in categorical_features:
            ind[i] = (col, "==", thr)  # Categorical: ONLY ==
        else:
            ind[i] = (col, random.choice([">", "<", ">=", "<="]), thr)  # Continuous: comparison operators
    elif op == "replace" and len(ind) > 0:
        ind[random.randrange(len(ind))] = random_condition(features, threshold_bank, categorical_features)
    elif op == "add" and len(ind) < max_k:
        ind.append(random_condition(features, threshold_bank, categorical_features))
    elif op == "drop" and len(ind) > min_k:
        ind.pop(random.randrange(len(ind)))
    return (ind,)

def crossover(ind1, ind2):
    """Crossover operator"""
    if len(ind1) > 0 and len(ind2) > 0:
        i = random.randrange(len(ind1))
        j = random.randrange(len(ind2))
        ind1[i], ind2[j] = ind2[j], ind1[i]
    return ind1, ind2

def run_ga(toolbox, features, threshold_bank, X_val, y_val, 
           pop_size, max_generations, mutation_rate, crossover_rate, elite_size=2, patience=50, categorical_features=None):
    """Run the genetic algorithm with generation tracking, timing, and early stopping"""
    if categorical_features is None:
        categorical_features = []
    
    toolbox.register("evaluate", lambda ind: (f1_score(y_val, apply_rule(ind, X_val), zero_division=0),))
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", lambda i1, i2: crossover(i1, i2))
    toolbox.register("mutate", lambda ind: mutate(ind, features, threshold_bank, categorical_features))
    
    pop = toolbox.population(n=pop_size)
    
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    
    generation_stats = []
    algorithm_start_time = time.time()  # Track total algorithm time
    
    # Early stopping variables
    best_f1_overall = 0
    patience_counter = 0
    
    for g in range(max_generations):
        gen_start_time = time.time()  # Start timing this generation
        
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_rate:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values
        
        for ind in offspring:
            if random.random() < mutation_rate:
                toolbox.mutate(ind)
                del ind.fitness.values
        
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)
        
        # ELITISM: Preserve best individuals from previous generation
        elite = tools.selBest(pop, elite_size)
        pop_with_elite = elite + offspring
        # Sort by fitness and keep the best pop_size individuals
        pop[:] = tools.selBest(pop_with_elite, pop_size)
        
        best_ind = tools.selBest(pop, 1)[0]
        best_f1 = best_ind.fitness.values[0]
        
        # Get all F1 scores for this generation
        fits = [ind.fitness.values[0] for ind in pop]
        
        gen_end_time = time.time()  # End timing for this generation
        gen_duration = gen_end_time - gen_start_time
        
        generation_stats.append({
            "generation": g,
            "best_f1": best_f1,
            "avg_f1": np.mean(fits),
            "best_rule": pretty_rule(best_ind),
            "rule_obj": copy.deepcopy(best_ind),
            "duration": gen_duration
        })
        
        # EARLY STOPPING: Check if fitness has improved
        if best_f1 > best_f1_overall:
            best_f1_overall = best_f1
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            plateau_gen = g - patience
            st.info(f"Early stopping triggered: Fitness plateaued for {patience} generations at Generation {plateau_gen} with F1 Score: {best_f1_overall:.4f}")
            break
    
    algorithm_end_time = time.time()
    total_time = algorithm_end_time - algorithm_start_time
    
    best = tools.selBest(pop, 1)[0]
    return best, generation_stats, total_time

# ============================================================================
# STREAMLIT INTERFACE
# ============================================================================

st.title("Genetic Algorithm for Fraud Detection")
st.write("Evolve interpretable fraud detection rules using evolutionary algorithms")

# Auto-load dataset from repository
@st.cache_data
def load_default_data():
    """Load dataset from repository"""
    try:
        return pd.read_csv("creditcard.csv")
    except FileNotFoundError:
        st.error("creditcard.csv not found in repository root")
        st.stop()

df = load_default_data()

# Sidebar - only show data loaded status
with st.sidebar:
    st.header("Configuration")
    st.success("Dataset loaded from repository!")
    st.write(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# STEP 1: EDA STATISTICS
st.header("Step 1: Exploratory Data Analysis")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Dataset Shape", f"{df.shape[0]} × {df.shape[1]}")
with col2:
    fraud_rate = df["Class"].mean() * 100
    st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
with col3:
    st.metric("Legitimate Transactions (Labeled as 0)", df[df["Class"]==0].shape[0])
with col4:
    st.metric("Fraudulent Transactions (Labeled as 1)", df[df["Class"]==1].shape[0])

st.subheader("Dataset Preview")
st.dataframe(df.head(5), use_container_width=True)


# Data preparation
df = df.drop_duplicates().reset_index(drop=True)
df["Class"] = df["Class"].astype(int)

X = df.drop(columns=["Class"])
y = df["Class"]

# Define binning function (before split)
def apply_v10_binning(X_data, edges):
    """
    Replace the V10 column with V10_bin_code (integer 0–9) in the same column
    position using pre-fitted bin edges.
    """
    codes = pd.cut(X_data["V10"], bins=edges, labels=False, include_lowest=True).astype(int)
    idx   = list(X_data.columns).index("V10")
    X_data     = X_data.drop(columns=["V10"]).copy()
    X_data.insert(idx, "V10_bin_code", codes.values)
    return X_data

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Train/Val/Test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_SEED
)

# ── Fit bin edges on TRAINING set only (prevents data leakage) ────────────────
_, bin_edges = pd.qcut(X_train["V10"], q=10, duplicates="drop", retbins=True)

# Extend outermost edges to ±∞
bin_edges[0]  = -np.inf
bin_edges[-1] =  np.inf

# Apply to all splits using training-fitted edges
X_train = apply_v10_binning(X_train, bin_edges)
X_val   = apply_v10_binning(X_val,   bin_edges)
X_test  = apply_v10_binning(X_test,  bin_edges)

st.info(f"Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")


st.divider()


# STEP 2: FEATURE SCREENING
st.header("Step 2: Feature Screening")

# NEW: Example of binning and how MI/Spread are calculated
st.subheader("How Mutual Information & Spread Are Calculated")

st.markdown("""
**Example:** Let's visualize how one feature is binned and how we calculate MI and Spread.
""")

# Select a feature to demonstrate
rank_df = feature_screen(X_train, y_train)
demo_feature = rank_df["feature"].iloc[0]  # Top feature

st.write(f"**Demonstrating with feature: `{demo_feature}`**")

col1, col2 = st.columns([1, 1])

with col1:
    st.write("**Step 1: Bin the Feature**")
    st.write(f"We divide `{demo_feature}` into equal-sized bins based on data percentiles.")
    
    # Create bins
    uniq = X_train[demo_feature].nunique()
    try:
        bins = pd.qcut(X_train[demo_feature], q=min(10, uniq), duplicates="drop")
    except:
        bins = X_train[demo_feature]
    
    bin_df = pd.DataFrame({
        "Bin": bins,
        "Class": y_train
    }).dropna()
    
    # Show fraud rate per bin
    fraud_by_bin = bin_df.groupby("Bin", observed=True)["Class"].agg([
        ("Count", "count"),
        ("Frauds", "sum"),
        ("Fraud Rate", "mean")
    ]).reset_index()
    fraud_by_bin["Fraud Rate"] = fraud_by_bin["Fraud Rate"].apply(lambda x: f"{x:.2%}")
    
    st.dataframe(fraud_by_bin, use_container_width=True)

with col2:
    st.write("**Step 2: Calculate Fraud Rate Per Bin**")
    st.write("Higher variation in fraud rates = higher Spread")
    
    # Visualize fraud rate per bin
    fraud_rates = bin_df.groupby("Bin", observed=True)["Class"].mean().reset_index()
    fraud_rates.columns = ["Bin", "Fraud Rate"]
    fraud_rates["Bin"] = fraud_rates["Bin"].astype(str)
    
    fig_spread = px.bar(fraud_rates, x="Bin", y="Fraud Rate",
                       title=f"Fraud Rate Variation in {demo_feature}",
                       labels={"Fraud Rate": "Fraud Rate", "Bin": "Bin Range"},
                       color="Fraud Rate",
                       color_continuous_scale="RdYlGn")
    st.plotly_chart(fig_spread, use_container_width=True)

# Explanation
st.write("---")
col1 = st.columns([1])
with col1[0]:
    st.markdown("""
    **In this example:**
    
    - We binned the feature into 10 equal parts (or fewer if there are duplicates)
    - We calculated the fraud rate in each bin
    - The variation in fraud rates across bins gives us the Spread
    - The Mutual Information is calculated based on how well the bins separate fraud vs legitimate by taking into account the distribution of classes in each bin and how much knowing the bin reduces uncertainty about fraud
    """)

explanation_col1, explanation_col2 = st.columns([1, 1])

with explanation_col1:
    st.markdown("""
    **What is Mutual Information (MI)?**
    
    - Measures **dependency** between the feature and fraud label
    - Higher MI = Feature is **strongly related** to fraud
    - Ranges from 0 (no relationship) to 1 (perfect prediction)
    - Example: If MI = 0.15, knowing this feature value gives 15% reduction in uncertainty about fraud
    """)

with explanation_col2:
    st.markdown("""
    **What is Spread?**
    
    - Measures **variation** in fraud rates across bins
    - Spread = Max fraud rate - Min fraud rate
    - Higher Spread = Feature **discriminates well** between fraud/legitimate
    - Example: If one bin has 5% fraud and another has 45%, spread = 0.40
    """)

st.write("---")

st.markdown(f"""
**Why Both Matter?**

- **MI alone** = Feature provides information but bins might be similar
- **Spread alone** = Extreme values matter but relationship might be weak
- **Both high** = Feature reliably separates fraudulent from legitimate transactions

For `{demo_feature}`: MI = {rank_df[rank_df['feature']==demo_feature]['Mutual Information'].values[0]:.4f}, Spread = {rank_df[rank_df['feature']==demo_feature]['Spread'].values[0]:.4f}
""")

st.divider()

rank_df = feature_screen(X_train, y_train)

st.subheader("Feature Importance Ranking")
st.dataframe(rank_df, use_container_width=True)

# Visualization of feature importance
rank_df_sorted = rank_df.sort_values("Mutual Information", ascending=True)
fig = px.bar(rank_df_sorted, x="Mutual Information", y="feature", orientation='h',
              labels={"Mutual Information": "Mutual Information", "feature": "Feature"},
              title="Feature Importance (Mutual Information)")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# STEP 3: GA CONFIGURATION
st.header("Step 3: Genetic Algorithm Configuration")

# Fixed GA parameters
POP_SIZE = 75
MAX_GENERATIONS = 200
PATIENCE = 50

st.info(f"**Fixed Algorithm Settings:** Population Size: {POP_SIZE} | Algorithm will run until it reaches {MAX_GENERATIONS} Generations OR Fitness plateaued for {PATIENCE} generations")

col1, col2 = st.columns(2)
with col1:
    top_k = st.selectbox("Number of Top Features to Include", options = list(range(1, len(rank_df)+1)), index=4)
    mutation_rate = st.slider("Mutation Rate (activates when random number < mutation_rate)", min_value=0.01, max_value=0.10, value=0.05, step=0.01)

with col2:
    crossover_rate = st.slider("Crossover Rate (activates when random number < crossover_rate)", min_value=0.6, max_value=0.95, value=0.6, step=0.05)
    elite_size = st.slider("Elite Members to Preserve", min_value=1, max_value=int(POP_SIZE*0.2), value=2, step=1)

# Select top features (V10 renamed to V10_bin_code from earlier binning)
top_features = rank_df["feature"].head(top_k).tolist()
X_train_selected = X_train[top_features]
X_val_selected = X_val[top_features]
X_test_selected = X_test[top_features]

st.write(f"**Selected {len(top_features)} features:** {', '.join(top_features)}")

# Build threshold bank
quantiles = np.linspace(0.05, 0.95, 19)
threshold_bank = {
    col: np.quantile(X_train_selected[col].values, quantiles)
    for col in top_features
}

# ── Display Threshold Bank ────────────────────────────────────────────────────
st.subheader("Threshold Bank (Possible Values the GA Can Use)")

st.markdown("""
The Genetic Algorithm randomly selects thresholds from this bank when generating and mutating rules.
Each feature has 19 quantile-based thresholds spanning from the 5th to 95th percentile of the training data.
""")

# Create a display-friendly threshold bank table with quantiles as columns
threshold_display_pivot = {}
for feature in top_features:
    feature_thresholds = {}
    for i, threshold in enumerate(threshold_bank[feature]):
        quantile_pct = f"{(i + 1) * (100 / 20):.1f}%"  # Quantile percentage as column header
        feature_thresholds[quantile_pct] = f"{threshold:.6f}"
    threshold_display_pivot[feature] = feature_thresholds

threshold_df = pd.DataFrame(threshold_display_pivot).T  # Transpose so features are rows, quantiles are columns

st.dataframe(threshold_df, use_container_width=True)




st.divider()

# STEP 4: RUN GENETIC ALGORITHM
st.header("Step 4: Evolution Process")

run_button = st.button("Run Genetic Algorithm", type="primary", use_container_width=True)

if run_button:
    with st.spinner("Evolution in progress..."):
        # Identify categorical features (features with < 15 unique values)
        categorical_features = [col for col in top_features if X_train_selected[col].nunique() < 15]
        
        # Setup GA
        toolbox = setup_ga(top_features, threshold_bank, categorical_features)
        
        # Run GA with fixed parameters
        best_rule, generation_stats, total_time = run_ga(
            toolbox, top_features, threshold_bank,
            X_val_selected, y_val,
            pop_size=POP_SIZE,
            max_generations=MAX_GENERATIONS,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            elite_size=elite_size,
            patience=PATIENCE,
            categorical_features=categorical_features
        )
        
        st.session_state.best_rule = best_rule
        st.session_state.generation_stats = generation_stats
        st.session_state.total_time = total_time
        st.success(f"Evolution complete! Total runtime: {total_time:.2f} seconds")
    
    # Display generation progress
    st.subheader("Generation Progress")
    
    gen_data = {
        "Generation": [s["generation"] for s in generation_stats],
        "Best F1": [s["best_f1"] for s in generation_stats],
        "Avg F1": [s["avg_f1"] for s in generation_stats]
    }
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gen_data["Generation"], y=gen_data["Best F1"],
                             mode='lines+markers', name="Best F1", 
                             line=dict(color='green', width=3)))
    fig.add_trace(go.Scatter(x=gen_data["Generation"], y=gen_data["Avg F1"],
                             mode='lines+markers', name="Average F1",
                             line=dict(color='blue', width=2)))
    fig.update_layout(title="F1 Score Evolution", 
                     xaxis_title="Generation", 
                     yaxis_title="F1 Score",
                     hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    # EXECUTION TIME VISUALIZATION
    
    
    timing_data = {
        "Generation": [s["generation"] for s in generation_stats],
        "Time (seconds)": [s["duration"] for s in generation_stats]
    }
        # Display timing statistics
    st.subheader("Timing Statistics")
    
    total_time = st.session_state.total_time
    avg_time = np.mean(timing_data["Time (seconds)"])
    max_time = max(timing_data["Time (seconds)"])
    min_time = min(timing_data["Time (seconds)"])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Runtime", f"{total_time:.2f}s")
    with col2:
        st.metric("Avg Time/Gen", f"{avg_time:.4f}s")
    with col3:
        st.metric("Max Time/Gen", f"{max_time:.4f}s")
    with col4:
        st.metric("Min Time/Gen", f"{min_time:.4f}s")


    st.subheader("Algorithm Execution Time")
    col1, col2 = st.columns(2)
    
    with col1:
        # Time per generation bar chart
        fig_time = px.bar(x=timing_data["Generation"], y=timing_data["Time (seconds)"],
                         labels={"x": "Generation", "y": "Time (seconds)"},
                         title="Time Per Generation",
                         color=timing_data["Time (seconds)"],
                         color_continuous_scale='Viridis')
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        # Cumulative time line chart
        cumulative_time = np.cumsum(timing_data["Time (seconds)"])
        fig_cumul = px.line(x=timing_data["Generation"], y=cumulative_time,
                           labels={"x": "Generation", "y": "Cumulative Time (seconds)"},
                           title="Cumulative Execution Time",
                           markers=True)
        fig_cumul.update_traces(line=dict(color='red', width=3))
        st.plotly_chart(fig_cumul, use_container_width=True)
    

    
    # Detailed timing table
    timing_df = pd.DataFrame({
        "Generation": timing_data["Generation"],
        "Time (seconds)": [f"{t:.4f}" for t in timing_data["Time (seconds)"]],
        "Cumulative (seconds)": [f"{t:.4f}" for t in cumulative_time]
    })
    
    st.subheader("Detailed Timing Table")
    st.dataframe(timing_df, use_container_width=True)

    st.divider()
    
    # Display generation-by-generation results
    st.subheader("Generation-by-Generation Results")
    
    tabs = st.tabs([f"Gen {s['generation']}" for s in generation_stats])
    
    for tab, stat in zip(tabs, generation_stats):
        with tab:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Best F1 Score", f"{stat['best_f1']:.4f}")
            with col2:
                st.metric("Average F1 Score", f"{stat['avg_f1']:.4f}")
            
            st.write("**Best Rule:**")
            st.code(stat['best_rule'], language="text")
            
            # Evaluate best rule on validation set
            rule_pred = apply_rule(stat['rule_obj'], X_val_selected)
            metrics = evaluate_metrics(y_val, rule_pred)
            conf = confusion(y_val, rule_pred)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Precision", f"{metrics['precision']:.4f}")
            with col2:
                st.metric("Recall", f"{metrics['recall']:.4f}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("True Positives", conf['TP'])
            with col2:
                st.metric("False Positives", conf['FP'])
            with col3:
                st.metric("False Negatives", conf['FN'])
            with col4:
                st.metric("True Negatives", conf['TN'])
                
    st.divider()
    
    # STEP 5: BEST RULE EVALUATION (ACROSS ALL SETS)
    st.header("Step 5: Best Rule Evaluation (Across All Sets)")
    
    st.subheader("Best Rule Found")
    
    # Find which generation had the best rule (highest F1 score)
    best_gen_idx = max(range(len(generation_stats)), 
                       key=lambda i: generation_stats[i]["best_f1"])
    best_generation = generation_stats[best_gen_idx]["generation"]
    best_f1_value = generation_stats[best_gen_idx]["best_f1"]
    
    st.info(f"Best rule discovered in **Generation {best_generation}** with F1 Score: **{best_f1_value:.4f}**")
    
    st.code(pretty_rule(best_rule), language="text")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Number of Conditions:**")
        st.write(f"### {len(best_rule)}")
    with col2:
        st.write("**Rule Complexity:**")
        st.write(f"### {len(best_rule)}/5")
    with col3:
        st.write("**Total Generations:**")
        st.write(f"### {len(generation_stats)}")
    
    # Evaluate on all sets
    st.subheader("Performance Metrics")
    
    train_pred = apply_rule(best_rule, X_train_selected)
    val_pred = apply_rule(best_rule, X_val_selected)
    test_pred = apply_rule(best_rule, X_test_selected)
    
    train_metrics = evaluate_metrics(y_train, train_pred)
    val_metrics = evaluate_metrics(y_val, val_pred)
    test_metrics = evaluate_metrics(y_test, test_pred)
    
    # Create comparison dataframe
    metrics_df = pd.DataFrame({
        "Dataset": ["Train", "Validation", "Test"],
        "Precision": [train_metrics['precision'], val_metrics['precision'], test_metrics['precision']],
        "Recall": [train_metrics['recall'], val_metrics['recall'], test_metrics['recall']],
        "F1 Score": [train_metrics['f1'], val_metrics['f1'], test_metrics['f1']]
    })
    
    st.dataframe(metrics_df, use_container_width=True)
    
    # Metrics visualization
    fig = px.bar(metrics_df, x="Dataset", y=["Precision", "Recall", "F1 Score"],
                title="Performance Comparison Across Datasets",
                barmode="group")
    st.plotly_chart(fig, use_container_width=True)
    
    # Confusion matrices
    st.subheader("Confusion Matrices")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Training Set**")
        train_conf = confusion(y_train, train_pred)
        cm_train = np.array([[train_conf['TN'], train_conf['FP']], 
                            [train_conf['FN'], train_conf['TP']]])
        fig = px.imshow(cm_train, text_auto=True, labels=dict(color="Count"),
                       x=["Negative", "Positive"], y=["Negative", "Positive"],
                       title="Training Data", color_continuous_scale='Greens')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Validation Set**")
        val_conf = confusion(y_val, val_pred)
        cm_val = np.array([[val_conf['TN'], val_conf['FP']], 
                          [val_conf['FN'], val_conf['TP']]])
        fig = px.imshow(cm_val, text_auto=True, labels=dict(color="Count"),
                       x=["Negative", "Positive"], y=["Negative", "Positive"],
                       title="Validation Data", color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.write("**Test Set**")
        test_conf = confusion(y_test, test_pred)
        cm_test = np.array([[test_conf['TN'], test_conf['FP']], 
                           [test_conf['FN'], test_conf['TP']]])
        fig = px.imshow(cm_test, text_auto=True, labels=dict(color="Count"),
                       x=["Negative", "Positive"], y=["Negative", "Positive"],
                       title="Testing Data", color_continuous_scale='Purples')
        st.plotly_chart(fig, use_container_width=True)
