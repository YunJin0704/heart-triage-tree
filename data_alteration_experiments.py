"""
Data Alteration & Inconsistency Experiments
Section 6 & 7 of Assignment 1

This script demonstrates how data alterations and inconsistencies affect model outputs.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================
# LOAD BASELINE DATA
# ============================================================================
USE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", 
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

df_original = pd.read_csv("heart.csv", usecols=USE_COLS)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def train_and_evaluate(df, experiment_name):
    """Train model and return key metrics"""
    X = df.drop(columns=["target"])
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    
    # Train with same parameters as baseline
    base_tree = DecisionTreeClassifier(
        criterion="gini",
        class_weight={0: 1, 1: 3},
        min_samples_leaf=10,
        random_state=RANDOM_STATE
    )
    
    param_grid = {"ccp_alpha": np.linspace(0.0, 0.02, 21)}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(base_tree, param_grid, scoring="average_precision", cv=cv, verbose=0)
    grid_search.fit(X_train, y_train)
    
    best_tree = grid_search.best_estimator_
    y_proba = best_tree.predict_proba(X_test)[:, 1]
    
    # Metrics
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    mask = fpr <= 0.10
    recall_010 = float(np.max(tpr[mask])) if np.any(mask) else 0.0
    
    # Test on 3 examples
    example1 = pd.DataFrame({
        "age": [62], "sex": [1], "cp": [3], "trestbps": [150], "chol": [268],
        "thalach": [120], "exang": [1], "oldpeak": [2.3], "slope": [0], 
        "ca": [1], "thal": [3]
    })
    example2 = pd.DataFrame({
        "age": [54], "sex": [0], "cp": [1], "trestbps": [122], "chol": [205],
        "thalach": [170], "exang": [0], "oldpeak": [0.0], "slope": [2], 
        "ca": [0], "thal": [2]
    })
    example3 = pd.DataFrame({
        "age": [67], "sex": [1], "cp": [2], "trestbps": [138], "chol": [240],
        "thalach": [108], "exang": [1], "oldpeak": [1.8], "slope": [1], 
        "ca": [2], "thal": [7]
    })
    
    prob1 = best_tree.predict_proba(example1)[0, 1]
    prob2 = best_tree.predict_proba(example2)[0, 1]
    prob3 = best_tree.predict_proba(example3)[0, 1]
    
    results = {
        "experiment": experiment_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "recall_at_010": recall_010,
        "example1_prob": prob1,
        "example2_prob": prob2,
        "example3_prob": prob3
    }
    
    return results, best_tree

# ============================================================================
# SECTION 6: DATA ALTERATION FOR CHANGED RESULTS
# ============================================================================
print("="*80)
print("SECTION 6: DATA ALTERATION EXPERIMENTS")
print("="*80)

results_list = []

# Baseline
print("\n[1] Training BASELINE model...")
baseline_results, baseline_tree = train_and_evaluate(df_original, "Baseline")
results_list.append(baseline_results)

# Alteration A: Oversample age >= 60
print("\n[2] Alteration A: 1.5x oversampling age >= 60...")
df_A = df_original.copy()
elderly = df_A[df_A.age >= 60]
oversample = elderly.sample(frac=0.5, replace=True, random_state=RANDOM_STATE)
df_A = pd.concat([df_A, oversample], ignore_index=True)
print(f"   New shape: {df_A.shape} (was {df_original.shape})")
results_A, _ = train_and_evaluate(df_A, "Alteration A: Age>=60 Oversample")
results_list.append(results_A)

# Alteration B1: Binarize cholesterol at threshold 200
print("\n[3] Alteration B1: Binarize chol at 200...")
df_B1 = df_original.copy()
df_B1["chol"] = (df_B1["chol"] >= 200).astype(int)
results_B1, _ = train_and_evaluate(df_B1, "Alteration B1: Chol>=200")
results_list.append(results_B1)

# Alteration B2: Binarize cholesterol at threshold 240
print("\n[4] Alteration B2: Binarize chol at 240...")
df_B2 = df_original.copy()
df_B2["chol"] = (df_B2["chol"] >= 240).astype(int)
results_B2, _ = train_and_evaluate(df_B2, "Alteration B2: Chol>=240")
results_list.append(results_B2)

# Alteration C: Add 5% Gaussian noise to trestbps
print("\n[5] Alteration C: 5% noise on trestbps...")
df_C = df_original.copy()
noise_idx = df_C.sample(frac=0.05, random_state=RANDOM_STATE).index
noise = np.random.normal(0, 10, len(noise_idx))
df_C.loc[noise_idx, "trestbps"] = df_C.loc[noise_idx, "trestbps"] + noise
results_C, _ = train_and_evaluate(df_C, "Alteration C: 5% Noise on BP")
results_list.append(results_C)

# Combined alteration (A + B2)
print("\n[6] Combined: A + B2...")
df_AB = df_A.copy()
df_AB["chol"] = (df_AB["chol"] >= 240).astype(int)
results_AB, _ = train_and_evaluate(df_AB, "Combined A+B2")
results_list.append(results_AB)

# ============================================================================
# SECTION 7: INCONSISTENT DATA
# ============================================================================
print("\n" + "="*80)
print("SECTION 7: INCONSISTENT DATA EXPERIMENTS")
print("="*80)

# Label flipping: randomly flip 5% labels
print("\n[7] Inconsistency: 5% label flip...")
df_flip = df_original.copy()
flip_idx = df_flip.sample(frac=0.05, random_state=RANDOM_STATE).index
df_flip.loc[flip_idx, "target"] = 1 - df_flip.loc[flip_idx, "target"]
print(f"   Flipped {len(flip_idx)} labels")
results_flip, _ = train_and_evaluate(df_flip, "Inconsistency: 5% Label Flip")
results_list.append(results_flip)

# Duplicate conflicts: 10 pairs with same features but opposite labels
print("\n[8] Inconsistency: 10 conflicting duplicate pairs...")
df_conflict = df_original.copy()
pairs = df_original.sample(n=10, random_state=RANDOM_STATE).copy()
pairs_opposite = pairs.copy()
pairs_opposite["target"] = 1 - pairs_opposite["target"]
df_conflict = pd.concat([df_conflict, pairs, pairs_opposite], ignore_index=True)
print(f"   Added 20 duplicate rows (10 conflicting pairs)")
results_conflict, _ = train_and_evaluate(df_conflict, "Inconsistency: Duplicate Conflicts")
results_list.append(results_conflict)

# Combined inconsistency
print("\n[9] Combined: Label flip + Conflicts...")
df_both = df_flip.copy()
df_both = pd.concat([df_both, pairs, pairs_opposite], ignore_index=True)
results_both, _ = train_and_evaluate(df_both, "Inconsistency: Flip + Conflicts")
results_list.append(results_both)

# ============================================================================
# COMPARISON TABLE
# ============================================================================
print("\n" + "="*80)
print("RESULTS COMPARISON")
print("="*80)

results_df = pd.DataFrame(results_list)
print("\nOverall Metrics:")
print(results_df[["experiment", "roc_auc", "pr_auc", "recall_at_010"]].to_string(index=False))

print("\n\nI/O Example Probabilities:")
print(results_df[["experiment", "example1_prob", "example2_prob", "example3_prob"]].to_string(index=False))

# ============================================================================
# DETAILED ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("DETAILED CHANGE ANALYSIS")
print("="*80)

baseline = results_df[results_df["experiment"] == "Baseline"].iloc[0]

print("\n--- SECTION 6 ANALYSIS: How Outputs Changed ---")
for exp in ["Alteration A: Age>=60 Oversample", "Alteration B2: Chol>=240", "Alteration C: 5% Noise on BP"]:
    row = results_df[results_df["experiment"] == exp].iloc[0]
    print(f"\n{exp}:")
    print(f"  Example 1: {baseline['example1_prob']:.4f} → {row['example1_prob']:.4f} (Δ={row['example1_prob']-baseline['example1_prob']:+.4f})")
    print(f"  Example 2: {baseline['example2_prob']:.4f} → {row['example2_prob']:.4f} (Δ={row['example2_prob']-baseline['example2_prob']:+.4f})")
    print(f"  Example 3: {baseline['example3_prob']:.4f} → {row['example3_prob']:.4f} (Δ={row['example3_prob']-baseline['example3_prob']:+.4f})")
    print(f"  PR-AUC: {baseline['pr_auc']:.4f} → {row['pr_auc']:.4f} (Δ={row['pr_auc']-baseline['pr_auc']:+.4f})")

print("\n--- SECTION 7 ANALYSIS: How Inconsistencies Degraded Performance ---")
for exp in ["Inconsistency: 5% Label Flip", "Inconsistency: Duplicate Conflicts", "Inconsistency: Flip + Conflicts"]:
    row = results_df[results_df["experiment"] == exp].iloc[0]
    print(f"\n{exp}:")
    print(f"  PR-AUC: {baseline['pr_auc']:.4f} → {row['pr_auc']:.4f} (Δ={row['pr_auc']-baseline['pr_auc']:+.4f})")
    print(f"  Recall@FPR≤0.10: {baseline['recall_at_010']:.4f} → {row['recall_at_010']:.4f} (Δ={row['recall_at_010']-baseline['recall_at_010']:+.4f})")
    print(f"  Example 1: {baseline['example1_prob']:.4f} → {row['example1_prob']:.4f} (Δ={row['example1_prob']-baseline['example1_prob']:+.4f})")
    print(f"  Example 2: {baseline['example2_prob']:.4f} → {row['example2_prob']:.4f} (Δ={row['example2_prob']-baseline['example2_prob']:+.4f})")

# Save results
results_df.to_csv("alteration_results.csv", index=False)
print("\n\nResults saved to 'alteration_results.csv'")

print("\n" + "="*80)
print("EXPERIMENTS COMPLETE")
print("="*80)


