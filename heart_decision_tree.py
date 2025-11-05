"""
Heart Disease Decision Tree Analysis
Assignment 1: Decision Trees
Author: Your Name
Date: 10/21/25

This script implements a cost-sensitive decision tree for heart disease triage classification.
Reproducibility: Fixed random seed, versioned dependencies, stratified validation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    roc_curve, 
    confusion_matrix,
    classification_report,
    precision_recall_curve
)
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys

# ============================================================================
# CONFIGURATION & REPRODUCIBILITY
# ============================================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Print versions for reproducibility
print("="*80)
print("ENVIRONMENT & VERSIONS")
print("="*80)
print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Scikit-learn version: {__import__('sklearn').__version__}")
print(f"Random State: {RANDOM_STATE}")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================
print("LOADING DATA...")

# Use only triage-feasible features (can be obtained at intake/basic lab)
USE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", 
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

df = pd.read_csv("heart.csv", usecols=USE_COLS)

print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['target'].value_counts()}\n")

# Prepare features and target
X = df.drop(columns=["target"])
y = df["target"]

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}\n")

# ============================================================================
# MODEL TRAINING WITH COST-SENSITIVE LEARNING & CCP PRUNING
# ============================================================================
print("TRAINING DECISION TREE WITH COST-SENSITIVE LEARNING...")

# Base tree with cost-sensitive weights (FN >> FP for triage scenario)
base_tree = DecisionTreeClassifier(
    criterion="gini",
    class_weight={0: 1, 1: 3},  # False Negative cost is 3x False Positive
    min_samples_leaf=10,
    random_state=RANDOM_STATE
)

# Grid search for optimal CCP alpha (pruning parameter)
param_grid = {"ccp_alpha": np.linspace(0.0, 0.02, 21)}

# 5-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Use average precision (PR-AUC) as scoring metric
grid_search = GridSearchCV(
    base_tree, 
    param_grid=param_grid, 
    scoring="average_precision", 
    cv=cv,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Best model
best_tree = grid_search.best_estimator_
print(f"\nBest CCP alpha: {grid_search.best_params_['ccp_alpha']:.4f}")
print(f"Best CV score (PR-AUC): {grid_search.best_score_:.4f}\n")

# ============================================================================
# MODEL EVALUATION
# ============================================================================
print("="*80)
print("MODEL EVALUATION ON TEST SET")
print("="*80)

# Predictions
y_pred = best_tree.predict(X_test)
y_proba = best_tree.predict_proba(X_test)[:, 1]

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {roc_auc:.4f}")

# PR-AUC
pr_auc = average_precision_score(y_test, y_proba)
print(f"PR-AUC: {pr_auc:.4f}")

# Recall at FPR <= 0.10
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
mask = fpr <= 0.10
recall_at_010_fpr = float(np.max(tpr[mask])) if np.any(mask) else 0.0
optimal_threshold = thresholds[np.argmax(tpr[mask])] if np.any(mask) else 0.5
print(f"Recall @ FPR≤0.10: {recall_at_010_fpr:.4f}")
print(f"Optimal threshold @ FPR≤0.10: {optimal_threshold:.4f}")

# Cost-weighted utility (TP=+1, FP=-0.1, FN=-1, TN=+0)
y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_optimal).ravel()
utility = 1*tp - 0.1*fp - 1*fn
print(f"\nCost-weighted utility: {utility:.2f}")
print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")

# Confusion matrix
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_optimal)}")

# Classification report
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_optimal)}")

# ============================================================================
# TREE RULES & INTERPRETATION
# ============================================================================
print("\n" + "="*80)
print("DECISION TREE RULES")
print("="*80)
tree_rules = export_text(best_tree, feature_names=list(X.columns))
print(tree_rules)

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*80)
print("FEATURE IMPORTANCE (Permutation)")
print("="*80)

perm_importance = permutation_importance(
    best_tree, X_test, y_test, 
    n_repeats=50, 
    random_state=RANDOM_STATE,
    scoring="average_precision"
)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": perm_importance.importances_mean,
    "Std": perm_importance.importances_std
}).sort_values("Importance", ascending=False)

print(importance_df.to_string(index=False))

# ============================================================================
# I/O EXAMPLES (3 CASES)
# ============================================================================
print("\n" + "="*80)
print("I/O EXAMPLES")
print("="*80)

# Example 1: High risk case
example1 = pd.DataFrame({
    "age": [62], "sex": [1], "cp": [3], "trestbps": [150], "chol": [268],
    "thalach": [120], "exang": [1], "oldpeak": [2.3], "slope": [0], 
    "ca": [1], "thal": [3]
})
prob1 = best_tree.predict_proba(example1)[0, 1]
pred1 = "High Risk" if prob1 >= optimal_threshold else "Low Risk"
print(f"\nExample 1 (High Risk Profile):")
print(f"  Input: {example1.iloc[0].to_dict()}")
print(f"  Output: {pred1}, Probability = {prob1:.4f}")

# Example 2: Low risk case
example2 = pd.DataFrame({
    "age": [54], "sex": [0], "cp": [1], "trestbps": [122], "chol": [205],
    "thalach": [170], "exang": [0], "oldpeak": [0.0], "slope": [2], 
    "ca": [0], "thal": [2]
})
prob2 = best_tree.predict_proba(example2)[0, 1]
pred2 = "High Risk" if prob2 >= optimal_threshold else "Low Risk"
print(f"\nExample 2 (Low Risk Profile):")
print(f"  Input: {example2.iloc[0].to_dict()}")
print(f"  Output: {pred2}, Probability = {prob2:.4f}")

# Example 3: Medium risk case
example3 = pd.DataFrame({
    "age": [67], "sex": [1], "cp": [2], "trestbps": [138], "chol": [240],
    "thalach": [108], "exang": [1], "oldpeak": [1.8], "slope": [1], 
    "ca": [2], "thal": [7]
})
prob3 = best_tree.predict_proba(example3)[0, 1]
pred3 = "High Risk" if prob3 >= optimal_threshold else "Low Risk"
print(f"\nExample 3 (Medium-High Risk Profile):")
print(f"  Input: {example3.iloc[0].to_dict()}")
print(f"  Output: {pred3}, Probability = {prob3:.4f}")

# ============================================================================
# FAIRNESS EVALUATION (by Sex and Age)
# ============================================================================
print("\n" + "="*80)
print("FAIRNESS EVALUATION - GROUP METRICS")
print("="*80)

def compute_group_metrics(X, y, y_proba, group_col, group_name):
    """Compute metrics for each group"""
    results = []
    for group_val in sorted(X[group_col].unique()):
        mask = X[group_col] == group_val
        if mask.sum() < 5:  # Skip if too few samples
            continue
        
        y_g = y[mask]
        proba_g = y_proba[mask]
        
        pr_auc_g = average_precision_score(y_g, proba_g)
        fpr_g, tpr_g, _ = roc_curve(y_g, proba_g)
        mask_fpr = fpr_g <= 0.10
        recall_g = np.max(tpr_g[mask_fpr]) if np.any(mask_fpr) else 0.0
        
        results.append({
            group_name: group_val,
            "n": mask.sum(),
            "PR-AUC": pr_auc_g,
            "Recall@FPR≤0.10": recall_g
        })
    
    return pd.DataFrame(results)

# Sex-based metrics
sex_metrics = compute_group_metrics(
    X_test, y_test, y_proba, "sex", "Sex"
)
print("\nMetrics by Sex (0=Female, 1=Male):")
print(sex_metrics.to_string(index=False))

# Age-based metrics (>=60 vs <60)
X_test_age = X_test.copy()
X_test_age["age_group"] = (X_test["age"] >= 60).astype(int)
age_metrics = compute_group_metrics(
    X_test_age, y_test, y_proba, "age_group", "Age>=60"
)
print("\nMetrics by Age Group (0=<60, 1=>=60):")
print(age_metrics.to_string(index=False))

print("\n" + "="*80)
print("BASELINE TRAINING COMPLETE")
print("="*80)

# Save results for later reference
baseline_results = {
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "recall_at_010": recall_at_010_fpr,
    "utility": utility,
    "example1_prob": prob1,
    "example2_prob": prob2,
    "example3_prob": prob3
}

print("\nBaseline results saved for comparison in experiments.")


