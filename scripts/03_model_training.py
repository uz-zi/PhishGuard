# ============================================================
# PhishGuard — Step 3: Model Training & Evaluation
# Input  : data/processed/X_train.csv, y_train.csv
# Output : model/model.pkl, model/model_results.json
# Author : Uzman Zahid
# ============================================================

# --- IMPORT LIBRARIES ----------------------------------------
import pandas as pd               # for loading X and y CSV files
import numpy as np                # for numerical operations
import matplotlib.pyplot as plt   # for plotting confusion matrix and charts
import seaborn as sns             # for better looking heatmaps
import json                       # for saving model results as JSON
import joblib                     # for saving the trained model as .pkl file
import os                         # for creating folders
import time                       # for measuring training time

# import the two ML models we are training
from sklearn.linear_model import LogisticRegression    # simple baseline model
from sklearn.ensemble import RandomForestClassifier    # main model

# import tools for splitting data and evaluating models
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,      # overall % correct predictions
    precision_score,     # of all URLs flagged as phishing, how many were actually phishing
    recall_score,        # of all actual phishing URLs, how many did we catch
    f1_score,            # harmonic mean of precision and recall
    confusion_matrix,    # table showing TP, FP, TN, FN
    classification_report  # full summary of all metrics
)

# --- PRINT HEADER --------------------------------------------
print("=" * 60)
print("PHISHGUARD — MODEL TRAINING SCRIPT")
print("=" * 60)

# ── 1. LOAD X AND y ──────────────────────────────────────────
# loads the feature matrix saved by Script 02
X = pd.read_csv('data/processed/X_train.csv')

# loads the label column saved by Script 02
# .values.ravel() converts from a 2D column to a 1D array (required by sklearn)
y = pd.read_csv('data/processed/y_train.csv').values.ravel()

# confirms the data loaded correctly
print(f"\n✅ Data loaded successfully")
print(f"   X shape : {X.shape}")
print(f"   y shape : {y.shape}")
print(f"   Phishing   (1) : {(y == 1).sum():,}")
print(f"   Legitimate (0) : {(y == 0).sum():,}")

# ── 2. SPLIT INTO TRAIN AND TEST SETS ────────────────────────
# splits data into 70% training and 30% testing
# test_size=0.3 means 30% goes to testing
# random_state=42 means the split is reproducible (same result every time)
# stratify=y ensures both sets have the same phishing/legitimate ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# confirms the split sizes
print(f"\n── Train/Test Split ────────────────────────────────────")
print(f"   Training set : {X_train.shape[0]:,} rows (70%)")
print(f"   Testing set  : {X_test.shape[0]:,} rows (30%)")

# ── 3. TRAIN LOGISTIC REGRESSION (BASELINE) ──────────────────
print(f"\n── Training Logistic Regression ────────────────────────")

# records the start time so we can measure how long training takes
lr_start = time.time()

# creates the Logistic Regression model
# max_iter=1000 gives the model enough iterations to converge
# class_weight='balanced' handles any minor class imbalance
# n_jobs=-1 uses all CPU cores to speed up training
lr_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)

# trains the model on the training data
lr_model.fit(X_train, y_train)

# calculates how long training took
lr_time = time.time() - lr_start
print(f"   ✅ Training complete in {lr_time:.1f} seconds")

# uses the trained model to predict labels for the test set
lr_predictions = lr_model.predict(X_test)

# ── 4. TRAIN RANDOM FOREST (MAIN MODEL) ──────────────────────
print(f"\n── Training Random Forest ──────────────────────────────")
print(f"   This may take 5-10 minutes on your i5 — please wait...")

# records the start time
rf_start = time.time()

# creates the Random Forest model
# n_estimators=100 = builds 100 decision trees and combines their votes
# class_weight='balanced' handles class imbalance automatically
# n_jobs=-1 uses ALL CPU cores (all 8 threads on your i5-8250U) — maximum speed
# random_state=42 makes results reproducible
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)

# trains the model on the training data
# this is the slow step — building 100 decision trees on 235k rows
rf_model.fit(X_train, y_train)

# calculates how long training took
rf_time = time.time() - rf_start
print(f"   ✅ Training complete in {rf_time:.1f} seconds")

# uses the trained model to predict labels for the test set
rf_predictions = rf_model.predict(X_test)

# ── 5. EVALUATE BOTH MODELS ──────────────────────────────────
print(f"\n── Model Evaluation Results ────────────────────────────")

# function to calculate and print all metrics for a model
def evaluate_model(name, y_true, y_pred):
    # calculates all 4 metrics
    acc  = accuracy_score(y_true, y_pred)           # overall correctness
    prec = precision_score(y_true, y_pred)           # quality of phishing predictions
    rec  = recall_score(y_true, y_pred)              # how many phishing URLs we caught
    f1   = f1_score(y_true, y_pred)                  # balance between precision and recall

    # prints a formatted results table
    print(f"\n   {name}")
    print(f"   {'─' * 40}")
    print(f"   Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"   Precision : {prec:.4f}  ({prec*100:.2f}%)")
    print(f"   Recall    : {rec:.4f}  ({rec*100:.2f}%)")
    print(f"   F1-Score  : {f1:.4f}  ({f1*100:.2f}%)")

    # returns metrics as a dictionary for saving later
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

# evaluates both models and stores results
lr_results = evaluate_model("Logistic Regression", y_test, lr_predictions)
rf_results = evaluate_model("Random Forest", y_test, rf_predictions)

# ── 6. COMPARE MODELS ────────────────────────────────────────
print(f"\n── Model Comparison ────────────────────────────────────")
print(f"   {'Metric':<15} {'Log. Regression':>18} {'Random Forest':>15}")
print(f"   {'─' * 50}")
for metric in ['accuracy', 'precision', 'recall', 'f1']:
    lr_val = lr_results[metric]
    rf_val = rf_results[metric]
    # adds a ✅ next to the better score
    winner = '✅' if rf_val > lr_val else '  '
    print(f"   {metric:<15} {lr_val:>17.4f}  {rf_val:>13.4f} {winner}")

# ── 7. CROSS VALIDATION ON RANDOM FOREST ─────────────────────
print(f"\n── Cross Validation (Random Forest, k=5) ───────────────")
print(f"   Running 5-fold cross validation — please wait...")

# runs 5-fold cross validation — splits data into 5 parts
# trains on 4 parts, tests on 1 part, repeats 5 times
# gives a more reliable accuracy estimate than a single train/test split
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='f1', n_jobs=-1)

# prints the score for each fold
print(f"   Fold scores : {[f'{s:.4f}' for s in cv_scores]}")
print(f"   Mean F1     : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ── 8. PRINT CLASSIFICATION REPORTS ──────────────────────────
print(f"\n── Classification Report — Logistic Regression ─────────")
# prints detailed per-class metrics (0 = legitimate, 1 = phishing)
print(classification_report(y_test, lr_predictions,
      target_names=['Legitimate', 'Phishing']))

print(f"\n── Classification Report — Random Forest ───────────────")
print(classification_report(y_test, rf_predictions,
      target_names=['Legitimate', 'Phishing']))

# ── 9. PLOT CONFUSION MATRICES ───────────────────────────────
# creates a side by side figure with 2 confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# function to plot a single confusion matrix
def plot_confusion(ax, y_true, y_pred, title):
    # calculates the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # plots it as a heatmap
    # annot=True shows numbers inside each cell
    # fmt='d' shows integers not decimals
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    ax.set_title(title, fontweight='bold', fontsize=13)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('Actual Label')

# plots confusion matrix for Logistic Regression
plot_confusion(axes[0], y_test, lr_predictions, 'Logistic Regression')

# plots confusion matrix for Random Forest
plot_confusion(axes[1], y_test, rf_predictions, 'Random Forest')

# adjusts spacing
plt.tight_layout()

# saves the figure
plt.savefig('data/processed/confusion_matrices.png', dpi=150)

# shows on screen
plt.show()
print(f"✅ Confusion matrices saved!")

# ── 10. PLOT FEATURE IMPORTANCE (Random Forest only) ─────────
# Random Forest can tell us how much each feature contributed to decisions
# this is different from correlation — it shows actual model importance
importances = rf_model.feature_importances_

# pairs each feature name with its importance score
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

# creates the chart
plt.figure(figsize=(10, 7))
colors = ['#e74c3c' if v > 0.05 else '#3498db' if v > 0.02 else '#95a5a6'
          for v in feature_importance_df['importance']]

# plots horizontal bars
plt.barh(feature_importance_df['feature'][::-1],
         feature_importance_df['importance'][::-1],
         color=colors[::-1], edgecolor='black', height=0.6)

plt.xlabel('Feature Importance Score', fontsize=12)
plt.title('Random Forest — Feature Importance',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('data/processed/feature_importance.png', dpi=150)
plt.show()
print(f"✅ Feature importance chart saved!")

# ── 11. SAVE BEST MODEL ───────────────────────────────────────
print(f"\n── Saving Best Model ───────────────────────────────────")

# creates model/ folder if it doesn't exist
os.makedirs('model', exist_ok=True)

# saves the Random Forest model as a .pkl file
# this file is loaded by the FastAPI backend to make live predictions
joblib.dump(rf_model, 'model/model.pkl')
print(f"   ✅ Random Forest saved to: model/model.pkl")

# also saves Logistic Regression for comparison
joblib.dump(lr_model, 'model/lr_model.pkl')
print(f"   ✅ Logistic Regression saved to: model/lr_model.pkl")

# ── 12. SAVE RESULTS TO JSON ─────────────────────────────────
# saves all results to a JSON file for documentation and README
results = {
    'logistic_regression': {**lr_results, 'training_time_seconds': round(lr_time, 2)},
    'random_forest': {**rf_results, 'training_time_seconds': round(rf_time, 2),
                      'cv_mean_f1': round(cv_scores.mean(), 4),
                      'cv_std_f1': round(cv_scores.std(), 4)},
    'best_model': 'random_forest',
    'training_rows': len(X_train),
    'testing_rows': len(X_test),
    'total_features': X.shape[1]
}

with open('model/model_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"   ✅ Results saved to: model/model_results.json")

# --- FINAL SUCCESS MESSAGE -----------------------------------
print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE — Ready for Explainability!")
print("=" * 60)
print(f"\n   Best Model  : Random Forest")
print(f"   Saved to    : model/model.pkl")