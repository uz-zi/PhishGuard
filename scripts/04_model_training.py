# ============================================================
# PhishGuard v3 — Step 4: Model Training & Evaluation
# Input  : data/processed/X_train.csv, y_train.csv
# Output : model/model.pkl, model/lr_model.pkl
#          model/model_results.json
# Author : Uzman Zahid
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

print("=" * 60)
print("PHISHGUARD — STEP 4: MODEL TRAINING")
print("=" * 60)

# ── 1. LOAD DATA ──────────────────────────────────────────────
X = pd.read_csv('data/processed/X_train.csv')
# converts the label file into a flat 1D array.
y = pd.read_csv('data/processed/y_train.csv').values.ravel()

print(f"\n Data loaded")
print(f"   X shape        : {X.shape}")
print(f"   Phishing   (1) : {(y==1).sum():,}")
print(f"   Legitimate (0) : {(y==0).sum():,}")

# ── 2. TRAIN/TEST SPLIT ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
# random_state=42 → ensures that the split is reproducible. Every time you run the code, you'll get the same train/test split, which is important for consistent results and debugging.
# stratify=y → ensures that the train and test sets have the same proportion of phishing and legitimate samples as the original dataset. This is important for imbalanced datasets to maintain representativeness in both sets.
print(f"\n── Train/Test Split ────────────────────────────────────")
print(f"   Training : {X_train.shape[0]:,} rows (70%)")
print(f"   Testing  : {X_test.shape[0]:,} rows (30%)")

# ── 3. LOGISTIC REGRESSION ────────────────────────────────────
print(f"\n── Training Logistic Regression ────────────────────────")
lr_start = time.time()
lr_model  = LogisticRegression(
    max_iter=1000, class_weight='balanced', random_state=42
)
lr_model.fit(X_train, y_train)
lr_time = time.time() - lr_start
# Uses the trained model to predict labels for the test set. so it contains the predicted labels (0 or 1) for each URL in the test set, which will be used for evaluation.
lr_pred = lr_model.predict(X_test)
print(f" Done in {lr_time:.1f}s")

# ── 4. RANDOM FOREST ──────────────────────────────────────────
print(f"\n── Training Random Forest ──────────────────────────────")
print(f"   May take 10-20 minutes...")
rf_start = time.time()
rf_model  = RandomForestClassifier(
    n_estimators=100, class_weight='balanced',
    n_jobs=-1, random_state=42
)
# The forest contains 100 decision trees.
# Balances class importance automatically.
# n_jobs Use all CPU cores available.
rf_model.fit(X_train, y_train)
rf_time = time.time() - rf_start
rf_pred = rf_model.predict(X_test)
print(f" Done in {rf_time:.1f}s")

# ── 5. EVALUATE ───────────────────────────────────────────────
print(f"\n── Model Evaluation ────────────────────────────────────")

def evaluate(name, y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred) # Overall percentage correct.
    prec = precision_score(y_true, y_pred) # Of all predicted phishing, how many were correct?
    rec  = recall_score(y_true, y_pred) # Of all actual phishing, how many did we catch?
    f1   = f1_score(y_true, y_pred) # Balance between precision and recall.
    print(f"\n   {name}")
    print(f"   {'─'*42}")
    print(f"   Accuracy  : {acc*100:.2f}%")
    print(f"   Precision : {prec*100:.2f}%")
    print(f"   Recall    : {rec*100:.2f}%")
    print(f"   F1-Score  : {f1*100:.2f}%")
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

lr_res = evaluate("Logistic Regression", y_test, lr_pred)
rf_res = evaluate("Random Forest",       y_test, rf_pred)

# ── 6. COMPARISON ─────────────────────────────────────────────
print(f"\n── Model Comparison ────────────────────────────────────")
print(f"   {'Metric':<12} {'LR':>10} {'RF':>10}")
print(f"   {'─'*35}")
for m in ['accuracy', 'precision', 'recall', 'f1']:
    w = 'ok' if rf_res[m] > lr_res[m] else '  '
    print(f"   {m:<12} {lr_res[m]*100:>9.2f}% {rf_res[m]*100:>9.2f}% {w}")

# ── 7. CROSS VALIDATION ───────────────────────────────────────
print(f"\n── Cross Validation (k=5) ──────────────────────────────")
# The dataset is split into 5 parts. model is tested 5 times, each time using a different part as the test set and the remaining 4 parts as the training set.
print(f"   Running...")
cv = cross_val_score(rf_model, X, y, cv=5, scoring='f1', n_jobs=-1)
print(f"   Folds  : {[f'{s:.4f}' for s in cv]}")
print(f"   Mean   : {cv.mean():.4f} (+/- {cv.std():.4f})")

# ── 8. CLASSIFICATION REPORT ──────────────────────────────────
print(f"\n── Classification Report — Random Forest ───────────────")
print(classification_report(y_test, rf_pred,
      target_names=['Legitimate', 'Phishing']))

# ── 9. CONFUSION MATRIX ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, pred, title in [
    (axes[0], lr_pred, 'Logistic Regression'),
    (axes[1], rf_pred, 'Random Forest')
]:
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legit', 'Phishing'],
                yticklabels=['Legit', 'Phishing'])
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('data/processed/confusion_matrices.png', dpi=150)
plt.show()
print(f"  Confusion matrices saved!")

# ── 10. FEATURE IMPORTANCE ────────────────────────────────────
# tells you which features the Random Forest used most for decision making.
imp_df = pd.DataFrame({
    'feature'   : X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n── Top 10 Feature Importances ──────────────────────────")
for _, row in imp_df.head(10).iterrows():
    print(f"   {row['feature']:<45} {row['importance']:.4f}")

plt.figure(figsize=(10, 9))
colors = ['#e74c3c' if v > 0.05 else '#3498db' if v > 0.02 else '#95a5a6'
          for v in imp_df['importance']]
plt.barh(imp_df['feature'][::-1], imp_df['importance'][::-1],
         color=colors[::-1], edgecolor='black', height=0.6)
plt.xlabel('Importance Score')
plt.title('Random Forest — Feature Importance', fontweight='bold')
plt.tight_layout()
plt.savefig('data/processed/feature_importance.png', dpi=150)
plt.show()
print(f" Feature importance saved!")

# ── 11. SAVE MODELS ───────────────────────────────────────────
os.makedirs('model', exist_ok=True)
joblib.dump(rf_model, 'model/model.pkl')
joblib.dump(lr_model, 'model/lr_model.pkl')
print(f"\n model/model.pkl saved")
print(f" model/lr_model.pkl saved")

results = {
    'dataset'             : 'Raw URLs 822k + Tranco 650k augmented',
    'label_convention'    : '1=phishing, 0=legitimate',
    'training_rows'       : len(X_train),
    'testing_rows'        : len(X_test),
    'total_features'      : X.shape[1],
    'logistic_regression' : {**lr_res, 'time': round(lr_time, 2)},
    'random_forest'       : {
        **rf_res,
        'time'       : round(rf_time, 2),
        'cv_mean_f1' : round(cv.mean(), 4),
        'cv_std_f1'  : round(cv.std(), 4)
    },
    'best_model' : 'random_forest'
}
with open('model/model_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f" model/model_results.json saved")

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE!")
print("=" * 60)
print(f"\n   Accuracy  : {rf_res['accuracy']*100:.2f}%")
print(f"   F1-Score  : {rf_res['f1']*100:.2f}%")
print(f"   CV Mean   : {cv.mean()*100:.2f}% (+/- {cv.std()*100:.2f}%)")
print(f"\n   Next: python scripts/05_test_url.py")