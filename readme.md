## 📦 Dataset — Challenges & Decisions

This section documents the full dataset research process, every problem encountered, and the final decisions made.

---

### 🔴 Problem 1 — PhishTank Registration is Closed

The first and most obvious data source for any phishing detection project is **PhishTank** (phishtank.org).

**What happened:**
PhishTank permanently closed new user registrations due to platform abuse. Without a registered account and API key, the public download endpoint is rate-limited to almost nothing — our download attempt returned only **4 rows of data**, completely unusable for training a machine learning model.

**What tried:**
- Direct CSV download: `http://data.phishtank.com/data/online-valid.csv` → returned 4 rows
- Direct JSON feed → same result
- Checking for guest/anonymous workarounds → none available
- Waiting and retrying → no change

**Resolution:** PhishTank was ruled out entirely. The project moved to alternative sources that do not require registration.

---

### 🟡 Problem 2 — Kaggle Datasets Were Outdated

The next step was **Kaggle**, which hosts several popular phishing URL datasets frequently used in student projects and research papers.

**What happened:**
The most downloaded phishing datasets on Kaggle were found to be **2–3 years old**, with data collected in 2021–2022. Since phishing tactics, URL patterns, and domain strategies evolve extremely rapidly, training a model on URLs from 2021 would risk poor generalisation to current threats — directly weakening the research validity. Evaluating whether dataset age mattered for this research → concluded it significantly impacts thesis credibility for a 2025 submission

---

### ✅ Resolution — Two Datasets Selected from Mendeley Data (2025)

After evaluating multiple sources including the UCI ML Repository, OpenPhish, URLhaus, IEEE DataPort, and Mendeley Data, two datasets published in 2025 were selected from **Mendeley Data** — a peer-reviewed research data repository run by Elsevier.

---

#### 🏋️ Training Dataset — StealthPhisher (January 2025)

| Property | Detail |
|---|---|
| **Source** | Mendeley Data — `datasets/m2479kmybx/1` |
| **Published** | January 2025 |
| **Total URLs** | 336,749 |
| **Legitimate URLs** | 160,943 (47.8%) |
| **Phishing URLs** | 175,806 (52.2%) |
| **Original sources** | PhishTank, spam email repositories, user submissions |
| **Academic backing** | Published in *Expert Systems with Applications* (peer-reviewed) |

**Why this was chosen:**
- **Nearly balanced** (47/53 split) — no resampling or SMOTE required for training
- **Large scale** (336k rows) — sufficient for robust Random Forest generalisation
- **Peer-reviewed** — strong academic credibility for thesis citation
- **Recent** (2025) — reflects current phishing URL patterns and tactics

---

#### 🧪 Validation Dataset — URL-Phish (September 2025)

| Property | Detail |
|---|---|
| **Source** | Mendeley Data — `datasets/65z9twcx3r/1` |
| **Published** | September 2025 |
| **Total URLs** | 111,660 |
| **Legitimate URLs** | 100,000 (89.5%) |
| **Phishing URLs** | 11,660 (10.5%) |
| **Original sources** | PhishTank (Nov 2024 – Sept 2025) |
| **Pre-extracted features** | 22 lexical and structural URL features included |

**Why this was chosen:**
- **Completely independent** from the training dataset — zero URL overlap
- **Most recent available** (September 2025) — freshest phishing samples
- **Used only for final blind validation** — never seen during any training phase
- **Class imbalance noted** (89/11) — handled with `class_weight='balanced'`

---

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Data Science | Python, Pandas, NumPy | Data loading, cleaning, feature extraction |
| Machine Learning | Scikit-learn | Logistic Regression, Random Forest, evaluation metrics |
| Explainability | SHAP, LIME | Feature importance, per-prediction explanations |
| Visualisation | Matplotlib, Seaborn | Confusion matrix, SHAP summary plots |
| Backend API | FastAPI, Uvicorn | Serve live predictions via REST endpoint |
| Model Storage | Joblib | Serialise and load trained model (.pkl) |
| Chrome Extension | HTML, CSS, JavaScript | Popup UI, background service worker, content script |
| Extension Standard | Manifest V3 | Current Chrome extension API standard |
| Hosting | Render.com / Railway | Free cloud deployment for the API |
| IDE | VS Code + Jupyter Notebook | Development and model training environment |
| Version Control | GitHub | Code storage and project tracking |

---

## 🚀 Getting Started

### Prerequisites

```bash
python --version
pip --version
```

### Install Python Dependencies

```bash
pip install pandas numpy scikit-learn shap lime matplotlib seaborn joblib fastapi uvicorn imbalanced-learn
```

### Download Datasets

1. **StealthPhisher (Training):** https://data.mendeley.com/datasets/m2479kmybx/1
2. **URL-Phish (Blind Validation):** https://data.mendeley.com/datasets/65z9twcx3r/1

Place both downloaded CSV files inside the `data/raw/` folder. Do **not** open or pre-process the URL-Phish dataset until model training is complete.


# 🧹 Phase 2 — Data Cleaning

> **Scripts:** `01_data_cleaning.py` → `02_feature_engineering.py`
> **Goal:** Take the raw StealthPhisher dataset and prepare it for machine learning model training.

---

## 📋 Overview

Before training any machine learning model, the raw dataset needs to be cleaned and prepared. This phase covers two scripts that transform the original 65-column raw CSV into a clean, numeric, ML-ready feature matrix.

---

## 📁 Files Involved

| File | Type | Description |
|---|---|---|
| `data/raw/StealthPhisher2025.csv` | Input | Original downloaded dataset — never modified |
| `data/processed/stealthphisher_clean.csv` | Output | Cleaned dataset with URL-only features |
| `data/processed/X_train.csv` | Output | Feature matrix ready for model training |
| `data/processed/y_train.csv` | Output | Label column (0 = Legitimate, 1 = Phishing) |
| `model/features.json` | Output | Feature names in exact order — critical for API |
| `data/processed/feature_correlations.png` | Output | Top 15 features ranked by importance |
| `data/processed/feature_distributions.png` | Output | Top 6 feature distributions phishing vs legitimate |

---

## 📜 Script 01 — Data Cleaning (`01_data_cleaning.py`)

### What it does

Loads the raw StealthPhisher2025 dataset and prepares a clean, URL-only version ready for feature engineering.

### Steps

**Step 1 — Load Dataset**
Loads `StealthPhisher2025.csv` from `data/raw/`:
```
Rows    : 336,749
Columns : 65
```

**Step 2 — Check Label Distribution**
Checks how many phishing vs legitimate URLs exist before any changes.
At this point the Label column contains text values (`"Phishing"` / `"Legitimate"`).
```
Phishing      : 175,804
Legitimate    : 160,943
```

**Step 3 — Drop HTML/Page-Based Columns**
Drops 32 columns that require loading the actual webpage to compute — these are not suitable for a URL-only detection system as per the thesis methodology:
```
LineOfCode, HasTitle, HasFavicon, HasPopup, HasIFrame,
CntImages, CntFilesCSS, CntFilesJS, WAPLegitimate, WAPPhishing ...
```
These features would make the extension slow, privacy-invasive, and unsuitable for real-time detection.

**Step 4 — Keep URL-Only Features**
Keeps only the 32 URL-based features aligned with the thesis research question:
```
LengthOfURL, URLComplexity, CharacterComplexity, IsDomainIP,
HasSSL, ShannonEntropy, KolmogorovComplexity, NumberOfSubdomains ...
```

**Step 5 — Check for Missing Values**
Scans all columns for empty/null values and removes affected rows if found.
```
Result: ✅ No missing values found
```

**Step 6 — Check for Duplicate URLs**
Removes any rows where the same URL appears more than once to avoid training bias.
```
Result: ✅ No duplicates found
```

**Step 7 — Final Dataset Summary**
Prints a summary of the clean dataset before saving.
```
Total rows     : 336,747
Total features : 31 (excluding URL and Label)
Legitimate     : 160,943
Phishing       : 175,804
Phishing ratio : 52.2%  ✅ Well balanced
```

**Step 8 — Convert Label to Numbers**
ML models require numeric labels — converts text to integers:
```
"Phishing"   → 1
"Legitimate" → 0
```
This conversion happens AFTER all text-based checks in Steps 2 and 7 to avoid comparison errors.

**Step 9 — Save Cleaned Dataset**
Saves the cleaned dataset to `data/processed/stealthphisher_clean.csv`.

### ⚠️ Issues Encountered & Fixed

**Issue — Label column was text not numbers**
The original dataset used `"Phishing"` and `"Legitimate"` as string labels. Running the script without fixing this would cause Steps 2 and 7 (which compared `== 0` and `== 1`) to return 0 for both classes — silently giving wrong counts.

**Fix:** Changed all label comparisons in Steps 2 and 7 to use text strings (`== 'Phishing'`, `== 'Legitimate'`). Then added `.map({'Phishing': 1, 'Legitimate': 0})` in Step 8 to convert to numbers before saving.

**Issue — `HasCopyrightInfoKey ` had a trailing space**
One column name had a hidden trailing space in the original dataset. This was handled by including the exact string with the space in the `html_columns` drop list.

### Run Command
```powershell
python scripts/01_data_cleaning.py
```

# Phase 3 — Model Training & Evaluation

> **Script:** `03_model_training.py`
> **Goal:** Train Logistic Regression and Random Forest models on 336,747 URLs and evaluate their performance in detecting phishing websites using URL-based features only.

---

## 📋 Overview

This phase trains two machine learning models on the cleaned and engineered dataset from Phase 2. The models learn to classify URLs as phishing (1) or legitimate (0) based on 29 URL structural features. Both models are evaluated using standard classification metrics and compared to identify the best performer for deployment.

---

## 📁 Files Involved

| File | Type | Description |
|---|---|---|
| `data/processed/X_train.csv` | Input | Feature matrix (336,747 rows × 29 features) |
| `data/processed/y_train.csv` | Input | Label column (0 = Legitimate, 1 = Phishing) |
| `model/model.pkl` | Output | Trained Random Forest model — used by API |
| `model/lr_model.pkl` | Output | Trained Logistic Regression model — baseline |
| `model/model_results.json` | Output | All evaluation metrics saved as JSON |
| `data/processed/confusion_matrices.png` | Output | Side-by-side confusion matrix charts |
| `data/processed/feature_importance.png` | Output | Random Forest feature importance chart |

---

## 📜 Script — Model Training (`03_model_training.py`)

### Steps

**Step 1 — Load Data**
Loads `X_train.csv` and `y_train.csv` from `data/processed/`.
Converts y from a 2D column to a 1D array using `.values.ravel()` as required by scikit-learn.
```
X shape : (336,747, 29)
y shape : (336,747,)
```

**Step 2 — Train/Test Split**
Splits the dataset into 70% training and 30% testing:
```
Training set : 235,722 rows (70%)
Testing set  : 101,025 rows (30%)
```
`stratify=y` ensures both sets have the same phishing/legitimate ratio.
`random_state=42` makes the split reproducible.

**Step 3 — Train Logistic Regression (Baseline)**
Trains a simple linear classifier as the baseline model:
```python
LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
```
`class_weight='balanced'` handles any minor class imbalance automatically.
`n_jobs=-1` uses all CPU cores for maximum speed.

**Step 4 — Train Random Forest (Main Model)**
Trains an ensemble of 100 decision trees:
```python
RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
```
Random Forest combines votes from 100 trees to make each prediction — significantly more robust than a single decision tree or linear model.

**Step 5 — Evaluate Both Models**
Both models are evaluated on the 30% test set using 4 standard metrics:
- **Accuracy** — overall percentage of correct predictions
- **Precision** — of all URLs flagged as phishing, how many were actually phishing
- **Recall** — of all actual phishing URLs, how many did the model catch
- **F1-Score** — harmonic mean of precision and recall

**Step 6 — Cross Validation**
5-fold cross validation is applied to the Random Forest model to confirm it generalises well and is not overfitting to the training data.

**Step 7 — Save Models**
Both models are saved as `.pkl` files using joblib. The Random Forest model is the primary model loaded by the FastAPI backend.

---

## 📊 Results

### Model Performance on Test Set (101,025 URLs)

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| **Accuracy** | 99.72% | **99.76%** ✅ |
| **Precision** | **99.97%** | 99.91% |
| **Recall** | 99.49% | **99.63%** ✅ |
| **F1-Score** | 99.73% | **99.77%** ✅ |

### Cross Validation — Random Forest (k=5)

| Fold | F1-Score |
|---|---|
| Fold 1 | 0.9980 |
| Fold 2 | 0.9981 |
| Fold 3 | 0.9978 |
| Fold 4 | 0.9978 |
| Fold 5 | 0.9979 |
| **Mean** | **0.9979** |
| **Std Dev** | **±0.0001** |

The extremely low standard deviation (±0.0001) confirms the model is highly consistent across all folds — no overfitting detected.

### Classification Report — Logistic Regression

```
              precision    recall  f1-score   support
  Legitimate       0.99      1.00      1.00     48,283
    Phishing       1.00      0.99      1.00     52,742
    accuracy                           1.00    101,025
   macro avg       1.00      1.00      1.00    101,025
weighted avg       1.00      1.00      1.00    101,025
```

### Classification Report — Random Forest

```
              precision    recall  f1-score   support
  Legitimate       1.00      1.00      1.00     48,283
    Phishing       1.00      1.00      1.00     52,742
    accuracy                           1.00    101,025
   macro avg       1.00      1.00      1.00    101,025
weighted avg       1.00      1.00      1.00    101,025
```

---

## How Good Are These Results?

To put the results in context:

| System | Accuracy |
|---|---|
| Most published phishing detection papers | 92–96% |
| Traditional blacklist-based tools | ~70–80% |
| **This project — Logistic Regression** | **99.72%** |
| **This project — Random Forest** | **99.76%** |

Both models significantly exceed the 90% accuracy target set at the start of this research and outperform most results reported in the existing literature.

---

## 🔍 Why Random Forest Was Selected as the Primary Model

Random Forest outperforms Logistic Regression on 3 out of 4 metrics:
- Higher accuracy (99.76% vs 99.72%)
- Higher recall (99.63% vs 99.49%) — catches more real phishing URLs
- Higher F1-score (99.77% vs 99.73%)

While Logistic Regression achieves slightly higher precision (99.97% vs 99.91%), recall is the more critical metric for a security tool — missing a real phishing URL (false negative) is more dangerous than occasionally flagging a safe URL (false positive).

Random Forest is therefore selected as the primary model and saved to `model/model.pkl` for API deployment.

---

## 🖥️ Training Environment

| Property | Detail |
|---|---|
| **CPU** | Intel Core i5-8250U @ 1.60GHz (4 cores / 8 threads) |
| **RAM** | 16GB |
| **Storage** | 238GB SSD |
| **OS** | Windows 10 64-bit |
| **Python** | 3.12.1 |
| **Scikit-learn** | latest |
| **Training rows** | 235,722 |
| **Testing rows** | 101,025 |
| **Features** | 29 URL-based features |

---

## ✅ Phase 3 Outcome

| Task | Result |
|---|---|
| Logistic Regression trained | ✅ 99.72% accuracy |
| Random Forest trained | ✅ 99.76% accuracy |
| Cross validation passed | ✅ 99.79% mean F1 (±0.0001) |
| No overfitting detected | ✅ Confirmed |
| Best model saved | ✅ model/model.pkl |
| Results documented | ✅ model/model_results.json |
| Confusion matrices generated | ✅ data/processed/confusion_matrices.png |
| Feature importance generated | ✅ data/processed/feature_importance.png |

---