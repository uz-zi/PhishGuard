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

---