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