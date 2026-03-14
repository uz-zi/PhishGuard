# ============================================================
# PhishGuard — Step 2: Feature Engineering
# Input  : data/processed/stealthphisher_clean.csv
# Output : data/processed/X_train.csv, y_train.csv
#          model/features.json
# Author : Uzman Zahid
# ============================================================

# --- IMPORT LIBRARIES ----------------------------------------
import pandas as pd        # for loading and manipulating the dataset (tables/dataframes)
import numpy as np         # for numerical operations (checking variance, correlations)
import matplotlib.pyplot as plt  # for creating charts and saving them as images
import seaborn as sns      # for better looking charts (built on top of matplotlib)
import json                # for saving feature names as a .json file (used by API later)
import os                  # for creating folders if they don't exist

# --- PRINT HEADER --------------------------------------------
print("=" * 60)            # prints a divider line of 60 = signs
print("PHISHGUARD — FEATURE ENGINEERING SCRIPT")  # prints script title
print("=" * 60)            # prints another divider line

# ── 1. LOAD CLEANED DATASET ──────────────────────────────────
# reads the cleaned CSV file we saved in Script 01 into a pandas DataFrame
df = pd.read_csv('data/processed/stealthphisher_clean.csv')

# prints confirmation that the file loaded successfully
print(f"\n✅ Cleaned dataset loaded")

# prints total number of rows in the dataset
print(f"   Rows    : {len(df):,}")

# prints total number of columns in the dataset
print(f"   Columns : {df.shape[1]}")

# ── 2. DROP TEXT COLUMNS ─────────────────────────────────────
# these three columns are text-based — ML models need numbers only
# URL    = the actual website address (text, not useful as a number)
# Domain = the domain name extracted from URL (text)
# TLD    = top level domain like .com .org .net (text)
drop_cols = ['URL', 'Domain', 'TLD']

# removes the three text columns from the dataframe permanently
df.drop(columns=drop_cols, inplace=True)

# confirms which columns were dropped
print(f"\n✅ Dropped non-numeric columns: {drop_cols}")

# ── 3. SEPARATE FEATURES (X) AND LABEL (y) ───────────────────
# X = feature matrix — everything EXCEPT the Label column
# this is what the ML model will learn FROM
X = df.drop(columns=['Label'])

# y = label column only — 0 = legitimate, 1 = phishing
# this is what the ML model will learn TO predict
y = df['Label']

# prints the shape of X — should be (336749, number_of_features)
print(f"\n── Feature Matrix ──────────────────────────────────────")
print(f"   X shape : {X.shape}  (rows x features)")

# prints the shape of y — should be (336749,)
print(f"   y shape : {y.shape}  (labels)")

# prints the list of all feature column names
print(f"\n   Features: {X.columns.tolist()}")

# ── 4. CHECK DATA TYPES ───────────────────────────────────────
print(f"\n── Data Types ──────────────────────────────────────────")

# finds any columns that are NOT numeric (int or float)
# ML models cannot process text columns directly
non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()

# if any non-numeric columns are found, drop them
if non_numeric:
    # warns us which columns are non-numeric
    print(f"   ⚠️  Non-numeric columns found: {non_numeric}")

    # removes those non-numeric columns from X
    X.drop(columns=non_numeric, inplace=True)

    # prints how many features remain after dropping
    print(f"   Dropped. Remaining features: {X.shape[1]}")
else:
    # all columns are numeric — no action needed
    print(f"   ✅ All features are numeric — no encoding needed!")

# ── 5. CHECK FOR ZERO-VARIANCE FEATURES ──────────────────────
print(f"\n── Zero-Variance Features ──────────────────────────────")

# calculates the variance of every column
# variance = how much the values in a column change between rows
# if variance = 0, every row has the exact same value → useless for ML
variances = X.var()

# finds columns where variance is exactly 0
zero_var = variances[variances == 0].index.tolist()

# if zero variance columns exist, drop them
if zero_var:
    # warns us which columns have zero variance
    print(f"Zero-variance features found (useless for ML): {zero_var}")

    # removes those zero-variance columns from X
    X.drop(columns=zero_var, inplace=True)

    # prints how many features remain
    print(f"Dropped. Remaining features: {X.shape[1]}")
else:
    # all features have some variation — good to keep
    print(f"No zero-variance features found!")

# ── 6. CALCULATE FEATURE CORRELATIONS WITH LABEL ─────────────
print(f"\n── Feature Correlation with Label ──────────────────────")

# calculates how strongly each feature in X is correlated with y (the label)
# .abs() takes the absolute value — we care about strength not direction
# .sort_values(ascending=False) sorts strongest to weakest
correlations = X.corrwith(y).abs().sort_values(ascending=False)

# prints the top 10 most important features (strongest link to phishing)
print(f"\n   Top 10 most important features:")
print(correlations.head(10).to_string())

# prints the 5 weakest features (least useful for detecting phishing)
print(f"\n   Bottom 5 least important features:")
print(correlations.tail(5).to_string())

# ── 7. DROP WEAK FEATURES ────────────────────────────────────
# finds features with very weak correlation (less than 0.01)
# these features add noise without helping the model predict phishing
weak_features = correlations[correlations < 0.01].index.tolist()

# if weak features exist, drop them
if weak_features:
    # warns us which features are too weak to be useful
    print(f"\n Dropping weak features (correlation < 0.01): {weak_features}")

    # removes weak features from X
    X.drop(columns=weak_features, inplace=True)

    # prints how many features remain after dropping weak ones
    print(f" Remaining features: {X.shape[1]}")
else:
    # all features have meaningful correlation — nothing to drop
    print(f"\n All features have meaningful correlation with label!")

# ── 8. FINAL FEATURE SUMMARY ─────────────────────────────────
print(f"\n── Final Feature Set ───────────────────────────────────")

# prints total number of features that made it through all checks
print(f"   Total features : {X.shape[1]}")

# prints the final list of feature names used for training
print(f"   Feature names  : {X.columns.tolist()}")

# ── 9. SAVE FEATURE NAMES TO JSON ────────────────────────────
# creates the model/ folder if it doesn't already exist
os.makedirs('model', exist_ok=True)

# saves the list of feature column names in the exact order they appear in X
# THIS IS CRITICAL — the API must extract features in the exact same order
# if the order is wrong, every prediction the API makes will be incorrect
feature_names = X.columns.tolist()

# opens features.json and writes the feature names as a JSON list
# indent=2 makes the file human readable (nicely formatted)
with open('model/features.json', 'w') as f:
    json.dump(feature_names, f, indent=2)

# confirms the file was saved
print(f"\n Feature names saved to: model/features.json")
print(f"   (This file is CRITICAL — the API uses it to align features)")

# ── 10. SAVE X AND y AS CSV FILES ────────────────────────────
# creates data/processed/ folder if it doesn't already exist
os.makedirs('data/processed', exist_ok=True)

# saves the feature matrix X as a CSV — used in Script 03 for model training
# index=False means don't save the row numbers (0,1,2...) as a column
X.to_csv('data/processed/X_train.csv', index=False)

# saves the label column y as a CSV — used in Script 03 for model training
y.to_csv('data/processed/y_train.csv', index=False)

# confirms both files were saved successfully
print(f"X saved to: data/processed/X_train.csv")
print(f"y saved to: data/processed/y_train.csv")

# ── 11. CHART 1 — TOP 15 FEATURE CORRELATIONS ────────────────
# takes only the top 15 most correlated features for the chart
top_features = correlations.head(15)

# creates a figure 10 inches wide and 6 inches tall
plt.figure(figsize=(10, 6))

# assigns bar colors based on how strong the correlation is:
# red  = very strong (> 0.3) — most important for detecting phishing
# blue = moderate   (> 0.1) — somewhat important
# grey = weak       (≤ 0.1) — less important
colors = ['#e74c3c' if v > 0.3 else '#3498db' if v > 0.1 else '#95a5a6'
          for v in top_features.values]

# creates a horizontal bar chart
# [::-1] reverses the list so the strongest feature appears at the TOP of the chart
bars = plt.barh(top_features.index[::-1], top_features.values[::-1],
                color=colors[::-1], edgecolor='black', height=0.6)

# labels the x-axis
plt.xlabel('Absolute Correlation with Label', fontsize=12)

# sets the chart title
plt.title('Top 15 Features — Correlation with Phishing Label',
          fontsize=13, fontweight='bold')

# adds a vertical red dashed line at x=0.3 to visually mark "strong" threshold
plt.axvline(x=0.3, color='red', linestyle='--', alpha=0.5, label='Strong (>0.3)')

# adds a vertical blue dashed line at x=0.1 to mark "moderate" threshold
plt.axvline(x=0.1, color='blue', linestyle='--', alpha=0.5, label='Moderate (>0.1)')

# shows the legend explaining what the colored lines mean
plt.legend()

# adjusts spacing so labels and title don't get cut off
plt.tight_layout()

# saves the chart as a PNG image at 150 DPI (good quality)
plt.savefig('data/processed/feature_correlations.png', dpi=150)

# displays the chart on screen in a popup window
plt.show()

# confirms the chart image was saved
print(f" Feature correlation chart saved!")

# ── 12. CHART 2 — TOP 6 FEATURE DISTRIBUTIONS ────────────────
# gets the names of the top 6 most correlated features
top6 = correlations.head(6).index.tolist()

# creates a grid of 6 subplots — 2 rows and 3 columns, 15x8 inches total
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# flattens the 2D grid [[ax1,ax2,ax3],[ax4,ax5,ax6]] into a flat list
# so we can loop through them easily with a single index
axes = axes.flatten()

# loops through each of the top 6 features one at a time
for i, feature in enumerate(top6):

    # plots a histogram of this feature for LEGITIMATE URLs (label == 0)
    # bins=50 = splits values into 50 buckets
    # alpha=0.6 = 60% opacity so bars overlap visibly
    # density=True = normalises height so both classes are comparable
    axes[i].hist(X[y == 0][feature], bins=50, alpha=0.6,
                 color='#2ecc71', label='Legitimate', density=True)

    # plots a histogram of this feature for PHISHING URLs (label == 1)
    axes[i].hist(X[y == 1][feature], bins=50, alpha=0.6,
                 color='#e74c3c', label='Phishing', density=True)

    # sets the title of this subplot to the feature name
    axes[i].set_title(feature, fontweight='bold')

    # labels the x-axis of this subplot
    axes[i].set_xlabel('Value')

    # labels the y-axis of this subplot
    axes[i].set_ylabel('Density')

    # shows the green/red legend on each subplot
    axes[i].legend(fontsize=8)

# sets one big title above all 6 subplots
# y=1.02 pushes the title slightly above the charts so it doesn't overlap
plt.suptitle('Top 6 Feature Distributions — Phishing vs Legitimate',
             fontsize=14, fontweight='bold', y=1.02)

# adjusts spacing between the 6 subplots
plt.tight_layout()

# saves all 6 charts as one PNG image
# bbox_inches='tight' makes sure the overall title doesn't get cut off
plt.savefig('data/processed/feature_distributions.png', dpi=150,
            bbox_inches='tight')

# displays all 6 charts on screen
plt.show()

# confirms the chart was saved
print(f"Feature distribution chart saved!")

# --- FINAL SUCCESS MESSAGE -----------------------------------
print("\n" + "=" * 60)       # prints divider line
print("FEATURE ENGINEERING COMPLETE — Ready for Model Training!")
print("=" * 60)              # prints divider line