# ============================================================
# PhishGuard — Step 1: Data Cleaning & Feature Selection
# Dataset: StealthPhisher2025.csv
# Author: Uzman Zahid
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── 1. LOAD DATASET ─────────────────────────────────────────
print("=" * 60)
print("PHISHGUARD — DATA CLEANING SCRIPT")
print("=" * 60)

df = pd.read_csv('data/raw/StealthPhisher2025.csv')
print(f"\n Dataset loaded successfully")
print(f"   Rows    : {df.shape[0]:,}")
print(f"   Columns : {df.shape[1]}")

# ── 2. CHECK LABEL DISTRIBUTION ─────────────────────────────
print("\n── Label Distribution ──────────────────────────────────")
print(df['Label'].value_counts())
print(f"\n   Legitimate (0) : {(df['Label'] == 0).sum():,}")
print(f"   Phishing   (1) : {(df['Label'] == 1).sum():,}")

# ── 3. DROP HTML/PAGE-BASED COLUMNS (keep URL-only) ─────────
# These require loading the webpage — not suitable for URL-only thesis
html_columns = [
    'LineOfCode', 'LongestLineLength', 'HasTitle', 'HasFavicon',
    'HasRobotsBlocked', 'IsResponsive', 'IsURLRedirects', 'IsSelfRedirects',
    'HasDescription', 'HasPopup', 'HasIFrame', 'IsFormSubmitExternal',
    'HasSocialMediaPage', 'HasSubmitButton', 'HasHiddenFields',
    'HasPasswordFields', 'HasBankingKey', 'HasPaymentKey', 'HasCryptoKey',
    'HasCopyrightInfoKey ', 'CntImages', 'CntFilesCSS', 'CntFilesJS',
    'CntSelfHRef', 'CntEmptyRef', 'CntExternalRef', 'CntPopup',
    'CntIFrame', 'UniqueFeatureCnt', 'WAPLegitimate', 'WAPPhishing',
    'IsUnreachable'
]

df.drop(columns=html_columns, inplace=True)
print(f"\n Dropped {len(html_columns)} HTML/page-based columns")
print(f"   Remaining columns: {df.shape[1]}")

# ── 4. KEEP ONLY URL-BASED FEATURES ─────────────────────────
url_features = [
    'URL',
    'LengthOfURL',
    'Domain',
    'URLComplexity',
    'CharacterComplexity',
    'DomainLengthOfURL',
    'IsDomainIP',
    'TLD',
    'TLDLength',
    'LetterCntInURL',
    'URLLetterRatio',
    'DigitCntInURL',
    'URLDigitRatio',
    'EqualCharCntInURL',
    'QuesMarkCntInURL',
    'AmpCharCntInURL',
    'OtherSpclCharCntInURL',
    'URLOtherSpclCharRatio',
    'NumberOfHashtags',
    'NumberOfSubdomains',
    'HavingPath',
    'PathLength',
    'HavingQuery',
    'HavingFragment',
    'HavingAnchor',
    'HasSSL',
    'ShannonEntropy',
    'FractalDimension',
    'KolmogorovComplexity',
    'HexPatternCnt',
    'Base64PatternCnt',
    'LikelinessIndex',
    'Label'
]

df = df[url_features]
print(f"\n Kept {len(url_features)} URL-based features + Label")

# ── 5. CHECK FOR MISSING VALUES ──────────────────────────────
print("\n── Missing Values ──────────────────────────────────────")
missing = df.isnull().sum()
missing_cols = missing[missing > 0]

if len(missing_cols) == 0:
    print("No missing values found!")
else:
    print(f"Found missing values in {len(missing_cols)} columns:")
    print(missing_cols)
    df.dropna(inplace=True)
    print(f"Rows after dropping NaN: {len(df):,}")

# ── 6. CHECK FOR DUPLICATES ──────────────────────────────────
print("\n── Duplicate URLs ──────────────────────────────────────")
duplicates = df.duplicated(subset='URL').sum()
print(f"   Duplicate URLs found: {duplicates:,}")

if duplicates > 0:
    df.drop_duplicates(subset='URL', inplace=True)
    print(f" Duplicates removed. Rows remaining: {len(df):,}")
else:
    print(" No duplicates found!")

# ── 7. FINAL DATASET SUMMARY ────────────────────────────────
print("\n── Final Dataset Summary ───────────────────────────────")
print(f"   Total rows     : {len(df):,}")
print(f"   Total features : {df.shape[1] - 2} (excluding URL and Label)")
print(f"   Legitimate (0) : {(df['Label'] == 0).sum():,}")
print(f"   Phishing   (1) : {(df['Label'] == 1).sum():,}")

balance_pct = (df['Label'] == 1).sum() / len(df) * 100
print(f"   Phishing ratio : {balance_pct:.1f}%")

if 40 <= balance_pct <= 60:
    print("Dataset is well balanced — no resampling needed!")
else:
    print("Class imbalance detected — will use class_weight='balanced'")

# ── 8. SAVE CLEANED DATASET ─────────────────────────────────
os.makedirs('data/processed', exist_ok=True)
output_path = 'data/processed/stealthphisher_clean.csv'
df.to_csv(output_path, index=False)
print(f"\n Cleaned dataset saved to: {output_path}")

# ── 9. PLOT LABEL DISTRIBUTION ──────────────────────────────
plt.figure(figsize=(6, 4))
colors = ['#2ecc71', '#e74c3c']
df['Label'].value_counts().plot(
    kind='bar',
    color=colors,
    edgecolor='black',
    width=0.5
)
plt.title('Label Distribution — StealthPhisher2025', fontsize=13, fontweight='bold')
plt.xlabel('Label (0 = Legitimate, 1 = Phishing)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()

os.makedirs('data/processed', exist_ok=True)
plt.savefig('data/processed/label_distribution.png', dpi=150)
plt.show()
print(" Label distribution chart saved!")

print("\n" + "=" * 60)
print("CLEANING COMPLETE — Ready for Feature Engineering!")
print("=" * 60)