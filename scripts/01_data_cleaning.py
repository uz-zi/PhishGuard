# ============================================================
# PhishGuard v3 — Step 1: Data Cleaning
# Dataset: new_data_urls.csv (822,010 URLs)
# Input  : data/raw/new_data_urls.csv
# Output : data/processed/clean_urls.csv
#
# Label convention (from dataset author):
#   status = 0 → PHISHING
#   status = 1 → LEGITIMATE
#
# Author : Uzman Zahid
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import os

print("=" * 60)
print("PHISHGUARD v3 — STEP 1: DATA CLEANING")
print("=" * 60)

# ── 1. LOAD DATASET ──────────────────────────────────────────
df = pd.read_csv('data/raw/new_data_urls.csv')
print(f"\n✅ Dataset loaded")
# Shows number of rows
print(f"   Rows    : {len(df):,}")
# Displays column names as a list
print(f"   Columns : {df.columns.tolist()}")

# ── 2. CHECK LABELS ──────────────────────────────────────────
print(f"\n── Raw Label Distribution ──────────────────────────────")
# Counts how many: 0 (phishing) 1 (legitimate)
print(df['status'].value_counts())
print(f"\n   status=0 : PHISHING   (per dataset author)")
print(f"   status=1 : LEGITIMATE (per dataset author)")

# ── 3. REMOVE DUPLICATES ─────────────────────────────────────
print(f"\n── Removing Duplicates ─────────────────────────────────")
# Saves number of rows before cleaning
before = len(df)
#Removes duplicate URLs
#subset='url' → only checks duplicates in the url column
#inplace=True → modifies df directly (no copy)
df.drop_duplicates(subset='url', inplace=True)
print(f"   Removed  : {before - len(df):,}")
print(f"   Remaining: {len(df):,}")

# ── 4. DROP MISSING ──────────────────────────────────────────
# Removes rows where: url is missing OR status is missing
df.dropna(subset=['url', 'status'], inplace=True)
print(f"   After dropna: {len(df):,}")

# ── 5. SAVE ──────────────────────────────────────────────────
os.makedirs('data/processed', exist_ok=True)
# index=False: avoids saving row numbers
df.to_csv('data/processed/clean_urls.csv', index=False)
print(f"\n✅ Saved: data/processed/clean_urls.csv")

# ── 6. PLOT ──────────────────────────────────────────────────
plt.figure(figsize=(6, 4))
df['status'].value_counts().sort_index().plot(
    kind='bar', color=['#e74c3c', '#2ecc71'],
    edgecolor='black', width=0.5
)
plt.title('Label Distribution — Raw Dataset',
          fontsize=13, fontweight='bold')
plt.xlabel('Status (0=Phishing, 1=Legitimate)')
plt.ylabel('Count')
plt.xticks([0, 1], ['Phishing (0)', 'Legitimate (1)'], rotation=0)
plt.tight_layout()
plt.savefig('data/processed/raw_distribution.png', dpi=150)
plt.show()
print(f"Chart saved!")