# ============================================================
# PhishGuard v5 — Step 1: Data Cleaning
# Dataset: new_data_urls.csv (822,010 URLs)
# Input  : data/raw/new_data_urls.csv
# Output : data/processed/clean_urls.csv
#
# Label convention:
#   status = 0 → PHISHING
#   status = 1 → LEGITIMATE
#
# Author : Uzman Zahid
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import os

print("=" * 60)
print("PHISHGUARD v5 — STEP 1: DATA CLEANING")
print("=" * 60)

# ── 1. LOAD DATASET ──────────────────────────────────────────
df = pd.read_csv('data/raw/new_data_urls.csv')
print(f"\nDataset loaded")
print(f"   Rows    : {len(df):,}")
print(f"   Columns : {df.columns.tolist()}")

# ── 2. CHECK LABELS ──────────────────────────────────────────
print(f"\n── Raw Label Distribution ──────────────────────────────")
print(df['status'].value_counts())
print(f"\n   status=0 : PHISHING")
print(f"   status=1 : LEGITIMATE")

# ── 3. REMOVE DUPLICATES ─────────────────────────────────────
print(f"\n── Removing Duplicates ─────────────────────────────────")
before = len(df)
df.drop_duplicates(subset='url', inplace=True)
df.dropna(subset=['url', 'status'], inplace=True)
print(f"   Removed  : {before - len(df):,}")
print(f"   Remaining: {len(df):,}")

# ── 4. BALANCE CHECK ─────────────────────────────────────────
phish = (df['status']==0).sum()
legit = (df['status']==1).sum()
total = len(df)
print(f"\n── Label Distribution ──────────────────────────────────")
print(f"   Phishing   (0): {phish:,} ({phish/total*100:.1f}%)")
print(f"   Legitimate (1): {legit:,} ({legit/total*100:.1f}%)")

# ── 5. SAVE ──────────────────────────────────────────────────
os.makedirs('data/processed', exist_ok=True)
df.to_csv('data/processed/clean_urls.csv', index=False)
print(f"\nSaved: data/processed/clean_urls.csv")

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
plt.savefig('data/processed/label_distribution.png', dpi=150)
plt.close()
print(f"Chart saved!")

print("\n" + "=" * 60)
print("STEP 1 COMPLETE!")
print(f"  Total URLs : {total:,}")
print(f"  Phishing   : {phish:,} ({phish/total*100:.1f}%)")
print(f"  Legitimate : {legit:,} ({legit/total*100:.1f}%)")
print("=" * 60)