# ============================================================
# PhishGuard v3 — Step 3: Feature Extraction (Improved)
# Input  : data/processed/augmented_urls.csv
# Output : data/processed/clean_dataset.csv
#          data/processed/X_train.csv
#          data/processed/y_train.csv
#          model/features.json
#
# New features added:
#   - subdomain_depth: number of subdomain levels
#     e.g. m365.cloud.microsoft = depth 2
#          www.google.com        = depth 1
#          google.com            = depth 0
#
# Author : Uzman Zahid
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import re
import math
import os

print("=" * 60)
print("PHISHGUARD v3 — STEP 3: FEATURE EXTRACTION (IMPROVED)")
print("=" * 60)

# ── 1. LOAD AUGMENTED DATASET ────────────────────────────────
df = pd.read_csv('data/processed/augmented_urls.csv')
print(f"\n✅ Augmented dataset loaded")
print(f"   Rows       : {len(df):,}")
print(f"   Phishing   : {(df['status']==0).sum():,}")
print(f"   Legitimate : {(df['status']==1).sum():,}")

# ── 2. CONVERT LABELS ────────────────────────────────────────
df['label'] = df['status'].map({0: 1, 1: 0})
df.drop(columns=['status'], inplace=True)
print(f"\n✅ Labels converted: 1=phishing, 0=legitimate")

# ── 3. FEATURE EXTRACTION FUNCTION ──────────────────────────
def extract_features(raw_url):
    """
    Extracts 42 features from any URL format.
    NEW: subdomain_depth feature added.
    Works on bare domains, http://, https:// equally.
    """
    try:
        url = str(raw_url).strip()
        # makes sure the URL is treated as a string and removes extra spaces at the

        # ── scheme detection ──────────────────────────────────
        has_https = 1 if url.lower().startswith('https://') else 0
        has_http  = 1 if url.lower().startswith('http://') else 0

        # ── extract components ────────────────────────────────
        url_no_scheme = url
        #removes http:// or https:// so parsing becomes easier.
        if '://' in url:
            url_no_scheme = url.split('://', 1)[1]
        # take the part after the scheme, e.g. "www.google.com/search?q=abc#section" from "http://www.google.com/search?q=abc#section"    

        # isolates the domain part. isolates the domain part. to  mail.google.com:8080
        domain_with_port = url_no_scheme.split('/')[0].split('?')[0].split('#')[0]
        # removes any port number.
        domain_clean     = domain_with_port.split(':')[0]

        # separates the path from the domain. mail.google.com/u/0/ => path = /u/0/
        parts    = url_no_scheme.split('/', 1)
        path     = '/' + parts[1] if len(parts) > 1 else ''
        # extracts the query string. mail.google.com/search?q=abc#section => query = q=abc
        query    = url.split('?', 1)[1].split('#')[0] if '?' in url else ''
        # extracts the fragment part after #. mail.google.com/search?q=abc#section => fragment = section
        fragment = url.split('#', 1)[1] if '#' in url else ''

        # domain starts with www., this removes it.
        domain_no_www = domain_clean
        if domain_no_www.lower().startswith('www.'):
            domain_no_www = domain_no_www[4:]

        # makes sure the domain is treated as if it has www.
        if domain_clean.lower().startswith('www.'):
            domain_normalised = domain_clean
        else:
            domain_normalised = 'www.' + domain_clean

        # splits the normalized domain into pieces.
        domain_parts_full = domain_normalised.split('.')

        # ── URL features ──────────────────────────────────────
        url_length                    = len(url)
        number_of_dots_in_url         = url.count('.')
        # finds all digit characters in the URL.
        digits_in_url                 = re.findall(r'\d', url)
        # checks whether any digit repeats. 1234 → no repeated digits  1123 → repeated digit exists
        having_repeated_digits_in_url = 1 if len(digits_in_url) != len(
            set(digits_in_url)) and len(digits_in_url) > 0 else 0
        # set(digits_in_url) removes duplicates
        
        # Counts characters that are not letters or digits.
        number_of_digits_in_url       = sum(c.isdigit() for c in url)
        number_of_special_char_in_url = sum(not c.isalnum() for c in url)
        number_of_hyphens_in_url      = url.count('-')
        number_of_underline_in_url    = url.count('_')
        number_of_slash_in_url        = url.count('/')
        number_of_questionmark_in_url = url.count('?')
        number_of_equal_in_url        = url.count('=')
        number_of_at_in_url           = url.count('@')
        number_of_dollar_in_url       = url.count('$')
        number_of_exclamation_in_url  = url.count('!')
        number_of_hashtag_in_url      = url.count('#')
        number_of_percent_in_url      = url.count('%')

        # ── domain features ───────────────────────────────────
        # Length of the domain excluding www.
        domain_length                          = len(domain_no_www)
        number_of_dots_in_domain               = domain_no_www.count('.')
        number_of_hyphens_in_domain            = domain_no_www.count('-')
        # Checks whether the domain contains unusual characters.
        having_special_characters_in_domain    = 1 if re.search(
            r'[^a-zA-Z0-9\.\-]', domain_no_www) else 0
        # Counts how many unusual special characters are in the domain.
        number_of_special_characters_in_domain = sum(
            not c.isalnum() and c not in '.-' for c in domain_no_www)
        # Binary feature: does the domain contain digits?
        having_digits_in_domain = 1 if any(
            c.isdigit() for c in domain_no_www) else 0
        # Counts how many digit characters are in the domain.
        number_of_digits_in_domain = sum(
            c.isdigit() for c in domain_no_www)
        # Extracts all digits from the domain.
        digits_in_domain  = re.findall(r'\d', domain_no_www)
        # Checks whether any digit is repeated inside the domain.
        having_repeated_digits_in_domain = 1 if len(
            digits_in_domain) != len(set(digits_in_domain)) and \
            len(digits_in_domain) > 0 else 0

        # ── subdomain features ────────────────────────────────
        # This calculates how many pieces come before the main domain + TLD.
        number_of_subdomains = max(0, len(domain_parts_full) - 2)
        # This stores the actual subdomain parts.
        subdomains = domain_parts_full[:-2] if len(
            domain_parts_full) > 2 else []

        # measures how many levels of subdomains exist
        # www.google.com         → depth 1 (just www)
        # mail.google.com        → depth 1 (just mail)
        # m365.cloud.microsoft   → depth 2 (m365 + cloud)
        # myaccount.google.com   → depth 1
        # Phishing rarely has multi-level numeric subdomains
        subdomain_depth = len(subdomains)

        having_hyphen_in_subdomain = 1 if any(
            '-' in s for s in subdomains) else 0
        average_subdomain_length               = sum(
            len(s) for s in subdomains) / len(subdomains) \
            if subdomains else 0.0
        average_number_of_hyphens_in_subdomain = sum(
            s.count('-') for s in subdomains) / len(subdomains) \
            if subdomains else 0.0
        having_special_characters_in_subdomain = 1 if any(
            re.search(r'[^a-zA-Z0-9\-]', s)
            for s in subdomains) else 0
        # Counts those unusual characters across all subdomains.
        number_of_special_characters_in_subdomain = sum(
            sum(not c.isalnum() and c != '-' for c in s)
            for s in subdomains)
        # Checks whether any subdomain contains a digit.
        having_digits_in_subdomain = 1 if any(
            any(c.isdigit() for c in s) for s in subdomains) else 0
        # Counts total digits across all subdomains.
        number_of_digits_in_subdomain = sum(
            sum(c.isdigit() for c in s) for s in subdomains)
        # Joins all subdomains into one string and extracts all digits.
        all_sub_digits = re.findall(
            r'\d', ''.join(subdomains))
        # Checks whether any digit repeats across subdomains.
        having_repeated_digits_in_subdomain = 1 if len(
            all_sub_digits) != len(set(all_sub_digits)) and \
            len(all_sub_digits) > 0 else 0

        # ── path/query features ───────────────────────────────
        having_path    = 1 if len(path) > 1 else 0
        path_segments  = [p for p in path.split('/') if p]
        path_length    = len(path_segments)
        having_query   = 1 if len(query) > 0 else 0
        having_fragment = 1 if len(fragment) > 0 else 0
        having_anchor  = 1 if '#' in url else 0

        # ── entropy features ──────────────────────────────────
        # this computes Shannon entropy for the full URL.
        # set(url) gets unique characters
        # url.count(c) / len(url) computes probability of each character
        # -sum(p * log2(p)) computes entropy
        if len(url) > 0:
            prob_url       = [url.count(c) / len(url) for c in set(url)]
            entropy_of_url = -sum(p * math.log2(p)
                                  for p in prob_url if p > 0)
        else:
            entropy_of_url = 0.0

        if len(domain_no_www) > 0:
            prob_dom          = [domain_no_www.count(c) /
                                 len(domain_no_www)
                                 for c in set(domain_no_www)]
            entropy_of_domain = -sum(p * math.log2(p)
                                     for p in prob_dom if p > 0)
        else:
            entropy_of_domain = 0.0

        return {
            'has_https'                                : has_https,
            'has_http'                                 : has_http,
            'url_length'                               : url_length,
            'number_of_dots_in_url'                    : number_of_dots_in_url,
            'having_repeated_digits_in_url'            : having_repeated_digits_in_url,
            'number_of_digits_in_url'                  : number_of_digits_in_url,
            'number_of_special_char_in_url'            : number_of_special_char_in_url,
            'number_of_hyphens_in_url'                 : number_of_hyphens_in_url,
            'number_of_underline_in_url'               : number_of_underline_in_url,
            'number_of_slash_in_url'                   : number_of_slash_in_url,
            'number_of_questionmark_in_url'            : number_of_questionmark_in_url,
            'number_of_equal_in_url'                   : number_of_equal_in_url,
            'number_of_at_in_url'                      : number_of_at_in_url,
            'number_of_dollar_in_url'                  : number_of_dollar_in_url,
            'number_of_exclamation_in_url'             : number_of_exclamation_in_url,
            'number_of_hashtag_in_url'                 : number_of_hashtag_in_url,
            'number_of_percent_in_url'                 : number_of_percent_in_url,
            'domain_length'                            : domain_length,
            'number_of_dots_in_domain'                 : number_of_dots_in_domain,
            'number_of_hyphens_in_domain'              : number_of_hyphens_in_domain,
            'having_special_characters_in_domain'      : having_special_characters_in_domain,
            'number_of_special_characters_in_domain'   : number_of_special_characters_in_domain,
            'having_digits_in_domain'                  : having_digits_in_domain,
            'number_of_digits_in_domain'               : number_of_digits_in_domain,
            'having_repeated_digits_in_domain'         : having_repeated_digits_in_domain,
            'number_of_subdomains'                     : number_of_subdomains,
            'subdomain_depth'                          : subdomain_depth,
            'having_hyphen_in_subdomain'               : having_hyphen_in_subdomain,
            'average_subdomain_length'                 : average_subdomain_length,
            'average_number_of_hyphens_in_subdomain'   : average_number_of_hyphens_in_subdomain,
            'having_special_characters_in_subdomain'   : having_special_characters_in_subdomain,
            'number_of_special_characters_in_subdomain': number_of_special_characters_in_subdomain,
            'having_digits_in_subdomain'               : having_digits_in_subdomain,
            'number_of_digits_in_subdomain'            : number_of_digits_in_subdomain,
            'having_repeated_digits_in_subdomain'      : having_repeated_digits_in_subdomain,
            'having_path'                              : having_path,
            'path_length'                              : path_length,
            'having_query'                             : having_query,
            'having_fragment'                          : having_fragment,
            'having_anchor'                            : having_anchor,
            'entropy_of_url'                           : entropy_of_url,
            'entropy_of_domain'                        : entropy_of_domain,
        }
    except Exception:
        return None

# ── 4. EXTRACT FEATURES ──────────────────────────────────────
print(f"\n── Extracting Features ─────────────────────────────────")

features_list = []
failed        = 0

# itertuples() is faster than some other row-by-row methods.
for i, row in enumerate(df.itertuples(), 1):
    # This calls the extract_features function for each URL and collects the results in a list.
    feat = extract_features(row.url)
    if feat is not None:
        feat['label'] = row.label
        features_list.append(feat)
    else:
        failed += 1
    if i % 100000 == 0:
        print(f"   Processed {i:,} / {len(df):,} ...")

print(f"\n Done! Successful: {len(features_list):,}  Failed: {failed:,}")

# ── 5. CREATE DATAFRAME ──────────────────────────────────────
feat_df = pd.DataFrame(features_list)
print(f"\n Feature DataFrame: {feat_df.shape}")

# ── 6. DROP ZERO VARIANCE ────────────────────────────────────
#A zero-variance feature has the same value for every row. feature is useless for training because it contains no information. so it will be removed
variances = feat_df.drop(columns=['label']).var()
zero_var  = variances[variances == 0].index.tolist()
if zero_var:
    print(f"\n   Dropping zero-variance: {zero_var}")
    feat_df.drop(columns=zero_var, inplace=True)
else:
    print(f" No zero-variance features!")

# ── 7. SAVE FEATURES JSON ────────────────────────────────────
os.makedirs('model', exist_ok=True)
feature_names = [c for c in feat_df.columns if c != 'label']
with open('model/features.json', 'w') as f:
    json.dump(feature_names, f, indent=2)
print(f"\n model/features.json saved ({len(feature_names)} features)")

# ── 8. SAVE DATASET ──────────────────────────────────────────
os.makedirs('data/processed', exist_ok=True)
feat_df.to_csv('data/processed/clean_dataset.csv', index=False)
X = feat_df.drop(columns=['label'])
y = feat_df['label']
X.to_csv('data/processed/X_train.csv', index=False)
y.to_csv('data/processed/y_train.csv', index=False)
print(f" Saved X_train.csv, y_train.csv, clean_dataset.csv")

# ── 9. BALANCE CHECK ─────────────────────────────────────────
print(f"\n── Final Label Distribution ────────────────────────────")
print(y.value_counts())
pct = (y == 1).sum() / len(y) * 100
print(f"\n   Phishing ratio : {pct:.1f}%")

# ── 10. TOP CORRELATIONS ─────────────────────────────────────
print(f"\n── Top 10 Feature Correlations ─────────────────────────")
corr = X.corrwith(y).abs().sort_values(ascending=False)
print(corr.head(10).to_string())
