# ============================================================
# PhishGuard v5 — Step 3: Feature Extraction (48 Hybrid Features)
# Input  : data/processed/augmented_urls.csv
# Output : data/processed/clean_dataset.csv
#          data/processed/X_train.csv
#          data/processed/y_train.csv
#          model/features.json
#
# Feature Engineering Approach: HYBRID
#   42 original structural features (unchanged)
#    + 6 new features:
#       2 knowledge-based (justified in thesis):
#         43. has_suspicious_tld     — known bad TLDs
#         44. is_known_safe_sld      — known safe brands
#       4 pure structural (no hardcoding):
#         45. consonant_vowel_ratio  — unnatural domains
#         46. longest_digit_sequence — digit runs
#         47. digit_letter_ratio     — digit heavy domains
#         48. path_to_url_ratio      — bare domain detection
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
print("PHISHGUARD — STEP 3: FEATURE EXTRACTION (48 HYBRID)")
print("=" * 60)

# ── 1. LOAD DATASET ──────────────────────────────────────────
df = pd.read_csv('data/processed/augmented_urls.csv')
print(f"\nAugmented dataset loaded")
print(f"   Rows       : {len(df):,}")
# Return Length with format like # "1,500"
print(f"   Phishing   : {(df['status']==0).sum():,}")
print(f"   Legitimate : {(df['status']==1).sum():,}")

# ── 2. CONVERT LABELS ────────────────────────────────────────
df['label'] = df['status'].map({0: 1, 1: 0})
df.drop(columns=['status'], inplace=True)
# removes old column
print(f"\nLabels: 1=phishing, 0=legitimate")

# ── 3. KNOWLEDGE LOOKUP TABLES ───────────────────────────────

SUSPICIOUS_TLDS = {
    # Free TLDs massively abused for phishing
    'tk', 'ml', 'ga', 'cf', 'gq',
    # Cheap TLDs with high phishing rates
    'xyz', 'top', 'club', 'online', 'site',
    'fun', 'icu', 'vip', 'cyou', 'lat',
    'space', 'live', 'pw', 'cc', 'su',
    'ws', 'bz', 'name', 'mobi', 'link',
    'click', 'download', 'loan', 'win',
    'racing', 'stream', 'trade', 'review',
    'accountant', 'science', 'work', 'party',
    'faith', 'date', 'cricket', 'ninja',
    'bid', 'webcam', 'rocks', 'country',
}

KNOWN_SAFE_SLDS = {
    # Major tech companies
    'google', 'microsoft', 'apple', 'amazon',
    'facebook', 'youtube', 'netflix', 'github',
    'stackoverflow', 'wikipedia', 'twitter',
    'linkedin', 'instagram', 'spotify', 'stripe',
    'paypal', 'dropbox', 'slack', 'zoom',
    'cloudflare', 'netlify', 'vercel', 'heroku',
    'reddit', 'pinterest', 'tiktok', 'discord',
    'notion', 'figma', 'canva', 'shopify',
    'wordpress', 'adobe', 'salesforce', 'hubspot',
    'openai', 'anthropic', 'huggingface', 'kaggle',
    'coursera', 'udemy', 'edx', 'khanacademy',
    'duolingo', 'brilliant', 'pluralsight',
    # News/Media
    'bbc', 'cnn', 'reuters', 'bloomberg',
    'techcrunch', 'theguardian', 'nytimes',
    # Universities
    'mit', 'stanford', 'harvard', 'oxford',
    'cambridge', 'dbs', 'ucd', 'tcd', 'dcu',
    # Developer platforms
    'npmjs', 'pypi', 'kubernetes', 'mozilla',
    'docker', 'digitalocean', 'linode',
    # Tools
    'spinbot', 'ilovepdf', 'smallpdf', 'mp3cut',
    'convertio', 'tinypng', 'grammarly', 'canva',
    # Finance
    'wise', 'revolut', 'coinbase', 'binance',
    # Streaming
    'vimeo', 'twitch', 'dailymotion', 'soundcloud',
    # Government
    'gov', 'nasa', 'nih',
}

# ── 4. FEATURE EXTRACTION ────────────────────────────────────
def extract_features(raw_url):
    """
    Extracts 48 hybrid features:
    42 structural + 2 knowledge-based + 4 pure structural
    """
    try:
        # step A: Pre processing
        # cleans url and convert it to string
        url = str(raw_url).strip() 

        has_https = 1 if url.lower().startswith('https://') else 0
        has_http  = 1 if url.lower().startswith('http://') else 0

        url_no_scheme = url
        #removes http:// or https:// so parsing becomes easier.
        if '://' in url:
            url_no_scheme = url.split('://', 1)[1]
        # remove https part https://google.com → google.com

        # Step B: Split URL
        #https://example.com:8080/path/page?search=abc#section1
        # isolates the domain part. isolates the domain part. to  mail.google.com:8080
        domain_with_port = url_no_scheme.split('/')[0].split('?')[0].split('#')[0] 
        # removes any port number.
        domain_clean     = domain_with_port.split(':')[0]
        # separates the path from the domain. mail.google.com/u/0/ => path = /u/0/
        parts            = url_no_scheme.split('/', 1) 
        path             = '/' + parts[1] if len(parts) > 1 else '' # /path/page?search=abc#section1 it will add the / in from
        # extracts the query string. mail.google.com/search?q=abc#section => query = q=abc
        query            = url.split('?', 1)[1].split('#')[0] if '?' in url else '' # ['https://example.com:8080/path/page', 'search=abc#section1'] and then it convert it to search=abc
        fragment         = url.split('#', 1)[1] if '#' in url else '' # "section1" extract the part after the #

        # Step C: Domain Processing
        # Remove the www for tld and std
        domain_no_www = domain_clean
        if domain_no_www.lower().startswith('www.'):
            domain_no_www = domain_no_www[4:]

        # ensures every domain has www.
        if domain_clean.lower().startswith('www.'):
            domain_normalised = domain_clean
        else:
            domain_normalised = 'www.' + domain_clean

        # splits the normalized domain into pieces.
        domain_parts_full = domain_normalised.split('.') # ['www', 'google', 'com']

        # TLD and SLD extraction
        domain_parts = domain_no_www.split('.') # ['google', 'com']
        tld = domain_parts[-1].lower() if domain_parts else ''
        sld = domain_parts[-2].lower() if len(domain_parts) >= 2 else ''
        # handle country code TLDs
        if len(domain_parts) >= 3 and domain_parts[-1] in [
            'uk','au','in','jp','nz','za','br','sg','ie','pk'
        ]:
            tld = f"{domain_parts[-2]}.{domain_parts[-1]}"
            sld = domain_parts[-3].lower() if len(domain_parts) >= 3 else ''

            # "bbc.co.uk".split('.')  tld = "co.uk" sld = "bbc"

        # ── URL features ──────────────────────────────────────
        url_length                    = len(url)
        number_of_dots_in_url         = url.count('.')
        digits_in_url                 = re.findall(r'\d', url) # find all the digits from the urls and made a list of them
        # checks whether any digit repeats. 1234 → no repeated digits  1123 → repeated digit exists
        having_repeated_digits_in_url = 1 if len(digits_in_url) != len(
            set(digits_in_url)) and len(digits_in_url) > 0 else 0
        # set(digits_in_url) removes duplicates
        
         # Counts characters that are not letters or digits.
        number_of_digits_in_url       = sum(c.isdigit() for c in url)
        number_of_special_char_in_url = sum(not c.isalnum() for c in url) # alphabets or digits
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
        domain_length                          = len(domain_no_www)
        number_of_dots_in_domain               = domain_no_www.count('.')
        number_of_hyphens_in_domain            = domain_no_www.count('-')
        # Checks whether the domain contains unusual characters.
        having_special_characters_in_domain    = 1 if re.search(r'[^a-zA-Z0-9\.\-]', domain_no_www) else 0
        # Counts how many unusual special characters are in the domain.
        number_of_special_characters_in_domain = sum(not c.isalnum() and c not in '.-' for c in domain_no_www)
        # Binary feature: does the domain contain digits?
        having_digits_in_domain                = 1 if any(c.isdigit() for c in domain_no_www) else 0
        # Counts how many digit characters are in the domain.
        number_of_digits_in_domain             = sum(c.isdigit() for c in domain_no_www)
        # Extracts all digits from the domain.
        digits_in_domain                       = re.findall(r'\d', domain_no_www)
        # Checks whether any digit is repeated inside the domain.
        having_repeated_digits_in_domain       = 1 if len(
            digits_in_domain) != len(set(digits_in_domain)) and \
            len(digits_in_domain) > 0 else 0

        # ── subdomain features ────────────────────────────────
        # This calculates how many pieces come before the main domain + TLD.
        number_of_subdomains = max(0, len(domain_parts_full) - 2)
        # This stores the actual subdomain parts. Takes all the parts excepts last 2 
        subdomains           = domain_parts_full[:-2] if len(domain_parts_full) > 2 else []
        subdomain_depth      = len(subdomains)

        having_hyphen_in_subdomain = 1 if any('-' in s for s in subdomains) else 0
        average_subdomain_length = sum(len(s) for s in subdomains) / len(subdomains) \
            if subdomains else 0.0
        average_number_of_hyphens_in_subdomain = sum(s.count('-') for s in subdomains) / len(subdomains) \
            if subdomains else 0.0
        having_special_characters_in_subdomain = 1 if any(re.search(r'[^a-zA-Z0-9\-]', s)for s in subdomains) else 0
        number_of_special_characters_in_subdomain = sum(sum(not c.isalnum() and c != '-' for c in s)for s in subdomains)
        having_digits_in_subdomain = 1 if any(any(c.isdigit() for c in s) for s in subdomains) else 0
        number_of_digits_in_subdomain = sum(sum(c.isdigit() for c in s) for s in subdomains)
        # Joins all subdomains into one string and extracts all digits.
        all_sub_digits = re.findall(r'\d', ''.join(subdomains))
        # Checks whether any digit repeats across subdomains.
        having_repeated_digits_in_subdomain = 1 if len(all_sub_digits) != len(set(all_sub_digits)) and \
            len(all_sub_digits) > 0 else 0

        # ── path/query features ───────────────────────────────
        having_path    = 1 if len(path) > 1 else 0
        path_segments  = [p for p in path.split('/') if p]
        path_length    = len(path_segments)
        having_query   = 1 if len(query) > 0 else 0
        having_fragment = 1 if len(fragment) > 0 else 0
        having_anchor  = 1 if '#' in url else 0

        # ── entropy ───────────────────────────────────────────
        # this computes Shannon entropy for the full URL.
        # set(url) gets unique characters
        # url.count(c) / len(url) computes probability of each character
        # -sum(p * log2(p)) computes entropy
        if len(url) > 0:
            prob_url       = [url.count(c)/len(url) for c in set(url)] #url = "ababa" set(url) → {'a', 'b'}
            entropy_of_url = -sum(p*math.log2(p)
                                  for p in prob_url if p > 0)
        else:
            entropy_of_url = 0.0

        if len(domain_no_www) > 0:
            prob_dom          = [domain_no_www.count(c)/len(domain_no_www)
                                 for c in set(domain_no_www)]
            entropy_of_domain = -sum(p*math.log2(p)
                                     for p in prob_dom if p > 0)
        else:
            entropy_of_domain = 0.0

        # ── NEW FEATURE 43: Has Suspicious TLD ────────────────
        # Knowledge-based: TLDs with documented high phishing rates
        # Source: APWG Phishing Activity Trends Reports
        has_suspicious_tld = 1 if tld.lower() in SUSPICIOUS_TLDS else 0

        # ── NEW FEATURE 44: Is Known Safe SLD ─────────────────
        # Knowledge-based: Well-known legitimate second level domains
        # Addresses false positives for major brands
        is_known_safe_sld = 1 if sld.lower() in KNOWN_SAFE_SLDS else 0

        # ── NEW FEATURE 45: Consonant to Vowel Ratio ──────────
        # Pure structural: phishing domains are linguistically unnatural
        # xn--yetherallet = 0.89, google = 0.50
        vowels          = set('aeiouAEIOU')
        letters         = [c for c in domain_no_www if c.isalpha()]
        vowel_count     = sum(1 for c in letters if c in vowels)
        consonant_count = sum(1 for c in letters if c not in vowels)
        consonant_vowel_ratio = round(consonant_count / (vowel_count + 1), 4)

        # ── NEW FEATURE 46: Longest Digit Sequence ────────────
        # Pure structural: 82376g76g → 5, mp3cut → 1
        digit_sequences   = re.findall(r'\d+', domain_no_www)
        longest_digit_seq = max(
            (len(s) for s in digit_sequences), default=0)

        # ── NEW FEATURE 47: Digit to Letter Ratio ─────────────
        # Pure structural: 0000111service → 0.5, google → 0.0
        alpha_count        = sum(c.isalpha() for c in domain_no_www)
        digit_count        = sum(c.isdigit() for c in domain_no_www)
        digit_letter_ratio = round(digit_count / (alpha_count + 1), 4)

        # ── NEW FEATURE 48: Path to URL Ratio ─────────────────
        # Pure structural: bare phishing domains → 0.0
        # Legitimate URLs usually have paths
        path_to_url_ratio = round(len(path) / len(url), 4) if len(url) > 0 else 0.0

        return {
            # Original 42
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
            # New 6 hybrid features
            'has_suspicious_tld'                       : has_suspicious_tld,
            'is_known_safe_sld'                        : is_known_safe_sld,
            'consonant_vowel_ratio'                    : consonant_vowel_ratio,
            'longest_digit_sequence'                   : longest_digit_seq,
            'digit_letter_ratio'                       : digit_letter_ratio,
            'path_to_url_ratio'                        : path_to_url_ratio,
        }
    except Exception:
        return None

# ── 5. EXTRACT ───────────────────────────────────────────────
print(f"\n── Extracting Features ─────────────────────────────────")
print(f"   48 hybrid features (42 structural + 6 new)")
print(f"   Knowledge-based: has_suspicious_tld, is_known_safe_sld")
print(f"   Pure structural: consonant_vowel_ratio, longest_digit_sequence,")
print(f"                    digit_letter_ratio, path_to_url_ratio")

features_list = []
failed        = 0

# itertuples() is faster than some other row-by-row methods.
for i, row in enumerate(df.itertuples(), 1):
    feat = extract_features(row.url)
    if feat is not None:
        feat['label'] = row.label
        features_list.append(feat)
    else:
        failed += 1
    if i % 100000 == 0:
        print(f"   Processed {i:,} / {len(df):,} ...")

print(f"\nDone! Successful: {len(features_list):,}  Failed: {failed:,}")

# ── 6. DATAFRAME ─────────────────────────────────────────────
feat_df = pd.DataFrame(features_list)
print(f"\nFeature DataFrame: {feat_df.shape}")

#A zero-variance feature has the same value for every row. feature is useless for training because it contains no information. so it will be removed
variances = feat_df.drop(columns=['label']).var()
zero_var  = variances[variances == 0].index.tolist()
if zero_var:
    feat_df.drop(columns=zero_var, inplace=True)
    print(f"   Dropped zero-variance: {zero_var}")
else:
    print(f"No zero-variance features!")

# ── 7. SAVE ───────────────────────────────────────────────────
os.makedirs('model', exist_ok=True)
feature_names = [c for c in feat_df.columns if c != 'label']
with open('model/features.json', 'w') as f:
    json.dump(feature_names, f, indent=2)
print(f"\nmodel/features.json saved ({len(feature_names)} features)")

os.makedirs('data/processed', exist_ok=True)
feat_df.to_csv('data/processed/clean_dataset.csv', index=False)
X = feat_df.drop(columns=['label'])
y = feat_df['label']
X.to_csv('data/processed/X_train.csv', index=False)
y.to_csv('data/processed/y_train.csv', index=False)
print(f"\nAll datasets saved!")

# ── 10. TOP CORRELATIONS ─────────────────────────────────────
print(f"\n── Top 10 Feature Correlations ─────────────────────────")
corr = X.corrwith(y).abs().sort_values(ascending=False)
print(corr.head(10).to_string())

print("\n" + "=" * 60)
print("STEP 3 COMPLETE!")
print(f"  Total features    : {len(feature_names)}")
print(f"  Structural        : 46")
print(f"  Knowledge-based   : 2")
print("=" * 60)