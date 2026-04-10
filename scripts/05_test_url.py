# ============================================================
# PhishGuard v3 — Step 5: URL Tester (Pure ML — No Whitelist)
# Tests any URL against the trained model only
# No whitelist — pure machine learning prediction
# Author : Uzman Zahid
# ============================================================

import pandas as pd
import joblib
import json
import re
import math

# ── 1. LOAD MODEL AND FEATURES ───────────────────────────────
rf_model = joblib.load('model/model.pkl')
with open('model/features.json') as f:
    features = json.load(f)

print("=" * 60)
print("PHISHGUARD — URL TESTER (PURE ML)")
print("=" * 60)
print(f"Total features : {len(features)}")

# ── 2. FEATURE EXTRACTION ────────────────────────────────────
def extract_features(raw_url):
    """
    Extracts features from any URL format.
    Works on bare domains, http://, https:// equally.
    Same function used during training.
    """
    try:
        url = str(raw_url).strip()

        has_https = 1 if url.lower().startswith('https://') else 0
        has_http  = 1 if url.lower().startswith('http://') else 0

        url_no_scheme = url
        if '://' in url:
            url_no_scheme = url.split('://', 1)[1]

        domain_with_port = url_no_scheme.split('/')[0].split('?')[0].split('#')[0]
        domain_clean     = domain_with_port.split(':')[0]

        parts    = url_no_scheme.split('/', 1)
        path     = '/' + parts[1] if len(parts) > 1 else ''
        query    = url.split('?', 1)[1].split('#')[0] if '?' in url else ''
        fragment = url.split('#', 1)[1] if '#' in url else ''

        domain_no_www = domain_clean
        if domain_no_www.lower().startswith('www.'):
            domain_no_www = domain_no_www[4:]

        if domain_clean.lower().startswith('www.'):
            domain_normalised = domain_clean
        else:
            domain_normalised = 'www.' + domain_clean

        domain_parts_full = domain_normalised.split('.')

        url_length                    = len(url)
        number_of_dots_in_url         = url.count('.')
        digits_in_url                 = re.findall(r'\d', url)
        having_repeated_digits_in_url = 1 if len(digits_in_url) != len(
            set(digits_in_url)) and len(digits_in_url) > 0 else 0
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

        domain_length                          = len(domain_no_www)
        number_of_dots_in_domain               = domain_no_www.count('.')
        number_of_hyphens_in_domain            = domain_no_www.count('-')
        having_special_characters_in_domain    = 1 if re.search(
            r'[^a-zA-Z0-9\.\-]', domain_no_www) else 0
        number_of_special_characters_in_domain = sum(
            not c.isalnum() and c not in '.-' for c in domain_no_www)
        having_digits_in_domain                = 1 if any(
            c.isdigit() for c in domain_no_www) else 0
        number_of_digits_in_domain             = sum(
            c.isdigit() for c in domain_no_www)
        digits_in_domain                       = re.findall(r'\d', domain_no_www)
        having_repeated_digits_in_domain       = 1 if len(
            digits_in_domain) != len(set(digits_in_domain)) and \
            len(digits_in_domain) > 0 else 0

        number_of_subdomains = max(0, len(domain_parts_full) - 2)
        subdomains           = domain_parts_full[:-2] if len(
            domain_parts_full) > 2 else []

        subdomain_depth = len(subdomains)

        having_hyphen_in_subdomain             = 1 if any(
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
        number_of_special_characters_in_subdomain = sum(
            sum(not c.isalnum() and c != '-' for c in s)
            for s in subdomains)
        having_digits_in_subdomain             = 1 if any(
            any(c.isdigit() for c in s) for s in subdomains) else 0
        number_of_digits_in_subdomain          = sum(
            sum(c.isdigit() for c in s) for s in subdomains)
        all_sub_digits                         = re.findall(
            r'\d', ''.join(subdomains))
        having_repeated_digits_in_subdomain    = 1 if len(
            all_sub_digits) != len(set(all_sub_digits)) and \
            len(all_sub_digits) > 0 else 0

        having_path    = 1 if len(path) > 1 else 0
        path_segments  = [p for p in path.split('/') if p]
        path_length    = len(path_segments)
        having_query   = 1 if len(query) > 0 else 0
        having_fragment = 1 if len(fragment) > 0 else 0
        having_anchor  = 1 if '#' in url else 0

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

def predict_url(url):
    """Pure ML prediction — no whitelist."""
    feat_dict = extract_features(url)
    if feat_dict is None:
        return 0, [0.5, 0.5]
    X    = pd.DataFrame([feat_dict])[features]
    pred = rf_model.predict(X)[0]
    prob = rf_model.predict_proba(X)[0]
    return pred, prob

# ── 3. TEST URLs ──────────────────────────────────────────────
test_urls = [
    # LEGITIMATE
    ("https://login.dbs.ie/idp/profile/SAML2/Redirect/SSO?execution=e2s2",                                              "LEGIT"),
    ("https://www.youtube.com/watch?v=6WFJCR4GKo4&list=RD6WFJCR4GKo4&start_radio=1",                                            "LEGIT"),
    ("https://www.youtube.com/watch?v=Or4N3PxqEdg",                                             "LEGIT"),
    ("youtube.com/watch?v=h4kUiFOb_v0&list=PLIY8eNdw5tW_o8gsLqNBu8gmScCAqKm2Q&index=33",                         "LEGIT"),
    ("https://data.mendeley.com/datasets/hx4m73v2sf/1",      "LEGIT"),
    ("https://huggingface.co/datasets/ealvaradob/phishing-dataset",                  "LEGIT"),
    ("https://drive.google.com/drive/folders/1EtIwhZG4bXk9p2m",             "LEGIT"),
    ("https://huggingface.co/datasets/ealvaradob/phishing",                  "LEGIT"),
    ("https://mp3cut.net/change-speed",                                      "LEGIT"),
    ("https://claude.ai/chat/afcf252c-ad56-4191-b8af-82",                   "LEGIT"),
    ("https://www.kaggle.com/code/elnahas/phishing-email-detection-using-svm-rfc/input",                    "LEGIT"),
    ("https://aistudio.google.com/generate-speech?model=gemini-2.5-pro-preview-tts",                   "LEGIT"),
    ("https://myflixerz.to/genre/action",                                "LEGIT"),
    ("https://elearning.dbs.ie/user/profile.php",                            "LEGIT"),
    ("https://elearning.dbs.ie/pluginfile.php/2518047/mod_resource/content/1/B9IS105%20Module%20Guide%202025.pdf",                                                  "LEGIT"),
    ("https://elearning.dbs.ie/course/view.php?id=56980",                                        "LEGIT"),
    ("https://login.dbs.ie/idp/profile/SAML2/Redirect/SSO?execution=e1s2",                   "LEGIT"),
    # PHISHING
    ("https://ca22b18a5e142fa310f0cecbe15d3437e8f45agricle.qctheme.com/0vKqOcJjHdfn9wOH",                "PHISHING"),
    ("https://pv-amendesgouv.com",                                                "PHISHING"),
    ("https://mail-amendes-gouv.com/",                                      "PHISHING"),
    ("https://tracking.soudergift.com/twint/",                      "PHISHING"),
    ("https://djpeepsproductions.com/",                       "PHISHING"),
    ("0000111servicehelpdesk.godaddysites.com",                              "PHISHING"),
    ("https://recover-signer.com/",                                     "PHISHING"),
    ("https://www.fungibleapparel.com/",                         "PHISHING"),
    ("https://tracking.soudergift.com/twint/",                   "PHISHING"),
]

print(f"\n── URL Predictions ─────────────────────────────────────")
print(f"{'URL':<52} {'Expected':<10} {'Result':<12} {'Conf':>6}")
print(f"{'─' * 85}")

correct = 0
for url, expected in test_urls:
    pred, prob    = predict_url(url)
    confidence    = max(prob) * 100
    result        = '🔴 PHISHING' if pred == 1 else '🟢 LEGIT'
    exp_icon      = '🔴' if expected == 'PHISHING' else '🟢'
    is_correct    = (pred == 1 and expected == 'PHISHING') or \
                    (pred == 0 and expected == 'LEGIT')
    correct      += 1 if is_correct else 0
    status        = '✅' if is_correct else '❌'
    short_url     = url[:49] + '...' if len(url) > 52 else url
    print(f"{short_url:<52} {exp_icon} {expected:<8} "
          f"{result:<12} {confidence:>5.1f}% {status}")

total = len(test_urls)
print(f"\n   Score: {correct}/{total} ({correct/total*100:.0f}%)")
print(f"\n{'=' * 60}")
print("URL TESTING COMPLETE — PURE ML")
print(f"{'=' * 60}")