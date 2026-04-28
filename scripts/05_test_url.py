# ============================================================
# PhishGuard v5 — Step 5: URL Tester (48 Hybrid Features)
# 42 structural + 2 knowledge-based + 4 pure structural
# Author : Uzman Zahid
# ============================================================

import pandas as pd
import joblib
import json
import re
import math

# ── LOAD MODEL ───────────────────────────────────────────────
rf_model = joblib.load('model/model.pkl')
with open('model/features.json') as f:
    features = json.load(f)

print("=" * 60)
print("PHISHGUARD v5 — URL TESTER (48 HYBRID FEATURES)")
print("=" * 60)
print(f"Total features    : {len(features)}")
print(f"Structural        : 46")
print(f"Knowledge-based   : 2 (has_suspicious_tld, is_known_safe_sld)")

# ── KNOWLEDGE TABLES ─────────────────────────────────────────
SUSPICIOUS_TLDS = {
    'tk', 'ml', 'ga', 'cf', 'gq',
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
    'duolingo', 'bbc', 'cnn', 'reuters', 'bloomberg',
    'techcrunch', 'mit', 'stanford', 'harvard',
    'oxford', 'cambridge', 'dbs', 'ucd', 'tcd', 'dcu',
    'npmjs', 'pypi', 'kubernetes', 'mozilla', 'docker',
    'spinbot', 'ilovepdf', 'smallpdf', 'mp3cut',
    'convertio', 'tinypng', 'grammarly',
    'wise', 'revolut', 'coinbase', 'binance',
    'vimeo', 'twitch', 'dailymotion', 'soundcloud',
    'gov', 'nasa', 'digitalocean',
}

# ── FEATURE EXTRACTION ───────────────────────────────────────
def extract_features(raw_url):
    try:
        url = str(raw_url).strip()

        has_https = 1 if url.lower().startswith('https://') else 0
        has_http  = 1 if url.lower().startswith('http://') else 0

        url_no_scheme = url
        if '://' in url:
            url_no_scheme = url.split('://', 1)[1]

        domain_with_port = url_no_scheme.split('/')[0].split('?')[0].split('#')[0]
        domain_clean     = domain_with_port.split(':')[0]
        parts            = url_no_scheme.split('/', 1)
        path             = '/' + parts[1] if len(parts) > 1 else ''
        query            = url.split('?', 1)[1].split('#')[0] if '?' in url else ''
        fragment         = url.split('#', 1)[1] if '#' in url else ''

        domain_no_www = domain_clean
        if domain_no_www.lower().startswith('www.'):
            domain_no_www = domain_no_www[4:]

        if domain_clean.lower().startswith('www.'):
            domain_normalised = domain_clean
        else:
            domain_normalised = 'www.' + domain_clean

        domain_parts_full = domain_normalised.split('.')

        # TLD and SLD
        domain_parts = domain_no_www.split('.')
        tld = domain_parts[-1].lower() if domain_parts else ''
        sld = domain_parts[-2].lower() if len(domain_parts) >= 2 else ''
        if len(domain_parts) >= 3 and domain_parts[-1] in [
            'uk','au','in','jp','nz','za','br','sg','ie','pk'
        ]:
            tld = f"{domain_parts[-2]}.{domain_parts[-1]}"
            sld = domain_parts[-3].lower() if len(domain_parts) >= 3 else ''

        # URL features
        url_length                    = len(url)
        number_of_dots_in_url         = url.count('.')
        digits_in_url                 = re.findall(r'\d', url)
        having_repeated_digits_in_url = 1 if len(digits_in_url) != len(set(digits_in_url)) and len(digits_in_url) > 0 else 0
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

        # Domain features
        domain_length                          = len(domain_no_www)
        number_of_dots_in_domain               = domain_no_www.count('.')
        number_of_hyphens_in_domain            = domain_no_www.count('-')
        having_special_characters_in_domain    = 1 if re.search(r'[^a-zA-Z0-9\.\-]', domain_no_www) else 0
        number_of_special_characters_in_domain = sum(not c.isalnum() and c not in '.-' for c in domain_no_www)
        having_digits_in_domain                = 1 if any(c.isdigit() for c in domain_no_www) else 0
        number_of_digits_in_domain             = sum(c.isdigit() for c in domain_no_www)
        digits_in_domain                       = re.findall(r'\d', domain_no_www)
        having_repeated_digits_in_domain       = 1 if len(digits_in_domain) != len(set(digits_in_domain)) and len(digits_in_domain) > 0 else 0

        # Subdomain features
        number_of_subdomains = max(0, len(domain_parts_full) - 2)
        subdomains           = domain_parts_full[:-2] if len(domain_parts_full) > 2 else []
        subdomain_depth      = len(subdomains)

        having_hyphen_in_subdomain                = 1 if any('-' in s for s in subdomains) else 0
        average_subdomain_length                  = sum(len(s) for s in subdomains) / len(subdomains) if subdomains else 0.0
        average_number_of_hyphens_in_subdomain    = sum(s.count('-') for s in subdomains) / len(subdomains) if subdomains else 0.0
        having_special_characters_in_subdomain    = 1 if any(re.search(r'[^a-zA-Z0-9\-]', s) for s in subdomains) else 0
        number_of_special_characters_in_subdomain = sum(sum(not c.isalnum() and c != '-' for c in s) for s in subdomains)
        having_digits_in_subdomain                = 1 if any(any(c.isdigit() for c in s) for s in subdomains) else 0
        number_of_digits_in_subdomain             = sum(sum(c.isdigit() for c in s) for s in subdomains)
        all_sub_digits                            = re.findall(r'\d', ''.join(subdomains))
        having_repeated_digits_in_subdomain       = 1 if len(all_sub_digits) != len(set(all_sub_digits)) and len(all_sub_digits) > 0 else 0

        # Path/query
        having_path     = 1 if len(path) > 1 else 0
        path_segments   = [p for p in path.split('/') if p]
        path_length     = len(path_segments)
        having_query    = 1 if len(query) > 0 else 0
        having_fragment = 1 if len(fragment) > 0 else 0
        having_anchor   = 1 if '#' in url else 0

        # Entropy
        prob_url       = [url.count(c)/len(url) for c in set(url)] if len(url) > 0 else []
        entropy_of_url = -sum(p*math.log2(p) for p in prob_url if p > 0) if prob_url else 0.0
        prob_dom          = [domain_no_www.count(c)/len(domain_no_www) for c in set(domain_no_www)] if len(domain_no_www) > 0 else []
        entropy_of_domain = -sum(p*math.log2(p) for p in prob_dom if p > 0) if prob_dom else 0.0

        # ── 6 NEW HYBRID FEATURES ─────────────────────────────
        has_suspicious_tld    = 1 if tld.lower() in SUSPICIOUS_TLDS else 0
        is_known_safe_sld     = 1 if sld.lower() in KNOWN_SAFE_SLDS else 0
        vowels                = set('aeiouAEIOU')
        letters               = [c for c in domain_no_www if c.isalpha()]
        vowel_count           = sum(1 for c in letters if c in vowels)
        consonant_count       = sum(1 for c in letters if c not in vowels)
        consonant_vowel_ratio = round(consonant_count / (vowel_count + 1), 4)
        digit_sequences       = re.findall(r'\d+', domain_no_www)
        longest_digit_seq     = max((len(s) for s in digit_sequences), default=0)
        alpha_count           = sum(c.isalpha() for c in domain_no_www)
        digit_count           = sum(c.isdigit() for c in domain_no_www)
        digit_letter_ratio    = round(digit_count / (alpha_count + 1), 4)
        path_to_url_ratio     = round(len(path) / len(url), 4) if len(url) > 0 else 0.0

        return {
            'has_https': has_https, 'has_http': has_http,
            'url_length': url_length,
            'number_of_dots_in_url': number_of_dots_in_url,
            'having_repeated_digits_in_url': having_repeated_digits_in_url,
            'number_of_digits_in_url': number_of_digits_in_url,
            'number_of_special_char_in_url': number_of_special_char_in_url,
            'number_of_hyphens_in_url': number_of_hyphens_in_url,
            'number_of_underline_in_url': number_of_underline_in_url,
            'number_of_slash_in_url': number_of_slash_in_url,
            'number_of_questionmark_in_url': number_of_questionmark_in_url,
            'number_of_equal_in_url': number_of_equal_in_url,
            'number_of_at_in_url': number_of_at_in_url,
            'number_of_dollar_in_url': number_of_dollar_in_url,
            'number_of_exclamation_in_url': number_of_exclamation_in_url,
            'number_of_hashtag_in_url': number_of_hashtag_in_url,
            'number_of_percent_in_url': number_of_percent_in_url,
            'domain_length': domain_length,
            'number_of_dots_in_domain': number_of_dots_in_domain,
            'number_of_hyphens_in_domain': number_of_hyphens_in_domain,
            'having_special_characters_in_domain': having_special_characters_in_domain,
            'number_of_special_characters_in_domain': number_of_special_characters_in_domain,
            'having_digits_in_domain': having_digits_in_domain,
            'number_of_digits_in_domain': number_of_digits_in_domain,
            'having_repeated_digits_in_domain': having_repeated_digits_in_domain,
            'number_of_subdomains': number_of_subdomains,
            'subdomain_depth': subdomain_depth,
            'having_hyphen_in_subdomain': having_hyphen_in_subdomain,
            'average_subdomain_length': average_subdomain_length,
            'average_number_of_hyphens_in_subdomain': average_number_of_hyphens_in_subdomain,
            'having_special_characters_in_subdomain': having_special_characters_in_subdomain,
            'number_of_special_characters_in_subdomain': number_of_special_characters_in_subdomain,
            'having_digits_in_subdomain': having_digits_in_subdomain,
            'number_of_digits_in_subdomain': number_of_digits_in_subdomain,
            'having_repeated_digits_in_subdomain': having_repeated_digits_in_subdomain,
            'having_path': having_path, 'path_length': path_length,
            'having_query': having_query, 'having_fragment': having_fragment,
            'having_anchor': having_anchor,
            'entropy_of_url': entropy_of_url, 'entropy_of_domain': entropy_of_domain,
            'has_suspicious_tld': has_suspicious_tld,
            'is_known_safe_sld': is_known_safe_sld,
            'consonant_vowel_ratio': consonant_vowel_ratio,
            'longest_digit_sequence': longest_digit_seq,
            'digit_letter_ratio': digit_letter_ratio,
            'path_to_url_ratio': path_to_url_ratio,
        }
    except Exception:
        return None

def predict_url(url):
    feat_dict = extract_features(url)
    if feat_dict is None:
        return 0, [0.5, 0.5]
    X    = pd.DataFrame([feat_dict])[features]
    pred = rf_model.predict(X)[0]
    prob = rf_model.predict_proba(X)[0]
    return pred, prob

# ── TEST URLs ─────────────────────────────────────────────────
test_urls = [
    # Educational
    ("https://chromewebstore.google.com/category/extensions?utm_source=ext_sidebar&hl=en-US",                                                        "LEGIT"),
    ("https://checkphish.bolster.ai/",                                                      "LEGIT"),
    ("https://tools.zvelo.com/",                                                    "LEGIT"),
    ("https://www.phishtank.net/phish_archive.php?page=3",                                                        "LEGIT"),
    # Government
    ("https://www.gov.ie/en/",                                                      "LEGIT"),
    ("https://www.gov.uk/",                                                         "LEGIT"),
    ("https://www.revenue.ie/",                                                     "LEGIT"),
    ("https://myaccount.revenue.ie/",                                               "LEGIT"),
    # Developer/API
    ("https://developers.openai.com/api/docs/guides/agent-builder",                "LEGIT"),
    ("https://api.github.com/repos/user/repo",                                     "LEGIT"),
    ("https://platform.openai.com/playground",                                     "LEGIT"),
    ("https://docs.python.org/3/library/urllib.html",                              "LEGIT"),
    ("https://developer.mozilla.org/en-US/docs/Web/JavaScript",                    "LEGIT"),
    ("https://kubernetes.io/docs/concepts/",                                        "LEGIT"),
    # Google Services
    ("https://mail.google.com/mail/u/0/",                                          "LEGIT"),
    ("https://mail.google.com/mail/u/0/#spam",                                     "LEGIT"),
    ("https://drive.google.com/drive/",                                             "LEGIT"),
    ("https://chromewebstore.google.com/?hl=en",                                   "LEGIT"),
    ("https://console.firebase.google.com/project/abc",                            "LEGIT"),
    ("https://colab.research.google.com/drive/abc123",                             "LEGIT"),
    # Tools
    ("https://spinbot.com/",                                                        "LEGIT"),
    ("https://www.ilovepdf.com/",                                                   "LEGIT"),
    ("https://mp3cut.net/change-speed",                                             "LEGIT"),
    ("https://smallpdf.com/compress-pdf",                                           "LEGIT"),
    # Financial
    ("https://dashboard.stripe.com/login?redirect=%2Ftest%2Fdashboard",            "LEGIT"),
    ("https://dashboard.stripe.com/",                                               "LEGIT"),
    # Personal sites
    ("https://uzmanzahid.netlify.app/",                                             "LEGIT"),
    ("https://app.netlify.com/sites/phishguard/deploys",                           "LEGIT"),
    ("https://vercel.com/uzman/phishguard/deployments",                            "LEGIT"),
    # Raw/CDN
    ("https://raw.githubusercontent.com/openphish/public_feed/refs/heads/main/feed.txt", "LEGIT"),
    ("https://cdn.jsdelivr.net/npm/bootstrap/",                                    "LEGIT"),
    # Social Media
    ("https://www.linkedin.com/in/username/",                                       "LEGIT"),
    ("https://www.facebook.com/profile.php?id=100095",                             "LEGIT"),
    ("https://medium.com/@username/article-title-abc123",                          "LEGIT"),
    ("https://twitter.com/username/status/1234567890",                             "LEGIT"),
    # Video
    ("https://www.youtube.com/watch?v=1hC3cxsVaDs",                                "LEGIT"),
    ("https://vimeo.com/123456789",                                                 "LEGIT"),
    ("https://www.twitch.tv/username",                                              "LEGIT"),
    # E-commerce
    ("https://www.amazon.com/dp/B08N5WRWNW",                                       "LEGIT"),
    ("https://www.ebay.com/itm/item-name/123456789",                               "LEGIT"),
    ("https://www.etsy.com/listing/123456/item-name",                              "LEGIT"),
    ("https://www.aliexpress.com/item/1005012000364163.html",                      "LEGIT"),
    # News/Blog
    ("https://www.bbc.com/news/technology-12345678",                                "LEGIT"),
    ("https://techcrunch.com/2024/01/15/article-name/",                            "LEGIT"),
    # Package repos
    ("https://pypi.org/project/fastapi/",                                           "LEGIT"),
    ("https://www.npmjs.com/package/express",                                       "LEGIT"),
    ("https://hub.docker.com/r/username/image",                                    "LEGIT"),
    # Cloud/DevOps
    ("https://portal.azure.com/#home",                                              "LEGIT"),
    ("https://huggingface.co/spaces/Uzmann/phishguard",                            "LEGIT"),
    ("https://www.kaggle.com/code/username/notebook",                              "LEGIT"),
    # Communication
    ("https://discord.com/channels/123/456",                                        "LEGIT"),
    ("https://zoom.us/j/123456789",                                                 "LEGIT"),
    ("https://teams.microsoft.com/l/meetup-join/abc",                              "LEGIT"),
    # Standard
    ("https://www.google.com/",                                                     "LEGIT"),
    ("https://stackoverflow.com/questions/tagged/python",                           "LEGIT"),
    ("https://github.com/uz-zi/Zameen.com-Programming-f",                          "LEGIT"),
    ("https://claude.ai/chat/afcf252c-ad56-4191-b8af-82",                          "LEGIT"),
    ("https://www.amazon.co.uk/dp/B08N5WRWNW",                                     "LEGIT"),
    ("https://www.bbc.co.uk/sport/football",                                        "LEGIT"),
    # OAuth
    ("https://accounts.google.com/signin?continue=https%3A%2F%2Fmail.google.com", "LEGIT"),
    ("https://login.microsoftonline.com/",                                          "LEGIT"),
    ("https://developers.openai.com/api/docs/guides/agent-builder",                                              "LEGIT"),

    # ── PHISHING ──────────────────────────────────────────────
    ("https://rax3ri.lat/c7uup3ox/TFGIhF/n4mEoo",                       "PHISHING"),
    ("https://shorturl.at/jliTD",                                                       "PHISHING"),
    ("http://allegro.pl-alebilet-3427.sbs",                                             "PHISHING"),
    ("https://allegrolokalnie.pl-alebilet-3427.sbs",                             "PHISHING"),
    ("https://allegro.0728723889735.cyou/",                              "PHISHING"),
    ("https://allegro.6380423831130.cfd/",                                     "PHISHING"),
    ("http://mata-masxxk-logijiiz.godaddysites.com",                                            "PHISHING"),
    ("https://kucoinuylogin.webflow.io/",                                            "PHISHING"),
    ("http://allegro-lokalnie.82376g76g.cyou/",                                     "PHISHING"),
    ("http://allegro.674353.lat",                                                   "PHISHING"),
    ("https://exodas-web3.zapier.app/wallet-portal",                                "PHISHING"),
    ("https://verify-seed.com/",                                                    "PHISHING"),
    ("https://metamaskk-io-ext.wixstudio.com/en-us",                               "PHISHING"),
    ("https://ledgrlive-desktop.wixstudio.com/start",                               "PHISHING"),
    ("https://tronslink.cc/",                                                       "PHISHING"),
    ("https://pv-amendesgouv.com",                                                  "PHISHING"),
    ("https://recover-signer.com/",                                                 "PHISHING"),
    ("https://dplussecurity.com/wp-sion/",                                          "PHISHING"),
    ("http://xn--yetherallet-tv8eu6a.com.cp-in-12.webho",                          "PHISHING"),
    ("https://xn--80akhbyknj4f.xn--p1ai/",                                         "PHISHING"),
    ("https://globalintro.wixstudio.com/live-us",                                   "PHISHING"),
    ("https://ledgrocom.wixstudio.com/start",                                       "PHISHING"),
    ("https://rax3ri.lat/c7uup3ox/TFGIhF/n4mEoo",                                  "PHISHING"),
    ("http://kreditdkb-mydkb.ooguy.com//login.php",                                        "PHISHING"),
    ("https://paypal-login.pages.dev/security-check",                                                     "PHISHING"),
    ("https://microsoft-authentication.github.io/login",                                  "PHISHING"),
    ("https://secure-login-coinbase.web.app/",                          "PHISHING"),
    ("http://malainrdimalafamslaridmaalcoiemlaashkfjh-2548.twil.io/lezflkelzfkjezjkhfjezghgfzehj.html",                              "PHISHING"),
]

# ── RUN PREDICTIONS ───────────────────────────────────────────
print(f"\n── URL Predictions ─────────────────────────────────────")
print(f"{'URL':<55} {'Expected':<10} {'Result':<12} {'Conf':>6}")
print(f"{'─' * 88}")

correct = 0
legit_correct = legit_total = phish_correct = phish_total = 0

for url, expected in test_urls:
    pred, prob    = predict_url(url)
    confidence    = max(prob) * 100
    result        = '🔴 PHISHING' if pred == 1 else '🟢 LEGIT'
    exp_icon      = '🔴' if expected == 'PHISHING' else '🟢'
    is_correct    = (pred == 1 and expected == 'PHISHING') or \
                    (pred == 0 and expected == 'LEGIT')
    correct      += 1 if is_correct else 0
    status        = '✅' if is_correct else '❌'
    short_url     = url[:52] + '...' if len(url) > 55 else url

    if expected == 'LEGIT':
        legit_total += 1
        if is_correct: legit_correct += 1
    else:
        phish_total += 1
        if is_correct: phish_correct += 1

    print(f"{short_url:<55} {exp_icon} {expected:<8} "
          f"{result:<12} {confidence:>5.1f}% {status}")

total = len(test_urls)
print(f"\n{'─' * 88}")
print(f"   Overall  : {correct}/{total} ({correct/total*100:.0f}%)")
print(f"   Legit    : {legit_correct}/{legit_total} ({legit_correct/legit_total*100:.0f}%)")
print(f"   Phishing : {phish_correct}/{phish_total} ({phish_correct/phish_total*100:.0f}%)")
print(f"\n{'=' * 60}")
print("URL TESTING COMPLETE — 48 HYBRID FEATURES")
print(f"{'=' * 60}")