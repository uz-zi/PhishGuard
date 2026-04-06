# ============================================================
# PhishGuard v3 — Step 2: Data Augmentation (Improved)
# Uses Tranco top-1M list + manual additions
# Input  : data/processed/clean_urls.csv
#          data/raw/top-1m.csv
# Output : data/processed/augmented_urls.csv
#
# Covers ALL URL patterns including:
#   ✅ Simple homepages
#   ✅ No-www domains
#   ✅ HTTP versions
#   ✅ Hyphenated paths (mp3cut.net/change-speed)
#   ✅ Deep paths
#   ✅ Query parameters
#   ✅ UTM tracking parameters
#   ✅ Alphanumeric IDs (YouTube, Google Drive)
#   ✅ UUID paths (Claude, Notion)
#   ✅ User account paths (/u/0/)
#   ✅ Subdomains (myaccount., drive., mail.)
#   ✅ Multi-level subdomains (m365.cloud.microsoft) 
#   ✅ HTTP + subdomain + UTM combinations
#   ✅ Country code TLDs (.co.uk, .com.au)
#   ✅ Newer domains not in Tranco 2023
#   ✅ URL shortener patterns
#   ✅ Numeric IDs in path (Amazon style)
#
# Author : Uzman Zahid
# ============================================================

import pandas as pd
import random
import string
import uuid
import os

print("=" * 60)
print("PHISHGUARD v3 — Script 2: DATA AUGMENTATION (IMPROVED)")
print("=" * 60)

# ── 1. LOAD EXISTING DATASET ─────────────────────────────────
df = pd.read_csv('data/processed/clean_urls.csv')
print(f"\n✅ Existing dataset loaded: {len(df):,} URLs")
print(f"   Phishing   (0): {(df['status']==0).sum():,}")
print(f"   Legitimate (1): {(df['status']==1).sum():,}")

# ── 2. LOAD TRANCO LIST ──────────────────────────────────────
# names=['rank', 'domain'] assigns column names manually
tranco = pd.read_csv('data/raw/top-1m.csv', header=None,
                     names=['rank', 'domain'])
print(f"\n✅ Tranco list loaded: {len(tranco):,} domains")

top_domains = tranco[tranco['rank'] <= 10000]['domain'].tolist()
print(f"   Using top 10,000 domains")

# ── 3. MANUALLY ADD NEWER DOMAINS (not in Tranco 2023) ──────
# These are important legitimate domains that may not appear
# in the 2023 Tranco list but are widely used in 2025/2026
newer_domains = [
    # AI platforms
    'openai.com', 'claude.ai', 'perplexity.ai', 'gemini.google.com',
    'copilot.microsoft.com', 'bard.google.com', 'chat.openai.com',
    'midjourney.com', 'stability.ai', 'huggingface.co',
    # Developer platforms
    'vercel.app', 'netlify.app', 'railway.app', 'render.com',
    'fly.io', 'supabase.com', 'planetscale.com', 'neon.tech',
    # Cloud storage
    'drive.google.com', 'onedrive.live.com', 'dropbox.com',
    'box.com', 'icloud.com', 'mega.nz',
    # Communication
    'discord.com', 'slack.com', 'teams.microsoft.com',
    'zoom.us', 'meet.google.com', 'webex.com',
    # Education
    'coursera.org', 'udemy.com', 'edx.org', 'khanacademy.org',
    'duolingo.com', 'brilliant.org',
    # Productivity
    'notion.so', 'airtable.com', 'trello.com', 'asana.com',
    'monday.com', 'clickup.com', 'basecamp.com',
    # Shopping
    'etsy.com', 'ebay.com', 'alibaba.com', 'aliexpress.com',
    'shopify.com', 'woocommerce.com',
    # Entertainment
    'netflix.com', 'spotify.com', 'twitch.tv', 'tiktok.com',
    'pinterest.com', 'reddit.com', 'quora.com',
    # Finance
    'paypal.com', 'stripe.com', 'wise.com', 'revolut.com',
    'coinbase.com', 'binance.com',
    # Microsoft services
    'm365.cloud.microsoft', 'office.com', 'outlook.com',
    'azure.microsoft.com', 'portal.azure.com',
    # Google services
    'accounts.google.com', 'myaccount.google.com',
    'photos.google.com', 'calendar.google.com',
    # URL shorteners (legitimate)
    'bit.ly', 't.co', 'goo.gl', 'ow.ly', 'tinyurl.com',
    'rb.gy', 'short.io', 'tiny.cc',
]

# combine Tranco + newer domains
# set(...) removes duplicates
all_domains = list(set(top_domains + newer_domains))
print(f"   Added {len(newer_domains)} newer domains manually")
print(f"   Total domains: {len(all_domains):,}")

# ── 4. HELPER FUNCTIONS ──────────────────────────────────────

# This function returns a random string of letters and digits. A7d9Kx3Pq2Lm
def rand_alnum(n=12):
    """Alphanumeric ID like YouTube video IDs"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=n))

# This generates a UUID-style identifier.
def rand_uuid():
    """UUID like Claude chat IDs"""
    return str(uuid.UUID(int=random.getrandbits(128)))

# This creates an Amazon-like product ID. B0X9A2KQ7Z
def rand_amazon_id():
    """Amazon product ID like B08N5WRWNW"""
    return 'B0' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

#This generates a random integer and converts it to a string.
def rand_numeric():
    """Numeric ID"""
    return str(random.randint(1000000, 999999999))

def rand_word():
    """Common legitimate URL path words"""
    words = [
        'about', 'contact', 'home', 'page', 'search', 'login',
        'register', 'profile', 'settings', 'help', 'faq', 'blog',
        'news', 'products', 'services', 'team', 'pricing', 'docs',
        'api', 'support', 'terms', 'privacy', 'careers', 'features',
        'download', 'upload', 'dashboard', 'account', 'explore',
        'trending', 'popular', 'latest', 'archive', 'category',
        'watch', 'listen', 'read', 'view', 'share', 'edit',
        'create', 'manage', 'report', 'analytics', 'overview',
        'chat', 'inbox', 'feed', 'library', 'collection',
        'playlist', 'channel', 'studio', 'drive', 'folder',
        'project', 'workspace', 'repository', 'issues', 'releases',
        'convert', 'compress', 'resize', 'crop', 'merge', 'split',
        'change', 'speed', 'volume', 'quality', 'format',
        'questions', 'answers', 'tagged', 'users', 'jobs',
    ]
    return random.choice(words)

# This Function generates a random hyphenated path like change-speed or video-converter.
def rand_hyphen_path():
    """Hyphenated path like change-speed, video-converter"""
    w1, w2 = rand_word(), rand_word()
    w3 = rand_word()
    n = random.randint(1, 3)
    if n == 1:
        return f"{w1}-{w2}"
    elif n == 2:
        return f"{w1}-{w2}-{w3}"
    else:
        return f"{w1}-{w2}"

# This function generates a random multi-word slug like "best-video-converter-online" or "how-to-change-speed-of-video".
def rand_slug():
    """Multi-word slug"""
    return '-'.join([rand_word() for _ in range(random.randint(2, 5))])

# This function generates random UTM parameters for Google Analytics tracking, like "?utm_source=google&utm_medium=cpc&utm_campaign=spring_sale".
def rand_utm():
    """Google Analytics UTM parameters"""
    sources  = ['google', 'youtube', 'facebook', 'twitter',
                'instagram', 'email', 'newsletter', 'bing',
                'linkedin', 'reddit', 'organic', 'direct',
                'YouTube', 'Gmail', 'Chrome']
    mediums  = ['cpc', 'social', 'email', 'organic',
                'referral', 'display', 'video', 'banner']
    source   = random.choice(sources)
    medium   = random.choice(mediums)
    campaign = rand_word()
    n = random.randint(1, 3)
    if n == 1:
        return f"?utm_source={source}"
    elif n == 2:
        return f"?utm_source={source}&utm_medium={medium}"
    else:
        return f"?utm_source={source}&utm_medium={medium}&utm_campaign={campaign}"

# This function generates common legitimate subdomains like "myaccount", "drive", "mail", "docs", "maps", "play", "store", "photos", "calendar", "meet", "chat", "support", "help", "blog", "news", "shop", "app", "api", "dev", "portal", "dashboard", "secure", "login", "signin", "auth", and "id".
def rand_subdomain():
    """Common legitimate subdomains"""
    return random.choice([
        'myaccount', 'account', 'accounts', 'drive', 'mail',
        'docs', 'maps', 'play', 'store', 'photos', 'calendar',
        'meet', 'chat', 'support', 'help', 'blog', 'news',
        'shop', 'app', 'api', 'dev', 'portal', 'dashboard',
        'secure', 'login', 'signin', 'auth', 'id',
    ])

# This function generates numeric-style subdomains like "m365", "o365", "web3", "app1", "app2", "api1", "api2", "v2", "v3", "beta", "staging", "dev", and "test".
def rand_numeric_sub():
    """Numeric-style subdomains like m365, o365"""
    return random.choice([
        'm365', 'o365', 'web3', 'app1', 'app2', 'api1', 'api2',
        'v2', 'v3', 'beta', 'staging', 'dev', 'test',
    ])

#This function generates cloud-style subdomains like "cloud", "azure", "aws", "gcp", "cdn", "static", "assets", "media", "storage", "files", and "data".
def rand_cloud_sub():
    """Cloud-style subdomains"""
    return random.choice([
        'cloud', 'azure', 'aws', 'gcp', 'cdn', 'static',
        'assets', 'media', 'storage', 'files', 'data',
    ])

# this function generates random country code TLDs like .co.uk, .com.au, .co.in, .co.jp, .co.nz, .co.za, .com.br, .co.ke, .com.sg, and .co.id.
def rand_cctld():
    """Country code TLD extensions"""
    return random.choice([
        'co.uk', 'com.au', 'co.in', 'co.jp', 'co.nz',
        'co.za', 'com.br', 'co.ke', 'com.sg', 'co.id',
    ])
# This function generates random shortener IDs like bit.ly/abc123 or t.co/xyz789.
def rand_shortener_id():
    """Short ID like bit.ly uses"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(5, 8)))

# ── 5. MAIN URL GENERATOR ────────────────────────────────────
def generate_urls(domain):
    """
    Generates ~65 realistic URL variations per domain.
    Covers every URL pattern the model needs to learn.
    """
    urls = []

    # extract base domain for country code TLD variations
    parts = domain.split('.')
    base  = parts[0] if len(parts) >= 2 else domain

    # ── HOMEPAGE VARIATIONS ───────────────────────────────────
    urls.append(f"https://www.{domain}/")
    urls.append(f"https://{domain}/")
    urls.append(f"http://www.{domain}/")
    urls.append(f"http://{domain}/")

    # ── SIMPLE WORD PATHS ─────────────────────────────────────
    urls.append(f"https://www.{domain}/{rand_word()}")
    urls.append(f"https://{domain}/{rand_word()}")
    urls.append(f"http://www.{domain}/{rand_word()}")
    urls.append(f"http://{domain}/{rand_word()}")

    # ── HYPHENATED PATHS (mp3cut.net/change-speed style) ─────
    urls.append(f"https://www.{domain}/{rand_hyphen_path()}")
    urls.append(f"https://{domain}/{rand_hyphen_path()}")
    urls.append(f"http://www.{domain}/{rand_hyphen_path()}")
    urls.append(f"http://{domain}/{rand_hyphen_path()}")
    urls.append(f"https://{domain}/{rand_word()}/{rand_hyphen_path()}")
    urls.append(f"https://www.{domain}/{rand_hyphen_path()}.html")
    urls.append(f"https://www.{domain}/{rand_hyphen_path()}/{rand_word()}")

    # ── DEEP PATHS ────────────────────────────────────────────
    urls.append(f"https://www.{domain}/{rand_word()}/{rand_word()}")
    urls.append(f"https://www.{domain}/{rand_word()}/{rand_word()}/{rand_word()}")
    urls.append(f"https://{domain}/{rand_word()}/{rand_word()}")
    urls.append(f"https://www.{domain}/{rand_word()}/{rand_slug()}")

    # ── QUERY PARAMETERS ─────────────────────────────────────
    urls.append(f"https://www.{domain}/search?q={rand_word()}")
    urls.append(f"https://www.{domain}/{rand_word()}?id={rand_numeric()}")
    urls.append(f"https://{domain}/results?search_query={rand_word()}")
    urls.append(f"https://www.{domain}/{rand_word()}?page={random.randint(1,10)}")
    urls.append(f"https://www.{domain}/{rand_word()}?ref={rand_word()}")
    urls.append(f"https://www.{domain}/{rand_word()}?tab={rand_word()}&sort=newest")

    # ── UTM TRACKING PARAMETERS ──────────────────────────────
    urls.append(f"https://www.{domain}/{rand_word()}{rand_utm()}")
    urls.append(f"https://www.{domain}/{rand_word()}/{rand_word()}{rand_utm()}")
    urls.append(f"http://www.{domain}/{rand_word()}{rand_utm()}")
    urls.append(f"https://{domain}/{rand_word()}{rand_utm()}")

    # ── HTTP + SUBDOMAIN + UTM (the failing pattern) ──────────
    sub = rand_subdomain()
    urls.append(f"http://{sub}.{domain}/u/0/?utm_source={random.choice(['YouTube','google','email','Gmail'])}")
    urls.append(f"http://{sub}.{domain}/{rand_word()}?utm_source={random.choice(['google','YouTube','email'])}&utm_medium=cpc")
    urls.append(f"https://{sub}.{domain}/u/0/?utm_source={random.choice(['YouTube','google'])}")
    urls.append(f"http://{sub}.{domain}/u/{random.randint(0,5)}/?utm_source=YouTube")

    # ── ALPHANUMERIC ID PATHS (YouTube, Drive) ────────────────
    urls.append(f"https://www.{domain}/watch?v={rand_alnum(11)}")
    urls.append(f"https://www.{domain}/watch?v={rand_alnum(11)}&list={rand_alnum(20)}&index={random.randint(1,50)}")
    urls.append(f"https://www.{domain}/file/{rand_alnum(20)}")
    urls.append(f"https://www.{domain}/{rand_word()}/{rand_alnum(15)}")
    urls.append(f"https://www.{domain}/d/{rand_alnum(20)}/view")
    urls.append(f"https://www.{domain}/drive/folders/{rand_alnum(20)}")
    urls.append(f"https://{domain}/{rand_word()}/{rand_alnum(12)}")

    # ── UUID PATHS (Claude, Notion, Figma) ───────────────────
    urls.append(f"https://www.{domain}/chat/{rand_uuid()}")
    urls.append(f"https://www.{domain}/doc/{rand_uuid()}")
    urls.append(f"https://www.{domain}/{rand_word()}/{rand_uuid()}")
    urls.append(f"https://{domain}/{rand_word()}/{rand_uuid()}")
    urls.append(f"https://www.{domain}/{rand_uuid()}")

    # ── AMAZON-STYLE PRODUCT IDS ──────────────────────────────
    urls.append(f"https://www.{domain}/dp/{rand_amazon_id()}")
    urls.append(f"https://www.{domain}/dp/{rand_amazon_id()}?ref=sr_1_1")
    urls.append(f"https://www.{domain}/product/{rand_amazon_id()}")

    # ── USER ACCOUNT PATHS (/u/0/, /user/123/) ────────────────
    urls.append(f"https://www.{domain}/u/0/")
    urls.append(f"https://www.{domain}/u/{random.randint(0, 5)}/")
    urls.append(f"https://www.{domain}/user/{rand_numeric()}/")
    urls.append(f"https://www.{domain}/users/{rand_numeric()}/profile")
    urls.append(f"http://www.{domain}/u/0/")
    urls.append(f"http://www.{domain}/u/{random.randint(0,5)}/")

    # ── SUBDOMAIN VARIATIONS ─────────────────────────────────
    sub = rand_subdomain()
    urls.append(f"https://{sub}.{domain}/")
    urls.append(f"https://{sub}.{domain}/{rand_word()}")
    urls.append(f"https://{sub}.{domain}/{rand_word()}/{rand_alnum(10)}")
    urls.append(f"http://{sub}.{domain}/")
    urls.append(f"https://{sub}.{domain}/{rand_word()}{rand_utm()}")

    # ── MULTI-LEVEL SUBDOMAINS (m365.cloud.microsoft) ─────────
    nsub  = rand_numeric_sub()
    csub  = rand_cloud_sub()
    urls.append(f"https://{nsub}.{csub}.{domain}/")
    urls.append(f"https://{nsub}.{csub}.{domain}/{rand_word()}")
    urls.append(f"https://{nsub}.{csub}.{domain}/{rand_word()}/{rand_alnum(10)}")
    urls.append(f"http://{nsub}.{csub}.{domain}/")

    # ── NUMERIC ID IN PATH ────────────────────────────────────
    urls.append(f"https://www.{domain}/{rand_word()}/{rand_numeric()}/")
    urls.append(f"https://www.{domain}/{rand_word()}/{rand_numeric()}/{rand_hyphen_path()}")

    # ── COUNTRY CODE TLD VARIATIONS ──────────────────────────
    cctld = rand_cctld()
    # this creates variations like https://www.example.co.uk/ and https://example.com.au/change-speed
    urls.append(f"https://www.{base}.{cctld}/")
    urls.append(f"https://www.{base}.{cctld}/{rand_word()}")
    urls.append(f"https://{base}.{cctld}/{rand_word()}/{rand_word()}")
    urls.append(f"http://www.{base}.{cctld}/")

    return urls

# ── 6. GENERATE URL SHORTENER LEGITIMATE PATTERNS ────────────
def generate_shortener_urls():
    """Generate legitimate URL shortener patterns"""
    shorteners = ['bit.ly', 't.co', 'tinyurl.com', 'ow.ly', 'rb.gy', 'short.io']
    urls = []
    for s in shorteners:
        for _ in range(100):
            urls.append(f"https://{s}/{rand_shortener_id()}")
            urls.append(f"http://{s}/{rand_shortener_id()}")
    return urls

# ── 7. GENERATE AUGMENTED URLS ───────────────────────────────
print(f"\n── Generating Augmented Legitimate URLs ────────────────")
print(f"   ~65 URL variations per domain")
print(f"   10,000 domains + {len(newer_domains)} newer domains")
print(f"   Please wait...")

random.seed(42)
augmented_urls = []

for i, domain in enumerate(all_domains, 1):
    urls = generate_urls(domain)
    for url in urls:
        augmented_urls.append({'url': url, 'status': 1})
    if i % 10000 == 0:
        print(f"   Processed {i:,} / {len(all_domains):,} domains...")

# add URL shortener patterns
shortener_urls = generate_shortener_urls()
for url in shortener_urls:
    augmented_urls.append({'url': url, 'status': 1})
print(f"   Added {len(shortener_urls):,} URL shortener patterns")

aug_df = pd.DataFrame(augmented_urls)
print(f"\n Generated {len(aug_df):,} augmented legitimate URLs")

# ── 8. SAMPLE PREVIEW ────────────────────────────────────────
print(f"\n── Sample Generated URLs ───────────────────────────────")
for url in aug_df['url'].sample(20, random_state=42).tolist():
    print(f"   {url}")

# ── 9. COMBINE WITH ORIGINAL DATASET ─────────────────────────
print(f"\n── Combining Datasets ──────────────────────────────────")
combined = pd.concat([df, aug_df], ignore_index=True)
print(f"   Original URLs  : {len(df):,}")
print(f"   Augmented URLs : {len(aug_df):,}")
print(f"   Combined total : {len(combined):,}")

before = len(combined)
combined.drop_duplicates(subset='url', inplace=True)
print(f"   After dedup    : {len(combined):,} "
      f"(removed {before - len(combined):,})")

# ── 10. CHECK BALANCE ─────────────────────────────────────────
print(f"\n── New Label Distribution ──────────────────────────────")
print(combined['status'].value_counts())
pct_phish = (combined['status'] == 0).sum() / len(combined) * 100
pct_legit = (combined['status'] == 1).sum() / len(combined) * 100
print(f"\n   Phishing   : {pct_phish:.1f}%")
print(f"   Legitimate : {pct_legit:.1f}%")

# ── 11. SAVE ──────────────────────────────────────────────────
os.makedirs('data/processed', exist_ok=True)
combined.to_csv('data/processed/augmented_urls.csv', index=False)
print(f"\n✅ Saved: data/processed/augmented_urls.csv")
