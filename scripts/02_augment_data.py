# ============================================================
# PhishGuard v4 — Step 2: Data Augmentation (COMPLETE FINAL)
# Uses Tranco top-5000 + manual domains
#
# Author : Uzman Zahid
# ============================================================

import pandas as pd
import random
import string
import uuid
import os

print("=" * 60)
print("PHISHGUARD v4 — STEP 2: DATA AUGMENTATION")
print("=" * 60)

# ── 1. LOAD EXISTING DATASET ─────────────────────────────────
df = pd.read_csv('data/processed/clean_urls.csv')
print(f"\nExisting dataset loaded: {len(df):,} URLs")
print(f"Phishing   (0): {(df['status']==0).sum():,}")
print(f"Legitimate (1): {(df['status']==1).sum():,}")

# ── 2. LOAD TRANCO LIST ──────────────────────────────────────
tranco = pd.read_csv('data/raw/top-1m.csv', header=None,
                     names=['rank', 'domain'])
print(f"\nTranco list loaded: {len(tranco):,} domains")
top_domains = tranco[tranco['rank'] <= 3000]['domain'].tolist()
print(f"Using top 3,000 Tranco domains")

# ── 3. SPECIFIC DOMAINS ──────────────────────────────────────
specific_domains = [
    # AI platforms
    'openai.com', 'claude.ai', 'anthropic.com', 'perplexity.ai',
    'huggingface.co', 'chatgpt.com', 'character.ai',
    'gemini.google.com', 'copilot.microsoft.com',
    # Developer/hosting
    'netlify.app', 'netlify.com', 'vercel.app', 'vercel.com',
    'railway.app', 'render.com', 'fly.io', 'supabase.com',
    'githubusercontent.com', 'githubassets.com',
    'pages.github.com', 'github.io',
    # CDN/Raw
    'cdn.jsdelivr.net', 'unpkg.com', 'cdnjs.cloudflare.com',
    'raw.githubusercontent.com', 'assets.vercel.com',
    'dl.google.com', 'objects.githubusercontent.com',
    # Cloud/Storage
    'drive.google.com', 'onedrive.live.com', 'dropbox.com',
    'box.com', 'icloud.com', 'mega.nz', 'wetransfer.com',
    # Communication
    'discord.com', 'slack.com', 'teams.microsoft.com',
    'zoom.us', 'meet.google.com', 'webex.com',
    'mail.google.com', 'outlook.live.com',
    'hooks.slack.com', 'api.telegram.org',
    # Education
    'coursera.org', 'udemy.com', 'edx.org', 'khanacademy.org',
    'duolingo.com', 'brilliant.org', 'pluralsight.com',
    'nu.edu.pk', 'dbs.ie', 'ucd.ie', 'tcd.ie',
    'mit.edu', 'stanford.edu', 'cam.ac.uk',
    # Tools/Converters (failing domains)
    'spinbot.com', 'ilovepdf.com', 'smallpdf.com',
    'mp3cut.net', 'convertio.co', 'tinypng.com',
    'canva.com', 'grammarly.com', 'notion.so',
    'airtable.com', 'trello.com', 'asana.com',
    'figma.com', 'miro.com',
    # Shopping/Finance
    'etsy.com', 'ebay.com', 'shopify.com', 'stripe.com',
    'wise.com', 'revolut.com', 'coinbase.com', 'paypal.com',
    'dashboard.stripe.com', 'app.paypal.com',
    'checkout.stripe.com', 'buy.stripe.com',
    # Developer consoles
    'chromewebstore.google.com', 'developers.google.com',
    'developers.openai.com', 'platform.openai.com',
    'console.firebase.google.com', 'console.cloud.google.com',
    'portal.azure.com', 'app.netlify.com',
    'docs.python.org', 'docs.microsoft.com',
    'developer.mozilla.org', 'api.github.com',
    'kubernetes.io', 'hub.docker.com',
    # Entertainment/Media (failing domains)
    'netflix.com', 'spotify.com', 'twitch.tv', 'tiktok.com',
    'pinterest.com', 'reddit.com', 'quora.com',
    'open.spotify.com', 'music.apple.com', 'soundcloud.com',
    'vimeo.com', 'dailymotion.com',
    # Social Media
    'linkedin.com', 'twitter.com', 'instagram.com',
    'facebook.com', 'medium.com', 'dev.to',
    # News/Media
    'bbc.com', 'techcrunch.com', 'theguardian.com',
    'reuters.com', 'bloomberg.com',
    # Package repos
    'npmjs.com', 'pypi.org', 'packagist.org',
    # Maps
    'maps.google.com', 'openstreetmap.org', 'maps.apple.com',
    # Government
    'gov.ie', 'gov.uk', 'usa.gov', 'europa.eu',
    'hse.ie', 'rte.ie',
    # URL shorteners
    'bit.ly', 't.co', 'tinyurl.com', 'ow.ly', 'rb.gy',
    'link.medium.com', 'go.onelink.me', 'app.adjust.com',
    # .io domains (kubernetes, etc)
    'kubernetes.io', 'socket.io', 'zeit.co', 'fastapi.io',
    'python.org', 'rust-lang.org', 'golang.org',
    # Personal
    'uzmanzahid.netlify.app',
]

all_domains = list(set(top_domains + specific_domains))
print(f"Added {len(specific_domains)} specific domains")
print(f"Total domains: {len(all_domains):,}")

# ── 4. HELPER FUNCTIONS ──────────────────────────────────────
def rand_alnum(n=12):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=n))

def rand_uuid():
    return str(uuid.UUID(int=random.getrandbits(128)))

def rand_amazon_id():
    return 'B0' + ''.join(random.choices(
        string.ascii_uppercase + string.digits, k=8))

def rand_numeric():
    return str(random.randint(1000000, 999999999))

def rand_short_numeric():
    """Short numeric ID like vimeo uses"""
    return str(random.randint(100000, 999999999))

def rand_word():
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
        'mail', 'spam', 'inbox', 'sent', 'drafts', 'trash',
        'guides', 'tutorials', 'examples', 'reference',
        'extensions', 'plugins', 'themes', 'templates',
        'productivity', 'tools', 'communication', 'education',
        'technology', 'science', 'sports', 'health', 'finance',
        'package', 'library', 'module', 'framework', 'plugin',
        'track', 'album', 'artist', 'playlist', 'podcast',
        'listing', 'item', 'product', 'order', 'checkout',
        'article', 'post', 'story', 'video', 'image', 'photo',
        'concepts', 'configuration', 'installation', 'getting-started',
        'compress-pdf', 'merge-pdf', 'split-pdf', 'convert-pdf',
    ]
    return random.choice(words)

def rand_username():
    names = ['john', 'jane', 'alex', 'sara', 'mike', 'uzman',
             'developer', 'user123', 'techguy', 'coder',
             'datascientist', 'student', 'professor', 'admin',
             'uzman-zahid', 'myportfolio', 'personal-blog']
    return random.choice(names)

def rand_hyphen_path():
    w1, w2, w3 = rand_word(), rand_word(), rand_word()
    n = random.randint(1, 3)
    if n == 1:   return f"{w1}-{w2}"
    elif n == 2: return f"{w1}-{w2}-{w3}"
    else:        return f"{w1}-{w2}"

def rand_slug():
    return '-'.join([rand_word() for _ in range(random.randint(2, 5))])

def rand_lang():
    return random.choice([
        'en-US', 'en-GB', 'en-IE', 'fr-FR',
        'de-DE', 'es-ES', 'it-IT', 'pt-BR', 'en'
    ])

def rand_utm():
    sources  = ['google', 'youtube', 'facebook', 'twitter',
                'instagram', 'email', 'newsletter', 'bing',
                'linkedin', 'reddit', 'organic', 'direct',
                'YouTube', 'Gmail', 'Chrome', 'ext_sidebar',
                'sidebar', 'nav', 'header', 'footer']
    mediums  = ['cpc', 'social', 'email', 'organic',
                'referral', 'display', 'video', 'banner']
    source   = random.choice(sources)
    medium   = random.choice(mediums)
    campaign = rand_word()
    lang     = rand_lang()
    n        = random.randint(1, 4)
    if n == 1:   return f"?utm_source={source}"
    elif n == 2: return f"?utm_source={source}&utm_medium={medium}"
    elif n == 3: return f"?utm_source={source}&utm_medium={medium}&utm_campaign={campaign}"
    else:        return f"?hl={lang}&utm_source={source}"

def rand_subdomain():
    return random.choice([
        'myaccount', 'account', 'accounts', 'drive', 'mail',
        'docs', 'maps', 'play', 'store', 'photos', 'calendar',
        'meet', 'chat', 'support', 'help', 'blog', 'news',
        'shop', 'app', 'api', 'dev', 'portal', 'dashboard',
        'secure', 'login', 'signin', 'auth', 'id',
        'raw', 'developers', 'platform', 'console', 'admin',
        'elearning', 'learn', 'courses', 'library', 'wiki',
        'lhr', 'khi', 'isb', 'open', 'music', 'dl', 'download',
        'checkout', 'buy', 'hooks', 'link', 'go',
        'chromewebstore', 'marketplace', 'extensions',
    ])

def rand_numeric_sub():
    return random.choice(['m365', 'o365', 'web3', 'app1', 'v2', 'v3', 'beta'])

def rand_cloud_sub():
    return random.choice(['cloud', 'azure', 'aws', 'gcp', 'cdn', 'static'])

def rand_anchor():
    return random.choice([
        'inbox', 'spam', 'sent', 'drafts', 'trash',
        'section1', 'overview', 'details', 'comments',
        'top', 'main', 'content', 'nav', 'footer',
        'themes', 'extensions', 'productivity', 'tools',
        'map', 'location', 'results', 'reviews',
    ])

def rand_encoded_path():
    words = [rand_word(), rand_word()]
    return '%2F' + '%2F'.join(words)

def rand_cctld():
    return random.choice([
        'co.uk', 'com.au', 'co.in', 'co.jp', 'co.nz',
        'co.za', 'com.br', 'co.ke', 'com.sg', 'ie',
        'ac.uk', 'edu.pk', 'gov.uk', 'org.uk',
    ])

def rand_oauth_param():
    return random.choice([
        f"client_id={rand_alnum(20)}",
        f"redirect_uri=https%3A%2F%2F{rand_word()}.com",
        f"scope={rand_word()}+{rand_word()}",
        f"response_type=code&client_id={rand_alnum(10)}",
    ])

def rand_coordinates():
    lat = round(random.uniform(-90, 90), 4)
    lng = round(random.uniform(-180, 180), 4)
    return f"{lat},{lng}"

def rand_year():
    return str(random.randint(2020, 2026))

def rand_month():
    return str(random.randint(1, 12)).zfill(2)

def rand_day():
    return str(random.randint(1, 28)).zfill(2)

def rand_io_domain():
    """Common .io domains"""
    return random.choice([
        'kubernetes.io', 'socket.io', 'fastapi.io',
        'fly.io', 'railway.app', 'render.com',
        'python.org', 'rust-lang.org', 'golang.org',
        'pypi.org', 'npmjs.com',
    ])

# ── 5. MAIN URL GENERATOR ─────────────────────────────────────
def generate_urls(domain):
    """Generates ~130 URL variations covering all 25 categories + 8 fixes."""
    urls = []
    parts = domain.split('.')
    base  = parts[0] if len(parts) >= 2 else domain

    # ── FIX 1: SIMPLE HOMEPAGES (www.google.com/) ────────────
    # More homepage variations to fix bare domain failures
    urls.append(f"https://www.{domain}/")
    urls.append(f"https://{domain}/")
    urls.append(f"http://www.{domain}/")
    urls.append(f"http://{domain}/")
    # extra homepage patterns with trailing content
    urls.append(f"https://www.{domain}/#")
    urls.append(f"https://{domain}/#home")
    urls.append(f"https://www.{domain}/?ref=homepage")
    urls.append(f"https://{domain}/?lang={rand_lang()}")

    # ── FIX 2: BARE NO-WWW TOOLS (spinbot.com, ilovepdf.com) ─
    # More no-www short domain patterns
    urls.append(f"https://{domain}/{rand_hyphen_path()}")
    urls.append(f"https://{domain}/{rand_word()}-{rand_word()}")
    urls.append(f"https://{domain}/{rand_word()}/")
    urls.append(f"http://{domain}/{rand_hyphen_path()}")
    urls.append(f"https://{domain}/{rand_word()}.php")
    urls.append(f"https://{domain}/{rand_word()}.html")

    # ── FIX 3: NUMERIC-ONLY PATHS (vimeo.com/123456789) ──────
    urls.append(f"https://www.{domain}/{rand_short_numeric()}")
    urls.append(f"https://{domain}/{rand_short_numeric()}")
    urls.append(f"https://www.{domain}/{rand_short_numeric()}/")
    urls.append(f"https://{domain}/{rand_short_numeric()}/")
    urls.append(f"https://www.{domain}/{rand_numeric()}")
    urls.append(f"https://{domain}/{rand_numeric()}")

    # ── FIX 4: .IO TLD DOMAINS (kubernetes.io) ───────────────
    # .io domains with docs paths
    urls.append(f"https://{domain}/docs/")
    urls.append(f"https://{domain}/docs/{rand_word()}/")
    urls.append(f"https://{domain}/docs/{rand_word()}/{rand_word()}/")
    urls.append(f"https://{domain}/docs/{rand_word()}/{rand_hyphen_path()}/")
    urls.append(f"https://docs.{domain}/")
    urls.append(f"https://docs.{domain}/{rand_word()}/")

    # ── FIX 5: LONG SUBDOMAIN HOMEPAGE (chromewebstore.google.com/) ─
    urls.append(f"https://chromewebstore.{domain}/")
    urls.append(f"https://chromewebstore.{domain}/?hl={rand_lang()}")
    urls.append(f"https://marketplace.{domain}/")
    urls.append(f"https://marketplace.{domain}/?hl={rand_lang()}")
    urls.append(f"https://extensions.{domain}/")
    urls.append(f"https://extensions.{domain}/?hl={rand_lang()}")

    # ── FIX 6: DRIVE SUBDOMAIN (drive.google.com/drive/) ─────
    urls.append(f"https://drive.{domain}/drive/")
    urls.append(f"https://drive.{domain}/drive/folders/{rand_alnum(20)}")
    urls.append(f"https://drive.{domain}/drive/my-drive")
    urls.append(f"https://drive.{domain}/drive/shared-with-me")
    urls.append(f"https://drive.{domain}/")

    # ── FIX 7: PERSONAL NETLIFY/VERCEL (uzmanzahid.netlify.app/) ─
    names = ['uzman', 'john', 'jane', 'alex', 'mike', 'sara',
             'portfolio', 'mysite', 'personal', 'blog',
             'uzmanzahid', 'johnsmith', 'developer']
    name  = random.choice(names)
    urls.append(f"https://{name}.{domain}/")
    urls.append(f"https://{name}.{domain}/{rand_word()}")
    urls.append(f"https://{name}-{rand_word()}.{domain}/")
    urls.append(f"https://{name}.{domain}/projects")
    urls.append(f"https://{name}.{domain}/about")

    # ── CAT 4: UTM TRACKING ───────────────────────────────────
    urls.append(f"https://www.{domain}/{rand_word()}{rand_utm()}")
    urls.append(f"https://www.{domain}/{rand_word()}/{rand_word()}{rand_utm()}")
    urls.append(f"http://www.{domain}/{rand_word()}{rand_utm()}")
    urls.append(f"https://{domain}/{rand_word()}{rand_utm()}")

    # ── CAT 4: SIDEBAR/NAVIGATION (chromewebstore) ────────────
    urls.append(
        f"https://www.{domain}/category/{rand_word()}/{rand_word()}"
        f"?hl={rand_lang()}&utm_source=ext_sidebar"
    )
    urls.append(f"https://www.{domain}/category/{rand_word()}?hl={rand_lang()}")
    urls.append(f"https://www.{domain}/?hl={rand_lang()}")
    urls.append(f"https://{domain}/?hl={rand_lang()}")

    # ── CAT 4: ANCHOR/HASH (Gmail #spam) ──────────────────────
    urls.append(f"https://mail.{domain}/mail/u/0/#{rand_anchor()}")
    urls.append(f"https://mail.{domain}/mail/u/0/")
    urls.append(f"https://mail.{domain}/mail/u/{random.randint(0,3)}/")
    urls.append(f"https://www.{domain}/{rand_word()}/#{rand_anchor()}")
    urls.append(f"https://{domain}/#{rand_anchor()}")

    # ── CAT 6: FINANCIAL DASHBOARD + REDIRECT ─────────────────
    urls.append(f"https://dashboard.{domain}/login?redirect={rand_encoded_path()}")
    urls.append(f"https://dashboard.{domain}/{rand_word()}")
    urls.append(f"https://dashboard.{domain}/")
    urls.append(f"https://app.{domain}/login?redirect=%2F{rand_word()}%2F{rand_word()}")
    urls.append(f"https://secure.{domain}/{rand_word()}?next=%2F{rand_word()}")

    # ── CAT 9: URL ENCODED ────────────────────────────────────
    urls.append(
        f"https://accounts.{domain}/signin"
        f"?continue=https%3A%2F%2F{rand_word()}.{domain}"
    )
    urls.append(
        f"https://www.{domain}/{rand_word()}"
        f"?redirect_uri=https%3A%2F%2F{rand_word()}.com%2F{rand_word()}"
    )
    urls.append(f"https://{domain}/{rand_word()}?next=%2F{rand_word()}%2F{rand_word()}")

    # ── CAT 12: OAUTH/REDIRECT ────────────────────────────────
    urls.append(f"https://{domain}/login/oauth/authorize?{rand_oauth_param()}")
    urls.append(
        f"https://accounts.{domain}/o/oauth2/auth"
        f"?redirect_uri={rand_word()}&{rand_oauth_param()}"
    )
    urls.append(
        f"https://login.{domain}/common/oauth2/authorize"
        f"?{rand_oauth_param()}"
    )

    # ── CAT 3: DEVELOPER/API ──────────────────────────────────
    urls.append(f"https://developers.{domain}/{rand_word()}/{rand_word()}")
    urls.append(
        f"https://developers.{domain}"
        f"/{rand_word()}/{rand_word()}/{rand_hyphen_path()}"
    )
    urls.append(f"https://api.{domain}/{rand_word()}/{rand_word()}")
    urls.append(f"https://api.{domain}/v{random.randint(1,3)}/{rand_word()}")
    urls.append(f"https://console.{domain}/{rand_word()}/{rand_word()}")
    urls.append(f"https://platform.{domain}/{rand_word()}")

    # ── CAT 11: RAW/CDN ───────────────────────────────────────
    urls.append(
        f"https://raw.{domain}/{rand_word()}/{rand_word()}"
        f"/refs/heads/main/{rand_word()}.txt"
    )
    urls.append(f"https://cdn.{domain}/npm/{rand_word()}/")
    urls.append(f"https://cdn.{domain}/{rand_word()}/{rand_alnum(8)}.js")
    urls.append(f"https://dl.{domain}/{rand_word()}/{rand_word()}.exe")
    urls.append(f"https://download.{domain}/?product={rand_word()}")

    # ── CAT 8: LONG QUERY STRINGS ─────────────────────────────
    urls.append(
        f"https://www.{domain}/search"
        f"?q={rand_word()}+{rand_word()}&hl={rand_lang()}&source=hp"
    )
    urls.append(
        f"https://www.{domain}/results"
        f"?search_query={rand_word()}+{rand_word()}&sp=CAI"
    )
    urls.append(
        f"https://www.{domain}/{rand_word()}"
        f"?page={random.randint(1,10)}&sort=newest&filter={rand_word()}"
    )

    # ── CAT 13: SOCIAL MEDIA PROFILES ─────────────────────────
    username = rand_username()
    urls.append(f"https://www.{domain}/in/{username}/")
    urls.append(f"https://www.{domain}/profile.php?id={rand_numeric()}")
    urls.append(f"https://www.{domain}/{username}/status/{rand_numeric()}")
    urls.append(f"https://www.{domain}/p/{rand_alnum(11)}/")
    urls.append(f"https://www.{domain}/@{username}/video/{rand_numeric()}")
    urls.append(f"https://www.{domain}/@{username}/")

    # ── CAT 14: E-COMMERCE PRODUCTS ───────────────────────────
    urls.append(f"https://www.{domain}/dp/{rand_amazon_id()}")
    urls.append(f"https://www.{domain}/dp/{rand_amazon_id()}?ref=sr_1_1")
    urls.append(f"https://www.{domain}/itm/{rand_hyphen_path()}/{rand_numeric()}")
    urls.append(f"https://www.{domain}/listing/{rand_numeric()}/{rand_slug()}")
    urls.append(f"https://www.{domain}/item/{rand_numeric()}.html")

    # ── CAT 15: NEWS/BLOG ARTICLES ────────────────────────────
    urls.append(f"https://www.{domain}/news/{rand_word()}-{rand_numeric()}")
    urls.append(
        f"https://{domain}/{rand_year()}/{rand_month()}/{rand_day()}"
        f"/{rand_slug()}/"
    )
    urls.append(f"https://{domain}/@{rand_username()}/{rand_slug()}-{rand_alnum(6)}")
    urls.append(f"https://{domain}/{rand_username()}/{rand_slug()}-{rand_numeric()}")

    # ── CAT 16: VIDEO PLATFORMS ───────────────────────────────
    urls.append(f"https://www.{domain}/watch?v={rand_alnum(11)}")
    urls.append(
        f"https://www.{domain}/watch?v={rand_alnum(11)}"
        f"&list={rand_alnum(20)}&index={random.randint(1,50)}"
    )
    urls.append(f"https://www.{domain}/{rand_username()}/clip/{rand_alnum(10)}")
    # numeric-only paths (vimeo style)
    urls.append(f"https://www.{domain}/{rand_short_numeric()}")
    urls.append(f"https://{domain}/{rand_short_numeric()}")
    urls.append(f"https://www.{domain}/video/x{rand_alnum(6)}")

    # ── CAT 17: PACKAGE REPOSITORIES ─────────────────────────
    urls.append(f"https://www.{domain}/package/{rand_hyphen_path()}")
    urls.append(f"https://www.{domain}/project/{rand_hyphen_path()}/")
    urls.append(f"https://www.{domain}/packages/{rand_word()}/{rand_hyphen_path()}")
    urls.append(f"https://www.{domain}/r/{rand_username()}/{rand_word()}")

    # ── CAT 18: DOCUMENTATION ─────────────────────────────────
    urls.append(
        f"https://docs.{domain}/{random.randint(1,4)}/"
        f"{rand_word()}/{rand_word()}.html"
    )
    urls.append(f"https://docs.{domain}/en-us/{rand_word()}/{rand_word()}/")
    urls.append(f"https://developer.{domain}/en-US/docs/{rand_word()}/{rand_word()}/")
    urls.append(f"https://{domain}/docs/{rand_word()}/{rand_word()}/")
    urls.append(f"https://{domain}/docs/{rand_word()}/")
    urls.append(f"https://{domain}/docs/")

    # ── CAT 20: STREAMING/MEDIA ───────────────────────────────
    urls.append(f"https://open.{domain}/track/{rand_alnum(22)}")
    urls.append(f"https://open.{domain}/playlist/{rand_alnum(22)}")
    urls.append(f"https://music.{domain}/us/album/{rand_hyphen_path()}/{rand_numeric()}")
    urls.append(f"https://www.{domain}/{rand_username()}/{rand_hyphen_path()}")

    # ── CAT 21: MAPS/LOCATION ─────────────────────────────────
    coords = rand_coordinates()
    urls.append(
        f"https://www.{domain}/maps/place/{rand_word()}+{rand_word()}"
        f"/@{coords}"
    )
    urls.append(f"https://maps.{domain}/?q={rand_word()}&ll={coords}")
    urls.append(f"https://www.{domain}/#{rand_word()}=12/{coords}")

    # ── CAT 22: PAYMENT/CHECKOUT ──────────────────────────────
    urls.append(f"https://checkout.{domain}/pay/{rand_alnum(20)}")
    urls.append(f"https://buy.{domain}/{rand_alnum(10)}")
    urls.append(f"https://www.{domain}/checkoutnow?token={rand_alnum(15)}")

    # ── CAT 23: SEARCH RESULTS ────────────────────────────────
    urls.append(
        f"https://www.{domain}/search"
        f"?q={rand_word()}+{rand_word()}&hl={rand_lang()}"
        f"&start={random.randint(0,100)}"
    )
    urls.append(f"https://www.{domain}/search?q={rand_word()}&tbm=isch&hl={rand_lang()}")

    # ── CAT 24: MOBILE DEEP LINKS ─────────────────────────────
    urls.append(f"https://app.{domain}/{rand_alnum(8)}?campaign={rand_word()}")
    urls.append(f"https://link.{domain}/{rand_alnum(8)}")
    urls.append(f"https://go.{domain}/{rand_word()}/{rand_alnum(6)}")

    # ── CAT 25: WEBHOOKS/CALLBACKS ────────────────────────────
    urls.append(
        f"https://hooks.{domain}/services"
        f"/{rand_alnum(8)}/{rand_alnum(8)}/{rand_alnum(24)}"
    )
    urls.append(
        f"https://api.{domain}/bot{rand_numeric()}:{rand_alnum(35)}/sendMessage"
    )
    urls.append(
        f"https://{domain}/api/webhooks/{rand_numeric()}/{rand_alnum(20)}"
    )

    # ── UUID PATHS ────────────────────────────────────────────
    urls.append(f"https://www.{domain}/chat/{rand_uuid()}")
    urls.append(f"https://www.{domain}/doc/{rand_uuid()}")
    urls.append(f"https://www.{domain}/{rand_word()}/{rand_uuid()}")
    urls.append(f"https://{domain}/{rand_word()}/{rand_uuid()}")

    # ── USER ACCOUNT PATHS ────────────────────────────────────
    urls.append(f"https://www.{domain}/u/0/")
    urls.append(f"https://www.{domain}/u/{random.randint(0,5)}/")
    urls.append(f"https://www.{domain}/user/{rand_numeric()}/")
    urls.append(f"http://www.{domain}/u/0/")

    # ── SUBDOMAIN VARIATIONS ──────────────────────────────────
    sub = rand_subdomain()
    urls.append(f"https://{sub}.{domain}/")
    urls.append(f"https://{sub}.{domain}/{rand_word()}")
    urls.append(f"http://{sub}.{domain}/")
    urls.append(f"http://{sub}.{domain}/u/0/?utm_source=YouTube")

    # ── MULTI-LEVEL SUBDOMAINS ────────────────────────────────
    nsub = rand_numeric_sub()
    csub = rand_cloud_sub()
    urls.append(f"https://{nsub}.{csub}.{domain}/")
    urls.append(f"https://{nsub}.{csub}.{domain}/{rand_word()}")

    # ── CAT 1: EDUCATIONAL PATTERNS ───────────────────────────
    cities = ['lhr', 'khi', 'isb', 'mtn', 'psh', 'fsd', 'lnd', 'nyc']
    city   = random.choice(cities)
    urls.append(f"https://{city}.{domain}/")
    urls.append(f"https://elearning.{domain}/user/profile.php")
    urls.append(f"https://elearning.{domain}/course/view.php?id={rand_numeric()}")
    urls.append(
        f"https://elearning.{domain}/pluginfile.php"
        f"/{rand_numeric()}/mod_{rand_word()}/{rand_word()}"
    )

    # ── CAT 2: GOVERNMENT PATTERNS ────────────────────────────
    urls.append(f"https://www.{domain}/en/")
    urls.append(f"https://www.{domain}/en/{rand_word()}/")

    # ── CAT 7: COUNTRY CODE TLD VARIATIONS ───────────────────
    cctld = rand_cctld()
    urls.append(f"https://www.{base}.{cctld}/")
    urls.append(f"https://www.{base}.{cctld}/{rand_word()}")
    urls.append(f"https://{base}.{cctld}/{rand_word()}/{rand_word()}")
    urls.append(f"http://www.{base}.{cctld}/")

    return urls


# ── 6. URL SHORTENER PATTERNS ─────────────────────────────────
def generate_shortener_urls():
    shorteners = ['bit.ly', 't.co', 'tinyurl.com', 'ow.ly', 'rb.gy']
    urls = []
    for s in shorteners:
        for _ in range(50):
            sid = ''.join(random.choices(
                string.ascii_letters + string.digits,
                k=random.randint(5, 8)))
            urls.append(f"https://{s}/{sid}")
            urls.append(f"http://{s}/{sid}")
    return urls


# ── 7. GENERATE ──────────────────────────────────────────────
print(f"\n── Generating Augmented Legitimate URLs ────────────────")
print(f"~130 URL variations per domain")
print(f"{len(all_domains):,} total domains")

random.seed(42)
augmented_urls = []

for i, domain in enumerate(all_domains, 1):
    urls = generate_urls(domain)
    for url in urls:
        augmented_urls.append({'url': url, 'status': 1})
    if i % 1000 == 0:
        print(f"Processed {i:,} / {len(all_domains):,} domains...")

shortener_urls = generate_shortener_urls()
for url in shortener_urls:
    augmented_urls.append({'url': url, 'status': 1})
print(f"   Added {len(shortener_urls):,} URL shortener patterns")

aug_df = pd.DataFrame(augmented_urls)
print(f"\nGenerated {len(aug_df):,} augmented legitimate URLs")

# ── 8. SAMPLE PREVIEW ────────────────────────────────────────
print(f"\n── Sample Generated URLs ───────────────────────────────")
for url in aug_df['url'].sample(20, random_state=42).tolist():
    print(f"   {url}")

# ── 9. COMBINE ────────────────────────────────────────────────
print(f"\n── Combining Datasets ──────────────────────────────────")
combined = pd.concat([df, aug_df], ignore_index=True)
print(f"   Original URLs  : {len(df):,}")
print(f"   Augmented URLs : {len(aug_df):,}")
print(f"   Combined total : {len(combined):,}")

before = len(combined)
combined.drop_duplicates(subset='url', inplace=True)
print(f"   After dedup    : {len(combined):,} "
      f"(removed {before - len(combined):,})")

# ── 10. BALANCE CHECK ─────────────────────────────────────────
print(f"\n── New Label Distribution ──────────────────────────────")
phish = (combined['status'] == 0).sum()
legit = (combined['status'] == 1).sum()
total = len(combined)
print(combined['status'].value_counts())
print(f"\n   Phishing   : {phish:,} ({phish/total*100:.1f}%)")
print(f"   Legitimate : {legit:,} ({legit/total*100:.1f}%)")

if 45 <= phish/total*100 <= 55:
    print(f"Excellent balance!")
elif 35 <= phish/total*100 <= 65:
    print(f"Acceptable balance!")
else:
    print(f"Imbalanced!")

# ── 11. SAVE ──────────────────────────────────────────────────
os.makedirs('data/processed', exist_ok=True)
combined.to_csv('data/processed/augmented_urls.csv', index=False)
print(f"\n✅ Saved: data/processed/augmented_urls.csv")

print("\n" + "=" * 60)
print("STEP 2 COMPLETE — Ready for Feature Extraction!")
print("=" * 60)