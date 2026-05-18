// ============================================================
// PhishGuard — background.js
// Automatically checks every URL the user visits
// Uses live HuggingFace API (PhishGuard — 48 features)
//
// Protection layers:
//   1. Skip internal browser pages
//   2. Check known safe SLD → skip API if safe
//   3. Call PhishGuard API
// ============================================================

const API_URL = "https://uzmann-phish-guard.hf.space/predict";

// ── Known Safe Second Level Domains ──────────────────────────
// These domains are well-established legitimate platforms.
// If SLD matches → mark as legitimate without calling API.
// This prevents false positives on long search/query URLs.
const KNOWN_SAFE_SLDS = new Set([
    // Search/Tech giants
    'google', 'microsoft', 'apple', 'amazon', 'meta',
    // Social Media
    'facebook', 'youtube', 'twitter', 'instagram', 'linkedin',
    'tiktok', 'reddit', 'pinterest', 'discord', 'snapchat',
    // Developer platforms
    'github', 'stackoverflow', 'gitlab', 'bitbucket',
    'npmjs', 'pypi', 'kubernetes', 'mozilla', 'docker',
    // Cloud/Storage
    'netlify', 'vercel', 'heroku', 'cloudflare', 'dropbox',
    'onedrive', 'icloud', 'box', 'wetransfer',
    // Communication
    'zoom', 'slack', 'webex', 'teams',
    // Entertainment
    'netflix', 'spotify', 'twitch', 'vimeo', 'soundcloud',
    'dailymotion', 'tiktok',
    // AI platforms
    'openai', 'anthropic', 'huggingface', 'claude', 'chatgpt',
    'perplexity', 'midjourney',
    // Education
    'coursera', 'udemy', 'edx', 'khanacademy', 'duolingo',
    'mit', 'stanford', 'harvard', 'dbs', 'ucd', 'tcd',
    // Shopping/Finance
    'ebay', 'etsy', 'shopify', 'stripe', 'paypal',
    'wise', 'revolut', 'coinbase',
    // Tools
    'notion', 'figma', 'canva', 'grammarly', 'trello',
    'asana', 'airtable', 'miro', 'zapier',
    'spinbot', 'ilovepdf', 'smallpdf', 'mp3cut',
    // News/Media
    'bbc', 'cnn', 'reuters', 'bloomberg', 'techcrunch',
    'theguardian', 'nytimes', 'wikipedia',
    // Package repos/Dev tools
    'kaggle', 'colab', 'jupyter',
]);

// ── Extract SLD from URL ──────────────────────────────────────
function getSLD(url) {
    try {
        let hostname = new URL(url).hostname;
        if (hostname.startsWith('www.')) {
            hostname = hostname.slice(4);
        }
        const parts = hostname.split('.');
        // handle country code TLDs like co.uk, com.au
        const ccTLDs = ['uk', 'au', 'in', 'jp', 'nz', 'za', 'br', 'sg', 'ie', 'pk'];
        if (parts.length >= 3 && ccTLDs.includes(parts[parts.length - 1])) {
            return parts[parts.length - 3].toLowerCase();
        }
        return parts.length >= 2
            ? parts[parts.length - 2].toLowerCase()
            : '';
    } catch {
        return '';
    }
}

// ── Store result in storage ───────────────────────────────────
async function storeResult(tabId, url, prediction, confidence, label, message) {
    await chrome.storage.local.set({
        [tabId]: { url, prediction, label, confidence, message, timestamp: Date.now() }
    });
}

// ── Update badge ──────────────────────────────────────────────
function setBadgeGreen(tabId) {
    chrome.action.setBadgeText({ text: '✓', tabId });
    chrome.action.setBadgeBackgroundColor({ color: '#2ed573', tabId });
}

function setBadgeRed(tabId) {
    chrome.action.setBadgeText({ text: '!', tabId });
    chrome.action.setBadgeBackgroundColor({ color: '#ff4757', tabId });
}

function setBadgeGrey(tabId) {
    chrome.action.setBadgeText({ text: '?', tabId });
    chrome.action.setBadgeBackgroundColor({ color: '#555555', tabId });
}

// ── Main URL Check Function ───────────────────────────────────
async function checkUrl(url, tabId) {

    // ── Layer 1: Skip internal browser pages ─────────────────
    if (!url ||
        url.startsWith('chrome://') ||
        url.startsWith('chrome-extension://') ||
        url.startsWith('edge://')) {
        return;
    }

    // ── Layer 2: Check known safe SLD ─────────────────────────
    // If domain is a known legitimate platform →
    // mark as LEGIT without calling API.
    // Prevents false positives on long Google/YouTube URLs.
    const sld = getSLD(url);
    if (sld && KNOWN_SAFE_SLDS.has(sld)) {
        setBadgeGreen(tabId);
        await storeResult(
            tabId, url, 0, 0.99, 'legitimate',
            'Known safe domain — PhishGuard'
        );
        return; // skip API call
    }

    // ── Layer 3: Call PhishGuard API ──────────────────────────
    try {
        const response = await fetch(API_URL, {
            method : 'POST',
            headers: { 'Content-Type': 'application/json' },
            body   : JSON.stringify({ url })
        });

        if (!response.ok) return;

        const data = await response.json();

        await storeResult(
            tabId, url,
            data.prediction,
            data.confidence,
            data.label,
            data.message
        );

        if (data.prediction === 1) {
            setBadgeRed(tabId);
            // send to content script to show warning banner
            try {
                await chrome.tabs.sendMessage(tabId, {
                    type      : 'PHISHGUARD_RESULT',
                    prediction: data.prediction,
                    label     : data.label,
                    confidence: data.confidence,
                    url       : data.url
                });
            } catch (e) {
                // content script not ready yet — ignore
            }
        } else if (data.prediction === 0) {
            setBadgeGreen(tabId);
        } else {
            // phishing but low confidence — show grey badge
            setBadgeGrey(tabId);
        }

    } catch (error) {
        // API not reachable
        setBadgeGrey(tabId);
        await storeResult(
            tabId, url, -1, 0, 'unknown',
            'Cannot connect to PhishGuard API'
        );
    }
}

// ── Listen for Tab Updates ────────────────────────────────────
chrome.tabs.onUpdated.addListener(function(tabId, changeInfo, tab) {
    if (changeInfo.status === 'loading' && tab.url) {
        checkUrl(tab.url, tabId);
    }
});

// ── Listen for Tab Activation ─────────────────────────────────
chrome.tabs.onActivated.addListener(async function(activeInfo) {
    try {
        const tab = await chrome.tabs.get(activeInfo.tabId);
        if (tab.url) {
            checkUrl(tab.url, activeInfo.tabId);
        }
    } catch (e) {
        // tab may have been closed
    }
});

// ── On Install ────────────────────────────────────────────────
chrome.runtime.onInstalled.addListener(function() {
    console.log('PhishGuard installed!');
    console.log('API:', API_URL);
});