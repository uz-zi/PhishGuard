// ============================================================
// PhishGuard — popup.js
// Shows result when user clicks extension icon
// ============================================================

const API_URL = "https://uzmann-phish-guard.hf.space/predict";

// Google Form for reporting incorrect classifications
const REPORT_FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSefiGMhkce7JbdXTN4Fh34ZIQnmEPkFmbYUpJiHqogGtrgYwA/viewform?usp=sharing&ouid=109684724618784885160";

// ── SAME known safe list as background.js ──────────────────
const KNOWN_SAFE_SLDS = new Set([
  "google","youtube","facebook","instagram","twitter","x",
  "microsoft","apple","amazon","netflix","spotify","linkedin",
  "github","stackoverflow","wikipedia","reddit","whatsapp",
  "tiktok","snapchat","pinterest","dropbox","adobe","salesforce",
  "zoom","slack","discord","twitch","paypal","stripe","shopify",
  "wordpress","blogger","medium","substack","notion","figma",
  "canva","trello","asana","jira","atlassian","hubspot",
  "mailchimp","sendgrid","cloudflare","digitalocean","heroku",
  "netlify","vercel","firebase","mongodb","postgresql","mysql",
  "npm","pypi","docker","kubernetes","jenkins","gitlab",
  "bitbucket","codecov","travis","circleci","amazonaws","azure",
  "googleapis","gstatic","doubleclick","ggpht","googleusercontent",
  "icloud","live","outlook","hotmail","yahoo","aol","protonmail",
  "bbc","cnn","nytimes","theguardian","reuters","bloomberg",
  "dublinbusinessschool","dbs","ucd","tcd","nuigalway","dcu"
]);

// ── DOM references ─────────────────────────────────────────
const currentUrlEl   = document.getElementById("currentUrl");
const loadingState   = document.getElementById("loadingState");
const resultState    = document.getElementById("resultState");
const errorState     = document.getElementById("errorState");
const resultIcon     = document.getElementById("resultIcon");
const resultLabel    = document.getElementById("resultLabel");
const confidenceFill = document.getElementById("confidenceFill");
const confidenceText = document.getElementById("confidenceText");
const resultMessage  = document.getElementById("resultMessage");
const errorText      = document.getElementById("errorText");
const recheckBtn     = document.getElementById("recheckBtn");
const reportBtn      = document.getElementById("reportBtn");

// ── Check if URL belongs to known safe domain ──────────────
function isKnownSafe(url) {
  try {
    const hostname = new URL(url).hostname.toLowerCase();
    // Remove www. prefix
    const clean = hostname.startsWith("www.") 
      ? hostname.slice(4) 
      : hostname;
    // Get second-level domain (e.g. "google" from "google.com")
    const parts = clean.split(".");
    const sld = parts.length >= 2 ? parts[parts.length - 2] : parts[0];
    return KNOWN_SAFE_SLDS.has(sld);
  } catch {
    return false;
  }
}

// ── UI helpers ─────────────────────────────────────────────
function showLoading() {
  loadingState.classList.remove("hidden");
  resultState.classList.add("hidden");
  errorState.classList.add("hidden");
}

function showResult(data) {
  loadingState.classList.add("hidden");
  errorState.classList.add("hidden");
  resultState.classList.remove("hidden");

  const isPhishing    = data.prediction === 1;
  const confidencePct = Math.round(data.confidence * 100);

  resultIcon.textContent     = isPhishing ? "🚨" : "✅";
  resultLabel.textContent    = isPhishing ? "PHISHING" : "LEGITIMATE";
  resultLabel.className      = isPhishing ? "result-label phishing" : "result-label legitimate";
  confidenceFill.style.width = confidencePct + "%";
  confidenceFill.className   = isPhishing ? "confidence-fill phishing" : "confidence-fill legitimate";
  confidenceText.textContent = confidencePct + "% confidence";
  resultMessage.textContent  = isPhishing
    ? "⚠️ Do NOT enter any personal information on this site!"
    : "This URL appears to be safe to browse.";
}

function showSafe(url) {
  loadingState.classList.add("hidden");
  errorState.classList.add("hidden");
  resultState.classList.remove("hidden");

  resultIcon.textContent     = "✅";
  resultLabel.textContent    = "LEGITIMATE";
  resultLabel.className      = "result-label legitimate";
  confidenceFill.style.width = "100%";
  confidenceFill.className   = "confidence-fill legitimate";
  confidenceText.textContent = "100% confidence";
  resultMessage.textContent  = "Trusted domain — known safe website.";
}

function showError(msg) {
  loadingState.classList.add("hidden");
  resultState.classList.add("hidden");
  errorState.classList.remove("hidden");
  errorText.textContent = msg;
}

// ── Core analysis function — same logic as background.js ───
async function analyseUrl(url) {
  showLoading();
  currentUrlEl.textContent = url.length > 55
    ? url.substring(0, 52) + "..."
    : url;

  // LAYER 1 — Check known safe domains first
  if (isKnownSafe(url)) {
    showSafe(url);
    return;
  }

  // LAYER 2 — Call API for everything else
  try {
    const response = await fetch(API_URL, {
      method  : "POST",
      headers : { "Content-Type": "application/json" },
      body    : JSON.stringify({ url: url })
    });

    if (!response.ok) throw new Error(`API error: ${response.status}`);

    const data = await response.json();
    showResult(data);

  } catch (error) {
    showError("⚠️ Cannot connect to PhishGuard API.\nPlease check your internet connection.");
  }
}

// ── Init on popup open ─────────────────────────────────────
async function init() {
  chrome.tabs.query({ active: true, currentWindow: true }, async function(tabs) {
    if (!tabs || !tabs[0]) return;

    const tab = tabs[0];
    const url = tab.url;

    // Skip internal browser pages
    if (!url ||
        url.startsWith("chrome://") ||
        url.startsWith("chrome-extension://") ||
        url.startsWith("about:") ||
        url.startsWith("edge://") ||
        url.startsWith("file://")) {
      currentUrlEl.textContent = url || "N/A";
      showError("PhishGuard cannot analyse browser internal pages.");
      return;
    }

    currentUrlEl.textContent = url.length > 55
      ? url.substring(0, 52) + "..."
      : url;

    // Check storage cache first
    const stored = await chrome.storage.local.get(String(tab.id));
    const result = stored[String(tab.id)];

    if (result && result.url === url && !result.error) {
      showResult(result);
    } else {
      // Run full analysis including known safe check
      analyseUrl(url);
    }
  });
}

// ── Recheck button — runs FULL analysis with protection layer
recheckBtn.addEventListener("click", function() {
  chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
    if (tabs && tabs[0] && tabs[0].url) {
      analyseUrl(tabs[0].url);  // includes known safe check
    }
  });
});

// ── Report button ──────────────────────────────────────────
reportBtn.addEventListener("click", function() {
  chrome.tabs.create({ url: REPORT_FORM_URL });
});

document.addEventListener("DOMContentLoaded", init);