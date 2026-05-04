// ============================================================
// PhishGuard — popup.js
// Shows result when user clicks extension icon
// ============================================================

const API_URL = "https://uzmann-phish-guard.hf.space/predict";

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
    ? " Do NOT enter any personal information on this site!"
    : " This URL appears to be safe to browse.";
}

function showError(msg) {
  loadingState.classList.add("hidden");
  resultState.classList.add("hidden");
  errorState.classList.remove("hidden");
  errorText.textContent = msg;
}

async function analyseUrl(url) {
  showLoading();
  currentUrlEl.textContent = url.length > 55
    ? url.substring(0, 52) + "..."
    : url;

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
    showError(" Cannot connect to PhishGuard API.\nPlease check your internet connection.");
  }
}

async function init() {
  chrome.tabs.query({ active: true, currentWindow: true }, async function(tabs) {
    if (!tabs || !tabs[0]) return;

    const tab = tabs[0];
    const url = tab.url;

    if (!url ||
        url.startsWith("chrome://") ||
        url.startsWith("chrome-extension://") ||
        url.startsWith("about:")) {
      currentUrlEl.textContent = url || "N/A";
      showError("ℹ️ PhishGuard cannot analyse browser internal pages.");
      return;
    }

    // check storage first
    const stored = await chrome.storage.local.get(String(tab.id));
    const result = stored[String(tab.id)];

    if (result && result.url === url && !result.error) {
      currentUrlEl.textContent = url.length > 55
        ? url.substring(0, 52) + "..."
        : url;
      showResult(result);
    } else {
      analyseUrl(url);
    }
  });
}

recheckBtn.addEventListener("click", function() {
  chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
    if (tabs && tabs[0] && tabs[0].url) {
      analyseUrl(tabs[0].url);
    }
  });
});

document.addEventListener("DOMContentLoaded", init);