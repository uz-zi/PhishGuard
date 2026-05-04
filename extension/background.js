// ============================================================
// PhishGuard — background.js
// Automatically checks every URL the user visits
// Uses live HuggingFace API
// ============================================================

const API_URL = "https://uzmann-phish-guard.hf.space/predict";

async function checkUrl(url, tabId) {
  // skip internal browser pages
  if (!url ||
      url.startsWith("chrome://") ||
      url.startsWith("chrome-extension://") ||
      url.startsWith("edge://") ||
      url.startsWith("about:") ||
      url.startsWith("data:")) {
    return;
  }

  try {
    const response = await fetch(API_URL, {
      method  : "POST",
      headers : { "Content-Type": "application/json" },
      body    : JSON.stringify({ url: url })
    });

    if (!response.ok) return;

    const data = await response.json();

    // store result for popup
    await chrome.storage.local.set({
      [tabId]: {
        url        : data.url,
        prediction : data.prediction,
        label      : data.label,
        confidence : data.confidence,
        message    : data.message,
        timestamp  : Date.now()
      }
    });

    // update badge
    if (data.prediction === 1) {
      chrome.action.setBadgeText({ text: "!", tabId: tabId });
      chrome.action.setBadgeBackgroundColor({ color: "#ff4757", tabId: tabId });
    } else {
      chrome.action.setBadgeText({ text: "✓", tabId: tabId });
      chrome.action.setBadgeBackgroundColor({ color: "#2ed573", tabId: tabId });
    }

    // send to content script to show banner if phishing
    try {
      await chrome.tabs.sendMessage(tabId, {
        type       : "PHISHGUARD_RESULT",
        prediction : data.prediction,
        label      : data.label,
        confidence : data.confidence,
        url        : data.url
      });
    } catch (e) {
      // content script not ready yet — ignore
    }

  } catch (error) {
    // API not reachable
    chrome.action.setBadgeText({ text: "?", tabId: tabId });
    chrome.action.setBadgeBackgroundColor({ color: "#555555", tabId: tabId });

    await chrome.storage.local.set({
      [tabId]: { url: url, error: true, timestamp: Date.now() }
    });
  }
}

// check every time a tab loads
chrome.tabs.onUpdated.addListener(function(tabId, changeInfo, tab) {
  if (changeInfo.status === "loading" && tab.url) {
    checkUrl(tab.url, tabId);
  }
});

// check when switching tabs
chrome.tabs.onActivated.addListener(async function(activeInfo) {
  const tab = await chrome.tabs.get(activeInfo.tabId);
  if (tab.url) {
    checkUrl(tab.url, activeInfo.tabId);
  }
});

chrome.runtime.onInstalled.addListener(function() {
  console.log("PhishGuard installed!");
});