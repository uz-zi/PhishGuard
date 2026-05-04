// ============================================================
// PhishGuard — content.js
// Injected into every page
// Shows red warning banner if URL is phishing
// ============================================================

chrome.runtime.onMessage.addListener(function(message) {
  if (message.type === "PHISHGUARD_RESULT" && message.prediction === 1) {
    showWarningBanner(message.confidence, message.url);
  }
});

function showWarningBanner(confidence, url) {
  if (document.getElementById("phishguard-banner")) return;

  const pct    = Math.round(confidence * 100);
  const banner = document.createElement("div");
  banner.id    = "phishguard-banner";

  banner.innerHTML = `
    <div id="pg-inner">
      <div id="pg-left">
        <div id="pg-icon">🚨</div>
        <div id="pg-content">
          <div id="pg-title">⚠️ PHISHING WARNING — PhishGuard</div>
          <div id="pg-subtitle">This website has been flagged as a potential phishing site (${pct}% confidence). Your personal information may be at risk!</div>
          <div id="pg-url">${url}</div>
        </div>
      </div>
      <div id="pg-actions">
        <button id="pg-leave">🔒 Leave This Site</button>
        <button id="pg-ignore">Ignore</button>
      </div>
      <button id="pg-close">✕</button>
    </div>
  `;

  banner.style.cssText = `
    position:fixed!important;top:0!important;left:0!important;
    width:100%!important;z-index:2147483647!important;
    background:linear-gradient(135deg,#1a0505,#2d0a0a)!important;
    border-bottom:3px solid #ff4757!important;
    font-family:'Segoe UI',Arial,sans-serif!important;
    box-shadow:0 4px 20px rgba(255,71,87,0.5)!important;
  `;

  const style       = document.createElement("style");
  style.textContent = `
    #pg-inner{display:flex!important;align-items:center!important;justify-content:space-between!important;padding:12px 20px!important;gap:16px!important;}
    #pg-left{display:flex!important;align-items:center!important;gap:14px!important;flex:1!important;}
    #pg-icon{font-size:30px!important;flex-shrink:0!important;}
    #pg-content{flex:1!important;}
    #pg-title{font-size:14px!important;font-weight:700!important;color:#ff4757!important;margin-bottom:4px!important;}
    #pg-subtitle{font-size:12px!important;color:#ffaaaa!important;line-height:1.4!important;margin-bottom:4px!important;}
    #pg-url{font-size:11px!important;color:#884444!important;word-break:break-all!important;font-family:monospace!important;}
    #pg-actions{display:flex!important;gap:8px!important;flex-shrink:0!important;}
    #pg-leave{background:#ff4757!important;color:white!important;border:none!important;padding:8px 16px!important;border-radius:6px!important;cursor:pointer!important;font-size:13px!important;font-weight:600!important;}
    #pg-ignore{background:transparent!important;color:#884444!important;border:1px solid #442222!important;padding:8px 12px!important;border-radius:6px!important;cursor:pointer!important;font-size:12px!important;}
    #pg-close{background:transparent!important;color:#884444!important;border:none!important;font-size:20px!important;cursor:pointer!important;padding:4px!important;flex-shrink:0!important;}
    #pg-close:hover{color:#ff4757!important;}
  `;

  document.head.appendChild(style);

  const appendBanner = () => {
    document.body.insertBefore(banner, document.body.firstChild);
    document.getElementById("pg-leave").onclick  = () => {
      window.history.length > 1 ? window.history.back() : (window.location.href = "https://www.google.com");
    };
    document.getElementById("pg-ignore").onclick = () => banner.remove();
    document.getElementById("pg-close").onclick  = () => banner.remove();
  };

  if (document.body) {
    appendBanner();
  } else {
    document.addEventListener("DOMContentLoaded", appendBanner);
  }
}