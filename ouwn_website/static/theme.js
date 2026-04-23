// Theme toggle (light/dark) with localStorage persistence
(function () {
  const STORAGE_KEY = "ouwn_theme"; // 'light' | 'dark'

  function safeGet(key) {
    try { return localStorage.getItem(key); }
    catch (_) { return null; }
  }

  function safeSet(key, val) {
    try { localStorage.setItem(key, val); }
    catch (_) {}
  }

  function currentTheme() {
    return document.body.classList.contains("dark-mode") ? "dark" : "light";
  }

  function applyTheme(theme) {
    if (!document.body) return;
    document.body.classList.toggle("dark-mode", theme === "dark");
    updateButton();
  }

  function toggleTheme() {
    const next = currentTheme() === "dark" ? "light" : "dark";
    safeSet(STORAGE_KEY, next);
    applyTheme(next);
  }

  let btn = null;

  function ensureButton() {
    if (btn) return btn;

    btn = document.querySelector(".theme-toggle");
    if (btn) return btn;

    btn = document.createElement("button");
    btn.type = "button";
    btn.className = "theme-toggle";
    btn.setAttribute("aria-label", "Toggle dark mode");
    btn.addEventListener("click", toggleTheme);

    const span = document.createElement("span");
    span.className = "theme-toggle__icon";
    span.setAttribute("aria-hidden", "true");
    btn.appendChild(span);

    document.body.appendChild(btn);
    return btn;
  }

  function updateButton() {
    if (!btn) return;
    const theme = currentTheme();
    const icon = btn.querySelector(".theme-toggle__icon");
    if (icon) icon.textContent = theme === "dark" ? "☀️" : "🌙";
    btn.title = theme === "dark" ? "Switch to light mode" : "Switch to dark mode";
  }

  document.addEventListener("DOMContentLoaded", function () {
    const saved = safeGet(STORAGE_KEY);
    applyTheme(saved === "dark" ? "dark" : "light");
    ensureButton();
    updateButton();
  });
})();
