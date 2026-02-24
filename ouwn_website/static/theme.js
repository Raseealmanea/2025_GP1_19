// Theme toggle (light/dark) with localStorage persistence
(function () {
  const STORAGE_KEY = "ouwn_theme"; // 'light' | 'dark'
  const root = document.documentElement;

  function safeGet(key) { try { return localStorage.getItem(key); } catch (_) { return null; } }
  function safeSet(key, val) { try { localStorage.setItem(key, val); } catch (_) {} }

  function currentTheme() {
    return root.getAttribute("data-theme") === "dark" ? "dark" : "light";
  }

  function applyTheme(theme) {
    root.setAttribute("data-theme", theme === "dark" ? "dark" : "light");
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

  // Apply saved theme ASAP
  const saved = safeGet(STORAGE_KEY);
  applyTheme(saved === "dark" ? "dark" : "light");

  document.addEventListener("DOMContentLoaded", function () {
    ensureButton();
    updateButton();
  });
})();

