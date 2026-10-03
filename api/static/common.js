/**
 * Shared helpers for the Session Studio and the Progress report pages.
 * Exposed as window.EY. Everything that touches user/LLM text builds DOM
 * nodes with textContent (never innerHTML) so content can't inject markup.
 */
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);

  function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class') node.className = v;
      else if (k === 'text') node.textContent = v;
      else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? '' : v);
    }
    for (const c of children.flat()) {
      if (c == null || c === false) continue;
      node.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return node;
  }

  function icon(name, cls = '') {
    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('class', `icon ${cls}`.trim());
    svg.setAttribute('aria-hidden', 'true');
    const use = document.createElementNS(ns, 'use');
    use.setAttribute('href', `icons.svg#i-${name}`);
    svg.append(use);
    return svg;
  }

  async function api(path, options = {}) {
    const res = await fetch(path, options);
    let body = null;
    try { body = await res.json(); } catch (_) { /* non-JSON body */ }
    if (!res.ok) {
      const detail = body && body.detail;
      const msg = Array.isArray(detail)
        ? detail.map((d) => d.msg).join('; ')
        : (detail || `Request failed (${res.status})`);
      const err = new Error(msg);
      err.status = res.status;
      throw err;
    }
    return body;
  }

  // Browser storage can be unavailable (private mode, blocked site data).
  const storage = {
    get(key, fallback) {
      try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; } catch (_) { return fallback; }
    },
    set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch (_) { /* ignore */ } },
    remove(key) { try { localStorage.removeItem(key); } catch (_) { /* ignore */ } },
  };
  const sessionStore = {
    get(key) { try { return sessionStorage.getItem(key); } catch (_) { return null; } },
    set(key, v) { try { sessionStorage.setItem(key, v); } catch (_) { /* ignore */ } },
    remove(key) { try { sessionStorage.removeItem(key); } catch (_) { /* ignore */ } },
  };

  function toast(msg) {
    document.querySelectorAll('.toast').forEach((t) => t.remove());
    const t = el('div', { class: 'toast', role: 'status' }, icon('check-circle'), msg);
    document.body.append(t);
    setTimeout(() => t.remove(), 3000);
  }

  function copy(text, msg = 'Copied') {
    navigator.clipboard.writeText(text).then(() => toast(msg), () => toast('Copy failed'));
  }

  const fmt = {
    time(s) {
      if (!Number.isFinite(s)) return '00:00';
      return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
    },
    srt(s) {
      const ms = Math.round(s * 1000);
      const h = Math.floor(ms / 3600000);
      const m = Math.floor((ms % 3600000) / 60000);
      const sec = Math.floor((ms % 60000) / 1000);
      return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')},${String(ms % 1000).padStart(3, '0')}`;
    },
    duration(s) { return `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`; },
    bytes(b) {
      if (!b) return '0 B';
      const i = Math.floor(Math.log(b) / Math.log(1024));
      return `${(b / 1024 ** i).toFixed(1)} ${['B', 'KB', 'MB', 'GB'][i]}`;
    },
    speaker(label) { return label ? label.replace('SPEAKER_', 'Speaker ') : 'Speaker ?'; },
    date(iso, opts = { day: 'numeric', month: 'short', year: 'numeric' }) {
      if (!iso) return '';
      const d = new Date(`${String(iso).slice(0, 10)}T00:00:00`);
      return Number.isNaN(d.getTime()) ? String(iso) : d.toLocaleDateString(undefined, opts);
    },
    plural(n, word) { return `${n} ${word}${n === 1 ? '' : 's'}`; },
  };

  window.EY = { $, el, icon, api, storage, session: sessionStore, toast, copy, fmt };
})();
