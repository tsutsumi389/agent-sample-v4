/* ============================================================
   汎用AIエージェント 設計書 — 共通スクリプト
   - サイドバーナビ / 前後ページャ / 目次(TOC) を一元生成
   - テーマ切替（ライト/ダーク）
   - Mermaid 図のレンダリング
   全ページが同じナビ構造を共有するため、リンク定義はここ1か所だけ。
   ============================================================ */

// ---- ページ一覧（順序がそのままナビ・ページャの並び） ----
const PAGES = [
  { slug: "01-overview",      ix: "01", icon: "🚀", title: "はじめに",                 desc: "全体像と用語" },
  { slug: "02-architecture",  ix: "02", icon: "🏛️", title: "システムアーキテクチャ",   desc: "構成要素と技術スタック" },
  { slug: "03-multi-agent",   ix: "03", icon: "🧠", title: "マルチエージェント設計",   desc: "4役のグラフと制御" },
  { slug: "04-memory",        ix: "04", icon: "🧩", title: "記憶アーキテクチャ",       desc: "短期/長期メモリ" },
  { slug: "05-streaming",     ix: "05", icon: "📡", title: "ストリーミング設計",       desc: "SSE 配信の仕組み" },
  { slug: "06-extensibility", ix: "06", icon: "🔌", title: "拡張ガイド",               desc: "ツール / MCP 追加" },
  { slug: "07-api-reference", ix: "07", icon: "📖", title: "API リファレンス",         desc: "全エンドポイント" },
  { slug: "08-data-model",    ix: "08", icon: "🗃️", title: "データモデルと永続化",     desc: "DB と namespace" },
  { slug: "09-frontend",      ix: "09", icon: "💻", title: "フロントエンド設計",       desc: "React の構成" },
  { slug: "10-setup",         ix: "10", icon: "⚙️", title: "セットアップと運用",       desc: "起動と運用" },
];

const inGuide = location.pathname.includes("/guide/");
const homeHref  = inGuide ? "../index.html" : "index.html";
const guideHref = (slug) => (inGuide ? `${slug}.html` : `guide/${slug}.html`);
const currentSlug = (() => {
  const m = location.pathname.match(/([^/]+)\.html$/);
  return m ? m[1] : "index";
})();

/* ---------- サイドバーナビ生成 ---------- */
function buildNav() {
  const nav = document.getElementById("site-nav");
  if (!nav) return;
  const home = `<a href="${homeHref}" class="${currentSlug === "index" ? "active" : ""}">
      <span class="ix">🏠</span><span>トップ（目次）</span></a>`;
  const items = PAGES.map((p) => `
    <a href="${guideHref(p.slug)}" class="${p.slug === currentSlug ? "active" : ""}">
      <span class="ix">${p.ix}</span><span>${p.title}</span>
    </a>`).join("");
  nav.innerHTML =
    `<div class="nav-group-title">ナビゲーション</div>${home}` +
    `<div class="nav-group-title">設計書</div>${items}`;
}

/* ---------- 前後ページャ生成 ---------- */
function buildPager() {
  const pager = document.getElementById("pager");
  if (!pager || currentSlug === "index") return;
  const idx = PAGES.findIndex((p) => p.slug === currentSlug);
  if (idx < 0) return;
  const prev = idx > 0 ? PAGES[idx - 1] : { slug: null, title: "目次", icon: "🏠" };
  const next = idx < PAGES.length - 1 ? PAGES[idx + 1] : null;
  const prevHref = idx > 0 ? guideHref(prev.slug) : homeHref;
  let html = `<a class="prev" href="${prevHref}"><span class="dir">← 前へ</span><span class="ttl">${prev.title}</span></a>`;
  html += next
    ? `<a class="next" href="${guideHref(next.slug)}"><span class="dir">次へ →</span><span class="ttl">${next.title}</span></a>`
    : `<span class="ph"></span>`;
  pager.innerHTML = html;
}

/* ---------- 見出しアンカー + 目次(TOC) 生成 ---------- */
function buildToc() {
  const article = document.querySelector(".content article");
  const toc = document.getElementById("toc");
  if (!article) return;
  const heads = article.querySelectorAll("h2, h3");
  const links = [];
  heads.forEach((h) => {
    if (!h.id) {
      h.id = (h.textContent || "")
        .trim().toLowerCase()
        .replace(/[^\w぀-ヿ一-鿿-]+/g, "-")
        .replace(/^-+|-+$/g, "") || `sec-${links.length}`;
    }
    const a = document.createElement("a");
    a.className = "heading-anchor"; a.href = `#${h.id}`; a.textContent = "#";
    a.setAttribute("aria-label", "この見出しへのリンク");
    h.appendChild(a);
    if (toc) {
      const t = document.createElement("a");
      t.href = `#${h.id}`;
      t.className = h.tagName === "H3" ? "lvl-3" : "lvl-2";
      t.textContent = (h.textContent || "").replace(/#$/, "").trim();
      toc.appendChild(t);
    }
    links.push({ id: h.id, el: h });
  });
  if (toc && links.length) {
    const title = document.createElement("div");
    title.className = "toc-title"; title.textContent = "このページの目次";
    toc.prepend(title);
    setupScrollSpy(links, toc);
  }
}

function setupScrollSpy(links, toc) {
  const tocLinks = toc.querySelectorAll("a:not(.toc-title)");
  const byId = {};
  tocLinks.forEach((a) => { byId[a.getAttribute("href").slice(1)] = a; });
  const obs = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        tocLinks.forEach((a) => a.classList.remove("active"));
        const a = byId[e.target.id];
        if (a) a.classList.add("active");
      }
    });
  }, { rootMargin: "-70px 0px -75% 0px", threshold: 0 });
  links.forEach((l) => obs.observe(l.el));
}

/* ---------- テーマ切替 ---------- */
function applyTheme(t) {
  document.documentElement.setAttribute("data-theme", t);
  try { localStorage.setItem("docs-theme", t); } catch (_) {}
  const btn = document.getElementById("theme-toggle");
  if (btn) btn.textContent = t === "dark" ? "☀️" : "🌙";
}
function initTheme() {
  let t = "light";
  try { t = localStorage.getItem("docs-theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"); } catch (_) {}
  applyTheme(t);
  const btn = document.getElementById("theme-toggle");
  if (btn) btn.addEventListener("click", () =>
    applyTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark"));
}

/* ---------- モバイルメニュー ---------- */
function initMobileMenu() {
  const btn = document.querySelector(".menu-toggle");
  const sidebar = document.getElementById("site-nav");
  if (!btn || !sidebar) return;
  let scrim = document.querySelector(".scrim");
  if (!scrim) { scrim = document.createElement("div"); scrim.className = "scrim"; document.body.appendChild(scrim); }
  const close = () => { sidebar.classList.remove("open"); scrim.classList.remove("show"); };
  btn.addEventListener("click", () => {
    sidebar.classList.toggle("open"); scrim.classList.toggle("show");
  });
  scrim.addEventListener("click", close);
  sidebar.addEventListener("click", (e) => { if (e.target.closest("a")) close(); });
}

/* ---------- Mermaid ---------- */
async function initMermaid() {
  if (!document.querySelector(".mermaid")) return;
  const { default: mermaid } = await import("https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs");
  const dark = document.documentElement.getAttribute("data-theme") === "dark";
  mermaid.initialize({
    startOnLoad: false,
    theme: dark ? "dark" : "default",
    securityLevel: "loose",
    fontFamily: getComputedStyle(document.body).fontFamily,
    flowchart: { curve: "basis", htmlLabels: true },
    sequence: { useMaxWidth: true, wrap: true },
  });
  await mermaid.run({ querySelector: ".mermaid" });
}

/* ---------- 起動 ---------- */
initTheme();
document.addEventListener("DOMContentLoaded", () => {
  buildNav();
  buildPager();
  buildToc();
  initMobileMenu();
  initMermaid();
});
