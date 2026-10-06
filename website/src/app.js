/* open every page at its top unless a #section was asked for; the viewer may keep the previous page's scroll */
try { history.scrollRestoration = "manual"; } catch (e) {}
addEventListener("load", () => { if (location.hash) return;
  scrollTo(0, 0); try { document.body.scrollIntoView({ block: "start" }); } catch (e) {} });
/* Page logic: leaderboard views and every figure. Data: window.__D (see build.py). */
const D = window.__D, F = D.figures;
const HC = { prime: "--prime", ch: "--ch", rag: "--rag", mem0: "--mem0", skillopt: "--skillopt", blind: "--ref", oracle: "--ink2" };
const HN = { prime: "Prime", ch: "CH", rag: "RAG", mem0: "Mem0", skillopt: "SkillOpt", blind: "Blind", oracle: "Oracle" };
const BYNAME = { Prime: "--prime", CH: "--ch", RAG: "--rag", Mem0: "--mem0", SkillOpt: "--skillopt" };
const HEX2VAR = { "#3F5F95": "--prime", "#5FA37E": "--ch", "#8DB6D6": "--rag", "#D9895B": "--mem0", "#D4B04A": "--skillopt" };
const MODEL_SHAPE = { "glm-5p3-flash": "o", "deepseek-v4p1-flash": "d", "glm-5p3": "s", "kimi-k3": "^", "claude-opus-5": "h", "gpt-5.6-terra": "p" };
const MODEL_SHORT = { "glm-5p3-flash": "GLM-5.3 Flash", "deepseek-v4p1-flash": "DS-V4.1 Flash", "glm-5p3": "GLM-5.3", "kimi-k3": "Kimi K3", "claude-opus-5": "Opus 5", "gpt-5.6-terra": "GPT-5.6 Terra" };
const hv = c => HEX2VAR[(c || "").toUpperCase()] || c;
const money = v => v == null ? "–" : "$" + Math.round(v).toLocaleString("en-US");

/* ------------------------------------------------------------ theme */
(() => {
  const b = document.getElementById("theme");
  try { const s = localStorage.getItem("slb-theme"); if (s) document.documentElement.dataset.theme = s; } catch (e) {}
  b.onclick = () => {
    const r = document.documentElement, cur = r.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    r.dataset.theme = cur === "dark" ? "light" : "dark";
    try { localStorage.setItem("slb-theme", r.dataset.theme); } catch (e) {}
  };
})();
document.querySelectorAll(".copy").forEach(btn => btn.addEventListener("click", () => {
  const pre = btn.parentElement, text = pre.dataset.copy || pre.innerText.replace(/^Copy\n?/, "");
  const done = () => { btn.textContent = "Copied"; setTimeout(() => (btn.textContent = "Copy"), 1400); };
  (navigator.clipboard ? navigator.clipboard.writeText(text) : Promise.reject()).then(done, () => {
    const r = document.createRange(); r.selectNodeContents(pre); const s = getSelection(); s.removeAllRanges(); s.addRange(r);
  });
}));

/* ------------------------------------------------------ leaderboard */
const LB = { models: new Set(D.models.map(m => m.key)), refs: true, sort: "avg_h", asc: false, view: "rank" };
function lbControls() {
  const c = document.getElementById("lb-models"); if (!c) return; c.innerHTML = "";
  D.models.forEach(m => {
    const b = document.createElement("button"); b.className = "chip"; b.type = "button";
    b.textContent = m.name; b.setAttribute("aria-pressed", LB.models.has(m.key));
    b.onclick = () => { LB.models.has(m.key) ? LB.models.delete(m.key) : LB.models.add(m.key); if (!LB.models.size) D.models.forEach(x => LB.models.add(x.key)); lbControls(); lbRender(); if (LB.view === "frontier") drawFrontier(); };
    c.appendChild(b);
  });
  const r = document.createElement("button"); r.className = "chip"; r.type = "button"; r.textContent = "Blind & Oracle";
  r.setAttribute("aria-pressed", LB.refs); r.style.marginLeft = "8px";
  r.onclick = () => { LB.refs = !LB.refs; lbControls(); lbRender(); };
  c.appendChild(r);
}
document.querySelectorAll("#lb-view button").forEach(b => b.addEventListener("click", () => {
  LB.view = b.dataset.view;
  document.querySelectorAll("#lb-view button").forEach(x => x.setAttribute("aria-pressed", x === b));
  document.querySelectorAll(".view").forEach(v => (v.hidden = v.dataset.view !== LB.view));
  if (LB.view === "frontier") drawFrontier();
  lbRender();
}));
function lbRender() {
  if (LB.view === "frontier" || !document.getElementById("lb")) return;
  const rows = D.rows.filter(r => LB.models.has(r.model) && (LB.refs || !r.ref));
  const key = LB.sort;
  const get = r => key.startsWith("v") ? r.values[+key.slice(1)] : key === "model" ? r.model_name : key === "harness" ? r.harness_name : r[key];
  rows.sort((a, b) => {
    if (a.ref !== b.ref) return a.ref ? 1 : -1;
    if (a.ref && a.harness !== b.harness) return a.harness === "oracle" ? -1 : 1;
    const x = get(a), y = get(b);
    if (x == null) return 1; if (y == null) return -1;
    return typeof x === "string" ? (LB.asc ? x.localeCompare(y) : y.localeCompare(x)) : (LB.asc ? x - y : y - x);
  });
  const best = {}; D.rows.filter(r => !r.ref).forEach(r => { if (best[r.model] == null || r.avg_h > best[r.model]) best[r.model] = r.avg_h; });
  const t = document.getElementById(LB.view === "rank" ? "lb" : "lb2"); t.innerHTML = "";
  const cols = LB.view === "rank"
    ? [["#", "rank", "l"], ["Model", "model", "l"], ["Method", "harness", "l"], ["Avg. Hidden", "avg_h"], ["Avg. Fully Spec.", "avg_f"], ["API cost", "cost"]]
    : [["Model", "model", "l"], ["Method", "harness", "l"]].concat(D.columns.map((c, i) => [c.replace(/ (H|F) /, " $1·"), "v" + i])).concat([["Avg. H", "avg_h"], ["Avg. F", "avg_f"]]);
  const thead = document.createElement("thead"), tr = document.createElement("tr");
  cols.forEach(([lab, k, cls]) => {
    const th = document.createElement("th"); th.textContent = lab; if (cls) th.className = cls;
    if (LB.sort === k) th.setAttribute("aria-sort", LB.asc ? "ascending" : "descending");
    if (k !== "rank") th.onclick = () => { if (LB.sort === k) LB.asc = !LB.asc; else { LB.sort = k; LB.asc = ["model", "harness", "cost"].includes(k); } lbRender(); };
    tr.appendChild(th);
  });
  thead.appendChild(tr); t.appendChild(thead);
  const tb = document.createElement("tbody"); let rank = 0, sep = false;
  rows.forEach(r => {
    if (r.ref && !sep) { sep = true; const g = document.createElement("tr"); g.className = "group"; g.innerHTML = `<td class="l" colspan="${cols.length}">Reference settings</td>`; tb.appendChild(g); }
    const row = document.createElement("tr"); if (r.ref) row.className = "ref";
    const isBest = !r.ref && best[r.model] === r.avg_h;
    let h = "";
    if (LB.view === "rank") h += `<td class="rank l">${r.ref ? "" : ++rank}</td>`;
    h += `<td class="l"><span class="mname">${r.model_name}</span>${r.size ? `<span class="msize">${r.size}</span>` : ""}</td>`;
    h += `<td class="l"><span class="hdot" style="background:var(${HC[r.harness]})"></span>${r.harness_name}${isBest ? '<span class="tag">best</span>' : ""}</td>`;
    if (LB.view === "rank") {
      h += `<td><div class="scorecell"><span class="num${isBest ? " best" : ""}">${fmt(r.avg_h)}</span><span class="scorebar"><i style="width:${r.avg_h}%"></i></span></div></td>`;
      h += `<td class="num">${fmt(r.avg_f)}</td><td class="num">${money(r.cost)}</td>`;
    } else {
      r.values.forEach((v, i) => { h += `<td class="num${!r.ref && D.columns[i].includes(" H ") && v != null && v < 20 ? " cell-lo" : ""}">${fmt(v)}</td>`; });
      h += `<td class="num best">${fmt(r.avg_h)}</td><td class="num">${fmt(r.avg_f)}</td>`;
    }
    row.innerHTML = h; tb.appendChild(row);
  });
  t.appendChild(tb);
}
let FRONTIER_X = "tokens";
document.querySelectorAll("#fx button").forEach(b => b.addEventListener("click", () => {
  FRONTIER_X = b.dataset.x; document.querySelectorAll("#fx button").forEach(x => x.setAttribute("aria-pressed", x === b)); drawFrontier();
}));
function pareto(pts, key) {
  const s = pts.slice().sort((a, b) => a[key] - b[key] || b.reward - a.reward); let best = -1; const out = [];
  s.forEach(p => { if (p.reward > best) { best = p.reward; out.push(p); } }); return out;
}
let _frontierDraw = null;
function frontierChart(host, key, models) {
  const pts = F.frontier.filter(p => models.has(p.model));
  const front = new Set(pareto(pts, key));
  const all = F.frontier.map(p => p[key]);
  scatterChart(host, {
    height: Math.min(420, Math.max(280, host.clientWidth * 0.7)), xLog: true,
    xDomain: [Math.min(...all) / 1.4, Math.max(...all) * 1.4], yDomain: [15, 80], yTicks: [20, 40, 60, 80],
    xTicks: key === "tokens" ? [1e4, 3e4, 1e5, 3e5, 1e6] : [0.003, 0.01, 0.03, 0.1, 0.3],
    xFmt: v => key === "tokens" ? (v >= 1e6 ? v / 1e6 + "M" : v / 1e3 + "k") : v,
    xLabel: key === "tokens" ? "Token cost per task (log)" : "API cost per task, USD (log)",
    yLabel: "Reward on Hidden (%)", labelBoxes: true,
    stairs: [...front].map(p => ({ x: p[key], y: p.reward })),
    points: pts.map(p => {
      const on = front.has(p);
      return { x: p[key], y: p.reward, color: HC[p.harness], shape: MODEL_SHAPE[p.model], emph: on, dim: !on,
        label: on ? `${MODEL_SHORT[p.model]} ${HN[p.harness]}` : null,
        tip: `<b>${MODEL_SHORT[p.model]} · ${HN[p.harness]}</b><br>Avg. Hidden <span class="m">${fmt(p.reward)}</span><br>` +
          `${fmt(p.tokens / 1e3, 0)}k tokens · $${p.usd.toFixed(4)} per task${on ? "<br><i>on the Pareto frontier</i>" : ""}` };
    }) });
}
const ALL_MODELS = new Set(D.models.map(m => m.key));
chart("fig1-tokens", host => frontierChart(host, "tokens", ALL_MODELS));
chart("fig1-usd", host => frontierChart(host, "usd", ALL_MODELS));
function drawFrontier() {
  if (_frontierDraw) return _frontierDraw();
  chart("frontier", host => {
  if (host.closest(".view").hidden) return;
  const pts = F.frontier.filter(p => LB.models.has(p.model)), key = FRONTIER_X;
  const front = new Set(pareto(pts, key));
  const all = F.frontier.map(p => p[key]);
  scatterChart(host, {
    height: Math.min(460, Math.max(300, host.clientWidth * 0.5)), xLog: true,
    xDomain: [Math.min(...all) / 1.4, Math.max(...all) * 1.4], yDomain: [15, 80], yTicks: [20, 40, 60, 80],
    xTicks: key === "tokens" ? [1e4, 3e4, 1e5, 3e5, 1e6] : [0.003, 0.01, 0.03, 0.1, 0.3],
    xFmt: v => key === "tokens" ? (v >= 1e6 ? v / 1e6 + "M" : v / 1e3 + "k") : "$" + v,
    xLabel: key === "tokens" ? "Tokens per test task (prompt, cache hits and output; log scale)" : "API cost per test task (USD, log scale)",
    yLabel: "Avg. Hidden (%)",
    stairs: [...front].map(p => ({ x: p[key], y: p.reward })),
    points: pts.map(p => {
      const on = front.has(p);
      return { x: p[key], y: p.reward, color: HC[p.harness], shape: MODEL_SHAPE[p.model], emph: on, dim: !on,
        label: on ? `${MODEL_SHORT[p.model]} ${HN[p.harness]}` : null, dx: 9, dy: 4,
        tip: `<b>${MODEL_SHORT[p.model]} · ${HN[p.harness]}</b><br>Avg. Hidden <span class="m">${fmt(p.reward)}</span><br>` +
          `${fmt(p.tokens / 1e3, 0)}k tokens · $${p.usd.toFixed(4)} per task${on ? "<br><i>on the Pareto frontier</i>" : ""}` };
    }) });
  });
  _frontierDraw = REGISTRY[REGISTRY.length - 1];
}
lbControls(); lbRender();

/* ---------------------------------------------------------- figures */
function legend(id, items) {
  const L = document.getElementById(id); if (!L) return;
  L.innerHTML = items.map(([name, color, kind]) => `<span><i class="${kind || ""}" style="background:${color.startsWith("--") ? `var(${color})` : color};color:${color.startsWith("--") ? `var(${color})` : color}"></i>${name}</span>`).join("");
}
const HARN_ORDER = ["Prime", "CH", "RAG", "Mem0", "SkillOpt"];
legend("leg-harness-1", HARN_ORDER.map(n => [n, BYNAME[n], "dot"]));
legend("leg-harness-2", HARN_ORDER.map(n => [n, BYNAME[n]]));
legend("leg-harness-3", HARN_ORDER.map(n => [n, BYNAME[n]]));
legend("leg-harness-4", HARN_ORDER.map(n => [n, BYNAME[n]]));

// average performance
[["avg-model", "(a) By Model"], ["avg-harness", "(b) By Harness"]].forEach(([id, key]) => chart(id, host => {
  const p = F.avg[key];
  const isH = key.includes("Harness");
  barChart(host, { cats: p.labels, yMin: 0, yMax: 100, yTicks: [0, 25, 50, 75, 100], yLabel: "Score (%)", unit: "%", height: 250,
    series: [{ name: "Hidden", colors: p.labels.map(l => isH ? (l === "Blind" || l === "Oracle" ? "--ref" : BYNAME[l]) : "--accent"), values: p.hidden },
             { name: "Fully Specified", color: "--line2", values: p.fs }],
    marks: isH ? [] : [{ name: "Oracle (Hidden)", color: "--good", values: p.oracle_hidden }, { name: "Blind (Hidden)", color: "--warn", values: p.blind_hidden, dash: true }] });
}));
legend("leg-avg", [["Hidden", "--accent", "sq"], ["Fully Specified", "--line2", "sq"], ["Oracle (Hidden)", "--good"], ["Blind (Hidden)", "--warn", "dash"]]);

// cost vs reward (serving + test pooled)
const cm = g => F.costmerged.filter(r => r.grouped_by === g);
const niceName = s => s.replace("DeepSeek-v4.1 Flash", "DS-V4.1 Flash").replace("mem0", "Mem0");
[["cm-model", "model"], ["cm-harness", "harness"]].forEach(([id, g]) => chart(id, host => comboChart(host, {
  cats: cm(g).map(r => niceName(r.column)), cost: cm(g).map(r => +r.usd_per_task), costMax: 0.22, rMin: 25, rMax: 80, height: 270,
  costFmt: v => v.toFixed(2),
  lines: [{ name: "Reward on Hidden", color: "--warn", values: cm(g).map(r => +r.hidden_pct) },
          { name: "Reward on all cases", color: "--good", shape: "s", values: cm(g).map(r => +r.all_pct) }] })));
legend("leg-cm", [["Cost per task (left axis)", "--accent", "sq"], ["Reward on Hidden (right)", "--warn"], ["Reward on all cases (right)", "--good"]]);

// learning speed
[["ls-bank", "banking"], ["ls-retail", "retail"]].forEach(([id, d]) => chart(id, host => lineChart(host, {
  x: F.speed.x, series: F.speed[d].slice().reverse().map(s => ({ name: s.name, color: BYNAME[s.name], y: s.y })),
  yMin: 0, yMax: 85, yTicks: [0, 20, 40, 60, 80], xTicks: [0, 2, 4, 6, 8, 10], xLabel: "Earlier tickets of the policy seen",
  yLabel: "Correct (%)", xName: "after", unit: "%", height: 260 })));
[["hm-bank", "heat_banking"], ["hm-retail", "heat_retail"]].forEach(([id, k]) => chart(id, host => heatmap(host, F.speed[k])));

// FS harm
["retail", "banking"].forEach(dom => chart("fs-" + dom, host => {
  const bars = F.fs.filter(b => b.domain === dom), names = bars.map(b => HN[b.method] || b.method);
  barChart(host, { cats: names, yMin: 75, yMax: 100, yTicks: [75, 80, 85, 90, 95, 100], yLabel: "FS accuracy (%)", unit: "%", labels: true, height: 250,
    series: [{ colors: names.map(n => n === "Blind" ? "--ref" : BYNAME[n]), values: bars.map(b => b.value) }] });
}));

// exploration
const curvesFrom = arr => arr.map(s => ({ name: s.name, color: hv(s.color), y: s.y }));
const fitPts = (fp, labels) => fp.points.map(p => ({ x: p.x, y: p.y, color: hv(p.color), label: p.label, emph: true,
  tip: `<b>${p.label}</b><br>exploration AUC <span class="m">${p.x.toFixed(2)}</span><br>Hidden reward <span class="m">${fmt(p.y)}</span>` }));
chart("ex-h-curve", host => lineChart(host, { x: F.explore.x, series: curvesFrom(F.explore.harness_curves), yMin: 1, yMax: 3.2, yTicks: [1, 1.5, 2, 2.5, 3], xLabel: "Ticket k of the policy", yLabel: "Distinct answers", xName: "k =", digits: 2, height: 250 }));
chart("ex-h-fit", host => scatterChart(host, { points: fitPts(F.explore.harness_fit), fit: F.explore.harness_fit.fit, xLabel: "Exploration AUC", yLabel: "Hidden reward (%)", note: "ρ = 1.00", height: 250, margin: { r: 60 } }));
chart("ex-m-curve", host => lineChart(host, { x: F.explore.x, series: curvesFrom(F.explore.model_curves), yMin: 1, yMax: 2.8, yTicks: [1, 1.5, 2, 2.5], xLabel: "Ticket k of the policy", yLabel: "Distinct answers", xName: "k =", digits: 2, height: 250 }));
chart("ex-m-fit", host => scatterChart(host, { points: fitPts(F.explore.model_fit), fit: F.explore.model_fit.fit, xLabel: "Exploration AUC", yLabel: "Hidden reward (%)", note: "ρ = 0.80", height: 250, margin: { r: 86 } }));
legend("leg-models", F.explore.model_curves.map(s => [s.name, hv(s.color)]));

// knowledge acquisition
chart("ka-curve", host => lineChart(host, { x: F.ka.x, series: curvesFrom(F.ka.curves), yMin: 0, yMax: 100, yTicks: [0, 25, 50, 75, 100], vline: 0,
  xLabel: "Ticket relative to first success", yLabel: "Correct (%)", xName: "ticket", unit: "%", height: 260 }));
chart("ka-fit", host => scatterChart(host, { points: fitPts(F.ka.fit).map(p => ({ ...p, tip: p.tip.replace("exploration AUC", "AUC<sub>KA</sub>") })), fit: F.ka.fit.fit,
  xLabel: "AUC_KA: correct over the next five tickets (%)", yLabel: "Hidden reward (%)", note: F.ka.rho, height: 260, margin: { r: 60 } }));

// learn vs drop
F.drop.panels.forEach((p, i) => chart("drop-" + i, host => lineChart(host, {
  x: [0, 1, 2, 3], xTicks: [0, 1, 2, 3], xFmt: v => F.drop.bins[v], yMin: 0, yMax: 100, yTicks: [0, 50, 100], height: 190,
  margin: { l: 36, b: 32 }, xName: "ticket", unit: "%",
  series: [{ name: "Learn the policy", color: hv(p.color), y: p.learn }, { name: "Drop the same policy", color: hv(p.color), y: p.drop, dash: true }] })));

// exploration by domain
F.expdomain.panels.forEach((p, i) => chart("ed-" + i, host => p.kind === "curves"
  ? lineChart(host, { x: F.expdomain.x, series: curvesFrom(p.data), yMin: 1, yMax: 3.4, yTicks: [1, 2, 3], xLabel: "Ticket k", xName: "k =", digits: 2, height: 210, margin: { l: 34 } })
  : scatterChart(host, { points: fitPts(p.data), fit: p.data.fit, xLabel: "Exploration AUC", height: 210, margin: { l: 34, r: 78 } })));

// exploration per cell
const TIER_SHAPE = { o: "o", s: "s", "^": "^" }, TIER_NAME = { o: "L1", s: "L2", "^": "L3" };
[["cells-all", null], ["cells-retail", "Retail"], ["cells-bank", "Banking"]].forEach(([id, dom], i) => chart(id, host => scatterChart(host, {
  points: F.cells.points.filter(p => !dom || p.domain === dom).map(p => ({ x: p.x, y: p.y, color: BYNAME[p.harness], shape: TIER_SHAPE[p.marker],
    tip: `<b>${p.harness}</b> · ${p.domain} ${TIER_NAME[p.marker]}<br>AUC <span class="m">${p.x.toFixed(2)}</span> · reward <span class="m">${fmt(p.y)}</span>` })),
  fit: F.cells.fits[i], note: F.cells.stats[i][0], xLabel: "Exploration AUC", yLabel: i ? null : "Hidden reward (%)", height: 250, margin: { l: i ? 34 : 46 } })));

// cost by phase
[["ph-sm", "serving", "model"], ["ph-sh", "serving", "harness"], ["ph-tm", "test", "model"], ["ph-th", "test", "harness"]].forEach(([id, ph, g]) => chart(id, host => {
  const rows = F.phase.filter(r => r.phase === ph && r.grouped_by === g);
  comboChart(host, { cats: rows.map(r => niceName(r.column)), cost: rows.map(r => +r.usd_per_episode), costMax: 0.25, rMin: 30, rMax: 75, height: 250, costFmt: v => v.toFixed(2),
    lines: [{ name: "Reward on Hidden", color: "--warn", values: rows.map(r => +r.adapt_pct) }, { name: "Reward on all cases", color: "--good", shape: "s", values: rows.map(r => +r.total_pct) }] });
}));

// learning cost breakdown
const LC_MODELS = [["glm-5p3-flash", "GLM-5.3 Flash"], ["glm-5p3", "GLM-5.3"], ["kimi-k3", "Kimi K3"], ["deepseek-v4p1-flash", "DS-V4.1 Flash"]];
LC_MODELS.forEach(([m, name], i) => chart("lc-" + i, host => {
  const rows = ["skillopt", "mem0", "ch", "prime"].map(h => F.learncost.find(r => r.model === m && r.harness === h)).filter(Boolean);
  const per = r => +r.serving_episodes;
  stackChart(host, { cats: rows.map(r => HN[r.harness]), height: 230, yFmt: v => "$" + v.toFixed(2),
    valFmt: v => "$" + v.toFixed(4), pct: rows.map(r => r.learning_share_usd_pct),
    parts: [{ name: "Serving the stream", color: "--test", values: rows.map(r => +r.serving_usd / per(r)) },
            { name: "Updating learned state", color: "--serve", values: rows.map(r => +r.learning_usd / per(r)) }] });
}));
legend("leg-lc", [["Serving the stream", "--test", "sq"], ["Updating the learned state", "--serve", "sq"]]);

/* ------------------------------------------------ home: leaderboard preview */
(() => {
  const t = document.getElementById("lb-preview"); if (!t) return;
  const rows = D.rows.filter(r => !r.ref).sort((a, b) => b.avg_h - a.avg_h).slice(0, 10);
  const medal = ["🥇", "🥈", "🥉"];
  t.innerHTML = `<thead><tr><th class="l">Rank</th><th class="l">System</th><th>Avg. Hidden</th><th>Avg. FS</th><th>API cost</th></tr></thead><tbody>` +
    rows.map((r, i) => `<tr><td class="l ${i < 3 ? "medal" : "rank"}">${i < 3 ? medal[i] : i + 1}</td>` +
      `<td class="l"><span class="hdot" style="background:var(${HC[r.harness]})"></span><span class="sys">${r.harness_name}<span class="sep">·</span>${r.model_name}</span></td>` +
      `<td><div class="scorecell"><span class="num best">${fmt(r.avg_h)}</span><span class="scorebar"><i style="width:${r.avg_h}%"></i></span></div></td>` +
      `<td class="num">${fmt(r.avg_f)}</td><td class="num">${money(r.cost)}</td></tr>`).join("") + "</tbody>";
})();

/* ---------------------------------------- leaderboard page: setting heatmap */
chart("lb-heat", host => {
  const hid = D.columns.map((c, i) => [c, i]).filter(([c]) => c.includes(" H "));
  const rows = D.rows.filter(r => !r.ref).sort((a, b) => b.avg_h - a.avg_h);
  heatmap(host, { rows: rows.map(r => `${r.harness_name} · ${r.model_name}`), cols: hid.map(([c]) => c.replace(" H ", " ")),
    values: rows.map(r => hid.map(([, i]) => r.values[i])), max: 100, labelWidth: 170, metric: "Hidden score", boldLast: false });
});
