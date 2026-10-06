/* A small SVG chart kit for the ServeLearnBench page.
   Every chart draws at its container's width, redraws on resize, takes all
   colors from CSS variables (so it follows the theme), and shows a tooltip on
   hover. No dependencies. */
const NS = "http://www.w3.org/2000/svg";
const TIP = (() => { const d = document.createElement("div"); d.className = "tip"; document.body.appendChild(d); return d; })();
function tipShow(ev, html) { TIP.innerHTML = html; TIP.classList.add("on"); tipMove(ev); }
function tipMove(ev) {
  const x = ev.clientX, y = ev.clientY, w = TIP.offsetWidth, h = TIP.offsetHeight;
  TIP.style.left = Math.min(window.innerWidth - w - 8, x + 14) + "px";
  TIP.style.top = (y - h - 12 < 4 ? y + 16 : y - h - 12) + "px";
}
function tipHide() { TIP.classList.remove("on"); }

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) if (attrs[k] !== undefined && attrs[k] !== null) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function txt(parent, x, y, s, attrs = {}) { const t = el("text", { x, y, ...attrs }, parent); t.textContent = s; return t; }
function hover(node, html) {
  node.style.cursor = "default";
  node.addEventListener("mouseenter", e => tipShow(e, typeof html === "function" ? html() : html));
  node.addEventListener("mousemove", tipMove);
  node.addEventListener("mouseleave", tipHide);
}
const fmt = (v, d = 1) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d));

function linear(d0, d1, r0, r1) { const f = v => r0 + (v - d0) / (d1 - d0 || 1) * (r1 - r0); f.inv = p => d0 + (p - r0) / (r1 - r0) * (d1 - d0); return f; }
function logscale(d0, d1, r0, r1) { const a = Math.log10(d0), b = Math.log10(d1); return v => r0 + (Math.log10(v) - a) / (b - a) * (r1 - r0); }
function niceTicks(lo, hi, n = 5) {
  const span = hi - lo, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) || mag * 10;
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

/* Frame: an <svg> sized to the container, with margins, axes and grid. */
function frame(host, opt) {
  host.innerHTML = "";
  const W = Math.max(260, host.clientWidth), H = opt.height || Math.round(Math.min(340, Math.max(220, W * 0.62)));
  const m = { t: 12, r: 14, b: 40, l: 46, ...(opt.margin || {}) };
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opt.aria || "" }, host);
  return { svg, W, H, m, x0: m.l, x1: W - m.r, y0: H - m.b, y1: m.t };
}
function axes(f, sx, sy, opt) {
  const g = el("g", { class: "grid" }, f.svg), a = el("g", { class: "axis" }, f.svg);
  (opt.yTicks || []).forEach(v => {
    const y = sy(v);
    el("line", { x1: f.x0, x2: f.x1, y1: y, y2: y }, g);
    txt(a, f.x0 - 7, y + 4, opt.yFmt ? opt.yFmt(v) : v, { "text-anchor": "end" });
  });
  (opt.xTicks || []).forEach(v => {
    const x = sx(v);
    if (opt.xGrid) el("line", { x1: x, x2: x, y1: f.y1, y2: f.y0 }, g);
    txt(a, x, f.y0 + 16, opt.xFmt ? opt.xFmt(v) : v, { "text-anchor": "middle" });
  });
  el("line", { x1: f.x0, x2: f.x1, y1: f.y0, y2: f.y0 }, a);
  if (opt.xLabel) txt(a, (f.x0 + f.x1) / 2, f.H - 6, opt.xLabel, { "text-anchor": "middle", class: "lbl" });
  if (opt.yLabel) txt(a, 0, 0, opt.yLabel, { class: "lbl", transform: `translate(12 ${(f.y0 + f.y1) / 2}) rotate(-90)`, "text-anchor": "middle" });
}
function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || name; }
const col = c => (c && c.startsWith("--") ? cssVar(c) : c);

/* ---------------------------------------------------------------- line */
function lineChart(host, o) {
  const f = frame(host, o);
  const xs = o.x, ys = o.series.flatMap(s => s.y.filter(v => v != null));
  const lo = o.yMin ?? Math.min(...ys), hi = o.yMax ?? Math.max(...ys);
  const sx = linear(o.xMin ?? xs[0], o.xMax ?? xs[xs.length - 1], f.x0 + 8, f.x1 - 8), sy = linear(lo, hi, f.y0, f.y1);
  axes(f, sx, sy, { yTicks: o.yTicks || niceTicks(lo, hi, 4), xTicks: o.xTicks || xs, xLabel: o.xLabel, yLabel: o.yLabel,
    xFmt: o.xFmt, yFmt: o.yFmt });
  if (o.vline != null) el("line", { x1: sx(o.vline), x2: sx(o.vline), y1: f.y1, y2: f.y0, stroke: col("--line2"), "stroke-dasharray": "3 3" }, f.svg);
  o.series.forEach(s => {
    const pts = xs.map((x, i) => [x, s.y[i]]).filter(p => p[1] != null);
    if (s.band) {
      const up = s.band[1], dn = s.band[0];
      const d = xs.map((x, i) => `${i ? "L" : "M"}${sx(x)},${sy(up[i])}`).join("") + xs.slice().reverse().map((x, j) => `L${sx(x)},${sy(dn[xs.length - 1 - j])}`).join("") + "Z";
      el("path", { d, fill: col(s.color), opacity: 0.12 }, f.svg);
    }
    el("path", { d: pts.map((p, i) => `${i ? "L" : "M"}${sx(p[0])},${sy(p[1])}`).join(""), fill: "none", stroke: col(s.color),
      "stroke-width": s.width || 2, "stroke-dasharray": s.dash ? "5 4" : null, "stroke-linejoin": "round", "stroke-linecap": "round" }, f.svg);
    pts.forEach(p => {
      const c = el("circle", { cx: sx(p[0]), cy: sy(p[1]), r: o.dots === false ? 0 : 3, fill: col(s.color), stroke: col("--surface"), "stroke-width": 1.2 }, f.svg);
      const hit = el("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 9, fill: "transparent" }, f.svg);
      hover(hit, `<b>${s.name}</b><br>${o.xName || "x"} ${o.xFmt ? o.xFmt(p[0]) : p[0]} · <span class="m">${fmt(p[1], o.digits ?? 1)}${o.unit || ""}</span>`);
    });
  });
}

/* ------------------------------------------------------------- scatter */
function scatterChart(host, o) {
  const f = frame(host, { margin: { r: 26, ...(o.margin || {}) }, ...o });
  const xs = o.points.map(p => p.x), ys = o.points.map(p => p.y);
  const pad = (a, b, k) => [a - (b - a) * k, b + (b - a) * k];
  let [xlo, xhi] = o.xDomain || pad(Math.min(...xs), Math.max(...xs), 0.12);
  let [ylo, yhi] = o.yDomain || pad(Math.min(...ys), Math.max(...ys), 0.14);
  const sx = o.xLog ? logscale(xlo, xhi, f.x0 + 6, f.x1 - 6) : linear(xlo, xhi, f.x0 + 6, f.x1 - 6), sy = linear(ylo, yhi, f.y0, f.y1);
  axes(f, sx, sy, { yTicks: o.yTicks || niceTicks(ylo, yhi, 4), xTicks: o.xTicks || niceTicks(xlo, xhi, 4), xLabel: o.xLabel,
    yLabel: o.yLabel, xFmt: o.xFmt, yFmt: o.yFmt, xGrid: o.xLog });
  if (o.stairs) {
    const s = o.stairs.slice().sort((a, b) => a.x - b.x);
    let d = `M${sx(s[0].x)},${sy(s[0].y)}`;
    for (let i = 1; i < s.length; i++) d += `H${sx(s[i].x)}V${sy(s[i].y)}`;
    d += `H${f.x1}`;
    el("path", { d: d + `V${f.y0}H${sx(s[0].x)}Z`, fill: col("--accent"), opacity: 0.06 }, f.svg);
    el("path", { d, fill: "none", stroke: col("--accent"), "stroke-width": 1.6 }, f.svg);
  }
  if (o.fit) el("path", { d: o.fit[0].map((x, i) => `${i ? "L" : "M"}${sx(x)},${sy(o.fit[1][i])}`).join(""), stroke: col("--muted"),
    "stroke-dasharray": "5 4", "stroke-width": 1.3, fill: "none" }, f.svg);
  if (o.note) txt(f.svg, f.x0 + 8, f.y1 + 12, o.note, { class: "val" });
  const placed = [];
  const place = (x, y, w, anchor) => {
    const box = (yy) => ({ x0: anchor === "end" ? x - w : x, x1: anchor === "end" ? x : x + w, y0: yy - 10, y1: yy + 3 });
    for (const d of [0, 12, -12, 24, -24, 36]) {
      const b = box(y + d);
      if (!placed.some(q => b.x0 < q.x1 && b.x1 > q.x0 && b.y0 < q.y1 && b.y1 > q.y0)) { placed.push(b); return y + d; }
    }
    placed.push(box(y)); return y;
  };
  const order = o.points.map((p, i) => i).sort((a, b) => (o.points[a].emph ? 1 : 0) - (o.points[b].emph ? 1 : 0));
  order.forEach(i => {
    const p = o.points[i], cx = sx(p.x), cy = sy(p.y), r = p.r || (p.emph ? 6.5 : 5);
    let node;
    if (p.shape === "s") node = el("rect", { x: cx - r * .85, y: cy - r * .85, width: r * 1.7, height: r * 1.7, rx: 1.5 }, f.svg);
    else if (p.shape === "^") node = el("path", { d: `M${cx},${cy - r * 1.1}L${cx + r},${cy + r * .75}L${cx - r},${cy + r * .75}Z` }, f.svg);
    else if (p.shape === "d") node = el("path", { d: `M${cx},${cy - r * 1.1}L${cx + r},${cy}L${cx},${cy + r * 1.1}L${cx - r},${cy}Z` }, f.svg);
    else if (p.shape === "h") node = el("path", { d: [0, 1, 2, 3, 4, 5].map(k => { const a = Math.PI / 3 * k + Math.PI / 6; return `${k ? "L" : "M"}${cx + r * 1.08 * Math.cos(a)},${cy + r * 1.08 * Math.sin(a)}`; }).join("") + "Z" }, f.svg);
    else if (p.shape === "p") { const w = r * 0.42, R = r * 1.1; node = el("path", { d: `M${cx - w},${cy - R}H${cx + w}V${cy - w}H${cx + R}V${cy + w}H${cx + w}V${cy + R}H${cx - w}V${cy + w}H${cx - R}V${cy - w}H${cx - w}Z` }, f.svg); }
    else node = el("circle", { cx, cy, r }, f.svg);
    node.setAttribute("fill", col(p.color));
    node.setAttribute("stroke", p.emph ? col("--ink") : col("--surface"));
    node.setAttribute("stroke-width", p.emph ? 1.4 : 1);
    node.setAttribute("opacity", p.dim ? 0.55 : 1);
    if (p.tip) hover(node, p.tip);
    if (p.label) {
      let anchor = p.anchor || "start", lx = cx + (p.dx ?? 9);
      const w = p.label.length * 7;
      if (anchor === "start" && lx + w > f.W - 2) { anchor = "end"; lx = cx - 9; }
      const ly = place(lx, cy + (p.dy ?? 4), w, anchor);
      const t = txt(f.svg, lx, ly, p.label, { class: "lbl", "text-anchor": anchor });
      if (o.labelBoxes) { try { const bb = t.getBBox(); const r = el("rect", { x: bb.x - 2, y: bb.y - 1, width: bb.width + 4, height: bb.height + 2, rx: 2, class: "lblbox", opacity: 0.9 }); f.svg.insertBefore(r, t); } catch (e) {} }
    }
  });
}

/* ------------------------------------------------------ grouped bars */
function barChart(host, o) {
  const f = frame(host, { margin: { b: o.rotate ? 58 : 40 }, ...o });
  const n = o.cats.length, k = o.series.length, band = (f.x1 - f.x0) / n, bw = Math.min(26, band * 0.78 / k);
  const lo = o.yMin ?? 0, hi = o.yMax ?? Math.max(...o.series.flatMap(s => s.values));
  const sy = linear(lo, hi, f.y0, f.y1);
  axes(f, v => v, sy, { yTicks: o.yTicks || niceTicks(lo, hi, 4), yLabel: o.yLabel, yFmt: o.yFmt });
  o.cats.forEach((c, i) => {
    const cx = f.x0 + band * (i + .5);
    const t = txt(f.svg, cx, f.y0 + 16, c, { "text-anchor": o.rotate ? "end" : "middle" });
    if (o.rotate) t.setAttribute("transform", `rotate(-30 ${cx} ${f.y0 + 16})`);
    o.series.forEach((s, j) => {
      const v = s.values[i]; if (v == null) return;
      const x = cx - bw * k / 2 + j * bw, y = sy(v);
      const r = el("rect", { x: x + 1, y, width: bw - 2, height: Math.max(0, f.y0 - y), rx: 2, fill: col(s.colors ? s.colors[i] : s.color) }, f.svg);
      hover(r, `<b>${c}</b>${s.name ? " · " + s.name : ""}<br><span class="m">${fmt(v, o.digits ?? 1)}${o.unit || ""}</span>`);
      if (o.labels) txt(f.svg, x + bw / 2, y - 4, fmt(v, 0), { class: "val", "text-anchor": "middle" });
    });
    (o.marks || []).forEach(mk => {
      const v = mk.values[i]; if (v == null) return;
      const y = sy(v);
      const ln = el("line", { x1: cx - band * .38, x2: cx + band * .38, y1: y, y2: y, stroke: col(mk.color), "stroke-width": 2.2, "stroke-dasharray": mk.dash ? "4 3" : null }, f.svg);
      hover(ln, `<b>${c}</b> · ${mk.name}<br><span class="m">${fmt(v)}${o.unit || ""}</span>`);
    });
  });
}

/* ------------------------------------------------------------ heatmap */
function heatmap(host, o) {
  const W = Math.max(260, host.clientWidth); host.innerHTML = "";
  const lw = o.labelWidth || 64, th = 12, cw = (W - lw - 4) / o.cols.length, ch = 28, H = th + ch * o.rows.length + 58;
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}` }, host);
  const max = o.max || 60, acc = cssVar("--accent");
  o.cols.forEach((c, j) => { const x = lw + cw * (j + .5) + (j === o.cols.length - 1 ? 3 : 0), y = th + ch * o.rows.length + 18;
    txt(svg, x, y, c, { "text-anchor": "end", transform: `rotate(-35 ${x} ${y})`, class: j === o.cols.length - 1 ? "lbl" : "" }); });
  o.rows.forEach((r, i) => {
    txt(svg, lw - 8, th + ch * (i + .5) + 4, r, { "text-anchor": "end", class: "lbl" });
    o.cols.forEach((c, j) => {
      const v = o.values[i][j], x = lw + cw * j, y = th + ch * i;
      const gap = (o.boldLast !== false && i === o.rows.length - 1 ? 3 : 0), gx = (o.boldLast !== false && j === o.cols.length - 1 ? 3 : 0);
      const cell = el("rect", { x: x + 1 + gx, y: y + 1 + gap, width: cw - 2, height: ch - 2, rx: 3,
        fill: v == null ? "transparent" : acc, "fill-opacity": v == null ? 0 : 0.08 + 0.85 * Math.min(1, v / max) }, svg);
      if (v == null) return;
      const t = txt(svg, x + cw / 2 + gx, y + ch / 2 + 4 + gap, fmt(v), { "text-anchor": "middle", class: "val" });
      if (v / max > 0.55) t.setAttribute("style", "fill:var(--accent-ink)");
      if (o.boldLast !== false && (i === o.rows.length - 1 || j === o.cols.length - 1)) t.setAttribute("font-weight", "700");
      hover(cell, `<b>${r}</b> · ${c}<br>${o.metric || "AULC<sub>10</sub>"} <span class="m">${fmt(v)}</span>`);
    });
  });
}

/* -------------------------------------------------- bars + lines (dual) */
function comboChart(host, o) {
  const f = frame(host, { margin: { r: 44, b: 58 }, ...o });
  const n = o.cats.length, band = (f.x1 - f.x0) / n, bw = Math.min(34, band * .55);
  const cmax = o.costMax || Math.max(...o.cost) * 1.1;
  const sc = linear(0, cmax, f.y0, f.y1), sr = linear(o.rMin ?? 20, o.rMax ?? 80, f.y0, f.y1);
  const g = el("g", { class: "grid" }, f.svg), a = el("g", { class: "axis" }, f.svg);
  niceTicks(0, cmax, 4).forEach(v => { el("line", { x1: f.x0, x2: f.x1, y1: sc(v), y2: sc(v) }, g); txt(a, f.x0 - 7, sc(v) + 4, o.costFmt ? o.costFmt(v) : v, { "text-anchor": "end" }); });
  niceTicks(o.rMin ?? 20, o.rMax ?? 80, 3).forEach(v => txt(a, f.x1 + 7, sr(v) + 4, v, { "text-anchor": "start" }));
  el("line", { x1: f.x0, x2: f.x1, y1: f.y0, y2: f.y0 }, a);
  txt(a, 0, 0, o.leftLabel || "Cost / task (USD)", { class: "lbl", transform: `translate(11 ${(f.y0 + f.y1) / 2}) rotate(-90)`, "text-anchor": "middle" });
  txt(a, 0, 0, "Reward (%)", { class: "lbl", transform: `translate(${f.W - 6} ${(f.y0 + f.y1) / 2}) rotate(90)`, "text-anchor": "middle" });
  o.cats.forEach((c, i) => {
    const cx = f.x0 + band * (i + .5), y = sc(o.cost[i]);
    const r = el("rect", { x: cx - bw / 2, y, width: bw, height: f.y0 - y, rx: 3, fill: col("--accent"), opacity: 0.28 }, f.svg);
    hover(r, `<b>${c}</b><br>cost <span class="m">$${o.cost[i].toFixed(4)}</span> / task`);
    const t = txt(f.svg, cx, f.y0 + 16, c, { "text-anchor": "end" }); t.setAttribute("transform", `rotate(-28 ${cx} ${f.y0 + 16})`);
  });
  o.lines.forEach(L => {
    el("path", { d: L.values.map((v, i) => `${i ? "L" : "M"}${f.x0 + band * (i + .5)},${sr(v)}`).join(""), fill: "none", stroke: col(L.color), "stroke-width": 2 }, f.svg);
    L.values.forEach((v, i) => {
      const c = el(L.shape === "s" ? "rect" : "circle", L.shape === "s"
        ? { x: f.x0 + band * (i + .5) - 4, y: sr(v) - 4, width: 8, height: 8 } : { cx: f.x0 + band * (i + .5), cy: sr(v), r: 4.2 }, f.svg);
      c.setAttribute("fill", col(L.color)); c.setAttribute("stroke", col("--surface")); c.setAttribute("stroke-width", 1.2);
      hover(c, `<b>${o.cats[i]}</b> · ${L.name}<br><span class="m">${fmt(v)}%</span>`);
    });
  });
}

/* ------------------------------------------------------ stacked bars */
function stackChart(host, o) {
  const f = frame(host, { margin: { b: 58, t: 22 }, ...o });
  const n = o.cats.length, band = (f.x1 - f.x0) / n, bw = Math.min(30, band * .6);
  const tot = o.cats.map((_, i) => o.parts.reduce((s, p) => s + p.values[i], 0));
  const hi = o.max || Math.max(...tot) * 1.12, sy = linear(0, hi, f.y0, f.y1);
  axes(f, v => v, sy, { yTicks: niceTicks(0, hi, 4), yLabel: o.yLabel, yFmt: o.yFmt });
  o.cats.forEach((c, i) => {
    const cx = f.x0 + band * (i + .5); let base = 0;
    o.parts.forEach(p => {
      const v = p.values[i], y0 = sy(base), y1 = sy(base + v);
      const r = el("rect", { x: cx - bw / 2, y: y1, width: bw, height: Math.max(0, y0 - y1), fill: col(p.color) }, f.svg);
      hover(r, `<b>${c}</b> · ${p.name}<br><span class="m">${o.valFmt ? o.valFmt(v) : fmt(v)}</span> (${fmt(100 * v / tot[i], 0)}%)`);
      base += v;
    });
    if (o.pct) txt(f.svg, cx, sy(tot[i]) - 5, o.pct[i] + "%", { class: "val", "text-anchor": "middle" });
    const t = txt(f.svg, cx, f.y0 + 16, c, { "text-anchor": "end" }); t.setAttribute("transform", `rotate(-28 ${cx} ${f.y0 + 16})`);
  });
}

/* Render on load, on resize and on theme change. */
const REGISTRY = [];
function chart(id, fn) {
  const host = document.getElementById(id);
  if (!host) return;
  const draw = () => { try { fn(host); } catch (e) { host.textContent = "Chart failed: " + e.message; } };
  REGISTRY.push(draw); draw();
}
let _rt;
window.addEventListener("resize", () => { clearTimeout(_rt); _rt = setTimeout(() => REGISTRY.forEach(d => d()), 120); });
new MutationObserver(() => REGISTRY.forEach(d => d())).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => REGISTRY.forEach(d => d()));
