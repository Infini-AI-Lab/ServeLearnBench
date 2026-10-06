/* The ServeLearnBench protocol as one animated figure.
   Serve & learn: tasks flow into one agent. A serving task's score flows into the
   agent's memory, from which it infers the hidden policy; a test task goes to the
   evaluator and returns nothing. The timeline below holds the actual hidden policy of
   each window, which the agent never sees, so its inference can be compared with it.
   Scale comes first: the size of the benchmark, then one stream plays.
   Everything is a pure function of time t, so any moment can be drawn. */
(() => {
  const host = document.getElementById("stream-viz"); if (!host) return;
  const NSV = "http://www.w3.org/2000/svg";
  const mk = (tag, a = {}, p) => { const e = document.createElementNS(NSV, tag); for (const k in a) e.setAttribute(k, a[k]); if (p) p.appendChild(e); return e; };
  const tx = (p, x, y, s, a = {}) => { const t = mk("text", { x, y, ...a }, p); t.textContent = s; return t; };
  const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
  const ease = u => 1 - Math.pow(1 - clamp(u), 3);

  /* ------------------------------------------------ one simulated stream */
  const POLICY = [["C1"], ["C1", "C2", "C6"], ["C1", "C3"], ["C1", "C4", "C5", "C6"], ["C1", "C4", "C6"]];
  const NS = 8, NT = 3, SLOW = { C4: 2, C5: 3 }, UNLEARN = 2;
  const tickets = []; let pos = 0;
  POLICY.forEach((act, w) => {
    const withdrawn = w ? POLICY[w - 1].filter(c => !act.includes(c)) : [];
    const rel = [...act, ...withdrawn];
    for (let i = 0; i < NS; i++) tickets.push({ w, kind: "serve", c: rel[(i + w) % rel.length], ts: ++pos });
    for (let i = 0; i < NT; i++) tickets.push({ w, kind: "test", c: rel[(i * 2 + w + 1) % rel.length], ts: pos });
  });
  { let belief = new Set(); const fails = {};
    tickets.forEach(t => {
      const truth = POLICY[t.w].includes(t.c), knows = belief.has(t.c);
      t.ok = truth === knows; t.before = new Set(belief);
      if (t.kind === "serve") {
        if (!t.ok) { fails[t.c] = (fails[t.c] || 0) + 1; if (fails[t.c] >= (truth ? (SLOW[t.c] || 1) : UNLEARN)) { truth ? belief.add(t.c) : belief.delete(t.c); fails[t.c] = 0; } }
        else fails[t.c] = 0;
      }
      t.after = new Set(belief);
    }); }

  /* ------------------------------------------------ timing */
  const DT = 0.55, TRAVEL = 2.4, PROC = 0.3, FLY = 0.6;
  const arrive = k => k * DT + TRAVEL, done = k => arrive(k) + PROC, stored = k => done(k) + FLY;
  const STREAM_END = stored(tickets.length - 1) + 0.6;
  const SCALE_DUR = 3.2, SCALE_SPEED = 1.3, OFF = (SCALE_DUR + 1.2) / SCALE_SPEED, TOTAL = OFF + STREAM_END + 4;
  const PHASES = [0, OFF];

  /* ------------------------------------------------ canvas */
  const W = 1000, H = 600;
  const svg = mk("svg", { viewBox: `0 0 ${W} ${H}`, class: "sv", role: "img",
    "aria-label": "Animated protocol: tasks stream into one agent; serving scores feed its memory, from which it infers the hidden policy; test tasks return nothing; the hidden policy changes between windows. The scale of the benchmark is shown first." }, host);
  const defs = mk("defs", {}, svg);
  const arrow = (id, cls) => { const m = mk("marker", { id, viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs); mk("path", { d: "M0,0 L10,5 L0,10 z", class: cls }, m); };
  arrow("sv-a1", "sv-arrowhead"); arrow("sv-a2", "sv-returnhead"); arrow("sv-a3", "sv-greyhead");
  const chip = (g, x, y, c, cls, w = 40) => { mk("rect", { x, y, width: w, height: 24, rx: 12, class: "sv-bl " + cls }, g);
    tx(g, x + w / 2, y + 16.5, c, { class: "sv-chiptxt" + (cls.includes("sv-bl-stale") ? " sv-strike" : ""), "text-anchor": "middle" }); };
  const partTitle = (g, x, y, n, s) => { const t = tx(g, x, y, "", { class: "sv-head" }); mk("tspan", { class: "sv-num" }, t).textContent = n + "  "; mk("tspan", {}, t).textContent = s; };

  /* ================= serve & learn ================= */
  const g1 = mk("g", { transform: "translate(0,152)" }, svg);
  partTitle(g1, 20, 24, "2", "Serve and learn: the agent infers the hidden policy from its own scores");
  const tsTxt = tx(g1, 976, 24, "", { class: "sv-mono", "text-anchor": "end" });
  // conveyor and the one agent
  const beltY = 118, beltX0 = 24, ax = 420, aw = 96, ah = 80;
  mk("line", { x1: beltX0, x2: ax - aw / 2 - 6, y1: beltY, y2: beltY, class: "sv-belt" }, g1);
  tx(g1, beltX0, beltY - 30, "Incoming tasks", { class: "sv-lab" });
  // agent icon: a robot head; while busy its eyes narrow and the antenna blinks
  const agentG = mk("g", {}, g1);
  mk("line", { x1: ax, x2: ax, y1: beltY - 34, y2: beltY - 48, class: "sv-ant" }, agentG);
  const tip = mk("circle", { cx: ax, cy: beltY - 50, r: 5, class: "sv-tip" }, agentG);
  mk("rect", { x: ax - 46, y: beltY - 12, width: 8, height: 22, rx: 4, class: "sv-bot" }, agentG);
  mk("rect", { x: ax + 38, y: beltY - 12, width: 8, height: 22, rx: 4, class: "sv-bot" }, agentG);
  mk("rect", { x: ax - 40, y: beltY - 34, width: 80, height: 70, rx: 22, class: "sv-bot sv-bot-head" }, agentG);
  mk("rect", { x: ax - 29, y: beltY - 16, width: 58, height: 26, rx: 13, class: "sv-visor" }, agentG);
  const eyes = [-12, 12].map(dx => mk("ellipse", { cx: ax + dx, cy: beltY - 3, rx: 4.5, ry: 4.5, class: "sv-glow" }, agentG));
  const agentBusy = (b, t) => { eyes.forEach(e => e.setAttribute("ry", b ? 1.4 : 4.5)); tip.style.opacity = b ? 0.5 + 0.5 * Math.sin(t * 12) : 1; };
  tx(g1, ax, beltY + ah / 2 + 18, "Agent", { class: "sv-lab", "text-anchor": "middle" });
  [["serving task: returns a score", "sv-s"], ["test task: returns nothing", "sv-x"]].forEach(([l, c], i) => {
    mk("rect", { x: beltX0 + i * 190, y: beltY + 40, width: 11, height: 15, rx: 4, class: "sv-t sv-live " + c }, g1);
    tx(g1, beltX0 + 17 + i * 190, beltY + 52, l, { class: "sv-small" }); });
  // evaluator (tests)
  const evX = 496, evY = 176;
  mk("path", { d: `M${ax + aw / 2},${beltY + 22} C${ax + 80},${beltY + 22} ${evX - 30},${evY + 26} ${evX - 4},${evY + 26}`, class: "sv-belt sv-dash", fill: "none" }, g1);
  mk("rect", { x: evX, y: evY, width: 150, height: 52, rx: 14, class: "sv-eval" }, g1);
  tx(g1, evX + 75, evY + 22, "Evaluator", { class: "sv-lab", "text-anchor": "middle" });
  tx(g1, evX + 75, evY + 39, "tests graded, never shown", { class: "sv-small", "text-anchor": "middle" });
  // memory panel (serving scores flow in) + inferred policy
  const mX = 660, mY = 40, mW = 328, mH = 236, rowH = 22, NROWS = 5;
  mk("rect", { x: mX, y: mY, width: mW, height: mH, rx: 16, class: "sv-panel" }, g1);
  tx(g1, mX + 14, mY + 22, "Agent's memory: serving outcomes", { class: "sv-lab" });
  mk("path", { d: `M${ax + aw / 2},${beltY - 18} C${ax + 120},${beltY - 30} ${mX - 70},${mY + 50} ${mX - 6},${mY + 50}`, class: "sv-loop", "marker-end": "url(#sv-a1)" }, g1);
  tx(g1, (ax + mX) / 2 + 10, beltY - 48, "the score returns", { class: "sv-small", "text-anchor": "middle" });
  const memG = mk("g", {}, g1);
  mk("line", { x1: mX + 14, x2: mX + mW - 14, y1: mY + 152, y2: mY + 152, class: "sv-divider" }, g1);
  tx(g1, mX + 14, mY + 170, "Inferred policy", { class: "sv-lab" });
  const belG = mk("g", {}, g1);
  const legendY = beltY + 104;
  [["learned", "sv-bl-ok"], ["stale (withdrawn)", "sv-bl-stale"], ["not found yet", "sv-bl-miss"]].forEach(([l, c], i) => {
    const x = mX + 14 + [0, 80, 204][i];
    mk("rect", { x, y: mY + mH - 22, width: 12, height: 12, rx: 6, class: "sv-bl " + c }, g1);
    tx(g1, x + 18, mY + mH - 13, l, { class: "sv-small" }); });
  // timeline with the actual hidden policy
  const tlY = 318, tlX0 = 24, tlX1 = 976, gap = 18, segW = (tlX1 - tlX0 - gap * 4) / 5, tw = segW / (NS + NT);
  const segX = w => tlX0 + w * (segW + gap);
  tx(g1, tlX0, tlY - 30, "Environment windows and the actual hidden policy (never shown to the agent)", { class: "sv-lab" });
  const winBox = POLICY.map((_, w) => mk("rect", { x: segX(w) - 5, y: tlY - 20, width: segW + 10, height: 84, rx: 14, class: "sv-winbox" }, g1));
  const ticks = [];
  POLICY.forEach((act, w) => {
    const x = segX(w);
    tx(g1, x + segW / 2, tlY - 6, `W${w}`, { class: "sv-win", "text-anchor": "middle" });
    for (let i = 0; i < NS + NT; i++) ticks.push(mk("rect", { x: x + i * tw + 1, y: tlY, width: tw - 2, height: 28, rx: 4, class: "sv-t " + (i < NS ? "sv-s" : "sv-x") }, g1));
    if (w) mk("line", { x1: x - gap / 2, x2: x - gap / 2, y1: tlY - 20, y2: tlY + 70, class: "sv-bound" }, g1);
    const cw = 34, cg = 6, tot = act.length * cw + (act.length - 1) * cg;
    act.forEach((c, i) => { const cx = x + segW / 2 - tot / 2 + i * (cw + cg);
      mk("rect", { x: cx, y: tlY + 38, width: cw, height: 22, rx: 11, class: "sv-chip" }, g1);
      tx(g1, cx + cw / 2, tlY + 53.5, c, { class: "sv-chiptxt", "text-anchor": "middle" }); });
  });
  { const a = segX(1) + segW / 2, b = segX(3) + segW / 2;
    mk("path", { d: `M${a},${tlY + 64} C${a},${tlY + 98} ${b},${tlY + 98} ${b},${tlY + 66}`, class: "sv-return", "marker-end": "url(#sv-a2)" }, g1);
    tx(g1, (a + b) / 2, tlY + 106, "C6 withdrawn, then returns", { class: "sv-returntxt", "text-anchor": "middle" }); }
  const cursor = mk("rect", { x: 0, y: tlY - 4, width: 3, height: 36, rx: 1.5, class: "sv-cursor" }, g1);
  const mov = mk("g", {}, g1);

  /* ================= scale ================= */
  const sY = 8, g3 = mk("g", {}, svg);
  mk("rect", { x: 12, y: sY, width: 976, height: 132, rx: 16, class: "sv-panel" }, g3);
  partTitle(g3, 28, sY + 26, "1", "At scale: nine streams like the one below");
  const g3c = mk("g", {}, g3);
  const big = tx(g3c, 28, sY + 84, "0", { class: "sv-big" });
  tx(g3c, 30, sY + 108, "tasks in 9 streams", { class: "sv-lab" });
  const sub = tx(g3c, 190, sY + 108, "", { class: "sv-small2" });
  const DOM = [["Retail", 2154, 1686], ["Banking", 1436, 1116], ["Pitch", 918, 408]];
  const BX = 500, BW = 380;
  const bars = DOM.map(([d, s, te], i) => {
    const y = sY + 30 + i * 30;
    tx(g3c, BX - 12, y + 15, d, { class: "sv-small2", "text-anchor": "end" });
    const r1 = mk("rect", { x: BX, y: y + 3, width: 0, height: 17, rx: 8.5, class: "sv-t sv-s sv-done" }, g3c);
    const r2 = mk("rect", { x: BX, y: y + 3, width: 0, height: 17, rx: 8.5, class: "sv-t sv-x sv-done" }, g3c);
    const lab = tx(g3c, 0, y + 16, "", { class: "sv-m2" });
    return { s, te, r1, r2, lab };
  });
  [["serving", "sv-s"], ["test", "sv-x"]].forEach(([l, c], i) => { mk("rect", { x: BX + i * 80, y: sY + 118, width: 11, height: 11, rx: 5.5, class: "sv-t sv-done " + c }, g3c);
    tx(g3c, BX + 16 + i * 80, sY + 128, l, { class: "sv-small" }); });

  /* ------------------------------------------------ draw at time t */
  function draw(tAll) {
    const t = tAll - OFF;
    g1.classList.toggle("sv-dim", t < 0);
    mov.innerHTML = ""; let lastDone = -1, busy = false;
    tickets.forEach((tk, k) => {
      const t0 = k * DT; if (t < t0) return;
      if (t >= done(k)) lastDone = k;
      const cls = tk.kind === "serve" ? "sv-s" : "sv-x";
      if (t < arrive(k)) mk("rect", { x: beltX0 + (t - t0) / TRAVEL * (ax - aw / 2 - 28 - beltX0), y: beltY - 16, width: 20, height: 32, rx: 7, class: "sv-t sv-live " + cls }, mov);
      else if (t < done(k)) busy = true;
      else if (t < stored(k)) {
        const u = ease((t - done(k)) / FLY);
        if (tk.kind === "serve") {             // the score flies into memory
          const x = ax + aw / 2 + u * (mX + 14 - ax - aw / 2), y = beltY - 20 + u * (mY + 36 - beltY + 20) + Math.sin(u * Math.PI) * 12;
          const gg = mk("g", {}, mov);
          mk("rect", { x, y, width: 92, height: 20, rx: 10, class: tk.ok ? "sv-score-ok" : "sv-score-no" }, gg);
          tx(gg, x + 46, y + 14, tk.ok ? "Score: 100/100" : "Score: 0/100", { class: "sv-scoretxt " + (tk.ok ? "sv-oktxt" : "sv-notxt"), "text-anchor": "middle" });
        } else {                                // a test goes to the evaluator
          const x = ax + aw / 2 + u * (evX - 12 - ax - aw / 2), y = beltY + 6 + u * (evY + 13 - beltY - 6);
          mk("rect", { x, y, width: 16, height: 26, rx: 4, class: "sv-t sv-live sv-x", opacity: String(1 - clamp((u - 0.7) / 0.3)) }, mov);
        }
      }
    });
    agentBusy(busy, tAll);
    ticks.forEach((r, k) => r.classList.toggle("sv-done", t >= done(k)));
    const ck = Math.min(tickets.length - 1, lastDone + 1);
    cursor.setAttribute("x", +ticks[ck].getAttribute("x") - 2);
    cursor.style.opacity = t < STREAM_END ? 1 : 0;
    const curW = tickets[ck].w;
    winBox.forEach((m, i) => m.classList.toggle("sv-on", t < STREAM_END && i === curW));
    tsTxt.textContent = `Current timestamp: ${lastDone >= 0 ? tickets[lastDone].ts : 0}`;
    // memory rows: the latest stored serving outcomes, newest on top
    memG.innerHTML = "";
    const storedK = tickets.map((tk, k) => k).filter(k => tickets[k].kind === "serve" && t >= stored(k));
    const recent = storedK.slice(-NROWS).reverse();
    recent.forEach((k, i) => {
      const tk = tickets[k], age = t - stored(k), y = mY + 34 + i * rowH, gg = mk("g", { opacity: String(i === 0 ? clamp(age / 0.3) : 1 - i * 0.12) }, memG);
      tx(gg, mX + 14, y + 14, `t=${tk.ts}`, { class: "sv-m2" });
      tx(gg, mX + 62, y + 14, `depends on ${tk.c}`, { class: "sv-small2" });
      mk("rect", { x: mX + 200, y: y + 1, width: 110, height: 19, rx: 9.5, class: tk.ok ? "sv-score-ok" : "sv-score-no" }, gg);
      tx(gg, mX + 255, y + 14.5, tk.ok ? "✓ 100/100" : "✗ 0/100", { class: "sv-scoretxt " + (tk.ok ? "sv-oktxt" : "sv-notxt"), "text-anchor": "middle" });
    });
    // inferred policy, compared with the current window's actual policy
    belG.innerHTML = "";
    const lastK = storedK.length ? storedK[storedK.length - 1] : -1;
    const b = lastK >= 0 ? tickets[lastK].after : new Set(), truth = POLICY[curW];
    const changed = lastK >= 0 && t - stored(lastK) < 1.0 ? new Set([...tickets[lastK].after].filter(c => !tickets[lastK].before.has(c)).concat([...tickets[lastK].before].filter(c => !tickets[lastK].after.has(c)))) : new Set();
    const all = [...new Set([...truth, ...b])].sort();
    all.forEach((c, i) => chip(belG, mX + 14 + i * 46, mY + 178, c, (b.has(c) ? (truth.includes(c) ? "sv-bl-ok" : "sv-bl-stale") : "sv-bl-miss") + (changed.has(c) ? " sv-flash" : "")));
    // scale
    const t3 = tAll * SCALE_SPEED, on3 = clamp(t3 / 0.6);
    g3c.style.opacity = on3; g3.classList.toggle("sv-dim", on3 === 0);
    const u = ease(t3 / SCALE_DUR);
    big.textContent = Math.round(7718 * u).toLocaleString("en-US");
    sub.textContent = t3 > 1 ? "53 windows · 4,508 serving · 3,210 test" : "";
    bars.forEach((bb, i) => { const v = ease((t3 - 0.4 - i * 0.3) / 1.6), sw = bb.s / 3840 * BW * v, tw2 = bb.te / 3840 * BW * v;
      bb.r1.setAttribute("width", sw); bb.r2.setAttribute("x", BX + sw + (v > 0 ? 1.5 : 0)); bb.r2.setAttribute("width", tw2);
      bb.lab.setAttribute("x", BX + 8 + sw + tw2); bb.lab.textContent = v > 0.02 ? Math.round((bb.s + bb.te) * v).toLocaleString("en-US") : ""; });
    const i = tAll >= PHASES[1] ? 1 : 0, ends = [OFF, TOTAL];
    steps.forEach((s, j) => { s.style.setProperty("--p", j < i ? 1 : j > i ? 0 : clamp((tAll - PHASES[j]) / (ends[j] - PHASES[j])));
      s.setAttribute("aria-current", j === i ? "step" : "false"); });
  }

  /* ------------------------------------------------ playback */
  const ctl = document.getElementById("stream-ctl");
  const play = ctl.querySelector("[data-a=play]"), speedBtn = ctl.querySelector("[data-a=speed]");
  const steps = [...ctl.querySelectorAll("[data-step]")];
  let t = 0, speed = 1, playing = !matchMedia("(prefers-reduced-motion: reduce)").matches, last = null;
  if (!playing) t = OFF + STREAM_END;
  const setPlay = p => { playing = p; play.textContent = p ? "Pause" : "Play"; play.setAttribute("aria-pressed", p); };
  setPlay(playing);
  play.onclick = () => setPlay(!playing);
  ctl.querySelector("[data-a=restart]").onclick = () => { t = 0; setPlay(true); draw(t); };
  speedBtn.onclick = () => { speed = speed === 1 ? 2 : 1; speedBtn.textContent = speed + "×"; };
  steps.forEach((s, i) => s.onclick = () => { t = PHASES[i]; draw(t); });
  function frame(now) {
    if (last != null && playing) { t += (now - last) / 1000 * speed; if (t >= TOTAL) t = 0; draw(t); }
    last = now; requestAnimationFrame(frame);
  }
  draw(t); requestAnimationFrame(frame);
  new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting && playing) { setPlay(false); host.dataset.autopaused = "1"; }
    else if (e.isIntersecting && host.dataset.autopaused) { delete host.dataset.autopaused; setPlay(true); } })).observe(host);
})();
