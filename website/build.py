#!/usr/bin/env python3
"""Build the ServeLearnBench project website (website/index.html).

Inputs, all in this directory:

    src/home.html             the home page
    src/leaderboard.html      the full leaderboard page
    src/_head.html, _nav.html, _foot.html   shared page parts
    src/style.css             styles (light and dark themes)
    src/charts.js             the SVG chart kit
    src/app.js                leaderboard and figure rendering
    src/protocol_figure.html  the protocol figure (inline SVG)
    data/main_results_v2.json per-setting scores of every model and method (the paper's Table 2)
    data/cost_total.json      total API cost of every row
    data/tiers.csv            scenario scale
    data/figures.json         the data behind every figure
    project.json              authors, affiliations and links
    ../dataset/data/          sample task instructions for the domain cards

The output is one self-contained file (inline CSS, JS and data; only the web
fonts load from Google Fonts).

    python3 website/build.py
    python3 -m http.server 9393 -d website       # local preview
"""
import csv
import html
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHORT = {"Claude Opus 5": "Opus 5"}
DOMAIN_TEXT = {
    "retail": ("Retail", "Multi-turn customer support over an order database: cancel or modify pending orders, return or exchange "
               "delivered items. Correct only if the final database state is the intended one, or the request is refused with "
               "the right reason code.", ["tool use", "categorical rules", "refusal codes"], "retail_l1", 1),
    "banking": ("Banking", "Case review: authorize card transactions, review credit-limit increases, screen outbound transfers. "
                "Harder tiers add ranked alternatives and payment batches. Hidden policies are thresholds, caps and corridors.",
                ["decisions", "numeric thresholds", "composition"], "banking_l1", 0),
    "pitch": ("Pitch", "Write a 40–80 word sales pitch from a factual product sheet for a customer whose taste is never stated. "
              "A rubric rewards liked attributes and penalizes disliked, false or invented claims.",
              ["generation", "latent preferences", "LLM judge"], "pitch_l1", 0),
}


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def leaderboard_rows(snap, cost):
    hidden = {m["key"]: set(m.get("hidden_harnesses", [])) for m in snap["models"]}
    names = {m["key"]: SHORT.get(m["name"], m["name"]) for m in snap["models"]}
    size = {m["key"]: m["latex"][m["latex"].index("(") + 1:m["latex"].index(")")] for m in snap["models"] if "(" in m.get("latex", "")}
    rows = []
    for r in snap["rows"]:
        if r["harness"] in hidden.get(r["model"], ()):
            continue
        rows.append(dict(model=r["model"], model_name=names[r["model"]], size=size.get(r["model"], ""),
                         harness=r["harness"], harness_name=snap["harnesses"][r["harness"]],
                         ref=r["harness"] in ("blind", "oracle"), avg_h=num(r["avg_h"]), avg_f=num(r["avg_f"]),
                         cost=cost.get(f"{r['model']}/{r['harness']}"), values=[num(v) for v in r["values"]]))
    return rows


def people(project):
    out = []
    if project.get("authors"):
        names = []
        for a in project["authors"]:
            sup = ",".join(str(i) for i in a.get("affiliations", [])) + a.get("mark", "")
            name = html.escape(a["name"])
            if a.get("url"):
                name = f'<a href="{html.escape(a["url"])}">{name}</a>'
            names.append(f"{name}<sup>{sup}</sup>" if sup else name)
        out.append('<p class="authors">' + ", ".join(names) + "</p>")
        if project.get("affiliations"):
            out.append('<p class="affils">' + " &nbsp;·&nbsp; ".join(
                f"<sup>{i}</sup>{html.escape(x)}" for i, x in enumerate(project["affiliations"], 1)) + "</p>")
        if project.get("note"):
            out.append(f'<p class="affils">{html.escape(project["note"])}</p>')
    return "\n".join(out)


ICONS = {
    "Paper": '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M4 1.5h5.5L12.5 4.5v10H4z"/><path d="M9.5 1.5v3h3M6 8h4.5M6 10.5h4.5"/></svg>',
    "Code": '<svg viewBox="0 0 16 16" fill="currentColor"><path d="M8 .2a8 8 0 0 0-2.5 15.6c.4 0 .5-.2.5-.4v-1.4c-2.2.5-2.7-1-2.7-1-.4-.9-.9-1.2-.9-1.2-.7-.5.1-.5.1-.5.8.1 1.2.8 1.2.8.7 1.2 1.9.9 2.3.7.1-.5.3-.9.5-1.1-1.8-.2-3.6-.9-3.6-4 0-.9.3-1.6.8-2.1-.1-.2-.4-1 .1-2.1 0 0 .7-.2 2.2.8a7.6 7.6 0 0 1 4 0c1.5-1 2.2-.8 2.2-.8.4 1.1.2 1.9.1 2.1.5.6.8 1.3.8 2.1 0 3.1-1.9 3.8-3.6 4 .3.3.6.8.6 1.5v2.2c0 .2.1.5.6.4A8 8 0 0 0 8 .2z"/></svg>',
    "Dataset": '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4"><ellipse cx="8" cy="3.5" rx="5.5" ry="2"/><path d="M2.5 3.5v9c0 1.1 2.5 2 5.5 2s5.5-.9 5.5-2v-9M2.5 8c0 1.1 2.5 2 5.5 2s5.5-.9 5.5-2"/></svg>',
}


def buttons(project):
    out = []
    for name in ("Paper", "Code", "Dataset"):
        url = project.get("links", {}).get(name, "")
        label = name if url or name != "Paper" else "Paper (coming soon)"
        cls = "btn" + ("" if url else " disabled")
        href = f' href="{html.escape(url)}"' if url else ' aria-disabled="true"'
        out.append(f'<a class="{cls}"{href}>{ICONS[name]}{label}</a>')
    out.append('<a class="btn primary" href="#start">Evaluate an agent →</a>')
    return "".join(out)


def nav_links(project):
    links = project.get("links", {})
    return "".join(f'<a class="icon-btn" href="{html.escape(links[n])}" aria-label="{n}" title="{n}">{ICONS[n]}</a>'
                   for n in ("Code", "Dataset") if links.get(n))


def stats(tiers):
    serving = sum(int(t["serving"]) for t in tiers)
    test = sum(int(t["test"]) for t in tiers)
    windows = sum(int(t["stages"]) for t in tiers)
    items = [(3, "domains"), (9, "scenarios"), (windows, "environment windows"), (f"{serving:,}", "serving tasks"),
             (f"{test:,}", "held-out test tasks"), ("28", "model–method pairs")]
    return "".join(f'<div class="stat"><b>{v}</b><span>{k}</span></div>' for v, k in items)


def findings_html(rows, columns, figures):
    learners = [r for r in rows if not r["ref"]]
    oracle = statistics.mean(r["avg_h"] for r in rows if r["harness"] == "oracle")
    blind = statistics.mean(r["avg_h"] for r in rows if r["harness"] == "blind")
    ci = columns.index("Retail H L3")
    rl3 = [r["values"][ci] for r in learners]
    ratio = [r["cost"] / next(o["cost"] for o in rows if o["model"] == r["model"] and o["harness"] == "oracle") for r in learners]
    kimi_prime = next(r["cost"] for r in rows if r["model"] == "kimi-k3" and r["harness"] == "prime")
    kimi_oracle = next(r["cost"] for r in rows if r["model"] == "kimi-k3" and r["harness"] == "oracle")
    acquisition = {p["label"]: p["x"] for p in figures["ka"]["fit"]["points"]}
    cards = [
        ("Capability is not learning", f"{oracle:.1f}<small>vs {blind:.1f}</small>",
         f"Mean Hidden score with the policy disclosed (Oracle) and without it (Blind). Every model can do the tasks once told the policy, "
         f"yet on Retail L3 the best of {len(learners)} learning pairs reaches only {max(rl3):.1f} (median {statistics.median_low(rl3):.1f})."),
        ("Adaptation is costly", f"{min(ratio):.0f}–{max(ratio):.0f}<small>× Oracle's cost</small>",
         f"Inferring a hidden policy costs {min(ratio):.0f} to {max(ratio):.0f} times as much as following a disclosed one "
         f"(Kimi K3: Prime ${kimi_prime:,.0f} vs. Oracle ${kimi_oracle:,.0f}), and it can degrade behavior that needed no learning."),
        ("Exploration is the bottleneck", "ρ = 1.00",
         "A method's answer diversity while it is still failing ranks the five methods exactly as their Hidden reward does "
         "(ρ = 0.75 over 30 method–domain–tier cells)."),
        ("Learning can be fast after first success", f"{acquisition['Prime']:.0f}% / {acquisition['CH']:.0f}%",
         "Once Prime and CH find the correct behavior, they quickly reuse it, averaging "
         f"{acquisition['Prime']:.0f}% and {acquisition['CH']:.0f}% correctness over the next five encounters. "
         "For these harnesses, finding the right behavior is a larger obstacle than reusing it."),
    ]
    return "".join(f'<div class="finding"><h3>{t}</h3><span class="big">{b}</span><p>{p}</p></div>' for t, b, p in cards)


def domains_html():
    out = []
    for key, (name, text, pills, scen, idx) in DOMAIN_TEXT.items():
        p = HERE.parent / "dataset" / "data" / scen / "serving.jsonl"
        example = ""
        if p.exists():
            rows = [json.loads(line) for line in open(p)]
            rows.sort(key=lambda r: len(r["instruction"]))
            ins = rows[len(rows) // 3 + idx]["instruction"]
            example = f'<pre class="ex">{html.escape(ins)}</pre>'
        out.append(f'<div class="domain"><div class="dh"><h3>{name}</h3><span>L1 · L2 · L3</span></div><p>{text}</p>{example}'
                   f'<div class="pills">' + "".join(f'<span class="pill">{x}</span>' for x in pills) + "</div></div>")
    return "".join(out)


def tier_rows(tiers):
    rows = "".join(f"<tr><td class='l mono'>{t['domain'].lower()}_{t['tier'].lower()}</td><td class='l'>{html.escape(t['latent_structure'])}</td>"
                   f"<td class='num'>{t['stages']}</td><td class='num'>{int(t['serving']):,}</td><td class='num'>{int(t['test']):,}</td></tr>" for t in tiers)
    tot = (sum(int(t["stages"]) for t in tiers), sum(int(t["serving"]) for t in tiers), sum(int(t["test"]) for t in tiers))
    return rows + f"<tr><td class='l'><b>Total</b></td><td></td><td class='num'><b>{tot[0]}</b></td><td class='num'><b>{tot[1]:,}</b></td><td class='num'><b>{tot[2]:,}</b></td></tr>"


def bibtex(project):
    names = " and ".join(f"{a['name'].split()[-1]}, {' '.join(a['name'].split()[:-1])}" for a in project.get("authors", []))
    publication = ""
    if project.get("arxiv_id"):
        publication = (f",\n  eprint = {{{project['arxiv_id']}}},"
                       "\n  archivePrefix = {arXiv},"
                       f"\n  primaryClass = {{{project.get('primary_class', 'cs.LG')}}},"
                       f"\n  url    = {{{project['links']['Paper']}}}")
    return html.escape(f"""@misc{{zheng2026servelearnbench,
  title  = {{{project.get('title', 'ServeLearnBench')}}},
  author = {{{names}}},
  year   = {{2026}}{publication}
}}""")


SHAPE_SVG = {
    "o": '<circle cx="6" cy="6" r="4.5"/>', "d": '<path d="M6 .8 11.2 6 6 11.2.8 6z"/>', "s": '<rect x="1.8" y="1.8" width="8.4" height="8.4" rx="1"/>',
    "^": '<path d="M6 1 11 10.5H1z"/>', "h": '<path d="M6 .8 10.5 3.4v5.2L6 11.2 1.5 8.6V3.4z"/>', "p": '<path d="M4.3 1h3.4v3.3H11v3.4H7.7V11H4.3V7.7H1V4.3h3.3z"/>',
}


def frontier_legend():
    harn = [("Prime", "--prime"), ("CH", "--ch"), ("RAG", "--rag"), ("Mem0", "--mem0"), ("SkillOpt", "--skillopt")]
    models = [("GLM-5.3 Flash", "o"), ("DS-V4.1 Flash", "d"), ("GLM-5.3", "s"), ("Kimi K3", "^"), ("Opus 5", "h"), ("GPT-5.6 Terra", "p")]
    a = "".join(f'<span><i class="dot" style="background:var({c})"></i>{n}</span>' for n, c in harn)
    b = "".join(f'<span><svg width="12" height="12" viewBox="0 0 12 12" style="fill:var(--muted)">{SHAPE_SVG[s]}</svg>{n}</span>' for n, s in models)
    return ('<div class="legend-group"><span class="legend-label">Method</span>' + a + '</div>'
            '<div class="legend-group"><span class="legend-label">Model</span>' + b + '</div>')


def cell_legend():
    harn = [("Prime", "--prime"), ("CH", "--ch"), ("RAG", "--rag"), ("Mem0", "--mem0"), ("SkillOpt", "--skillopt")]
    tiers = [("L1", "o"), ("L2", "s"), ("L3", "^")]
    return ("".join(f'<span><i class="dot" style="background:var({c})"></i>{n}</span>' for n, c in harn)
            + "".join(f'<span><svg width="12" height="12" viewBox="0 0 12 12" style="fill:var(--muted)">{SHAPE_SVG[s]}</svg>{n}</span>' for n, s in tiers))


FONTS = {
    "default": "IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600",
    "terminal": "JetBrains+Mono:wght@400;500;600;700;800",
    "journal": "Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Source+Sans+3:wght@400;500;600&family=Source+Code+Pro:wght@400;500",
    "observatory": "Manrope:wght@400;500;600;700;800&family=DM+Mono:wght@400;500",
    "swiss": "Archivo:wght@400;500;600;700;800;900&family=Archivo+Narrow:wght@500;700&family=JetBrains+Mono:wght@400;500",
    "soft": "Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500",
}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build website/index.html")
    ap.add_argument("--theme", default="journal", choices=sorted(FONTS))
    ap.add_argument("--out", default=str(HERE), help="output directory")
    a = ap.parse_args()
    snap = json.loads((HERE / "data/main_results_v2.json").read_text())
    cost = json.loads((HERE / "data/cost_total.json").read_text())
    tiers = list(csv.DictReader((HERE / "data/tiers.csv").open()))
    figures = json.loads((HERE / "data/figures.json").read_text())
    project = json.loads((HERE / "project.json").read_text())
    rows = leaderboard_rows(snap, cost)
    data = {"models": [{"key": m["key"], "name": SHORT.get(m["name"], m["name"])} for m in snap["models"]],
            "columns": snap["columns"], "rows": rows, "figures": figures}
    drop = "".join(f'<div><p class="panel-title"><b>{html.escape(p["title"])}</b> pairs</p><div class="chart" id="drop-{i}"></div></div>'
                   for i, p in enumerate(figures["drop"]["panels"]))
    ed = "".join(f'<div><p class="panel-title">{html.escape(p["caption"])}</p><div class="chart" id="ed-{i}"></div></div>'
                 for i, p in enumerate(figures["expdomain"]["panels"]))
    phase_legend = ('<span><i class="sq" style="background:var(--accent);opacity:.35"></i>Cost per task (left axis)</span>'
                    '<span><i style="background:var(--warn)"></i>Reward on Hidden</span><span><i style="background:var(--good)"></i>Reward on all cases</span>')
    src = HERE / "src"
    chrome = {"%%HEAD%%": (src / "_head.html").read_text(), "%%NAV%%": (src / "_nav.html").read_text(),
              "%%FOOT%%": (src / "_foot.html").read_text()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for tpl, name, title in (("home.html", "index.html", "ServeLearnBench"),
                             ("leaderboard.html", "leaderboard.html", "ServeLearnBench Leaderboard"),
                             ("benchmark.html", "benchmark.html", "ServeLearnBench Benchmark")):
        page = (src / tpl).read_text()
        for k2, v2 in chrome.items():
            page = page.replace(k2, v2)
        page = page.replace("<title>ServeLearnBench</title>", f"<title>{title}</title>")
        subs = {
            "%%CSS%%": (src / "style.css").read_text() + ("" if a.theme == "default" else (src / "themes" / f"{a.theme}.css").read_text()),
            "%%FONTS%%": ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
                      f'<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family={FONTS[a.theme]}&display=swap">'),
            "%%PROTOCOL%%": (src / "protocol_figure.html").read_text(),
            "%%PEOPLE%%": people(project), "%%BUTTONS%%": buttons(project), "%%NAVLINKS%%": nav_links(project),
            "%%STATS%%": stats(tiers), "%%FINDINGS%%": findings_html(rows, snap["columns"], figures), "%%DOMAINS%%": domains_html(),
            "%%TIERROWS%%": tier_rows(tiers), "%%BIBTEX%%": bibtex(project), "%%SNAPSHOT%%": html.escape(snap.get("generated", "")[:10]),
            "%%FRONTIER_LEGEND%%": frontier_legend(), "%%CELL_LEGEND%%": cell_legend(), "%%PHASE_LEGEND%%": phase_legend,
            "%%DROP_PANELS%%": drop, "%%ED_PANELS%%": ed,
            "%%DATA%%": json.dumps(data, separators=(",", ":")).replace("</", "<\\/"),
            "%%CHARTS_JS%%": (src / "charts.js").read_text(), "%%APP_JS%%": (src / "app.js").read_text(), "%%STREAM_JS%%": (src / "stream.js").read_text(),
        }
        for k, v in subs.items():
            page = page.replace(k, v)
        (out / name).write_text(page)
        print(f"wrote {out / name} [{a.theme}] ({len(page) // 1024} KB)" + (" UNREPLACED %%" if "%%" in page else ""))


if __name__ == "__main__":
    main()
