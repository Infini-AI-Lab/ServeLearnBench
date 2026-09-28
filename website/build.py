#!/usr/bin/env python3
"""Build the ServeLearnBench project website (website/index.html).

The page is generated from the files in this directory:

    data/main_results_v2.json   per-setting scores of every model and method (the paper's Table 2)
    data/cost_total.json        total API cost of every row
    data/tiers.csv              scenario scale
    assets/fig/*.png            the paper's figures
    project.json                authors, affiliations and links

and is self-contained (inline CSS/JS, no network access needed).

    python3 website/build.py
    python3 -m http.server 9393 -d website       # local preview

`--refresh-from DIR` first copies the data files from DIR/data and renders the
figures from their PDFs under DIR (the paper sources), then builds.
Averages follow the paper: one score per model-method-setting cell, then an
unweighted mean over settings.
"""
import argparse
import csv
import html
import json
import shutil
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent

FIGS = [  # (key, paper pdf, title, caption)  -- captions condensed from the paper
    ('overview', 'figures/flywheelbench_fig1.pdf', 'Protocol',
     'The stream is divided into environment windows. Each window holds serving tasks with outcome feedback, followed by held-out test '
     'tasks without feedback. The active hidden policy may introduce, withdraw, or reactivate behavioral components at window '
     'boundaries; the learner never sees the policy or any correction and must infer it from serving interactions and scores.'),
    ('frontier', 'figures/frontier/frontier.pdf', 'Cost vs. performance',
     'Hidden reward (Avg. H of the leaderboard) against (a) total tokens and (b) API cost per held-out test task, for all 28 model-harness '
     'pairs. The staircase is the Pareto frontier; the shaded region is dominated by it.'),
    ('avg', 'figures/avg_performance/avg_performance.pdf', 'Average performance',
     'Hidden and Fully Specified scores of the four open-weight models, averaged (a) by model and (b) by harness, with the Blind and '
     'Oracle references.'),
    ('costmerged', 'figures/cost_merged/cost_merged.pdf', 'Cost per task against reward',
     'Average API cost per task (bars, serving stream including learning work plus held-out test) and reward (lines, Hidden and all '
     'cases, averaged over the nine settings), (a) by model and (b) by learning harness.'),
    ('speed', 'figures/learning_speed/learning_speed.pdf', 'Learning speed for new policies',
     '(a, b) Correctness on Hidden serving tickets against the number of earlier tickets of the same policy, Banking and Retail. '
     '(c, d) AULC10, the mean correctness over the first ten examples of a new policy, by model and harness.'),
    ('fs', 'figures/fs_method_comparison/fs_method_comparison.pdf', 'Does adaptation harm Fully Specified cases?',
     'Fully Specified accuracy with and without adaptation, averaged over the four open-weight models and three tiers.'),
    ('explore', 'figures/exploration/exploration_strip.pdf', 'Exploration predicts adaptation',
     '(a, c) Mean number of distinct answers after k Hidden serving tickets of the same policy, by harness and by model. '
     '(b, d) Exploration AUC against mean Hidden test reward over the six Retail and Banking settings; rho is the Spearman rank correlation.'),
    ('ka', 'figures/exploration/consolidation_half.pdf', 'Knowledge acquisition after first success',
     'Left: correctness on Hidden serving tickets aligned at the first correct response. Right: AUC_KA, mean correctness over the next '
     'five tickets, against mean Hidden test reward.'),
    ('drop', 'figures/learning_speed/learn_vs_drop.pdf', 'Learning a policy versus dropping it',
     'Correctness when a hidden policy first applies (learning) compared with when the same policy is withdrawn and the documented '
     'behavior returns (dropping).'),
    ('expdomain', 'figures/exploration/exploration_by_domain.pdf', 'Exploration by domain',
     'Answer-diversity curves and their relation to reward, shown separately for Retail and Banking.'),
    ('expcells', 'figures/exploration/exploration_cells.pdf', 'Exploration per harness, domain and tier',
     'One point per harness-domain-tier cell: exploration AUC against the cell\'s Hidden reward (rho = 0.75 over 30 cells).'),
    ('phase', 'figures/updated/cost_correctness_row.pdf', 'Cost and reward by phase',
     'The cost comparison with the serving stream (including learning work) and the held-out test kept apart.'),
    ('learncost', 'figures/updated/run_cost_all.pdf', 'Where the serving budget goes',
     'Tokens and API cost per serving task, split between serving the stream and updating the learned state.'),
]

SHORT = {'Claude Opus 5': 'Opus 5'}

HARNESS_INFO = [  # condensed from the paper, Appendix "Learning Harnesses"
    ('Blind', 'Reference. Only learner-visible task information; retains no serving experience.'),
    ('Oracle', 'Reference. Additionally receives the active hidden policy of each window.'),
    ('RAG', 'Adds each finished block of twelve served cases, with their score lines, to a BM25 index and retrieves the eight most similar cases into the prompt. No learning-side model calls.'),
    ('Mem0', 'Memory layer with its own extraction model: two memory tools during the task, and a post-outcome memory turn after every serving ticket.'),
    ('SkillOpt', 'Prompt optimizer: each block of twelve serving tickets is one step (six to train, six to select); a candidate skill document replaces the current one only if it passes a strict gate.'),
    ('CH', 'Continual Harness: tools for memories, skills, subagents and code. The acting agent never sees its score; a Refiner uses it to revise prompt, memories, skills and subagents on a fixed step cadence.'),
    ('Prime', 'Agent runtime with its own sandbox and code interpreter, a fresh session per ticket, and a post-score turn in which it writes its own memory, skill and prompt files.'),
]


def render_pdf(pdf: Path, png: Path, dpi=220):
    import pymupdf
    doc = pymupdf.open(pdf)
    doc[0].get_pixmap(dpi=dpi, alpha=False).save(png)


def load(_unused, site: Path):
    snap = json.loads((site / 'data/main_results_v2.json').read_text())
    cost = json.loads((site / 'data/cost_total.json').read_text())
    tiers = list(csv.DictReader((site / 'data/tiers.csv').open()))
    return snap, cost, tiers


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def rows_for_page(snap, cost):
    hidden = {m['key']: set(m.get('hidden_harnesses', [])) for m in snap['models']}
    names = {m['key']: SHORT.get(m['name'], m['name']) for m in snap['models']}
    size = {}
    for m in snap['models']:
        lx = m.get('latex', '')
        if '(' in lx:
            size[m['key']] = lx[lx.index('(') + 1:lx.index(')')]
    out = []
    for r in snap['rows']:
        if r['harness'] in hidden.get(r['model'], ()):
            continue
        out.append(dict(model=r['model'], model_name=names[r['model']], size=size.get(r['model'], ''),
                        harness=r['harness'], harness_name=snap['harnesses'][r['harness']],
                        ref=r['harness'] in ('blind', 'oracle'),
                        avg_h=num(r['avg_h']), avg_f=num(r['avg_f']),
                        cost=cost.get(f"{r['model']}/{r['harness']}"),
                        values=[num(v) for v in r['values']]))
    return out


CSS = r"""
:root{--bg:#fbfaf7;--panel:#ffffff;--ink:#1f2328;--ink2:#57606a;--muted:#8b949e;--line:#e6e2da;--accent:#3f5f95;--accent2:#5fa37e;
--chip:#f1eee7;--ref:#f6f4ef;--bar:#dbe5f2;--barH:#3f5f95;--warn:#b35900;--shadow:0 1px 2px rgba(0,0,0,.04),0 4px 16px rgba(0,0,0,.04)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#111418;--panel:#181c21;--ink:#e6e8eb;--ink2:#b3b9c1;--muted:#7d8590;
--line:#2a3038;--accent:#86a8dc;--accent2:#7cc49b;--chip:#222830;--ref:#1c2127;--bar:#26344a;--barH:#86a8dc;--shadow:none}}
:root[data-theme="dark"]{--bg:#111418;--panel:#181c21;--ink:#e6e8eb;--ink2:#b3b9c1;--muted:#7d8590;--line:#2a3038;--accent:#86a8dc;
--accent2:#7cc49b;--chip:#222830;--ref:#1c2127;--bar:#26344a;--barH:#86a8dc;--shadow:none}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.wrap{max-width:1180px;margin:0 auto;padding:0 20px}
nav{position:sticky;top:0;z-index:10;background:color-mix(in srgb,var(--bg) 88%,transparent);backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
nav .wrap{display:flex;align-items:center;gap:18px;height:52px;overflow-x:auto;white-space:nowrap}
nav .brand{font-weight:700;letter-spacing:-.01em;color:var(--ink)}nav a.l{color:var(--ink2);font-size:14px}nav .sp{flex:1}
nav button{background:none;border:1px solid var(--line);color:var(--ink2);border-radius:8px;padding:4px 9px;cursor:pointer;font-size:13px}
header.hero{padding:64px 0 28px}
.kicker{color:var(--accent);font-weight:600;font-size:13px;letter-spacing:.08em;text-transform:uppercase}
h1{font-size:44px;line-height:1.1;letter-spacing:-.025em;margin:10px 0 12px}
.sub{font-size:20px;color:var(--ink2);max-width:820px;margin:0 0 18px}
.lede{max-width:860px;color:var(--ink2)}
.badges{display:flex;flex-wrap:wrap;gap:8px;margin:20px 0 6px}
.authors{font-size:17px;margin:4px 0 2px}.authors a{color:var(--ink)}.affils{color:var(--ink2);font-size:14px;margin:2px 0}
.badge.link{color:var(--ink);font-weight:600}.badge.link:hover{border-color:var(--accent);text-decoration:none}
.badge{border:1px solid var(--line);background:var(--panel);border-radius:999px;padding:5px 12px;font-size:13px;color:var(--ink2)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin:28px 0 8px}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px 16px;box-shadow:var(--shadow)}
.stat b{display:block;font-size:26px;letter-spacing:-.02em}.stat span{color:var(--muted);font-size:13px}
section{padding:40px 0 16px;scroll-margin-top:60px}
h2{font-size:28px;letter-spacing:-.02em;margin:0 0 6px}h3{font-size:18px;margin:26px 0 8px}
.secsub{color:var(--ink2);margin:0 0 20px;max-width:880px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow)}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:14px 16px;border-bottom:1px solid var(--line)}
.chip{border:1px solid var(--line);background:var(--chip);color:var(--ink2);border-radius:999px;padding:4px 11px;font-size:13px;cursor:pointer;user-select:none}
.chip.on{background:var(--accent);border-color:var(--accent);color:#fff}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:10px;overflow:hidden}
.seg button{border:0;background:var(--panel);color:var(--ink2);padding:5px 12px;font-size:13px;cursor:pointer}.seg button.on{background:var(--accent);color:#fff}
.ctl-label{font-size:12px;color:var(--muted);margin-right:2px;text-transform:uppercase;letter-spacing:.06em}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{font-size:12px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.04em;cursor:pointer;position:sticky;top:0;background:var(--panel)}
th.l,td.l{text-align:left}th.sorted::after{content:" ↓"}th.sorted.asc::after{content:" ↑"}
tr.ref td{background:var(--ref);color:var(--ink2)}tr.sep td{background:var(--panel);color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em;padding-top:14px}tr:hover td{background:color-mix(in srgb,var(--accent) 6%,var(--panel))}
td.rank{color:var(--muted);width:32px}.mname{font-weight:600}.msize{color:var(--muted);font-size:12px;margin-left:6px}
.hdot{display:inline-block;width:9px;height:9px;border-radius:3px;margin-right:7px;vertical-align:0}
.barcell{position:relative;min-width:120px}.barcell .bar{position:absolute;left:10px;top:50%;height:6px;margin-top:9px;border-radius:3px;background:var(--bar)}
.barcell .bar i{display:block;height:100%;border-radius:3px;background:var(--barH)}
.best{font-weight:700}.tag{font-size:11px;border-radius:6px;padding:1px 6px;margin-left:6px;background:var(--chip);color:var(--ink2)}
.foot{padding:12px 16px;color:var(--muted);font-size:13px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}
.finding{padding:20px 22px}.finding .n{font-size:13px;font-weight:700;color:var(--accent);letter-spacing:.06em}
.finding h3{margin:6px 0 8px}.finding p{margin:0;color:var(--ink2)}
.big{font-size:30px;font-weight:700;letter-spacing:-.02em;color:var(--ink);display:block;margin:10px 0 2px}
figure{margin:0;padding:18px}figure img{width:100%;height:auto;display:block;border-radius:8px;background:#fff;cursor:zoom-in}
figcaption{font-size:14px;color:var(--ink2);margin-top:12px}figcaption b{color:var(--ink)}
details{margin-top:18px}summary{cursor:pointer;font-weight:600;color:var(--accent);padding:6px 0}
.dom{padding:20px 22px}.dom h3{margin-top:0}.dom p{color:var(--ink2);margin:0 0 10px}
.pill{display:inline-block;font-size:12px;border-radius:999px;padding:2px 9px;background:var(--chip);color:var(--ink2);margin:2px 4px 2px 0}
.proto{padding:22px 24px}.proto p{color:var(--ink2)}.proto h3:first-child{margin-top:0}
code{background:var(--chip);padding:1px 5px;border-radius:5px;font-size:13px}
footer{border-top:1px solid var(--line);margin-top:40px;padding:26px 0 40px;color:var(--muted);font-size:13px}
#zoom{position:fixed;inset:0;background:rgba(0,0,0,.75);display:none;align-items:center;justify-content:center;z-index:50;padding:20px;cursor:zoom-out}
#zoom img{max-width:100%;max-height:100%;background:#fff;border-radius:8px}
@media (max-width:640px){h1{font-size:32px}.sub{font-size:17px}header.hero{padding:40px 0 18px}th,td{padding:7px 8px}}
"""

JS = r"""
const D = window.__DATA__;
const HC = {rag:'#8DB6D6',mem0:'#D9895B',ch:'#5FA37E',prime:'#3F5F95',skillopt:'#D4B04A',blind:'#9aa0a6',oracle:'#6e7781'};
const st = {models:new Set(D.models.map(m=>m.key)), refs:true, view:'summary', sort:'avg_h', asc:false};
const fmt = (v,d=1)=> v==null ? '–' : v.toFixed(d);
const money = v => v==null ? '–' : '$'+Math.round(v).toLocaleString('en-US');
function el(t,a={},...c){const e=document.createElement(t);for(const k in a){if(k==='class')e.className=a[k];else if(k==='html')e.innerHTML=a[k];else e.setAttribute(k,a[k]);}c.flat().forEach(x=>e.append(x instanceof Node?x:document.createTextNode(x)));return e;}
function controls(){
  const c=document.getElementById('lb-controls'); c.innerHTML='';
  c.append(el('span',{class:'ctl-label'},'Models'));
  D.models.forEach(m=>{const b=el('span',{class:'chip'+(st.models.has(m.key)?' on':'')},m.name);
    b.onclick=()=>{st.models.has(m.key)?st.models.delete(m.key):st.models.add(m.key); if(!st.models.size) D.models.forEach(x=>st.models.add(x.key)); controls(); table();}; c.append(b);});
  const r=el('span',{class:'chip'+(st.refs?' on':''),style:'margin-left:10px'},'Blind / Oracle');
  r.onclick=()=>{st.refs=!st.refs;controls();table();}; c.append(r);
  const seg=el('span',{class:'seg',style:'margin-left:auto'});
  [['summary','Summary'],['setting','Per setting']].forEach(([k,t])=>{const b=el('button',{class:st.view===k?'on':''},t);b.onclick=()=>{st.view=k;controls();table();};seg.append(b);});
  c.append(seg);
}
function table(){
  const rows=D.rows.filter(r=>st.models.has(r.model)&&(st.refs||!r.ref));
  const key=st.sort, get=r=> key.startsWith('v')? r.values[+key.slice(1)] : key==='model'? r.model_name : key==='harness'? r.harness_name : r[key];
  rows.sort((a,b)=>{ if(a.ref!==b.ref) return a.ref?1:-1; if(a.ref&&b.ref&&a.harness!==b.harness) return a.harness==='oracle'?-1:1; let x=get(a),y=get(b); if(x==null)return 1; if(y==null)return -1; if(typeof x==='string') return st.asc? x.localeCompare(y): y.localeCompare(x); return st.asc? x-y : y-x;});
  const best={}; D.rows.filter(r=>!r.ref).forEach(r=>{ if(r.avg_h!=null && (best[r.model]==null||r.avg_h>best[r.model])) best[r.model]=r.avg_h;});
  const t=document.getElementById('lb'); t.innerHTML='';
  const cols = st.view==='summary'
    ? [['#','rank','l'],['Model','model','l'],['Harness','harness','l'],['Avg. H','avg_h'],['Avg. F','avg_f'],['API cost','cost']]
    : [['Model','model','l'],['Harness','harness','l']].concat(D.columns.map((c,i)=>[c.replace(' H ',' H·').replace(' F ',' F·'),'v'+i])).concat([['Avg. H','avg_h'],['Avg. F','avg_f']]);
  const thead=el('thead'), tr=el('tr');
  cols.forEach(([lab,k,cls])=>{const th=el('th',{class:(cls||'')+(st.sort===k?' sorted'+(st.asc?' asc':''):'')},lab);
    if(k!=='rank') th.onclick=()=>{ if(st.sort===k) st.asc=!st.asc; else {st.sort=k; st.asc=(k==='model'||k==='harness'||k==='cost');} table();}; tr.append(th);});
  thead.append(tr); t.append(thead);
  const tb=el('tbody'); let rank=0, sepDone=false;
  rows.forEach(r=>{
    if(r.ref && !sepDone){ sepDone=true; const sp=el('tr',{class:'sep'}); sp.append(el('td',{colspan:String(cols.length),class:'l'},'Reference settings')); tb.append(sp);}
    const trr=el('tr',{class:r.ref?'ref':''});
    if(st.view==='summary') trr.append(el('td',{class:'rank l'}, r.ref?'':String(++rank)));
    const m=el('td',{class:'l'},el('span',{class:'mname'},r.model_name)); if(r.size) m.append(el('span',{class:'msize'},r.size)); trr.append(m);
    const h=el('td',{class:'l'},el('span',{class:'hdot',style:'background:'+HC[r.harness]}),r.harness_name);
    if(!r.ref && best[r.model]===r.avg_h) h.append(el('span',{class:'tag'},'best learner')); trr.append(h);
    if(st.view==='summary'){
      const bc=el('td',{class:'barcell'+(!r.ref&&best[r.model]===r.avg_h?' best':'')},fmt(r.avg_h));
      if(r.avg_h!=null){const b=el('span',{class:'bar',style:'width:calc(100% - 20px)'}),i=el('i',{style:'width:'+r.avg_h+'%'});b.append(i);bc.append(b);} trr.append(bc);
      trr.append(el('td',{},fmt(r.avg_f))); trr.append(el('td',{},money(r.cost)));
    } else {
      r.values.forEach(v=>trr.append(el('td',{},fmt(v)))); trr.append(el('td',{class:'best'},fmt(r.avg_h))); trr.append(el('td',{},fmt(r.avg_f)));
    }
    tb.append(trr);});
  t.append(tb);
}
controls(); table();
document.querySelectorAll('figure img').forEach(i=>i.onclick=()=>{const z=document.getElementById('zoom');z.querySelector('img').src=i.src;z.style.display='flex';});
document.getElementById('zoom').onclick=e=>e.currentTarget.style.display='none';
const tbtn=document.getElementById('theme');
tbtn.onclick=()=>{const r=document.documentElement;const dark=r.dataset.theme? r.dataset.theme==='dark' : matchMedia('(prefers-color-scheme: dark)').matches;r.dataset.theme=dark?'light':'dark';try{localStorage.setItem('slb-theme',r.dataset.theme)}catch(e){}};
try{const s=localStorage.getItem('slb-theme'); if(s) document.documentElement.dataset.theme=s;}catch(e){}
"""


def fig_html(k, figs, cls='card'):
    key, _, title, cap = figs[k]
    return (f'<div class="{cls}"><figure><img src="assets/fig/{key}.png" alt="{html.escape(title)}">'
            f'<figcaption><b>{html.escape(title)}{'' if title.endswith('?') else '.'}</b> {html.escape(cap)}</figcaption></figure></div>')


def people(project):
    """Author line, affiliations and link buttons from project.json."""
    aff = project.get("affiliations", [])
    out = []
    if project.get("authors"):
        names = []
        for a in project["authors"]:
            sup = ",".join(str(i) for i in a.get("affiliations", []))
            mark = a.get("mark", "")
            name = html.escape(a["name"])
            name = f'<a href="{html.escape(a["url"])}">{name}</a>' if a.get("url") else name
            names.append(f'{name}<sup>{sup}{mark}</sup>' if (sup or mark) else name)
        out.append('<p class="authors">' + ", ".join(names) + "</p>")
        if aff:
            out.append('<p class="affils">' + " &nbsp; ".join(
                f"<sup>{i}</sup>{html.escape(x)}" for i, x in enumerate(aff, 1)) + "</p>")
        if project.get("note"):
            out.append(f'<p class="affils">{html.escape(project["note"])}</p>')
    links = [(k, v) for k, v in project.get("links", {}).items() if v]
    if links:
        out.append('<div class="badges">' + "".join(
            f'<a class="badge link" href="{html.escape(v)}">{html.escape(k)}</a>' for k, v in links) + "</div>")
    return "\n".join(out)


def page(snap, rows, tiers, project=None):
    people_html = people(project or {})
    figs = {f[0]: f for f in FIGS}
    learners = [r for r in rows if not r['ref']]
    oracle = [r['avg_h'] for r in rows if r['harness'] == 'oracle']
    blind = [r['avg_h'] for r in rows if r['harness'] == 'blind']
    best = max(learners, key=lambda r: r['avg_h'])
    import statistics
    ci = snap['columns'].index('Retail H L3')
    rl3 = [r['values'][ci] for r in learners]
    rl3_best, rl3_med = max(rl3), statistics.median_low(rl3)
    ratio = [r['cost'] / next(o['cost'] for o in rows if o['model'] == r['model'] and o['harness'] == 'oracle') for r in learners]
    tot_serv = sum(int(t['serving']) for t in tiers)
    tot_test = sum(int(t['test']) for t in tiers)
    tot_win = sum(int(t['stages']) for t in tiers)
    data = dict(models=[dict(key=m['key'], name=SHORT.get(m['name'], m['name'])) for m in snap['models']], columns=snap['columns'], rows=rows)
    tier_rows = ''.join(f"<tr><td class='l'>{t['domain']} {t['tier']}</td><td class='l'>{html.escape(t['latent_structure'])}</td>"
                        f"<td>{t['stages']}</td><td>{int(t['serving']):,}</td><td>{int(t['test']):,}</td></tr>" for t in tiers)
    harness_rows = ''.join(f"<tr><td class='l'><b>{n}</b></td><td class='l' style='white-space:normal'>{html.escape(d)}</td></tr>"
                           for n, d in HARNESS_INFO)
    appendix_figs = ''.join(fig_html(k, figs) for k in ('drop', 'expdomain', 'expcells', 'phase', 'learncost'))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ServeLearnBench</title>
<meta name="description" content="ServeLearnBench: how well can agents self-improve from serving experience? Benchmark, leaderboard and analyses.">
<style>{CSS}</style></head>
<body>
<nav><div class="wrap"><span class="brand">ServeLearnBench</span>
<a class="l" href="#leaderboard">Leaderboard</a><a class="l" href="#findings">Findings</a><a class="l" href="#results">Results</a>
<a class="l" href="#analysis">Analysis</a><a class="l" href="#domains">Domains</a><a class="l" href="#protocol">Protocol</a>
<span class="sp"></span><button id="theme" title="Toggle theme">◐</button></div></nav>

<header class="hero"><div class="wrap">
<div class="kicker">Benchmark · Continual learning from serving data</div>
<h1>ServeLearnBench</h1>
<p class="sub">How well can agents self-improve from serving experience?</p>
{people_html}
<p class="lede" style="margin-top:22px">The knowledge an agent needs in deployment is often implicit, undisclosed, and subject to change. Continual-learning
harnesses aim to let agents improve from serving experience, but how well they do so is not well characterized.
ServeLearnBench formalizes the setting as an <b>evolving-environment streaming dataset</b> (EESD): a chronological task stream in which
agents must infer, apply, and revise latent environment knowledge from interaction and outcome feedback as hidden policies evolve.
It spans retail support, banking, and sales-pitch generation with 53 environment windows and 7,718 tasks, and we evaluate five learning
harnesses across six models: 28 model&ndash;harness pairs and 252 learning runs.</p>
<div class="stats">
<div class="stat"><b>3</b><span>domains</span></div><div class="stat"><b>9</b><span>scenarios (3 tiers each)</span></div>
<div class="stat"><b>{tot_win}</b><span>environment windows</span></div><div class="stat"><b>{tot_serv:,}</b><span>serving tasks</span></div>
<div class="stat"><b>{tot_test:,}</b><span>held-out test tasks</span></div><div class="stat"><b>28</b><span>model&ndash;harness pairs</span></div>
<div class="stat"><b>252</b><span>learning runs</span></div>
</div></div></header>

<main class="wrap">
<section id="leaderboard"><h2>Leaderboard</h2>
<p class="secsub"><b>Avg. H</b> is the mean Hidden-Dependent score over the nine settings; <b>Avg. F</b> the mean Fully Specified score over the six
Retail and Banking settings (Pitch has none). Each setting counts once. Retail and Banking scores are accuracy × 100, Pitch the 0–100 rubric
score. <b>API cost</b> is the total over the nine settings (serving stream, learning work and held-out test) at official list prices.
Click a column to sort; "Per setting" shows all fifteen scores.</p>
<div class="card"><div class="controls" id="lb-controls"></div><div class="tablewrap"><table id="lb"></table></div>
<div class="foot">Snapshot {html.escape(snap.get('generated', '')[:10])}. Prime was not run with Opus 5 and GPT-5.6 Terra because of its cost.
Blind and Oracle are reference settings, not learners. One run per cell; no confidence intervals are implied.</div></div>
</section>

<section id="findings"><h2>Key findings</h2><p class="secsub">From the evaluation of five learning harnesses across six models.</p>
<div class="grid3">
<div class="card finding"><div class="n">FINDING 1</div><h3>Capability is not learning</h3>
<span class="big">{sum(oracle)/len(oracle):.1f} vs {sum(blind)/len(blind):.1f}</span><p>Oracle vs. Blind mean Hidden score across the six models: every model can perform the tasks once the policy is disclosed.
Yet no learner recovers it reliably: the best pair ({html.escape(best['model_name'])} + {best['harness_name']}) reaches {best['avg_h']:.1f}, and on Retail L3 the best of {len(learners)} pairs reaches only {rl3_best:.1f} (median {rl3_med:.1f}).</p></div>
<div class="card finding"><div class="n">FINDING 2</div><h3>Adaptation is costly</h3>
<span class="big">{min(ratio):.0f}–{max(ratio):.0f}×</span><p>Inferring a hidden policy costs {min(ratio):.0f} to {max(ratio):.0f} times as much as following a disclosed one (Kimi K3: Prime $2,479 vs. Oracle $102).
Adaptation can also degrade behavior that needed no learning: averaged over the four open-weight models, every learner loses Fully Specified accuracy against Blind in Retail (98.9 vs. 89.5–95.5).</p></div>
<div class="card finding"><div class="n">FINDING 3</div><h3>Exploration is a bottleneck</h3>
<span class="big">ρ = 1.00</span><p>A harness's answer diversity while it is still failing ranks the harnesses exactly as their Hidden reward does
(Spearman ρ over five harnesses; 0.75 over 30 harness–domain–tier cells). Prime, CH, RAG, Mem0 and SkillOpt explore in that order (AUC 1.16, 1.02, 0.88, 0.65, 0.51).</p></div>
</div></section>

<section id="results"><h2>Results</h2><p class="secsub">Performance, cost, and the cost–performance frontier. Click a figure to enlarge.</p>
{fig_html('frontier', figs)}
<div class="grid2" style="margin-top:16px">{fig_html('avg', figs)}{fig_html('costmerged', figs)}</div>
</section>

<section id="analysis"><h2>Analysis</h2><p class="secsub">How scores come about: how fast a harness picks up a changed policy, what adaptation does to cases that
needed no learning, and whether the way a harness explores and retains predicts its reward. The analyses use the recorded serving streams of
the four open-weight models and run no new inference.</p>
{fig_html('speed', figs)}
<div class="grid2" style="margin-top:16px">{fig_html('fs', figs)}{fig_html('ka', figs)}</div>
<div style="margin-top:16px">{fig_html('explore', figs)}</div>
<details><summary>More analyses (appendix figures)</summary><div class="grid2" style="margin-top:12px">{appendix_figs}</div></details>
</section>

<section id="domains"><h2>Domains and tiers</h2><p class="secsub">Three domains crossed with three difficulty tiers give nine scenarios. Serving and test
tasks of a window cover the same active policy components with disjoint instances.</p>
<div class="grid3">
<div class="card dom"><h3>Retail</h3><p>Multi-turn customer support over a mutable order database: cancel or modify pending orders, return or exchange
delivered items. Correct only if the final database state is the intended one, or the request is refused with the right reason code.</p>
<span class="pill">tool use</span><span class="pill">categorical rules</span><span class="pill">refusal codes</span></div>
<div class="card dom"><h3>Banking</h3><p>Case review: authorize card transactions, review credit-limit increases, screen outbound transfers. Harder tiers
compose decisions into ranked alternatives and payment batches. Hidden policies are numeric thresholds, caps and corridors.</p>
<span class="pill">decision making</span><span class="pill">numeric thresholds</span><span class="pill">composition</span></div>
<div class="card dom"><h3>Pitch</h3><p>Single-turn generation: write a 40–80 word sales pitch from a factual product sheet. The hidden state is the
active customer's taste; a rubric rewards praising liked attributes and penalizes disliked ones, contradictions and invented claims.</p>
<span class="pill">open-ended generation</span><span class="pill">latent preferences</span><span class="pill">LLM judge</span></div>
</div>
<div style="margin-top:16px;display:grid;grid-template-columns:minmax(0,1fr);gap:16px">
<div class="card"><div class="tablewrap"><table><thead><tr><th class="l">Scenario</th><th class="l">Latent structure</th><th>Windows</th><th>Serving</th><th>Test</th></tr></thead>
<tbody>{tier_rows}<tr><td class="l"><b>Total</b></td><td></td><td><b>{tot_win}</b></td><td><b>{tot_serv:,}</b></td><td><b>{tot_test:,}</b></td></tr></tbody></table></div></div>
<div class="card proto"><h3>Difficulty tiers</h3>
<p><b>L1 · acquisition.</b> Previously unseen states, mostly single-policy Retail and Banking requests, non-repeating Pitch profiles.</p>
<p><b>L2 · revision and composition.</b> Earlier states are revised; multi-policy Retail requests, ranked or batched Banking decisions, recurring Pitch profiles over more attributes.</p>
<p><b>L3 · non-monotonic evolution.</b> Reversals, removals and re-entry of earlier states on top of L2's composition; Pitch cycles back to earlier profiles.</p></div>
</div></section>

<section id="protocol"><h2>Protocol</h2>
<div style="margin-bottom:16px">{fig_html('overview', figs)}</div>
<div class="grid2">
<div class="card proto"><h3>Serving and test</h3>
<p>Each environment window places <b>serving</b> tasks, which return a score after the agent acts, before <b>held-out test</b> tasks, which return
nothing and never update the learner. The hidden state is fixed within a window and may change only at its boundary.</p>
<h3>Hidden-Dependent and Fully Specified</h3>
<p>A task is <b>Hidden-Dependent</b> if the visible information alone cannot determine the correct behavior, and <b>Fully Specified</b> otherwise.
The goal is to raise Hidden scores without losing Fully Specified ones.</p>
<h3>What the learner sees</h3>
<p>The task, its own trajectory or response, a scalar score (<code>Score: n/100</code>), a monotonic stream timestamp, and its own persistent state.
Never: the hidden policy, corrections, slice labels, window boundaries or evaluator explanations. Every task is attempted once, within 50 tool calls and 100 turns.</p></div>
<div class="card proto"><h3>Harnesses</h3><div class="tablewrap"><table><tbody>{harness_rows}</tbody></table></div>
<h3>Aggregation</h3>
<p>Each model–harness–setting cell yields one score; averages across settings weight every setting equally. Costs are totals over tasks.</p></div>
</div></section>
</main>

<footer><div class="wrap">ServeLearnBench · results snapshot {html.escape(snap.get('generated', '')[:10])} · page built {date.today().isoformat()}.
Numbers on this page are generated from the same pinned data as the paper's tables and figures.</div></footer>
<div id="zoom"><img alt=""></div>
<script>window.__DATA__={json.dumps(data, separators=(',', ':'))};</script>
<script>{JS}</script>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--refresh-from', type=Path, default=None,
                    help='copy data files and render figures from these paper sources first')
    a = ap.parse_args()
    if a.refresh_from:
        src = a.refresh_from
        (HERE / 'data').mkdir(exist_ok=True)
        (HERE / 'assets/fig').mkdir(parents=True, exist_ok=True)
        for f in ('main_results_v2.json', 'cost_total.json', 'tiers.csv'):
            shutil.copy2(src / 'data' / f, HERE / 'data' / f)
        for key, pdf, *_ in FIGS:
            render_pdf(src / pdf, HERE / 'assets/fig' / f'{key}.png')
    snap, cost, tiers = load(HERE / "..", HERE)
    rows = rows_for_page(snap, cost)
    pj = HERE / 'project.json'
    project = json.loads(pj.read_text()) if pj.exists() else {}
    (HERE / 'index.html').write_text(page(snap, rows, tiers, project))
    print(f'wrote website/index.html: {len(rows)} leaderboard rows, {len(FIGS)} figures')


if __name__ == '__main__':
    main()
