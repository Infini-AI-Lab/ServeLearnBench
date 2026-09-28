"""Command-line interface: `slb <command>` (or `python -m servelearnbench`)."""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__


def _model_cfg(a, prefix=""):
    from .llm import ModelConfig
    g = lambda k: getattr(a, prefix + k)  # noqa: E731
    return ModelConfig(model=g("model"), base_url=g("base_url"),
                       api_key=os.environ.get(g("api_key_env")) if g("api_key_env") else None,
                       reasoning_effort=None if g("reasoning_effort") in ("", "none", "default")
                       else g("reasoning_effort"))


def cmd_list(a):
    from .methods import METHODS
    from .scenarios import SCENARIOS
    print("scenarios:", " ".join(SCENARIOS))
    print("methods:  ", " ".join(METHODS))


def cmd_run(a):
    from . import llm
    from .methods import METHODS
    from .runner import run
    scenarios = a.scenario.split(",")
    for sc in scenarios:
        if sc.startswith("pitch"):
            llm.set_judge(_model_cfg(a, "judge_"))
            break
    windows = [int(w) for w in a.windows.split(",")] if a.windows else None
    for sc in scenarios:
        out = a.out if len(scenarios) == 1 and a.out else os.path.join(
            a.out_dir, a.model.split("/")[-1], f"{sc}_{a.method}.jsonl")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        print(f"== {sc} · {a.method} · {a.model} -> {out}", flush=True)
        run(sc, METHODS[a.method](), _model_cfg(a), out, workers=a.workers,
            resume=a.resume, windows=windows)


def cmd_score(a):
    from .scoring import SCENARIOS, summarize
    table = summarize(a.files)
    short = [s.replace("retail", "R").replace("banking", "B").replace("pitch", "P").replace("_l", "") for s in SCENARIOS]
    print(f"{'model':28} {'method':8} " + " ".join(f"{s:>9}" for s in short) + f" {'Avg.H':>6} {'Avg.F':>6}")
    fmt = lambda v: "   -  " if v is None else f"{v:6.1f}"  # noqa: E731
    for (model, method), cells in sorted(table.items()):
        vals = []
        for s in SCENARIOS:
            c = cells.get(s)
            if c is None:
                vals.append(f"{'-':>9}")
            elif s.startswith("pitch"):
                vals.append(f"{fmt(c['hidden']):>9}")
            else:
                vals.append(f"{fmt(c['hidden']).strip():>4}/{fmt(c['fully_specified']).strip():<4}")
        print(f"{model.split('/')[-1]:28} {method:8} " + " ".join(vals)
              + f" {fmt(cells['avg_h'])} {fmt(cells['avg_f'])}")
    if a.json:
        json.dump({f"{k[0]}|{k[1]}": v for k, v in table.items()}, open(a.json, "w"), indent=1)


def cmd_verify(a):
    from pathlib import Path

    from .fingerprint import fingerprint
    pinned = json.loads((Path(__file__).parent / "fingerprints.json").read_text())
    ok = True
    for name in (a.scenario.split(",") if a.scenario else pinned):
        match = fingerprint(name) == pinned[name]
        ok &= match
        print(f"{name:11} {'OK' if match else 'MISMATCH'}", flush=True)
    sys.exit(0 if ok else 1)


def cmd_export(a):
    from .export import export
    export(a.out, scenarios=a.scenario.split(",") if a.scenario else None)


def main(argv=None):
    p = argparse.ArgumentParser(prog="slb", description="ServeLearnBench")
    p.add_argument("--version", action="version", version=f"servelearnbench {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list scenarios and methods").set_defaults(fn=cmd_list)

    r = sub.add_parser("run", help="run a method on one or more scenarios")
    r.add_argument("--scenario", required=True, help="comma-separated, e.g. retail_l1,banking_l2")
    r.add_argument("--method", required=True, choices=["blind", "oracle", "rag"])
    r.add_argument("--model", required=True)
    r.add_argument("--base-url", default="https://api.fireworks.ai/inference/v1")
    r.add_argument("--api-key-env", default="SLB_API_KEY",
                   help="environment variable holding the API key")
    r.add_argument("--reasoning-effort", default="high",
                   help="sent to the provider as reasoning_effort; 'none' to omit")
    r.add_argument("--judge-model", default="accounts/fireworks/models/glm-5p3-flash",
                   help="Pitch judge (the paper uses glm-5p3-flash at high effort)")
    r.add_argument("--judge-base-url", default="https://api.fireworks.ai/inference/v1")
    r.add_argument("--judge-api-key-env", default="SLB_API_KEY")
    r.add_argument("--judge-reasoning-effort", default="high")
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--windows", default="", help="comma-separated window indices (default: all)")
    r.add_argument("--out", default="", help="output file (single scenario)")
    r.add_argument("--out-dir", default="results")
    r.add_argument("--resume", action="store_true")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("score", help="setting scores and averages from result files")
    s.add_argument("files", nargs="+")
    s.add_argument("--json", default="", help="also write the table as JSON")
    s.set_defaults(fn=cmd_score)

    v = sub.add_parser("verify", help="check generated scenarios against the pinned fingerprints")
    v.add_argument("--scenario", default="")
    v.set_defaults(fn=cmd_verify)

    e = sub.add_parser("export-data", help="write the dataset files (tasks, worlds, documents)")
    e.add_argument("--out", required=True)
    e.add_argument("--scenario", default="")
    e.set_defaults(fn=cmd_export)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
