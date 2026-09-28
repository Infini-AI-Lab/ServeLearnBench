"""From result rows to the numbers reported in the paper.

Each task belongs to one of two categories. A task is Hidden-Dependent
("hidden") if the visible information cannot determine the correct behavior,
and Fully Specified ("fully_specified") otherwise. In Retail and Banking a
task's slice `adapt` is Hidden; `general` and `reopen` (a rule that has
returned to its documented behavior) are Fully Specified. Every Pitch task is
Hidden: the customer's taste is never documented.

A setting score is the mean reward x 100 over a scenario's held-out test
tasks of one category. Averages across settings weight every setting
equally: Avg. H over the nine Hidden scores, Avg. F over the six Retail and
Banking Fully Specified scores.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Iterable

SCENARIOS = [f"{d}_l{t}" for d in ("retail", "banking", "pitch") for t in (1, 2, 3)]


def category(domain: str, slice_: str | None) -> str:
    if domain == "pitch":
        return "hidden"
    return "hidden" if slice_ == "adapt" else "fully_specified"


def read_rows(path: str) -> tuple[dict, list[dict]]:
    config, rows = {}, []
    for line in open(path):
        if not line.strip():
            continue
        d = json.loads(line)
        if "config" in d:
            config = d["config"]
        else:
            rows.append(d)
    return config, rows


def setting_scores(rows: Iterable[dict]) -> dict:
    """{'hidden': score, 'fully_specified': score or None, 'n_test': n}."""
    acc = defaultdict(list)
    n = 0
    for r in rows:
        if r.get("phase") != "test":
            continue
        n += 1
        acc[r["category"]].append(r["reward"])
    mean = lambda xs: 100 * sum(xs) / len(xs) if xs else None  # noqa: E731
    return {"hidden": mean(acc["hidden"]), "fully_specified": mean(acc["fully_specified"]),
            "n_test": n}


def summarize(paths: Iterable[str]) -> dict:
    """{(model, method): {scenario: scores, 'avg_h': ..., 'avg_f': ...}}.
    Averages are reported only when every constituent setting is present."""
    table: dict = defaultdict(dict)
    for p in paths:
        cfg, rows = read_rows(p)
        key = (cfg.get("model", "?"), cfg.get("method", "?"))
        table[key][cfg.get("scenario", "?")] = setting_scores(rows)
    for key, cells in table.items():
        hs = [cells[s]["hidden"] for s in SCENARIOS if s in cells]
        fs = [cells[s]["fully_specified"] for s in SCENARIOS
              if s in cells and not s.startswith("pitch")]
        cells["avg_h"] = sum(hs) / 9 if len(hs) == 9 else None
        cells["avg_f"] = sum(fs) / 6 if len(fs) == 6 else None
    return dict(table)
