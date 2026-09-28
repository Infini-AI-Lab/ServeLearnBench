"""Content hashes of a scenario, used to check that the generated tasks, world
and prompts are exactly the ones evaluated in the paper."""

from __future__ import annotations

import hashlib
import json

from . import agent
from .scenarios import get_bundle, n_windows


def task_dict(t) -> dict:
    return {"task_id": t.task_id, "instruction": t.instruction,
            "actions": [{"name": a.name, "kwargs": a.kwargs} for a in t.actions],
            "answer": t.answer, "z": t.z}


def _h(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


def fingerprint(name: str) -> dict:
    b = get_bundle(name, fresh=True)
    rules = b["rules"]
    nw = n_windows(b)
    tests = {w: [task_dict(t) for t in b["test_set"](w)] for w in range(nw)}
    return {
        "n_windows": nw,
        "n_serving": len(b["serving"]),
        "n_test": sum(len(v) for v in tests.values()),
        "serving": _h([task_dict(t) for t in b["serving"]]),
        "test": _h(tests),
        "world": _h(b["load_data"]()),
        "documents": _h([b["policy_md"], b["workflow_md"], b["docs_toc"]]),
        "system_prompt": _h(b["meta"]["system_prompt"]),
        "tools": _h(agent.tool_specs([t.get_info() for t in b["tools"]], b["meta"]["domain"])),
        "oracle_bulletins": _h([rules.oracle_note(rules.WINDOW_STARTS[w]) for w in range(nw)]),
    }
