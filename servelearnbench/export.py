"""Write the dataset files: every scenario's task stream and environment.

Layout (one directory per scenario):

    <out>/data/<scenario>/serving.jsonl   the serving stream, in order
    <out>/data/<scenario>/test.jsonl      the held-out test tasks of every window
    <out>/environments/<scenario>/
        world.json          the initial database (empty for Pitch)
        policy.md           the retrievable policy document (Retail, Banking)
        workflow.md         the retrievable workflow document (Retail, Banking)
        system_prompt.txt   the acting agent's system prompt
        tools.json          tool schemas, `finish` included
        windows.json        per window: first stream position and the Oracle bulletin
    <out>/fingerprints.json content hashes, identical to `slb verify`

Instructions are written exactly as the agent receives them, timestamp line
included. Fields under `evaluator` are for scoring only and are never shown
to an agent.
"""

from __future__ import annotations

import json
import os

from . import agent, protocol
from .fingerprint import fingerprint
from .scenarios import SCENARIOS, get_bundle, n_windows
from .scoring import category

_EVAL_ONLY_Z = {"window", "split", "slice", "family", "tier", "stream_pos"}


def _row(name, domain, task, split, window, timestamp):
    z = dict(task.z or {})
    slice_ = z.get("slice")
    row = {"task_id": task.task_id, "scenario": name, "domain": domain,
           "tier": name[-2:].upper(), "split": split, "window": window,
           "timestamp": timestamp, "category": category(domain, slice_),
           "instruction": task.instruction,
           "evaluator": {"slice": slice_, "family": z.get("family"),
                         "actions": [{"name": a.name, "kwargs": a.kwargs} for a in task.actions],
                         "answer": task.answer,
                         "metadata": {k: v for k, v in z.items() if k not in _EVAL_ONLY_Z}}}
    return row


def _dump_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def export(out: str, scenarios=None) -> None:
    scenarios = scenarios or list(SCENARIOS)
    prints = {}
    for name in scenarios:
        b = get_bundle(name, fresh=True)
        domain = b["meta"]["domain"]
        rules = b["rules"]
        serving = list(b["serving"])
        protocol.stamp_serving(serving)
        d_data = os.path.join(out, "data", name)
        d_env = os.path.join(out, "environments", name)
        os.makedirs(d_data, exist_ok=True)
        os.makedirs(d_env, exist_ok=True)
        _dump_jsonl(os.path.join(d_data, "serving.jsonl"),
                    [_row(name, domain, t, "serving", t.z["window"], t.z["stream_pos"])
                     for t in serving])
        tests = []
        for w in range(n_windows(b)):
            ts = b["test_set"](w)
            stamp = protocol.stamp_test(ts, b["serving"], w)
            tests += [_row(name, domain, t, "test", w, stamp) for t in ts]
        _dump_jsonl(os.path.join(d_data, "test.jsonl"), tests)

        with open(os.path.join(d_env, "world.json"), "w") as f:
            json.dump(b["load_data"](), f, ensure_ascii=False, default=str)
        if b["policy_md"]:
            open(os.path.join(d_env, "policy.md"), "w").write(b["policy_md"])
            open(os.path.join(d_env, "workflow.md"), "w").write(b["workflow_md"])
        open(os.path.join(d_env, "system_prompt.txt"), "w").write(b["meta"]["system_prompt"])
        json.dump(agent.tool_specs([t.get_info() for t in b["tools"]], domain),
                  open(os.path.join(d_env, "tools.json"), "w"), indent=1)
        json.dump([{"window": w, "first_position": protocol.window_end(b["serving"], w - 1) + 1
                    if w else 1, "oracle_bulletin": rules.oracle_note(rules.WINDOW_STARTS[w])}
                   for w in range(n_windows(b))],
                  open(os.path.join(d_env, "windows.json"), "w"), indent=1, ensure_ascii=False)
        prints[name] = fingerprint(name)
        print(f"{name}: {len(serving)} serving, {len(tests)} test", flush=True)
    json.dump(prints, open(os.path.join(out, "fingerprints.json"), "w"), indent=1)
