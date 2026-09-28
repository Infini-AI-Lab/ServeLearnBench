"""The runner end to end, offline: a fake model that submits `finish`
immediately, on one window of Banking L1."""

import json
from types import SimpleNamespace

from servelearnbench.llm import ModelConfig
from servelearnbench.methods import RAG, Blind, Oracle
from servelearnbench.runner import Runner
from servelearnbench.scenarios import get_bundle
from servelearnbench.scoring import read_rows


class FinishAtOnce:
    def __init__(self):
        self.system_prompts = []

    def complete(self, messages, tools=None, **_):
        self.system_prompts.append(messages[0]["content"])
        return SimpleNamespace(content="", reasoning=None, finish_reason="tool_calls",
                               tool_calls=[{"id": "c", "name": "finish", "arguments": "{}"}])


CFG = ModelConfig(model="fake")


def counts(path):
    _, rows = read_rows(path)
    return sum(r["phase"] == "serving" for r in rows), sum(r["phase"] == "test" for r in rows)


def window0(name="banking_l1"):
    b = get_bundle(name)
    return sum(t.z["window"] == 0 for t in b["serving"]), len(b["test_set"](0))


def test_blind_and_oracle_run_tests_only(tmp_path):
    n_serving, n_test = window0()
    for method in (Blind(), Oracle()):
        model = FinishAtOnce()
        out = tmp_path / f"{method.name}.jsonl"
        Runner("banking_l1", method, CFG, str(out), workers=4, windows=[0], log=lambda *_: None,
               model=model).run()
        assert counts(out) == (0, n_test)
    assert "CURRENT ACTUAL POLICY" in model.system_prompts[0]


def test_rag_serves_in_blocks_and_resumes(tmp_path):
    n_serving, n_test = window0()
    out = tmp_path / "rag.jsonl"
    model = FinishAtOnce()
    Runner("banking_l1", RAG(), CFG, str(out), workers=4, windows=[0], log=lambda *_: None,
           model=model).run()
    assert counts(out) == (n_serving, n_test)
    assert any("# Past cases" in p for p in model.system_prompts)   # later blocks retrieve

    # interrupt: drop the last 20 lines (tests and part of the last serving block), then resume
    lines = open(out).read().splitlines()
    open(out, "w").write("\n".join(lines[:-(n_test + 5)]) + "\n")
    Runner("banking_l1", RAG(), CFG, str(out), workers=4, windows=[0], resume=True,
           log=lambda *_: None, model=FinishAtOnce()).run()
    assert counts(out) == (n_serving, n_test)
    ids = [json.loads(line).get("task_id") for line in open(out)][1:]
    assert len(ids) == len(set(ids))
