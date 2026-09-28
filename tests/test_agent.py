"""The agent loop's handling of valid and invalid turns, with a scripted model."""

import json
from types import SimpleNamespace

from servelearnbench import agent


class Scripted:
    """Stands in for llm.LLM: returns the queued turns in order."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []

    def complete(self, messages, tools=None, **_):
        self.requests.append([dict(m) for m in messages])
        return self.turns.pop(0)


def call(name, args, cid="c1"):
    return {"id": cid, "name": name, "arguments": json.dumps(args)}


def turn(*calls, content="", finish_reason="tool_calls"):
    return SimpleNamespace(content=content, reasoning="thinking", tool_calls=list(calls),
                           finish_reason=finish_reason)


def run(turns, domain="retail"):
    model = Scripted(turns)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    executed, finished = [], []

    def execute(name, args):
        if name != "lookup":
            raise agent.UnknownTool(agent.MSG_UNKNOWN.format(name=name))
        executed.append(args)
        return "ok"

    trace = agent.run_turns(model, messages, agent.tool_specs([], domain), execute,
                            on_finish=finished.append)
    return trace, messages, executed, finished


def lookup_spec():
    return {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}


def test_tool_then_finish():
    trace, messages, executed, finished = run([turn(call("lookup", {"q": 1})),
                                               turn(call("finish", {}, "c2"))])
    assert executed == [{"q": 1}] and finished == [{}]
    assert [s["kind"] for s in trace] == ["tool", "finish"]
    assert messages[-1] == {"role": "tool", "tool_call_id": "c2", "content": agent.FINISH_ACK}


def test_two_calls_in_one_turn_execute_nothing():
    trace, messages, executed, _ = run([turn(call("lookup", {}, "a"), call("lookup", {}, "b")),
                                        turn(call("finish", {}, "c"))])
    assert executed == []
    assert trace[0]["invalid_reason"] == "multi_action"
    errors = [m for m in messages if m.get("role") == "tool" and m["tool_call_id"] in ("a", "b")]
    assert len(errors) == 2


def test_text_only_turn_gets_a_corrective_message():
    trace, messages, _, _ = run([turn(content="I think..."), turn(call("finish", {}))])
    assert trace[0]["invalid_reason"] == "no_action"
    assert {"role": "user", "content": agent.MSG_NO_ACTION} in messages


def test_invalid_arguments_and_unknown_tool():
    bad = {"id": "x", "name": "lookup", "arguments": "{not json"}
    trace, _, executed, _ = run([turn(bad), turn(call("nope", {}, "y")), turn(call("finish", {}, "z"))])
    assert [s.get("invalid_reason") for s in trace[:2]] == ["schema_invalid", "unknown_tool"]
    assert executed == []


def test_finish_payload_is_validated():
    trace, _, _, finished = run([turn(call("finish", {"decision": "refuse"})),
                                 turn(call("finish", {"decision": "refuse", "reason": "not_found"}, "c2"))])
    assert trace[0].get("rejected") and finished == [{"decision": "refuse", "reason": "not_found"}]


def test_pitch_finish_requires_a_pitch():
    trace, _, _, finished = run([turn(call("finish", {})), turn(call("finish", {"pitch": "Buy it."}, "c2"))],
                                domain="pitch")
    assert trace[0].get("rejected") and finished == [{"pitch": "Buy it."}]
