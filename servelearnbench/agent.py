"""The acting agent: a ReAct loop over the provider's native tool-calling API.

Per turn: the model request completes (retries included), then the single
tool call is validated and executed exactly once. Every tool call id receives
exactly one tool-result message; a turn without a call receives a corrective
user message. The episode ends when the model calls `finish`, when it runs out
of turns, or when it exceeds the per-episode output-token budget.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from . import llm as L
from . import protocol
from .engine.env import Env
from .engine.types import FINISH_ACTION_NAME, RESPOND_ACTION_NAME, Action

# Messages the harness sends back (identical for every model).
MSG_NO_ACTION = ("No tool call was received. Act with exactly ONE tool "
                 "call (call `finish` to end the episode).")
MSG_NO_ACTION_LENGTH = ("Your reply was cut off by the output limit before "
                        "a tool call was produced. Act with exactly ONE "
                        "tool call.")
MSG_MULTI = ("not executed: {n} tool calls were received in one turn; "
             "only ONE call per turn is allowed. Nothing was executed — "
             "re-issue the calls one per turn.")
MSG_SCHEMA = ("not executed: arguments must be a JSON object that follows "
              "the tool's schema ({detail}).")
MSG_UNKNOWN = "not executed: unknown tool '{name}'."
FINISH_ACK = '{"accepted": true}'

_TEMPLATE_MARKUP = re.compile(r"</?arg_key>|</?arg_value>|</?tool_call>|<\|[a-z_]+\|>")


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------
def finish_tool_spec(domain: str) -> dict:
    """The `finish` tool; its payload is what the verifier grades."""
    if domain == "pitch":
        params = {"type": "object",
                  "properties": {"pitch": {"type": "string",
                                           "description": "Your 40-80 word pitch."}},
                  "required": ["pitch"]}
        desc = ("Submit your final pitch. This ends the episode: call it "
                "exactly once, as the only call of its turn.")
    elif domain == "banking":
        params = {"type": "object", "properties": {}, "required": []}
        desc = ("End the episode after your decision tool call. Takes no "
                "arguments. Call it exactly once, as the only call of its "
                "turn.")
    else:
        params = {"type": "object",
                  "properties": {
                      "answer": {"description": ("Your final answer to the "
                                                 "customer's question — a "
                                                 "number, a string, or a list "
                                                 "of strings. Omit for action "
                                                 "tasks.")},
                      "decision": {"type": "string", "enum": ["refuse"],
                                   "description": ("Set to \"refuse\" to decline "
                                                   "the request instead of "
                                                   "carrying it out.")},
                      "reason": {"type": "string",
                                 "description": ("The refusal reason code "
                                                 "(with decision=\"refuse\").")}},
                  "required": []}
        desc = ("Submit the outcome and end the episode: finish(answer=...) "
                "for a question, finish(decision=\"refuse\", reason=\"<code>\") "
                "to decline the request, finish() after a completed action "
                "task. Call it exactly once, as the only call of its turn.")
    return {"type": "function",
            "function": {"name": FINISH_ACTION_NAME, "description": desc,
                         "parameters": params}}


def tool_specs(tools_info: list, domain: str) -> list:
    """The `tools=` list: the domain's own schemas plus `finish`."""
    names = [t["function"]["name"] for t in tools_info]
    if FINISH_ACTION_NAME in names or RESPOND_ACTION_NAME in names:
        raise ValueError("a domain tool uses a reserved name")
    return list(tools_info) + [finish_tool_spec(domain)]


def finish_args_error(args: dict, spec: dict) -> str | None:
    """Check a finish call against its own schema. A rejected finish is an
    ordinary tool error and the model may finish again."""
    params = spec["function"]["parameters"]
    props = params.get("properties") or {}
    missing = [k for k in params.get("required") or [] if k not in args]
    if missing:
        return ("not executed: finish requires " +
                ", ".join(f"'{k}' ({props[k].get('type', 'value')})" for k in missing) + ".")
    unknown = sorted(k for k in args if k not in props)
    if unknown:
        return ("not executed: finish takes no argument named " +
                ", ".join(f"'{k}'" for k in unknown) +
                (f"; allowed: {', '.join(props)}." if props else "; it takes no arguments."))
    for k, v in args.items():
        if props[k].get("type") == "string" and (not isinstance(v, str) or not v.strip()):
            return f"not executed: finish argument '{k}' must be a non-empty string."
        if isinstance(v, str) and _TEMPLATE_MARKUP.search(v):
            return (f"not executed: finish argument '{k}' contains tool-call "
                    f"markup ({_TEMPLATE_MARKUP.search(v).group(0)!r}); re-issue "
                    f"the call with plain values, one argument per field.")
        if "enum" in props[k] and v not in props[k]["enum"]:
            return f"not executed: finish argument '{k}' must be one of {props[k]['enum']}."
    if args.get("decision") == "refuse" and not str(args.get("reason") or "").strip():
        return "not executed: finish(decision=\"refuse\") requires 'reason' (the refusal code)."
    return None


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------
def _assistant_message(res) -> dict:
    """The assistant turn as returned, reasoning included, back into history."""
    content = res.content or ""
    calls = res.tool_calls or []
    turn = {"role": "assistant", "content": content if (content or not calls) else None}
    if res.reasoning:
        turn["reasoning_content"] = res.reasoning
    if calls:
        turn["tool_calls"] = [{"id": c["id"], "type": "function",
                               "function": {"name": c["name"],
                                            "arguments": _passback_args(c["arguments"])}}
                              for c in calls]
    return turn


def _passback_args(raw: str) -> str:
    """Arguments as they go back into history. Providers reject a history
    whose tool-call arguments are not a JSON object, so an invalid string is
    wrapped instead of sent verbatim."""
    if not raw or not raw.strip():
        return "{}"
    try:
        obj = json.loads(raw)
    except ValueError:
        obj = None
    return raw if isinstance(obj, dict) else json.dumps({"_invalid_arguments": raw})


class UnknownTool(KeyError):
    pass


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------
def run_turns(model: L.LLM, messages: list, specs: list,
              execute: Callable[[str, dict], str],
              on_finish: Callable[[dict], Any],
              max_steps: int = protocol.MAX_STEPS) -> list:
    """Drive one episode inside `messages` (system + instruction already
    present). Returns the step trace. Raises llm errors that are
    infrastructure failures; a context overflow ends the episode instead."""
    trace: list = []
    turn_input = messages[-1].get("content") or ""
    for _ in range(max_steps):
        if L.agent_usage()["output"] >= protocol.MAX_OUTPUT_PER_TASK:
            trace.append({"kind": "capped", "name": "(token_cap)", "args": {},
                          "obs": f"task exceeded {protocol.MAX_OUTPUT_PER_TASK} output tokens",
                          "input": turn_input, "raw": "", "reasoning": ""})
            break
        u0 = L.agent_usage()
        try:
            res = model.complete(messages, tools=specs)
        except Exception as e:  # noqa: BLE001
            if L.is_context_overflow(e):
                trace.append({"kind": "error", "name": "(context_overflow)", "args": {},
                              "obs": str(e)[:300], "input": turn_input, "raw": "",
                              "reasoning": ""})
                break
            raise L.InfraError(f"{type(e).__name__}: {str(e)[:300]}") from e
        u1 = L.agent_usage()
        calls = list(res.tool_calls or [])
        fr = res.finish_reason
        messages.append(_assistant_message(res))
        step = {"input": turn_input, "raw": res.content or "", "reasoning": res.reasoning or "",
                "tool_calls": calls, "finish_reason": fr,
                "usage": {k: u1[k] - u0[k] for k in u1}}

        if fr == "length" and calls:
            err = MSG_NO_ACTION_LENGTH
            for c in calls:
                messages.append({"role": "tool", "tool_call_id": c["id"],
                                 "content": "not executed: " + err})
            trace.append({"kind": "invalid", "name": "(no_action)",
                          "args": {"calls": [c["name"] for c in calls]},
                          "obs": "not executed: " + err, "invalid_reason": "length", **step})
            turn_input = f"Observation: not executed: {err}"
            continue

        if not calls:
            reason = "length" if fr == "length" else "no_action"
            trace.append({"kind": "invalid", "name": "(no_action)", "args": {}, "obs": "",
                          "invalid_reason": reason, **step})
            turn_input = MSG_NO_ACTION_LENGTH if fr == "length" else MSG_NO_ACTION
            messages.append({"role": "user", "content": turn_input})
            continue

        if len(calls) > 1:
            names = [c["name"] for c in calls]
            reason = "finish_with_tool" if FINISH_ACTION_NAME in names else "multi_action"
            err = MSG_MULTI.format(n=len(calls))
            for c in calls:
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": err})
            trace.append({"kind": "invalid", "name": f"({reason})", "args": {"calls": names},
                          "obs": err, "invalid_reason": reason, **step})
            turn_input = f"Observation: {err}"
            continue

        c = calls[0]
        name, raw_args = c["name"], c["arguments"]
        try:
            args = json.loads(raw_args) if raw_args.strip() else {}
            bad = None if isinstance(args, dict) else f"got {type(args).__name__}, not an object"
        except ValueError as e:
            args, bad = None, f"invalid JSON: {str(e)[:80]}"
        if bad:
            err = MSG_SCHEMA.format(detail=bad)
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": err})
            trace.append({"kind": "invalid", "name": "(schema_invalid)",
                          "args": {"tool": name, "arguments": raw_args[:500]}, "obs": err,
                          "invalid_reason": "schema_invalid", "call_id": c["id"], **step})
            turn_input = f"Observation: {err}"
            continue

        if name == FINISH_ACTION_NAME:
            fspec = next(t for t in specs if t["function"]["name"] == FINISH_ACTION_NAME)
            ferr = finish_args_error(args, fspec)
            if ferr:
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": ferr})
                trace.append({"kind": "tool", "name": FINISH_ACTION_NAME, "args": args,
                              "obs": ferr, "call_id": c["id"], "rejected": True, **step})
                turn_input = f"Observation: {ferr}"
                continue
            on_finish(args)
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": FINISH_ACK})
            trace.append({"kind": "finish", "name": FINISH_ACTION_NAME, "args": args,
                          "obs": "", "call_id": c["id"], **step})
            break

        try:
            obs = execute(name, args)
        except UnknownTool as e:
            err = str(e.args[0]) if e.args else MSG_UNKNOWN.format(name=name)
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": err})
            trace.append({"kind": "invalid", "name": "(unknown_tool)",
                          "args": {"tool": name, "arguments": args}, "obs": err,
                          "invalid_reason": "unknown_tool", "call_id": c["id"], **step})
            turn_input = f"Observation: {err}"
            continue
        messages.append({"role": "tool", "tool_call_id": c["id"], "content": obs})
        trace.append({"kind": "tool", "name": name, "args": args, "obs": obs,
                      "call_id": c["id"], **step})
        turn_input = f"Observation: {obs}"
    else:
        trace.append({"kind": "capped", "name": "(step_cap)", "args": {},
                      "obs": f"episode reached the step cap ({max_steps} turns) without finish",
                      "input": turn_input, "raw": "", "reasoning": ""})
    return trace


def new_env(bundle: dict) -> Env:
    return Env(load_data=bundle["load_data"], tools=bundle["tools"],
               policy=bundle["policy_md"], workflow=bundle["workflow_md"],
               docs_toc=bundle["docs_toc"], verifier=bundle.get("verifier"))


def run_episode(model: L.LLM, bundle: dict, task, extra_system: str = ""):
    """One scored episode of `task`. `extra_system` is appended to the
    domain system prompt (a method's retrieved cases, an Oracle bulletin).
    Returns (RewardResult, trace, agent_usage, judge_usage)."""
    L.usage_reset()
    meta = bundle["meta"]
    system = meta["system_prompt"] + ("\n\n" + extra_system if extra_system else "")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": task.instruction}]
    env = new_env(bundle)
    env.reset(task)
    specs = tool_specs(env.tools_info, meta["domain"])

    def execute(name, args):
        if name not in env.tools_map:
            raise UnknownTool(MSG_UNKNOWN.format(name=name))
        return env.step(Action(name=name, kwargs=args)).observation

    trace = run_turns(model, messages, specs, execute,
                      on_finish=lambda args: env.step(Action(name=FINISH_ACTION_NAME, kwargs=args)))
    reward = env.compute_reward()
    return reward, trace, L.agent_usage(), L.judge_usage()
