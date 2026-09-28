"""The evaluation protocol: budgets, the outcome line, the episode record,
and the stream timestamp.

Budgets are per episode (one agent rollout on one task). Every task is
attempted once.
"""

from __future__ import annotations

import json

ENGINE_TOOL_BUDGET = 50      # tool calls per episode; `finish` is exempt
MAX_STEPS = 100              # model turns per episode (runaway guard above the tool budget)
MAX_OUTPUT_PER_TASK = 256_000  # output tokens per episode
MAX_TOKENS = 65_536          # max_tokens per model call

# ---------------------------------------------------------------------------
# Outcome feedback. Serving tasks return this line and nothing else: no
# correct answer, no slice label, no window index, no explanation.
# ---------------------------------------------------------------------------
SCORE_LINE = "Score: {score:g}/100."


def score_line(reward: float) -> str:
    """Binary domains render as 0/100 or 100/100; Pitch rewards are continuous."""
    return SCORE_LINE.format(score=reward * 100)


CUT_OFF_REPLY = "(reasoning ran too long and was cut off)"


def split_reply(raw: str, reasoning: str = "") -> tuple[str, str]:
    """(reasoning, visible reply) for one model turn, for providers that
    return reasoning in a separate field and for models that think inline
    between <think> tags."""
    raw = raw or ""
    reasoning = reasoning or ""
    if "</think>" in raw:
        head, _, tail = raw.partition("</think>")
        return (reasoning or head.replace("<think>", "").strip(), tail.strip())
    if "<think>" in raw:
        return (reasoning or raw.replace("<think>", "").strip(), CUT_OFF_REPLY)
    return reasoning.strip(), raw.strip()


def episode_record(instruction: str, trace: list, reward: float) -> str:
    """The text record of one episode that a learning method may read:
    the instruction, every step (what the model received when it was not
    simply the previous observation, its reasoning, its reply, the action,
    the observation), and the score line. Steps that produced no action are
    kept in place."""
    lines = [f"INSTRUCTION: {instruction}", ""]
    prev_obs = None
    for n, s in enumerate(trace, 1):
        kind = s.get("kind")
        lines.append(f"[step {n}]")
        got = s.get("input") or ""
        if got and got != instruction and got != f"Observation: {prev_obs}":
            lines.append(f"  received: {got}")
        thought, reply = split_reply(s.get("raw"), s.get("reasoning"))
        if thought:
            lines.append(f"  thought: {thought}")
        if reply:
            lines.append(f"  replied: {reply}")
        if kind in ("tool", "finish"):
            args = json.dumps(s.get("args") or {}, ensure_ascii=False)
            lines.append(f"  action: {s.get('name')}({args})")
        else:
            lines.append(f"  action: NONE [{kind}: {s.get('name')}]")
        obs = s.get("obs")
        if obs:
            lines.append(f"  observation: {obs}")
        prev_obs = obs
        lines.append("")
    lines.append(score_line(reward))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stream timestamp. Every instruction starts with "Current timestamp: t".
# A serving task carries its 1-based position in the serving stream
# (continuous across windows); a window's test batch carries the position of
# that window's last serving task. Test tasks do not advance the counter, so
# nothing on the time axis marks a window boundary.
# ---------------------------------------------------------------------------
TIMESTAMP_PREFIX = "Current timestamp: "


def _stamp(task, t: int) -> None:
    if not task.instruction.startswith(TIMESTAMP_PREFIX):
        task.instruction = f"{TIMESTAMP_PREFIX}{t}\n\n{task.instruction}"


def stamp_serving(tasks) -> None:
    """Stamp the full serving list, in stream order, with its positions."""
    for i, task in enumerate(tasks, 1):
        task.z["stream_pos"] = i
        _stamp(task, i)


def window_end(serving, w: int) -> int:
    """Stream position of window w's last serving task."""
    n = sum(1 for t in serving if t.z["window"] <= w)
    if n == 0:
        raise ValueError(f"window {w} has no serving tasks at or before it")
    return n


def stamp_test(tasks, serving, w: int) -> int:
    """Stamp a window's test batch with the window-end position."""
    t = window_end(serving, w)
    for task in tasks:
        _stamp(task, t)
    return t


def strip_timestamp(instruction: str) -> str:
    if instruction.startswith(TIMESTAMP_PREFIX):
        return instruction.split("\n\n", 1)[-1]
    return instruction
