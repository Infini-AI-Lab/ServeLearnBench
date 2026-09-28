"""Acting-agent system prompt: a domain-specific head followed by the shared
tool-calling protocol.

Every domain renders its own head (identity, environment note, reference
documents, tool signatures); the protocol section and the one-line `finish`
guidance are shared, so every model and every method reads the same rules.
"""

from __future__ import annotations

PROTOCOL_RULES = """# Protocol
You act ONLY through tool calls. Plain text is not an action.
- Each turn: exactly ONE tool call. Wait for its result, then decide the next call.
- A turn with two or more tool calls is rejected: none of them is executed and each receives an error result. Re-issue them one per turn.
- A turn with no tool call is not an action; you will be prompted to act again.
- Arguments must be a JSON object that follows the tool's schema; a tool reports argument problems in its result.
- The episode ends when you call `finish`, exactly once, as the only call of its turn. Nothing after `finish` is executed."""

FINISH_LINES = {
    "retail": ("Use `finish` to submit: finish(answer=<value>) to answer a "
               "question; finish(decision=\"refuse\", reason=\"<code>\") to "
               "decline the request, citing WHY; finish() with no arguments "
               "after you have completed an action task."),
    "banking": "Call finish() with no arguments after your decision tool call.",
    "pitch": "Submit your pitch with finish(pitch=\"<your 40-80 word pitch>\").",
}


def compose_system_prompt(head: str, domain: str, notes: str = "") -> str:
    """Head, then the shared protocol, the domain's finish line and any
    domain-specific protocol notes."""
    parts = [head.rstrip(), "", PROTOCOL_RULES, FINISH_LINES[domain]]
    if notes:
        parts.append(notes)
    return "\n".join(parts)


def tool_signatures(tools) -> str:
    """One line per tool, `- name(required, [optional])`, from Tool classes."""
    lines = []
    for t in tools:
        fn = t.get_info()["function"]
        required = set(fn["parameters"].get("required", []))
        params = ", ".join(p if p in required else f"[{p}]"
                           for p in fn["parameters"]["properties"])
        lines.append(f"- {fn['name']}({params})")
    return "\n".join(lines)
