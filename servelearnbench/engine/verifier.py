"""Reward computation, decoupled from Env.

reward = r_state AND r_answer   (binary, pure rule-based, zero LLM-judge, outcome-only)

r_state: replay the GT recipe from a CLEAN DB, hash the terminal state, and compare
to the agent's terminal DB. GT is the terminal state the recipe produces, implicitly
coupled to the current tools and DB -- that coupling keeps the environment a
consistent bundle across drift stages. We do NOT check how the agent got there.

r_answer: the agent must SUBMIT a structured final answer via a `finish` action.
  - task.answer is None            -> state-only task; always passes (cancel/return).
  - {"kind":"value","expect",...}  -> typed/canonical compare of the submitted answer.
  - {"kind":"refuse"}              -> the submission must be decision=="refuse".
The submitted answer is read from a dedicated field, not scanned from free text, so
there are no accidental substring hits. Comparison is canonicalized (not raw exact)
to avoid penalizing correct answers that differ only in formatting.

Kept as its own class (not inside Env) so domains can supply their own
verifier (e.g. the pitch rubric verifier).
"""

from __future__ import annotations

import re
from hashlib import sha256
from typing import Any, Callable, Dict, List, Optional, Type

from .tool import Tool
from .types import (
    FINISH_ACTION_NAME,
    RESPOND_ACTION_NAME,
    Action,
    RewardResult,
    Task,
)


def to_hashable(item: Any) -> Any:
    """dict/set -> order-independent; list -> order-sensitive."""
    if isinstance(item, dict):
        return tuple((k, to_hashable(v)) for k, v in sorted(item.items()))
    if isinstance(item, list):
        return tuple(to_hashable(e) for e in item)
    if isinstance(item, set):
        return tuple(sorted(to_hashable(e) for e in item))
    return item


def hash_db(data: Dict[str, Any]) -> str:
    return sha256(str(to_hashable(data)).encode("utf-8")).hexdigest()


def _norm(s: Any) -> str:
    """Canonical string form: lowercase, alphanumeric only."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _num_tokens(s: Any) -> List[str]:
    return re.findall(r"-?\d+(?:\.\d+)?", str(s).replace(",", "").replace("$", ""))


def _single_number(s: Any) -> Optional[float]:
    """The submission's numeric content — REQUIRING exactly one number.

    Formatting tolerance (currency symbols, thousands separators, surrounding
    words) is kept; a second numeric token is not — the answer must commit to
    ONE number (e.g. "$12.70 or $99" yields None).
    """
    toks = _num_tokens(s)
    return float(toks[0]) if len(toks) == 1 else None


def compare_answer(got: Any, expect: Any, mode: str) -> bool:
    """Canonical comparison -- format-tolerant, but single-valued and typed."""
    if got is None:
        return False
    if mode == "int":
        gv, ev = _single_number(got), _single_number(expect)
        # fractional submissions must not round down onto the answer ("2.9" != 2)
        return (gv is not None and ev is not None
                and gv.is_integer() and gv == ev)
    if mode == "float":
        gv, ev = _single_number(got), _single_number(expect)
        return gv is not None and ev is not None and abs(gv - ev) < 0.01
    if mode == "str_norm":
        # exact normalized equality, not containment ("not eligible" must
        # not match "eligible")
        ng, ne = _norm(got), _norm(expect)
        return ne != "" and ne == ng
    if mode == "set":
        try:
            return {_norm(x) for x in got} == {_norm(x) for x in expect}
        except TypeError:
            return False
    # unknown mode -> strict normalized equality
    return _norm(got) == _norm(expect)


class EvalVerifier:
    """Clean, strict, outcome-only scoring."""

    def replay_gt(
        self,
        task: Task,
        load_data: Callable[[], Dict[str, Any]],
        tools_map: Dict[str, Type[Tool]],
    ) -> str:
        """Reload a clean DB, replay the tool part of the GT recipe, return its hash.

        Non-tool actions (respond, finish) never touch the DB and are skipped, so
        the GT terminal state depends only on the tool calls.
        """
        from .env import apply_episode_view
        gt_data = load_data()
        apply_episode_view(gt_data, task)  # same view as the agent episode
        for action in task.actions:
            if action.name in (RESPOND_ACTION_NAME, FINISH_ACTION_NAME):
                continue
            tool = tools_map.get(action.name)
            # Fail closed: a GT recipe that references a missing tool or errors is
            # corrupt DATA, not agent behavior — computing a hash off a partial
            # replay would silently verify against the wrong target state.
            # Explicit raises (not assert): `python -O` strips asserts, which would
            # quietly turn fail-closed back into fail-open.
            if tool is None:
                raise RuntimeError(f"GT action references unknown tool: {action.name}")
            obs = tool.invoke(data=gt_data, **action.kwargs)
            if isinstance(obs, str) and obs.startswith("Error"):
                raise RuntimeError(
                    f"GT action {action.name} errored during replay: {obs[:80]}")
        return hash_db(gt_data)

    def submitted_answer(self, trajectory: List[Action]) -> Optional[Dict[str, Any]]:
        """The payload of the single `finish` action.

        The protocol requires EXACTLY ONE terminal finish. Runners stop at the
        first finish, so this matters only for direct Env use: zero or multiple
        finishes -> no answer.
        """
        finishes = [a.kwargs for a in trajectory if a.name == FINISH_ACTION_NAME]
        if len(finishes) != 1:
            return None
        return finishes[0]

    @staticmethod
    def _declared_refusal(finish: Dict[str, Any]) -> bool:
        # normalized, so "Refuse" and "refused" also count as a refusal
        return str(finish.get("decision", "")).strip().lower().startswith("refus")

    def check_answer(self, task: Task, trajectory: List[Action]):
        spec = task.answer
        if not spec:
            # Action task: the work is judged by DB state, but the protocol still
            # requires a terminal finish, and the DECLARATION must be consistent
            # with what was done — carrying the request out and then declaring a
            # refusal contradicts the carry-out-vs-refuse judging basis.
            finish = self.submitted_answer(trajectory)
            if finish is None:
                return False, {"kind": "none", "reason": "no finish submitted"}
            if self._declared_refusal(finish):
                return False, {"kind": "none",
                               "reason": "declared refusal on a carried-out request"}
            return True, {"kind": "none"}
        finish = self.submitted_answer(trajectory)
        if finish is None:
            return False, {"kind": spec.get("kind"), "reason": "no finish submitted"}
        if spec["kind"] == "refuse":
            ok = self._declared_refusal(finish)
            accepted = spec.get("reasons")
            if ok and accepted:
                # the refusal must cite a correct reason code (any of the accepted set)
                ok = finish.get("reason") in accepted
            return ok, {"kind": "refuse", "submitted": finish.get("decision"),
                        "reason": finish.get("reason"), "accepted": accepted}
        if spec["kind"] == "value":
            # a refusal declaration and a value answer are mutually exclusive:
            # {"decision":"refuse","answer":X} must not score as X — the
            # customer was told "no"
            if self._declared_refusal(finish):
                return False, {"kind": "value", "expect": spec["expect"],
                               "reason": "declared refusal on a value question"}
            got = finish.get("answer")
            ok = compare_answer(got, spec["expect"], spec.get("compare", "str_norm"))
            return ok, {"kind": "value", "expect": spec["expect"], "got": got, "compare": spec.get("compare")}
        return False, {"kind": spec.get("kind"), "reason": "unknown spec"}

    def compute(
        self,
        agent_data: Dict[str, Any],
        task: Task,
        load_data: Callable[[], Dict[str, Any]],
        tools_map: Dict[str, Type[Tool]],
        trajectory: List[Action],
    ) -> RewardResult:
        agent_hash = hash_db(agent_data)
        gt_hash = self.replay_gt(task, load_data, tools_map)
        r_state = agent_hash == gt_hash

        r_answer, detail = self.check_answer(task, trajectory)

        reward = 1.0 if (r_state and r_answer) else 0.0
        return RewardResult(
            reward=reward,
            r_state=r_state,
            r_answer=r_answer,
            agent_data_hash=agent_hash,
            gt_data_hash=gt_hash,
            answer_detail=detail,
        )
