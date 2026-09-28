"""Banking L3 tool shim: identical protocol to the shared banking tools, but

- the reason-code whitelist is L3's (adds daily_cap, new_payee_review);
- ORDERED FALLBACK cases: one case may carry several ranked
  alternatives, and `execute_leg` carries out the chosen one;
- refusals on a fallback case normalise their code onto the case's accepted
  set, so any admissible code lands on one graded database state.

The shared module validates against L1's REASONS at import time, so every L3 behaviour change is a subclass
here rather than a widening of the shared code.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from ....engine.tool import Tool
from ..tools import (ApproveCase, DenyCase, EscalateCase, GetAccountDetails,
                     GetCaseDetails, GetMerchantDetails, GetPayeeDetails,
                     _decide, _find_case)
from .rules import REASONS

__all__ = ["ApproveCaseL3", "DecideBatchL3", "DenyCaseL3", "EscalateCaseL3",
           "ExecuteLegL3", "GetAccountDetailsL3", "GetCaseDetailsL3",
           "GetMerchantDetails", "GetPayeeDetailsL3"]


class GetCaseDetailsL3(GetCaseDetails):
    """Same observation as L1 plus `legs` on fallback cases.

    Evaluator-private keys (leading underscore) are stripped: the accepted
    reason-code set rides on the case record so the episode view scopes it
    automatically, but exposing it would hand the agent the answer.
    """

    @staticmethod
    def invoke(data: Dict[str, Any], case_id: str) -> str:
        c = _find_case(data, case_id)
        if c is None:
            return "Error: case not found"
        return json.dumps({k: v for k, v in c.items()
                           if not k.startswith("_")})

    @staticmethod
    def get_info() -> Dict[str, Any]:
        info = GetCaseDetails.get_info()
        info["function"]["description"] = (
            "Look up the case under review — a pending card transaction "
            "(TXN-...), a limit-increase request (LIM-...) or an outbound "
            "transfer (TRF-...). A case may list several ranked alternatives "
            "under `legs`; leg 1 is the customer's first preference.")
        return info


class GetAccountDetailsL3(GetAccountDetails):
    """L3 accounts carry today_total (the daily-cap input); the
    description names the field, since it decides the daily-cap tickets."""
    @staticmethod
    def get_info():
        info = GetAccountDetails.get_info()
        info["function"]["description"] = (
            "Look up an account: holder, home state, verification level, "
            "standing, tenure, current credit limit, and today's approved "
            "same-day total (today_total).")
        return info


class GetPayeeDetailsL3(GetPayeeDetails):
    @staticmethod
    def get_info():
        info = GetPayeeDetails.get_info()
        info["function"]["description"] = (
            "Look up a transfer payee: name, type, jurisdiction, and how "
            "many days ago the payee was added (added_days_ago).")
        return info


def _accepted(data, case_id):
    c = _find_case(data, case_id)
    return (c or {}).get("_accept") or []


def _normalise(data, case_id, reason):
    """Any code in the case's accepted set lands on the same stored value.

    Refusals are graded through the database hash here (unlike retail, where
    they live in the answer channel and the verifier can compare against a
    set), so equivalence has to be established when the value is written.
    """
    acc = _accepted(data, case_id)
    return acc[0] if (acc and reason in acc) else reason


def _has_legs(data, case_id) -> bool:
    return bool((_find_case(data, case_id) or {}).get("legs"))


def _is_batch(data, case_id) -> bool:
    return bool((_find_case(data, case_id) or {}).get("lines"))


class ApproveCaseL3(ApproveCase):
    """Approving is only meaningful for a single-request case: a fallback
    case has to name WHICH alternative was carried out. Rejected as a
    correctable error rather than silently writing an unreachable state."""

    @staticmethod
    def invoke(data: Dict[str, Any], case_id: str) -> str:
        if _has_legs(data, case_id):
            return ("Error: this case lists ranked alternatives — carry one "
                    "out with execute_leg(case_id, leg)")
        if _is_batch(data, case_id):
            return ("Error: this case is a payment batch — settle it with "
                    "decide_batch(case_id, decisions)")
        return _decide(data, case_id, "approved", None)


class ExecuteLegL3(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], case_id: str, leg: Any) -> str:
        c = _find_case(data, case_id)
        if c is None:
            return "Error: case not found"
        if not c.get("legs"):
            return ("Error: this case has no alternatives — decide it with "
                    "approve_case / deny_case / escalate_case")
        if c["status"] != "pending":
            return f"Error: case already {c['status']} — decisions are final"
        try:
            k = int(leg)
        except (TypeError, ValueError):
            return "Error: leg must be an integer"
        if not any(item["leg"] == k for item in c["legs"]):
            return (f"Error: no such alternative (this case lists legs "
                    f"1..{len(c['legs'])})")
        c["status"] = "approved"
        c["chosen_leg"] = k
        c["reason"] = None
        return f"Case {case_id} approved on alternative {k}"

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {"type": "function", "function": {
            "name": "execute_leg",
            "description": "Carry out one of a case's ranked alternatives, "
                           "identified by its leg number. Final.",
            "parameters": {"type": "object", "properties": {
                "case_id": {"type": "string"},
                "leg": {"type": "integer"}},
                "required": ["case_id", "leg"]}}}


class DecideBatchL3(Tool):
    """One call settles every line of a payment batch.

    Submitting the whole answer at once is deliberate: per-line calls would
    leak which lines were accepted before the agent commits to the rest, and
    the tool-call count would scale with the batch size — this form raises the
    hidden-information load without raising the operation count.
    """

    @staticmethod
    def invoke(data: Dict[str, Any], case_id: str, decisions: Any) -> str:
        c = _find_case(data, case_id)
        if c is None:
            return "Error: case not found"
        if not c.get("lines"):
            return ("Error: this case is not a payment batch — decide it with "
                    "approve_case / deny_case / escalate_case")
        if c["status"] != "pending":
            return f"Error: case already {c['status']} — decisions are final"
        if not isinstance(decisions, list):
            return "Error: decisions must be a list of per-line objects"
        want = {item["line"] for item in c["lines"]}
        got = {}
        for entry in decisions:
            if not isinstance(entry, dict):
                return "Error: each decision must be an object"
            try:
                n = int(entry.get("line"))
            except (TypeError, ValueError):
                return "Error: each decision needs an integer `line`"
            action = str(entry.get("action", "")).strip().lower()
            if action not in ("approve", "escalate", "deny"):
                return ("Error: each decision needs action = approve | "
                        "escalate | deny")
            reason = entry.get("reason")
            if action == "approve":
                reason = None
            elif reason not in REASONS:
                return "Error: invalid reason code"
            got[n] = (action, reason)
        if set(got) != want:
            return (f"Error: decide every line exactly once "
                    f"(this batch has lines {sorted(want)})")
        for item in c["lines"]:
            action, reason = got[item["line"]]
            item["status"] = {"approve": "approved", "deny": "denied",
                              "escalate": "escalated"}[action]
            item["reason"] = reason
        c["status"] = "settled"
        return f"Batch {case_id} settled ({len(got)} lines)"

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {"type": "function", "function": {
            "name": "decide_batch",
            "description": "Settle every line of a payment batch in one call. "
                           "`decisions` is a list of {line, action, reason} "
                           "objects — action is approve, deny or escalate, and "
                           "reason is required unless the action is approve. "
                           "Every line must appear exactly once. Final.",
            "parameters": {"type": "object", "properties": {
                "case_id": {"type": "string"},
                "decisions": {"type": "array"}},
                "required": ["case_id", "decisions"]}}}


class DenyCaseL3(DenyCase):
    @staticmethod
    def invoke(data: Dict[str, Any], case_id: str, reason: str) -> str:
        if reason not in REASONS:
            return "Error: invalid reason code"
        if _is_batch(data, case_id):
            return ("Error: this case is a payment batch — settle it with "
                    "decide_batch(case_id, decisions)")
        return _decide(data, case_id, "denied",
                       _normalise(data, case_id, reason))


class EscalateCaseL3(EscalateCase):
    @staticmethod
    def invoke(data: Dict[str, Any], case_id: str, reason: str) -> str:
        if reason not in REASONS:
            return "Error: invalid reason code"
        if _is_batch(data, case_id):
            return ("Error: this case is a payment batch — settle it with "
                    "decide_batch(case_id, decisions)")
        return _decide(data, case_id, "escalated",
                       _normalise(data, case_id, reason))
