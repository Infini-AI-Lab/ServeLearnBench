"""Banking L1 frozen risk manual — TRUE BUT INCOMPLETE (incomplete in
exactly four places).

Structure is public (what gets checked, which codes exist, that a restricted
payee roster exists); the DISCRIMINATING content is not:
- unwritten rules: the overseas-jewelry corridor (born mid-timeline), the
  gift-card verification tier, which payees are restricted;
- one stale printed value: electronics limit 500 (reality 650).
Everything else is documented CORRECTLY (limit cap 5000, chargeback blocks
increases, reporting threshold 8000) — trusting the manual there is right.

The manual is frozen for the whole timeline; judged outcomes are the only
ground truth about current practice.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from ....engine.tool import Tool
from .rules import DOCS, REASONS

POLICY_SECTIONS = {
    "decisions": (
        "Every pending case — card transaction, limit-increase request or "
        "outbound transfer — receives exactly one final decision: approve, "
        "deny, or escalate to manual review. Decisions are final and cannot "
        "be revised. Deny and escalate must cite a reason code."),
    "transactions": (
        "Card transactions are checked against the category instant-approve "
        f"limits: electronics ${DOCS['electronics_limit']:.2f}. Categories "
        "without a listed limit have no amount restriction. A transaction "
        "above its limit is escalated with reason over_limit. Groceries, "
        "utilities, dining, fitness, electronics, travel, jewelry and gift "
        "cards are all serviceable; gift cards are available to all account "
        "tiers, and no category is restricted by merchant region."),
    "limits": (
        "A limit-increase request from an account with a recent chargeback "
        "is denied with reason account_standing, whatever the amount. "
        "Otherwise the request is approved when the requested limit is "
        f"within the approvable cap of ${DOCS['limit_cap']:.2f}; above the "
        "cap, deny with reason cap_exceeded."),
    "transfers": (
        "Outbound transfers are screened against the restricted-payee roster "
        "(maintained by the compliance desk; entries are not published) and "
        "against the reporting threshold of "
        f"${DOCS['reporting_threshold']:.2f}. A transfer to a restricted "
        "payee is denied with reason restricted_payee; a transfer above the "
        "threshold is escalated with reason reportable_amount."),
    "accounts": (
        "Accounts carry a verification level (basic or verified), a standing "
        "flag, tenure in years, and a current credit limit."),
}

WORKFLOW_SECTIONS = {
    "review_steps": (
        "1) get_case_details for the case under review. 2) get_account_details, "
        "and get_merchant_details or get_payee_details as the case requires. "
        "3) Apply policy and issue exactly one decision tool call. 4) finish."),
    # code definitions are deliberately ABSTRACT for the codes whose rules
    # are unwritten, so the glossary does not reveal the hidden condition
    # behind each code; documented-rule codes keep their full definitions.
    "reasons": (
        "Reason codes: " + " | ".join(REASONS) + ". corridor_blocked: this "
        "payment corridor is closed under current practice. "
        "verification_required: the account must complete additional "
        "verification for this purchase. over_limit: the amount exceeds the "
        "category's instant-approve limit. account_standing: the account has "
        "a recent chargeback, which blocks limit increases. cap_exceeded: the requested "
        "limit is above the approvable cap. restricted_payee: the payee is "
        "on the restricted roster. reportable_amount: the transfer is above "
        "the reporting threshold."),
}


def merged_md(which: str) -> str:
    secs = POLICY_SECTIONS if which == "policy" else WORKFLOW_SECTIONS
    return "\n\n".join(f"## {k}\n{v}" for k, v in secs.items())


def system_prompt_toc() -> str:
    return ("  policy: " + ", ".join(POLICY_SECTIONS) + "\n"
            "  workflow: " + ", ".join(WORKFLOW_SECTIONS))


class ReadDocs(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], doc: str, section: str = "") -> str:
        secs = {"policy": POLICY_SECTIONS, "workflow": WORKFLOW_SECTIONS}.get(doc)
        if secs is None:
            return "Error: unknown doc (policy | workflow)"
        if not section:
            return json.dumps(list(secs))
        if section not in secs:
            return f"Error: unknown section (have: {', '.join(secs)})"
        return secs[section]

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {"type": "function", "function": {
            "name": "read_docs",
            "description": "Retrieve a section of the frozen risk manual "
                           "(doc: policy | workflow).",
            "parameters": {"type": "object", "properties": {
                "doc": {"type": "string"}, "section": {"type": "string"}},
                "required": ["doc"]}}}
