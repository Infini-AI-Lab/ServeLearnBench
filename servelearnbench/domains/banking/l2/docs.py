"""Banking L2 frozen risk manual — wrong in exactly six places, correct
everywhere else.

Structure is public; the DISCRIMINATING content is not:
- unwritten rules: the overseas corridor (scope migrates mid-timeline) and
  the daily cap;
- stale printed values: electronics limit 500 (reality 650, tightening to
  460 — ACROSS the print), travel limit 1200 (reality 900, relaxing to
  1150);
- a printed value that GOES stale: the cap 4200 is correct until W4, then
  reality tightens to 3800;
- an over-strict clause: the manual escalates every new payee (<30 days),
  while current practice applies that review only above a hidden amount
  threshold.
Everything else is documented CORRECTLY (chargeback blocks increases,
reporting threshold 8000, gift cards unrestricted, no payee roster) —
trusting the manual there is right.

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
    "case_forms": (
        "A case takes one of three forms.\n"
        "1) A single request. Decide it with approve_case, deny_case or "
        "escalate_case.\n"
        "2) A case listing RANKED ALTERNATIVES under `legs`, most preferred "
        "first. The customer will accept any of them. Carry out the FIRST "
        "alternative that current practice approves automatically, using "
        "execute_leg. An alternative that would be denied or escalated is not "
        "carried out — move on to the next one. If NO alternative can be "
        "approved automatically, the case takes the disposition of leg 1 "
        "(deny_case or escalate_case as that leg requires); the reason code "
        "may be the code of any blocked alternative that carries the same "
        "disposition as leg 1.\n"
        "3) A PAYMENT BATCH listing several payment lines under `lines`. "
        "Settle it with a single decide_batch call that gives EVERY line its "
        "own action and, unless the action is approve, its own reason code. "
        "There is no partial submission and no way to revise a settled "
        "batch."),
    "transactions": (
        "Card transactions are checked against the category instant-approve "
        f"limits: electronics ${DOCS['electronics_limit']:.2f}, travel "
        f"${DOCS['travel_limit']:.2f}. Each pending transaction lists the "
        "account's same-day approved total alongside the charge. Categories "
        "without a listed limit "
        "have no amount restriction. A transaction above its limit is "
        "escalated with reason over_limit. Groceries, utilities, dining, "
        "fitness, electronics, travel, jewelry and gift cards are all "
        "serviceable; gift cards are available to all account tiers, and no "
        "category is restricted by merchant region."),
    "limits": (
        "A limit-increase request from an account with a recent chargeback "
        "is denied with reason account_standing, whatever the amount. "
        "Otherwise the request is approved when the requested limit is "
        f"within the approvable cap of ${DOCS['limit_cap']:.2f}; above the "
        "cap, deny with reason cap_exceeded."),
    "transfers": (
        "Outbound transfers to payees added within the last "
        f"{DOCS['new_payee_days']} days must be escalated for review with "
        "reason new_payee_review. All transfers are additionally screened "
        "against the reporting threshold of "
        f"${DOCS['reporting_threshold']:.2f}: a transfer above the "
        "threshold is escalated with reason reportable_amount. If both "
        "transfer checks apply, use new_payee_review."),
    "accounts": (
        "Accounts carry a verification level (basic or verified), a standing "
        "flag, tenure in years, a current credit limit, and a running total "
        "of today's approved card activity."),
}

WORKFLOW_SECTIONS = {
    "review_steps": (
        "1) get_case_details for the case under review; it tells you whether "
        "the case is a single request, a set of ranked alternatives (`legs`) "
        "or a payment batch (`lines`). 2) get_account_details, and "
        "get_merchant_details or get_payee_details as the case requires — for "
        "a multi-part case, look up every alternative or line you need. "
        "3) Apply policy and issue exactly one decision tool call: "
        "approve_case / deny_case / escalate_case for a single request, "
        "execute_leg for ranked alternatives, decide_batch for a payment "
        "batch. 4) finish."),
    # code definitions are deliberately ABSTRACT for the codes whose rules
    # are unwritten (corridor, daily cap): spelling out the exact hidden
    # condition would hand a zero-feedback agent the unwritten rule;
    # documented-rule codes keep their full definitions.
    "reasons": (
        "Reason codes: " + " | ".join(REASONS) + ". corridor_blocked: this "
        "payment corridor is closed under current practice. "
        "over_limit: the amount exceeds the "
        "category's instant-approve limit. account_standing: the account has "
        "a recent chargeback, which blocks limit increases. cap_exceeded: "
        "the requested "
        "limit is above the approvable cap. reportable_amount: the transfer "
        "is above "
        "the reporting threshold. daily_cap: the account's same-day activity "
        "does not permit instant approval. new_payee_review: the payee was "
        "added recently and the transfer requires review."),
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
