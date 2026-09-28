"""Rule registry + drift schedule: the evaluator-only source of truth.

Documented (F) rules keep living in policy_config/docs. This module holds what
the docs never mention:

  H1  split-payment (card + gift card) orders cannot be modified   [constant]
  D1  gift-card-paid orders cancellable                            [drifts]
  D2  electronics items exchangeable                               [drifts]
  D3  exchanges for Florida-bound orders                           [drifts]
  D4  refund to the original payment method honored on request     [drifts]

Two dimensions run PERMISSIVELY: the DOCS forbid, reality allows, so their
adapt tickets are CARRY-OUT tasks that punish blanket refusal. If every hidden
dimension ran doc-says-allowed / reality-forbids, the adapt slice would be all
refusals and a policy-blind "always refuse" strategy would score 100% on it.

  D5  clearance items ARE returnable (docs: "final sale, never returnable")
      [constant permissive]
  D6  the return window is NOT enforced (docs: 14/30 days after delivery)
      [permissive drift: reality tightens... then relaxes at W3]

The system under test never sees this file's content; tools do not enforce
these rules (the API lags policy). Only task GT / the verifier read truth(t).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

TIMELINE_LEN = 720

# Truth flags before any event (documented behavior + undocumented perks).
_DEFAULTS: Dict[str, bool] = {
    "D1_gift_card_cancel": True,
    "D2_electronics_exchange": True,
    "D3_florida_exchange": True,
    "D4_original_refund": True,
    "H1_split_modify": False,   # hidden-static: forbidden from day one
    "D5_clearance_return": True,   # PERMISSIVE-static: docs call it final sale
    "D6_window_return": False,     # matches the docs until E6 relaxes it
}

# (t, flag, new_value): at episode t the flag silently becomes new_value.
EVENTS: List[Tuple[int, str, bool]] = [
    (80,  "D1_gift_card_cancel", False),
    (220, "D2_electronics_exchange", False),
    (360, "D1_gift_card_cancel", True),   # re-drift: the ban is lifted
    (360, "D6_window_return", True),      # RELAXATION: the window stops being
                                          # enforced while the docs still state
                                          # it — carry-out adapt tickets appear
    (480, "D3_florida_exchange", False),
    (600, "D4_original_refund", False),
]

# Window boundaries (start episodes); W0 starts at 1.
# de-duplicated: correlated events share a timestamp (t=360 is the
# "relaxation day" — the gift-card ban lifts and the return window stops
# being enforced in the same beat), so the tier keeps six windows.
WINDOW_STARTS = [1] + sorted({t for t, _, _ in EVENTS})


def truth(t: int) -> Dict[str, bool]:
    """Effective hidden-rule truth at episode t (1-indexed)."""
    state = dict(_DEFAULTS)
    for te, flag, value in EVENTS:
        if t >= te:
            state[flag] = value
    return state


def window_of(t: int) -> int:
    """W-index of episode t (W0..W5)."""
    w = 0
    for i, ws in enumerate(WINDOW_STARTS):
        if t >= ws:
            w = i
    return w


def window_range(w: int) -> Tuple[int, int]:
    starts = WINDOW_STARTS + [TIMELINE_LEN + 1]
    return starts[w], starts[w + 1] - 1


# Refusal codes accepted when a hidden rule is the (single) root cause.
# ONE accepted code per rule. Codes are assigned so that two rules reachable
# from the SAME request type never share one (exchange: D2 vs D3; cancel:
# D1 vs H1): the agent must identify WHICH rule blocks the request, and a
# fixed request-type -> code map cannot score without knowing the rule state.
NARROW_CODES: Dict[str, str] = {
    "H1_split_modify": "invalid_status",
    "D1_gift_card_cancel": "service_not_offered",
    "D2_electronics_exchange": "service_not_offered",
    "D3_florida_exchange": "destination_not_allowed",
    "D4_original_refund": "destination_not_allowed",
    # permissive dims refuse only while truth still matches the docs
    # (D5: never in this timeline; D6: windows W0-W2)
    "D5_clearance_return": "final_sale",
    "D6_window_return": "window_expired",
}
ACCEPTED_CODES: Dict[str, List[str]] = {k: [v] for k, v in NARROW_CODES.items()}


# What the docs imply on the hidden dimensions (the doc-follower's view).
DOCS_VIEW: Dict[str, bool] = {
    "D1_gift_card_cancel": True, "D2_electronics_exchange": True,
    "D3_florida_exchange": True, "D4_original_refund": True,
    "H1_split_modify": True,   # docs never forbid it -> implied allowed
    # the permissive axis: the documents DO forbid these, in writing
    "D5_clearance_return": False,   # "clearance items are final sale"
    "D6_window_return": False,      # "14/30 days after delivery"
}

# Difficulty tier of each hidden rule (L1 = enumerable answer space); tiers
# are metadata inside a scenario, never separate datasets.
RULE_TIER = {"H1_split_modify": "L1", "D1_gift_card_cancel": "L1",
             "D2_electronics_exchange": "L1", "D3_florida_exchange": "L1",
             "D4_original_refund": "L1", "D5_clearance_return": "L1",
             "D6_window_return": "L1"}

RULE_FAMILY = {
    "H1_split_modify": "h1_affected", "D1_gift_card_cancel": "d1_affected",
    "D2_electronics_exchange": "d2_affected", "D3_florida_exchange": "d3_affected",
    "D4_original_refund": "d4_affected", "D5_clearance_return": "d5_affected",
    "D6_window_return": "d6_affected",
}

# Rules whose adapt tickets are CARRY-OUT tasks (docs forbid, reality allows).
# Reported separately: accuracy here is what a blanket-refusal strategy cannot
# fake, so it is the honest read of "did the agent learn the rule".
PERMISSIVE_RULES = ("D5_clearance_return", "D6_window_return")


def active_rules(t: int) -> List[str]:
    """Rules whose truth at t differs from the docs-implied view — exactly the
    rules a doc-follower gets wrong; defines the Adapt space at t."""
    s = truth(t)
    order = ["H1_split_modify", "D1_gift_card_cancel", "D2_electronics_exchange",
             "D3_florida_exchange", "D4_original_refund", "D5_clearance_return",
             "D6_window_return"]
    return [r for r in order if s[r] != DOCS_VIEW[r]]


def oracle_note(t: int) -> str:
    """Ground-truth addendum injected into the oracle prompt: the full
    current truth of every hidden rule, phrased as an internal policy bulletin."""
    s = truth(t)
    lines = ["# Internal policy bulletin (authoritative)",
             "Each bulletin below applies ONLY to the exact condition it states; where it"
             " conflicts with the documents, the bulletin wins — but ALL other documented"
             " policies (thresholds, windows, final-sale, refund destinations, ...) remain"
             " in force and must still be checked. Declare a refusal ONLY when nothing"
             " in the request could be carried out; once any part has been executed,"
             " end with finish() and no refusal. The reasons below apply to full"
             " refusals only."]
    lines.append("- Orders paid with a split payment (credit card + gift card) can NOT be "
                 f"modified. Refuse such modification requests "
                 f"(reason: {NARROW_CODES['H1_split_modify']}).")
    if s["D1_gift_card_cancel"]:
        lines.append("- Orders paid with a gift card CAN be cancelled normally.")
    else:
        lines.append("- Orders paid with a gift card can currently NOT be cancelled. "
                     f"Refuse such cancellation requests "
                     f"(reason: {NARROW_CODES['D1_gift_card_cancel']}).")
    if s["D2_electronics_exchange"]:
        lines.append("- Electronics items CAN be exchanged normally.")
    else:
        lines.append("- Electronics items can currently NOT be exchanged. "
                     f"Refuse such exchange requests "
                     f"(reason: {NARROW_CODES['D2_electronics_exchange']}).")
    if s["D3_florida_exchange"]:
        lines.append("- Exchanges for orders shipping to Florida operate normally.")
    else:
        lines.append("- Exchanges are currently suspended for orders shipping to Florida. "
                     f"Refuse such exchange requests "
                     f"(reason: {NARROW_CODES['D3_florida_exchange']}).")
    if s["D4_original_refund"]:
        lines.append("- Return refunds MAY go to the original payment method when the "
                     "customer asks for it.")
    else:
        lines.append("- Return refunds can currently go to a GIFT CARD ONLY. Refuse "
                     "requests to refund the original payment method "
                     f"(reason: {NARROW_CODES['D4_original_refund']}).")
    if s["D5_clearance_return"]:
        lines.append("- Clearance items ARE accepted for return in practice, despite the "
                     "final-sale wording in the documents. Carry such returns out.")
    else:
        lines.append("- Clearance items are final sale and cannot be returned "
                     "(reason: final_sale).")
    if s["D6_window_return"]:
        lines.append("- The return window is NOT being enforced in practice: returns "
                     "past the documented 14/30-day window ARE accepted. Carry them out.")
    else:
        lines.append("- The documented return window applies: returns past it must be "
                     "refused (reason: window_expired).")
    return "\n".join(lines)
