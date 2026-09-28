"""L2 rule registry: SCOPE-STATE drift (the evaluator-only source of truth).

L2's difficulty axis is HOW policies change: a rule's state is not a boolean
but a SCOPE — the set of world states it currently blocks. Scopes narrow
("all electronics" -> "audio products only"), migrate ("gift-paid" ->
"split-paid"), and revert, on a denser event schedule with correlated bundles.

Docs stay frozen and silent on every hidden dimension; reality only tightens
(a scope move = local tightening + local vacating — the vacated side lands in
the General slice). Parameterized rules, permissive drift, and contradiction
forms are L3 scope by design.

The system under test never sees this file; tools do not enforce these rules.
Only task GT and the evaluator read truth(t).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

TIMELINE_LEN = 720

# Audio-class electronics (scope target of R1's narrow step). world.py asserts
# these names exist and that >=5 NON-audio electronics names remain.
AUDIO_PRODUCTS = frozenset({
    "Noise-Cancelling Headphones", "Wireless Earbuds", "Bluetooth Speaker",
})

# rule -> state; "on" = docs-consistent (allowed). Every off_* state is a scope.
_DEFAULTS: Dict[str, str] = {
    "R1_exch_category": "on",      # exchange ban by category scope
    "R2_cancel_payment": "on",     # cancel ban by payment-form scope
    "R3_refund_original": "on",    # original-method refund ban by card scope
    "R4_exch_geo": "on",           # exchange ban by destination scope
    "R5_modify_split": "off_split",   # static anchor: split-paid modify ban
    "R6_partial_return": "on",     # partial-return ban by order-composition scope
    "R7_partial_exchange": "on",   # partial-exchange ban on multi-item orders
    "R8_pobox_exchange": "off_pobox",  # static cold-dimension anchor
    # PERMISSIVE dimension (docs forbid, reality allows — the same shape as
    # L1's D5): gift-card-funded exchange upcharges ARE accepted. Its adapt
    # tickets are CARRY-OUT tasks, the structural counter to blanket refusal.
    "R9_upcharge_gift": "on",
}

# (t, rule, new_state): correlated bundles share a timestamp.
EVENTS: List[Tuple[int, str, str]] = [
    (80,  "R1_exch_category", "off_full"),      # E1 "scope day"
    (80,  "R6_partial_return", "off_multi"),
    (200, "R2_cancel_payment", "off_gift"),     # E2
    (200, "R3_refund_original", "off_all"),
    (200, "R7_partial_exchange", "off"),
    (320, "R1_exch_category", "off_audio"),     # E3: NARROW (cameras vacated)
    (320, "R4_exch_geo", "off_fl"),
    (440, "R3_refund_original", "off_amex"),    # E4: narrow + MIGRATE bundle
    (440, "R2_cancel_payment", "off_split"),    # migrate gift -> split
    (440, "R6_partial_return", "off_multi_elec"),
    (560, "R4_exch_geo", "off_tx"),             # E5: MIGRATE FL -> TX
    (560, "R1_exch_category", "on"),            # revert
    (640, "R2_cancel_payment", "on"),           # E6: revert bundle
    (640, "R7_partial_exchange", "on"),
]

WINDOW_STARTS = [1] + sorted({t for t, _, _ in EVENTS})   # 7 windows W0..W6


def truth(t: int) -> Dict[str, str]:
    state = dict(_DEFAULTS)
    for te, rule, new in EVENTS:
        if t >= te:
            state[rule] = new
    return state


def window_of(t: int) -> int:
    w = 0
    for i, ws in enumerate(WINDOW_STARTS):
        if t >= ws:
            w = i
    return w


def window_range(w: int) -> Tuple[int, int]:
    starts = WINDOW_STARTS + [TIMELINE_LEN + 1]
    return starts[w], starts[w + 1] - 1


# --- the single source of truth for "does rule R block this request" ---------
# ctx keys (populate what the op needs):
#   category, product_name          — of the ITEM being exchanged
#   pay_kind                        — "gift_only" | "split" | "card"
#   dest_original_card (bool), card_brand
#   addr_state (str), pobox (bool)
#   order_multi (bool), partial (bool)  — request covers a strict subset
#   order_has_elec (bool)

def blocks(rule: str, state: str, ctx: Dict[str, Any]) -> bool:
    if state == "on":
        return False
    if rule == "R1_exch_category":
        if state == "off_full":
            return ctx.get("category") == "electronics"
        if state == "off_audio":
            return ctx.get("product_name") in AUDIO_PRODUCTS
    if rule == "R2_cancel_payment":
        if state == "off_gift":
            return ctx.get("pay_kind") == "gift_only"
        if state == "off_split":
            return ctx.get("pay_kind") == "split"
    if rule == "R3_refund_original":
        if state == "off_all":
            return bool(ctx.get("dest_original_card"))
        if state == "off_amex":
            return bool(ctx.get("dest_original_card")) and \
                ctx.get("card_brand") == "amex"
    if rule == "R4_exch_geo":
        if state == "off_fl":
            return ctx.get("addr_state") == "FL"
        if state == "off_tx":
            return ctx.get("addr_state") == "TX"
    if rule == "R5_modify_split":
        return ctx.get("pay_kind") == "split"
    if rule == "R9_upcharge_gift":
        if state == "off_gift_up":
            return bool(ctx.get("upcharge_gift"))
    if rule == "R6_partial_return":
        if state == "off_multi":
            return bool(ctx.get("order_multi")) and bool(ctx.get("partial"))
        if state == "off_multi_elec":
            return bool(ctx.get("order_multi")) and bool(ctx.get("partial")) \
                and bool(ctx.get("order_has_elec"))
    if rule == "R7_partial_exchange":
        return bool(ctx.get("order_multi")) and bool(ctx.get("partial"))
    if rule == "R8_pobox_exchange":
        return bool(ctx.get("pobox"))
    return False


# Which op each rule can bite (dispatch helper).
RULE_OPS = {
    "R1_exch_category": ("exchange",),
    "R2_cancel_payment": ("cancel",),
    "R3_refund_original": ("return",),
    "R4_exch_geo": ("exchange",),
    "R5_modify_split": ("modify",),
    "R6_partial_return": ("return",),
    "R7_partial_exchange": ("exchange",),
    "R8_pobox_exchange": ("exchange",),
}

# Refusals are IRREVOCABLE (delivered to the customer -> K=1), so an
# accepted-code union would be enumerable surface. Every question bakes ONE
# code per rule (NARROW_CODES) — chosen inside ACCEPTED_CODES, matching the
# oracle bulletin, and balanced across the three codes. The remaining
# same-request-type collisions (R1/R7 both service_not_offered on exchanges,
# R4/R8 both destination_not_allowed) are unavoidable with 8 rules over 3
# codes; the discriminating layer is the rule condition, and the code stays
# semantically primary (pobox/geo = shipping destination; partial/category =
# service scope).
NARROW_CODES: Dict[str, str] = {
    "R1_exch_category": "service_not_offered",
    "R2_cancel_payment": "invalid_status",
    "R3_refund_original": "destination_not_allowed",
    "R4_exch_geo": "destination_not_allowed",
    "R5_modify_split": "invalid_status",
    "R6_partial_return": "invalid_status",
    "R7_partial_exchange": "service_not_offered",
    "R8_pobox_exchange": "destination_not_allowed",
    # R9 never blocks under truth — the code exists only for the docs-view
    # refusal a docs-follower would issue (a trap code, like L1's final_sale)
    "R9_upcharge_gift": "service_not_offered",
}

ACCEPTED_CODES: Dict[str, List[str]] = {
    "R1_exch_category": ["service_not_offered"],
    "R2_cancel_payment": ["invalid_status", "service_not_offered"],
    "R3_refund_original": ["destination_not_allowed"],
    "R4_exch_geo": ["service_not_offered", "destination_not_allowed"],
    "R5_modify_split": ["invalid_status", "service_not_offered"],
    "R6_partial_return": ["invalid_status"],
    "R7_partial_exchange": ["invalid_status", "service_not_offered"],
    "R8_pobox_exchange": ["destination_not_allowed"],
    "R9_upcharge_gift": ["service_not_offered"],
}

# docs imply everything is allowed (silent on all hidden dimensions)
DOCS_VIEW = {r: "on" for r in _DEFAULTS}
# the docs claim gift cards cannot fund exchange upcharges; truth allows it
DOCS_VIEW["R9_upcharge_gift"] = "off_gift_up"

RULE_ORDER = list(_DEFAULTS)


R9_TRUTH_LINE = ("- Exchange upcharges MAY be settled with a gift card with "
                 "sufficient balance, despite the manual's credit-card-only "
                 "wording. Carry such exchanges out.")


def active_rules(t: int) -> List[str]:
    """Rules whose truth differs from the docs view at t."""
    s = truth(t)
    # adapt space = truth-vs-DOCS mismatch, NOT truth-vs-"on": the static
    # permissive R9 (truth on, docs off_gift_up) is adapt in every window
    return [r for r in RULE_ORDER if s[r] != DOCS_VIEW[r]]


_STATE_LINES = {
    ("R1_exch_category", "off_full"):
        "- Exchanges are NOT available for ELECTRONICS items (per-item; other "
        "requested items are unaffected). Code if refusing: service_not_offered.",
    ("R1_exch_category", "off_audio"):
        "- Exchanges are NOT available for AUDIO products (headphones, earbuds, "
        "speakers, soundbars) — other electronics ARE exchangeable again "
        "(per-item). Code if refusing: service_not_offered.",
    ("R2_cancel_payment", "off_gift"):
        "- Cancellation is NOT available for orders paid ENTIRELY with a gift "
        "card. Code if refusing: invalid_status.",
    ("R2_cancel_payment", "off_split"):
        "- Cancellation is NOT available for SPLIT-paid orders (credit card + "
        "gift card); gift-only orders CAN be cancelled again. Code if refusing: "
        "invalid_status.",
    ("R3_refund_original", "off_all"):
        "- Return refunds to the ORIGINAL card are NOT available (gift card "
        "destinations only). Code if refusing: destination_not_allowed.",
    ("R3_refund_original", "off_amex"):
        "- Return refunds to an original AMEX card are NOT available; refunds "
        "to original VISA/MASTERCARD cards ARE available again. Code if "
        "refusing: destination_not_allowed.",
    ("R4_exch_geo", "off_fl"):
        "- Exchanges are NOT available for orders shipping to FLORIDA. Code if "
        "refusing: destination_not_allowed.",
    ("R4_exch_geo", "off_tx"):
        "- Exchanges are NOT available for orders shipping to TEXAS; Florida "
        "orders ARE served again. Code if refusing: destination_not_allowed.",
    ("R5_modify_split", "off_split"):
        "- Modification is NOT available for split-paid orders. Code if "
        "refusing: invalid_status.",
    ("R6_partial_return", "off_multi"):
        "- PARTIAL return REQUESTS (listing fewer than all items) are NOT "
        "accepted on multi-item orders; a request LISTING every item stays "
        "whole-order — and proceeds — even if some items get excluded. "
        "Code if refusing: invalid_status.",
    ("R6_partial_return", "off_multi_elec"):
        "- PARTIAL returns are NOT accepted on multi-item orders CONTAINING "
        "ELECTRONICS; multi-item orders without electronics accept partial "
        "returns again. Code if refusing: invalid_status.",
    ("R7_partial_exchange", "off"):
        "- Exchanging a SUBSET of a multi-item order is NOT available (whole-"
        "order requests and single-item orders unaffected). Code if refusing: "
        "service_not_offered.",
    ("R8_pobox_exchange", "off_pobox"):
        "- Exchanges are NOT available for orders shipping to a PO Box "
        "address. Code if refusing: destination_not_allowed.",
}


def oracle_note(t: int) -> str:
    """Authoritative bulletin: facts + citation codes, never 'refuse the
    ticket' imperatives — closing semantics stay with the workflow doc."""
    s = truth(t)
    lines = ["# Internal policy bulletin (authoritative)",
             "Each bulletin states what is currently AVAILABLE, overriding the"
             " documents where they conflict. The bulletin is INFORMATION ONLY —"
             " it performs nothing: every operation still requires the documented"
             " tool procedure, and nothing has happened until the tool call"
             " succeeds. Everything else stays in force: all"
             " other documented policies must still be checked, items are assessed"
             " independently, and when the customer states an ordered list of"
             " alternatives, carry out the FIRST permissible one (checking each"
             " in the stated order). For hypothetical or quote questions ('how"
             " much would I get back', 'how many items could be...'), reply with"
             " the number only and do NOT execute the operation. Declare a"
             " refusal ONLY when nothing in the request could be carried out;"
             " once any part has been executed, end with finish() and no"
             " refusal. The codes below apply to full refusals only."]
    for r in RULE_ORDER:
        if s[r] != "on":
            lines.append(_STATE_LINES[(r, s[r])])
        elif r in ("R1_exch_category", "R2_cancel_payment", "R3_refund_original",
                   "R4_exch_geo", "R6_partial_return", "R7_partial_exchange"):
            lines.append({
                "R1_exch_category": "- Exchanges ARE available for every category.",
                "R2_cancel_payment": "- Cancellation IS available regardless of payment form.",
                "R3_refund_original": "- Return refunds to the original card ARE available.",
                "R4_exch_geo": "- Exchanges ARE available for every US "
                               "STATE destination (no region-based "
                               "restriction; the separate PO Box rule "
                               "below still applies).",
                "R6_partial_return": "- Partial returns of multi-item orders ARE accepted.",
                "R7_partial_exchange": "- Exchanging a subset of a multi-item order IS available.",
            }[r])
    # permissive dimension: the docs FORBID gift-funded upcharges and reality
    # accepts them, so the bulletin must say so (otherwise a bulletin+docs
    # follower refuses every R9 ticket)
    lines.append(R9_TRUTH_LINE)
    return "\n".join(lines)
