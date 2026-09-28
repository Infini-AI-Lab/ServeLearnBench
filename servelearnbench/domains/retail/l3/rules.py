"""L3 rule registry: OSCILLATION + PERMISSIVE drift (evaluator-only truth).

L3's difficulty axis is HOSTILE CHANGE DYNAMICS:
  - states RE-ENTER (A -> B -> A): evidence has a shelf life, notes need
    recency weighting, and "accumulate more proof" actively backfires;
  - one dimension drifts PERMISSIVELY (docs explicitly forbid clearance
    returns; reality allows them in some windows): scoring requires DOING
    what the documents prohibit — blanket refusal stops paying, and a
    blind-EXECUTE bot becomes the adversarial baseline.

This is an independent bundle; the only shared-layer hook it uses is the
config-gated `clearance_returnable` tool flag (default off).

The system under test never sees this file; tools do not enforce these rules
(the clearance gate makes the return tool mechanically permissive here).
Only task GT and the evaluator read truth(t).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

TIMELINE_LEN = 900

AUDIO_PRODUCTS = frozenset({
    "Noise-Cancelling Headphones", "Wireless Earbuds", "Bluetooth Speaker",
})

# rule -> DOCS-view state (what following the documents implies).
# "on" = docs allow; "off*" = docs FORBID (explicit prohibition in the text).
# Z3 and Z4 are the PERMISSIVE axes: docs forbid (gift-card-only refunds /
# final-sale clearance), truth relaxes them in permissive windows.
DOCS_VIEW: Dict[str, str] = {
    "Z1_exch_category": "on",
    "Z2_cancel_payment": "on",
    "Z3_refund_original": "off_all",   # docs: gift-card destinations only
    "Z4_clearance_return": "off_clear",   # docs: clearance is final sale
    "Z5_modify_split": "on",
    "Z6_pobox_exchange": "on",
    "Z7_exch_geo": "on",
    # Z8 docs-view: the manual forbids gift-funded upcharges
    "Z8_upcharge_gift": "off_gift_up",
}

# Truth defaults at t=1 (W0 = calibration window: everything matches docs
# except the two static anchors).
_DEFAULTS: Dict[str, str] = {
    "Z1_exch_category": "on",
    "Z2_cancel_payment": "on",
    "Z3_refund_original": "off_all",
    "Z4_clearance_return": "off_clear",
    "Z5_modify_split": "off_split",     # static anchor (hidden, as in L2)
    "Z6_pobox_exchange": "off_pobox",   # static anchor
    "Z7_exch_geo": "on",
    # STATIC PERMISSIVE anchor (docs forbid, reality allows — the
    # cross-tier shape: L1 D5 / L2 R9 / here Z8). Supplies carry-out adapt
    # tickets in EVERY window, including the all-restrictive phases W0/W4
    # where blanket refusal would otherwise score every adapt ticket.
    "Z8_upcharge_gift": "on",
}

# 10 windows, 90 ticks each: starts at 1, 90, 180, ..., 810.
# Design targets: every oscillating rule RE-ENTERS at least one previous
# state; every (rule, state-period) spans >=1 full window so the suite can
# give it >=8 adapt questions (learnability floor).
#
#          W0     W1      W2      W3      W4      W5      W6      W7      W8      W9
# Z1       on     offA    offA    on      on      offFULL offFULL on      on      on
# Z2       on     offG    offG    offS    offS    offG    offG    offS    offS    on
# Z3(perm) offAll offAll  ON      ON      offAll  offAll  onVISA  onVISA  offAll  offAll
# Z4(perm) offC   offC    ON      ON      offC    offC    offC    offC    ON      ON
# Z5       offSp  (static)
# Z6       offPo  (static)
# Z7       on     offFL   offFL   on      offTX   offTX   on      offFL   offFL   on
EVENTS: List[Tuple[int, str, str]] = [
    (90,  "Z1_exch_category", "off_audio"),
    (90,  "Z2_cancel_payment", "off_gift"),
    (90,  "Z7_exch_geo", "off_fl"),
    (180, "Z3_refund_original", "on"),           # PERMISSIVE: truth relaxes docs
    (180, "Z4_clearance_return", "on"),          # PERMISSIVE: truth relaxes docs
    (270, "Z1_exch_category", "on"),             # re-entry #1 (Z1 back to on)
    (270, "Z2_cancel_payment", "off_split"),
    (270, "Z7_exch_geo", "on"),
    (360, "Z3_refund_original", "off_all"),      # permissive window closes
    (360, "Z4_clearance_return", "off_clear"),   # permissive window closes
    (360, "Z7_exch_geo", "off_tx"),
    (450, "Z1_exch_category", "off_full"),       # Z1 re-closes DIFFERENTLY (all electronics)
    (450, "Z2_cancel_payment", "off_gift"),      # Z2 RE-ENTERS off_gift
    (540, "Z3_refund_original", "on_visa"),      # Z3 re-OPENS DIFFERENTLY (visa only)
    (540, "Z7_exch_geo", "on"),
    (630, "Z1_exch_category", "on"),
    (630, "Z2_cancel_payment", "off_split"),     # Z2 re-enters off_split
    (630, "Z7_exch_geo", "off_fl"),              # Z7 RE-ENTERS off_fl
    (720, "Z3_refund_original", "off_all"),
    (720, "Z4_clearance_return", "on"),          # permissive re-opens (W8-9)
    (810, "Z2_cancel_payment", "on"),
    (810, "Z7_exch_geo", "on"),
]

WINDOW_STARTS = [1] + sorted({t for t, _, _ in EVENTS})   # 10 windows W0..W9


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


# --- the single source of truth for "does rule Z block this request" ---------
# ctx keys (populate what the op needs):
#   category, product_name, clearance   — of the ITEM in question
#   pay_kind                            — "gift_only" | "split" | "card"
#   dest_original_card (bool)
#   addr_state (str), pobox (bool)
#   order_multi (bool), partial (bool)

def blocks(rule: str, state: str, ctx: Dict[str, Any]) -> bool:
    if state == "on":
        return False
    if rule == "Z8_upcharge_gift":
        return state == "off_gift_up" and bool(ctx.get("upcharge_gift"))
    if rule == "Z1_exch_category":
        if state == "off_audio":
            return ctx.get("product_name") in AUDIO_PRODUCTS
        if state == "off_full":
            return ctx.get("category") == "electronics"
    if rule == "Z2_cancel_payment":
        if state == "off_gift":
            return ctx.get("pay_kind") == "gift_only"
        if state == "off_split":
            return ctx.get("pay_kind") == "split"
    if rule == "Z3_refund_original":
        if state == "off_all":
            return bool(ctx.get("dest_original_card"))
        if state == "on_visa":
            return bool(ctx.get("dest_original_card")) and \
                ctx.get("card_brand") != "visa"
    if rule == "Z4_clearance_return":
        if state == "off_clear":
            return bool(ctx.get("clearance"))
    if rule == "Z5_modify_split":
        return ctx.get("pay_kind") == "split"
    if rule == "Z6_pobox_exchange":
        return bool(ctx.get("pobox"))
    if rule == "Z7_exch_geo":
        if state == "off_fl":
            return ctx.get("addr_state") == "FL"
        if state == "off_tx":
            return ctx.get("addr_state") == "TX"
    return False


RULE_OPS = {
    "Z1_exch_category": ("exchange",),
    "Z2_cancel_payment": ("cancel",),
    "Z3_refund_original": ("return",),
    "Z4_clearance_return": ("return",),    # item-level, return legs
    "Z5_modify_split": ("modify",),
    "Z6_pobox_exchange": ("exchange",),
    "Z7_exch_geo": ("exchange",),
}

# Refusals are K=1 (delivered code is irrevocable) — every rule bakes ONE
# accepted code, balanced three ways across the suite.
ACCEPTED_CODES: Dict[str, List[str]] = {
    "Z1_exch_category": ["service_not_offered"],
    "Z2_cancel_payment": ["invalid_status"],
    "Z3_refund_original": ["destination_not_allowed"],
    "Z4_clearance_return": ["final_sale"],
    "Z5_modify_split": ["invalid_status"],
    "Z6_pobox_exchange": ["service_not_offered"],
    "Z7_exch_geo": ["destination_not_allowed"],
    "Z8_upcharge_gift": ["service_not_offered"],
}
NARROW_CODES = {r: c[0] for r, c in ACCEPTED_CODES.items()}

RULE_ORDER = list(_DEFAULTS)


# EXPOSURE: generation keeps all ten underlying windows; two
# second-measurement windows are HIDDEN from serving and test to reduce
# evaluation cost. The pair (2, 4) keeps every policy PHASE SEGMENT with
# >=1 exposed window plus all reopen beats and the first/last windows, and
# minimizes serving shell-imbalance cells among such pairs. The remaining
# one-sided shell cells are serving-side attribution noise, not a scoring
# shortcut (test uses a disjoint phrasing pool).
HIDDEN_WINDOWS = (2, 4)
PRESENTED_WINDOWS = [w for w in range(len(WINDOW_STARTS))
                     if w not in HIDDEN_WINDOWS]      # underlying indices


def presented_of(underlying: int):
    """Presented index (W0..W7) of an underlying window, None if hidden."""
    return (PRESENTED_WINDOWS.index(underlying)
            if underlying in PRESENTED_WINDOWS else None)


def active_rules(t: int) -> List[str]:
    """Rules whose truth differs from the DOCS view at t (in EITHER
    direction — permissive divergence counts)."""
    s = truth(t)
    return [r for r in RULE_ORDER if s[r] != DOCS_VIEW[r]]


_STATE_LINES = {
    # Z8 static permissive: truth is always "on"; the line tells the
    # oracle the manual's credit-card-only wording is stale (every window
    # needs this entry)
    ("Z8_upcharge_gift", "on"):
        "- Exchange upcharges MAY be settled with a gift card with sufficient"
        " balance, despite the manual's credit-card-only wording. Carry such"
        " exchanges out.",
    ("Z1_exch_category", "off_audio"):
        "- Exchanges are NOT available for AUDIO products (headphones, earbuds,"
        " speakers) — per-item; other requested items are unaffected. Code if"
        " refusing: service_not_offered.",
    ("Z2_cancel_payment", "off_gift"):
        "- Cancellation is NOT available for orders paid ENTIRELY with a gift"
        " card. Code if refusing: invalid_status.",
    ("Z2_cancel_payment", "off_split"):
        "- Cancellation is NOT available for SPLIT-paid orders (card + gift"
        " card); gift-only orders CAN be cancelled. Code if refusing:"
        " invalid_status.",
    ("Z1_exch_category", "off_full"):
        "- Exchanges are NOT available for ANY ELECTRONICS item (per-item;"
        " other requested items are unaffected). Code if refusing:"
        " service_not_offered.",
    ("Z3_refund_original", "on"):
        "- Return refunds to the ORIGINAL card ARE currently available,"
        " overriding the documents' gift-card-only clause. This applies to the"
        " refund destination only — return eligibility rules are unchanged,"
        " and split-paid orders follow the documented proportional split.",
    ("Z3_refund_original", "on_visa"):
        "- Return refunds to an original VISA card ARE currently available"
        " (overriding the documents); refunds to any OTHER card brand remain"
        " unavailable — gift-card destination only. Code if refusing:"
        " destination_not_allowed.",
    ("Z4_clearance_return", "on"):
        "- CLEARANCE items ARE currently returnable, overriding the documents'"
        " final-sale clause (same fees and windows as regular items). This"
        " applies to RETURNS only — exchanging clearance items remains"
        " unavailable, and exchange eligibility counts are unchanged.",
    ("Z5_modify_split", "off_split"):
        "- Modification is NOT available for split-paid orders. Code if"
        " refusing: invalid_status.",
    ("Z6_pobox_exchange", "off_pobox"):
        "- Exchanges are NOT available for orders shipping to a PO Box"
        " address. Code if refusing: service_not_offered.",
    ("Z7_exch_geo", "off_fl"):
        "- Exchanges are NOT available for orders shipping to FLORIDA. Code if"
        " refusing: destination_not_allowed.",
    ("Z7_exch_geo", "off_tx"):
        "- Exchanges are NOT available for orders shipping to TEXAS. Code if"
        " refusing: destination_not_allowed.",
}


def oracle_note(t: int) -> str:
    """Authoritative bulletin: facts + codes; explicitly overrides the docs
    in BOTH directions (tightening and relaxing)."""
    s = truth(t)
    lines = ["# Internal policy bulletin (authoritative)",
             "Each bulletin states what is currently AVAILABLE or UNAVAILABLE,"
             " overriding the documents where they conflict — in either"
             " direction. The bulletin is INFORMATION ONLY — it performs"
             " nothing: every operation still requires the documented tool"
             " procedure, and nothing has happened until the tool call"
             " succeeds. Everything else stays in force: all other documented"
             " policies must still be checked, items are assessed"
             " independently, and when the customer states an ordered list of"
             " alternatives, carry out the FIRST permissible one (checking"
             " each in the stated order). For hypothetical or quote questions"
             " ('how much would I get back', 'how many items could be...'),"
             " reply with the number only and do NOT execute the operation."
             " Declare a refusal ONLY when nothing in the request could be"
             " carried out; once any part has been executed, end with finish()"
             " and no refusal. The codes below apply to full refusals only."]
    for r in RULE_ORDER:
        if s[r] != DOCS_VIEW[r]:
            lines.append(_STATE_LINES[(r, s[r])])
        elif r in ("Z1_exch_category", "Z2_cancel_payment",
                   "Z3_refund_original", "Z4_clearance_return", "Z7_exch_geo"):
            lines.append({
                "Z1_exch_category": "- Exchanges ARE available for every category.",
                "Z2_cancel_payment": "- Cancellation IS available regardless of payment form.",
                "Z3_refund_original": "- Return refunds to the original card"
                " remain UNAVAILABLE (gift-card destinations only), as"
                " documented. Code if refusing: destination_not_allowed.",
                "Z4_clearance_return": "- Clearance items remain FINAL SALE (not returnable), as "
        "documented. Code if refusing: final_sale.",
                "Z7_exch_geo": "- Exchanges ARE available for every destination.",
            }[r])
    return "\n".join(lines)
