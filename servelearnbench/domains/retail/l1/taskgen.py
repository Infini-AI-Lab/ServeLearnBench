"""L1 per-window independent task generation with two-space slice tags.

Every question owns a dedicated order (built by world.py from the registry this
module's `slot_sequence` defines). Wording never depends on t; GT and the slice
tag are derived from rules.truth(t) / rules.active_rules(t).
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import Any, Dict, List, Tuple

from ....engine.types import Action, Task
from . import rules

CANCEL_REASON = "no longer needed"
RETURN_REASON = "no longer needed"

MIRROR = {"h1_affected": "h1_control", "d1_affected": "d1_control",
          "d2_affected": "d23_control", "d3_affected": "d23_control",
          "d4_affected": "d4_control", "d5_affected": "d4_control",
          "d6_affected": "d4_control"}
# Clearance and the return window are the hidden PERMISSIVE dimensions D5/D6
# (docs forbid, reality allows), not documented-rule families: a fixed "refuse
# per docs" grade would be wrong once truth diverges. d6_affected covers
# window_expired while D6 still matches the docs; final_sale is a pure trap
# code (never correct in this tier).
F_ROT = ["f_thr", "f_thr_mid", "g_thr_mid", "g_win_home", "g_status", "g_balance"]

FAMILY_RULE = {v: k for k, v in rules.RULE_FAMILY.items()}


def slot_sequence(n: int = rules.TIMELINE_LEN, seed: int = 11) -> List[Tuple[int, str]]:
    """Deterministic (t, family) serving slots — shared by world.py and here."""
    ALL_RULE_FAMS = [rules.RULE_FAMILY[r] for r in
                     ("H1_split_modify", "D1_gift_card_cancel", "D2_electronics_exchange",
                      "D3_florida_exchange", "D4_original_refund",
                      "D5_clearance_return", "D6_window_return")]
    out, t, b = [], 1, 0
    a_i = m_i = f_i = 0
    while t <= n:
        # every rule family is sampled EVERY block, active or not — drift must be
        # observable (working -> broken -> reverted); density is tuned only via
        # the 4 extra slots that rotate over currently-active rules
        active = rules.active_rules(t)
        fams = [rules.RULE_FAMILY[r] for r in active]
        mirrors = sorted({MIRROR[f] for f in fams})
        slots = list(ALL_RULE_FAMS)
        for _ in range(4):
            slots.append(fams[a_i % len(fams)]); a_i += 1
        for _ in range(2):
            slots.append(mirrors[m_i % len(mirrors)]); m_i += 1
        slots.append(F_ROT[f_i % len(F_ROT)]); f_i += 1
        random.Random(seed + b).shuffle(slots)
        for fam in slots:
            if t > n:
                break
            out.append((t, fam))
            t += 1
        b += 1
    return out


# Explicit per-window adapt quotas (L1). Two constraints at once:
# (a) the PERMISSIVE dimensions D5/D6 give every window's adapt slice >= 20%
#     carry-out tickets (W3, the relaxation day, gets 40%), so blanket
#     refusal is penalized inside adapt itself;
# (b) suite totals per family stay >= 30 (H1 56, D1 33, D2 33, D3 30,
#     D4 30, D5 38) — D6 totals 20, a declared limit: it exists only from
#     W3 (late exposure).
# Totals are pinned independently in ADAPT_TOTALS below.
ADAPT_QUOTAS: Dict[int, Dict[str, int]] = {
    0: {"H1_split_modify": 16, "D5_clearance_return": 4},
    1: {"H1_split_modify": 12, "D1_gift_card_cancel": 20, "D5_clearance_return": 8},
    2: {"H1_split_modify": 8, "D1_gift_card_cancel": 13,
        "D2_electronics_exchange": 11, "D5_clearance_return": 8},
    3: {"H1_split_modify": 10, "D2_electronics_exchange": 14,
        "D5_clearance_return": 8, "D6_window_return": 8},
    4: {"H1_split_modify": 6, "D2_electronics_exchange": 4,
        "D3_florida_exchange": 20, "D5_clearance_return": 4, "D6_window_return": 6},
    5: {"H1_split_modify": 4, "D2_electronics_exchange": 4,
        "D3_florida_exchange": 10, "D4_original_refund": 30,
        "D5_clearance_return": 6, "D6_window_return": 6},
}


# Pinned per-window adapt totals: a second, independent anchor so a typo in
# ADAPT_QUOTAS cannot silently change the suite size.
ADAPT_TOTALS = {0: 20, 1: 40, 2: 40, 3: 40, 4: 40, 5: 60}


def test_plan(w: int) -> List[Tuple[str, str]]:
    """[(family, slice)] for window w's frozen test set: quota adapt + 24 general."""
    t0, _ = rules.window_range(w)
    active = rules.active_rules(t0)
    quotas = ADAPT_QUOTAS[w]
    stray = set(quotas) - set(active)
    assert not stray, (f"W{w} (t={t0}): quota assigns adapt questions to {sorted(stray)}, "
                       f"but only doc-mismatched rules {sorted(active)} can be adapt")
    assert sum(quotas.values()) == ADAPT_TOTALS[w], (
        f"W{w}: quota sum {sum(quotas.values())} != pinned total {ADAPT_TOTALS[w]}")
    plan: List[Tuple[str, str]] = []
    for r in active:  # keep the original active-rule order for stable generation
        plan += [(rules.RULE_FAMILY[r], "adapt")] * quotas.get(r, 0)
    # 4x contrast coverage for the exchange/return controls — they share
    # item draws with d3/d4 affected tasks and break item->refuse confounds.
    for fam, n in (("d1_control", 2), ("d23_control", 4),
                   ("d4_control", 4), ("h1_control", 2)):
        plan += [(fam, "general")] * n
    # documented F rules: threshold pair (refuse/allow), clearance, window pair.
    # (f_thr is excluded: its over_threshold coverage duplicates f_thr_mid.)
    for fam in ("f_thr_mid", "g_thr_mid", "g_win_home"):
        plan += [(fam, "general")]
    # documented-rule coverage for two refusal codes no other General family exercises:
    # invalid_status (cancel a processed order) and insufficient_balance
    # (gift-card upcharge that the balance cannot cover)
    for fam in ("g_status", "g_balance"):
        plan += [(fam, "general")] * 2
    consistent = [rules.RULE_FAMILY[r] for r in
                  ("D1_gift_card_cancel", "D2_electronics_exchange",
                   "D3_florida_exchange", "D4_original_refund", "D6_window_return")
                  if r not in active]
    # while the return window is still ENFORCED (pre-relaxation), its
    # refusals get 3 dedicated tickets per window — a 1-ticket sample cannot
    # carry the pre/post-relaxation comparison curve.
    n_fill = 5
    if "D6_window_return" not in active:
        plan += [("d6_affected", "general")] * 3
        n_fill = 2
        consistent = [f for f in consistent if f != "d6_affected"]
    for i in range(n_fill):  # doc-consistent affected shapes (stale-note trap)
        fam = consistent[i % len(consistent)] if consistent else "d23_control"
        plan.append((fam, "general"))
    return plan


def slice_of(family: str, t: int) -> str:
    r = FAMILY_RULE.get(family)
    return "adapt" if r and r in rules.active_rules(t) else "general"


def _ticket(spec, body):
    return (f"Customer: {spec['first']} {spec['last']} (zip {spec['zip']}) writes:\n"
            f"\"{body}\"")


# --- paraphrase pools (L1) -------------------------------------------------
# One pool per INTENT, shared across affected/control families of that intent so
# wording never correlates with the ground truth. Selection is a deterministic
# hash of task_id (independent of the world RNG stream). Every variant preserves
# the semantic anchors the GT depends on: literal ids, the valid cancel/return
# reason ("no longer needed"), the explicit refund destination, and the target
# variant + payment method for exchanges/modifications.
def _pick(task_id: str, pool):
    return pool[int(hashlib.md5(task_id.encode()).hexdigest(), 16) % len(pool)]


# SERVING-ONLY phrasing bodies: the frozen test suite never shares a
# phrasing with the serving stream.
SV_CANCEL_BODIES = [
    'Hey, I need order {oid} cancelled — I no longer need it.',
    "Please drop order {oid} from my account; it's no longer needed.",
    'Change of plans: order {oid} is no longer needed. Cancel it, please.',
    'Can you cancel order {oid}? No longer needed on my end.',
    "I'm requesting cancellation of {oid} — the items are no longer needed.",
    'Order {oid} — please cancel. Reason: no longer needed.',
    'Hello! Could you cancel my recent order {oid}? I no longer need it, thanks.',
    'Turns out we no longer need order {oid}. Please cancel it.',
]
SV_EXCHANGE_BODIES = [
    'For my order {oid}, swap the {item_name} (item {item_id}) to item {new_item_id} — the {new_opts} one. Bill any difference to my credit card ({pay_id}).',
    'Exchange request on {oid}: the {item_name}, item {item_id}, for the variant with {new_opts} (item {new_item_id}). Difference on my credit card ({pay_id}) is fine.',
    "Hi! Order {oid} — I'd rather have the {new_opts} version of the {item_name}. That's item {item_id} going out, item {new_item_id} coming in. My credit card ({pay_id}) can cover any difference.",
    'Can we do an exchange? Order {oid}, the {item_name} (item {item_id}) — I want item {new_item_id} with {new_opts} instead. Settle whatever difference there is via my credit card ({pay_id}).',
    'I picked the wrong variant on {oid}. Please exchange the {item_name} (item {item_id}) for item {new_item_id} ({new_opts}); price difference to my credit card ({pay_id}).',
    'About the {item_name} in order {oid} — item {item_id} — could it become the {new_opts} variant, item {new_item_id}? Use my credit card ({pay_id}) for any gap in price.',
    'Order {oid}: exchange item {item_id} (the {item_name}) to item {new_item_id}, {new_opts}. Card ({pay_id}) for the difference, thanks.',
    'Hello — hoping to trade the {item_name} (item {item_id}) on order {oid} for the {new_opts} version, item {new_item_id}. Any price change can hit my credit card ({pay_id}).',
]
SV_RETURN_BODIES = [
    'I want to send back the {item_name} (item {item_id}) from order {oid} — no longer needed. Refund to {dest}, please.',
    'Return request: order {oid}, item {item_id} (the {item_name}). Reason is no longer needed; put the refund on {dest}.',
    'Hi, the {item_name} from {oid} — item {item_id} — is no longer needed. Could you take it back and refund {dest}?',
    "Please start a return on order {oid} for the {item_name} (item {item_id}). It's no longer needed. Refund destination: {dest}.",
    'No longer needed: the {item_name}, item {item_id}, from my order {oid}. Send the money back to {dest}.',
    "Could I return item {item_id} — the {item_name} — off order {oid}? No longer needed. I'd like the refund on {dest}.",
    "Hello, I'm returning the {item_name} (item {item_id}) from {oid} since it's no longer needed. Refund should land on {dest}.",
    'One return for order {oid}: {item_name}, item {item_id}, no longer needed. Credit the refund to {dest}.',
]
SV_MODIFY_BODIES = [
    "Order {oid} hasn't gone out yet, right? Please change the {item_name} (item {item_id}) to item {new_item_id} — the {new_opts} one. Difference to my credit card ({pay_id}).",
    'Before {oid} ships: swap the {item_name}, item {item_id}, for item {new_item_id} with {new_opts}. Bill any difference to my credit card ({pay_id}).',
    'Modification for pending order {oid}: item {item_id} ({item_name}) should be item {new_item_id}, the {new_opts} variant. My credit card ({pay_id}) for any difference.',
    'Hi — my order {oid} is still pending. Can the {item_name} (item {item_id}) become the {new_opts} version (item {new_item_id})? Difference on my credit card ({pay_id}).',
    'Please adjust order {oid} before shipment. The {item_name} (item {item_id}) needs to be item {new_item_id} — {new_opts}. Use my credit card ({pay_id}) for the difference.',
    'Quick change on pending order {oid}: {item_name} item {item_id} to item {new_item_id} ({new_opts}). Settle the difference with my credit card ({pay_id}).',
]


def _pick2(task_id, spec, test_pool, serving_pool):
    pool = serving_pool if spec.get("split") == "train" else test_pool
    return _pick(task_id, pool)


CANCEL_BODIES = [
    "Please cancel my order {oid} — I no longer need it.",
    "I'd like to cancel order {oid}. It's no longer needed.",
    "Could you cancel {oid} for me? I don't need it anymore.",
    "Hi — order {oid} is no longer needed. Please cancel it.",
    "I want to cancel my order {oid} since I no longer need it.",
    "Please go ahead and cancel order {oid}; I no longer have any need for it.",
    "Cancel order {oid}, please — turns out I don't need it anymore.",
    "I need to cancel {oid}. Reason: no longer needed.",
]

EXCHANGE_BODIES = [
    "I'd like to exchange the {item_name} (item {item_id}) in my order {oid} for "
    "the variant with {new_opts} (item {new_item_id}). If there is a price "
    "difference, settle it with my credit card ({pay_id}).",
    "In order {oid}, I want to swap the {item_name} (item {item_id}) for the "
    "{new_opts} variant (item {new_item_id}). Any price difference can go on my "
    "credit card ({pay_id}).",
    "Could you exchange the {item_name} (item {item_id}) from order {oid}? I'd "
    "like the variant with {new_opts} (item {new_item_id}) instead. Charge or "
    "refund the difference to my credit card ({pay_id}).",
    "About order {oid}: I'd like an exchange — please replace the {item_name} "
    "(item {item_id}) with item {new_item_id}, the one with {new_opts}. Use my "
    "credit card ({pay_id}) for any difference.",
    "I'd like to swap for a different variant of the {item_name} in order "
    "{oid}: please replace item {item_id} with item {new_item_id} ({new_opts}). "
    "Settle any price gap with my credit card ({pay_id}).",
    "Hi, for order {oid} — can I trade the {item_name} (item {item_id}) in for "
    "the {new_opts} version (item {new_item_id})? Put any difference on my "
    "credit card ({pay_id}).",
    "Requesting an exchange on order {oid}: {item_name}, item {item_id} to item "
    "{new_item_id} ({new_opts}). Price difference to my credit card ({pay_id}).",
    "The {item_name} (item {item_id}) in my order {oid} isn't quite right — I'd "
    "like item {new_item_id} with {new_opts} instead. My credit card ({pay_id}) "
    "can cover any difference.",
]

RETURN_BODIES = [
    "Please return the {item_name} (item {item_id}) from my order {oid}; "
    "reason: no longer needed. Send the refund to {dest}.",
    "I'd like to return the {item_name} (item {item_id}) in order {oid} — it's "
    "no longer needed. Please issue the refund to {dest}.",
    "Could you process a return for item {item_id} (the {item_name}) from order "
    "{oid}? I no longer need it. The refund should go to {dest}.",
    "Returning the {item_name} (item {item_id}) from {oid}: no longer needed. "
    "I want the money back on {dest}.",
    "Hi — please take back the {item_name} (item {item_id}) from order {oid}. "
    "Reason: no longer needed. Refund to {dest}.",
    "Order {oid}: I want to return item {item_id} ({item_name}) since it's no "
    "longer needed, with the refund going to {dest}.",
    "Please set up a return for the {item_name} (item {item_id}) on order {oid} "
    "— no longer needed — and send the refund to {dest}.",
    "I don't need the {item_name} (item {item_id}) from order {oid} any longer. "
    "Please return it and credit {dest}.",
]

MODIFY_BODIES = [
    "In my pending order {oid}, please change the {item_name} (item {item_id}) "
    "to the variant with {new_opts} (item {new_item_id}). Settle any price "
    "difference with my credit card ({pay_id}).",
    "For my pending order {oid}: please swap the {item_name} (item {item_id}) "
    "to item {new_item_id} — the variant with {new_opts}. Any price difference "
    "goes on my credit card ({pay_id}).",
    "Can you update order {oid} before it ships? Change the {item_name} (item "
    "{item_id}) to the {new_opts} variant (item {new_item_id}); settle any "
    "difference with my credit card ({pay_id}).",
    "I'd like to change an item in my pending order {oid}: {item_name}, from "
    "item {item_id} to item {new_item_id} ({new_opts}). Use my credit card "
    "({pay_id}) for any price gap.",
    "Order {oid} hasn't shipped yet — please switch the {item_name} (item "
    "{item_id}) to item {new_item_id} with {new_opts}. My credit card "
    "({pay_id}) covers any difference.",
    "Please modify pending order {oid}: replace item {item_id} ({item_name}) "
    "with item {new_item_id}, the {new_opts} version. Charge any difference to "
    "my credit card ({pay_id}).",
    "Before order {oid} goes out, please swap the {item_name} (item {item_id}) "
    "for the variant with {new_opts} (item {new_item_id}), settling the "
    "difference with my credit card ({pay_id}).",
    "Quick change to my pending order {oid}: {item_name} (item {item_id}) to "
    "item {new_item_id} ({new_opts}). Any price difference on my credit card "
    "({pay_id}).",
]


def _refuse(reasons):
    return {"kind": "refuse", "reasons": list(reasons)}


def build_task(spec: Dict[str, Any], t: int, task_id: str) -> Task:
    fam = spec["family"]
    s = rules.truth(t)
    oid = spec["order_id"]

    if fam in ("d1_affected", "d1_control", "f_thr", "f_thr_mid", "g_thr_mid", "g_status"):
        body = _pick2(task_id, spec, CANCEL_BODIES, SV_CANCEL_BODIES).format(oid=oid)
        if fam == "g_status":
            allowed, codes = False, ["invalid_status"]   # processed order: docs say pending-only
        elif fam in ("f_thr", "f_thr_mid"):
            allowed, codes = False, ["over_threshold"]
        elif fam == "g_thr_mid":
            allowed, codes = True, []
        else:
            allowed = s["D1_gift_card_cancel"] if fam == "d1_affected" else True
            codes = rules.ACCEPTED_CODES["D1_gift_card_cancel"]
        actions = [Action(name="cancel_pending_order",
                          kwargs={"order_id": oid, "reason": CANCEL_REASON})] if allowed else []

    elif fam in ("d2_affected", "d23_control", "d3_affected", "g_balance"):
        body = _pick2(task_id, spec, EXCHANGE_BODIES, SV_EXCHANGE_BODIES).format(
            oid=oid, item_name=spec["item_name"], item_id=spec["item_id"],
            new_opts=spec["new_opts"], new_item_id=spec["new_item_id"],
            pay_id=spec["pay_id"])
        if fam == "g_balance":
            # upcharge exceeds the pinned gift card's balance; the wording pins
            # the gift card as the only instrument the customer will pay with
            # (case-insensitive: variants may open a sentence with "My credit card")
            body = re.sub(r"(m)y credit card", r"\1y gift card", body,
                          flags=re.IGNORECASE) + \
                " Please put it on that gift card only — do not charge my credit card."
            allowed, codes = False, ["insufficient_balance"]
        elif fam == "d2_affected":
            allowed, codes = s["D2_electronics_exchange"], rules.ACCEPTED_CODES["D2_electronics_exchange"]
        elif fam == "d3_affected":
            allowed, codes = s["D3_florida_exchange"], rules.ACCEPTED_CODES["D3_florida_exchange"]
        else:
            allowed, codes = True, []
        actions = [Action(name="exchange_delivered_order_items",
                          kwargs={"order_id": oid, "item_ids": [spec["item_id"]],
                                  "new_item_ids": [spec["new_item_id"]],
                                  "payment_method_id": spec["pay_id"]})] if allowed else []

    elif fam in ("d4_affected", "d4_control", "f_win", "f_clr", "f_win_elec",
                 "g_win_home", "d5_affected", "d6_affected"):
        dest_phrase = (f"my original credit card ({spec['dest_id']}) — the card I paid with"
                       if fam == "d4_affected" else f"my gift card ({spec['dest_id']})")
        body = _pick2(task_id, spec, RETURN_BODIES, SV_RETURN_BODIES).format(
            oid=oid, item_name=spec["item_name"], item_id=spec["item_id"],
            dest=dest_phrase)
        if fam in ("f_win", "f_win_elec"):
            allowed, codes = False, ["window_expired"]
        elif fam == "g_win_home":
            allowed, codes = True, []
        elif fam == "f_clr":
            allowed, codes = False, ["final_sale"]
        elif fam == "d5_affected":
            # permissive: docs say final sale, truth accepts the return
            allowed, codes = s["D5_clearance_return"], ["final_sale"]
        elif fam == "d6_affected":
            # permissive drift: refuse while the window is enforced (docs-
            # consistent -> general), carry out after the relaxation (adapt)
            allowed, codes = s["D6_window_return"], ["window_expired"]
        else:
            allowed = s["D4_original_refund"] if fam == "d4_affected" else True
            codes = rules.ACCEPTED_CODES["D4_original_refund"]
        actions = [Action(name="return_delivered_order_items",
                          kwargs={"order_id": oid, "item_ids": [spec["item_id"]],
                                  "payment_method_id": spec["dest_id"],
                                  "reason": RETURN_REASON})] if allowed else []

    elif fam in ("h1_affected", "h1_control"):
        body = _pick2(task_id, spec, MODIFY_BODIES, SV_MODIFY_BODIES).format(
            oid=oid, item_name=spec["item_name"], item_id=spec["item_id"],
            new_opts=spec["new_opts"], new_item_id=spec["new_item_id"],
            pay_id=spec["pay_id"])
        allowed = s["H1_split_modify"] if fam == "h1_affected" else True
        codes = rules.ACCEPTED_CODES["H1_split_modify"]
        actions = [Action(name="modify_pending_order_items",
                          kwargs={"order_id": oid, "item_ids": [spec["item_id"]],
                                  "new_item_ids": [spec["new_item_id"]],
                                  "payment_method_id": spec["pay_id"]})] if allowed else []
    else:
        raise AssertionError(fam)

    answer = None if actions else _refuse(codes)
    return Task(task_id=task_id, user_id=spec["user_id"],
                instruction=_ticket(spec, body), actions=actions, answer=answer,
                z={"family": fam, "t": t, "order_id": oid, "split": spec["split"],
                   "slice": slice_of(fam, t), "window": rules.window_of(t),
                   "tier": rules.RULE_TIER.get(FAMILY_RULE.get(fam), "L1")})


def _opaque_id(*parts) -> str:
    """Opaque task id: a hash of the parts, so the id carries no episode,
    family, window or slice information (a readable id would be an answer key
    if any harness path echoed it). slice/family/window live only in z."""
    return "v2_" + hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()[:10]


def serving_sequence(registry) -> List[Task]:
    """720 training questions, one dedicated order each, in timeline order."""
    return [build_task(spec, spec["t"], _opaque_id("s", spec["t"], spec["family"]))
            for spec in registry["train"]]


def test_set(registry, w: int) -> List[Task]:
    t0, _ = rules.window_range(w)
    return [build_task(spec, t0, _opaque_id("t", w, spec["family"], i))
            for i, spec in enumerate(registry["test"][w])]
