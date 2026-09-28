"""L2 task generation: scope-drift suite (single GT engine).

Every question's GT derives from ONE engine: the spec carries the request
(op, items, chain legs, ask) plus the ctx attributes each rule reads;
`_gt(spec, state)` walks rules.blocks() under truth(t) — and under DOCS_VIEW
for the slice tag. No per-family GT branching: families only decide WORLD
SHAPE and PHRASING, so adding a family cannot desynchronize GT logic.

Archetypes (family prefixes):
  p_*    D  single-op boundary probes (one per boundary SIDE; serving + test)
  mxr/mxe A  5-item mixed subset execution (return / exchange)
  agg    B  aggregate value questions ("how much refund...") — float compare
  chn    C  ordered-preference chains (4 legs, stop position varies)
  rfz    E  nothing-works refusals
  gl_*   General doc-derivable versions (limits, doc-blocked aggregates, ...)
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple

from ....engine.types import Action, Task
from . import rules

CANCEL_REASON = "no longer needed"
RETURN_REASON = "no longer needed"


def _pick(task_id: str, pool):
    return pool[int(hashlib.md5(task_id.encode()).hexdigest(), 16) % len(pool)]


# =============================================================================
# The GT engine
# =============================================================================
# spec["request"] is a list of LEGS in customer preference order; each leg:
#   {"op": "exchange"|"return"|"cancel"|"modify",
#    "item_ids": [...], "new_item_ids": [...] (swap ops),
#    "dest": payment_method_id (return), "pay": pm id (swap settle),
#    "mech_block": None | "unavailable" (leg impossible mechanically/doc-wise),
#    "ctx": {order/leg-level rule attributes},
#    "items_ctx": per-item ctx dicts aligned with item_ids (exchange),
#    "doc_bad_idx": indices doc-ineligible (clearance) — dropped per policy #1}
# spec["ask"]: None (action task) or
#   {"prices": [...aligned with leg item_ids...], "fee": float}
#   -> value answer = sum(eligible prices) * (1-fee) of the first viable leg.

ITEM_RULES = ("R1_exch_category",)          # per-item drops (exchange only)
ORDER_RULES = {                              # order/leg-level kills
    "exchange": ("R4_exch_geo", "R8_pobox_exchange",
                 "R7_partial_exchange", "R9_upcharge_gift"),
    "return": ("R6_partial_return", "R3_refund_original"),
    "cancel": ("R2_cancel_payment",),
    "modify": ("R5_modify_split",),
}


def _leg_outcome(leg, state) -> Tuple[str, Any]:
    """('exec', eligible_idx) | ('dead', blocking_rules)."""
    if leg.get("mech_block"):
        return ("dead", [leg["mech_block"]])
    ctx = leg["ctx"]
    killers = [r for r in ORDER_RULES[leg["op"]]
               if rules.blocks(r, state[r], ctx)]
    if killers:
        return ("dead", killers)
    idx = list(range(len(leg.get("item_ids") or [None])))
    if leg["op"] == "exchange" and leg.get("items_ctx"):
        idx = [i for i in idx
               if not any(rules.blocks(r, state[r], leg["items_ctx"][i])
                          for r in ITEM_RULES)]
        if not idx:
            return ("dead", ["R1_exch_category"])
    if leg.get("doc_bad_idx"):
        idx = [i for i in idx if i not in leg["doc_bad_idx"]]
        if not idx:
            return ("dead", ["doc"])
    return ("exec", idx)


def _leg_action(leg, idx, oid) -> Action:
    op = leg["op"]
    if op == "cancel":
        return Action(name="cancel_pending_order",
                      kwargs={"order_id": oid, "reason": CANCEL_REASON})
    if op == "return":
        return Action(name="return_delivered_order_items",
                      kwargs={"order_id": oid,
                              "item_ids": [leg["item_ids"][i] for i in idx],
                              "payment_method_id": leg["dest"],
                              "reason": RETURN_REASON})
    name = ("exchange_delivered_order_items" if op == "exchange"
            else "modify_pending_order_items")
    return Action(name=name,
                  kwargs={"order_id": oid,
                          "item_ids": [leg["item_ids"][i] for i in idx],
                          "new_item_ids": [leg["new_item_ids"][i] for i in idx],
                          "payment_method_id": leg["pay"]})


DOC_CODE = {"unavailable": ["service_not_offered"], "doc": ["final_sale"],
            "limit": ["invalid_status"]}


def _gt(spec, state) -> Dict[str, Any]:
    """{'actions': [...], 'answer': ..., 'stop': leg#|None}."""
    oid = spec["order_id"]
    for k, leg in enumerate(spec["request"]):
        out, idx = _leg_outcome(leg, state)
        if out == "exec":
            if spec.get("ask"):
                a = spec["ask"]
                if a.get("kind") == "count":
                    return {"actions": [], "stop": k,
                            "answer": {"kind": "value", "expect": str(len(idx)),
                                       "compare": "int"}}
                if a.get("kind") == "diff":
                    amt = round(sum(a["diffs"][i] for i in idx), 2)
                    return {"actions": [], "stop": k,
                            "answer": {"kind": "value", "expect": f"{amt:.2f}",
                                       "compare": "float"}}
                amt = round(sum(a["prices"][i] for i in idx) * (1.0 - a["fee"]), 2)
                return {"actions": [], "stop": k,
                        "answer": {"kind": "value", "expect": f"{amt:.2f}",
                                   "compare": "float"}}
            return {"actions": [_leg_action(leg, idx, oid)], "answer": None,
                    "stop": k}
    if spec.get("ask"):
        cmp_ = "int" if spec["ask"].get("kind") == "count" else "float"
        return {"actions": [], "stop": None,
                "answer": {"kind": "value", "expect": "0", "compare": cmp_}}
    codes: List[str] = []
    for leg in spec["request"]:
        _, kill = _leg_outcome(leg, state)
        # Every question bakes ONE code per rule (NARROW_CODES): refusals
        # are irrevocable (K=1), so a code union would be enumerable surface.
        sets = [set([rules.NARROW_CODES[r]]) if r in rules.NARROW_CODES
                else set(rules.ACCEPTED_CODES.get(r, DOC_CODE.get(r, [])))
                for r in kill]
        if not sets:
            continue
        inter = set.intersection(*sets)   # multi-cause leg: only codes valid
        codes += sorted(inter or set.union(*sets))  # for EVERY cause count
    codes = sorted(set(codes)) or ["service_not_offered"]
    return {"actions": [], "stop": None,
            "answer": {"kind": "refuse", "reasons": codes}}


def build_task(spec: Dict[str, Any], t: int, task_id: str) -> Task:
    truth_gt = _gt(spec, rules.truth(t))
    docs_gt = _gt(spec, rules.DOCS_VIEW)

    def canon(g):
        return ([(a.name, tuple(sorted(a.kwargs.get("item_ids", []) or [])),
                  a.kwargs.get("payment_method_id")) for a in g["actions"]],
                g["answer"])
    slice_ = "adapt" if canon(truth_gt) != canon(docs_gt) else "general"
    if spec.get("slice") and spec["split"] == "test":
        assert slice_ == spec["slice"], (
            f"{task_id}: planned {spec['slice']} but built {slice_} "
            f"(dead placement for {spec['family']})")

    # serving draws from SERVING_POOLS (probe pools); the test suite
    # draws from POOLS
    if spec.get("split") == "train" and spec["pool"] in SERVING_POOLS:
        pool = SERVING_POOLS[spec["pool"]]
        rank = spec.get("_seq")
        body = _pick(task_id, pool) if rank is None else pool[rank % len(pool)]
        body = body.format(**spec["fmt"])
    elif spec.get("_seq") is not None and spec["pool"] in POOLS:
        # test draws rotate by the per-family counter (even shell coverage)
        body = POOLS[spec["pool"]][spec["_seq"] % len(POOLS[spec["pool"]])]
        body = body.format(**spec["fmt"])
    else:
        body = _pick(task_id, POOLS[spec["pool"]]).format(**spec["fmt"])
    instruction = (f"Customer: {spec['first']} {spec['last']} "
                   f"(zip {spec['zip']}) writes:\n\"{body}\"")
    da = docs_gt["actions"]
    return Task(
        task_id=task_id, user_id=spec["user_id"], instruction=instruction,
        actions=truth_gt["actions"], answer=truth_gt["answer"],
        z={"family": spec["family"], "t": t, "order_id": spec["order_id"],
           "split": spec["split"], "slice": slice_,
           "window": rules.window_of(t), "tier": spec.get("tier", "L2"),
           "stop": truth_gt["stop"],
           **({"narrow": True} if spec.get("narrow") else {}),
           "docs_action": ({"name": da[0].name, "kwargs": dict(da[0].kwargs)}
                           if da else None),
           "docs_answer": (docs_gt["answer"].get("expect")
                           if docs_gt["answer"] and
                           docs_gt["answer"].get("kind") == "value" else None)})


# =============================================================================
# Phrasing pools (selected by spec["pool"] key; anchors: literal ids, stated
# reasons, ordered preferences with explicit conditionals)
# =============================================================================
POOLS: Dict[str, List[str]] = {
"cancel": [
    "Please cancel my order {oid} — I no longer need it.",
    "I'd like to cancel order {oid}. It's no longer needed.",
    "Could you cancel {oid} for me? I don't need it anymore.",
    "Hi — order {oid} is no longer needed. Please cancel it.",
    "I want to cancel my order {oid} since I no longer need it.",
    "Cancel order {oid}, please — turns out I don't need it anymore.",
],
"p9_up": [
    "I'd like to exchange the {item_name} (item {item_id}) in order {oid} for "
    "the variant with {new_opts} (item {new_item_id}). Please charge the price "
    "difference to my gift card ({pay_id}).",
    "In order {oid}, swap the {item_name} (item {item_id}) for the {new_opts} "
    "variant (item {new_item_id}); put the difference on my gift card "
    "({pay_id}).",
    "Could you exchange the {item_name} (item {item_id}) from order {oid} for "
    "item {new_item_id} ({new_opts})? Settle any difference with my gift card "
    "({pay_id}), please.",
    "Order {oid}: for my {item_name} (item {item_id}), I want the {new_opts} "
    "variant (item {new_item_id}) instead. The difference goes on my gift "
    "card ({pay_id}).",
],
"exchange": [
    "I'd like to exchange the {item_name} (item {item_id}) in my order {oid} for "
    "the variant with {new_opts} (item {new_item_id}). If there is a price "
    "difference, settle it with my credit card ({pay_id}).",
    "In order {oid}, I want to swap the {item_name} (item {item_id}) for the "
    "{new_opts} variant (item {new_item_id}). Any price difference can go on my "
    "credit card ({pay_id}).",
    "Could you exchange the {item_name} (item {item_id}) from order {oid}? I'd "
    "like the variant with {new_opts} (item {new_item_id}) instead. Charge or "
    "refund the difference to my credit card ({pay_id}).",
    "About order {oid}: please replace the {item_name} (item {item_id}) with "
    "item {new_item_id}, the one with {new_opts}. Use my credit card ({pay_id}) "
    "for any difference.",
    "Requesting an exchange on order {oid}: {item_name}, item {item_id} to item "
    "{new_item_id} ({new_opts}). Price difference to my credit card ({pay_id}).",
    "Hi, for order {oid} — can I trade the {item_name} (item {item_id}) in for "
    "the {new_opts} version (item {new_item_id})? Put any difference on my "
    "credit card ({pay_id}).",
],
"return_one": [
    'Order {oid}: the {item_name} (item {item_id}) is no longer needed — please take it back and send the refund to {dest}.',
    'Returning one thing from {oid}: the {item_name} (item {item_id}), reason: no longer needed. Refund goes to {dest}.',

    "Please return the {item_name} (item {item_id}) from my order {oid}; "
    "reason: no longer needed. Send the refund to {dest}.",
    "I'd like to return the {item_name} (item {item_id}) in order {oid} — it's "
    "no longer needed. Please issue the refund to {dest}.",
    "Could you process a return for item {item_id} (the {item_name}) from order "
    "{oid}? I no longer need it. The refund should go to {dest}.",
    "Returning the {item_name} (item {item_id}) from {oid}: no longer needed. "
    "I want the money back on {dest}.",
    "Order {oid}: I want to return item {item_id} ({item_name}) since it's no "
    "longer needed, with the refund going to {dest}.",
    "Please set up a return for the {item_name} (item {item_id}) on order {oid} "
    "— no longer needed — and send the refund to {dest}.",
],
"modify": [
    "In my pending order {oid}, please change the {item_name} (item {item_id}) "
    "to the variant with {new_opts} (item {new_item_id}). Settle any price "
    "difference with my credit card ({pay_id}).",
    "For my pending order {oid}: please swap the {item_name} (item {item_id}) "
    "to item {new_item_id} — the variant with {new_opts}. Any price difference "
    "goes on my credit card ({pay_id}).",
    "Can you update order {oid} before it ships? Change the {item_name} (item "
    "{item_id}) to the {new_opts} variant (item {new_item_id}); settle any "
    "difference with my credit card ({pay_id}).",
    "Order {oid} hasn't shipped yet — please switch the {item_name} (item "
    "{item_id}) to item {new_item_id} with {new_opts}. My credit card "
    "({pay_id}) covers any difference.",
    "Please modify pending order {oid}: replace item {item_id} ({item_name}) "
    "with item {new_item_id}, the {new_opts} version. Charge any difference to "
    "my credit card ({pay_id}).",
],
"ret_multi": [
    "From order {oid} I'm sending back: {clauses}. They're no longer needed. Refund to {dest}, please.",
    'Order {oid} — these are no longer needed: {clauses}. Process the return with the refund to {dest}.',

    "Please return these items from order {oid}: {clauses} — I no longer need "
    "them. Refund to {dest}.",
    "I'd like to return the following from my order {oid}: {clauses}. Reason: "
    "no longer needed. The refund should go to {dest}.",
    "Order {oid}: please process a return for {clauses} — no longer needed — "
    "with the money going to {dest}.",
    "Hi — from order {oid} I want to send back {clauses}; I no longer need "
    "them. Please refund {dest}.",
    "Returning from order {oid}: {clauses}. No longer needed; refund to {dest}.",
],
"exch_multi": [
    'Order {oid}: several swaps please — {clauses}. Any price difference belongs on my credit card ({pay_id}).',
    'For {oid} I need these exchanged: {clauses}. Put whatever difference there is on my credit card ({pay_id}).',
    'Multiple exchanges on order {oid}: {clauses}. My credit card ({pay_id}) takes care of any difference.',

    "In order {oid}, I'd like to exchange these items: {clauses}. Any price "
    "difference goes on my credit card ({pay_id}).",
    "About order {oid} — please swap the following: {clauses}. Settle any "
    "difference with my credit card ({pay_id}).",
    "Requesting exchanges on order {oid}: {clauses}. Price differences to my "
    "credit card ({pay_id}).",
    "Hi, for order {oid}, can I trade in {clauses}? Put any difference on my "
    "credit card ({pay_id}).",
    "Order {oid}: I want these exchanged — {clauses}. My credit card ({pay_id}) "
    "covers any difference.",
    "I'm refreshing my whole order {oid} — please swap every item: {clauses}. "
    "Any price differences go on my credit card ({pay_id}).",
    "For order {oid}, do what you can of the following: {clauses}. Anything "
    "that can't be exchanged, just leave as is; differences on my credit card "
    "({pay_id}).",
    "A few swaps for order {oid}, whichever of these are possible: {clauses}. "
    "Settle the differences with my credit card ({pay_id}).",
    "Order {oid} needs an overhaul. Exchange list: {clauses}. Put whatever "
    "difference results on my credit card ({pay_id}).",
],
"agg": [
    "Quick question about order {oid}: if I return {clauses} — all no longer "
    "needed, refund to {dest} — how much money would I get back in total? If "
    "such a return cannot be done at all, say 0. Reply with just the amount.",
    "For order {oid} — suppose I send back {clauses} (no longer needed), "
    "refunded to {dest}. What total refund would I receive after fees? Answer 0 if the "
    "return can't be done. Just the number, please.",
    "Before I decide about order {oid}: returning {clauses} with the refund to "
    "{dest} — what would the total refund come to, fees included? If it isn't possible at "
    "all, answer 0. Reply with just the amount.",
    "About my order {oid}: how much would I get back, net of any fees, if you processed a return "
    "of {clauses} to {dest}? Reply with just the number; 0 if it cannot be "
    "done.",
],
"chain": [
    # NOTE: every chain spec has FOUR legs; a template that states fewer
    # legs than the spec ships is an instruction/GT incoherence (the GT would
    # execute returns the prompt never mentions)
    "For order {oid}: exchange the {item_name} (item {item_id}) for the "
    "variant with {opts_a} (item {id_a}); if that variant can't be provided, "
    "the {opts_b} one (item {id_b}) works too — difference on my credit card "
    "({pay_id}). If no exchange is possible at all, return the item instead "
    "(no longer needed), refunding my original credit card ({pay_id}); and if "
    "the card refund can't be done, send it to my gift card ({gift_id}).",
    "About order {oid}: my first choice is exchanging the {item_name} (item "
    "{item_id}) to item {id_a} ({opts_a}); second choice item {id_b} "
    "({opts_b}), difference on my credit card ({pay_id}). If exchanging isn't "
    "possible, then return it — reason: no longer needed — to the credit card "
    "I paid with ({pay_id}), or to my gift card ({gift_id}) if the card "
    "refund fails.",
    "Order {oid}, in order of preference: 1) swap the {item_name} (item "
    "{item_id}) for item {id_a} ({opts_a}); 2) swap it for item {id_b} "
    "({opts_b}) — card ({pay_id}) covers differences; 3) return it (no longer "
    "needed) with the refund to my original card ({pay_id}); 4) same return "
    "but refund to my gift card ({gift_id}). Please do the first one that "
    "works.",
    "Hi — for order {oid} I'd like, in this order: exchange the {item_name} "
    "(item {item_id}) to the {opts_a} variant (item {id_a}); failing that, to "
    "the {opts_b} variant (item {id_b}), difference on my credit card "
    "({pay_id}); failing any exchange, a return (no longer needed) refunded "
    "to that same card ({pay_id}); and as a last resort the refund can go to "
    "my gift card ({gift_id}).",
],
"rfz": [
    "For order {oid}: exchange the {item_name} (item {item_id}) for the "
    "variant with {new_opts} (item {new_item_id}) — difference on my credit "
    "card ({pay_id}). If the exchange isn't possible, return it (no longer "
    "needed) with the refund to my original credit card ({pay_id}). Those are "
    "the only two options I'll accept.",
    "Order {oid}: either swap the {item_name} (item {item_id}) to item "
    "{new_item_id} ({new_opts}) with the difference on my credit card "
    "({pay_id}), or return it — no longer needed — refunding that same card "
    "({pay_id}). Nothing else works for me.",
    "About order {oid} — I want the {item_name} (item {item_id}) exchanged to "
    "the {new_opts} variant (item {new_item_id}), difference on my card "
    "({pay_id}); if you can't exchange, return it (no longer needed) to my "
    "original card ({pay_id}). No gift-card refunds, please.",
],
"chn_m": [
    "Order {oid} hasn't shipped yet — could you change the {item_name} (item "
    "{item_id}) to the {new_opts} version (item {new_item_id})? Charge any "
    "difference to my card ({pay_id}). If modifications aren't possible, just "
    "cancel the whole order instead — I no longer need it.",
    "About my pending order {oid}: first choice — swap the {item_name} (item "
    "{item_id}) for item {new_item_id} ({new_opts}), difference on my credit "
    "card ({pay_id}). If you can't modify it, cancel the order; no longer "
    "needed.",
    "Hi, regarding order {oid} (still pending): I'd like the {item_name} "
    "(item {item_id}) modified to the {new_opts} variant (item {new_item_id}) "
    "— card {pay_id} covers any difference. Failing that, please cancel it "
    "(no longer needed).",
],
"chn_p": [
    "For order {oid}: I'd like the {item_name} (item {item_id}) exchanged to "
    "the {new_opts} variant (item {new_item_id}) — difference on my card "
    "({pay_id}). If an exchange can't be done, return it (no longer needed) "
    "with the refund going to {dest_phrase}.",
    "Order {oid} — preference one: swap the {item_name} (item {item_id}) to "
    "item {new_item_id} ({new_opts}), card ({pay_id}) for the difference. "
    "Otherwise return it, no longer needed, refund to {dest_phrase}.",
    "Hello, on order {oid}: exchange the {item_name} (item {item_id}) for the "
    "{new_opts} version (item {new_item_id}) if you can (difference on "
    "{pay_id}); if not, a return works — no longer needed — refunded to "
    "{dest_phrase}.",
],
"rfz_m": [
    "Order {oid} is still pending — please change the {item_name} (item "
    "{item_id}) to the {new_opts} variant (item {new_item_id}), difference on "
    "my card ({pay_id}). I only want the modification; do not cancel the "
    "order or anything else.",
    "About order {oid}: swap the {item_name} (item {item_id}) to item "
    "{new_item_id} ({new_opts}) before it ships; card ({pay_id}) covers the "
    "difference. A modification is the only thing I'll accept.",
],
"rfz_x": [
    "For order {oid}: exchange the {item_name} (item {item_id}) to the "
    "variant with {opts_a} (item {id_a}); if that one doesn't work, the "
    "{opts_b} variant (item {id_b}) — difference on my card ({pay_id}). "
    "Exchange only — I don't want a return.",
    "Order {oid}: I want the {item_name} (item {item_id}) swapped, ideally to "
    "item {id_a} ({opts_a}), otherwise item {id_b} ({opts_b}); my card "
    "({pay_id}) covers differences. No returns please — exchange or nothing.",
],
"ret3_i": [
    'Order {oid}: everything goes back — {clauses} — no longer needed. The refund has to land on my original card ({pay_id}); no gift-card credit.',
    'Full return of {oid} ({clauses}), reason: no longer needed. Original card ({pay_id}) only for the refund — not a gift card.',

    "For order {oid}: I'm returning the whole order — every item: {clauses} "
    "— no longer needed. The refund must go to my original card ({pay_id}), "
    "the one I paid with. That card is the ONLY destination I'll accept.",
    "Order {oid} — sending everything back ({clauses}); reason: no longer "
    "needed. Refund the full amount to my original card ({pay_id}). No "
    "gift-card credit, please.",
    "Hi, about order {oid}: I want to return all of it — {clauses} — as no "
    "longer needed, with the money back on the card I paid with "
    "({pay_id}). Please don't offer store credit.",
    "Returning order {oid} in full: {clauses}. No longer needed. Put the "
    "refund on my original credit card ({pay_id}) — that's the only option "
    "that works for me.",
],
"exg_i": [
    'Order {oid}: exchange the entire order please — {clauses}. Differences go on my card ({pay_id}). I only want exchanges, not returns.',
    "Every item in {oid} needs swapping: {clauses}. Use my card ({pay_id}) for any price gaps. Exchange only — a refund doesn't help me.",

    "For order {oid}: I'd like the whole order exchanged — {clauses} — with "
    "any difference on my card ({pay_id}). Exchanges only; I don't want a "
    "return.",
    "Order {oid} — please swap every item: {clauses}. Card {pay_id} covers "
    "any differences. An exchange is the only thing that works for me.",
    "Hi, on order {oid} I want a full exchange: {clauses}; charge or refund "
    "the differences to {pay_id}. No returns please — exchange or nothing.",
    "Exchanging my whole order {oid}: {clauses}. Any price difference goes "
    "on my card ({pay_id}). Exchange only.",
],
"rfz_m2": [
    "My order {oid} hasn't shipped: change the {item_name} (item {item_id}) to item {new_item_id} ({new_opts}) and leave everything else; the difference goes on my credit card ({pay_id}). If you can't, leave the order untouched.",
    'Pending order {oid}: one modification — the {item_name} (item {item_id}) becomes item {new_item_id} ({new_opts}), rest unchanged, difference on my credit card ({pay_id}). Otherwise change nothing.',

    "Order {oid} is still pending — please change the {item_name} (item "
    "{item_id}) to the {new_opts} variant (item {new_item_id}); the other "
    "items stay as they are. Difference on my card ({pay_id}). I only want "
    "the modification — do not cancel the order or anything else.",
    "About my pending order {oid}: swap just the {item_name} (item "
    "{item_id}) to item {new_item_id} ({new_opts}) before it ships, keep "
    "the rest unchanged; card ({pay_id}) covers the difference. A "
    "modification is the only thing I'll accept.",
    "Hi — order {oid} hasn't shipped. Modify the {item_name} (item "
    "{item_id}) to the {new_opts} version (item {new_item_id}), leave the "
    "other items alone, difference on {pay_id}. Modification only, please.",
    "One change to pending order {oid}: the {item_name} (item {item_id}) "
    "should become item {new_item_id} ({new_opts}); everything else stays. "
    "Card {pay_id} for any difference. Do not cancel anything.",
],
"rfz_x2": [
    "For order {oid}: exchange the entire order — {clauses} — difference on "
    "my card ({pay_id}). Exchange only; I don't want a return.",
    "Order {oid}: I want every item swapped: {clauses}; my card ({pay_id}) "
    "covers differences. No returns please — exchange or nothing.",
    "Hi, whole-order exchange on {oid} please: {clauses}. Differences go on "
    "{pay_id}. An exchange is the only option I'll take.",
    "Swap my full order {oid}: {clauses}. Card {pay_id} for the "
    "differences. Exchanges only — no refunds, no store credit.",
],
"agg_x": [
    'On order {oid}: run the numbers for a full-order exchange — {clauses}. Restricting to items that are exchangeable today, what total price difference would I owe?',
    'Order {oid} question: were I to exchange everything currently exchangeable in it ({clauses}), what would the combined price difference come to?',
    "Take order {oid} and assume every eligible item gets exchanged — {clauses}. What is the total difference I'd pay, counting only what you can exchange right now?",

    "Question about order {oid}: if you exchanged every item in it that can "
    "currently be exchanged — {clauses} — how much total price difference "
    "would I end up paying? Reply with just the amount.",
    "For order {oid}, suppose a whole-order exchange: {clauses}. Counting only "
    "the items you can actually exchange right now, what is the total price "
    "difference I'd pay? Just the number, please.",
    "About my order {oid} — I'd want everything swapped: {clauses}. For the "
    "items that can currently be exchanged, how much extra would that cost me "
    "in total? Reply with just the amount.",
    "Checking on order {oid}: with a whole-order exchange request ({clauses}), "
    "what total price difference would you charge for the items that actually "
    "get exchanged under current rules? Reply with just the amount.",
    "Budget question on order {oid}: I'd refresh the whole thing — {clauses}. "
    "Only counting what you can actually exchange right now, what's the total "
    "extra cost? Reply with just the amount.",
    "Before committing on order {oid}: if I asked for all of these ({clauses}) "
    "and you did whichever are currently possible, what would the combined "
    "price difference be? Just the number, please.",
    "Thinking about upgrading order {oid} end to end: {clauses}. What total "
    "difference would the doable part cost me? Reply with just the amount.",
    "For order {oid} — full swap wishlist: {clauses}. Tally the price "
    "difference for the items that can be exchanged today and reply with just "
    "the amount.",
],
"advice_cnt": [
    "Quick question about my order {oid}: if I asked you to exchange the WHOLE "
    "order (every item, each for another variant of its product), how many of "
    "the items would actually get exchanged under current rules? Reply with "
    "just the number.",
    "For order {oid} — suppose I request a whole-order exchange, every item "
    "swapped to another variant. How many items would you actually be able to "
    "exchange? Please reply with just the number.",
    "Checking on order {oid}: on a whole-order exchange request (all items "
    "listed), what number of items would end up exchanged under the current "
    "rules? Reply with just the number.",
],
}


# SERVING-ONLY shells: the frozen test suite never shares a phrasing with
# the serving stream. Same variable schemas as POOLS, zero textual overlap.
SERVING_POOLS: Dict[str, List[str]] = {
"p9_up": [
    "Hi — for order {oid}, exchange my {item_name} (item {item_id}) to the "
    "{new_opts} variant (item {new_item_id}) and take the difference from my "
    "gift card ({pay_id}).",
    "Exchange request on {oid}: item {item_id} (the {item_name}) to item "
    "{new_item_id} ({new_opts}). Cover the extra cost with my gift card "
    "({pay_id}).",
    "Please swap the {item_name} (item {item_id}) in {oid} for item "
    "{new_item_id} ({new_opts}); my gift card ({pay_id}) should pay the "
    "difference.",
    "Regarding order {oid}: the {item_name} (item {item_id}) should become "
    "item {new_item_id} ({new_opts}), difference charged to my gift card "
    "({pay_id}).",
],
"rfz": [
    "About order {oid}: exchange the {item_name} (item {item_id}) for item {new_item_id} ({new_opts}), with any difference on my credit card ({pay_id}). If that's impossible, I'm out of options.",
    "Order {oid}: the only outcome I'll accept is exchanging the {item_name} (item {item_id}) to item {new_item_id} ({new_opts}), difference to my credit card ({pay_id}). Nothing else works.",
    "For {oid}: swap the {item_name} (item {item_id}) to item {new_item_id} ({new_opts}) — my credit card ({pay_id}) covers the gap — or nothing.",
    "One acceptable path for order {oid}: exchange item {item_id} (the {item_name}) for item {new_item_id} ({new_opts}), difference on my credit card ({pay_id}). That's it.",
],

"advice_cnt": [
    "Question on order {oid}: if I asked to exchange the whole order, item by item, how many items would actually go through under current rules? Just the number.",
    "For {oid} — a full-order exchange, every item to another variant: how many would you actually be able to exchange? Number only, please.",
    "Checking order {oid}: were I to request a whole-order exchange, what count of items ends up exchanged? Reply with just the number.",
],
"exch_multi": [
    'Order {oid} needs several swaps: {clauses}. Whatever the price difference is, bill my credit card ({pay_id}).',
    'Hi — multiple exchanges on {oid} please: {clauses}. Settle any difference via my credit card ({pay_id}).',
    "I'd like to exchange a few things in order {oid}: {clauses}. Difference to my credit card ({pay_id}) is fine.",
    'Several items in {oid} need different variants: {clauses}. Put any price change on my credit card ({pay_id}).',
    'Exchange request, order {oid}, more than one item: {clauses}. My credit card ({pay_id}) covers the difference.',
    'Can you process these swaps for order {oid}? {clauses}. Charge or refund the difference on my credit card ({pay_id}).',
    'For {oid}, please switch out the following: {clauses}. Any gap in price goes to my credit card ({pay_id}).',
    'Batch exchange on my order {oid}: {clauses}. Use my credit card ({pay_id}) for whatever difference results.',
],
"chain": [
    "About order {oid}: swap the {item_name} (item {item_id}) for the {opts_a} variant (item {id_a}) — any difference on my credit card ({pay_id}); if that's not possible, the {opts_b} one (item {id_b}); and if neither works, return it — no longer needed — with the refund to my gift card ({gift_id}).",
    'Plan for {oid}: first choice, exchange the {item_name} (item {item_id}) to item {id_a} ({opts_a}), difference to my credit card ({pay_id}); second choice, item {id_b} ({opts_b}); last resort, return it as no longer needed and refund my gift card ({gift_id}).',
    'Order {oid}, in order of preference: the {item_name} (item {item_id}) becomes item {id_a} with {opts_a} — my credit card ({pay_id}) covers differences; failing that, item {id_b} with {opts_b}; failing both, just return it (no longer needed), refund to gift card {gift_id}.',
    "Here's what I want for {oid}: try exchanging the {item_name} (item {item_id}) for item {id_a} ({opts_a}), differences on my credit card ({pay_id}). Can't? Then item {id_b} ({opts_b}). Still no? Return it — no longer needed — refund to my gift card ({gift_id}).",
    'For my order {oid}: about the {item_name} (item {item_id}) — preference one is item {id_a} ({opts_a}), price gap to my credit card ({pay_id}); preference two is item {id_b} ({opts_b}); otherwise return it as no longer needed with the refund to my gift card ({gift_id}).',
    'Ranked request on {oid}: exchange {item_name} (item {item_id}) to {opts_a} (item {id_a}) with my credit card ({pay_id}) for differences, else to {opts_b} (item {id_b}), else return it (no longer needed) and credit my gift card ({gift_id}).',
],
"agg": [
    "Question about order {oid}: if I sent back {clauses} — all no longer needed — with the refund to {dest}, what total would I get? If it can't be done at all, say 0. Just the number.",
    'Before deciding on {oid}: returning {clauses} (no longer needed), refund to {dest} — how much comes back after fees? Answer 0 if the return is impossible; number only.',
    "Could you total it up? Order {oid}, returning {clauses} as no longer needed, refund to {dest}. Reply with just the amount — 0 if it can't happen.",
    'What would I get back on {dest} for returning {clauses} from order {oid}? All no longer needed. Just the figure; 0 if not possible.',
],
"cancel": [
    'Hey, I need order {oid} cancelled — I no longer need it.',
    "Please drop order {oid} from my account; it's no longer needed.",
    'Change of plans: order {oid} is no longer needed. Cancel it, please.',
    'Can you cancel order {oid}? No longer needed on my end.',
    "I'm requesting cancellation of {oid} — the items are no longer needed.",
    'Order {oid} — please cancel. Reason: no longer needed.',
    'Hello! Could you cancel my recent order {oid}? I no longer need it, thanks.',
    'Turns out we no longer need order {oid}. Please cancel it.',
],
"exchange": [
    'For my order {oid}, swap the {item_name} (item {item_id}) to item {new_item_id} — the {new_opts} one. Bill any difference to my credit card ({pay_id}).',
    'Exchange request on {oid}: the {item_name}, item {item_id}, for the variant with {new_opts} (item {new_item_id}). Difference on my credit card ({pay_id}) is fine.',
    "Hi! Order {oid} — I'd rather have the {new_opts} version of the {item_name}. That's item {item_id} going out, item {new_item_id} coming in. My credit card ({pay_id}) can cover any difference.",
    'Can we do an exchange? Order {oid}, the {item_name} (item {item_id}) — I want item {new_item_id} with {new_opts} instead. Settle whatever difference there is via my credit card ({pay_id}).',
    'I picked the wrong variant on {oid}. Please exchange the {item_name} (item {item_id}) for item {new_item_id} ({new_opts}); price difference to my credit card ({pay_id}).',
    'About the {item_name} in order {oid} — item {item_id} — could it become the {new_opts} variant, item {new_item_id}? Use my credit card ({pay_id}) for any gap in price.',
    'Order {oid}: exchange item {item_id} (the {item_name}) to item {new_item_id}, {new_opts}. Card ({pay_id}) for the difference, thanks.',
    'Hello — hoping to trade the {item_name} (item {item_id}) on order {oid} for the {new_opts} version, item {new_item_id}. Any price change can hit my credit card ({pay_id}).',
    'Wrong pick on my part: in {oid}, the {item_name} (item {item_id}) should be the {new_opts} variant (item {new_item_id}). Difference on the credit card ({pay_id}) please.',
    'Please arrange an exchange for order {oid}. Outgoing: {item_name}, item {item_id}. Incoming: item {new_item_id} with {new_opts}. Balance any difference to my credit card ({pay_id}).',
    'Is it possible to switch the {item_name} (item {item_id}) in {oid} over to item {new_item_id} — the one with {new_opts}? Credit card ({pay_id}) for whatever the difference is.',
    'Exchange for {oid} please: {item_name} item {item_id} out, item {new_item_id} ({new_opts}) in. Put the difference, if any, on my credit card ({pay_id}).',
],
"return_one": [
    'I want to send back the {item_name} (item {item_id}) from order {oid} — no longer needed. Refund to {dest}, please.',
    'Return request: order {oid}, item {item_id} (the {item_name}). Reason is no longer needed; put the refund on {dest}.',
    'Hi, the {item_name} from {oid} — item {item_id} — is no longer needed. Could you take it back and refund {dest}?',
    "Please start a return on order {oid} for the {item_name} (item {item_id}). It's no longer needed. Refund destination: {dest}.",
    'No longer needed: the {item_name}, item {item_id}, from my order {oid}. Send the money back to {dest}.',
    "Could I return item {item_id} — the {item_name} — off order {oid}? No longer needed. I'd like the refund on {dest}.",
    "Hello, I'm returning the {item_name} (item {item_id}) from {oid} since it's no longer needed. Refund should land on {dest}.",
    'One return for order {oid}: {item_name}, item {item_id}, no longer needed. Credit the refund to {dest}.',
],
"modify": [
    "Order {oid} hasn't gone out yet, right? Please change the {item_name} (item {item_id}) to item {new_item_id} — the {new_opts} one. Difference to my credit card ({pay_id}).",
    'Before {oid} ships: swap the {item_name}, item {item_id}, for item {new_item_id} with {new_opts}. Bill any difference to my credit card ({pay_id}).',
    'Modification for pending order {oid}: item {item_id} ({item_name}) should be item {new_item_id}, the {new_opts} variant. My credit card ({pay_id}) for any difference.',
    'Hi — my order {oid} is still pending. Can the {item_name} (item {item_id}) become the {new_opts} version (item {new_item_id})? Difference on my credit card ({pay_id}).',
    'Please adjust order {oid} before shipment. The {item_name} (item {item_id}) needs to be item {new_item_id} — {new_opts}. Use my credit card ({pay_id}) for the difference.',
    'Quick change on pending order {oid}: {item_name} item {item_id} to item {new_item_id} ({new_opts}). Settle the difference with my credit card ({pay_id}).',
],
"ret_multi": [
    'A few things from order {oid} are coming back: {clauses}. No longer needed — refund to {dest}.',
    'Please return these from {oid}: {clauses}. All no longer needed; the refund goes to {dest}.',
    "I'm sending back several items in order {oid} — {clauses} — no longer needed. Refund {dest}.",
    'Multiple returns on {oid}: {clauses}. Reason: no longer needed. Please credit {dest}.',
    "From order {oid} I want to return {clauses} — they're no longer needed. Money back to {dest}.",
    'Return request for {oid}, several items: {clauses}, no longer needed. Refund destination: {dest}.',
    'These items in {oid} are no longer needed: {clauses}. Take them back and refund {dest}.',
    'Order {oid} — returning {clauses} as no longer needed, refund to {dest} please.',
],
}

# =============================================================================
# Families, quotas, serving
# =============================================================================
# Probe families: one per boundary SIDE (single-op form). The slice derives
# automatically per window: blocked side -> adapt; vacated/never-blocked ->
# general (the paranoid/stale punishment surface).
PROBES = [
    "p_aud_exch", "p_cam_exch", "p_soft_exch",
    "p_gift_cancel", "p_split_cancel", "p_card_cancel",
    "p_amex_ret", "p_visa_ret", "p_gift_ret",
    "p_fl_exch", "p_tx_exch",
    "p_split_mod", "p_card_mod",
    "p_part_ret", "p_part_ret_el", "p_full_ret",
    "p_part_exch", "p_pobox_exch",
]

ARCH = ["mxa", "mxr", "mxr_el", "mxe", "agg", "agg_el", "agg_cnt", "agg_x", "ret2", "exch2",
        "chn_au", "chn_av", "chn_cam", "chn_s", "chn_tx", "rfz"]
GENERAL_FAMS = ["gl_agg", "gl_mx", "gl_chn", "gl_limit", "gl_advice"]
FAMILIES = PROBES + ARCH + GENERAL_FAMS

# planned slots: ADAPT_TOTALS / GENERAL_TOTALS; the BUILT slice is derived
# per question (truth-vs-docs), so a planned-adapt placement whose
# truth==docs lands general — final cell counts come from the built suite,
# not from these anchors.
# p9_up (permissive carry-out) runs in every window as the counter to
# blanket refusal; W5/W6 get the heavy share (their adapt slice is otherwise
# almost all refusals).
ADAPT_TOTALS = {0: 29, 1: 51, 2: 51, 3: 51, 4: 51, 5: 42, 6: 37}
GENERAL_TOTALS = {w: (24 if w == 0 else (22 if w >= 5 else 21))
                  for w in range(7)}

# archetype variants are placed only where their SHAPE is doc-mismatched:
# W0 has just the static anchors; scope windows pick matching variants
# (e.g. mxr soft-only is adapt under off_multi (W1-3) but general under
# off_multi_elec (W4-6), where mxr_el takes over).
# adapt chains capped at ~5% (they measure execution, not inference);
# multi-policy single-item forms (ret2 R6xR3, exch2 R1xR4) carry joint
# hypothesis load; answers stay the un-guessable mass.
ARCH_ADAPT = {
    0: {"rfz": 1, "p9_up": 5},
    1: {"p9_up": 5, "mxa": 24, "mxr": 1, "mxe": 4, "agg_x": 9, "agg_cnt": 1, "chn_au": 2,
        "ret2": 2},
    2: {"p9_up": 5, "mxa": 25, "mxr": 2, "mxe": 2, "agg_x": 10, "agg_cnt": 1, "chn_av": 2},
    3: {"p9_up": 5, "mxa": 25, "mxr": 2, "mxe": 2, "agg_x": 10, "agg_cnt": 1, "chn_au": 2},
    4: {"p9_up": 5, "mxa": 25, "mxr_el": 2, "mxe": 2, "agg_x": 10, "agg_cnt": 1,
        "chn_av": 1, "exch2": 1},
    5: {"p9_up": 10, "mxr_el": 2, "mxe": 2, "agg_el": 3, "chn_tx": 1,
        "ret2": 2, "rfz": 2},
    6: {"p9_up": 7, "mxr_el": 2, "agg_el": 4, "chn_tx": 1, "ret2": 2, "rfz": 1},
}

# Extra W0 adapt bulk (static anchors R5/R8 only — the sole hidden rules
# live at W0). Appended AFTER the main generation pass with dedicated
# families and counters, so the main pass's user cursor / family counters /
# plan-shuffle RNG are consumed unchanged. No value questions here: R5/R8
# are order-level, all-blocked values would be enumerable zeros.
W0_EXTRA = {"chn_m": 5, "chn_p": 4, "rfz_m": 3, "rfz_x": 2}

# Per-window rule-balance bulk, appended AFTER the W0_EXTRA pass (same
# state-isolation discipline). Rule targets (headline cells >=5): R6 -> 50,
# R7 -> 42, R3 -> 42, R4 -> 30, R5 -> 38, R8 -> 37. Single-leg families:
# ret3 (R3, any-brand hosts W2-3) / ret3_ax (R3, amex hosts W4-6) / exg_fl
# (R4, FL hosts W3-4) / exg_tx (R4, TX hosts W5-6); each alternates an
# insist-refusal form and a fallback-executes form (chain share control).
# Single-fallback forms count as menu chains (<=5% cap), so every addition
# is a single-leg form on a MULTI-item host (single-item hosts capped <20%
# of adapt). Order-level rules' adapt side is therefore refusal-shaped by
# construction — resistance to blind refusal is measured (baseline bots +
# General co-report), not prevented.
V75_EXTRA = {
    1: {"mxr": 5, "rfz_m": 5, "rfz_x": 5},
    2: {"mxr": 6, "mxe": 8, "ret3": 6, "rfz_m": 5, "rfz_x": 5},
    3: {"mxr": 6, "mxe": 8, "ret3": 8, "exg_fl": 8, "rfz_m": 5, "rfz_x": 5},
    4: {"mxr_el": 6, "mxe": 9, "ret3_ax": 8, "exg_fl": 7,
        "rfz_m": 5, "rfz_x": 5},
    5: {"mxr_el": 2, "mxe": 9, "ret3_ax": 5, "exg_tx": 6,
        "rfz_m": 5, "rfz_x": 3},
    6: {"mxr_el": 1, "ret3_ax": 5, "exg_tx": 6, "rfz_m": 5, "rfz_x": 4},
}

# PHRASE-POOL TWINS (pool-answer degeneracy defense): the V75_EXTRA
# families' pools appear on OPEN-side hosts as general — same phrasing,
# opposite outcome — so a pool->answer memorizer pays for every gain.
# ret3_t only in windows where an original-card refund EXECUTES.
V76_TWINS = {
    0: {"exg_t": 2, "rfz_mt": 2, "rfz_xt": 2, "ret3_t": 3},
    1: {"exg_t": 2, "rfz_mt": 2, "rfz_xt": 2, "ret3_t": 3},
    2: {"exg_t": 2, "rfz_mt": 2, "rfz_xt": 2},
    3: {"exg_t": 2, "rfz_mt": 2, "rfz_xt": 2},
    4: {"exg_t": 2, "rfz_mt": 2, "rfz_xt": 2, "ret3_t": 3},
    5: {"exg_t": 2, "rfz_mt": 2, "rfz_xt": 2, "ret3_t": 3},
    6: {"exg_t": 2, "rfz_mt": 2, "rfz_xt": 2, "ret3_t": 3},
}

# Vacated-side probe top-up (general slice, tier L1 — attribution only,
# not headline). Every narrow/migrate/revert event keeps >=2 open-side
# probes per subsequent window.
PROBE_EXTRA = {
    3: {"p_cam_exch": 1},
    4: {"p_cam_exch": 1, "p_visa_ret": 1, "p_gift_cancel": 1},
    5: {"p_cam_exch": 2, "p_visa_ret": 2, "p_gift_cancel": 2},
    6: {"p_cam_exch": 1, "p_visa_ret": 2, "p_gift_cancel": 1},
}
GEN_QUOTA = {
    0: {"gl_agg": 4, "gl_mx": 4, "gl_chn": 4, "gl_limit": 3, "gl_advice": 3,
        "chn_s": 2, "mxr": 1, "agg": 1, "p_cam_exch": 1, "p_visa_ret": 1},
    1: {"gl_agg": 4, "gl_mx": 4, "gl_chn": 4, "gl_limit": 2, "gl_advice": 2,
        "p_visa_ret": 1, "p_gift_cancel": 1, "p_tx_exch": 1, "chn_s": 2},
    2: {"gl_agg": 4, "gl_mx": 4, "gl_chn": 4, "gl_limit": 2, "gl_advice": 2,
        "p_tx_exch": 1, "p_card_cancel": 1, "p_gift_ret": 1, "p_full_ret": 1,
        "p_card_mod": 1},
    3: {"gl_agg": 4, "gl_mx": 4, "gl_chn": 4, "gl_limit": 2, "gl_advice": 2,
        "p_cam_exch": 1, "p_tx_exch": 1, "chn_s": 2, "p_full_ret": 1},
    4: {"gl_agg": 4, "gl_mx": 4, "gl_chn": 4, "gl_limit": 2, "gl_advice": 2,
        "p_cam_exch": 1, "p_visa_ret": 1, "p_gift_cancel": 1, "p_part_ret": 1,
        "p_gift_ret": 1},
    5: {"gl_agg": 4, "gl_mx": 4, "gl_chn": 4, "gl_limit": 2, "gl_advice": 2,
        "p_aud_exch": 1, "p_fl_exch": 2, "chn_s": 2, "p_full_ret": 1},
    6: {"gl_agg": 4, "gl_mx": 4, "gl_chn": 4, "gl_limit": 2, "gl_advice": 2,
        "p_aud_exch": 1, "p_cam_exch": 1, "p_fl_exch": 2, "p_gift_cancel": 1,
        "p_part_exch": 1},
}

PROBE_TRIGGER = {
    "p_aud_exch": lambda s: s["R1_exch_category"] in ("off_full", "off_audio"),
    "p_cam_exch": lambda s: s["R1_exch_category"] == "off_full",
    "p_soft_exch": lambda s: False,
    "p_gift_cancel": lambda s: s["R2_cancel_payment"] == "off_gift",
    "p_split_cancel": lambda s: s["R2_cancel_payment"] == "off_split",
    "p_card_cancel": lambda s: False,
    "p_amex_ret": lambda s: s["R3_refund_original"] in ("off_all", "off_amex"),
    "p_visa_ret": lambda s: s["R3_refund_original"] == "off_all",
    "p_gift_ret": lambda s: False,
    "p_fl_exch": lambda s: s["R4_exch_geo"] == "off_fl",
    "p_tx_exch": lambda s: s["R4_exch_geo"] == "off_tx",
    "p_split_mod": lambda s: True,
    "p_card_mod": lambda s: False,
    "p_part_ret": lambda s: s["R6_partial_return"] == "off_multi",
    "p_part_ret_el": lambda s: s["R6_partial_return"] != "on",
    "p_full_ret": lambda s: False,
    "p_part_exch": lambda s: s["R7_partial_exchange"] == "off",
    "p_pobox_exch": lambda s: True,
}


def test_plan(w: int) -> List[Tuple[str, str]]:
    t0, _ = rules.window_range(w)
    s = rules.truth(t0)
    plan: List[Tuple[str, str]] = []
    arch = ARCH_ADAPT[w]
    for f, n in arch.items():
        plan += [(f, "adapt")] * n
    need = ADAPT_TOTALS[w] - len(plan)
    assert need >= 0, f"W{w}: archetype quota exceeds window total"
    sides = [p for p in PROBES if PROBE_TRIGGER[p](s)]
    assert sides, f"W{w}: no blocked probe side"
    for i in range(need):
        plan.append((sides[i % len(sides)], "adapt"))
    gen = GEN_QUOTA[w]
    for f, n in gen.items():
        if f.startswith("p_"):
            assert not PROBE_TRIGGER[f](s), \
                f"W{w}: {f} placed as general but its side is blocked at t={t0}"
        plan += [(f, "general")] * n
    n_gen = sum(1 for _, sl in plan if sl == "general")
    assert n_gen == GENERAL_TOTALS[w], f"W{w}: general quota {n_gen}"
    assert len(plan) == ADAPT_TOTALS[w] + GENERAL_TOTALS[w], f"W{w} size"
    return plan


def slot_sequence(n: int = rules.TIMELINE_LEN, seed: int = 11):
    """Serving slots: every probe side sampled every block (boundary mirrors:
    both sides of every boundary appear continuously), plus 4 density-boost
    slots rotating over currently-blocked sides.

    KNOWN LEAK (documented, accepted): the density boost makes request-type
    FREQUENCY correlate with the hidden state — a system aware of this
    generator could infer drift without rewards. Intended as realism (policy
    changes spike related tickets)."""
    out, t, b = [], 1, 0
    a_i = 0
    while t <= n:
        s = rules.truth(t)
        slots = list(PROBES)
        hot = [p for p in PROBES if PROBE_TRIGGER[p](s)]
        for _ in range(4):
            slots.append(hot[a_i % len(hot)]); a_i += 1
        random.Random(seed + b).shuffle(slots)
        for fam in slots:
            if t > n:
                break
            out.append((t, fam))
            t += 1
        b += 1
    return out


def serving_sequence(registry) -> List[Task]:
    # outcome-aware GLOBAL rotation per (pool, outcome): every serving shell
    # meets every outcome its pool produces (plain md5 picks can starve a
    # shell of one outcome). Test stamping is separate.
    cnt: Dict[tuple, int] = {}
    for sp in registry["train"]:
        g = _gt(sp, rules.truth(sp["t"]))
        a = g["answer"] or {}
        kind = a.get("kind") or "exec"
        if kind == "value":
            kind = "zero" if float(a.get("expect", 1) or 0) == 0 else "nonzero"
        k = (sp["pool"], kind)
        sp["_seq"] = cnt.get(k, 0)
        cnt[k] = sp["_seq"] + 1
    return [build_task(spec, spec["t"], f"l4_t{spec['t']:04d}_{spec['family']}")
            for spec in registry["train"]]


def test_set(registry, w: int) -> List[Task]:
    t0, _ = rules.window_range(w)
    return [build_task(spec, t0, f"l4_test_w{w}_{spec['family']}_{i}")
            for i, spec in enumerate(registry["test"][w])]
