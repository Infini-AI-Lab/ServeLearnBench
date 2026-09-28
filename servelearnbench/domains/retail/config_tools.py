"""Config-driven write tools: behavior derives from the stage PolicyConfig.

Rule interactions live here: membership-tier fee/threshold tables,
category return windows, tier-dependent refund destinations, clearance
(final-sale) exclusions. Money flow is fully materialized in the DB state so
the reward hash verifies every cent. Errors are neutral strings.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ...engine.tool import Tool
from .policy_config import TODAY, days_between
from .tools.place_order import PlaceOrder


def _tier(user: Dict[str, Any]) -> str:
    return user.get("membership", "bronze")


def _product_of(products: Dict[str, Any], item_id: str) -> Optional[Dict[str, Any]]:
    for p in products.values():
        if item_id in p["variants"]:
            return p
    return None


def _dest_allowed(cfg, user, order, payment_method_id) -> bool:
    """Refund destination rule: gift card always; original method for allowed tiers."""
    pm = user["payment_methods"].get(payment_method_id)
    if pm is None:
        return False
    if pm["source"] == "gift_card":
        return True
    original = order["payment_history"][0]["payment_method_id"]
    return _tier(user) in cfg["refund_original_tiers"] and payment_method_id == original


def _settle(user, payment_method_id, amount, order, kind) -> Optional[str]:
    """Apply a signed money movement; returns an error string or None.
    amount > 0 => charge the method; amount < 0 => refund to the method."""
    pm = user["payment_methods"][payment_method_id]
    if pm["source"] == "gift_card":
        if amount > 0 and pm["balance"] < amount:
            return (f"Error: insufficient gift card balance "
                    f"(balance ${pm['balance']:.2f} < ${amount:.2f})")
        pm["balance"] = round(pm["balance"] - amount, 2)
    order["payment_history"].append({
        "transaction_type": "payment" if amount > 0 else "refund",
        "amount": round(abs(amount), 2), "payment_method_id": payment_method_id})
    return None


def make_cancel_tool(cfg: Dict[str, Any]):
    class CancelPendingOrderCfg(Tool):
        @staticmethod
        def invoke(data: Dict[str, Any], order_id: str, reason: str) -> str:
            orders = data["orders"]
            if order_id not in orders:
                return "Error: order not found"
            order = orders[order_id]
            if order["status"] != "pending":
                return "Error: non-pending order cannot be cancelled"
            if reason not in cfg["cancel_reasons"]:
                return "Error: invalid reason"
            user = data["users"][order["user_id"]]
            # Signed ledger: per-method net = payments - refunds. On a pristine
            # order this equals the payment sum; after a cheaper modification it
            # reflects what the customer actually still paid, so cancelling
            # refunds each method's OUTSTANDING balance exactly once.
            net: Dict[str, float] = {}
            for p in order["payment_history"]:
                sign = 1 if p["transaction_type"] == "payment" else -1
                pid = p["payment_method_id"]
                net[pid] = round(net.get(pid, 0.0) + sign * p["amount"], 2)
            total = round(sum(net.values()), 2)
            thr_table = cfg["cancel_max_total"]
            if thr_table is not None:
                thr = thr_table[_tier(user)]
                if total > thr:
                    return (f"Error: orders with a total above ${thr:.0f} cannot be "
                            f"cancelled for this account (this order: ${total:.2f})")
            refunds = []
            for pid, bal in net.items():
                if bal <= 0:
                    continue
                refunds.append({"transaction_type": "refund", "amount": bal,
                                "payment_method_id": pid})
                if "gift_card" in pid:
                    pm = user["payment_methods"][pid]
                    pm["balance"] = round(pm["balance"] + bal, 2)
            order["status"] = "cancelled"
            order["cancel_reason"] = reason
            order["payment_history"].extend(refunds)
            return json.dumps(order)

        @staticmethod
        def get_info() -> Dict[str, Any]:
            return {"type": "function", "function": {
                "name": "cancel_pending_order",
                "description": "Cancel an order. See the policy and workflow documents for the rules and procedure.",
                "parameters": {"type": "object", "properties": {
                    "order_id": {"type": "string", "description": "The order id, e.g. '#W0000000'."},
                    "reason": {"type": "string", "description": "Cancellation reason (see policy for valid values)."},
                }, "required": ["order_id", "reason"]}}}

    return CancelPendingOrderCfg


def _eligibility_error(cfg, data, order, item_ids, op: str) -> Optional[str]:
    """Shared clearance + window checks for returns/exchanges (per item)."""
    products = data["products"]
    age = days_between(order["delivered_at"], TODAY) if "delivered_at" in order else None
    for iid in item_ids:
        p = _product_of(products, iid)
        if p is None:
            return "Error: some item not found"
        if p["variants"][iid].get("clearance") and not (
                op == "returned" and cfg.get("clearance_returnable")):
            # clearance_returnable (config flag, default off): the RETURN tool
            # becomes mechanically permissive on clearance; whether honoring the
            # request is correct is decided by rules.truth(t), not the tool.
            return f"Error: item {iid} is a clearance item (final sale) and cannot be {op}"
        w = cfg["return_window_days"][p["category"]]
        if (age is None or age > w) and not (
                op == "returned" and cfg.get("window_returnable")):
            # window_returnable (config flag, default off): the RETURN tool stops
            # enforcing the category window; whether honoring a late return is
            # correct is decided by rules.truth(t), not the tool.
            return (f"Error: the {p['category']} {op[:-1]} window ({w} days after delivery) "
                    f"has passed for this order")
    return None


def make_return_tool(cfg: Dict[str, Any]):
    class ReturnDeliveredOrderItemsCfg(Tool):
        @staticmethod
        def invoke(data: Dict[str, Any], order_id: str, item_ids: List[str],
                   payment_method_id: str, reason: str) -> str:
            if not item_ids:
                return "Error: item_ids must not be empty"
            orders = data["orders"]
            if order_id not in orders:
                return "Error: order not found"
            order = orders[order_id]
            if order["status"] != "delivered":
                return "Error: non-delivered order cannot be returned"
            if reason not in (cfg["return_reasons"] or []):
                return "Error: invalid or missing return reason"
            user = data["users"][order["user_id"]]
            all_item_ids = [it["item_id"] for it in order["items"]]
            for item_id in item_ids:
                if item_ids.count(item_id) > all_item_ids.count(item_id):
                    return "Error: some item not found"
            err = _eligibility_error(cfg, data, order, item_ids, "returned")
            if err:
                return err
            if not _dest_allowed(cfg, user, order, payment_method_id):
                return "Error: refund destination not permitted for this account"
            gross = sum(next(it["price"] for it in order["items"] if it["item_id"] == iid)
                        for iid in item_ids)
            fee = cfg["return_fee"][_tier(user)]
            refund = round(gross * (1.0 - fee), 2)
            order["status"] = "return requested"
            order["return_items"] = sorted(item_ids)
            order["return_reason"] = reason
            order["return_refund_amount"] = refund
            # Documented L2+ policy (config flag split_refund_proportional):
            # an original-method refund on an order paid with MULTIPLE methods is
            # split across those methods in proportion to what each paid (a
            # gift-card destination still takes the full amount).
            pm = user["payment_methods"][payment_method_id]
            paid: Dict[str, float] = {}
            for p in order["payment_history"]:
                if p["transaction_type"] == "payment":
                    paid[p["payment_method_id"]] = round(
                        paid.get(p["payment_method_id"], 0.0) + p["amount"], 2)
            if cfg.get("split_refund_proportional") and \
                    pm["source"] != "gift_card" and len(paid) > 1:
                total_paid = sum(paid.values())
                pids = list(paid)
                allotted = 0.0
                for k, pid in enumerate(pids):
                    part = (round(refund - allotted, 2) if k == len(pids) - 1
                            else round(refund * paid[pid] / total_paid, 2))
                    allotted = round(allotted + part, 2)
                    _settle(user, pid, -part, order, "refund")
            else:
                _settle(user, payment_method_id, -refund, order, "refund")
            return json.dumps(order)

        @staticmethod
        def get_info() -> Dict[str, Any]:
            return {"type": "function", "function": {
                "name": "return_delivered_order_items",
                "description": "Return items of an order. See the policy and workflow documents for the rules, fees, windows, and procedure.",
                "parameters": {"type": "object", "properties": {
                    "order_id": {"type": "string", "description": "The order id, e.g. '#W0000000'."},
                    "item_ids": {"type": "array", "minItems": 1, "items": {"type": "string"}, "description": "The item ids to return."},
                    "payment_method_id": {"type": "string", "description": "Refund destination payment method id."},
                    "reason": {"type": "string", "description": "Return reason (see policy for valid values)."},
                }, "required": ["order_id", "item_ids", "payment_method_id", "reason"]}}}

    return ReturnDeliveredOrderItemsCfg


def _swap_validate(data, order, item_ids, new_item_ids):
    """Common validation for exchange/modify pairs; returns (err, diff)."""
    products = data["products"]
    if len(item_ids) != len(new_item_ids):
        return "Error: item_ids and new_item_ids must have the same length", 0.0
    all_item_ids = [it["item_id"] for it in order["items"]]
    diff = 0.0
    for item_id, new_item_id in zip(item_ids, new_item_ids):
        if item_ids.count(item_id) > all_item_ids.count(item_id):
            return "Error: some item not found", 0.0
        item = next(it for it in order["items"] if it["item_id"] == item_id)
        product = products.get(item["product_id"])
        if product is None or new_item_id not in product["variants"]:
            return "Error: new item is not a variant of the same product", 0.0
        variant = product["variants"][new_item_id]
        if not variant["available"]:
            return "Error: new item is not available", 0.0
        if variant.get("clearance"):
            return "Error: new item is a clearance item (final sale) and cannot be selected", 0.0
        diff += variant["price"] - item["price"]
    return None, round(diff, 2)


def make_exchange_tool(cfg: Dict[str, Any]):
    class ExchangeDeliveredOrderItemsCfg(Tool):
        @staticmethod
        def invoke(data: Dict[str, Any], order_id: str, item_ids: List[str],
                   new_item_ids: List[str], payment_method_id: str) -> str:
            if not item_ids:
                return "Error: item_ids must not be empty"
            orders = data["orders"]
            if order_id not in orders:
                return "Error: order not found"
            order = orders[order_id]
            if order["status"] != "delivered":
                return "Error: non-delivered order cannot be exchanged"
            user = data["users"][order["user_id"]]
            pm = user["payment_methods"].get(payment_method_id)
            if pm is None:
                return "Error: payment method not found"
            err = _eligibility_error(cfg, data, order, item_ids, "exchanged")
            if err:
                return err
            err, diff = _swap_validate(data, order, item_ids, new_item_ids)
            if err:
                return err
            if diff > 0:
                original = order["payment_history"][0]["payment_method_id"]
                if not cfg.get("exchange_any_method") and \
                        pm["source"] != "gift_card" and payment_method_id != original:
                    return "Error: the price difference must be paid via the original payment method or a gift card"
                err = _settle(user, payment_method_id, diff, order, "payment")
                if err:
                    return err
            elif diff < 0:
                if not _dest_allowed(cfg, user, order, payment_method_id):
                    return "Error: refund destination not permitted for this account"
                _settle(user, payment_method_id, diff, order, "refund")
            order["status"] = "exchange requested"
            # MULTISET storage (tau-bench heritage): the two lists are sorted
            # independently and carry NO positional pairing — pairing was
            # validated pre-sort (each new id a variant of its paired item's
            # product) and is uniquely recoverable per product. Do not read
            # these columns as index-aligned pairs.
            order["exchange_items"] = sorted(item_ids)
            order["exchange_new_items"] = sorted(new_item_ids)
            order["exchange_payment_method_id"] = payment_method_id
            order["exchange_price_difference"] = diff
            return json.dumps(order)

        @staticmethod
        def get_info() -> Dict[str, Any]:
            return {"type": "function", "function": {
                "name": "exchange_delivered_order_items",
                "description": "Exchange items of an order for other variants of the same products. See the policy and workflow documents for the rules and procedure.",
                "parameters": {"type": "object", "properties": {
                    "order_id": {"type": "string", "description": "The order id, e.g. '#W0000000'."},
                    "item_ids": {"type": "array", "minItems": 1, "items": {"type": "string"}, "description": "The item ids to exchange away."},
                    "new_item_ids": {"type": "array", "minItems": 1, "items": {"type": "string"}, "description": "The replacement item ids, same length/order as item_ids."},
                    "payment_method_id": {"type": "string", "description": "Payment method for settling the price difference."},
                }, "required": ["order_id", "item_ids", "new_item_ids", "payment_method_id"]}}}

    return ExchangeDeliveredOrderItemsCfg


def make_modify_tool(cfg: Dict[str, Any]):
    class ModifyPendingOrderItemsCfg(Tool):
        @staticmethod
        def invoke(data: Dict[str, Any], order_id: str, item_ids: List[str],
                   new_item_ids: List[str], payment_method_id: str) -> str:
            if not item_ids:
                return "Error: item_ids must not be empty"
            orders, products = data["orders"], data["products"]
            if order_id not in orders:
                return "Error: order not found"
            order = orders[order_id]
            if order["status"] != "pending":
                return "Error: only pending orders can be modified"
            user = data["users"][order["user_id"]]
            pm = user["payment_methods"].get(payment_method_id)
            if pm is None:
                return "Error: payment method not found"
            original = order["payment_history"][0]["payment_method_id"]
            if pm["source"] != "gift_card" and payment_method_id != original:
                return "Error: the price difference must be settled via the original payment method or a gift card"
            err, diff = _swap_validate(data, order, item_ids, new_item_ids)
            if err:
                return err
            if diff != 0:
                err = _settle(user, payment_method_id, diff, order, "payment" if diff > 0 else "refund")
                if err:
                    return err
            # swap the items in place (first unswapped occurrence per pair)
            swapped = set()
            for item_id, new_item_id in zip(item_ids, new_item_ids):
                for idx, it in enumerate(order["items"]):
                    if it["item_id"] == item_id and idx not in swapped:
                        p = products[it["product_id"]]
                        v = p["variants"][new_item_id]
                        order["items"][idx] = {"name": p["name"], "product_id": p["product_id"],
                                               "item_id": new_item_id, "price": v["price"],
                                               "options": v["options"]}
                        swapped.add(idx)
                        break
            # L2+ (config flag record_modifications): modifications are recorded
            # on the order, making "was this order modified before?" legible state
            # rather than a ledger-shape inference.
            if cfg.get("record_modifications"):
                hist = order.setdefault("modification_history", [])
                # sorted: the same multiset of swaps must yield the same DB
                # hash regardless of the caller's argument order
                for item_id, new_item_id in sorted(zip(item_ids, new_item_ids)):
                    hist.append({"item_id": item_id, "new_item_id": new_item_id})
            return json.dumps(order)

        @staticmethod
        def get_info() -> Dict[str, Any]:
            return {"type": "function", "function": {
                "name": "modify_pending_order_items",
                "description": "Modify items of a pending order to other variants of the same products. See the policy and workflow documents for the rules and procedure.",
                "parameters": {"type": "object", "properties": {
                    "order_id": {"type": "string", "description": "The order id, e.g. '#W0000000'."},
                    "item_ids": {"type": "array", "minItems": 1, "items": {"type": "string"}, "description": "The current item ids to change."},
                    "new_item_ids": {"type": "array", "minItems": 1, "items": {"type": "string"}, "description": "The replacement item ids, same length/order as item_ids."},
                    "payment_method_id": {"type": "string", "description": "Payment method for settling the price difference."},
                }, "required": ["order_id", "item_ids", "new_item_ids", "payment_method_id"]}}}

    return ModifyPendingOrderItemsCfg


def make_write_tools(cfg: Dict[str, Any]) -> list:
    tools = [make_cancel_tool(cfg), PlaceOrder]
    if cfg["modify_offered"]:
        tools.append(make_modify_tool(cfg))
    if cfg["returns_offered"]:
        tools.append(make_return_tool(cfg))
    if cfg["exchanges_offered"]:
        tools.append(make_exchange_tool(cfg))
    return tools
