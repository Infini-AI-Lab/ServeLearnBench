from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ....engine.tool import Tool


def _new_order_id(orders: Dict[str, Any]) -> str:
    # Deterministic: derived from the current order count so the agent's path and
    # the GT replay (both starting from the same clean DB) mint the same id.
    n = 9000000 + len(orders)
    oid = f"#W{n}"
    while oid in orders:
        n += 1
        oid = f"#W{n}"
    return oid


class PlaceOrder(Tool):
    @staticmethod
    def invoke(
        data: Dict[str, Any], user_id: str, item_ids: List[str],
        payment_method_id: str, fallback_payment_method_id: Optional[str] = None,
    ) -> str:
        if not item_ids:
            return "Error: item_ids must not be empty"
        users, products, orders = data["users"], data["products"], data["orders"]
        if user_id not in users:
            return "Error: user not found"
        user = users[user_id]
        if payment_method_id not in user["payment_methods"]:
            return "Error: payment method not found"
        if fallback_payment_method_id is not None:
            if fallback_payment_method_id not in user["payment_methods"]:
                return "Error: fallback payment method not found"
            if "gift_card" in fallback_payment_method_id:
                return "Error: fallback payment method cannot be a gift card"

        # resolve items from the catalog
        items = []
        for item_id in item_ids:
            found = None
            for p in products.values():
                if item_id in p["variants"]:
                    v = p["variants"][item_id]
                    if not v["available"]:
                        return "Error: item is not available"
                    found = {"name": p["name"], "product_id": p["product_id"],
                             "item_id": item_id, "price": v["price"], "options": v["options"]}
                    break
            if found is None:
                return "Error: item not found"
            items.append(found)
        total = round(sum(it["price"] for it in items), 2)

        # payment: gift cards are limited by balance; insufficient balance needs a
        # non-gift-card fallback for the remainder, otherwise the order is rejected.
        pm = user["payment_methods"][payment_method_id]
        payment_history = []
        if pm["source"] == "gift_card":
            balance = pm["balance"]
            if balance >= total:
                payment_history.append({"transaction_type": "payment", "amount": total,
                                        "payment_method_id": payment_method_id})
                pm["balance"] = round(balance - total, 2)
            elif fallback_payment_method_id is not None:
                remainder = round(total - balance, 2)
                if balance > 0:
                    payment_history.append({"transaction_type": "payment", "amount": round(balance, 2),
                                            "payment_method_id": payment_method_id})
                payment_history.append({"transaction_type": "payment", "amount": remainder,
                                        "payment_method_id": fallback_payment_method_id})
                pm["balance"] = 0.0
            else:
                return ("Error: insufficient gift card balance "
                        f"(balance ${balance:.2f} < total ${total:.2f}) and no fallback payment method")
        else:
            payment_history.append({"transaction_type": "payment", "amount": total,
                                    "payment_method_id": payment_method_id})

        from ..policy_config import TODAY
        order_id = _new_order_id(orders)
        order = {
            "order_id": order_id, "user_id": user_id, "address": dict(user["address"]),
            "items": items, "status": "pending", "placed_at": TODAY,
            "payment_history": payment_history,
        }
        orders[order_id] = order
        user["orders"].append(order_id)
        return json.dumps(order)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "place_order",
                "description": (
                    "Place a new order for a user. Payment rules: if the payment method is a gift card, "
                    "its balance must cover the total; when it does not, a non-gift-card fallback payment "
                    "method is required — the gift card is drained and the remainder is charged to the "
                    "fallback. Without a fallback, an order exceeding the gift card balance is rejected. "
                    "Requires explicit user confirmation before calling."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "description": "The user id, e.g. 'noah_brown_6181'."},
                        "item_ids": {
                            "type": "array",
                            "minItems": 1, "items": {"type": "string"},
                            "description": "Catalog item ids (variants) to order, e.g. '1008292230'.",
                        },
                        "payment_method_id": {"type": "string", "description": "Primary payment method id."},
                        "fallback_payment_method_id": {
                            "type": "string",
                            "description": "Optional non-gift-card method charged for the remainder when a gift card cannot cover the total.",
                        },
                    },
                    "required": ["user_id", "item_ids", "payment_method_id"],
                },
            },
        }
