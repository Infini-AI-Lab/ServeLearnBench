from __future__ import annotations

import json
from typing import Any, Dict, List

from ....engine.tool import Tool


class ExchangeDeliveredOrderItems(Tool):
    @staticmethod
    def invoke(
        data: Dict[str, Any], order_id: str, item_ids: List[str],
        new_item_ids: List[str], payment_method_id: str,
    ) -> str:
        if not item_ids:
            return "Error: item_ids must not be empty"
        orders, products = data["orders"], data["products"]
        if order_id not in orders:
            return "Error: order not found"
        order = orders[order_id]
        if order["status"] != "delivered":
            return "Error: non-delivered order cannot be exchanged"
        if len(item_ids) != len(new_item_ids):
            return "Error: item_ids and new_item_ids must have the same length"

        user_methods = data["users"][order["user_id"]]["payment_methods"]
        if payment_method_id not in user_methods:
            return "Error: payment method not found"
        if (
            "gift_card" not in payment_method_id
            and payment_method_id != order["payment_history"][0]["payment_method_id"]
        ):
            return "Error: payment method should be either the original payment method or a gift card"

        all_item_ids = [it["item_id"] for it in order["items"]]
        diff = 0.0
        for item_id, new_item_id in zip(item_ids, new_item_ids):
            if item_ids.count(item_id) > all_item_ids.count(item_id):
                return "Error: some item not found"
            item = next(it for it in order["items"] if it["item_id"] == item_id)
            product = products.get(item["product_id"])
            if product is None or new_item_id not in product["variants"]:
                return "Error: new item is not a variant of the same product"
            variant = product["variants"][new_item_id]
            if not variant["available"]:
                return "Error: new item is not available"
            diff += variant["price"] - item["price"]

        diff = round(diff, 2)
        # gift-card settlement is immediate: a positive difference must be covered
        # by the balance; a negative difference is refunded onto the card.
        if "gift_card" in payment_method_id:
            pm = user_methods[payment_method_id]
            if diff > 0 and pm["balance"] < diff:
                return (f"Error: insufficient gift card balance "
                        f"(balance ${pm['balance']:.2f} < price difference ${diff:.2f})")
            pm["balance"] = round(pm["balance"] - diff, 2)

        order["status"] = "exchange requested"
        order["exchange_items"] = sorted(item_ids)          # sorted: list hash is order-sensitive
        order["exchange_new_items"] = sorted(new_item_ids)
        order["exchange_payment_method_id"] = payment_method_id
        order["exchange_price_difference"] = diff
        return json.dumps(order)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "exchange_delivered_order_items",
                "description": (
                    "Exchange items of a delivered order for different variants of the SAME products. "
                    "Each new item must be an available variant of the product of the item it replaces. "
                    "The order status becomes 'exchange requested'; the price difference is charged or "
                    "refunded via the given payment method. Gift-card settlement is immediate: a positive "
                    "difference requires sufficient balance, a negative difference is refunded onto the card. "
                    "Requires explicit user confirmation before calling."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "description": "The order id, e.g. '#W0000000'. Note the leading '#'."},
                        "item_ids": {
                            "type": "array",
                            "minItems": 1, "items": {"type": "string"},
                            "description": "The item ids to be exchanged, e.g. '1008292230'.",
                        },
                        "new_item_ids": {
                            "type": "array",
                            "minItems": 1, "items": {"type": "string"},
                            "description": "The replacement item ids, same length/order as item_ids; each must be a variant of the same product.",
                        },
                        "payment_method_id": {
                            "type": "string",
                            "description": "Payment method for the price difference: the original payment method or a gift card.",
                        },
                    },
                    "required": ["order_id", "item_ids", "new_item_ids", "payment_method_id"],
                },
            },
        }
