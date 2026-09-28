from __future__ import annotations

import json
from typing import Any, Dict, List

from ....engine.tool import Tool


class ReturnDeliveredOrderItems(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], order_id: str, item_ids: List[str], payment_method_id: str) -> str:
        if not item_ids:
            return "Error: item_ids must not be empty"
        orders = data["orders"]
        if order_id not in orders:
            return "Error: order not found"
        order = orders[order_id]
        if order["status"] != "delivered":
            return "Error: non-delivered order cannot be returned"

        user_methods = data["users"][order["user_id"]]["payment_methods"]
        if payment_method_id not in user_methods:
            return "Error: payment method not found"
        # refund must go to the original payment method or a gift card
        if (
            "gift_card" not in payment_method_id
            and payment_method_id != order["payment_history"][0]["payment_method_id"]
        ):
            return "Error: payment method should be either the original payment method or a gift card"

        # items to return must exist in the order (respecting multiplicity)
        all_item_ids = [it["item_id"] for it in order["items"]]
        for item_id in item_ids:
            if item_ids.count(item_id) > all_item_ids.count(item_id):
                return "Error: some item not found"

        order["status"] = "return requested"
        order["return_items"] = sorted(item_ids)  # sorted: list hash is order-sensitive
        order["return_payment_method_id"] = payment_method_id
        return json.dumps(order)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "return_delivered_order_items",
                "description": (
                    "Return some items of a delivered order. The status becomes 'return requested'. "
                    "Requires explicit user confirmation before calling."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "description": "The order id, e.g. '#W0000000'. Note the leading '#'."},
                        "item_ids": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string"},
                            "description": "The item ids to return, e.g. '1008292230'. Duplicates allowed.",
                        },
                        "payment_method_id": {
                            "type": "string",
                            "description": "Refund target: the original payment method or a gift card, e.g. 'gift_card_0000000'.",
                        },
                    },
                    "required": ["order_id", "item_ids", "payment_method_id"],
                },
            },
        }
