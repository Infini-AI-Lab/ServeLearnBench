from __future__ import annotations

import json
from typing import Any, Dict

from ....engine.tool import Tool


class GetOrderDetails(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], order_id: str) -> str:
        order = data["orders"].get(order_id)
        if order is None:
            return "Error: order not found"
        return json.dumps(order)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_order_details",
                "description": "Get the status, items, and payment history of an order.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "description": "The order id, e.g. '#W0000000'. Note the leading '#'."},
                    },
                    "required": ["order_id"],
                },
            },
        }
