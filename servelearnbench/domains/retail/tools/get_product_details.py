from __future__ import annotations

import json
from typing import Any, Dict

from ....engine.tool import Tool


class GetProductDetails(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], product_id: str) -> str:
        product = data["products"].get(product_id)
        if product is None:
            return "Error: product not found"
        return json.dumps(product)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_product_details",
                "description": "Get the variants (item ids, options, availability, price) of a product.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string", "description": "The product id, e.g. '9523456873'. This is not the item id."},
                    },
                    "required": ["product_id"],
                },
            },
        }
