from __future__ import annotations

import json
from typing import Any, Dict

from ....engine.tool import Tool


class GetUserDetails(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], user_id: str) -> str:
        user = data["users"].get(user_id)
        if user is None:
            return "Error: user not found"
        return json.dumps(user)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_user_details",
                "description": "Get the details of a user, including their orders, addresses, and payment methods.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "description": "The user id, e.g. 'noah_brown_6181'."},
                    },
                    "required": ["user_id"],
                },
            },
        }
