from __future__ import annotations

import json
from typing import Any, Dict

from ....engine.tool import Tool

VALID_REASONS = ["no longer needed", "ordered by mistake"]


class CancelPendingOrder(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], order_id: str, reason: str) -> str:
        orders = data["orders"]
        if order_id not in orders:
            return "Error: order not found"
        order = orders[order_id]
        if order["status"] != "pending":
            return "Error: non-pending order cannot be cancelled"
        if reason not in VALID_REASONS:
            return "Error: invalid reason"

        # Refund each payment method's OUTSTANDING balance (payments - refunds),
        # exactly once. Gift-card refunds top up the balance immediately.
        net = {}
        for p in order["payment_history"]:
            sign = 1 if p["transaction_type"] == "payment" else -1
            pid = p["payment_method_id"]
            net[pid] = round(net.get(pid, 0.0) + sign * p["amount"], 2)
        refunds = []
        for pid, bal in net.items():
            if bal <= 0:
                continue
            refunds.append({
                "transaction_type": "refund",
                "amount": bal,
                "payment_method_id": pid,
            })
            if "gift_card" in pid:
                pm = data["users"][order["user_id"]]["payment_methods"][pid]
                pm["balance"] = round(pm["balance"] + bal, 2)

        order["status"] = "cancelled"
        order["cancel_reason"] = reason
        order["payment_history"].extend(refunds)
        return json.dumps(order)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "cancel_pending_order",
                "description": (
                    "Cancel a pending order. If the order is already processed or delivered it cannot "
                    "be cancelled. Requires explicit user confirmation before calling. The status becomes "
                    "'cancelled' and the payment is refunded (gift card immediately, otherwise 5-7 business days)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "description": "The order id, e.g. '#W0000000'. Note the leading '#'."},
                        "reason": {
                            "type": "string",
                            "enum": VALID_REASONS,
                            "description": "Either 'no longer needed' or 'ordered by mistake'.",
                        },
                    },
                    "required": ["order_id", "reason"],
                },
            },
        }
