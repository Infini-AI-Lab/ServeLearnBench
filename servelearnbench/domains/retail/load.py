"""Load the generated retail DB and validate its consistency invariants.

A fresh dict is returned on every load_data() call (read from disk), so replay
and agent runs never share mutable state.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

FOLDER = os.path.dirname(__file__)
DATA_DIR = os.path.join(FOLDER, "data")


def load_data() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for table in ("users", "products", "orders"):
        with open(os.path.join(DATA_DIR, f"{table}.json")) as f:
            out[table] = json.load(f)
    return out


def validate(data: Dict[str, Any]) -> List[str]:
    """Return a list of violations; empty list means the DB is self-consistent.

    Inconsistent data would make reward provably wrong, so callers fail-fast.
    """
    errors: List[str] = []
    users, products, orders = data["users"], data["products"], data["orders"]

    # real item_id set
    valid_items = {
        item_id
        for p in products.values()
        for item_id in p["variants"]
    }

    # reverse index user -> orders actually owned
    owned: Dict[str, set] = {uid: set() for uid in users}

    for oid, order in orders.items():
        if order.get("order_id") != oid:
            errors.append(f"{oid}: order_id field mismatch")
        uid = order["user_id"]
        if uid not in users:
            errors.append(f"{oid}: user_id {uid} not in users")
            continue
        owned.setdefault(uid, set()).add(oid)
        # items reference real variants AND carry the catalog's exact
        # product/name/options/price (an in-memory tamper must not pass)
        for it in order["items"]:
            if it["item_id"] not in valid_items:
                errors.append(f"{oid}: item_id {it['item_id']} not a real variant")
                continue
            pid = it.get("product_id")
            prod = products.get(pid)
            if prod is None or it["item_id"] not in prod.get("variants", {}):
                errors.append(f"{oid}: item {it['item_id']} not a variant of "
                              f"its declared product {pid}")
                continue
            var = prod["variants"][it["item_id"]]
            if it.get("name") != prod.get("name"):
                errors.append(f"{oid}: item {it['item_id']} name mismatch")
            if abs(it.get("price", -1) - var.get("price", -2)) > 0.001:
                errors.append(f"{oid}: item {it['item_id']} price mismatch")
            if it.get("options") != var.get("options"):
                errors.append(f"{oid}: item {it['item_id']} options mismatch")
        # payment total == item total (initial payment)
        item_total = round(sum(it["price"] for it in order["items"]), 2)
        pay_total = round(
            sum(p["amount"] for p in order["payment_history"] if p["transaction_type"] == "payment"),
            2,
        )
        if abs(item_total - pay_total) > 0.001:
            errors.append(f"{oid}: payment {pay_total} != items {item_total}")
        # payment method belongs to the user
        for p in order["payment_history"]:
            if p["payment_method_id"] not in users[uid]["payment_methods"]:
                errors.append(f"{oid}: payment method {p['payment_method_id']} not owned by {uid}")
        # address completeness
        for field in ("address1", "city", "state", "zip", "country"):
            if field not in order["address"]:
                errors.append(f"{oid}: address missing {field}")

    # reverse-reference consistency
    for uid, user in users.items():
        if user.get("user_id") != uid:
            errors.append(f"{uid}: user_id field mismatch")
        if set(user["orders"]) != owned.get(uid, set()):
            errors.append(f"{uid}: user.orders {set(user['orders'])} != owned {owned.get(uid, set())}")

    # policy fields: membership / category / clearance / dates
    for uid, user in users.items():
        if user.get("membership") not in ("bronze", "silver", "gold"):
            errors.append(f"{uid}: missing/invalid membership tier")
    for pid, p in products.items():
        if p.get("category") not in ("apparel", "electronics", "home"):
            errors.append(f"{pid}: missing/invalid category")
        for iid, v in p["variants"].items():
            if "clearance" not in v:
                errors.append(f"{pid}/{iid}: missing clearance flag")
    for oid, order in orders.items():
        if "placed_at" not in order:
            errors.append(f"{oid}: missing placed_at")
        if order["status"] == "delivered":
            if "delivered_at" not in order:
                errors.append(f"{oid}: delivered without delivered_at")
            elif order["delivered_at"] < order["placed_at"]:
                errors.append(f"{oid}: delivered_at before placed_at")

    # status distribution: cancel/return feasibility
    statuses = [o["status"] for o in orders.values()]
    if statuses.count("pending") < 1:
        errors.append("no pending orders (cancel tasks infeasible)")
    if statuses.count("delivered") < 1:
        errors.append("no delivered orders (return tasks infeasible)")

    return errors
