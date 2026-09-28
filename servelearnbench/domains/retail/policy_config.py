"""Per-stage policy configuration -- the single source of truth for the rules.

Rules are INTERACTION TABLES, not flat values. A single return decision
depends on membership tier x product category x order age:
  fee   = f(tier)          window = f(category)      destination = f(tier)
  cancel threshold = f(tier, stage>=4)               clearance => final sale

Drift matrix:
  S0  strict shop : card payments only; cancellation + order modification only.
  S1  gift cards  : gift-card program launches.
  S2  growth      : catalog/orders expand, store_credit appears.
  S3  returns     : returns launch with tier fees, category windows,
                    tier-dependent refund destinations, final-sale exclusions.
  S4  exchanges   : exchange program launches (same window/final-sale rules)
                    + tier-dependent cancellation thresholds.

Tools and documents are both rendered from these configs, so the bundle can
never contradict itself.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

# The environment's calendar date (world parameter, not mutable state).
TODAY = "2026-07-15"

REFUSE_REASONS = [
    "service_not_offered",      # the requested operation does not exist (yet)
    "invalid_status",           # order status does not allow the action
    "destination_not_allowed",  # requested money destination violates policy
    "insufficient_balance",     # gift card cannot cover and user refuses alternatives
    "over_threshold",           # amount exceeds the policy threshold (tier-dependent)
    "window_expired",           # outside the category's return/exchange window
    "final_sale",               # clearance items are final sale
    "not_found",                # referenced order/item does not exist or does not match
]

TIERS = ["bronze", "silver", "gold"]
CATEGORIES = ["apparel", "electronics", "home"]

_S0: Dict[str, Any] = {
    "gift_cards": False,
    "cancel_reasons": ["no longer needed", "ordered by mistake"],
    # tier -> max cancellable order total (None = no threshold). Activated at S4.
    "cancel_max_total": None,
    "returns_offered": False,
    "return_reasons": None,
    # tier -> restocking-fee fraction withheld from the refund
    "return_fee": None,
    # category -> days after delivery during which returns/exchanges are allowed
    "return_window_days": None,
    # tiers whose refunds may ALSO go to the original payment method
    "refund_original_tiers": [],
    "exchanges_offered": False,
    # order modification (pending orders) has existed from day one
    "modify_offered": True,
}


def get_config(stage: int) -> Dict[str, Any]:
    cfg = copy.deepcopy(_S0)
    if stage >= 1:
        cfg["gift_cards"] = True
    if stage >= 3:
        cfg["returns_offered"] = True
        cfg["return_reasons"] = ["defective", "no longer needed", "wrong item"]
        cfg["return_fee"] = {"bronze": 0.15, "silver": 0.10, "gold": 0.05}
        cfg["return_window_days"] = {"apparel": 30, "electronics": 14, "home": 30}
        cfg["refund_original_tiers"] = ["gold"]
    if stage >= 4:
        cfg["exchanges_offered"] = True
        cfg["cancel_max_total"] = {"bronze": 200.0, "silver": 300.0, "gold": 500.0}
    return cfg


def days_between(d1: str, d2: str) -> int:
    """Days from ISO date d1 to d2."""
    from datetime import date
    return (date.fromisoformat(d2) - date.fromisoformat(d1)).days
