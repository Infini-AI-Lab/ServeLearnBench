"""Assemble the L2 bundle: world + mechanical tools + frozen docs + timeline.

Tools are MECHANICALLY PERMISSIVE on the hidden dimensions (the API lags
policy): the cancel tool doesn't care how an order was paid, the exchange tool
doesn't care about category/geography, the return tool honors any owned refund
destination, the modify tool accepts split-paid orders. Truth reaches the
system only through rewards.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

from ..config_tools import make_write_tools
from ..prompt import system_prompt
from ..policy_config import get_config
from ..tools import (
    FindUserIdByNameZip, GetOrderDetails, GetProductDetails, GetUserDetails,
)
from . import docs, rules, taskgen, world


def mechanical_config() -> Dict[str, Any]:
    """S4 config with the refund-destination gate opened for every tier: the
    API supports original-method refunds for anyone; whether honoring the
    request is CORRECT is decided by rules.truth(t), not the tool."""
    cfg = get_config(4)
    cfg["refund_original_tiers"] = ["bronze", "silver", "gold"]
    cfg["exchange_any_method"] = True  # docs say "any payment method on file" — tool must agree
    cfg["split_refund_proportional"] = True  # L2 policy #3 (config-gated in the shared tool layer)
    cfg["record_modifications"] = True   # D6 trigger state must be legible, not ledger-inferred
    return cfg


def get_bundle() -> Dict[str, Any]:
    data, registry = world.build_world()
    cfg = mechanical_config()
    tools = [docs.ReadDocsV2,
             FindUserIdByNameZip, GetUserDetails, GetOrderDetails, GetProductDetails]
    tools += make_write_tools(cfg)

    meta = {"scenario": "retail_l2", "domain": "retail", "tier": "L2",
            "system_prompt": system_prompt(tools, docs.system_prompt_toc())}
    return {
        "meta": meta,
        "id": "l2",
        "config": cfg,
        "data": data,
        "registry": registry,
        "load_data": lambda d=data: copy.deepcopy(d),
        "tools": tools,
        "policy_md": docs.merged_md("policy"),
        "workflow_md": docs.merged_md("workflow"),
        "docs_toc": docs.system_prompt_toc(),
        "serving": taskgen.serving_sequence(registry),
        "test_set": lambda w, r=registry: taskgen.test_set(r, w),
        "rules": rules,
    }
