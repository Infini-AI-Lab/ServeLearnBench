"""Assemble the L3 bundle: world + mechanical tools + frozen docs + timeline.

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
    cfg["clearance_returnable"] = True   # L3 permissive gate: the RETURN tool is
    # mechanically permissive on clearance; correctness comes from rules.truth(t)
    return cfg


def get_bundle() -> Dict[str, Any]:
    data, registry = world.build_world()
    cfg = mechanical_config()
    tools = [docs.ReadDocsV2,
             FindUserIdByNameZip, GetUserDetails, GetOrderDetails, GetProductDetails]
    tools += make_write_tools(cfg)

    meta = {"scenario": "retail_l3", "domain": "retail", "tier": "L3",
            "system_prompt": system_prompt(tools, docs.system_prompt_toc())}
    return {
        "meta": meta,
        "id": "l3",
        "config": cfg,
        "data": data,
        "registry": registry,
        "load_data": lambda d=data: copy.deepcopy(d),
        "tools": tools,
        "policy_md": docs.merged_md("policy"),
        "workflow_md": docs.merged_md("workflow"),
        "docs_toc": docs.system_prompt_toc(),
        "serving": _exposed_serving(registry),
        "test_set": lambda w, r=registry: _exposed_test(r, w),
        "rules": _ExposedRules(),
    }


def _exposed_serving(registry):
    """Hidden windows (rules.HIDDEN_WINDOWS) never serve; the survivors are
    renumbered W0..W7 in z['window']. Underlying t values keep their gaps —
    they are evaluator-internal and never shown to the agent."""
    out = []
    for t in taskgen.serving_sequence(registry):
        pw = rules.presented_of(rules.window_of(t.z["t"]))
        if pw is None:
            continue
        t.z["window"] = pw
        out.append(t)
    return out


def _exposed_test(registry, w_presented):
    if not (0 <= w_presented < len(rules.PRESENTED_WINDOWS)):
        return []
    uw = rules.PRESENTED_WINDOWS[w_presented]
    tasks = taskgen.test_set(registry, uw)
    for t in tasks:
        t.z["window"] = w_presented
    return tasks


class _ExposedRules:
    """rules proxy: consumers (runners, reports) see EIGHT windows;
    truth/oracle keep underlying t semantics untouched."""
    WINDOW_STARTS = [rules.WINDOW_STARTS[u] for u in rules.PRESENTED_WINDOWS]

    def __getattr__(self, name):
        return getattr(rules, name)

    @staticmethod
    def window_of(t):
        return rules.presented_of(rules.window_of(t))

    @staticmethod
    def window_range(w_presented):
        return rules.window_range(rules.PRESENTED_WINDOWS[w_presented])
