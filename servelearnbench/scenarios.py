"""The nine scenarios: three domains x three difficulty tiers.

A scenario bundle is a dict with the world (`load_data`), the agent's tools,
the reference documents, the ordered serving stream, the per-window test
sets, the hidden-policy module (`rules`, with WINDOW_STARTS and oracle_note),
an optional custom verifier, and `meta` (scenario, domain, tier and the
acting system prompt).
"""

from __future__ import annotations

from typing import Any, Callable, Dict


def _retail(tier: str) -> Callable[[], Dict[str, Any]]:
    def load():
        import importlib
        return importlib.import_module(f"servelearnbench.domains.retail.{tier}.timeline").get_bundle()
    return load


def _banking(tier: str) -> Callable[[], Dict[str, Any]]:
    def load():
        import importlib
        return importlib.import_module(f"servelearnbench.domains.banking.{tier}.world").get_bundle()
    return load


def _pitch(tier: str) -> Callable[[], Dict[str, Any]]:
    def load():
        from servelearnbench.domains.pitch.world import build_bundle
        return build_bundle(tier.upper())
    return load


SCENARIOS: Dict[str, Callable[[], Dict[str, Any]]] = {
    **{f"retail_{t}": _retail(t) for t in ("l1", "l2", "l3")},
    **{f"banking_{t}": _banking(t) for t in ("l1", "l2", "l3")},
    **{f"pitch_{t}": _pitch(t) for t in ("l1", "l2", "l3")},
}

_CACHE: Dict[str, Dict[str, Any]] = {}


def get_bundle(name: str, fresh: bool = False) -> Dict[str, Any]:
    """Build (or return the cached) bundle of a scenario. Pass fresh=True for
    an independent copy whose tasks can be stamped without affecting others."""
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; one of {sorted(SCENARIOS)}")
    if fresh:
        return SCENARIOS[name]()
    if name not in _CACHE:
        _CACHE[name] = SCENARIOS[name]()
    return _CACHE[name]


def n_windows(bundle: Dict[str, Any]) -> int:
    return len(bundle["rules"].WINDOW_STARTS)
