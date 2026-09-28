"""Drift schedules for the three pitch tiers.

The hidden preference is a CUSTOMER (customers.py). Each tier is a sequence of
windows; each window has one active customer. Difficulty rises by how the
customer drifts across windows (mirrors the retail tier spine):

  L1  4 windows, 4 distinct customers, never revisited, stable within
  L2  6 windows, hard switches, two customers recur at the end
  L3  oscillate + reopen: 7 windows, customers return (must re-recognise a
      taste seen earlier, not overwrite it); identity appears three times

  All switches are hard: exactly one customer is active per window, no
  mixtures or gradual transitions.

Each window holds 54 serving and 24 test items so per-window means are
statistically usable.

Time index t is 1-based over the serving stream; window boundaries are every
PER_WINDOW_SERVING products.
"""

from __future__ import annotations

PER_WINDOW_SERVING = 54
PER_WINDOW_TEST = 24

SCHEDULES = {
    # L1 uses only customers whose EFFECTIVE combo space (valence-nonzero
    # dims) supports serving/test isolation — eco (4 combos) and conformist
    # (5) cannot be isolated under 54 serving items, so eco appears in L2 and
    # conformist in L3, keeping all six covered.
    "L1": ["value", "identity", "anti_marketing", "risk_averse"],
    "L2": ["risk_averse", "value", "eco", "identity", "value", "risk_averse"],
    "L3": ["risk_averse", "identity", "anti_marketing", "identity", "risk_averse",
           "conformist", "identity"],
}


def n_windows(tier: str) -> int:
    return len(SCHEDULES[tier])


def customer_of_window(tier: str, w: int) -> str:
    return SCHEDULES[tier][w]


def is_reopen_window(tier: str, w: int) -> bool:
    """True if this window's customer already appeared in an earlier window."""
    c = SCHEDULES[tier][w]
    return c in SCHEDULES[tier][:w]


class Rules:
    """Per-tier window object with the interface the serving runner expects
    (WINDOW_STARTS, window_of, oracle_note)."""

    def __init__(self, tier: str):
        self.tier = tier
        self.WINDOW_STARTS = [w * PER_WINDOW_SERVING + 1 for w in range(n_windows(tier))]

    def window_of(self, t: int) -> int:
        return min((t - 1) // PER_WINDOW_SERVING, n_windows(self.tier) - 1)

    def oracle_note(self, t: int) -> str:
        """The oracle bulletin (current customer described) for the window at t.
        Restricted to the tier's dim set (L1 sheets carry 5 dims)."""
        from .customers import oracle_bulletin
        from .world import TIER_DIMS
        return oracle_bulletin(customer_of_window(self.tier, self.window_of(t)),
                               dims=TIER_DIMS[self.tier])
