"""Banking L1 world: three request types and deterministic tickets.

Construction disciplines:
- one dedicated account + one dedicated case per ticket, ZERO reuse across
  the tier (global freshness);
- serving and test amounts are exactly disjoint (numeric holdout): both
  splits draw cents from the same distribution, keyed by split;
- family quotas follow a DECLARED per-window table: the changing
  policy's supply peaks in its event window (beat thickening), general
  fillers absorb the delta;
- twins: same surface, flipped hidden field (overseas/domestic,
  basic/verified, chargeback/good, restricted/clean payee);
- instruction shells rotate by ticket index, independent of the verdict;
- episode view: an episode sees the shared world plus ITS OWN case only.
"""

from __future__ import annotations

import copy
import hashlib
import random as _random
from typing import Any, Dict, List

from ....engine.types import Action, Task
from . import docs as D
from ....prompts import compose_system_prompt, tool_signatures
from . import rules as R
from ..tools import (ApproveCase, DenyCase, EscalateCase, GetAccountDetails,
                    GetCaseDetails, GetMerchantDetails, GetPayeeDetails)
from ..test_prune import (align_clean_payee_anchors, balance_change_adapt,
                         balanced_shell_assignment,
                         categorical_alternation_probe_ids,
                         numeric_side_balance_ids,
                         per_window_constant_answer_balance_ids,
                         event_general_probe_ids, prune_general_repetitions,
                         value_coverage_probe_ids)

TODAY = "2026-08-08"
SEED = "banking_v3_L1"
ENTITY_ID_BASE = 100
ENTITY_ID_SPAN = 200
OPAQUE_ID_BASE = 100000
OPAQUE_ID_SPAN = 200000

FIRST = ["Ava", "Noah", "Mia", "Liam", "Zoe", "Ethan", "Ivy", "Lucas", "Nora",
         "Owen", "Ruth", "Felix", "June", "Hugo", "Lena", "Marco",
         "Tessa", "Bram", "Cleo", "Dario", "Elif", "Farid", "Greta", "Hadi", "Imre", "Jolan", "Katya", "Lior", "Mirek", "Nadia", "Oskar", "Priya"]
LAST = ["Alvarez", "Becker", "Chen", "Dawson", "Egan", "Fischer", "Grant",
        "Hopkins", "Iqbal", "Jensen", "Kovac", "Lindqvist", "Moreau", "Novak",
        "Ortiz", "Petrov",
        "Quill", "Rasch", "Soto", "Tanaka", "Ueda", "Varga", "Whitfield", "Xylander", "Yilmaz", "Zapata", "Abano", "Brekke", "Csordas", "Duarte", "Eriksen", "Farkas"]
STATES = ["CA", "NY", "TX", "WA", "IL", "MA", "CO", "GA"]

# 8-12 names per category (electronics 12): the first two-thirds serve
# the serving stream, the rest are test-only per
# (category, region) cell. The rules key on category x region, NOT on the
# entity — with shared merchants, "this MER id was denied before" would
# reproduce the corridor rule with no get_merchant_details call. Payees
# are the opposite case: the roster IS entity-level, so serving and test
# share payees by design (learning WHICH payees are restricted is the task).
MERCHANT_NAMES = {
    "electronics": ["Voltway", "Circuit Loft", "Ohm Depot", "Ampere House", "Gridline", "Watt & Main", "Cinder Circuits", "Novelle Tech", "Coilhouse", "Brightfuse", "Menlo Parts", "Statica"],
    "travel": ["Skyline Tours", "Harbor Trips", "Trailhead Co", "Meridian Air", "Atlas Fare", "Beacon Voyages", "Slipstream Travel", "Cairn & Compass"],
    "jewelry": ["Gilt & Stone", "Aurel Atelier", "Clasp House", "Opaline", "Karat Row", "Tessera Fine", "Onyx & Ash", "Silverbraid"],
    "gift_cards": ["CardNest", "PrepaidHub", "TokenTree", "StoreCredit Co", "GiftBridge", "StowCard", "Emberly Gifts", "Vault Voucher"],
    "groceries": ["Green Crate", "Daily Larder", "Pantry Union", "Field & Jar", "Cornermart", "Miller's Crate", "Fenwick Foods", "Orchard Row"],
    "utilities": ["MetroPower", "ClearWater Co", "CityGas", "Voltic Utility", "Aquaduct", "Brightmain Power", "Cobble Gas", "Rainline Water"],
    "dining": ["Fork & Flame", "Noodle Dock", "Cinder Table", "Brine & Bread", "Sagebrush", "Hearth & Rye", "Juniper Bowl", "Copperpot Diner"],
    "fitness": ["IronLeaf Gym", "Stride Studio", "Kettle Club", "Apex Fit", "Rowhouse", "Granite Gym", "Loop Athletics", "Emberfit"],
}

PAYEE_NAMES = ["Meridian Holdings", "Baltic Trade Co", "Sunrise Logistics",
               "Vector Capital", "Northgate Supply", "Orchid Ventures",
               "Lakeside Freight", "Halcyon Partners", "Pinewood Exports",
               "Cobalt Services", "Redstone Group", "Aurora Imports"]
# 4 jurisdictions so BOTH the roster (5 names) and the clean set (7) cover
# every value, so no jurisdiction excludes a payee from the roster
JURISDICTIONS = ["US", "UK", "SG", "AE"]

SHELLS = {
    "txn": [
        "Transaction review {cid}: {ename} ({eid}) placed a ${amt:.2f} "
        "authorization on account {acc}. Record the decision.",
        "Operations ticket {cid} concerns a ${amt:.2f} card purchase at "
        "{ename} ({eid}) for account {acc}.",
        "Account {acc} has a held ${amt:.2f} purchase with {ename} ({eid}); "
        "resolve transaction {cid}.",
        "Card queue entry {cid}: ${amt:.2f} at {ename} ({eid}) on account "
        "{acc}. Settle the decision.",
        "Flagged charge {cid} — account {acc}, ${amt:.2f}, merchant {ename} "
        "({eid}). Decide it.",
        "Pending card authorization {cid} links account {acc} with {ename} "
        "({eid}) for ${amt:.2f}. Record the outcome.",
    ],
    "limit": [
        "Credit request {cid}: account {acc} proposes a new ceiling of "
        "${amt:.2f}. Record the decision.",
        "Operations ticket {cid} asks whether account {acc} may receive a "
        "${amt:.2f} limit.",
        "Account {acc} has a ${amt:.2f} credit-line application awaiting "
        "resolution under case {cid}.",
        "Ceiling request {cid}: account {acc} wants ${amt:.2f}. Settle the "
        "decision.",
        "Flagged limit case {cid} — account {acc} proposes ${amt:.2f}. "
        "Decide it.",
        "Account {acc} seeks a ${amt:.2f} ceiling in request {cid}. Record "
        "the outcome.",
    ],
    "transfer": [
        "Payment review {cid}: account {acc} proposes ${amt:.2f} to "
        "{ename} ({eid}). Record the decision.",
        "Operations ticket {cid} covers a held ${amt:.2f} payment from "
        "account {acc} to {ename} ({eid}).",
        "Account {acc} has a ${amt:.2f} outbound payment to {ename} ({eid}) "
        "awaiting resolution as case {cid}.",
        "Payment order {cid}: ${amt:.2f} from {acc} to {ename} ({eid}). "
        "Settle the decision.",
        "Flagged transfer {cid} — {acc} to {ename} ({eid}), ${amt:.2f}. "
        "Decide it.",
        "Transfer screening case {cid}: {acc} would send ${amt:.2f} to "
        "{ename} ({eid}). Record the outcome.",
    ],
}


def _h(*parts) -> int:
    return int(hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest(), 16)


def _build_merchants() -> Dict[str, dict]:
    """Ids are assigned AFTER a deterministic shuffle, so the id does not
    reveal the hidden region without a lookup."""
    rows = [(name, cat, region)
            for cat, names in MERCHANT_NAMES.items() for name in names
            for region in ("domestic", "overseas")]
    _random.Random(f"mer:{SEED}").shuffle(rows)
    out = {}
    for i, (name, cat, region) in enumerate(rows):
        mid = f"MER-{ENTITY_ID_BASE + i:03d}"
        out[mid] = {"merchant_id": mid, "name": name,
                    "category": cat, "region": region}
    return out


def _build_payees():
    """Ids are assigned AFTER a deterministic shuffle of all payee names, so
    the roster does not occupy its own id block. Jurisdictions rotate over
    the shuffled order, so no jurisdiction is roster-exclusive.
    Returns (payees, roster_ids); roster_ids fills rules.RESTRICTED_ROSTER."""
    names = list(PAYEE_NAMES)
    _random.Random(f"pay:{SEED}").shuffle(names)
    out, roster = {}, []
    used = set()
    for i, name in enumerate(names):
        # hash-per-name ids, so the roster shares no residue class
        n = ENTITY_ID_BASE + _h(SEED, "pid", name) % ENTITY_ID_SPAN
        while n in used:
            n = ENTITY_ID_BASE + (n - ENTITY_ID_BASE + 1) % ENTITY_ID_SPAN
        used.add(n)
        pid = f"PAY-{n:03d}"
        out[pid] = {"payee_id": pid, "name": name, "type": "business",
                    "jurisdiction": JURISDICTIONS[i % len(JURISDICTIONS)]}
        if i < 5:
            roster.append(pid)
    return out, sorted(roster)


_MERCHANTS = _build_merchants()
_PAYEES, _ROSTER = _build_payees()
R.RESTRICTED_ROSTER[:] = _ROSTER
_CLEAN_PAYEES = [p for p in sorted(_PAYEES) if p not in set(_ROSTER)]


def _merchant(cat: str, region: str, salt, split: str, w: int) -> dict:
    """Merchant names are split-disjoint GLOBALLY per category: the first
    two-thirds of each pool (5 of 8, 8 of 12) only ever serve the serving
    stream, the rest only ever appear on test tickets — in every window.
    This prevents recalling verdicts by merchant id across windows. An
    unseen merchant id marks a ticket as test, which is harmless: serving
    and test are separate phases and the split is not secret."""
    pool = MERCHANT_NAMES[cat]
    n_sv = (2 * len(pool)) // 3          # 8-pools: 5/3; 12-pools: 8/4
    names = pool[:n_sv] if split != "test" else pool[n_sv:]
    cands = [m for m in _MERCHANTS.values()
             if m["category"] == cat and m["region"] == region
             and m["name"] in names]
    return cands[_h(SEED, "mer", salt) % len(cands)]


def _account(acc_id: str, verification: str, standing: str, tenure: int,
             h: int, holder_i: int) -> dict:
    return {"account_id": acc_id,
            "holder": (f"{FIRST[holder_i % len(FIRST)]} L. "
                       f"{LAST[(holder_i // len(FIRST)) % len(LAST)]}"),
            "state": STATES[(h // 7) % len(STATES)],
            "verification": verification, "standing": standing,
            "tenure_years": tenure,
            "credit_limit": 1000 + 500 * (h % 5),
            # opened is derived from TODAY so the stated tenure_years is exactly the number of full
            # years elapsed
            "opened": (lambda m: f"{2026 - tenure - (1 if m > 8 else 0)}-"
                                 f"{m:02d}")(1 + h % 12)}


def _cents(family: str, split: str, w: int, k: int) -> float:
    """Per-ticket cents, SAME distribution on both splits, so the cents
    pattern does not identify the split."""
    # IDENTICAL distribution on both splits (any deterministic offset —
    # +10, +0.05, integer-vs-not — is itself a readable split marker);
    # disjointness comes from the hash being split-keyed.
    return round(0.01 + 0.01 * (_h(SEED, "cents", family, split, w, k) % 1999), 2)


def _grid(lo: float, step: float, k: int, split: str, family: str = "",
          w: int = 0) -> float:
    return round(lo + step * k + _cents(family, split, w, k), 2)


def _spread(lo: float, hi: float, family: str, k: int, split: str, w: int) -> float:
    span = int((hi - lo) // 20)
    v = lo + 20 * (_h(SEED, "spread", family, split, w, k) % max(span, 1))
    v = v - (v % 20)
    return round(v + _cents(family, split, w, k), 2)


# ---- per-window quota tables (DECLARED window-varying quotas) ----
# Few policies, thick data, and the CHANGING policy's supply peaks in its
# event window (beat thickening). Quotas therefore vary by window according
# to this declared table; general filler families absorb the delta so the window total stays 84/69.
# Twin families keep a ~50/50 hot ratio in EVERY window (the boost grows
# both sides), so constant-fire bots stay capped.
SERVING_QUOTA_W = {
    "base": [("A_jewelry", 12), ("A_giftcard", 12), ("A_electronics", 14),
             ("A_plain", 6), ("A_offscope", 4), ("B_cap", 8),
             ("B_standing", 6), ("C_payee", 14), ("C_threshold", 8)],
    1: [("A_jewelry", 20), ("A_giftcard", 12), ("A_electronics", 14),
        ("A_plain", 4), ("A_offscope", 4), ("B_cap", 6),
        ("B_standing", 4), ("C_payee", 14), ("C_threshold", 6)],
}
TEST_QUOTA_W = {
    "base": [("A_jewelry", 8), ("A_giftcard", 8), ("A_electronics", 12),
             ("A_plain", 4), ("A_offscope", 4), ("B_cap", 6),
             ("B_standing", 6), ("C_payee", 14), ("C_threshold", 7)],
    1: [("A_jewelry", 24), ("A_giftcard", 8), ("A_electronics", 12),
        ("A_plain", 3), ("A_offscope", 4), ("B_cap", 5),
        ("B_standing", 4), ("C_payee", 12), ("C_threshold", 5)],
}


def serving_quota(w: int):
    return SERVING_QUOTA_W.get(w, SERVING_QUOTA_W["base"])


def test_quota(w: int):
    return TEST_QUOTA_W.get(w, TEST_QUOTA_W["base"])


# half of each twin family carries the rule-firing condition, in every
# window (beat windows boost hot and cold together)
def hot_n(split: str, family: str, w: int) -> int:
    base = {"serving": {"A_jewelry": 6, "A_giftcard": 6, "B_standing": 3},
            "test": {"A_jewelry": 4, "A_giftcard": 4, "B_standing": 3}}
    n = base[split].get(family, 0)
    if w == 1:      # D1 beat: jewelry doubles; B_standing shrinks with quota
        if family == "A_jewelry":
            n = 10 if split == "serving" else 12
        if family == "B_standing":
            n = 2
    return n


# C_payee draws a fixed roster POSITION per index; None = clean control.
# Positions 0-2 are live from W0; 3-4 activate at E2 (W3) — BEFORE the
# event they appear as clean-history payees (approve), so the flip is
# observable against each entity's own past. The multiset is FIXED per
# (split, window) — only the order is shuffled — so coverage is guaranteed.
def _payee_plan(split: str, w: int) -> list:
    live5 = R.truth(w)["C_restricted_payees"] >= 5
    if split == "serving":
        plan = ([0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, None, None] if live5
                else [0, 0, 0, 1, 1, 1, 2, 2, 3, 4, None, None, None, None])
    else:
        if live5:
            # Event-window test emphasizes the two newly activated roster
            # positions: eight tickets flip relative to W2, with older/live
            # and clean payees retained as controls.
            plan = [3, 3, 3, 3, 4, 4, 4, 4,
                    0, 1, None, None, None, None]
        elif w == 1:
            plan = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, None, None]
        else:
            plan = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, None, None, None, None]
    _random.Random(f"payee:{SEED}:{split}:{w}").shuffle(plan)
    return plan


def _hot_positions(amounts, hot: int, seed_key: str, ref=None) -> set:
    """WHICH instances carry the rule-firing condition.

    Two constraints:
    - read in amount order the condition must alternate >= 2 times (no
      monotone amount rule can express it);
    - the rank PATTERN must vary per (family, split, window), so no single
      global pattern (e.g. "the cheapest instance always fires") exists.
      Seeded rejection sampling gives every block its own pattern and lets
      rank 0 / rank n-1 be controls too.

    amounts: {k: amount}. Returns the set of k carrying the condition.
    """
    n = len(amounts)
    if n - hot <= 0 or n < 3:
        return set(amounts)
    order = [k for k, _ in sorted(amounts.items(), key=lambda kv: kv[1])]
    rnd = _random.Random(f"hot:{SEED}:{seed_key}")

    def nn_transfer(hot_keys):
        """Accuracy of 'copy the hot/cold label of the nearest-amount ref
        instance'; test placement keeps it low so the hidden condition
        cannot be recovered from amounts. ref = {amount: is_hot} of the
        serving block."""
        ref_pts = sorted(ref.items())
        ok = 0
        for k, a in amounts.items():
            nearest = min(ref_pts, key=lambda p: abs(p[0] - a))
            ok += nearest[1] == (k in hot_keys)
        return ok / n

    best, best_t = None, 2.0
    for _ in range(500):
        hot_ranks = set(rnd.sample(range(n), hot))
        pat = [r in hot_ranks for r in range(n)]
        if sum(1 for i in range(1, n) if pat[i] != pat[i - 1]) < 2:
            continue
        keys = {order[r] for r in hot_ranks}
        if ref is None:
            return keys
        t = nn_transfer(keys)
        if t <= 0.6:
            return keys
        if t < best_t:
            best, best_t = keys, t
    return best if best is not None else {order[r] for r in range(hot)}


def _tenure(family: str, k: int, split: str, w: int) -> int:
    """Account tenure — a real, varied field."""
    return 1 + (_h(SEED, "ten", family, split, w, k) % 9)


def _vf(family: str, k: int, split: str, w: int) -> str:
    """Non-causal verification variety outside the gift-card rule."""
    return "basic" if _h(SEED, "vf", family, k, split, w) % 10 < 3 else "verified"


def _spec(family: str, k: int, split: str, w: int = 0, hot_pos=None):
    """The k-th instance of a family -> (kind, fields).

    Rule families are twins: `hot_pos` (a permuted position set) decides which
    instances carry the rule-firing condition, so the hidden field is
    independent of the amount ladder. Boundary families walk a DESIGNED amount
    ladder that straddles every limit the timeline uses, so each boundary
    keeps >=4 discriminating probes per window.
    """
    hot = hot_n(split, family, w)
    is_hot = (k in hot_pos) if hot_pos is not None else (k < hot)
    ten = _tenure(family, k, split, w)
    if family == "A_jewelry":
        return ("txn", dict(verification=_vf(family, k, split, w), standing="good", tenure=ten,
                            category="jewelry",
                            region="overseas" if is_hot else "domestic",
                            amount=_spread(60, 900, family, k, split, w)))
    if family == "A_giftcard":
        return ("txn", dict(verification="basic" if is_hot else "verified",
                            standing="good", tenure=ten, category="gift_cards",
                            region="domestic",
                            amount=_spread(60, 900, family, k, split, w)))
    if family == "A_electronics":
        # S2: docs 500 vs truth 650. Supply target 8 serving / 6 test
        # in-gap probes per window. Serving BRACKETS the truth (626 approve /
        # 668 escalate); every base keeps a >=20 margin from 500/650 so the
        # 0-20 cents never cross a boundary, and test in-gap probes stay
        # >=20 under the serving bracket base (determinability).
        amt = ([404.0, 434.0, 464.0,                       # below 500: general
                502.0, 520.0, 538.0, 556.0, 574.0, 592.0,  # (500,650): adapt
                610.0, 626.0,
                668.0, 724.0, 784.0][k] if split == "serving"
               else [420.0, 450.0, 480.0,                  # below: general
                     504.0, 524.0, 544.0, 564.0, 584.0, 604.0,  # gap: adapt
                     688.0, 744.0, 800.0][k]) + _cents(family, split, w, k)
        return ("txn", dict(verification=_vf(family, k, split, w), standing="good", tenure=ten,
                            category="electronics", region="domestic",
                            amount=amt))
    if family == "A_offscope":
        # Off-scope controls put each hidden condition (overseas, basic
        # verification, chargeback) OUTSIDE its rule's scope with GT
        # approve, so an UNSCOPED rule ("deny anything overseas") is
        # penalized and rule SCOPE becomes measurable.
        form = k % 2
        if form == 0:      # overseas + basic, outside both txn scopes
            category = "travel" if w % 2 == 0 else "groceries"
            return ("txn", dict(verification="basic", standing="good",
                                tenure=ten, category=category,
                                region="overseas",
                                amount=_spread(60, 900, family, k, split, w)))
        # chargeback x transfer -> approve (standing rules limit requests only)
        return ("transfer", dict(verification=_vf(family, k, split, w),
                                 standing="recent_chargeback", tenure=ten,
                                 payee_id=_CLEAN_PAYEES[
                                     _h(SEED, "os", split, w, k) % len(_CLEAN_PAYEES)],
                                 amount=_spread(1200, 7900, family, k, split, w)))

    if family == "A_plain":
        cat = ["groceries", "utilities", "dining", "fitness", "travel"][
            (k + _h(SEED, "pc", split, w)) % 5]
        return ("txn", dict(verification=_vf(family, k, split, w), standing="good", tenure=ten,
                            category=cat, region="domestic",
                            amount=_spread(60, 900, family, k, split, w)))
    if family == "B_cap":
        # the cap is DOCUMENTED TRUE (5000/5000) — a pure general
        # boundary family. Lists are interleaved below/above so the W1
        # truncation (10 -> 8 serving, 8 -> 6 test) keeps the straddle and
        # the 4960/5020 serving bracket in every window.
        amt = ([4960.0, 5020.0, 3600.0, 5320.0, 3980.0, 5620.0, 4360.0,
                5920.0, 4740.0, 5240.0][k] if split == "serving"
               else [4940.0, 5040.0, 4020.0, 5340.0, 4420.0, 5640.0,
                     4820.0, 5940.0][k]) + _cents(family, split, w, k)
        return ("limit", dict(verification=_vf(family, k, split, w), standing="good",
                              tenure=ten, requested=amt))
    if family == "B_standing":
        # Amounts depend on k ONLY, never on is_hot. Slots k=0/k=1
        # are the HIGH band (above every cap): k=0 is forced HOT (standing
        # and cap co-fire -> account_standing, the precedence probe), k=1 is
        # forced CONTROL (good standing, same band -> cap_exceeded), so the
        # high band contains BOTH standings and BOTH answers. Remaining
        # slots sit under the docs cap; hot spread among them by amount rank.
        req = (_spread(5100, 6400, family, k, split, w) if k < 2
               else _spread(2600, 4000, family, k, split, w))
        return ("limit", dict(verification=_vf(family, k, split, w),
                              standing="recent_chargeback" if is_hot else "good",
                              tenure=ten, requested=req))
    if family == "C_payee":
        # the payee assignment is permuted per window too, so "which payee"
        # is independent of the amount rung
        plan = _payee_plan(split, w)
        pid = (R.RESTRICTED_ROSTER[plan[k]] if plan[k] is not None
               else _CLEAN_PAYEES[_h(SEED, "cp", split, w, k) % len(_CLEAN_PAYEES)])
        # above-threshold amounts are attached PLAN-AWARE: the first two
        # LIVE roster positions co-fire restricted > reportable each window.
        # Future-roster (pre-activation) and clean draws stay
        # under the threshold so their approve history is single-cause.
        live_n = R.truth(w)["C_restricted_payees"]
        live_ks = [i for i, p in enumerate(plan)
                   if p is not None and p < live_n][:2]
        amt = (_spread(8200, 9600, family, k, split, w) if k in live_ks
               else _spread(1200, 7900, family, k, split, w))
        return ("transfer", dict(verification=_vf(family, k, split, w), standing="good",
                                 tenure=ten, payee_id=pid, amount=amt))
    # C_threshold: the reporting threshold is DOCUMENTED TRUE
    # (8000/8000) — a pure general boundary family. Interleaved so the W1
    # truncation (8 -> 6 both splits) keeps the straddle and the 7960/8020
    # serving bracket in every window.
    amt = ([7960.0, 8020.0, 6200.0, 8620.0, 6800.0, 9220.0, 7400.0,
            9820.0][k] if split == "serving"
           else [7940.0, 8040.0, 6400.0, 8640.0, 7000.0, 9240.0, 7600.0,
                 9840.0][k]) + _cents(family, split, w, k)
    return ("transfer", dict(verification=_vf(family, k, split, w), standing="good",
                             tenure=ten,
                             payee_id=_CLEAN_PAYEES[
                                 _h(SEED, "ct", split, w, k) % len(_CLEAN_PAYEES)],
                             amount=amt))


def verdict_under(task, state, data):
    """Recompute a ticket verdict under an arbitrary L1 policy state."""
    acc = data["accounts"][task.z["account_id"]]
    kind = task.z["kind"]
    if kind == "txn":
        mer = data["merchants"][
            data["transactions"][task.z["case_id"]]["merchant_id"]
        ]
        if (state["A_R1_overseas_jewelry_blocked"]
                and mer["category"] == "jewelry"
                and mer["region"] == "overseas"):
            return "deny", "corridor_blocked"
        if (state["A_R2_giftcard_needs_verified"]
                and mer["category"] == "gift_cards"
                and acc["verification"] == "basic"):
            return "deny", "verification_required"
        if (mer["category"] == "electronics"
                and task.z["amount"] > state["A_T1_electronics_limit"]):
            return "escalate", "over_limit"
        return "approve", None
    if kind == "limit":
        if (state["B_R3_chargeback_blocks"]
                and acc["standing"] == "recent_chargeback"):
            return "deny", "account_standing"
        if task.z["amount"] > state["B_cap"]:
            return "deny", "cap_exceeded"
        return "approve", None
    live = set(R.RESTRICTED_ROSTER[:state["C_restricted_payees"]])
    if task.z["payee_id"] in live:
        return "deny", "restricted_payee"
    if task.z["amount"] > state["C_reporting_threshold"]:
        return "escalate", "reportable_amount"
    return "approve", None


def build_world_and_tasks(*, prune_test: bool = True):
    accounts: Dict[str, dict] = {}
    cases = {"transactions": {}, "limit_requests": {}, "transfers": {}}
    serving: List[Task] = []
    test_by_w: Dict[int, List[Task]] = {}
    acc_idx = 0
    gidx = 0
    t_serving = 0

    def make(family, k, w, split, hot_pos):
        nonlocal acc_idx, gidx, t_serving
        kind, f = _spec(family, k, split, w, hot_pos)
        # ids must carry NO information: they are opaque hashes of a
        # private key, not generation ordinals or window/split markers.
        acc_num = _h(SEED, 'accid', w, split, family, k) % OPAQUE_ID_SPAN
        while f"ACC-{acc_num + OPAQUE_ID_BASE}" in accounts:
            acc_num = (acc_num + 1) % OPAQUE_ID_SPAN
        acc_id = f"ACC-{acc_num + OPAQUE_ID_BASE}"
        acc = _account(acc_id, f["verification"], f["standing"], f["tenure"],
                       _h(SEED, "accd", w, split, family, k), gidx)
        accounts[acc["account_id"]] = acc
        shells = SHELLS[kind]
        shell_i = _h(SEED, "shell", w, split, family, k) % len(shells)
        tag = {"txn": "TXN", "limit": "LIM", "transfer": "TRF"}[kind]
        cid_num = _h(SEED, 'caseid', w, split, family, k) % OPAQUE_ID_SPAN
        _all_cases = set().union(*(cases[tb] for tb in cases))
        while f"{tag}-{cid_num + OPAQUE_ID_BASE}" in _all_cases:
            cid_num = (cid_num + 1) % OPAQUE_ID_SPAN
        cid = f"{tag}-{cid_num + OPAQUE_ID_BASE}"
        if kind == "limit":
            # a "limit increase" below the current limit is incoherent:
            # pin the current limit under the request
            amt_req = f["requested"]
            acc["credit_limit"] = min(acc["credit_limit"],
                                      int(amt_req - 400) // 100 * 100)
        if kind == "txn":
            mer = _merchant(f["category"], f["region"],
                            f"{split}:{family}:{k}:{w}", split, w)
            cases["transactions"][cid] = {
                "case_id": cid, "kind": "card_transaction",
                "account_id": acc["account_id"],
                "merchant_id": mer["merchant_id"], "amount": f["amount"],
                "status": "pending", "reason": None}
            decision, reason = R.decide_txn(acc, mer, f["amount"], w)
            sl = R.slice_of("txn", acc, mer, f["amount"], w)
            ename, eid, amt = mer["name"], mer["merchant_id"], f["amount"]
        elif kind == "limit":
            cases["limit_requests"][cid] = {
                "case_id": cid, "kind": "limit_increase",
                "account_id": acc["account_id"],
                "current_limit": acc["credit_limit"],
                "requested_limit": f["requested"], "status": "pending",
                "reason": None}
            decision, reason = R.decide_limit(acc, f["requested"], w)
            sl = R.slice_of("limit", acc, f["requested"], w)
            ename, eid, amt = "", "", f["requested"]
        else:
            payee = _PAYEES[f["payee_id"]]
            cases["transfers"][cid] = {
                "case_id": cid, "kind": "outbound_transfer",
                "account_id": acc["account_id"], "payee_id": payee["payee_id"],
                "amount": f["amount"], "status": "pending", "reason": None}
            decision, reason = R.decide_transfer(acc, payee, f["amount"], w)
            sl = R.slice_of("transfer", acc, payee, f["amount"], w)
            ename, eid, amt = payee["name"], payee["payee_id"], f["amount"]

        tool = {"approve": "approve_case", "deny": "deny_case",
                "escalate": "escalate_case"}[decision]
        kwargs = {"case_id": cid}
        if reason:
            kwargs["reason"] = reason
        instruction = shells[shell_i].format(cid=cid, acc=acc["account_id"],
                                             amt=amt, ename=ename, eid=eid)
        _rf = {"cid": cid, "acc": acc["account_id"], "amt": amt,
               "ename": ename, "eid": eid}
        if split == "serving":
            t_serving += 1
            t = t_serving
        else:
            t = w * R.PER_WINDOW_SERVING + 1
        gidx += 1
        # task ids are opaque (window/slice/split would be an answer key
        # if the id were ever echoed); metadata lives in z
        tid = "bank_" + hashlib.md5(
            f"{SEED}|{w}|{split}|{family}|{k}".encode()).hexdigest()[:10]
        task = Task(task_id=tid,
                    user_id="banking", instruction=instruction,
                    actions=[Action(name=tool, kwargs=kwargs)], answer=None,
                    z={"t": t, "window": w, "tier": "L1", "slice": sl,
                       "manual_slice": sl,
                       "family": family, "kind": kind, "split": split,
                       "case_id": cid, "account_id": acc["account_id"],
                       "amount": amt, "shell": shell_i,
                       "category": f.get("category"), "region": f.get("region"),
                       "payee_id": f.get("payee_id"), "tenure": f["tenure"],
                       "standing": f["standing"],
                       "verification": f["verification"],
                       "decision": decision, "reason": reason})
        task._rf = _rf
        return task


    _shell_rr = {}

    def _rebalance_shells(pool, w, split):
        """Deterministic shell balancing: within each
        (kind, decision) group the shells are assigned round-robin, and
        the rotation CONTINUES across windows (a per-group counter), so
        even small groups distribute evenly over the whole timeline and
        every shell's verdict mix collapses to the kind's base rate."""
        if split == "test":
            assigned = balanced_shell_assignment(
                pool,
                shell_count_by_kind={kind: len(shells)
                                     for kind, shells in SHELLS.items()},
                seed=SEED,
            )
            for t in pool:
                si = assigned[t.task_id]
                t.z["shell"] = si
                t.instruction = SHELLS[t.z["kind"]][si].format(**t._rf)
                del t._rf
            return
        groups = {}
        for t in pool:
            # keyed on (kind, FAMILY, decision): balancing inside each
            # family leaves no within-family correlation between shell
            # and verdict
            groups.setdefault((t.z["kind"], t.z["family"], t.z["decision"]),
                              []).append(t)
        for (kind, fam, dec), grp in sorted(
                groups.items(), key=lambda kv: (kv[0][0], kv[0][1],
                                                str(kv[0][2]))):
            grp = sorted(grp, key=lambda t: _h(SEED, "shb", t.task_id))
            start = _shell_rr.get((split, kind, fam, dec), 0)
            for i, t in enumerate(grp):
                si = (start + i) % len(SHELLS[kind])
                t.z["shell"] = si
                t.instruction = SHELLS[kind][si].format(**t._rf)
            _shell_rr[(split, kind, fam, dec)] = (start + len(grp)) % len(SHELLS[kind])
        for t in pool:
            del t._rf

    def hot_for(family, n, w, split):
        """Two-pass: amounts do not depend on the hidden condition, so compute
        them first, then place the condition by amount rank. B_standing pins
        k=0 hot / k=1 control — its HIGH band (above every cap) must carry
        BOTH standings."""
        hot = hot_n(split, family, w)
        if not hot:
            return set()
        amounts = {k: _spec(family, k, split, w, set())[1].get(
            "amount", _spec(family, k, split, w, set())[1].get("requested"))
            for k in range(n)}
        ref = None
        if split == "test":
            # condition the TEST placement on this window's SERVING placement
            # so nearest-amount label transfer stays <= 60%
            n_sv = dict(serving_quota(w))[family]
            sv_hp = hot_for(family, n_sv, w, "serving")
            sv_amounts = {k: _spec(family, k, "serving", w, set())[1].get(
                "amount", _spec(family, k, "serving", w, set())[1].get("requested"))
                for k in range(n_sv)}
            ref = {a: (k in sv_hp) for k, a in sv_amounts.items()}
        if family == "B_standing":
            low = {k: a for k, a in amounts.items() if k >= 2}
            low_ref = ({a: h for a, h in ref.items()} if ref else None)
            if len(low) < 3:
                # W1 shrinks the family to 4 (2 low slots): _hot_positions
                # would return BOTH low keys (its n<3 fallback), skewing the
                # hot ratio to 75% — pick the low hot by seeded coin instead
                pick = sorted(low)[_h(SEED, "bslow", split, w) % len(low)]
                hp = {0, pick} if hot >= 2 else {0}
            else:
                hp = set(_hot_positions(low, hot - 1, f"{family}:{split}:{w}",
                                        ref=low_ref))
                hp |= {0}
        else:
            hp = _hot_positions(amounts, hot, f"{family}:{split}:{w}", ref=ref)
        # PHANTOM-AMOUNT GUARD: the amounts used for placement must equal
        # what the final spec ships
        for k in range(n):
            f2 = _spec(family, k, split, w, hp)[1]
            real = f2.get("amount", f2.get("requested"))
            assert real == amounts[k], (family, k, w, split, amounts[k], real)
        return hp

    for w in range(R.N_WINDOWS):
        wserv = []
        for family, n in serving_quota(w):
            hp = hot_for(family, n, w, "serving")
            for k in range(n):
                wserv.append(make(family, k, w, "serving", hp))
        # shuffle so the position within a window does not predict the
        # family (hence most of the answer), then re-stamp t
        _random.Random(f"order:{SEED}:{w}").shuffle(wserv)
        for j, t_ in enumerate(wserv):
            t_.z["t"] = w * R.PER_WINDOW_SERVING + j + 1
        test_by_w[w] = []
        for family, n in test_quota(w):
            hp = hot_for(family, n, w, "test")
            for k in range(n):
                test_by_w[w].append(make(family, k, w, "test", hp))
        # the TEST list is shuffled too, so position does not reveal the
        # family and any TEST_PER_WINDOW truncation samples all families
        _random.Random(f"torder:{SEED}:{w}").shuffle(test_by_w[w])
        _rebalance_shells(wserv, w, "serving")
        serving.extend(wserv)

    data = {"accounts": accounts, "merchants": copy.deepcopy(_MERCHANTS),
            "payees": copy.deepcopy(_PAYEES),
            "transactions": cases["transactions"],
            "limit_requests": cases["limit_requests"],
            "transfers": cases["transfers"],
            "_episode_view": {"kind": "banking"}}
    if prune_test:
        alternating = categorical_alternation_probe_ids(
            test_by_w,
            specs=(("A_jewelry", "region", None),
                   ("A_giftcard", "verification", None),
                   ("B_standing", "standing", None)),
            seed=SEED,
        )
        test_by_w = balance_change_adapt(
            test_by_w,
            rules=R,
            data=data,
            verdict_fn=verdict_under,
            seed=SEED,
            # W2 deliberately has no policy event: retain a compact sample
            # that measures memory without letting it dominate adapt.
            retention_budget_by_window={2: 20},
            target_adapt_by_window={0: 16, 1: 24, 2: 20, 3: 24},
        )
        protected = event_general_probe_ids(
            test_by_w, rules=R, data=data, verdict_fn=verdict_under
        )
        coverage = value_coverage_probe_ids(
            test_by_w,
            # Twin alternation and constant-answer protection below already
            # imply two values for the categorical/adapt families.
            specs=(("A_electronics", "decision"),
                   ("B_cap", "decision"),
                   ("C_threshold", "decision")),
            seed=SEED,
        )
        for w, task_ids in coverage.items():
            protected[w].update(task_ids)
        for w, task_ids in alternating.items():
            protected[w].update(task_ids)
        constant_controls = per_window_constant_answer_balance_ids(
            test_by_w,
            families=("A_jewelry", "A_giftcard", "C_payee"),
            seed=SEED,
            max_rate=0.80,
            preselected_by_window=protected,
            active_windows_by_family={"A_jewelry": {1, 2, 3}},
        )
        for w, task_ids in constant_controls.items():
            protected[w].update(task_ids)
        numeric = numeric_side_balance_ids(
            test_by_w,
            specs=(
                ("A_electronics", lambda task, _w: task.z["amount"],
                 lambda w: R.truth(w)["A_T1_electronics_limit"],
                 lambda task, _w: task.z.get("reason") in
                 (None, "over_limit")),
                ("B_cap", lambda task, _w: task.z["amount"],
                 lambda w: R.truth(w)["B_cap"],
                 lambda task, _w: task.z.get("reason") in
                 (None, "cap_exceeded")),
                ("C_threshold", lambda task, _w: task.z["amount"],
                 lambda w: R.truth(w)["C_reporting_threshold"],
                 lambda task, _w: task.z.get("reason") in
                 (None, "reportable_amount")),
            ),
            seed=SEED,
            preselected_by_window=protected,
        )
        for w, task_ids in numeric.items():
            protected[w].update(task_ids)
        # Keep co-firing standing/cap controls: they establish that the
        # standing reason takes precedence when both rules apply.
        for w, tasks in test_by_w.items():
            state = R.truth(w)
            for task in tasks:
                if (task.z.get("slice") == "general"
                        and task.z.get("family") == "B_standing"
                        and task.z.get("standing") == "recent_chargeback"
                        and task.z.get("amount") > state["B_cap"]):
                    protected[w].add(task.task_id)
        test_by_w = prune_general_repetitions(
            test_by_w,
            target_per_window=R.PER_WINDOW_TEST,
            seed=SEED,
            preferred_families_by_window={
                w: {"B_cap", "B_standing"} for w in range(R.N_WINDOWS)
            },
            protected_task_ids_by_window=protected,
            protected_families_by_window={
                w: {"A_offscope"} for w in range(R.N_WINDOWS)
            },
        )
    align_clean_payee_anchors(
        serving,
        test_by_w,
        transfer_cases=cases["transfers"],
        payees=_PAYEES,
        clean_payees=_CLEAN_PAYEES,
        reporting_threshold_by_window=(
            lambda window: R.truth(window)["C_reporting_threshold"]
        ),
        seed=SEED,
    )
    _rebalance_shells(
        [task for w in range(R.N_WINDOWS) for task in test_by_w[w]],
        0,
        "test",
    )

    return data, serving, test_by_w


PROTOCOL_NOTES = """Issue one final decision per case, then finish
(scoring is outcome-based: the case's final state is what is judged, and
repeat decision calls are rejected by the tools)."""


def _system_prompt(tools) -> str:
    head = f"""You are a bank operations-review agent. Today's date is {TODAY}.

# Environment
You review pending cases of three kinds — card transactions (TXN-...),
credit-limit increase requests (LIM-...) and outbound transfers (TRF-...) —
and give each EXACTLY ONE final decision: approve, deny, or escalate to
manual review (deny/escalate cite a reason code). Decisions are final.
The bank's ACTUAL practice can differ from its documents: some rules are not
written down, and some written values may be out of date — in either
direction. You are judged against actual current practice, and judged
outcomes are your only ground truth about it.

# Reference documents
The risk POLICY and the WORKFLOW guide are NOT included here. Retrieve
sections with read_docs(doc, section).
Retrievable sections:
{D.system_prompt_toc()}

# Tools
{tool_signatures(tools)}
Ids are exact strings in the forms TXN-######, LIM-######, TRF-######,
ACC-######, MER-###, PAY-### — always use the exact id from the case."""
    return compose_system_prompt(head, "banking", PROTOCOL_NOTES)


def get_bundle() -> Dict[str, Any]:
    data, serving, test_by_w = build_world_and_tasks()
    tools = [D.ReadDocs, GetCaseDetails, GetAccountDetails, GetMerchantDetails,
             GetPayeeDetails, ApproveCase, DenyCase, EscalateCase]
    meta = {"scenario": "banking_l1", "domain": "banking", "tier": "L1",
            "system_prompt": _system_prompt(tools)}
    return {
        "meta": meta, "id": "banking_L1",
        "data": data,
        "load_data": lambda d=data: copy.deepcopy(d),
        "tools": tools,
        "policy_md": D.merged_md("policy"),
        "workflow_md": D.merged_md("workflow"),
        "docs_toc": D.system_prompt_toc(),
        "serving": serving,
        "test_set": lambda w, tb=test_by_w: tb.get(w, []),
        "rules": R,
    }
