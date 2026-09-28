"""Banking L2 world: the L1 request surface under a REVISION drift
schedule, with tier-disjoint visible identities and language.

Construction disciplines:
- one dedicated account + one dedicated case per ticket, ZERO reuse across
  the tier (global freshness);
- serving and test amounts are exactly disjoint (numeric holdout): both
  splits draw cents from the same distribution; DESIGNED ladders use
  split-distinct bases (a declared, answer-free split marker — verdicts are
  uniform inside every shared band) with half-range cents at band edges;
- family quotas follow a DECLARED per-window table: each drift's
  discriminating supply peaks in its event window (beat thickening),
  general fillers absorb the delta;
- twins: same surface, flipped hidden field (overseas/domestic,
  chargeback/good — L2 has no verification rule and no payee roster);
- instruction shells rotate by ticket index, independent of the verdict;
- episode view: an episode sees the shared world plus ITS OWN case only.
"""

from __future__ import annotations

import copy
import hashlib
import math
import random as _random
from typing import Any, Dict, List

from ....engine.types import Action, Task
from . import docs as D
from ....prompts import compose_system_prompt, tool_signatures
from . import rules as R
from .tools import (ApproveCaseL2 as ApproveCase, DecideBatchL2 as DecideBatch,
                    DenyCaseL2 as DenyCase,
                    EscalateCaseL2 as EscalateCase, ExecuteLegL2 as ExecuteLeg,
                    GetAccountDetailsL2 as GetAccountDetails,
                    GetCaseDetailsL2 as GetCaseDetails, GetMerchantDetails,
                    GetPayeeDetailsL2 as GetPayeeDetails)
from ..test_prune import (balance_change_adapt, balanced_shell_assignment,
                          categorical_alternation_probe_ids,
                          constant_answer_balance_ids,
                          categorical_value_adapt_ids,
                          multipart_control_ids,
                          numeric_side_adapt_ids,
                          numeric_side_balance_ids,
                          per_window_constant_answer_balance_ids,
                          event_general_probe_ids, prune_general_repetitions,
                          value_coverage_probe_ids)

TODAY = "2026-08-08"
SEED = "banking_v2_L2_compound_v1"
ENTITY_ID_BASE = 400
ENTITY_ID_SPAN = 200
OPAQUE_ID_BASE = 400000
OPAQUE_ID_SPAN = 200000

# the pool must exceed the account count — holders are FIRST x LAST and
# every one of them is unique (48x40 = 1920 names). Tier disjointness comes
# from the middle initial (L1 " L. " / L2 " M. " / L3 " R. "), so extending
# a pool cannot collide with another tier.
FIRST = ["Ava", "Noah", "Mia", "Liam", "Zoe", "Ethan", "Ivy", "Lucas", "Nora",
         "Owen", "Ruth", "Felix", "June", "Hugo", "Lena", "Marco",
         "Anouk", "Bassam", "Corin", "Dagny", "Emeka", "Fenna", "Gaspar", "Hana", "Ilias", "Jetta", "Kofi", "Lucia", "Matteo", "Nell", "Otthild", "Pavel",
         "Quinn", "Rasmus", "Sanne", "Tobias", "Ulla", "Viggo", "Wren", "Xenia",
         "Yusuf", "Zita", "Arne", "Berit", "Caius", "Dilara", "Eero", "Freja"]
LAST = ["Alvarez", "Becker", "Chen", "Dawson", "Egan", "Fischer", "Grant",
        "Hopkins", "Iqbal", "Jensen", "Kovac", "Lindqvist", "Moreau", "Novak",
        "Ortiz", "Petrov",
        "Galloway", "Hirsch", "Ibarra", "Joshi", "Kaminski", "Lindau", "Mbeki", "Nystrom", "Okafor", "Pellegrini", "Quiroga", "Rasmussen", "Sindelar", "Toivonen", "Ulmer", "Vieira",
        "Wexford", "Yaremchuk", "Zubiri", "Ashworth", "Bardem", "Cortese", "Drakos", "Eklund"]
STATES = ["CA", "NY", "TX", "WA", "IL", "MA", "CO", "GA"]

# 8-12 names per category (electronics 12): two-thirds serving-only,
# the rest test-only —
# GLOBALLY (the rules key on category x region, never on the entity, and
# shared merchants let id-recall reproduce the corridor rule). Payees are
# ALSO split-disjoint per age class (L2 has no roster — the payee-relevant
# divergence is the ATTRIBUTE clause added_days_ago < 30, and test
# entities are never served, so payee-id memory sits at base rate).
MERCHANT_NAMES = {
    "electronics": ["Bytecrest", "Signal Works", "Coil & Core", "Fuse Alley", "Diode Bay", "Nordwatt", "Cascade Circuitry", "Pylon & Sons", "Voltbench", "Copperwire Co", "Halide Electronics", "Twin Diode"],
    "travel": ["Latitude Trips", "Windward Fare", "Copper Compass", "Transit Blue", "Farline", "Vector Voyages", "Amberline Air", "Portage Trips"],
    "jewelry": ["Lumen Setting", "Vermeil House", "Stonebright", "Filigree & Co", "Bezel Row", "Astral Setting", "Marrowstone Fine", "Gilt Anchor"],
    "gift_cards": ["ChoiceToken", "Balance Tree", "CardHarbor", "Prepay Point", "VoucherWell", "Cardsmith Co", "Tinsel Token", "Redeemly"],
    "groceries": ["Harvest Aisle", "Butter & Stem", "Larder Lane", "Crisper Co", "Market Ninefold", "Bramblewood Market", "Split Oak Grocery", "Harrow Farms"],
    "utilities": ["GridNorth", "BluePipe Utility", "Wattage Co", "Cinder Gas", "Clearline Water", "Delta Main Power", "Quayside Water", "Emberline Gas"],
    "dining": ["Ember Counter", "Salt Meridian", "Copper Ladle", "Night Market Table", "Grove & Grain", "Sorrel & Smoke", "The Tin Ladle", "Meridian Supper"],
    "fitness": ["Pulse Loft", "Ironwood Gym", "Cadence Club", "Summit Circuit", "Tidal Fit", "Crag & Rope", "Northpace Gym", "Kindle Fitness"],
}

PAYEE_NAMES = ["Crescent Forwarding", "Alder & Vance Ltd", "Bluewater Consignment",
               "Ferrous Trading Co", "Kestrel Logistics", "Marrow Point Capital",
               "Osprey Freight", "Quarry Lane Partners", "Sable Exports",
               "Tern Harbor Group", "Umber Holdings", "Vantage Crossing",
               "Wicker & Frame Co", "Yarrow Imports", "Zephyr Clearing",
               "Granite Fold Ltd", "Copperline Remit", "Dunmore & Pratt",
               "Estuary Clearing", "Foxglove Trading", "Halyard Exports",
               "Ironbell Partners", "Juno Freightworks", "Kelp Harbor Co"]
# 4 jurisdictions rotate over the shuffled payee order so no jurisdiction
# marks an age class or a split (L2 has no roster)
JURISDICTIONS = ["US", "UK", "SG", "AE"]

SHELLS = {
    "txn": [
        "Review pending transaction {cid} on account {acc}: ${amt:.2f} at "
        "{ename} ({eid}). Decide per current practice.",
        "Queue item {cid}: account {acc} attempted a ${amt:.2f} charge at "
        "{ename} ({eid}). Please resolve it.",
        "A ${amt:.2f} charge from {ename} ({eid}) is on hold for account "
        "{acc} — case {cid}. Make the call.",
        "Authorization hold {cid}: {ename} ({eid}) submitted ${amt:.2f} "
        "against account {acc}. Give the final ruling.",
        "Card review needed — case {cid}. Account {acc}, merchant {ename} "
        "({eid}), amount ${amt:.2f}.",
        "Pending authorization {cid}: {ename} ({eid}) charged ${amt:.2f} "
        "to account {acc}. Enter a decision.",
        "Resolve card case {cid} for {acc}; the ${amt:.2f} merchant is "
        "{ename} ({eid}).",
    ],
    "limit": [
        "Limit-increase request {cid}: account {acc} asks to move its credit "
        "limit to ${amt:.2f}. Decide per current practice.",
        "Queue item {cid}: cardholder on account {acc} requests a "
        "${amt:.2f} credit limit. Please resolve it.",
        "Case {cid} — account {acc} applied for a limit of ${amt:.2f}. "
        "Make the call.",
        "Credit review {cid}: proposed new limit ${amt:.2f} on account "
        "{acc}. Give the final ruling.",
        "Underwriting queue item {cid} — account {acc}, requested limit "
        "${amt:.2f}.",
        "Credit-line case {cid}: {acc} asks for a ${amt:.2f} limit. Enter "
        "a decision.",
        "Resolve request {cid}; account {acc} proposes a new ceiling of "
        "${amt:.2f}.",
    ],
    "transfer": [
        "Outbound transfer {cid}: account {acc} is sending ${amt:.2f} to "
        "{ename} ({eid}). Decide per current practice.",
        "Queue item {cid}: a ${amt:.2f} transfer from account {acc} to "
        "{ename} ({eid}) is held for screening.",
        "Case {cid} — account {acc} instructed ${amt:.2f} to {ename} "
        "({eid}). Make the call.",
        "Wire review {cid}: ${amt:.2f} from account {acc}, beneficiary "
        "{ename} ({eid}). Give the final ruling.",
        "Screening hold — case {cid}. Payment of ${amt:.2f} to {ename} "
        "({eid}) from account {acc}.",
        "Beneficiary review {cid}: {acc} plans ${amt:.2f} to {ename} "
        "({eid}). Enter a decision.",
        "Resolve outbound-payment case {cid}; ${amt:.2f} leaves {acc} for "
        "{ename} ({eid}).",
    ],
}



# ORDERED-FALLBACK shells. Separate pools because the sentence has to rank
# three requests; they are balanced by the same machinery under their own pool
# key ("txn_fb" / "limit_fb" / "transfer_fb") so a fallback shell can never
# become a marker for the stop position.
SHELLS.update({
    "txn_fb": [
        "Card case {cid} on account {acc}: {ename} ({eid}) submitted "
        "${amt:.2f}. The cardholder would rather have it settled at "
        "${amt2:.2f}, and failing that at ${amt3:.2f}. Take the best one "
        "current practice allows.",
        "Queue item {cid}: account {acc} at {ename} ({eid}) — preferred "
        "${amt:.2f}, else ${amt2:.2f}, else ${amt3:.2f}. Resolve it.",
        "Authorization {cid} for {acc}: {ename} ({eid}) asked for "
        "${amt:.2f}; acceptable fallbacks are ${amt2:.2f} then ${amt3:.2f}.",
        "Ranked card request {cid} — account {acc}, merchant {ename} "
        "({eid}): first ${amt:.2f}, then ${amt2:.2f}, then ${amt3:.2f}.",
        "Case {cid}: {acc} wants ${amt:.2f} at {ename} ({eid}), or "
        "${amt2:.2f}, or ${amt3:.2f}. Give the final ruling.",
        "Cardholder on {acc} ranks three amounts at {ename} ({eid}) for case "
        "{cid}: ${amt:.2f}, ${amt2:.2f}, ${amt3:.2f}. Enter a decision.",
        "Resolve card case {cid} for {acc} at {ename} ({eid}); the ranked "
        "amounts are ${amt:.2f}, ${amt2:.2f}, ${amt3:.2f}.",
    ],
    "limit_fb": [
        "Credit case {cid}: account {acc} asks for a ${amt:.2f} limit, or "
        "${amt2:.2f}, or ${amt3:.2f} — in that order of preference.",
        "Queue item {cid}: {acc} would take ${amt:.2f}, else ${amt2:.2f}, "
        "else ${amt3:.2f}. Resolve it.",
        "Underwriting {cid} for account {acc}: preferred ceiling ${amt:.2f}, "
        "fallbacks ${amt2:.2f} and ${amt3:.2f}.",
        "Ranked limit request {cid} — account {acc}: ${amt:.2f}, then "
        "${amt2:.2f}, then ${amt3:.2f}.",
        "Case {cid}: {acc} proposes ${amt:.2f}, ${amt2:.2f} or ${amt3:.2f} "
        "as a new ceiling. Give the final ruling.",
        "Account {acc} ranks three ceilings in case {cid}: ${amt:.2f}, "
        "${amt2:.2f}, ${amt3:.2f}. Enter a decision.",
        "Resolve credit-line case {cid}; {acc} will accept ${amt:.2f}, "
        "${amt2:.2f} or ${amt3:.2f}.",
    ],
    "batch": [
        "Payment batch {cid} on account {acc}: {n} lines totalling "
        "${total:.2f} — {ename1} ({eid1}) ${amt1:.2f} · {ename2} ({eid2}) "
        "${amt2:.2f} · {ename3} ({eid3}) ${amt3:.2f} · {ename4} ({eid4}) "
        "${amt4:.2f}. Settle every line.",
        "Queue item {cid}: account {acc} submitted a {n}-line payment file "
        "worth ${total:.2f} — ${amt1:.2f} to {ename1} ({eid1}), ${amt2:.2f} to "
        "{ename2} ({eid2}), ${amt3:.2f} to {ename3} ({eid3}), ${amt4:.2f} "
        "to {ename4} ({eid4}). Resolve each line.",
        "Batch screening {cid} for {acc} (${total:.2f} across {n} lines): "
        "{ename1} ({eid1}) ${amt1:.2f}; {ename2} ({eid2}) ${amt2:.2f}; "
        "{ename3} ({eid3}) ${amt3:.2f}; {ename4} ({eid4}) ${amt4:.2f}.",
        "Outbound file {cid} — account {acc}, {n} beneficiaries, "
        "${total:.2f} total: {ename1} ({eid1}) ${amt1:.2f} / {ename2} "
        "({eid2}) ${amt2:.2f} / {ename3} ({eid3}) ${amt3:.2f} / {ename4} "
        "({eid4}) ${amt4:.2f}. Give a ruling per line.",
        "Case {cid}: {acc} is paying {n} beneficiaries (${total:.2f}) — "
        "{ename1} ({eid1}) ${amt1:.2f}, {ename2} ({eid2}) ${amt2:.2f}, "
        "{ename3} ({eid3}) ${amt3:.2f}, {ename4} ({eid4}) ${amt4:.2f}.",
        "Bulk payment {cid} from {acc}: ${amt1:.2f} {ename1} ({eid1}) · "
        "${amt2:.2f} {ename2} ({eid2}) · ${amt3:.2f} {ename3} ({eid3}) · "
        "${amt4:.2f} {ename4} ({eid4}). Total ${total:.2f} over {n} lines.",
        "Resolve payment file {cid} for {acc}; the {n} lines are {ename1} "
        "({eid1}) ${amt1:.2f}, {ename2} ({eid2}) ${amt2:.2f}, {ename3} "
        "({eid3}) ${amt3:.2f} and {ename4} ({eid4}) ${amt4:.2f}.",
    ],
    "transfer_fb": [
        "Payment case {cid}: account {acc} wants ${amt:.2f} to {ename} "
        "({eid}); failing that ${amt2:.2f} to {ename2} ({eid2}); failing "
        "that ${amt3:.2f} to {ename3} ({eid3}).",
        "Queue item {cid}: {acc} ranks three payments — ${amt:.2f} to "
        "{ename} ({eid}), ${amt2:.2f} to {ename2} ({eid2}), ${amt3:.2f} to "
        "{ename3} ({eid3}). Resolve it.",
        "Outbound case {cid} for {acc}: first choice ${amt:.2f} to {ename} "
        "({eid}), then ${amt2:.2f} to {ename2} ({eid2}), then ${amt3:.2f} "
        "to {ename3} ({eid3}).",
        "Ranked transfer {cid} — account {acc}: {ename} ({eid}) ${amt:.2f} · "
        "{ename2} ({eid2}) ${amt2:.2f} · {ename3} ({eid3}) ${amt3:.2f}.",
        "Case {cid}: {acc} would send ${amt:.2f} to {ename} ({eid}), or "
        "${amt2:.2f} to {ename2} ({eid2}), or ${amt3:.2f} to {ename3} "
        "({eid3}). Give the final ruling.",
        "Beneficiary options for case {cid} on {acc}: {ename} ({eid}) "
        "${amt:.2f}, {ename2} ({eid2}) ${amt2:.2f}, {ename3} ({eid3}) "
        "${amt3:.2f}. Enter a decision.",
        "Resolve payment case {cid}; {acc} ranks ${amt:.2f} to {ename} "
        "({eid}), ${amt2:.2f} to {ename2} ({eid2}), ${amt3:.2f} to "
        "{ename3} ({eid3}).",
    ],
})

# Shell wordings, per side.
#
# The verdict-balance contract is SCORED-side: a shell's majority share
# stays within 5pp of its pool's base rate over the test set, and a
# cell of n rows cannot resolve finer than 1/n.  Only txn has enough scored
# rows for seven cells (165 here); the rest sit at 30-76 and are scored
# over three.
#
# Serving is under no such cap, so the single-request pools keep all seven
# wordings there and none of their sentences is unreachable.  The four
# multi-part pools are the exception: ~30 serving rows over seven cells
# leaves verdict-pure shells, so they use three on both sides.  Their
# remaining four sentences keep every pool at >= 6 wordings and are the
# ONLY unreachable ones, so a sentence added to any other pool takes effect.
MULTIPART_POOLS = {p for p in SHELLS if p.endswith("_fb") or p == "batch"}
SERVING_SHELL_COUNT = {p: (3 if p in MULTIPART_POOLS else len(SHELLS[p]))
                       for p in SHELLS}
SCORED_SHELL_COUNT = {p: (7 if p == "txn" else 3) for p in SHELLS}
assert all(SCORED_SHELL_COUNT[p] <= SERVING_SHELL_COUNT[p] <= len(SHELLS[p])
           for p in SHELLS)
assert {p for p in SHELLS
        if SERVING_SHELL_COUNT[p] < len(SHELLS[p])} == MULTIPART_POOLS




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
    """Ids are assigned AFTER a deterministic shuffle of all payee names.
    L2 has NO restricted roster (the roster is an L1/L3 axis) — payees
    differ only in age (recent vs long-established, the docs-only
    new-payee clause) and jurisdiction texture. The last EIGHT shuffled
    names are RECENT (<30d), the rest long-established; ages hash on the
    NAME. Each age class is SPLIT-DISJOINT — the first
    half serves the serving stream, the second half is test-only — so a
    payee-id verdict memory carries nothing onto test and the new-payee
    clause must be learned as an ATTRIBUTE rule, not entity recall."""
    names = list(PAYEE_NAMES)
    _random.Random(f"pay:{SEED}").shuffle(names)
    out, roster = {}, []
    used = set()
    n_recent = 8
    for i, name in enumerate(names):
        n = ENTITY_ID_BASE + _h(SEED, "pid", name) % ENTITY_ID_SPAN
        while n in used:
            n = ENTITY_ID_BASE + (n - ENTITY_ID_BASE + 1) % ENTITY_ID_SPAN
        used.add(n)
        pid = f"PAY-{n:03d}"
        recent = i >= len(PAYEE_NAMES) - n_recent
        out[pid] = {"payee_id": pid, "name": name, "type": "business",
                    "jurisdiction": JURISDICTIONS[i % len(JURISDICTIONS)],
                    "added_days_ago": (3 + _h(SEED, "age", name) % 25
                                       if recent
                                       else 30 + _h(SEED, "age", name) % 900),
                    "_recent": recent, "_shuffle_i": i}
    return out, sorted(roster)


_MERCHANTS = _build_merchants()
_PAYEES, _ROSTER = _build_payees()
# split-disjoint pools per age class (order by shuffle index for stability)
def _pool(recent: bool, split: str):
    ids = sorted((p for p in _PAYEES if _PAYEES[p]["_recent"] == recent),
                 key=lambda p: _PAYEES[p]["_shuffle_i"])
    half = len(ids) // 2
    return ids[:half] if split == "serving" else ids[half:]
_RECENT_BY_SPLIT = {"serving": _pool(True, "serving"), "test": _pool(True, "test")}
_OLD_BY_SPLIT = {"serving": _pool(False, "serving"), "test": _pool(False, "test")}
for _s in ("serving", "test"):
    assert len(_RECENT_BY_SPLIT[_s]) == 4 and len(_OLD_BY_SPLIT[_s]) == 8
# the private split markers must not ship in the world data
for _p in _PAYEES.values():
    _p.pop("_recent"); _p.pop("_shuffle_i")


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
            "holder": (f"{FIRST[holder_i % len(FIRST)]} M. "
                       f"{LAST[(holder_i // len(FIRST)) % len(LAST)]}"),
            "state": STATES[(h // 7) % len(STATES)],
            "verification": verification, "standing": standing,
            "tenure_years": tenure,
            # floor 2500: today_total + amount tops out ~1940, so the
            # account's own limit can never contradict an approved charge
            "credit_limit": 2500 + 500 * (h % 5),
            # EVERY account carries a same-day approved total (if only the
            # daily-cap family had the field, its presence would name the
            # family); default <= 600 so no other family's amount can reach
            # the $2000 cap (max other txn 1280 + 600 = 1880)
            "today_total": round(20.0 * (h % 31), 2),
            # opened is derived from TODAY so the stated
            # tenure_years is exactly the number of full years elapsed
            "opened": (lambda m: f"{2026 - tenure - (1 if m > 8 else 0)}-"
                                 f"{m:02d}")(1 + h % 12)}


def _cents(family: str, split: str, w: int, k: int) -> float:
    """Per-ticket cents in {0.01 .. 19.99}, SAME distribution on both splits
    (any deterministic residue scheme is a readable split marker).
    Disjointness comes from the split-keyed hash."""
    # IDENTICAL distribution on both splits (any deterministic offset —
    # +10, +0.05, integer-vs-not — is itself a readable split marker);
    # disjointness comes from the hash being split-keyed.
    return round(0.01 + 0.01 * (_h(SEED, "cents", family, split, w, k) % 1999), 2)


def _grid(lo: float, step: float, k: int, split: str, family: str = "",
          w: int = 0) -> float:
    return round(lo + step * k + _cents(family, split, w, k), 2)


def _band_cents(family: str, split: str, w: int, k: int, edge) -> float:
    """Cents for DESIGNED ladder values. edge=None: full 0.01-19.99 range.
    At a band's EDGE base (shared by both splits) the two splits take
    disjoint half-ranges so every test probe is sandwiched by same-verdict
    serving evidence in ITS OWN window (determinability): edge='lo' = the
    band's lowest base -> serving low half, test high; edge='hi' = the
    band's highest base -> serving high half, test low. The half-range is a
    split-conditional residue ONLY at edge bases, where the verdict inside
    the band is uniform — knowing the split there reveals no answer."""
    if edge is None:
        return _cents(family, split, w, k)
    c = round(0.01 + 0.01 * (_h(SEED, "bcents2", family, split, w, k) % 999), 2)
    low_half = (split == "serving") == (edge == "lo")
    return c if low_half else round(c + 10.0, 2)


def _ladder(vals, family, k, split, w) -> float:
    base, edge = vals[k]
    return round(base + _band_cents(family, split, w, k, edge), 2)


def _spread(lo: float, hi: float, family: str, k: int, split: str, w: int) -> float:
    span = int((hi - lo) // 20)
    v = lo + 20 * (_h(SEED, "spread", family, split, w, k) % max(span, 1))
    v = v - (v % 20)
    return round(v + _cents(family, split, w, k), 2)


# ---- per-window quota tables (DECLARED window-varying quotas) ----
# Few policies, thick data, and the CHANGING policy's supply peaks in its
# event window (beat thickening: electronics at W1, corridor at W2, travel
# at W3, cap at W4 — one event per window, no consolidation window).
# Twins keep ~50/50 hot.
SERVING_QUOTA_W = {
    "base": [("A_jewelry", 8), ("A_elec_ovs", 8), ("A_electronics", 14),
             ("A_travel", 12), ("A_daily", 8), ("A_plain", 2),
             ("A_offscope", 2), ("B_cap", 7), ("B_standing", 6),
             ("C_newpayee", 12), ("C_threshold", 5)],
    2: [("A_jewelry", 12), ("A_elec_ovs", 12), ("A_electronics", 10),
        ("A_travel", 11), ("A_daily", 8), ("A_plain", 1),
        ("A_offscope", 2), ("B_cap", 7), ("B_standing", 5),
        ("C_newpayee", 12), ("C_threshold", 4)],
    3: [("A_jewelry", 8), ("A_elec_ovs", 8), ("A_electronics", 10),
        ("A_travel", 18), ("A_daily", 8), ("A_plain", 1),
        ("A_offscope", 2), ("B_cap", 7), ("B_standing", 6),
        ("C_newpayee", 12), ("C_threshold", 4)],
    4: [("A_jewelry", 8), ("A_elec_ovs", 8), ("A_electronics", 10),
        ("A_travel", 12), ("A_daily", 8), ("A_plain", 1),
        ("A_offscope", 2), ("B_cap", 12), ("B_standing", 6),
        ("C_newpayee", 12), ("C_threshold", 5)],
}
TEST_QUOTA_W = {
    "base": [("A_jewelry", 8), ("A_elec_ovs", 8), ("A_electronics", 14),
             ("A_travel", 12), ("A_daily", 10), ("A_plain", 2),
             ("A_offscope", 2), ("B_cap", 6), ("B_standing", 6),
             ("C_newpayee", 12), ("C_threshold", 7)],
    1: [("A_jewelry", 8), ("A_elec_ovs", 8), ("A_electronics", 19),
        ("A_travel", 12), ("A_daily", 10), ("A_plain", 1),
        ("A_offscope", 2), ("B_cap", 6), ("B_standing", 6),
        ("C_newpayee", 12), ("C_threshold", 5)],
    2: [("A_jewelry", 12), ("A_elec_ovs", 12), ("A_electronics", 11),
        ("A_travel", 12), ("A_daily", 8), ("A_plain", 2),
        ("A_offscope", 2), ("B_cap", 6), ("B_standing", 6),
        ("C_newpayee", 12), ("C_threshold", 4)],
    3: [("A_jewelry", 8), ("A_elec_ovs", 8), ("A_electronics", 11),
        ("A_travel", 22), ("A_daily", 10), ("A_plain", 2),
        ("A_offscope", 2), ("B_cap", 6), ("B_standing", 6),
        ("C_newpayee", 12), ("C_threshold", 4)],
    4: [("A_jewelry", 8), ("A_elec_ovs", 8), ("A_electronics", 11),
        ("A_travel", 12), ("A_daily", 10), ("A_plain", 2),
        ("A_offscope", 2), ("B_cap", 14), ("B_standing", 6),
        ("C_newpayee", 10), ("C_threshold", 6)],
}

# Ordered-fallback supply, APPENDED to every window's single-request plan.
# Appended (never interleaved) so the single-request families keep their own
# (family, k) stream: amounts, ids, holders and hidden-field placement are
# all keyed on that stream.
# Serving covers EVERY ladder row: a row that appears only in test would have
# no same-window serving evidence behind its verdict. Quotas therefore equal
# the ladder lengths, which also keeps L3's W0 quota-equivalent to L2's.
FALLBACK_SERVING = [("F_txn", 4), ("F_limit", 4), ("F_transfer", 4),
                    ("F_batch", 4)]
# Test leans harder on the fallback form than serving does: the
# determinability machinery needs SINGLE-REQUEST serving evidence to pin
# each threshold, but nothing requires the scored side to mirror that mix.
FALLBACK_TEST = [("F_txn", 12), ("F_limit", 8), ("F_transfer", 12),
                 ("F_batch", 12)]
FALLBACK_FAMILIES = {name for name, _ in FALLBACK_SERVING}
N_LEGS = 3


def serving_quota(w: int):
    return SERVING_QUOTA_W.get(w, SERVING_QUOTA_W["base"]) + FALLBACK_SERVING


def test_quota(w: int):
    return TEST_QUOTA_W.get(w, TEST_QUOTA_W["base"]) + FALLBACK_TEST


# ~half of each twin family carries the rule-firing condition, every window
# (the W2 corridor beat boosts hot and cold together: 6-of-12)
def hot_n(split: str, family: str, w: int) -> int:
    if family == "A_elec_ovs":
        return sum(_eo_mix(w))
    base = {"A_jewelry": 4, "B_standing": 3}
    n = base.get(family, 0)
    if w == 2 and family == "A_jewelry":
        n = 6
    return n



def _band_ok(hp, band_ks, amounts, min_flips=2):
    """Within one designed amount band, the hot/cold pattern read in amount
    order must alternate >= min_flips times, so no sub-band is monotone and
    no interval rule on one band transfers across splits."""
    rows = sorted((amounts[k], k) for k in band_ks)
    pat = [k in hp for _, k in rows]
    if len(set(pat)) < 2:
        return True     # uniform band carries no separable pattern
    return sum(1 for i in range(1, len(pat)) if pat[i] != pat[i - 1]) >= min_flips

def _hot_positions(amounts, hot: int, seed_key: str) -> set:
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
    for _ in range(500):
        hot_ranks = set(rnd.sample(range(n), hot))
        pat = [r in hot_ranks for r in range(n)]
        if sum(1 for i in range(1, n) if pat[i] != pat[i - 1]) >= 2:
            return {order[r] for r in hot_ranks}
    return {order[r] for r in range(hot)}


# multi-policy tickets: designed same-day totals that push the SUM over the
# $2000 daily cap while staying inside the normal today_total range (<=1980,
# overlapping every control), so neither marginal alone predicts the verdict.
# The today value rounds UP to the 10-grid: serving sums in [2040, 2170),
# test sums in [2060, 2190) —
# at or above every A_daily serving fire (2030+) and above every served
# control (<=1988), so no multi sum falls inside a window's served sum gap.
def _multi_today(family: str, k: int, split: str, w: int, amount: float) -> float:
    # TEST sums start at 2060: the A_daily serving fires floor each window
    # somewhere in [2030, 2050), so every test multi sum sits ABOVE the
    # served fire floor in every window (strict determinability)
    base = 2060.0 if split == "test" else 2040.0
    target = base + 20 * (_h(SEED, "mtgt", family, k, split, w) % 6)
    return round(max(200.0, -((amount - target) // 10) * 10), 2)


# per-window mix knobs for the corridor twins:
# jewelry hot: how many of the (amount-rank sorted) hot slots carry a HIGH
# same-day total (corridor x daily multi); elec_ovs hot: (n_low, n_high)
# split between the under-every-limit and above-every-limit halves
def _jw_multi_n(w: int, split: str) -> int:
    if split == "serving":
        return {2: 4}.get(w, 2)
    return {2: 3, 4: 3}.get(w, 2)


def _eo_mix(w: int):
    return {2: (3, 3)}.get(w, (2, 2))


def _vf(family: str, k: int, split: str, w: int) -> str:
    """Verification level. No L2 rule keys on it, but the manual advertises
    basic|verified, so it is a real field: hash-varied ~30/70, independent
    of every verdict."""
    return "basic" if _h(SEED, "vf", family, k, split, w) % 10 < 3 else "verified"


def _tenure(family: str, k: int, split: str, w: int) -> int:
    """Account tenure — a real, varied field."""
    return 1 + (_h(SEED, "ten", family, split, w, k) % 9)


def _spec(family: str, k: int, split: str, w: int = 0, hot_pos=None):
    """The k-th instance of a family -> (kind, fields). Twin families use
    hot_pos (amount-rank permuted, phantom-guarded); boundary families walk
    DESIGNED ladders with asymmetric brackets around every limit the
    timeline uses (650/520 electronics, 900/1150 travel, 4200 cap, 8000
    reporting, 2000 daily), so every probe is decided by serving evidence."""
    hot = hot_n(split, family, w)
    is_hot = (k in hot_pos) if hot_pos is not None else (k < hot)
    ten = _tenure(family, k, split, w)
    if family in FALLBACK_FAMILIES:
        return _fb_spec(family, k, split, w, ten)
    if family == "A_jewelry":
        # corridor scope W0-W1; VACATED at E2. The first _jw_multi_n(w) hot
        # slots (by hot_pos order) also carry a HIGH same-day total: corridor
        # x daily MULTI tickets — deny corridor while the corridor covers
        # jewelry (corridor > daily), escalate daily_cap after it migrates.
        amount = _spread(200, 900, family, k, split, w)
        is_multi = bool(hot_pos) and is_hot and k in sorted(hot_pos)[:_jw_multi_n(w, split)]
        f = dict(verification=_vf(family, k, split, w), standing="good",
                 tenure=ten, category="jewelry",
                 region="overseas" if is_hot else "domestic", amount=amount)
        if is_multi:
            f["today_total"] = _multi_today(family, k, split, w, amount)
            f["multi_path"] = True
        return ("txn", f)
    if family == "A_elec_ovs":
        # corridor's NEW scope from E2. Even ks sit UNDER every electronics
        # limit (corridor alone decides — single); odd ks sit ABOVE every
        # limit (corridor x limit MULTI: post-migration the corridor wins,
        # a partial learner who only knows the limit escalates over_limit).
        high = k % 2 == 1
        amount = (_spread(680, 900, family, k, split, w) if high
                  else _spread(60, 420, family, k, split, w))
        f = dict(verification=_vf(family, k, split, w), standing="good",
                 tenure=ten, category="electronics",
                 region="overseas" if is_hot else "domestic", amount=amount)
        if high and is_hot and R.truth(w)["A_R1_corridor_scope"] == "electronics":
            # only once the corridor actually covers electronics is this a
            # corridor x limit conjunction (pre-migration the ticket is
            # limit-only and docs-agreeing)
            f["multi_path"] = True
        return ("txn", f)
    if family == "A_electronics":
        # D1: docs 500, truth 650 -> 460 at W1 — the event CROSSES the
        # print. Bands: (460,500) general-approve at W0, adapt-ESCALATE from
        # W1 (docs approve there); (500,650) adapt-approve at W0, general
        # from W1; brackets at 440 / 652+. MULTI bases {530,558,586,534,562,
        # 590} carry a high same-day total: daily fires first (daily_cap vs
        # docs over_limit) — adapt multi in EVERY window. Beat W1 walks the
        # full (460,650) flip band.
        MULTI_BASES = {662.0, 686.0, 710.0, 666.0, 690.0, 714.0}
        if split == "serving":
            vals = ([(440.0, None), (462.0, "lo"), (470.0, None),
                     (478.0, "hi"), (502.0, "lo"), (540.0, None),
                     (578.0, None), (616.0, None), (628.0, "hi"),
                     (652.0, "lo"), (700.0, None),
                     (662.0, None), (686.0, None), (710.0, None)] if w <= 1
                    else [(440.0, None), (462.0, "lo"), (470.0, None),
                          (478.0, "hi"), (502.0, "lo"), (628.0, "hi"),
                          (652.0, "lo"), (700.0, None),
                          (662.0, None), (710.0, None)])
        else:
            vals = ([(410.0, None), (462.0, "lo"), (468.0, None),
                     (474.0, None), (478.0, "hi"), (502.0, "lo"),
                     (526.0, None), (544.0, None), (586.0, None),
                     (614.0, None), (628.0, "hi"),
                     (652.0, "lo"), (700.0, None),
                     (666.0, None), (690.0, None), (714.0, None)] if w <= 1
                    else [(410.0, None), (462.0, "lo"), (468.0, None),
                          (474.0, None), (478.0, "hi"), (628.0, "hi"),
                          (652.0, "lo"), (666.0, None), (690.0, None),
                          (714.0, None)])
        if split == "test" and w == 1:
            vals.extend([(420.0, None), (430.0, None), (440.0, None)])
        elif split == "test" and w >= 2:
            vals.append((430.0, None))
        base, edge = vals[k]
        amt = round(base + _band_cents(family, split, w, k, edge), 2)
        f = dict(verification=_vf(family, k, split, w), standing="good",
                 tenure=ten, category="electronics", region="domestic",
                 amount=amt)
        if base in MULTI_BASES:
            f["today_total"] = _multi_today(family, k, split, w, amt)
            f["multi_path"] = True
        return ("txn", f)
    if family == "A_travel":
        # D3: docs 1200, truth 900 -> 1150 at W3. Bands: (900,1150) the
        # RELAX flip band (adapt-escalate W0-W2, general-approve W3+);
        # (1150,1200) adapt-escalate in every window (docs-fallback stays
        # wrong). MULTI bases {1154,1162,1172,1176} add a high same-day
        # total: daily fires first (daily_cap vs docs approve) — adapt
        # multi in every window (travel x daily: the travel-limit condition
        # holds in every window because the bases sit above 1150 > 900).
        MID_MULTI = {1154.0, 1162.0, 1172.0, 1176.0}
        if split == "serving":
            vals = ({3: [(820.0, None), (846.0, None), (902.0, "lo"),
                         (936.0, None), (970.0, None), (1004.0, None),
                         (1038.0, None), (1072.0, None), (1106.0, None),
                         (1128.0, "hi"), (1152.0, "lo"), (1178.0, "hi"),
                         (1202.0, "lo"), (1260.0, None),
                         (1154.0, None), (1162.0, None), (1172.0, None),
                         (1176.0, None)],
                     4: [(902.0, "lo"), (978.0, None), (1054.0, None),
                         (1128.0, "hi"), (1152.0, "lo"), (1158.0, None),
                         (1166.0, None), (1178.0, "hi"), (1154.0, None),
                         (1162.0, None), (1172.0, None),
                         (1176.0, None)]}.get(
                w, [(820.0, None), (846.0, None), (902.0, "lo"),
                    (1128.0, "hi"), (1152.0, "lo"), (1178.0, "hi"),
                    (1202.0, "lo"), (1260.0, None),
                    (1154.0, None), (1162.0, None), (1172.0, None),
                    (1176.0, None)]))
        else:
            vals = ({3: [(820.0, None), (902.0, "lo"), (916.0, None),
                         (930.0, None), (958.0, None), (986.0, None),
                         (1014.0, None), (1028.0, None), (1042.0, None),
                         (1070.0, None), (1098.0, None), (1114.0, None),
                         (1128.0, "hi"),
                         (1152.0, "lo"), (1158.0, None), (1166.0, None),
                         (1178.0, "hi"), (1202.0, "lo"),
                         (1154.0, None), (1162.0, None), (1172.0, None),
                         (1176.0, None)],
                     4: [(902.0, "lo"), (978.0, None), (1054.0, None),
                         (1128.0, "hi"), (1152.0, "lo"), (1158.0, None),
                         (1166.0, None), (1178.0, "hi"), (1154.0, None),
                         (1162.0, None), (1172.0, None),
                         (1176.0, None)]}.get(
                w, [(820.0, None), (826.0, None), (902.0, "lo"),
                    (1006.0, None), (1128.0, "hi"), (1152.0, "lo"),
                    (1202.0, "lo"), (1240.0, None),
                    (1154.0, None), (1162.0, None), (1172.0, None),
                    (1176.0, None)]))
        base, edge = vals[k]
        amt = round(base + _band_cents(family, split, w, k, edge), 2)
        f = dict(verification=_vf(family, k, split, w), standing="good",
                 tenure=ten, category="travel", region="domestic", amount=amt)
        if base in MID_MULTI:
            f["today_total"] = _multi_today(family, k, split, w, amt)
            f["multi_path"] = True
        return ("txn", f)
    if family == "A_daily":
        # unwritten STATIC: same-day approved total + this charge > $2000 ->
        # escalate (daily_cap). (a) the firing
        # and non-firing sides OVERLAP on both marginals — a high-today
        # small-amount fire, a low-today big-amount fire, a high-today
        # control and a mid-today control — so neither today_total nor the
        # amount alone predicts the verdict; (b) served sums pin the cap from BOTH
        # sides (controls just under 2000, fires just over); test sums are
        # monotone-side decidable — every test control sum sits BELOW some
        # served approve sum and every test fire sum ABOVE some served fire.
        # k>=4 (serving; k==4 test) are the daily>limit PRECEDENCE probes:
        # electronics ABOVE every limit in play with a high total.
        n_pairs = 6
        cofire = False
        if k < n_pairs:
            # 6 pairs each split (3 fire / 3 ctrl) + co-fire beyond; the
            # fire and ctrl sides OVERLAP on both marginals (neither
            # today_total nor the amount alone predicts the verdict)
            pairs = ([(1610.0, 420.0), (990.0, 1100.0),     # fire: 2030/2090
                      (1500.0, 540.0),                       # fire: 2040
                      (1610.0, 340.0), (1050.0, 900.0),      # ctrl: 1950/1950
                      (1210.0, 760.0)]                       # ctrl: 1970
                     if split == "serving"
                     else [(1630.0, 420.0), (990.0, 1070.0),  # fire: 2050/2060
                           (1590.0, 480.0), (1200.0, 730.0),  # fire 2070/ctrl 1930
                           (1610.0, 320.0), (1040.0, 880.0)])  # ctrl: 1930/1920
            shift = _h(SEED, "dshift", split, w) % n_pairs
            today, base = pairs[(k + shift) % n_pairs]
            cat = "groceries"
        else:
            # beyond the pairs: TEST alternates daily>limit CO-FIRE probes
            # (high total -> daily_cap wins) with limit-only CONTROLS so the
            # test fire ratio stays ~50%; SERVING tail is all co-fire (its
            # controls come from the A_electronics ladder)
            base = 660.0 + 40 * (k % 2)
            cofire = ((k - n_pairs) % 2 == 0
                      or (split == "serving")
                      or (w == 4 and split == "test" and k == n_pairs + 1))
            today = (1600.0 + 20 * (k % 3)) if cofire else (940.0 + 20 * (k % 3))
            cat = "electronics"
        f = dict(verification=_vf(family, k, split, w), standing="good",
                 tenure=ten, category=cat, region="domestic",
                 amount=round(base + _cents(family, split, w, k), 2),
                 today_total=today)
        if k >= n_pairs and cofire:
            f["multi_path"] = True      # daily x limit co-fire (daily_cap wins)
        return ("txn", f)
    if family == "A_plain":
        cat = ["utilities", "dining", "fitness",
               "gift_cards"][(k + _h(SEED, "pc", split, w)) % 4]
        return ("txn", dict(verification=_vf(family, k, split, w), standing="good", tenure=ten,
                            category=cat, region="domestic",
                            amount=_spread(60, 900, family, k, split, w)))
    if family == "A_offscope":
        # Quota is two per window: keep the standing-scope control and vary
        # the corridor off-scope category across windows, so all declared
        # forms occur without sacrificing per-window scope coverage.
        form = k % 2
        if form == 0:      # overseas x an out-of-corridor category
            category = "travel" if w % 2 == 0 else "groceries"
            return ("txn", dict(verification=_vf(family, k, split, w), standing="good",
                                tenure=ten, category=category,
                                region="overseas",
                                amount=_spread(60, 820, family, k, split, w)))
        if form == 1:      # chargeback x transfer: standing is request-scoped
            return ("transfer", dict(verification=_vf(family, k, split, w),
                                 standing="recent_chargeback", tenure=ten,
                                 payee_id=_OLD_BY_SPLIT[split][
                                     _h(SEED, "os", split, w, k) % len(_OLD_BY_SPLIT[split])],
                                 amount=_spread(1200, 7900, family, k, split, w)))
        # overseas x groceries: corridor never covers it
        return ("txn", dict(verification=_vf(family, k, split, w), standing="good",
                            tenure=ten, category="groceries",
                            region="overseas",
                            amount=_spread(60, 900, family, k, split, w)))
    if family == "B_cap":
        # D4: docs print 4200 and are CORRECT until W4, when the cap
        # tightens to 3800 — the (3800,4200) band flips approve -> deny and
        # trusting the docs becomes wrong for the first time. Brackets at
        # 3700/3802 (W4 boundary) and 4178/4220 (pre-W4 boundary). The W4
        # beat walks the band densely.
        if split == "serving":
            vals = ([(3700.0, None), (3746.0, None), (3802.0, "lo"),
                     (3890.0, None), (3980.0, None), (4080.0, None),
                     (4178.0, "hi"), (4220.0, "lo"), (4400.0, None),
                     (4520.0, None), (4640.0, None), (4760.0, None)]
                    if w == 4
                    else [(3700.0, None), (3802.0, "lo"), (3900.0, None),
                          (4020.0, None), (4130.0, None), (4178.0, "hi"),
                          (4220.0, "lo"), (4400.0, None)])
        else:
            vals = ([(3680.0, None), (3830.0, None), (3860.0, None),
                     (3900.0, None), (3990.0, None), (4080.0, None),
                     (4130.0, None),
                     (4178.0, "hi"),
                     (4220.0, "lo"), (4400.0, None), (4560.0, None),
                     (4700.0, None)] if w == 4
                    else [(3680.0, None), (3830.0, None), (4090.0, None),
                          (4178.0, "hi"), (4220.0, "lo"), (4400.0, None)])
        if split == "test" and w == 4:
            vals.extend([(3700.0, None), (3740.0, None)])
        amt = _ladder(vals, family, k, split, w)
        return ("limit", dict(verification=_vf(family, k, split, w),
                              standing="good", tenure=ten, requested=amt))
    if family == "B_standing":
        req = (_spread(4700, 6000, family, k, split, w) if k < 2
               else _spread(2600, 3900, family, k, split, w))
        f = dict(verification=_vf(family, k, split, w),
                 standing="recent_chargeback" if is_hot else "good",
                 tenure=ten, requested=req)
        if is_hot and req > R.truth(w)["B_cap"]:
            f["multi_path"] = True
        return ("limit", f)
    if family == "C_newpayee":
        # Genuine payee-age x amount composition.  k0-3 exceed both the
        # recent-payee and reporting thresholds (two live conditions;
        # new_payee_review wins).  k4-5 exceed only the recent-payee
        # threshold.  k6-11 are below it: truth approves while the stale
        # manual's unconditional new-payee clause escalates.
        if k < 4:
            amt = _spread(8200, 9600, family, k, split, w)
        elif k < 6:
            amt = _spread(6700, 7900, family, k, split, w)
        else:
            # Same-window lower-bound evidence: every serving window has a
            # recent-payee approval above every held-out sub-$6500 probe.
            amt = (round(6420.0 + _cents(family, split, w, k), 2)
                   if split == "serving" and k == 6
                   else _spread(1200, 6300, family, k, split, w))
        # payee rotates with (k + w) so every recent payee accumulates BOTH
        # reportable and approve outcomes across the timeline, so per-payee
        # verdict histories stay balanced and payee-id recall does not help
        pool = _RECENT_BY_SPLIT[split]
        pid = pool[(k + w) % len(pool)]
        f = dict(verification=_vf(family, k, split, w), standing="good",
                 tenure=ten, payee_id=pid, amount=amt)
        if k < 4:
            f["multi_path"] = True
        return ("transfer", f)
    # C_threshold: the reporting threshold is DOCUMENTED TRUE
    # (8000/8000) — a pure general boundary family, interleaved so every
    # window keeps the straddle and the 7960/8020 serving bracket.
    amt = _ladder(
        [(7960.0, None), (8020.0, None), (6200.0, None), (8620.0, None),
         (6800.0, None), (9220.0, None), (7400.0, None), (9820.0, None),
         (7000.0, None), (8300.0, None), (6500.0, None)]
        if split == "serving"
        else [(7940.0, None), (8040.0, None), (6400.0, None), (8640.0, None),
              (7100.0, None), (9240.0, None), (7600.0, None), (9840.0, None),
              (6700.0, None), (8320.0, None), (6520.0, None)],
        family, k, split, w)
    return ("transfer", dict(verification=_vf(family, k, split, w), standing="good",
                             tenure=ten,
                             payee_id=_OLD_BY_SPLIT[split][
                                 _h(SEED, "ct", split, w, k) % len(_OLD_BY_SPLIT[split])],
                             amount=amt))


def _leg_cents(family, split, w, k, i) -> float:
    """Per-leg cents, keyed on the leg index too so the parts of one case
    never share a cents value (which would make the ranking readable off the
    decimals).

    The two splits take DISJOINT half-ranges (serving 0.01-9.99, test
    10.01-19.99). Multi-part ladders share their base values across splits —
    unlike the single-request ladders, which carry split-distinct bases — so
    without this the only separation would be a 1-in-1999 hash and exact
    collisions are certain at this ticket count. This is the same declared, answer-free split residue
    `_band_cents` already uses at band edges: every base sits far enough from
    every threshold that no verdict changes anywhere inside its cents range,
    asserted by _fb_margin_guard below."""
    c = round(0.01 + 0.01 * (_h(SEED, "legc", family, split, w, k, i) % 999), 2)
    return c if split == "serving" else round(c + 10.0, 2)


# ---- ordered-fallback ladders ------------------------------------------
# Each row is one case's ranked requests, most-preferred first. Rows are
# DESIGNED, not sampled: the stop position has to spread over {1, 2, 3, none}
# in every window, and each window's event has to
# move the stop position on some rows, or the fallback families carry no
# temporal signal. Every base keeps a >=20 margin from the thresholds that
# apply to it — asserted by _fb_margin_guard, because _leg_cents uses a
# split-conditional cents range.
#
# F_limit rows are read against the approvable cap (4200, tightening to 3800
# at W4); row 5 is a recent-chargeback account, so every leg dies on standing.
_FB_LIMIT = [(3600.0, 3300.0, 3000.0),   # always stop 1
             (4600.0, 3900.0, 3400.0),   # stop 2 -> stop 3 at W4
             (5200.0, 4800.0, 4400.0),   # always refuse (cap_exceeded)
             (4400.0, 4100.0, 3700.0),   # stop 2 -> stop 3 at W4
             (3900.0, 3500.0, 3100.0),   # stop 1 -> stop 2 at W4
             (4500.0, 4000.0, 3500.0),   # chargeback: refuse (account_standing)
             (4300.0, 4250.0, 4150.0),   # stop 3 -> refuse at W4
             (3700.0, 3400.0, 3200.0),   # always stop 1
             (4100.0, 3850.0, 3600.0),   # stop 1 -> stop 3 at W4
             (5000.0, 4300.0, 4150.0),   # stop 3 -> refuse at W4
             (3500.0, 3200.0, 3000.0),   # always stop 1
             (4800.0, 4400.0, 3900.0)]   # stop 3 -> refuse at W4
_FB_LIMIT_HOT = {5}                       # rows whose account has a chargeback

# F_txn rows: (category, region, today_total, ranked amounts). Read against the
# corridor scope (jewelry -> electronics at W2), the $2000 daily sum cap, and
# the category limits (electronics 650 -> 460 at W1; travel 900 -> 1150 at W3).
_FB_TXN = [
    ("electronics", "domestic", 200.0, (700.0, 600.0, 500.0)),   # stop2 -> refuse W1
    ("electronics", "domestic", 150.0, (620.0, 480.0, 420.0)),   # stop1 -> stop3 W1
    ("travel", "domestic", 200.0, (1250.0, 1100.0, 820.0)),      # stop3 -> stop2 W3
    ("travel", "domestic", 150.0, (1180.0, 1000.0, 920.0)),      # refuse -> stop2 W3
    ("jewelry", "overseas", 200.0, (500.0, 400.0, 300.0)),       # refuse -> stop1 W2
    ("electronics", "overseas", 200.0, (600.0, 500.0, 415.0)),   # stop1/3 -> refuse W2
    ("electronics", "domestic", 1500.0, (560.0, 400.0, 150.0)),  # daily-driven stop 2
    ("groceries", "domestic", 1600.0, (480.0, 250.0, 60.0)),     # daily-driven stop 2
    ("electronics", "domestic", 200.0, (900.0, 800.0, 700.0)),   # always refuse
    ("travel", "domestic", 250.0, (820.0, 700.0, 600.0)),        # always stop 1
    ("travel", "domestic", 200.0, (1160.0, 1050.0, 940.0)),      # refuse -> stop2 W3
    ("electronics", "domestic", 300.0, (615.0, 500.0, 410.0)),   # stop1 -> stop3 W1
    ("jewelry", "overseas", 150.0, (800.0, 650.0, 500.0)),       # refuse -> stop1 W2
    ("electronics", "overseas", 150.0, (430.0, 400.0, 350.0)),   # stop1 -> refuse W2
    ("dining", "domestic", 1450.0, (650.0, 200.0, 100.0)),       # daily-driven stop 2
    ("travel", "domestic", 200.0, (1300.0, 1200.0, 1100.0)),     # refuse -> stop3 W3
]

# F_transfer rows: ranked (payee age class, amount). Read against the
# recent-payee review threshold (6500) and the reporting threshold (8000);
# both are escalations, so an all-blocked row escalates rather than denies.
# The manual's unconditional new-payee clause makes many rows adapt: a docs
# reader stops on a later leg than current practice does.
_FB_TRANSFER = [
    (("old", 5200.0), ("old", 4800.0), ("recent", 4000.0)),      # stop 1
    (("recent", 9000.0), ("old", 7500.0), ("old", 5000.0)),      # stop 2
    (("recent", 8600.0), ("old", 8300.0), ("old", 7900.0)),      # stop 3
    (("recent", 7500.0), ("recent", 7400.0), ("recent", 6300.0)),  # stop 3
    (("old", 9500.0), ("old", 9000.0), ("old", 8500.0)),         # refuse (reportable)
    (("recent", 9800.0), ("recent", 9200.0), ("recent", 8800.0)),  # refuse (new_payee)
    (("recent", 6200.0), ("old", 5000.0), ("old", 4000.0)),      # stop 1
    (("old", 8100.0), ("old", 7600.0), ("recent", 5500.0)),      # stop 2
    (("recent", 5800.0), ("old", 8400.0), ("old", 6000.0)),      # stop 1
    (("recent", 7500.0), ("old", 7000.0), ("old", 6200.0)),      # stop 2
    (("old", 7900.0), ("old", 7000.0), ("old", 6000.0)),         # stop 1
    (("recent", 7500.0), ("recent", 6400.0), ("old", 5000.0)),   # stop 2
    (("old", 8200.0), ("recent", 7500.0), ("old", 7800.0)),      # stop 3
    (("recent", 8900.0), ("old", 8700.0), ("recent", 8100.0)),   # refuse, TWO codes
    (("old", 6400.0), ("recent", 9000.0), ("old", 4000.0)),      # stop 1
    (("recent", 7500.0), ("old", 7200.0), ("recent", 5900.0)),   # stop 2
    # All-blocked rows whose blocked legs carry DIFFERENT codes under the
    # same disposition: leg 1 escalates on the recent-payee clause, leg 2
    # on the reporting threshold.  Every such case accepts either code,
    # which exercises the accepted-set normalisation.
    (("old", 8600.0), ("recent", 7600.0), ("old", 8400.0)),     # refuse, TWO codes
    (("recent", 9400.0), ("old", 8900.0), ("recent", 7900.0)),  # refuse, TWO codes
]

# Recent-payee amounts avoid (6425, 7345]: that band is what the
# single-request serving stream leaves open for the new-payee
# threshold, and a batch/fallback part landing inside it would be a
# coin toss.
# F_batch rows: four payment lines. Screened LINE BY LINE — recent payee above
# 6500 -> new_payee_review, any line above 8000 -> reportable_amount, else
# approved. The manual only ever speaks about "a transfer", so whether the
# thresholds read a line or the batch total is hidden information; several
# rows carry a batch total far above 8000 with every line below it, so a
# learner that screens the total escalates a batch that should clear.
_FB_BATCH = [
    (("old", 2100.0), ("old", 1800.0), ("recent", 2400.0), ("old", 2600.0)),
    (("old", 8300.0), ("old", 1200.0), ("recent", 900.0), ("old", 2000.0)),
    (("recent", 7400.0), ("old", 3000.0), ("old", 2200.0), ("recent", 1500.0)),
    (("recent", 7400.0), ("old", 8600.0), ("old", 1100.0), ("recent", 6200.0)),
    (("old", 2500.0), ("old", 2700.0), ("old", 2900.0), ("old", 3100.0)),
    (("recent", 9200.0), ("recent", 8800.0), ("old", 900.0), ("old", 1000.0)),
    (("old", 8100.0), ("old", 7900.0), ("recent", 7500.0), ("recent", 6400.0)),
    (("recent", 3300.0), ("old", 4100.0), ("recent", 2000.0), ("old", 8900.0)),
    (("old", 7000.0), ("old", 7500.0), ("old", 6800.0), ("old", 7200.0)),
    (("recent", 7500.0), ("old", 8200.0), ("recent", 6300.0), ("old", 7800.0)),
    (("old", 1500.0), ("recent", 7500.0), ("old", 8400.0), ("recent", 5900.0)),
    (("recent", 8500.0), ("old", 2200.0), ("old", 3300.0), ("recent", 7600.0)),
    (("old", 6900.0), ("recent", 6100.0), ("old", 7400.0), ("old", 5200.0)),
    (("recent", 9500.0), ("old", 9100.0), ("recent", 8700.0), ("old", 8300.0)),
    (("old", 3600.0), ("old", 8050.0), ("recent", 7450.0), ("old", 2400.0)),
    (("recent", 7700.0), ("recent", 6200.0), ("old", 8600.0), ("old", 4500.0)),
    # line-vs-total PROBE rows (total over the reporting threshold, every
    # line under it) that still refuse on lines 2, 3 or 4, so that no fixed
    # per-line pattern such as "approve lines 3 and 4" is a free strategy.
    (("old", 3000.0), ("recent", 7600.0), ("old", 2500.0), ("recent", 7400.0)),
    (("old", 2000.0), ("old", 3000.0), ("recent", 7500.0), ("old", 2500.0)),
    (("recent", 7700.0), ("old", 2000.0), ("recent", 7400.0), ("old", 3000.0)),
    (("old", 2500.0), ("recent", 7600.0), ("recent", 7500.0), ("old", 2000.0)),
]


def _fb_margin_guard():
    """Every multi-part base must keep the whole 0-20 cents range on ONE side
    of every threshold that ACTUALLY applies to it; otherwise the
    split-conditional cents half-range in `_leg_cents` would decide a verdict
    and become an answer channel rather than a declared split residue.

    Bounds are scoped per ladder: a $900 transfer line has nothing to do with
    the $900 travel category limit, and checking every base against every
    boundary would raise on that coincidence."""
    CAP = {4200.0, 3800.0}                 # limit-increase cap
    XFER = {6500.0, 8000.0}                # recent-payee / reporting
    CATLIM = {"electronics": {650.0, 460.0}, "travel": {900.0, 1150.0}}
    DAILY = 2000.0

    def check(base, bounds, what):
        for bound in bounds:
            if base <= bound < base + 20.0:
                raise AssertionError(
                    f"{what}: base {base} sits within 20 of boundary {bound} "
                    f"— the split-conditional cents range would decide it")

    for row in _FB_LIMIT:
        for v in row:
            check(v, CAP, "F_limit")
    for cat, _region, today, row in _FB_TXN:
        for v in row:
            check(v, CATLIM.get(cat, set()), f"F_txn/{cat}")
            # the daily cap reads today_total + amount, so the SUM carries the
            # same margin requirement
            check(today + v, {DAILY}, f"F_txn/{cat} daily sum")
    for row in _FB_TRANSFER:
        for _age, amt in row:
            check(amt, XFER, "F_transfer")
    for row in _FB_BATCH:
        for _age, amt in row:
            check(amt, XFER, "F_batch")


def _policy_paths(kind, acc, mer, parts, state):
    """The DISTINCT hidden policies that participate in deciding a case.

    Counting occurrences instead would inflate a four-line batch that trips
    the reporting threshold four times into "four causal paths" when there is
    only one policy involved; multi_path is about how many policies the agent
    must know, not how many times one of them fires."""
    seen = set()
    for part in parts:
        amount = part.get("requested", part.get("amount"))
        if kind == "txn":
            if (mer["region"] == "overseas"
                    and state["A_R1_corridor_scope"] is not None):
                seen.add("corridor")
            if acc.get("today_total", 0.0) + amount > state["A_D_daily_cap"]:
                seen.add("daily")
            lim = {"electronics": state["A_T1_electronics_limit"],
                   "travel": state["A_T2_travel_limit"]}.get(mer["category"])
            if lim is not None and amount > lim:
                seen.add("limit")
        elif kind == "limit":
            if acc["standing"] == "recent_chargeback":
                seen.add("standing")
            if amount > state["B_cap"]:
                seen.add("cap")
        else:
            payee = part["payee"]
            if (payee.get("added_days_ago", 999) < R.DOCS["new_payee_days"]
                    and amount > state["C_new_payee_threshold"]):
                seen.add("newpayee")
            if amount > state["C_reporting_threshold"]:
                seen.add("reporting")
    return seen


def _form_verdict(kind, acc, mer, parts, state, form):
    """Whole-case verdict for a multi-part case under an arbitrary state.

    Used at BUILD time (no Task object exists yet) and shaped exactly like
    `_fallback_verdict_under` / `_batch_verdict_under`, so the slice label and
    the counterfactual machinery can never disagree."""
    outs = [_single_verdict_under(kind, acc, mer, part.get("payee"),
                                  part.get("requested", part.get("amount")),
                                  state) for part in parts]
    if form == "batch":
        return "batch", "|".join(f"{a}:{c}" for a, c in outs)
    for i, (action, _code) in enumerate(outs):
        if action == "approve":
            return "execute", f"leg{i + 1}"
    return outs[0]


def _form_slice(kind, acc, mer, parts, w, form):
    """adapt when a frozen-manual reader lands somewhere else. Uses the same
    state-parameterised engine as the counterfactual machinery, so the two
    can never disagree about what "the manual would do"."""
    return ("adapt" if _form_verdict(kind, acc, mer, parts, R.truth(w), form)
            != _form_verdict(kind, acc, mer, parts, R.DOCS_STATE, form)
            else "general")


_fb_margin_guard()


# the two splits draw DISJOINT ladder rows, so no scored multi-part ticket
# is a cents-perturbed copy of a same-window serving ticket that could be
# answered by replaying a remembered serving answer.
FALLBACK_ROWS = {
    "serving": {"F_txn": (0, 9, 10, 15),
                "F_limit": (0, 2, 6, 9),
                "F_transfer": (2, 9, 14, 16),
                "F_batch": (3, 4, 13, 19)},
    "test": {"F_txn": (1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14),
             "F_limit": (1, 3, 4, 5, 7, 8, 10, 11),
             "F_transfer": (0, 1, 3, 4, 5, 8, 10, 11, 12, 13, 15, 17),
             "F_batch": (1, 2, 5, 6, 7, 8, 9, 10, 14, 15, 16, 18)},
}


def _fb_row(family, split, k) -> int:
    rows = FALLBACK_ROWS[split][family]
    return rows[k % len(rows)]


def _fb_spec(family, k, split, w, ten):
    """Ordered-fallback rows -> (kind, fields). `legs` carries one dict per
    ranked request; the shared account attributes stay at the top level."""
    if family == "F_limit":
        ri = _fb_row("F_limit", split, k)
        row = _FB_LIMIT[ri]
        legs = [{"leg": i + 1,
                 "requested": round(v + _leg_cents(family, split, w, k, i), 2)}
                for i, v in enumerate(row)]
        return ("limit", dict(
            verification=_vf(family, k, split, w),
            standing=("recent_chargeback" if ri in _FB_LIMIT_HOT
                      else "good"),
            tenure=ten, fallback=True, legs=legs))
    if family == "F_txn":
        cat, region, today, row = _FB_TXN[_fb_row("F_txn", split, k)]
        legs = [{"leg": i + 1,
                 "amount": round(v + _leg_cents(family, split, w, k, i), 2)}
                for i, v in enumerate(row)]
        return ("txn", dict(
            verification=_vf(family, k, split, w), standing="good", tenure=ten,
            category=cat, region=region, today_total=today,
            fallback=True, legs=legs))
    if family == "F_batch":
        row = _FB_BATCH[_fb_row("F_batch", split, k)]
        lines = []
        for i, (age, amt) in enumerate(row):
            pool = (_RECENT_BY_SPLIT if age == "recent" else _OLD_BY_SPLIT)[split]
            lines.append({"line": i + 1,
                          "payee_id": pool[(k + i + w) % len(pool)],
                          "amount": round(
                              amt + _leg_cents(family, split, w, k, i), 2)})
        return ("transfer", dict(
            verification=_vf(family, k, split, w), standing="good", tenure=ten,
            batch=True, lines=lines))
    row = _FB_TRANSFER[_fb_row("F_transfer", split, k)]
    legs = []
    for i, (age, amt) in enumerate(row):
        pool = (_RECENT_BY_SPLIT if age == "recent" else _OLD_BY_SPLIT)[split]
        legs.append({"leg": i + 1,
                     "payee_id": pool[(k + i + w) % len(pool)],
                     "amount": round(amt + _leg_cents(family, split, w, k, i), 2)})
    return ("transfer", dict(
        verification=_vf(family, k, split, w), standing="good", tenure=ten,
        fallback=True, legs=legs))


def _single_verdict_under(kind, acc, mer, payee, amount, state):
    """One request's verdict under an arbitrary policy state (the leg-level
    primitive; `verdict_under` is the whole-case wrapper)."""
    if kind == "txn":
        if (mer["region"] == "overseas"
                and mer["category"] == state["A_R1_corridor_scope"]):
            return "deny", "corridor_blocked"
        if acc.get("today_total", 0.0) + amount > state["A_D_daily_cap"]:
            return "escalate", "daily_cap"
        limit = {"electronics": state["A_T1_electronics_limit"],
                 "travel": state["A_T2_travel_limit"]}.get(mer["category"])
        if limit is not None and amount > limit:
            return "escalate", "over_limit"
        return "approve", None
    if kind == "limit":
        if (state["B_R3_chargeback_blocks"]
                and acc["standing"] == "recent_chargeback"):
            return "deny", "account_standing"
        if amount > state["B_cap"]:
            return "deny", "cap_exceeded"
        return "approve", None
    if (payee.get("added_days_ago", 999) < R.DOCS["new_payee_days"]
            and amount > state["C_new_payee_threshold"]):
        return "escalate", "new_payee_review"
    if amount > state["C_reporting_threshold"]:
        return "escalate", "reportable_amount"
    return "approve", None


def _fallback_verdict_under(task, state, data):
    """Whole-case verdict for an ordered-fallback ticket under `state`.

    The returned pair is an ENCODING, not a tool call: an executed case is
    ("execute", "leg{k}") so that stopping on a different leg counts as a
    changed verdict for the temporal-slice machinery. Two states that stop on
    the same leg are the same verdict even if the blocked legs above it were
    blocked for different reasons — what the customer gets is identical."""
    acc = data["accounts"][task.z["account_id"]]
    kind = task.z["kind"]
    mer = (data["merchants"][task.z["merchant_id"]] if kind == "txn" else None)
    outcomes = []
    for leg in task.z["legs"]:
        payee = (data["payees"][leg["payee_id"]] if kind == "transfer" else None)
        amount = leg["requested"] if kind == "limit" else leg["amount"]
        outcomes.append(_single_verdict_under(kind, acc, mer, payee, amount, state))
    for i, (action, _code) in enumerate(outcomes):
        if action == "approve":
            return "execute", f"leg{i + 1}"
    lead_action, lead_code = outcomes[0]
    return lead_action, lead_code


def _batch_verdict_under(task, state, data):
    """A batch's verdict is the whole per-line tuple: two states differ as
    soon as ONE line's disposition differs, because the graded database keeps
    a status and a reason on every line."""
    acc = data["accounts"][task.z["account_id"]]
    parts = []
    for item in task.z["lines"]:
        payee = data["payees"][item["payee_id"]]
        action, code = _single_verdict_under(
            "transfer", acc, None, payee, item["amount"], state)
        parts.append(f"{action}:{code}")
    return "batch", "|".join(parts)


def verdict_under(task, state, data):
    """Recompute a ticket verdict under an arbitrary L2 policy state."""
    if task.z.get("batch"):
        return _batch_verdict_under(task, state, data)
    if task.z.get("fallback"):
        return _fallback_verdict_under(task, state, data)
    acc = data["accounts"][task.z["account_id"]]
    kind = task.z["kind"]
    if kind == "txn":
        mer = data["merchants"][
            data["transactions"][task.z["case_id"]]["merchant_id"]
        ]
        if (mer["region"] == "overseas"
                and mer["category"] == state["A_R1_corridor_scope"]):
            return "deny", "corridor_blocked"
        if acc.get("today_total", 0.0) + task.z["amount"] > state["A_D_daily_cap"]:
            return "escalate", "daily_cap"
        limit = {
            "electronics": state["A_T1_electronics_limit"],
            "travel": state["A_T2_travel_limit"],
        }.get(mer["category"])
        if limit is not None and task.z["amount"] > limit:
            return "escalate", "over_limit"
        return "approve", None
    if kind == "limit":
        if (state["B_R3_chargeback_blocks"]
                and acc["standing"] == "recent_chargeback"):
            return "deny", "account_standing"
        if task.z["amount"] > state["B_cap"]:
            return "deny", "cap_exceeded"
        return "approve", None
    payee = data["payees"][task.z["payee_id"]]
    if (payee.get("added_days_ago", 999) < R.DOCS["new_payee_days"]
            and task.z["amount"] > state["C_new_payee_threshold"]):
        return "escalate", "new_payee_review"
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
        if "today_total" in f:
            acc["today_total"] = f["today_total"]
        elif kind == "txn":
            # non-daily transactions carry totals up to ~1880, bounded
            # per-ticket so the sum can never reach the $2000 cap (amount +
            # total <= 1940) — deep overlap with the A_daily range, so the
            # value range does not name the family, and zero accidental
            # firings.
            bound = int((1940.0 - f["amount"]) // 10)
            acc["today_total"] = round(
                10.0 * (_h(SEED, "tdt", w, split, family, k) % (bound + 1)), 2)
            # 10-step grid: residues cover both 0 and 10 (mod 20), so the
            # residue is not an A_daily family fingerprint
        accounts[acc["account_id"]] = acc
        shells = SHELLS[kind]
        shell_i = _h(SEED, "shell", w, split, family, k) % len(shells)
        tag = {"txn": "TXN", "limit": "LIM", "transfer": "TRF"}[kind]
        cid_num = _h(SEED, 'caseid', w, split, family, k) % OPAQUE_ID_SPAN
        _all_cases = set().union(*(cases[tb] for tb in cases))
        while f"{tag}-{cid_num + OPAQUE_ID_BASE}" in _all_cases:
            cid_num = (cid_num + 1) % OPAQUE_ID_SPAN
        cid = f"{tag}-{cid_num + OPAQUE_ID_BASE}"
        if f.get("batch"):
            lines = f["lines"]
            rec_lines = [{"line": item["line"], "payee_id": item["payee_id"],
                          "payee_added_days_ago":
                              _PAYEES[item["payee_id"]]["added_days_ago"],
                          "amount": item["amount"], "status": "pending",
                          "reason": None} for item in lines]
            rec = {"case_id": cid, "kind": "payment_batch",
                   "account_id": acc["account_id"],
                   "lines": rec_lines, "status": "pending"}
            cases["transfers"][cid] = rec
            gts, docs_gts = [], []
            for item in lines:
                payee = _PAYEES[item["payee_id"]]
                gts.append(R.decide_transfer(acc, payee, item["amount"], w))
                docs_gts.append(R.docs_transfer(acc, payee, item["amount"], w))
            sl = _form_slice("transfer", acc, None,
                             [{"payee": _PAYEES[item["payee_id"]],
                               "amount": item["amount"]} for item in lines],
                             w, "batch")
            decisions = [{"line": item["line"],
                          "action": {"approve": "approve", "deny": "deny",
                                     "escalate": "escalate"}[a]}
                         for item, (a, _c) in zip(lines, gts)]
            for entry, (_a, code) in zip(decisions, gts):
                if code is not None:
                    entry["reason"] = code
            rf = {"cid": cid, "acc": acc["account_id"], "n": len(lines),
                  "total": round(math.fsum(item["amount"] for item in lines), 2)}
            for i, item in enumerate(lines):
                p = _PAYEES[item["payee_id"]]
                rf[f"ename{i + 1}"] = p["name"]
                rf[f"eid{i + 1}"] = p["payee_id"]
                rf[f"amt{i + 1}"] = item["amount"]
            pool_key = "batch"
            shells_b = SHELLS[pool_key]
            shell_i = _h(SEED, "shell", w, split, family, k) % len(shells_b)
            instruction = shells_b[shell_i].format(**rf)
            if split == "serving":
                t_serving += 1
                t = t_serving
            else:
                t = w * R.PER_WINDOW_SERVING + 1
            gidx += 1
            tid = "bank2_" + hashlib.md5(
                f"{SEED}|{w}|{split}|{family}|{k}".encode()).hexdigest()[:10]
            task = Task(task_id=tid, user_id="banking_l2",
                        instruction=instruction,
                        actions=[Action(name="decide_batch",
                                        kwargs={"case_id": cid,
                                                "decisions": decisions})],
                        answer=None,
                        z={"t": t, "window": w, "tier": "L2", "slice": sl,
                           "manual_slice": sl, "family": family,
                           "kind": "transfer", "shell_pool": pool_key,
                           "split": split, "case_id": cid,
                           "account_id": acc["account_id"],
                           "amount": lines[0]["amount"], "shell": shell_i,
                           "category": None, "region": None,
                           "payee_id": lines[0]["payee_id"],
                           "tenure": f["tenure"], "standing": f["standing"],
                           "verification": f["verification"],
                           "batch": True, "lines": rec_lines,
                           "n_lines": len(lines),
                           "batch_total": rf["total"],
                           "line_verdicts": [list(g) for g in gts],
                           "multi_path": len(_policy_paths(
                               "transfer", acc, None,
                               [{"payee": _PAYEES[item["payee_id"]],
                                 "amount": item["amount"]} for item in lines],
                               R.truth(w))) >= 2,
                           "n_firing": sum(1 for a, _c in gts if a != "approve"),
                           "decision": "batch", "reason": None})
            task._rf = rf
            return task
        if f.get("fallback"):
            legs = f["legs"]
            if kind == "limit":
                # every leg must be a real increase, so pin the current limit
                # under the SMALLEST ranked request
                low = min(item["requested"] for item in legs)
                acc["credit_limit"] = min(acc["credit_limit"],
                                          int(low - 400) // 100 * 100)
                rec_legs = [{"leg": item["leg"], "requested": item["requested"]}
                            for item in legs]
                resolved = [{"requested": item["requested"]} for item in legs]
                rec = {"case_id": cid, "kind": "limit_increase",
                       "account_id": acc["account_id"],
                       "current_limit": acc["credit_limit"],
                       "legs": rec_legs, "status": "pending", "reason": None}
                cases["limit_requests"][cid] = rec
                rf = {"cid": cid, "acc": acc["account_id"],
                      "amt": legs[0]["requested"], "amt2": legs[1]["requested"],
                      "amt3": legs[2]["requested"]}
                mer = None
            elif kind == "txn":
                mer = _merchant(f["category"], f["region"],
                                f"{split}:{family}:{k}:{w}", split, w)
                rec_legs = [{"leg": item["leg"], "amount": item["amount"]}
                            for item in legs]
                resolved = [{"merchant": mer, "amount": item["amount"]}
                            for item in legs]
                rec = {"case_id": cid, "kind": "card_transaction",
                       "account_id": acc["account_id"],
                       "merchant_id": mer["merchant_id"],
                       "same_day_approved_total": acc["today_total"],
                       "legs": rec_legs, "status": "pending", "reason": None}
                cases["transactions"][cid] = rec
                rf = {"cid": cid, "acc": acc["account_id"],
                      "ename": mer["name"], "eid": mer["merchant_id"],
                      "amt": legs[0]["amount"], "amt2": legs[1]["amount"],
                      "amt3": legs[2]["amount"]}
            else:
                rec_legs = [{"leg": item["leg"], "payee_id": item["payee_id"],
                             "payee_added_days_ago":
                                 _PAYEES[item["payee_id"]]["added_days_ago"],
                             "amount": item["amount"]} for item in legs]
                resolved = [{"payee": _PAYEES[item["payee_id"]],
                             "amount": item["amount"]} for item in legs]
                rec = {"case_id": cid, "kind": "outbound_transfer",
                       "account_id": acc["account_id"],
                       "legs": rec_legs, "status": "pending", "reason": None}
                cases["transfers"][cid] = rec
                rf = {"cid": cid, "acc": acc["account_id"]}
                for i, item in enumerate(legs):
                    p = _PAYEES[item["payee_id"]]
                    sfx = "" if i == 0 else str(i + 1)
                    rf[f"ename{sfx}"] = p["name"]
                    rf[f"eid{sfx}"] = p["payee_id"]
                    rf[f"amt{sfx}"] = item["amount"]
                mer = None
            chosen, action, reason, accepted = R.decide_legs(kind, acc, resolved, w)
            sl = _form_slice(kind, acc, mer, resolved, w, "fallback")
            if chosen is None and accepted:
                # evaluator-private: GetCaseDetailsL2 strips leading-underscore
                # keys, so the accepted set rides on the record (and is scoped
                # by the episode view) without being observable
                rec["_accept"] = accepted
            if chosen is not None:
                tool = "execute_leg"
                kwargs = {"case_id": cid, "leg": chosen + 1}
                decision, reason_z = "execute", None
            else:
                tool = {"deny": "deny_case", "escalate": "escalate_case"}[action]
                kwargs = {"case_id": cid, "reason": reason}
                decision, reason_z = action, reason
            pool_key = f"{kind}_fb"
            shells_fb = SHELLS[pool_key]
            shell_i = _h(SEED, "shell", w, split, family, k) % len(shells_fb)
            instruction = shells_fb[shell_i].format(**rf)
            if split == "serving":
                t_serving += 1
                t = t_serving
            else:
                t = w * R.PER_WINDOW_SERVING + 1
            gidx += 1
            tid = "bank2_" + hashlib.md5(
                f"{SEED}|{w}|{split}|{family}|{k}".encode()).hexdigest()[:10]
            task = Task(task_id=tid, user_id="banking_l2",
                        instruction=instruction,
                        actions=[Action(name=tool, kwargs=kwargs)], answer=None,
                        z={"t": t, "window": w, "tier": "L2", "slice": sl,
                           "manual_slice": sl, "family": family, "kind": kind,
                           "shell_pool": pool_key, "split": split,
                           "case_id": cid, "account_id": acc["account_id"],
                           "amount": (legs[0]["requested"] if kind == "limit"
                                      else legs[0]["amount"]),
                           "shell": shell_i,
                           "category": f.get("category"),
                           "region": f.get("region"),
                           "merchant_id": (mer["merchant_id"] if mer else None),
                           "payee_id": rec_legs[0].get("payee_id"),
                           "tenure": f["tenure"], "standing": f["standing"],
                           "verification": f["verification"],
                           "fallback": True, "legs": rec_legs,
                           "n_legs": len(legs),
                           "chosen_leg": (chosen + 1) if chosen is not None else None,
                           "accepted_reasons": accepted,
                           "multi_path": len(_policy_paths(
                               kind, acc, mer, resolved, R.truth(w))) >= 2,
                           "n_firing": sum(
                               1 for item in resolved
                               if _single_verdict_under(
                                   kind, acc, mer,
                                   item.get("payee"),
                                   item.get("requested", item.get("amount")),
                                   R.truth(w))[0] != "approve"),
                           "decision": decision, "reason": reason_z})
            task._rf = rf
            return task
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
                # the account's same-day approved total rides ON the case
                # record (get_case_details always fetches it), so no extra
                # account lookup is needed. The SUM judgment and daily>limit
                # precedence are unaffected.
                "same_day_approved_total": acc["today_total"],
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

        # strict co-firing count: how many policy CONDITIONS actually hold
        # (fires or would fire if reached) under the current truth — the
        # z.multi flag additionally counts mandatory checks (corridor scope
        # on overseas tickets, roster membership); both are stamped
        # so reports can use either definition
        st_now = R.truth(w)
        n_firing = 0
        if kind == "txn":
            if (mer["region"] == "overseas"
                    and mer["category"] == st_now["A_R1_corridor_scope"]):
                n_firing += 1
            if acc.get("today_total", 0.0) + f["amount"] > st_now["A_D_daily_cap"]:
                n_firing += 1
            lim = {"electronics": st_now["A_T1_electronics_limit"],
                   "travel": st_now["A_T2_travel_limit"]}.get(f.get("category"))
            if lim is not None and f["amount"] > lim:
                n_firing += 1
        elif kind == "limit":
            if f["standing"] == "recent_chargeback":
                n_firing += 1
            if f["requested"] > st_now["B_cap"]:
                n_firing += 1
        else:
            payee = _PAYEES[f["payee_id"]]
            if (payee.get("added_days_ago", 999) < R.DOCS["new_payee_days"]
                    and f["amount"] > st_now["C_new_payee_threshold"]):
                n_firing += 1
            if f["amount"] > st_now["C_reporting_threshold"]:
                n_firing += 1
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
        tid = "bank2_" + hashlib.md5(
            f"{SEED}|{w}|{split}|{family}|{k}".encode()).hexdigest()[:10]
        task = Task(task_id=tid,
                    user_id="banking_l2", instruction=instruction,
                    actions=[Action(name=tool, kwargs=kwargs)], answer=None,
                    z={"t": t, "window": w, "tier": "L2", "slice": sl,
                       "manual_slice": sl,
                       "family": family, "kind": kind, "shell_pool": kind,
                       "split": split,
                       "case_id": cid, "account_id": acc["account_id"],
                       "amount": amt, "shell": shell_i,
                       "category": f.get("category"), "region": f.get("region"),
                       "payee_id": f.get("payee_id"), "tenure": f["tenure"],
                       "standing": f["standing"],
                       "verification": f["verification"],
                       "multi_path": f.get("multi_path", False),
                       "n_firing": n_firing,
                       "decision": decision, "reason": reason})
        task._rf = _rf
        return task


    def _verdict_key(t):
        """What the shell balancer must spread. z["decision"] is a constant on
        multi-part cases, so balancing it would let the real answer pile up in
        one shell."""
        if t.z.get("batch"):
            return "batch:" + "|".join(f"{a}:{c}" for a, c in t.z["line_verdicts"])
        if t.z.get("fallback") and t.z.get("chosen_leg"):
            return f"execute:leg{t.z['chosen_leg']}"
        return f"{t.z['decision']}:{t.z['reason']}"

    _shell_rr = {}

    def _rebalance_shells(pool, w, split):
        """Deterministic shell balancing: within each
        (kind, decision) group the shells are assigned round-robin, and
        the rotation CONTINUES across windows (a per-group counter), so
        even small groups distribute evenly over the whole timeline and
        every shell's verdict mix collapses to the kind's base rate."""
        for t in pool:
            t.z["verdict_key"] = _verdict_key(t)
        if split == "test":
            assigned = balanced_shell_assignment(
                pool,
                shell_count_by_kind=SCORED_SHELL_COUNT,
                seed=SEED,
                key_field="shell_pool",
                decision_field="verdict_key",
            )
            for t in pool:
                si = assigned[t.task_id]
                t.z["shell"] = si
                t.instruction = SHELLS[t.z["shell_pool"]][si].format(**t._rf)
                del t._rf
            return
        groups = {}
        for t in pool:
            # keyed on (shell pool, FAMILY, decision): balancing inside each
            # family leaves no within-family correlation between shell
            # and verdict
            groups.setdefault((t.z["shell_pool"], t.z["family"],
                               t.z["verdict_key"]), []).append(t)
        for (pool_key, fam, dec), grp in sorted(
                groups.items(), key=lambda kv: (kv[0][0], kv[0][1],
                                                str(kv[0][2]))):
            grp = sorted(grp, key=lambda t: _h(SEED, "shb", t.task_id))
            start = _shell_rr.get((split, pool_key, fam, dec), 0)
            for i, t in enumerate(grp):
                si = (start + i) % SERVING_SHELL_COUNT[pool_key]
                t.z["shell"] = si
                t.instruction = SHELLS[pool_key][si].format(**t._rf)
            _shell_rr[(split, pool_key, fam, dec)] = (
                (start + len(grp)) % SERVING_SHELL_COUNT[pool_key])
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
        if family == "B_standing":
            low = {k: a for k, a in amounts.items() if k >= 2}
            hp = set(_hot_positions(low, hot - 1, f"{family}:{split}:{w}"))
            hp |= {0}
        elif family == "A_elec_ovs":
            # low (even ks, under every limit) and high (odd ks, above every
            # limit) halves get their hot counts from the declared mix, so
            # the corridor x limit MULTI supply is designed, not sampled
            nl, nh = _eo_mix(w)
            evens = [k for k in amounts if k % 2 == 0]
            odds = [k for k in amounts if k % 2 == 1]
            rnd = _random.Random(f"eo:{SEED}:{split}:{w}")
            order = [k for k, _ in sorted(amounts.items(), key=lambda kv: kv[1])]
            for _ in range(500):
                hp = set(rnd.sample(evens, nl)) | set(rnd.sample(odds, nh))
                pat = [k in hp for k in order]
                if (sum(1 for i in range(1, len(pat)) if pat[i] != pat[i-1]) >= 2
                        and _band_ok(hp, evens, amounts)
                        and _band_ok(hp, odds, amounts)):
                    break
        else:
            hp = _hot_positions(amounts, hot, f"{family}:{split}:{w}")
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
        # The adapt thinner has to see the same numeric
        # contract the later balance pass enforces, or it can
        # evict a side the pass then cannot rebuild.
        _COVERAGE_SPECS = (("A_jewelry", "region"),
                   ("A_elec_ovs", "region"),
                   ("B_standing", "standing"),
                   ("A_daily", "decision"),
                   ("A_electronics", "decision"),
                   ("A_travel", "decision"),
                   ("B_cap", "decision"),
                   ("C_threshold", "decision"))
        _NUMERIC_SPECS = (
                ("A_electronics", lambda task, _w: task.z["amount"],
                 lambda w: R.truth(w)["A_T1_electronics_limit"],
                 lambda task, _w: task.z.get("reason") in
                 (None, "over_limit")),
                ("A_travel", lambda task, _w: task.z["amount"],
                 lambda w: R.truth(w)["A_T2_travel_limit"],
                 lambda task, _w: task.z.get("reason") in
                 (None, "over_limit")),
                ("A_daily", lambda task, _w:
                 (data["accounts"][task.z["account_id"]]
                  .get("today_total", 0.0) + task.z["amount"]),
                 lambda w: R.truth(w)["A_D_daily_cap"],
                 lambda task, _w: task.z.get("reason") in
                 (None, "daily_cap")),
                ("B_cap", lambda task, _w: task.z["amount"],
                 lambda w: R.truth(w)["B_cap"],
                 lambda task, _w: task.z.get("reason") in
                 (None, "cap_exceeded")),
                ("C_newpayee", lambda task, _w: task.z["amount"],
                 lambda w: R.truth(w)["C_new_payee_threshold"],
                 lambda task, _w: task.z.get("reason") in
                 (None, "new_payee_review")),
                ("C_threshold", lambda task, _w: task.z["amount"],
                 lambda w: R.truth(w)["C_reporting_threshold"],
                 lambda task, _w: task.z.get("reason") in
                 (None, "reportable_amount")),
            )
        # Reserve, before any thinning, the adapt rows the numeric
        # and categorical coverage contracts depend on: the probe
        # builders run after `balance_change_adapt` and cannot
        # rebuild a side or a value it has already deleted.
        _ADAPT_KEEP = categorical_value_adapt_ids(
            test_by_w, specs=_COVERAGE_SPECS, seed=SEED)
        for _w, _ids in numeric_side_adapt_ids(
                test_by_w, specs=_NUMERIC_SPECS, seed=SEED).items():
            _ADAPT_KEEP[_w].update(_ids)
        # Multi-part tickets carry this tier's hidden-information
        # load.  Thinning them to hit a size target would cut exactly
        # the rows the size target exists to preserve, so they are
        # reserved whole on both slices.
        for _w, _tasks in test_by_w.items():
            _ADAPT_KEEP[_w].update(
                task.task_id for task in _tasks
                if task.z.get("family") in FALLBACK_FAMILIES)
        test_by_w = balance_change_adapt(
            test_by_w,
            rules=R,
            data=data,
            verdict_fn=verdict_under,
            seed=SEED,
            protected_persistent_task_ids_by_window=_ADAPT_KEEP,
            # sized to absorb the ordered-fallback rows without evicting
            # the single-request adapt rows the numeric boundaries depend on
            target_adapt_by_window={0: 35, 1: 37, 2: 36, 3: 39, 4: 39},
            prefer_composite=True,
        )
        alternating = categorical_alternation_probe_ids(
            test_by_w,
            specs=(("A_jewelry", "region", None),
                   ("A_elec_ovs", "region", None),
                   ("B_standing", "standing", None)),
            seed=SEED,
        )
        protected = event_general_probe_ids(
            test_by_w, rules=R, data=data, verdict_fn=verdict_under
        )
        coverage = value_coverage_probe_ids(
            test_by_w,
            specs=_COVERAGE_SPECS,
            seed=SEED,
        )
        for w, task_ids in coverage.items():
            protected[w].update(task_ids)
        # Static daily-cap adapt rows are all escalations. Keep two approve
        # controls in the early windows so a constant daily_cap strategy
        # cannot profit from the change-sensitive thinning.
        for w in range(R.N_WINDOWS):
            controls = sorted(
                (task for task in test_by_w[w]
                 if task.z.get("slice") == "general"
                 and task.z.get("family") == "A_daily"
                 and task.z.get("decision") == "approve"),
                key=lambda task: task.task_id,
            )
            n_controls = 4 if w == 4 else 3
            protected[w].update(task.task_id for task in controls[:n_controls])
        for w, task_ids in alternating.items():
            protected[w].update(task_ids)
        # Keep a recent-payee-only general witness as well as the co-fire
        # representative selected by the composite-aware family sampler.
        for w, tasks in test_by_w.items():
            recent_only = sorted(
                (task for task in tasks
                 if task.z.get("slice") == "general"
                 and task.z.get("family") == "C_newpayee"
                 and task.z.get("n_firing") == 1),
                key=lambda task: task.task_id,
            )
            protected[w].update(task.task_id for task in recent_only[:1])
            cofire = sorted(
                (task for task in tasks
                 if task.z.get("slice") == "general"
                 and task.z.get("family") == "C_newpayee"
                 and task.z.get("n_firing", 0) >= 2),
                key=lambda task: task.task_id,
            )
            protected[w].update(task.task_id for task in cofire[:1])
        # These are precedence controls, not expendable texture: keep every
        # chargeback request that also crosses the live cap.
        for w, tasks in test_by_w.items():
            state = R.truth(w)
            for task in tasks:
                if (task.z.get("slice") == "general"
                        and task.z.get("family") == "B_standing"
                        and task.z.get("standing") == "recent_chargeback"
                        and task.z.get("amount") > state["B_cap"]):
                    protected[w].add(task.task_id)
        categorical = per_window_constant_answer_balance_ids(
            test_by_w,
            families=("A_jewelry", "A_elec_ovs", "B_standing"),
            seed=SEED,
            max_rate=0.80,
            preselected_by_window=protected,
            active_windows_by_family={
                "A_jewelry": {0, 1},
                "A_elec_ovs": {2, 3, 4},
            },
        )
        for w, task_ids in categorical.items():
            protected[w].update(task_ids)
        daily_controls = constant_answer_balance_ids(
            test_by_w,
            families=("A_daily",),
            seed=SEED,
            max_rate=0.60,
            preselected_by_window=protected,
        )
        for w, task_ids in daily_controls.items():
            protected[w].update(task_ids)
        numeric = numeric_side_balance_ids(
            test_by_w,
            specs=_NUMERIC_SPECS,
            seed=SEED,
            preselected_by_window=protected,
            proportional=True,
        )
        for w, task_ids in numeric.items():
            protected[w].update(task_ids)
        for w, task_ids in multipart_control_ids(
                test_by_w, rules=R, families=FALLBACK_FAMILIES).items():
            protected[w].update(task_ids)
        test_by_w = prune_general_repetitions(
            test_by_w,
            target_per_window=R.PER_WINDOW_TEST,
            numeric_specs=_NUMERIC_SPECS,
            mark_controls=True,
            seed=SEED,
            preferred_families_by_window={
                w: {"A_jewelry", "A_elec_ovs", "B_standing"}
                for w in range(R.N_WINDOWS)
            },
            protected_task_ids_by_window=protected,
            protected_families_by_window={
                # Multi-part rows are protected only where they carry ADAPT signal.
                # A multi-part GENERAL ticket adds execution length --
                # three legs or four lines handled per the manual -- and no
                # hidden information, which is not what this suite measures,
                # so those rows are prunable and the general budget goes to
                # the single-request coverage controls.
                w: {"A_offscope"}
                for w in range(R.N_WINDOWS)
            },
            prefer_composite=True,
        )
    _rebalance_shells(
        [task for w in range(R.N_WINDOWS) for task in test_by_w[w]],
        0,
        "test",
    )

    return data, serving, test_by_w


PROTOCOL_NOTES = """Issue one final decision per case — for a payment batch
that single call must already cover every line — then finish
(scoring is outcome-based: the case's final state is what is judged, and
repeat decision calls are rejected by the tools)."""


def _system_prompt(tools) -> str:
    head = f"""You are a bank operations-review agent. Today's date is {TODAY}.

# Environment
You review pending cases of three kinds — card transactions (TXN-...),
credit-limit increase requests (LIM-...) and outbound transfers (TRF-...) —
and settle each with EXACTLY ONE final tool call. Decisions are final.
A case is one of three forms, and get_case_details tells you which:
  - a SINGLE request -> approve_case / deny_case / escalate_case
  - RANKED ALTERNATIVES (`legs`, most preferred first) -> execute_leg on the
    first alternative current practice approves automatically
  - a PAYMENT BATCH (`lines`) -> decide_batch, giving every line its own
    action and reason code
Deny and escalate cite a reason code. The workflow guide states how each
form is resolved; read it before deciding.
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
             GetPayeeDetails, ApproveCase, DenyCase, EscalateCase,
             ExecuteLeg, DecideBatch]
    meta = {"scenario": "banking_l2", "domain": "banking", "tier": "L2",
            "system_prompt": _system_prompt(tools)}
    return {
        "meta": meta, "id": "banking_L2",
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
