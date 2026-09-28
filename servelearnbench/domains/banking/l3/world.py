"""Banking L3 world: L2-hard single-policy primitives under a
multi-stage change schedule, with a tier-disjoint visible surface.

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
  chargeback/good — L3 has no verification rule and no payee roster);
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
from .tools import (ApproveCaseL3 as ApproveCase, DecideBatchL3 as DecideBatch,
                    DenyCaseL3 as DenyCase,
                    EscalateCaseL3 as EscalateCase, ExecuteLegL3 as ExecuteLeg,
                    GetAccountDetailsL3 as GetAccountDetails,
                    GetCaseDetailsL3 as GetCaseDetails, GetMerchantDetails,
                    GetPayeeDetailsL3 as GetPayeeDetails)
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
SEED = "banking_v6_L3_l2core_temporal"
ENTITY_ID_BASE = 700
ENTITY_ID_SPAN = 200
OPAQUE_ID_BASE = 700000
OPAQUE_ID_SPAN = 200000

FIRST = ["Ava", "Noah", "Mia", "Liam", "Zoe", "Ethan", "Ivy", "Lucas", "Nora",
         "Owen", "Ruth", "Felix", "June", "Hugo", "Lena", "Marco", "Anouk",
         "Bassam", "Corin", "Dagny", "Emeka", "Fenna", "Gaspar", "Hana",
         "Ilias", "Jetta", "Kofi", "Lucia", "Matteo", "Nell", "Otthild",
         "Pavel", "Rania", "Soren", "Tala", "Viktor", "Solveig", "Tarek",
         "Ulf", "Verena", "Wilmar", "Xiulan", "Yannick", "Zorana", "Aurel",
         "Bianca", "Ciaran", "Dorit", "Eluned", "Fabrizio", "Gudrun",
         "Halvor"]
LAST = ["Alvarez", "Becker", "Chen", "Dawson", "Egan", "Fischer", "Grant",
        "Hopkins", "Iqbal", "Jensen", "Kovac", "Lindqvist", "Moreau",
        "Novak", "Ortiz", "Petrov", "Galloway", "Hirsch", "Ibarra", "Joshi",
        "Kaminski", "Lindau", "Mbeki", "Nystrom", "Okafor", "Pellegrini",
        "Quiroga", "Rasmussen", "Sindelar", "Toivonen", "Ulmer", "Vieira",
        "Wolff", "Yamada", "Zamani", "Benkovic", "Wallenberg", "Ximenes",
        "Ysgaard", "Zamora", "Ansaldo", "Brannigan", "Cavendish",
        "Delacroix", "Engelhart", "Fontaine", "Gunnarsson", "Havelock"]
STATES = ["CA", "NY", "TX", "WA", "IL", "MA", "CO", "GA"]

# 8-12 names per category (electronics 12): two-thirds serving-only,
# the rest test-only —
# GLOBALLY (the rules key on category x region, never on the entity, and
# shared merchants let id-recall reproduce the corridor rule). Payees are
# ALSO split-disjoint per age class (L3 has no roster — the payee-relevant
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
# marks an age class or a split (L3 has no roster)
JURISDICTIONS = ["US", "UK", "SG", "AE"]

SHELLS = {
    "txn": [
        "Adjudicate card event {cid}: {ename} ({eid}) placed ${amt:.2f} on "
        "account {acc}.",
        "Transaction desk request {cid} concerns a ${amt:.2f} purchase by "
        "{acc} at {ename} ({eid}).",
        "Account {acc} has card item {cid} pending for ${amt:.2f} with "
        "{ename} ({eid}). Record the disposition.",
        "Determine the outcome of authorization {cid}: ${amt:.2f}, merchant "
        "{ename} ({eid}), account {acc}.",
        "Card event {cid} is unresolved. It links {acc} to {ename} ({eid}) "
        "for ${amt:.2f}.",
        "A decision is due on transaction {cid}, a ${amt:.2f} charge to "
        "{acc} from {ename} ({eid}).",
        "Process pending purchase {cid}: merchant {ename} ({eid}), amount "
        "${amt:.2f}, account {acc}.",
    ],
    "limit": [
        "Adjudicate credit request {cid}: account {acc} seeks a limit of "
        "${amt:.2f}.",
        "Credit desk item {cid} proposes a ${amt:.2f} ceiling for {acc}.",
        "Account {acc} has increase case {cid} pending at ${amt:.2f}. Record "
        "the disposition.",
        "Determine the outcome of limit application {cid}: {acc} requests "
        "${amt:.2f}.",
        "Credit-line event {cid} is unresolved for account {acc}; proposed "
        "limit ${amt:.2f}.",
        "A decision is due on ceiling request {cid}, submitted by {acc} for "
        "${amt:.2f}.",
        "Process pending limit case {cid}: account {acc}, requested amount "
        "${amt:.2f}.",
    ],
    "transfer": [
        "Adjudicate payment {cid}: account {acc} would send ${amt:.2f} to "
        "{ename} ({eid}).",
        "Transfer desk item {cid} covers ${amt:.2f} from {acc} to {ename} "
        "({eid}).",
        "Account {acc} has outbound case {cid} pending for ${amt:.2f} to "
        "{ename} ({eid}). Record the disposition.",
        "Determine the outcome of wire {cid}: ${amt:.2f}, beneficiary "
        "{ename} ({eid}), source {acc}.",
        "Payment event {cid} is unresolved. It sends ${amt:.2f} from {acc} "
        "to {ename} ({eid}).",
        "A decision is due on transfer {cid}, addressed to {ename} ({eid}) "
        "for ${amt:.2f} from {acc}.",
        "Process pending wire case {cid}: source {acc}, recipient {ename} "
        "({eid}), amount ${amt:.2f}.",
    ],
}




SHELLS.update({
    "txn_fb": [
        "Ranked card review {cid} on account {acc}: {ename} ({eid}) submitted "
        "${amt:.2f}. The cardholder would prefer it settled at "
        "${amt2:.2f}, and failing that at ${amt3:.2f}. Take the best one "
        "current practice allows.",
        "Work item {cid}: account {acc} at {ename} ({eid}) — preferred "
        "${amt:.2f}, else ${amt2:.2f}, else ${amt3:.2f}. Settle it.",
        "Hold {cid} on {acc}: {ename} ({eid}) asked for "
        "${amt:.2f}; acceptable alternatives are ${amt2:.2f} then ${amt3:.2f}.",
        "Preference list {cid} — account {acc}, merchant {ename} "
        "({eid}): first ${amt:.2f}, then ${amt2:.2f}, then ${amt3:.2f}.",
        "File {cid}: {acc} will take ${amt:.2f} at {ename} ({eid}), or "
        "${amt2:.2f}, or ${amt3:.2f}. Issue the ruling.",
        "The cardholder on {acc} lists three amounts at {ename} ({eid}) for case "
        "{cid}: ${amt:.2f}, ${amt2:.2f}, ${amt3:.2f}. Record the decision.",
        "Close card file {cid} for {acc} at {ename} ({eid}); the ranked "
        "amounts are ${amt:.2f}, ${amt2:.2f}, ${amt3:.2f}.",
    ],
    "limit_fb": [
        "Ranked credit file {cid}: account {acc} asks for a ${amt:.2f} limit, or "
        "${amt2:.2f}, or ${amt3:.2f} — in that order.",
        "Work item {cid}: {acc} would take ${amt:.2f}, else ${amt2:.2f}, "
        "else ${amt3:.2f}. Settle it.",
        "Credit review {cid} for account {acc}: preferred ceiling ${amt:.2f}, "
        "fallbacks ${amt2:.2f} and ${amt3:.2f}.",
        "Preference list {cid} — account {acc}: ${amt:.2f}, then "
        "${amt2:.2f}, then ${amt3:.2f}.",
        "File {cid}: {acc} offers ${amt:.2f}, ${amt2:.2f} or ${amt3:.2f} "
        "as a new ceiling. Issue the ruling.",
        "Account {acc} lists three ceilings under file {cid}: ${amt:.2f}, "
        "${amt2:.2f}, ${amt3:.2f}. Record the decision.",
        "Close credit-line file {cid}; {acc} will accept ${amt:.2f}, "
        "${amt2:.2f} or ${amt3:.2f}.",
    ],
    "batch": [
        "Payment file {cid} on account {acc}: {n} lines totalling "
        "${total:.2f} — {ename1} ({eid1}) ${amt1:.2f} · {ename2} ({eid2}) "
        "${amt2:.2f} · {ename3} ({eid3}) ${amt3:.2f} · {ename4} ({eid4}) "
        "${amt4:.2f}. Rule on every line.",
        "Work item {cid}: account {acc} submitted a {n}-line payment file "
        "worth ${total:.2f} — ${amt1:.2f} to {ename1} ({eid1}), ${amt2:.2f} to "
        "{ename2} ({eid2}), ${amt3:.2f} to {ename3} ({eid3}), ${amt4:.2f} "
        "to {ename4} ({eid4}). Settle each line.",
        "File screening {cid} for {acc} (${total:.2f} across {n} lines): "
        "{ename1} ({eid1}) ${amt1:.2f}; {ename2} ({eid2}) ${amt2:.2f}; "
        "{ename3} ({eid3}) ${amt3:.2f}; {ename4} ({eid4}) ${amt4:.2f}.",
        "Remittance run {cid} — account {acc}, {n} beneficiaries, "
        "${total:.2f} total: {ename1} ({eid1}) ${amt1:.2f} / {ename2} "
        "({eid2}) ${amt2:.2f} / {ename3} ({eid3}) ${amt3:.2f} / {ename4} "
        "({eid4}) ${amt4:.2f}. Issue a ruling per line.",
        "File {cid}: {acc} pays {n} beneficiaries (${total:.2f}) — "
        "{ename1} ({eid1}) ${amt1:.2f}, {ename2} ({eid2}) ${amt2:.2f}, "
        "{ename3} ({eid3}) ${amt3:.2f}, {ename4} ({eid4}) ${amt4:.2f}.",
        "Mass payment {cid} from {acc}: ${amt1:.2f} {ename1} ({eid1}) · "
        "${amt2:.2f} {ename2} ({eid2}) · ${amt3:.2f} {ename3} ({eid3}) · "
        "${amt4:.2f} {ename4} ({eid4}). Total ${total:.2f} over {n} lines.",
        "Close remittance file {cid} for {acc}; the {n} lines are {ename1} "
        "({eid1}) ${amt1:.2f}, {ename2} ({eid2}) ${amt2:.2f}, {ename3} "
        "({eid3}) ${amt3:.2f} and {ename4} ({eid4}) ${amt4:.2f}.",
    ],
    "transfer_fb": [
        "Ranked payment file {cid}: account {acc} wants ${amt:.2f} to {ename} "
        "({eid}); failing that ${amt2:.2f} to {ename2} ({eid2}); failing "
        "that ${amt3:.2f} to {ename3} ({eid3}).",
        "Work item {cid}: {acc} ranks three payments — ${amt:.2f} to "
        "{ename} ({eid}), ${amt2:.2f} to {ename2} ({eid2}), ${amt3:.2f} to "
        "{ename3} ({eid3}). Settle it.",
        "Outbound file {cid} for {acc}: first choice ${amt:.2f} to {ename} "
        "({eid}), then ${amt2:.2f} to {ename2} ({eid2}), then ${amt3:.2f} "
        "to {ename3} ({eid3}).",
        "Preference list {cid} — account {acc}: {ename} ({eid}) ${amt:.2f} · "
        "{ename2} ({eid2}) ${amt2:.2f} · {ename3} ({eid3}) ${amt3:.2f}.",
        "File {cid}: {acc} may send ${amt:.2f} to {ename} ({eid}), or "
        "${amt2:.2f} to {ename2} ({eid2}), or ${amt3:.2f} to {ename3} "
        "({eid3}). Issue the ruling.",
        "Beneficiary choices under file {cid} on {acc}: {ename} ({eid}) "
        "${amt:.2f}, {ename2} ({eid2}) ${amt2:.2f}, {ename3} ({eid3}) "
        "${amt3:.2f}. Record the decision.",
        "Close payment file {cid}; {acc} ranks ${amt:.2f} to {ename} "
        "({eid}), ${amt2:.2f} to {ename2} ({eid2}), ${amt3:.2f} to "
        "{ename3} ({eid3}).",
    ],
})

# Shell wordings, per side.
#
# The verdict-balance contract is SCORED-side: a shell's majority share is
# kept within 5pp of its pool's base rate over the test set, and a
# cell of n rows cannot resolve finer than 1/n.  Only txn has enough scored
# rows for seven cells (223 here); the rest sit at 30-76 and are scored
# over three.
#
# Serving is under no such cap, so the single-request pools keep all seven
# wordings there and none of their sentences is unreachable.  The four
# multi-part pools are the exception: ~30 serving rows over seven cells
# leaves verdict-pure shells, so they use three on both sides.  Their
# remaining four sentences keep every pool at >= 6 wordings and are the ONLY
# unreachable ones, so a sentence added to any other pool takes effect.
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
        out[mid] = {"merchant_id": mid, "name": f"{name} Group",
                    "category": cat, "region": region}
    return out


def _build_payees():
    """Ids are assigned AFTER a deterministic shuffle of all payee names.
    L3 has NO restricted roster — payees
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
        out[pid] = {"payee_id": pid, "name": f"{name} Network",
                    "type": "business",
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
    display_names = {f"{name} Group" for name in names}
    cands = [m for m in _MERCHANTS.values()
             if m["category"] == cat and m["region"] == region
             and m["name"] in display_names]
    return cands[_h(SEED, "mer", salt) % len(cands)]


def _account(acc_id: str, verification: str, standing: str, tenure: int,
             h: int, holder_i: int) -> dict:
    return {"account_id": acc_id,
            "holder": (f"{FIRST[holder_i % len(FIRST)]} N. "
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
    """Per-ticket cents in {0.01 .. 19.99}, SAME distribution on both splits,
    so the cents pattern does not identify the split."""
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
    1: [("A_jewelry", 8), ("A_elec_ovs", 8), ("A_electronics", 14),
        ("A_travel", 10), ("A_daily", 8), ("A_plain", 1),
        ("A_offscope", 2), ("B_cap", 10), ("B_standing", 6),
        ("C_newpayee", 12), ("C_threshold", 5)],
    2: [("A_jewelry", 8), ("A_elec_ovs", 8), ("A_electronics", 10),
        ("A_travel", 12), ("A_daily", 8), ("A_plain", 1),
        ("A_offscope", 2), ("B_cap", 6), ("B_standing", 6),
        ("C_newpayee", 18), ("C_threshold", 5)],
    3: [("A_jewelry", 8), ("A_elec_ovs", 8), ("A_electronics", 12),
        ("A_travel", 16), ("A_daily", 8), ("A_plain", 1),
        ("A_offscope", 2), ("B_cap", 7), ("B_standing", 6),
        ("C_newpayee", 12), ("C_threshold", 4)],
    4: [("A_jewelry", 8), ("A_elec_ovs", 8), ("A_electronics", 9),
        ("A_travel", 10), ("A_daily", 14), ("A_plain", 1),
        ("A_offscope", 2), ("B_cap", 10), ("B_standing", 6),
        ("C_newpayee", 12), ("C_threshold", 4)],
    5: [("A_jewelry", 12), ("A_elec_ovs", 8), ("A_electronics", 10),
        ("A_travel", 15), ("A_daily", 8), ("A_plain", 1),
        ("A_offscope", 2), ("B_cap", 6), ("B_standing", 6),
        ("C_newpayee", 12), ("C_threshold", 4)],
}
TEST_QUOTA_W = {
    "base": [("A_jewelry", 8), ("A_elec_ovs", 8), ("A_electronics", 14),
             ("A_travel", 13), ("A_daily", 10), ("A_plain", 1),
             ("A_offscope", 2), ("B_cap", 6), ("B_standing", 6),
             ("C_newpayee", 12), ("C_threshold", 7)],
    1: [("A_jewelry", 8), ("A_elec_ovs", 8), ("A_electronics", 15),
        ("A_travel", 11), ("A_daily", 8), ("A_plain", 1),
        ("A_offscope", 2), ("B_cap", 12), ("B_standing", 6),
        ("C_newpayee", 12), ("C_threshold", 6)],
    2: [("A_jewelry", 8), ("A_elec_ovs", 8), ("A_electronics", 11),
        ("A_travel", 13), ("A_daily", 8), ("A_plain", 1),
        ("A_offscope", 2), ("B_cap", 7), ("B_standing", 4),
        ("C_newpayee", 18), ("C_threshold", 4)],
    3: [("A_jewelry", 8), ("A_elec_ovs", 8), ("A_electronics", 13),
        ("A_travel", 20), ("A_daily", 10), ("A_plain", 2),
        ("A_offscope", 2), ("B_cap", 7), ("B_standing", 4),
        ("C_newpayee", 12), ("C_threshold", 5)],
    4: [("A_jewelry", 8), ("A_elec_ovs", 8), ("A_electronics", 9),
        ("A_travel", 10), ("A_daily", 18), ("A_plain", 2),
        ("A_offscope", 2), ("B_cap", 10), ("B_standing", 6),
        ("C_newpayee", 12), ("C_threshold", 7)],
    5: [("A_jewelry", 12), ("A_elec_ovs", 6), ("A_electronics", 11),
        ("A_travel", 26), ("A_daily", 10), ("A_plain", 1),
        ("A_offscope", 2), ("B_cap", 4), ("B_standing", 6),
        ("C_newpayee", 12), ("C_threshold", 5)],
}

# Multi-part supply, APPENDED to every window's single-request plan so the
# existing families keep an unchanged (family, k) stream — their amounts, ids,
# holders and hidden-field placement are all keyed on it. Test leans harder
# on the multi-part forms than serving does: a batch returns ONE binary reward
# for four lines and cannot pin a threshold, so the evidence that makes those
# lines decidable has to come from single-request serving tickets.
# Serving must cover EVERY ladder row: a row that appears only in test has no
# same-window serving counterpart, so the transition it lands on has no
# evidence. Quotas therefore equal the ladder lengths.
FALLBACK_SERVING = [("F_txn", 4), ("F_limit", 4), ("F_transfer", 4),
                    ("F_batch", 4)]
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
    if w == 5 and family == "A_jewelry":
        n = 8
    return n



def _band_ok(hp, band_ks, amounts, min_flips=2):
    """Within one designed amount band, the hot/cold pattern read in amount
    order must alternate >= min_flips times, so no sub-band is monotone
    and no interval rule on one band transfers across splits."""
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
    # Fixed across windows: the input distribution must not move with the
    # hidden cap.  These sums exceed both cap values (1700 and 2000).
    base = 2060.0 if split == "test" else 2040.0
    target = base + 20 * (_h(SEED, "mtgt", family, k, split, w) % 6)
    return round(max(200.0, -((amount - target) // 10) * 10), 2)


# per-window mix knobs for the corridor twins:
# jewelry hot: how many of the (amount-rank sorted) hot slots carry a HIGH
# same-day total (corridor x daily multi); elec_ovs hot: (n_low, n_high)
# split between the under-every-limit and above-every-limit halves
def _jw_multi_n(w: int, split: str) -> int:
    if split == "serving":
        # W2 migrates the corridor away from jewelry. Keep two clean
        # deny->approve landing examples so the deactivation outcome is
        # visible in serving rather than hidden behind daily co-fires.
        return 2
    return {2: 3, 4: 3}.get(w, 2)


def _eo_mix(w: int):
    return {2: (3, 3), 4: (1, 3)}.get(w, (2, 2))


def _vf(family: str, k: int, split: str, w: int) -> str:
    """Verification level. No L3 rule keys on it, but the manual advertises
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
        if w == 1:
            vals = ([(420.0, None), (440.0, None),
                     (462.0, "lo"), (480.0, None), (502.0, None),
                     (526.0, None), (550.0, None), (574.0, None),
                     (600.0, None), (628.0, "hi"),
                     (652.0, "lo"), (662.0, None),
                     (686.0, None), (710.0, None)]
                    if split == "serving"
                    else [(410.0, None), (440.0, None),
                          (462.0, "lo"), (478.0, "hi"),
                          (502.0, None), (526.0, None),
                          (550.0, None), (574.0, None),
                          (600.0, None), (628.0, "hi"),
                          (652.0, "lo"), (666.0, None),
                          (690.0, None), (714.0, None)])
        elif w == 3:
            vals = ([(420.0, None), (440.0, None),
                     (462.0, "lo"), (480.0, None), (502.0, None),
                     (526.0, None), (550.0, None), (574.0, None),
                     (600.0, None), (628.0, "hi"),
                     (652.0, "lo"), (700.0, None)]
                    if split == "serving"
                    else [(410.0, None), (440.0, None),
                          (462.0, "lo"), (478.0, "hi"),
                          (502.0, None), (526.0, None),
                          (550.0, None), (574.0, None),
                          (600.0, None), (628.0, "hi"),
                          (652.0, "lo"), (714.0, None)])
            if split == "test":
                vals.append((720.0, None))
        elif split == "serving" and w == 2:
            vals[0:3] = [(420.0, None), (440.0, None), (462.0, "lo")]
        if split == "test" and w == 2:
            vals.append((430.0, None))
        if split == "test" and w == 1:
            vals.append((420.0, None))
        if split == "test" and w == 4:
            # Isolated high-side probes. The other high rows co-fire
            # the daily cap, so they test precedence rather than the
            # electronics threshold itself.
            vals.insert(8, (680.0, None))
        elif split == "test" and w == 5:
            vals.append((680.0, None))
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
                     5: [(820.0, None), (846.0, None), (902.0, "lo"),
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
                     5: [(820.0, None), (902.0, "lo"), (916.0, None),
                         (930.0, None), (958.0, None), (986.0, None),
                         (1014.0, None), (1028.0, None), (1042.0, None),
                         (1070.0, None), (1098.0, None), (1114.0, None),
                         (1128.0, "hi"), (1152.0, "lo"), (1158.0, None),
                         (1166.0, None), (1178.0, "hi"), (1202.0, "lo"),
                         (1154.0, None), (1162.0, None), (1172.0, None),
                         (1176.0, None)],
                     4: [(902.0, "lo"), (978.0, None), (1054.0, None),
                         (1128.0, "hi"), (1152.0, "lo"), (1158.0, None),
                         (1166.0, None), (1178.0, "hi"), (1154.0, None),
                         (1162.0, None), (1172.0, None),
                         (1176.0, None)]}.get(
                w, [(820.0, None), (826.0, None), (902.0, "lo"),
                    (840.0, None), (1128.0, "hi"), (1152.0, "lo"),
                    (1202.0, "lo"), (1240.0, None),
                    (1154.0, None), (1162.0, None), (1172.0, None),
                    (1176.0, None), (1006.0, None)]))
        if split == "test" and w == 5:
            vals.extend([(780.0, None), (790.0, None),
                         (800.0, None), (810.0, None)])
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
        # and non-firing sides must OVERLAP on both marginals — a high-today
        # small-amount fire, a low-today big-amount fire, a high-today
        # control and a mid-today control — so neither today_total nor the
        # amount alone predicts the verdict; (b) served sums pin the cap from BOTH
        # sides (controls just under 2000, fires just over); test sums are
        # monotone-side decidable — every test control sum sits BELOW some
        # served approve sum and every test fire sum ABOVE some served fire.
        # k>=4 (serving; k==4 test) are the daily>limit PRECEDENCE probes:
        # electronics ABOVE every limit in play with a high total.
        # Fixed sums span both policy values.  W0-W3 use the 2000 boundary;
        # W4-W5 use 1700, so the middle rows become genuine flip probes.
        pairs = ([(1200.0, 440.0), (980.0, 700.0),
                  (1210.0, 488.0), (990.0, 840.0),
                  (1430.0, 540.0), (1510.0, 520.0),
                  (990.0, 1080.0), (1380.0, 680.0),
                  (1010.0, 740.0), (1250.0, 540.0),
                  (990.0, 880.0), (1410.0, 500.0),
                  (1190.0, 780.0), (1340.0, 720.0)]
                 if split == "serving"
                 else [(1210.0, 440.0), (990.0, 700.0),
                       (1220.0, 520.0), (1010.0, 870.0),
                       (1400.0, 550.0), (1520.0, 530.0),
                       (1000.0, 1090.0), (1390.0, 690.0),
                       (1020.0, 740.0), (1260.0, 550.0),
                       (1000.0, 890.0), (1420.0, 510.0),
                       (1200.0, 780.0), (1350.0, 720.0),
                       (930.0, 700.0), (1400.0, 720.0)])
        if split == "test" and w == 4:
            # Clean controls under the tightened $1700 cap. These are
            # groceries, so no lower-priority category limit masks them.
            pairs.extend([(900.0, 680.0), (1050.0, 550.0)])
        elif split == "test" and w == 5:
            pairs[8:8] = [(900.0, 680.0), (1050.0, 550.0)]
        today, base = pairs[k]
        cofire = k in ({7, 13} if split == "serving" else {7, 13, 15})
        cat = "electronics" if cofire else "groceries"
        f = dict(verification=_vf(family, k, split, w), standing="good",
                 tenure=ten, category=cat, region="domestic",
                 amount=round(base + _cents(family, split, w, k), 2),
                 today_total=today)
        if cofire:
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
            vals = ({0: [(3700.0, None), (3802.0, "lo"),
                         (3900.0, None), (4130.0, None),
                         (4178.0, "hi"), (4220.0, "lo"),
                         (4400.0, None)],
                     1: [(3600.0, None), (3740.0, None),
                         (3802.0, "lo"), (3840.0, None),
                         (3880.0, None), (3920.0, None),
                         (3980.0, None), (4040.0, None),
                         (4100.0, None), (4178.0, "hi")],
                     2: [(3600.0, None), (3740.0, None),
                         (3802.0, "lo"), (3900.0, None),
                         (4220.0, "lo"), (4400.0, None)],
                     3: [(3600.0, None), (3740.0, None),
                         (3802.0, "lo"), (3900.0, None),
                         (4050.0, None), (4220.0, "lo"),
                         (4400.0, None)],
                     5: [(3700.0, None), (3900.0, None),
                         (4178.0, "hi"), (4220.0, "lo"),
                         (4400.0, None), (4600.0, None)]}.get(
                w, [(3700.0, None), (3802.0, "lo"),
                     (3860.0, None), (3920.0, None), (3980.0, None),
                     (4040.0, None), (4100.0, None), (4178.0, "hi"),
                     (4220.0, "lo"), (4400.0, None)]
                    if w == 4
                    else [(3700.0, None), (3802.0, "lo"), (3900.0, None),
                          (4020.0, None), (4130.0, None), (4178.0, "hi"),
                          (4220.0, "lo"), (4400.0, None)]))
        else:
            vals = ({1: [(3650.0, None), (3750.0, None),
                         (3830.0, None), (3870.0, None),
                         (3910.0, None), (3950.0, None),
                         (3990.0, None), (4030.0, None),
                         (4070.0, None), (4110.0, None),
                         (4220.0, "lo"), (4400.0, None)]}.get(
                w, [(3680.0, None), (3830.0, None), (3860.0, None),
                     (3900.0, None), (3990.0, None), (4080.0, None),
                     (4130.0, None), (4178.0, "hi"),
                     (4220.0, "lo"), (4400.0, None), (4560.0, None),
                     (4700.0, None)] if w == 4
                    else [(3680.0, None), (3830.0, None), (4090.0, None),
                          (4178.0, "hi"), (4220.0, "lo"),
                          (4400.0, None)]))
        if split == "test" and w in (2, 3):
            vals.append((3700.0, None))
        if split == "test" and w == 5:
            vals = [(3680.0, None), (4090.0, None),
                    (4220.0, "lo"), (4400.0, None)]
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
        if k < 3:
            amt = _spread(8200, 9600, family, k, split, w)
        elif k == 3:
            amt = round(7800.0 + _cents(family, split, w, k), 2)
        elif k == 4:
            # Serving carries the closest positive threshold anchor; test
            # stays farther from the boundary.  This makes the positive
            # side strictly derivable rather than merely within a tolerance.
            base = 7220.0 if split == "serving" else 7300.0
            amt = round(base + _cents(family, split, w, k), 2)
        elif k in (5, 6, 7, 8, 9, 12, 13, 14):
            # Eight fixed counterfactual probes in the 6500..7200 band.
            # They fire before W2 and become controls after the threshold
            # tightens; the input distribution itself never moves.
            bases = {5: 6600.0, 6: 6700.0, 7: 6800.0, 8: 6900.0,
                     9: 7170.0 if split == "serving" else 7100.0,
                     12: 6750.0, 13: 6950.0,
                     14: 7180.0 if split == "serving" else 7150.0}
            amt = round(bases[k] + _cents(family, split, w, k), 2)
        elif k in (10, 11):
            # Near-threshold controls keep the negative side learnable in
            # every window, with serving negatives close to held-out ones.
            base = {10: 6100.0,
                    11: 6460.0 if split == "serving" else 6400.0}[k]
            amt = round(base + _cents(family, split, w, k), 2)
        else:
            amt = _spread(1200, 6200.0, family, k, split, w)
        # payee rotates with (k + w) so every recent payee accumulates BOTH
        # reportable and approve outcomes across the timeline, so per-payee
        # verdict histories do not predict the answer
        pool = _RECENT_BY_SPLIT[split]
        pid = pool[(k + w) % len(pool)]
        f = dict(verification=_vf(family, k, split, w), standing="good",
                 tenure=ten, payee_id=pid, amount=amt)
        if k < 3:
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


def _is_reopen(family: str, f: dict, w: int) -> bool:
    """Mark tickets whose earlier restrictive outcome becomes permissive."""
    if (family == "A_electronics" and w == 3
            and 460.0 < f["amount"] < 650.0):
        return True
    if (family == "B_cap" and w == 4 and f["standing"] == "good"
            and 3800.0 < f["requested"] < 4200.0):
        return True
    if (family == "A_elec_ovs" and w == 4
            and f.get("region") == "overseas"):
        return True
    if (family == "A_jewelry" and w == 5
            and f.get("region") == "overseas"):
        return True
    return False


def _leg_cents(family, split, w, k, i) -> float:
    """Per-part cents, keyed on the part index so the parts of one case never
    share a cents value. The two splits take DISJOINT half-ranges: multi-part
    ladders share base values across splits, so a shared cents space would
    collide exactly. Safe because
    _fb_margin_guard keeps every base 20 away from every value its threshold
    takes ANYWHERE on the timeline."""
    c = round(0.01 + 0.01 * (_h(SEED, "legc", family, split, w, k, i) % 999), 2)
    return c if split == "serving" else round(c + 10.0, 2)


# ---- ordered-fallback / batch ladders --------------------------------------
# Bases must ALSO stay outside the served numeric envelope of every
# window: L3's single-request ladders prove approvals only up to ~855
# (travel) and ~445 (electronics at the 460 setting), far tighter than
# L2's, so several L2 bases are undecidable here.
# L3 thresholds OSCILLATE, so a base has to clear every value its threshold
# ever takes: electronics {650,460}, travel {900,1150}, daily sum {2000,1700},
# cap {4200,3800}, recent-payee {6500,7200}, reporting {8000}. That leaves a
# much narrower usable range than L2 and is why several L2 rows do not carry
# over unchanged (e.g. daily sums of 1980 sit on the 2000 boundary).
_FB_LIMIT = [(3600.0, 3300.0, 3000.0),   # always stop 1
             (4600.0, 3900.0, 3400.0),   # stop2 -> stop3 (W1) -> stop2 (W4)
             (5200.0, 4800.0, 4400.0),   # always refuse
             (4400.0, 4100.0, 3700.0),   # stop2 -> stop3 -> stop2
             (3900.0, 3500.0, 3100.0),   # stop1 -> stop2 -> stop1
             (4500.0, 4000.0, 3500.0),   # chargeback: refuse (standing)
             (4300.0, 4250.0, 4150.0),   # stop3 -> refuse -> stop3
             (3700.0, 3400.0, 3200.0),   # always stop 1
             (4100.0, 3850.0, 3600.0),   # stop1 -> stop3 -> stop1
             (5000.0, 4300.0, 4150.0),   # stop3 -> refuse -> stop3
             (3500.0, 3200.0, 3000.0),   # always stop 1
             (4800.0, 4400.0, 3900.0)]   # stop3 -> refuse -> stop3
_FB_LIMIT_HOT = {5}

_FB_TXN = [
    ("electronics", "domestic", 200.0, (700.0, 600.0, 500.0)),
    ("electronics", "domestic", 150.0, (620.0, 480.0, 420.0)),
    ("travel", "domestic", 200.0, (1250.0, 1100.0, 820.0)),
    ("travel", "domestic", 150.0, (1180.0, 1000.0, 920.0)),
    ("jewelry", "overseas", 200.0, (500.0, 400.0, 300.0)),
    ("electronics", "overseas", 200.0, (600.0, 500.0, 410.0)),
    ("electronics", "domestic", 1500.0, (560.0, 400.0, 150.0)),   # daily: always/flips/never
    ("groceries", "domestic", 1600.0, (340.0, 250.0, 60.0)),      # daily flips W4
    ("electronics", "domestic", 200.0, (900.0, 800.0, 700.0)),    # always refuse
    ("travel", "domestic", 250.0, (810.0, 700.0, 600.0)),         # always stop 1
    ("travel", "domestic", 200.0, (1160.0, 1050.0, 940.0)),
    ("electronics", "domestic", 300.0, (610.0, 500.0, 410.0)),
    ("jewelry", "overseas", 150.0, (800.0, 650.0, 500.0)),
    ("electronics", "overseas", 150.0, (410.0, 400.0, 350.0)),
    ("dining", "domestic", 1450.0, (300.0, 200.0, 100.0)),        # daily flips W4
    ("travel", "domestic", 200.0, (1300.0, 1200.0, 1100.0)),
]

# Recent-payee amounts between 6500 and 7200 are the W2 event's flip band and
# are wanted; every one of them still has to be decidable from that window's
# SINGLE-REQUEST serving evidence.
_FB_TRANSFER = [
    (("old", 5200.0), ("old", 4800.0), ("recent", 4000.0)),
    (("recent", 9000.0), ("old", 7500.0), ("old", 5000.0)),
    (("recent", 8600.0), ("old", 8300.0), ("old", 7900.0)),
    (("recent", 7600.0), ("recent", 7400.0), ("recent", 6300.0)),
    (("old", 9500.0), ("old", 9000.0), ("old", 8500.0)),
    (("recent", 9800.0), ("recent", 9200.0), ("recent", 8800.0)),
    (("recent", 6200.0), ("old", 5000.0), ("old", 4000.0)),
    (("old", 8100.0), ("old", 7600.0), ("recent", 5500.0)),
    (("recent", 5800.0), ("old", 8400.0), ("old", 6000.0)),
    (("recent", 7500.0), ("old", 7000.0), ("old", 6200.0)),
    (("old", 7900.0), ("old", 7000.0), ("old", 6000.0)),
    (("recent", 7550.0), ("recent", 6400.0), ("old", 5000.0)),
    (("old", 8200.0), ("recent", 7500.0), ("old", 7800.0)),
    (("recent", 8900.0), ("old", 8700.0), ("recent", 8100.0)),
    (("old", 6400.0), ("recent", 9000.0), ("old", 4000.0)),
    (("recent", 7700.0), ("old", 7200.0), ("recent", 5900.0)),
    # All-blocked rows whose blocked legs carry DIFFERENT codes under the
    # same disposition: leg 1 escalates on the recent-payee clause, leg 2
    # on the reporting threshold.  Every such case accepts either code,
    # which exercises the accepted-set normalisation.
    (("old", 8600.0), ("recent", 7600.0), ("old", 8400.0)),     # refuse, TWO codes
    (("recent", 9400.0), ("old", 8900.0), ("recent", 7900.0)),  # refuse, TWO codes
    # Multi-part rows whose answer moves with the recent-payee threshold
    # step 6500 -> 7200 at W2 (amounts in (6500, 7200]).  6900/6950 clear
    # both served
    # uncertainty bands (W0-W1 ends at 6603, W2-W5 starts at 7173), so the
    # flip is decidable on both sides of the event.
    (("recent", 6900.0), ("old", 8600.0), ("old", 5000.0)),     # stop3 -> stop1 at W2
    (("recent", 9000.0), ("recent", 6950.0), ("old", 4000.0)),  # stop3 -> stop2 at W2
]

_FB_BATCH = [
    (("old", 2100.0), ("old", 1800.0), ("recent", 2400.0), ("old", 2600.0)),
    (("old", 8300.0), ("old", 1200.0), ("recent", 900.0), ("old", 2000.0)),
    (("recent", 7600.0), ("old", 3000.0), ("old", 2200.0), ("recent", 1500.0)),
    (("recent", 7400.0), ("old", 8600.0), ("old", 1100.0), ("recent", 6200.0)),
    (("old", 2500.0), ("old", 2700.0), ("old", 2900.0), ("old", 3100.0)),
    (("recent", 9200.0), ("recent", 8800.0), ("old", 900.0), ("old", 1000.0)),
    (("old", 8100.0), ("old", 7900.0), ("recent", 7500.0), ("recent", 6400.0)),
    (("recent", 3300.0), ("old", 4100.0), ("recent", 2000.0), ("old", 8900.0)),
    (("old", 7000.0), ("old", 7500.0), ("old", 6800.0), ("old", 7100.0)),
    (("recent", 7500.0), ("old", 8200.0), ("recent", 6300.0), ("old", 7800.0)),
    (("old", 1500.0), ("recent", 7500.0), ("old", 8400.0), ("recent", 5900.0)),
    (("recent", 8500.0), ("old", 2200.0), ("old", 3300.0), ("recent", 7600.0)),
    (("old", 6900.0), ("recent", 6100.0), ("old", 7400.0), ("old", 5200.0)),
    (("recent", 9500.0), ("old", 9100.0), ("recent", 8700.0), ("old", 8300.0)),
    (("old", 3600.0), ("old", 8050.0), ("recent", 7450.0), ("old", 2400.0)),
    (("recent", 7700.0), ("recent", 6200.0), ("old", 8600.0), ("old", 4500.0)),
    # line-vs-total PROBE rows (total over the reporting threshold, every
    # line under it) that still refuse on lines 2, 3 or 4, so no fixed
    # per-line strategy such as "approve lines 3 and 4" is free.
    (("old", 3000.0), ("recent", 7600.0), ("old", 2500.0), ("recent", 7400.0)),
    (("old", 2000.0), ("old", 3000.0), ("recent", 7500.0), ("old", 2500.0)),
    (("recent", 7700.0), ("old", 2000.0), ("recent", 7400.0), ("old", 3000.0)),
    (("old", 2500.0), ("recent", 7600.0), ("recent", 7500.0), ("old", 2000.0)),
    # batch counterpart of the two transfer rows above — lines 1 and 3
    # escalate before the W2 threshold step and clear after it.  Also a
    # line-vs-total probe (total 18900 over the reporting threshold, every
    # line under it).
    (("recent", 6900.0), ("old", 3000.0), ("recent", 7000.0), ("old", 2000.0)),
]


def _fb_margin_guard():
    """Every multi-part base keeps the whole 0-20 cents range on one side of
    every value its threshold takes ACROSS THE WHOLE TIMELINE — oscillation
    means a base that is safe in one window can straddle the boundary in
    another, so the check is over the union, not the current window."""
    CAP = {4200.0, 3800.0}
    # scoped by payee age: the recent-payee threshold simply does not apply to
    # a long-established payee, and checking every amount against every bound
    # raises on that coincidence (a $7200 payment to an old payee is not near
    # "the" boundary in any sense that matters)
    XFER_RECENT = {6500.0, 7200.0, 8000.0}
    XFER_OLD = {8000.0}
    CATLIM = {"electronics": {650.0, 460.0}, "travel": {900.0, 1150.0}}
    DAILY = {2000.0, 1700.0}

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
            check(today + v, DAILY, f"F_txn/{cat} daily sum")
    for row in _FB_TRANSFER:
        for age, amt in row:
            check(amt, XFER_RECENT if age == "recent" else XFER_OLD,
                  f"F_transfer/{age}")
    for row in _FB_BATCH:
        for age, amt in row:
            check(amt, XFER_RECENT if age == "recent" else XFER_OLD,
                  f"F_batch/{age}")


_fb_margin_guard()


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


def _policy_firings(kind, acc, mer, parts, state) -> int:
    """n_firing generalised to a multi-part case: the number of policy
    CONDITIONS that hold, summed over parts. The single-request definition is
    the one-part case of this, so the two stay comparable."""
    n = 0
    for part in parts:
        amount = part.get("requested", part.get("amount"))
        if kind == "txn":
            if (mer["region"] == "overseas"
                    and mer["category"] == state["A_R1_corridor_scope"]):
                n += 1
            if acc.get("today_total", 0.0) + amount > state["A_D_daily_cap"]:
                n += 1
            lim = {"electronics": state["A_T1_electronics_limit"],
                   "travel": state["A_T2_travel_limit"]}.get(mer["category"])
            if lim is not None and amount > lim:
                n += 1
        elif kind == "limit":
            if acc["standing"] == "recent_chargeback":
                n += 1
            if amount > state["B_cap"]:
                n += 1
        else:
            payee = part["payee"]
            if (payee.get("added_days_ago", 999) < R.DOCS["new_payee_days"]
                    and amount > state["C_new_payee_threshold"]):
                n += 1
            if amount > state["C_reporting_threshold"]:
                n += 1
    return n


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


# the two splits draw DISJOINT ladder rows, so no scored multi-part ticket is
# a cents-perturbed copy of a same-window serving ticket whose remembered
# answer could be replayed without inferring the policy.
FALLBACK_ROWS = {
    "serving": {"F_txn": (0, 3, 8, 9),
                "F_limit": (0, 2, 6, 9),
                "F_transfer": (5, 7, 10, 14),
                "F_batch": (5, 8, 13, 18)},
    "test": {"F_txn": (1, 2, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15),
             "F_limit": (1, 3, 4, 5, 7, 8, 10, 11),
             "F_transfer": (2, 3, 4, 6, 8, 9, 11, 15, 16, 17, 18, 19),
             "F_batch": (0, 1, 2, 3, 6, 7, 9, 10, 12, 15, 16, 20)},
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
    """Recompute a ticket verdict under an arbitrary L3 policy state."""
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
            # the today_total VALUE RANGE must not name the A_daily family:
            # non-daily transactions carry totals up to ~1880, bounded per-ticket so the sum can
            # never reach the $2000 cap (amount + total <= 1940) — deep
            # overlap with the A_daily range, zero accidental firings.
            # Fixed below the tightest cap: world inputs cannot reveal the
            # hidden 2000 -> 1700 transition through a distribution shift.
            safe_total = 1640.0
            bound = max(0, int((safe_total - f["amount"]) // 10))
            acc["today_total"] = round(
                10.0 * (_h(SEED, "tdt", w, split, family, k) % (bound + 1)), 2)
            # 10-step grid: residues cover both 0 and 10 (mod 20), so the
            # residue does not fingerprint the A_daily family
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
            tid = "bank3_" + hashlib.md5(
                f"{SEED}|{w}|{split}|{family}|{k}".encode()).hexdigest()[:10]
            task = Task(task_id=tid, user_id="banking_l3",
                        instruction=instruction,
                        actions=[Action(name="decide_batch",
                                        kwargs={"case_id": cid,
                                                "decisions": decisions})],
                        answer=None,
                        z={"t": t, "window": w, "tier": "L3", "slice": sl,
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
                           "n_firing": _policy_firings(
                               "transfer", acc, None,
                               [{"payee": _PAYEES[item["payee_id"]],
                                 "amount": item["amount"]} for item in lines],
                               R.truth(w)),
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
            tid = "bank3_" + hashlib.md5(
                f"{SEED}|{w}|{split}|{family}|{k}".encode()).hexdigest()[:10]
            task = Task(task_id=tid, user_id="banking_l3",
                        instruction=instruction,
                        actions=[Action(name=tool, kwargs=kwargs)], answer=None,
                        z={"t": t, "window": w, "tier": "L3", "slice": sl,
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
                           "n_firing": _policy_firings(kind, acc, mer, resolved,
                                                       R.truth(w)),
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
                # precedence are unchanged.
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
        # on overseas tickets, roster membership); both are stamped so
        # reports can use either definition
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
        manual_slice = sl
        if _is_reopen(family, f, w):
            sl = "reopen"
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
        tid = "bank3_" + hashlib.md5(
            f"{SEED}|{w}|{split}|{family}|{k}".encode()).hexdigest()[:10]
        task = Task(task_id=tid,
                    user_id="banking_l3", instruction=instruction,
                    actions=[Action(name=tool, kwargs=kwargs)], answer=None,
                    z={"t": t, "window": w, "tier": "L3", "slice": sl,
                       "manual_slice": manual_slice,
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
        """What the shell balancer must spread: z["decision"] is constant on
        multi-part cases, so balancing it would leave the real answer free to
        concentrate in one shell."""
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
            # keyed on (kind, FAMILY, decision): balancing inside each
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
                   ("C_newpayee", "decision"),
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
            # Preserve every manual-divergent row.  L3's temporal signal is
            # additive to the L2-strength single-policy core, not obtained
            # by deleting persistent anchors.
            # Set to the ceiling so the appended multi-part rows do not
            # evict the single-request adapt rows numeric coverage depends on
            target_adapt_by_window={0: 30, 1: 51, 2: 51, 3: 62, 4: 65, 5: 54},
            prefer_composite=False,
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
        constant_controls = per_window_constant_answer_balance_ids(
            test_by_w,
            families=("A_jewelry", "A_elec_ovs", "B_standing"),
            seed=SEED,
            max_rate=0.80,
            preselected_by_window=protected,
            active_windows_by_family={
                "A_jewelry": {0, 1, 4},
                "A_elec_ovs": {2, 3},
            },
        )
        for w, task_ids in constant_controls.items():
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
        # Before the W2 threshold relaxation, keep more than a token scored
        # sample in the affected numeric band.
        for w in (0, 1):
            early_band = sorted(
                (task for task in test_by_w[w]
                 if task.z.get("family") == "C_newpayee"
                 and 6500.0 < task.z.get("amount", 0.0) <= 7200.0),
                key=lambda task: task.task_id,
            )
            protected[w].update(task.task_id for task in early_band[:3])
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
        w0_travel_low = sorted(
            (task for task in test_by_w[0]
             if task.z.get("family") == "A_travel"
             and task.z.get("amount", 0.0)
                 <= R.truth(0)["A_T2_travel_limit"]),
            key=lambda task: task.task_id,
        )
        protected[0].update(task.task_id for task in w0_travel_low[:3])
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
            prefer_composite=False,
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
    meta = {"scenario": "banking_l3", "domain": "banking", "tier": "L3",
            "system_prompt": _system_prompt(tools)}
    return {
        "meta": meta, "id": "banking_L3",
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
