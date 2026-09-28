"""L3 world: per-window independent order pools, one order per question.

900 serving orders (one per episode) + the frozen test orders across 10
windows (W0-W9; per-window totals in taskgen.ADAPT_TOTALS/GENERAL_TOTALS) +
the 50-order base world. No order is shared between windows or between
serving and test. ~180 extra users added deterministically (unique names).
"""

from __future__ import annotations

import copy
import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple

from ..generate import _date_offset, _digits, CITIES, FIRST_NAMES, LAST_NAMES, STREETS
from ..load import load_data as _load_base
from . import rules
from .taskgen import slot_sequence, test_plan

# 12 Florida addresses, split into disjoint TRAIN/TEST pools so the D3
# rule cannot be learned as "these exact address strings" — a method that
# generalizes 'destination state = FL' transfers; string memorization does not.
FL_ADDRESSES_TRAIN = [
    {"address1": "100 Ocean Drive", "address2": "Suite 100", "city": "Miami",
     "state": "FL", "country": "USA", "zip": "33139"},
    {"address1": "421 Lake Eola Way", "address2": "Apt 12", "city": "Orlando",
     "state": "FL", "country": "USA", "zip": "32801"},
    {"address1": "77 Bayshore Blvd", "address2": "Suite 300", "city": "Tampa",
     "state": "FL", "country": "USA", "zip": "33606"},
    {"address1": "215 Riverside Ave", "address2": "Unit 5", "city": "Jacksonville",
     "state": "FL", "country": "USA", "zip": "32202"},
    {"address1": "58 Seagrape Lane", "address2": "Apt 4B", "city": "Naples",
     "state": "FL", "country": "USA", "zip": "34102"},
    {"address1": "902 University Ave", "address2": "Suite 210", "city": "Gainesville",
     "state": "FL", "country": "USA", "zip": "32601"},
    {"address1": "133 Mizner Blvd", "address2": "Unit 7", "city": "Boca Raton",
     "state": "FL", "country": "USA", "zip": "33432"},
    {"address1": "410 Duval Street", "address2": "Apt 2", "city": "Key West",
     "state": "FL", "country": "USA", "zip": "33040"},
]
FL_ADDRESSES_TEST = [
    {"address1": "764 Beach Drive NE", "address2": "Suite 12", "city": "St. Petersburg",
     "state": "FL", "country": "USA", "zip": "33701"},
    {"address1": "2301 Las Olas Blvd", "address2": "Apt 9C", "city": "Fort Lauderdale",
     "state": "FL", "country": "USA", "zip": "33301"},
    {"address1": "88 Monroe Street", "address2": "Unit 3", "city": "Tallahassee",
     "state": "FL", "country": "USA", "zip": "32301"},
    {"address1": "515 Palm Avenue", "address2": "Suite 8", "city": "Sarasota",
     "state": "FL", "country": "USA", "zip": "34236"},
]
FL_ADDRESSES = FL_ADDRESSES_TRAIN  # incidental FL flavor draws use the train pool
# PO Box pools for the H2 hidden-static rule (L2): disjoint TRAIN/TEST pools,
# same anti-memorization discipline as the Florida pools. Non-FL cities so H2
# tasks can never co-trigger D3.
PO_BOX_TRAIN = [
    {"address1": "PO Box 2211", "address2": "", "city": "Madison", "state": "WI", "country": "USA", "zip": "53701"},
    {"address1": "PO Box 4870", "address2": "", "city": "Denver", "state": "CO", "country": "USA", "zip": "80201"},
    {"address1": "PO Box 913", "address2": "", "city": "Portland", "state": "OR", "country": "USA", "zip": "97207"},
    {"address1": "PO Box 15522", "address2": "", "city": "Tucson", "state": "AZ", "country": "USA", "zip": "85702"},
    {"address1": "PO Box 3308", "address2": "", "city": "Nashville", "state": "TN", "country": "USA", "zip": "37202"},
    {"address1": "PO Box 771", "address2": "", "city": "Omaha", "state": "NE", "country": "USA", "zip": "68101"},
    {"address1": "PO Box 6042", "address2": "", "city": "Baltimore", "state": "MD", "country": "USA", "zip": "21203"},
    {"address1": "PO Box 1189", "address2": "", "city": "Pittsburgh", "state": "PA", "country": "USA", "zip": "15230"},
]
PO_BOX_TEST = [
    {"address1": "PO Box 8064", "address2": "", "city": "Minneapolis", "state": "MN", "country": "USA", "zip": "55480"},
    {"address1": "PO Box 425", "address2": "", "city": "Raleigh", "state": "NC", "country": "USA", "zip": "27602"},
    {"address1": "PO Box 7733", "address2": "", "city": "Salt Lake City", "state": "UT", "country": "USA", "zip": "84110"},
    {"address1": "PO Box 1906", "address2": "", "city": "Columbus", "state": "OH", "country": "USA", "zip": "43216"},
]
TX_ADDRESSES_TRAIN = [
    {"address1": "1400 Congress Ave", "address2": "Suite 210", "city": "Austin", "state": "TX", "country": "USA", "zip": "78701"},
    {"address1": "2205 Elm Street", "address2": "Apt 5B", "city": "Dallas", "state": "TX", "country": "USA", "zip": "75201"},
    {"address1": "910 Bayou Vista", "address2": "Unit 12", "city": "Houston", "state": "TX", "country": "USA", "zip": "77002"},
    {"address1": "334 Alamo Plaza", "address2": "Suite 8", "city": "San Antonio", "state": "TX", "country": "USA", "zip": "78205"},
    {"address1": "87 Sundance Square", "address2": "Apt 3C", "city": "Fort Worth", "state": "TX", "country": "USA", "zip": "76102"},
    {"address1": "512 Mesa Drive", "address2": "Unit 4", "city": "El Paso", "state": "TX", "country": "USA", "zip": "79901"},
    {"address1": "6100 Tech Ridge Rd", "address2": "Suite 330", "city": "Plano", "state": "TX", "country": "USA", "zip": "75024"},
    {"address1": "221 Riverwalk Lane", "address2": "Apt 9", "city": "Corpus Christi", "state": "TX", "country": "USA", "zip": "78401"},
]
TX_ADDRESSES_TEST = [
    {"address1": "845 Longhorn Blvd", "address2": "Suite 5", "city": "Lubbock", "state": "TX", "country": "USA", "zip": "79401"},
    {"address1": "132 Prairie View Dr", "address2": "Apt 2A", "city": "Amarillo", "state": "TX", "country": "USA", "zip": "79101"},
    {"address1": "58 Gulf Breeze Way", "address2": "Unit 7", "city": "Galveston", "state": "TX", "country": "USA", "zip": "77550"},
    {"address1": "410 Hill Country Rd", "address2": "Suite 12", "city": "Waco", "state": "TX", "country": "USA", "zip": "76701"},
]
# Pool of non-FL addresses used when an order must NOT be Florida-bound (the
# user's own address happens to be FL). Rotated by order sequence so no single
# address accumulates hundreds of uses.
SAFE_ADDRESSES = [
    {"address1": "742 River Road", "address2": "Suite 201", "city": "Denver",
     "state": "CO", "country": "USA", "zip": "80014"},
    {"address1": "18 Maple Hollow", "address2": "Apt 3A", "city": "Portland",
     "state": "OR", "country": "USA", "zip": "97205"},
    {"address1": "960 Canyon View Dr", "address2": "Unit 12", "city": "Phoenix",
     "state": "AZ", "country": "USA", "zip": "85018"},
    {"address1": "305 Birchwood Lane", "address2": "Suite 5", "city": "Madison",
     "state": "WI", "country": "USA", "zip": "53703"},
    {"address1": "1201 Harbor Point", "address2": "Apt 22", "city": "Baltimore",
     "state": "MD", "country": "USA", "zip": "21230"},
    {"address1": "47 Foothill Terrace", "address2": "Unit 9", "city": "Salt Lake City",
     "state": "UT", "country": "USA", "zip": "84103"},
    {"address1": "633 Prairie Street", "address2": "Suite 410", "city": "Omaha",
     "state": "NE", "country": "USA", "zip": "68102"},
    {"address1": "289 Beacon Hill Rd", "address2": "Apt 6C", "city": "Nashville",
     "state": "TN", "country": "USA", "zip": "37206"},
]


def _swap_pairs(products, categories):
    pairs = []
    for pid in sorted(products):
        p = products[pid]
        if p["category"] not in categories:
            continue
        ok = [p["variants"][iid] for iid in sorted(p["variants"])
              if p["variants"][iid]["available"] and not p["variants"][iid].get("clearance")]
        for old in ok:
            for new in ok:
                if new["item_id"] != old["item_id"] and new["price"] >= old["price"]:
                    pairs.append({"product_id": pid, "name": p["name"],
                                  "category": p["category"], "old": old, "new": new})
    return pairs


def _plain_items(products, max_price=None, clearance=False, categories=None):
    items = []
    for pid in sorted(products):
        p = products[pid]
        if categories and p["category"] not in categories:
            continue
        for iid in sorted(p["variants"]):
            v = p["variants"][iid]
            if bool(v.get("clearance")) != clearance:
                continue
            if max_price is not None and v["price"] > max_price:
                continue
            items.append({"product_id": pid, "name": p["name"],
                          "category": p["category"], "variant": v})
    return items


def _opts(o):
    return ", ".join(f"{k}: {v}" for k, v in sorted(o.items()))


def _add_users(data, n=60):
    rng = random.Random(4242)
    users = data["users"]
    # Full names must be UNIQUE across the whole user base: two customers named
    # "Ava Okafor" would make name+zip authentication depend on zip luck and
    # turn identical-looking tickets into different accounts.
    taken = {(u["name"]["first_name"].lower(), u["name"]["last_name"].lower())
             for u in users.values()}
    added = 0
    while added < n:
        first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
        if (first.lower(), last.lower()) in taken:
            continue
        uid = f"{first}_{last}_{_digits(rng, 4)}"
        if uid in users:
            continue
        taken.add((first.lower(), last.lower()))
        city, state, zp = rng.choice(CITIES)
        pms = {}
        # 1-3 credit cards (extras are distractors); ~75% hold a gift card.
        # Task builders pin explicit ids and add gift cards on demand (gift()).
        for _ in range(rng.choices([1, 2, 3], weights=[45, 40, 15])[0]):
            cc = f"credit_card_{_digits(rng, 7)}"
            pms[cc] = {"source": "credit_card", "id": cc,
                       "brand": rng.choice(["visa", "mastercard", "amex"]),
                       "last_four": _digits(rng, 4)}
        if rng.random() < 0.75:
            gc = f"gift_card_{_digits(rng, 7)}"
            pms[gc] = {"source": "gift_card", "id": gc,
                       "balance": round(rng.uniform(5, 140), 2)}
        users[uid] = {
            "user_id": uid, "membership": rng.choices(["bronze", "silver", "gold"],
                                                      weights=[40, 40, 20])[0],
            "name": {"first_name": first.capitalize(), "last_name": last.capitalize()},
            "address": {"address1": f"{rng.randint(100, 999)} {rng.choice(STREETS)}",
                        "address2": f"Suite {rng.randint(100, 999)}",
                        "city": city, "state": state, "country": "USA",
                        "zip": zp + _digits(rng, 3)},
            "email": f"{first}.{last}{_digits(rng, 4)}@example.com",
            "payment_methods": pms,
            "orders": [],
        }
        added += 1


class _Builder:
    def __init__(self, data):
        self.data = data
        self.uids = sorted(data["users"])
        self.seq = 0
        self.gseq = 0
        self.cursor = 0
        # Opaque-ID RNG: isolated from the world/timeline RNG streams so ids
        # carry NO information about episode time, window, family, or slice.
        self.idrng = random.Random(90210)
        p = data["products"]
        self.elec_pairs = _swap_pairs(p, {"electronics"})
        self.soft_pairs = _swap_pairs(p, {"apparel", "home"})
        # name-grouped views: drawing round-robin over PRODUCT NAMES (then over
        # variants within a name) keeps every product's share ~= 1/n_names even
        # when the raw pair pool is dominated by one product's many variants.
        def _by_name(pairs):
            groups = {}
            for pr in pairs:
                groups.setdefault(pr["name"], []).append(pr)
            return [groups[k] for k in sorted(groups)]
        self.elec_groups = _by_name(self.elec_pairs)
        self.soft_groups = _by_name(self.soft_pairs)
        self.cheap = _plain_items(p, max_price=150.0)
        self.mid = [e for e in _plain_items(p) if 210.0 <= e["variant"]["price"] <= 290.0]
        self.elec = _plain_items(p, categories={"electronics"})
        self.soft = _plain_items(p, categories={"apparel", "home"})
        self.clear = _plain_items(p, clearance=True)
        self.big = sorted(_plain_items(p), key=lambda e: -e["variant"]["price"])
        # --- L2 pools ---------------------------------------------------------
        # old available non-clearance -> new UNAVAILABLE variant (documented
        # blocker for the doc-derivable fallback family gl2_fallback)
        unavail = []
        for pid in sorted(p):
            pr = p[pid]
            if pr["category"] not in ("apparel", "home"):
                continue
            olds = [v for v in sorted(pr["variants"]) if pr["variants"][v]["available"]
                    and not pr["variants"][v].get("clearance")]
            uns = [v for v in sorted(pr["variants"]) if not pr["variants"][v]["available"]]
            for o in olds:
                for n in uns:
                    unavail.append({"product_id": pid, "name": pr["name"],
                                    "category": pr["category"],
                                    "old": pr["variants"][o], "new": pr["variants"][n]})
        self.unavail_groups = _by_name(unavail)
        # clearance variant with an available non-clearance SIBLING target on the
        # same product (the customer names a target; the clearance ITEM blocks)
        clear_sib = []
        for pid in sorted(p):
            pr = p[pid]
            if pr["category"] not in ("apparel", "home"):
                continue
            cls = [v for v in sorted(pr["variants"]) if pr["variants"][v].get("clearance")]
            sibs = [v for v in sorted(pr["variants"]) if pr["variants"][v]["available"]
                    and not pr["variants"][v].get("clearance")]
            for c in cls:
                clear_sib.append({"product_id": pid, "name": pr["name"],
                                  "category": pr["category"],
                                  "old": pr["variants"][c], "new": pr["variants"][sibs[0]]})
        self.clear_sib = clear_sib
        # cheap soft pairs whose totals sit under EVERY tier's cancel cap, so the
        # b2_d6 cancel fallback can never trip over_threshold
        from ..policy_config import get_config as _gc
        self._cancel_cap = min(_gc(4)["cancel_max_total"].values())
        self.cheap_soft_groups = _by_name(
            [x for x in self.soft_pairs if x["new"]["price"] <= self._cancel_cap * 0.8])
        # --- scope-drift pools ------------------------------------------------
        from . import rules as _R
        audio = [x for x in self.elec_pairs if x["name"] in _R.AUDIO_PRODUCTS]
        nonaudio = [x for x in self.elec_pairs if x["name"] not in _R.AUDIO_PRODUCTS]
        self.audio_groups = _by_name(audio)
        self.nonaudio_groups = _by_name(nonaudio)
        assert len(self.audio_groups) >= 3, "audio pair pool thin"
        assert len(self.nonaudio_groups) >= 5, "non-audio electronics pool thin"
        assert len(self.unavail_groups) >= 3, "unavailable-target pool thin"
        assert len({e["name"] for e in self.clear_sib}) >= 4, "clearance-sibling pool thin"
        assert len(self.cheap_soft_groups) >= 8, "cheap soft pair pool thin"
        # pool-sufficiency check: every task family must have generous supply
        assert len({e["name"] for e in self.elec_pairs}) >= 10, "elec pair names thin"
        assert len({e["name"] for e in self.soft_pairs}) >= 20, "soft pair names thin"
        assert len(self.cheap) >= 40, "cheap pool thin"
        assert len(self.mid) >= 20, "mid pool thin"
        assert len(self.clear) >= 8, "clearance pool thin"
        assert sum(e["variant"]["price"] for e in self.big[:3]) > 520.0, "big pool thin"

    def user(self):
        u = self.data["users"][self.uids[self.cursor % len(self.uids)]]
        self.cursor += 1
        return u

    def user_of_tier(self, tier):
        for _ in range(len(self.uids)):
            u = self.user()
            if u["membership"] == tier:
                return u
        raise AssertionError(f"no {tier} user")

    def card(self, u):
        return next(pid for pid, pm in u["payment_methods"].items()
                    if pm["source"] == "credit_card")

    def gift(self, u):
        for pid, pm in u["payment_methods"].items():
            if pm["source"] == "gift_card":
                return pid
        gid = f"gift_card_{''.join(str(self.idrng.randint(0, 9)) for _ in range(7))}"
        bal = round(20.0 + (self.gseq * 37) % 121 + ((self.gseq * 7) % 100) / 100, 2)
        self.gseq += 1
        u["payment_methods"][gid] = {"source": "gift_card", "id": gid, "balance": bal}
        return gid

    def card_of_brand(self, u, brand):
        """A credit card of the given brand ('amex' | 'visa'), or explicitly
        NOT amex ('non_amex'); minted deterministically if the user lacks
        one. 'visa' is strict — on_visa windows open ONLY that brand, so a
        mastercard host would silently change the GT."""
        for pid, pm in u["payment_methods"].items():
            if pm["source"] != "credit_card":
                continue
            if brand == "visa" and pm["brand"] == "visa":
                return pid
            if brand == "amex" and pm["brand"] == "amex":
                return pid
            if brand == "non_amex" and pm["brand"] != "amex":
                return pid
        cc = f"credit_card_{''.join(str(self.idrng.randint(0, 9)) for _ in range(7))}"
        u["payment_methods"][cc] = {
            "source": "credit_card", "id": cc,
            "brand": "amex" if brand == "amex" else "visa",
            "last_four": "".join(str(self.idrng.randint(0, 9)) for _ in range(4))}
        return cc

    def _order(self, u, items, status, payments, ago, address=None):
        while True:
            oid = "#W" + "".join(str(self.idrng.randint(0, 9)) for _ in range(7))
            if oid not in self.data["orders"]:
                break
        self.seq += 1
        addr = dict(address or u["address"])
        if address is None and addr.get("state") in ("FL", "TX"):
            addr = dict(SAFE_ADDRESSES[self.seq % len(SAFE_ADDRESSES)])
        self.data["orders"][oid] = {
            "order_id": oid, "user_id": u["user_id"], "address": addr, "items": items,
            "status": status, "placed_at": _date_offset((ago or 1) + 4),
            **({"delivered_at": _date_offset(ago)} if status == "delivered" else {}),
            "payment_history": [{"transaction_type": "payment", "amount": round(a, 2),
                                 "payment_method_id": pid} for pid, a in payments]}
        u["orders"].append(oid)
        return oid

    def _fee(self, u):
        from ..policy_config import get_config as _gc
        return _gc(4)["return_fee"][u["membership"]]

    SIZE_CYCLE = (3, 4, 5, 3, 4, 5, 3, 4, 5, 5)   # interleaved: 3:90 / 4:70 / 5:rest

    def _size(self, i):
        return self.SIZE_CYCLE[i % len(self.SIZE_CYCLE)]

    def _mixed5(self, i, n_audio=1, n_elec=1, n_soft=3, n_clear=0):
        """5 distinct-product items: audio + non-audio electronics + soft
        (+clearance replacing soft). Returns (items, pairs) aligned."""
        prs = []
        for k in range(n_audio):
            g = self.audio_groups[(i + k) % len(self.audio_groups)]
            prs.append(g[(i // len(self.audio_groups)) % len(g)])
        for k in range(n_elec):
            g = self.nonaudio_groups[(i * 2 + k) % len(self.nonaudio_groups)]
            prs.append(g[(i // len(self.nonaudio_groups)) % len(g)])
        for k in range(n_soft - n_clear):
            g = self.soft_groups[(i * 3 + k) % len(self.soft_groups)]
            prs.append(g[(i // len(self.soft_groups)) % len(g)])
        items = []
        for pr in prs:
            v = pr["old"]
            items.append({"name": pr["name"], "product_id": pr["product_id"],
                          "item_id": v["item_id"], "price": v["price"],
                          "options": v["options"]})
        for k in range(n_clear):
            e = self.clear[(i + k) % len(self.clear)]
            items.append({"name": e["name"], "product_id": e["product_id"],
                          "item_id": e["variant"]["item_id"],
                          "price": e["variant"]["price"],
                          "options": e["variant"]["options"]})
            prs.append(None)
        assert len({x["item_id"] for x in items}) == len(items)
        rot = i % len(items)
        return items[rot:] + items[:rot], prs[rot:] + prs[:rot]

    def make(self, family: str, i: int, split: str = "train"):
        u = self.user()
        card = self.card(u)
        win_ago = [4, 6, 8, 10][i % 4]
        spec = {"family": family, "user_id": u["user_id"],
                "first": u["name"]["first_name"], "last": u["name"]["last_name"],
                "zip": u["address"]["zip"], "tier": "L3", "ask": None}

        def pair(groups, off=0):
            g = groups[(i + off) % len(groups)]
            return g[(i // len(groups)) % len(g)]

        def item_of(pr):
            v = pr["old"]
            return {"name": pr["name"], "product_id": pr["product_id"],
                    "item_id": v["item_id"], "price": v["price"],
                    "options": v["options"]}

        def geo(kind):
            if kind == "fl":
                pool = FL_ADDRESSES_TEST if split == "test" else FL_ADDRESSES_TRAIN
            elif kind == "tx":
                pool = TX_ADDRESSES_TEST if split == "test" else TX_ADDRESSES_TRAIN
            else:
                pool = PO_BOX_TEST if split == "test" else PO_BOX_TRAIN
            return pool[i % len(pool)]

        # ---------- single-item EXCHANGE probes --------------------------------
        if family == "p9_up":
            # STATIC PERMISSIVE probe (Z8): single-item soft exchange
            # with a POSITIVE upcharge, settled with a gift card provisioned
            # with balance > diff. Docs forbid gift-funded upcharges; truth
            # accepts -> CARRY-OUT adapt in every window (incl. W0/W4, the
            # all-restrictive phases). Safe address isolates Z6/Z7.
            g_ = self.soft_groups[(i * 7 + 3) % len(self.soft_groups)]
            pr = next((x for x in g_
                       if x["new"]["price"] - x["old"]["price"] >= 10.0), None)
            if pr is None:
                pr = max(g_, key=lambda x: x["new"]["price"] - x["old"]["price"])
            diff = round(pr["new"]["price"] - pr["old"]["price"], 2)
            assert diff > 0, "p9_up: needs a positive upcharge"
            it = {"name": pr["name"], "product_id": pr["product_id"],
                  "item_id": pr["old"]["item_id"], "price": pr["old"]["price"],
                  "options": pr["old"]["options"]}
            safe = SAFE_ADDRESSES[(i * 3 + 1) % len(SAFE_ADDRESSES)]
            oid = self._order(u, [it], "delivered", [(card, it["price"])],
                              win_ago, address=safe)
            gid = f"gift_card_{''.join(str(self.idrng.randint(0, 9)) for _ in range(7))}"
            self.gseq += 1
            u["payment_methods"][gid] = {"source": "gift_card", "id": gid,
                                         "balance": round(diff + 20.0
                                                          + (i % 5) * 7, 2)}
            spec.update({"order_id": oid, "pool": "p9_up"})
            spec["request"] = [{
                "op": "exchange", "item_ids": [it["item_id"]],
                "new_item_ids": [pr["new"]["item_id"]], "pay": gid,
                "ctx": {"addr_state":
                            self.data["orders"][oid]["address"].get("state"),
                        "pobox": False, "order_multi": False,
                        "partial": False, "upcharge_gift": True},
                "items_ctx": [{"category": pr["category"],
                               "product_name": pr["name"]}]}]
            spec["fmt"] = {"oid": oid, "item_name": pr["name"],
                           "item_id": it["item_id"],
                           "new_item_id": pr["new"]["item_id"],
                           "new_opts": _opts(pr["new"]["options"]),
                           "pay_id": gid}

        elif family in ("p_aud_exch", "p_cam_exch", "p_soft_exch",
                      "p_fl_exch", "p_tx_exch", "p_pobox_exch"):
            pr = pair({"p_aud_exch": self.audio_groups,
                       "p_cam_exch": self.nonaudio_groups}.get(
                          family, self.soft_groups), off=3)
            it = item_of(pr)
            addr = {"p_fl_exch": geo("fl"), "p_tx_exch": geo("tx"),
                    "p_pobox_exch": geo("po")}.get(family)
            oid = self._order(u, [it], "delivered", [(card, it["price"])],
                              win_ago, address=addr)
            actual_state = self.data["orders"][oid]["address"].get("state")
            spec.update({"order_id": oid, "pool": "exchange", "tier": "L1"})
            spec["request"] = [{
                "op": "exchange", "item_ids": [it["item_id"]],
                "new_item_ids": [pr["new"]["item_id"]], "pay": card,
                "ctx": {"addr_state": actual_state,
                        "pobox": family == "p_pobox_exch",
                        "order_multi": False, "partial": False},
                "items_ctx": [{"category": pr["category"],
                               "product_name": pr["name"]}]}]
            spec["fmt"] = {"oid": oid, "item_name": pr["name"],
                           "item_id": it["item_id"],
                           "new_item_id": pr["new"]["item_id"],
                           "new_opts": _opts(pr["new"]["options"]),
                           "pay_id": card}

        # ---------- CANCEL probes ---------------------------------------------
        elif family in ("p_gift_cancel", "p_split_cancel", "p_card_cancel"):
            it = {"name": None}
            e = self.cheap[(i * 7 + len(family)) % len(self.cheap)]
            it = {"name": e["name"], "product_id": e["product_id"],
                  "item_id": e["variant"]["item_id"], "price": e["variant"]["price"],
                  "options": e["variant"]["options"]}
            if family == "p_gift_cancel":
                g = self.gift(u)
                pays, kind = [(g, it["price"])], "gift_only"
            elif family == "p_split_cancel":
                g = self.gift(u)
                a = round(it["price"] * 0.6, 2)
                pays, kind = [(card, a), (g, round(it["price"] - a, 2))], "split"
            else:
                pays, kind = [(card, it["price"])], "card"
            oid = self._order(u, [it], "pending", pays, None)
            spec.update({"order_id": oid, "pool": "cancel", "tier": "L1"})
            spec["request"] = [{"op": "cancel", "ctx": {"pay_kind": kind}}]
            spec["fmt"] = {"oid": oid}

        # ---------- RETURN destination probes ---------------------------------
        elif family in ("p_amex_ret", "p_visa_ret", "p_gift_ret"):
            it = item_of(pair(self.soft_groups, off=7))
            if family == "p_amex_ret":
                dest = self.card_of_brand(u, "amex"); brand = "amex"
            elif family == "p_visa_ret":
                dest = self.card_of_brand(u, "visa"); brand = "visa"
            else:
                dest = self.gift(u); brand = None
            paycard = dest if family != "p_gift_ret" else card
            oid = self._order(u, [it], "delivered", [(paycard, it["price"])],
                              win_ago)
            spec.update({"order_id": oid, "pool": "return_one", "tier": "L1"})
            spec["request"] = [{
                "op": "return", "item_ids": [it["item_id"]], "dest": dest,
                "ctx": {"dest_original_card": family != "p_gift_ret",
                        "card_brand": brand, "order_multi": False,
                        "partial": False, "order_has_elec": False}}]
            # the refusal GT must be DERIVABLE from the docs: the gift-card-
            # only clause refuses only a customer who INSISTS on the original
            # card and declines a gift card — so the ticket must insist
            dphrase = (f"my gift card ({dest})" if family == "p_gift_ret"
                       else f"my original credit card ({dest}) — the card I "
                            f"paid with. That card is the ONLY destination "
                            f"I'll accept; please don't credit a gift card")
            spec["fmt"] = {"oid": oid, "item_name": it["name"],
                           "item_id": it["item_id"], "dest": dphrase}

        # ---------- MODIFY probes ---------------------------------------------
        elif family in ("p_split_mod", "p_card_mod"):
            pr = pair(self.cheap_soft_groups, off=5)
            it = item_of(pr)
            if family == "p_split_mod":
                g = self.gift(u)
                a = round(it["price"] * 0.6, 2)
                pays, kind = [(card, a), (g, round(it["price"] - a, 2))], "split"
            else:
                pays, kind = [(card, it["price"])], "card"
            oid = self._order(u, [it], "pending", pays, None)
            spec.update({"order_id": oid, "pool": "modify", "tier": "L1"})
            spec["request"] = [{
                "op": "modify", "item_ids": [it["item_id"]],
                "new_item_ids": [pr["new"]["item_id"]], "pay": card,
                "ctx": {"pay_kind": kind}}]
            spec["fmt"] = {"oid": oid, "item_name": pr["name"],
                           "item_id": it["item_id"],
                           "new_item_id": pr["new"]["item_id"],
                           "new_opts": _opts(pr["new"]["options"]),
                           "pay_id": card}

        # ---------- partial/full RETURN + partial EXCHANGE probes --------------
        elif family in ("p_part_ret", "p_part_ret_el", "p_full_ret",
                        "p_part_exch"):
            n_elec = 1 if family == "p_part_ret_el" else 0
            items, prs = self._mixed5(i, n_audio=0, n_elec=n_elec,
                                      n_soft=3 - n_elec)
            total = round(sum(x["price"] for x in items), 2)
            oid = self._order(u, items, "delivered", [(card, total)], win_ago)
            gid = self.gift(u)
            spec["order_id"] = oid
            has_elec = n_elec > 0
            # target a SOFT item (index of a soft one)
            soft_ix = next(k for k, pr in enumerate(prs)
                           if pr and pr["category"] != "electronics")
            if family == "p_part_exch":
                pr = prs[soft_ix]
                spec.update({"pool": "exchange", "tier": "L1"})
                spec["request"] = [{
                    "op": "exchange", "item_ids": [items[soft_ix]["item_id"]],
                    "new_item_ids": [pr["new"]["item_id"]], "pay": card,
                    "ctx": {"addr_state": self.data["orders"][oid]["address"].get("state"),
                            "pobox": False, "order_multi": True, "partial": True},
                    "items_ctx": [{"category": pr["category"],
                                   "product_name": pr["name"]}]}]
                spec["fmt"] = {"oid": oid, "item_name": pr["name"],
                               "item_id": items[soft_ix]["item_id"],
                               "new_item_id": pr["new"]["item_id"],
                               "new_opts": _opts(pr["new"]["options"]),
                               "pay_id": card}
            elif family == "p_full_ret":
                spec.update({"pool": "ret_multi", "tier": "L1"})
                spec["request"] = [{
                    "op": "return", "item_ids": [x["item_id"] for x in items],
                    "dest": gid,
                    "ctx": {"dest_original_card": False, "card_brand": None,
                            "order_multi": True, "partial": False,
                            "order_has_elec": has_elec}}]
                cl = ", ".join(f"the {x['name']} (item {x['item_id']})" for x in items)
                spec["fmt"] = {"oid": oid, "clauses": cl,
                               "dest": f"my gift card ({gid})"}
            else:
                tgt = items[soft_ix]
                spec.update({"pool": "return_one", "tier": "L1"})
                spec["request"] = [{
                    "op": "return", "item_ids": [tgt["item_id"]], "dest": gid,
                    "ctx": {"dest_original_card": False, "card_brand": None,
                            "order_multi": True, "partial": True,
                            "order_has_elec": has_elec}}]
                spec["fmt"] = {"oid": oid, "item_name": tgt["name"],
                               "item_id": tgt["item_id"],
                               "dest": f"my gift card ({gid})"}

        # ---------- A: 5-item mixed subset execution ---------------------------
        elif family in ("mxr", "mxr_el", "gl_mx"):
            n_tot = self._size(i)
            n_clear = 0 if family == "gl_mx" else 1
            n_elec = 1 if family == "mxr_el" else 0
            items, prs = self._mixed5(i, n_audio=0, n_elec=n_elec,
                                      n_soft=n_tot - n_elec, n_clear=n_clear)
            total = round(sum(x["price"] for x in items), 2)
            oid = self._order(u, items, "delivered", [(card, total)], win_ago)
            gid = self.gift(u)
            spec["order_id"] = oid
            full = family == "gl_mx"          # whole-order request (R6-immune)
            req_ix = (list(range(len(items))) if full else
                      list(range(len(items)))[:len(items) - 1])
            def _is_clear(it):
                pv = self.data["products"][it["product_id"]]["variants"]
                return bool(pv[it["item_id"]].get("clearance"))
            bad = [j for j, k in enumerate(req_ix) if _is_clear(items[k])]
            has_el = any(pr and pr["category"] == "electronics" for pr in prs)
            spec["request"] = [{
                "op": "return", "item_ids": [items[k]["item_id"] for k in req_ix],
                "dest": gid, "doc_bad_idx": bad,
                "ctx": {"dest_original_card": False, "card_brand": None,
                        "order_multi": True,
                        "partial": len(req_ix) < len(items),
                        "order_has_elec": has_el}}]
            spec["pool"] = "ret_multi"
            cl = ", ".join(f"the {items[k]['name']} (item {items[k]['item_id']})"
                           for k in req_ix)
            spec["fmt"] = {"oid": oid, "clauses": cl,
                           "dest": f"my gift card ({gid})"}

        elif family == "mxe":
            n_tot = self._size(i)
            n_el = 1 if n_tot < 5 else 1 + (i % 2)
            items, prs = self._mixed5(i, n_audio=1, n_elec=n_el,
                                      n_soft=n_tot - 1 - n_el)
            total = round(sum(x["price"] for x in items), 2)
            oid = self._order(u, items, "delivered", [(card, total)], win_ago)
            spec["order_id"] = oid
            cand = [k for k, pr in enumerate(prs) if pr is not None]
            el_ix = [k for k in cand if prs[k]["category"] == "electronics"]
            req_ix = ([el_ix[0]] + [k for k in cand if k != el_ix[0]])[:len(items) - 1]
            spec["request"] = [{
                "op": "exchange",
                "item_ids": [items[k]["item_id"] for k in req_ix],
                "new_item_ids": [prs[k]["new"]["item_id"] for k in req_ix],
                "pay": card,
                "ctx": {"addr_state": self.data["orders"][oid]["address"].get("state"),
                        "pobox": False, "order_multi": True,
                        "partial": len(req_ix) < len(items)},
                "items_ctx": [{"category": prs[k]["category"],
                               "product_name": prs[k]["name"]} for k in req_ix]}]
            spec["pool"] = "exch_multi"
            cl = ", and ".join(
                f"the {prs[k]['name']} (item {items[k]['item_id']}) for the "
                f"variant with {_opts(prs[k]['new']['options'])} "
                f"(item {prs[k]['new']['item_id']})" for k in req_ix)
            spec["fmt"] = {"oid": oid, "clauses": cl, "pay_id": card}

        # ---------- B: aggregate refund questions ------------------------------
        elif family in ("agg", "agg_el", "gl_agg"):
            has_elec = family == "agg_el"
            n_clear = 1 if family != "gl_agg" and i % 3 == 0 else 0
            n_tot = self._size(i)
            # n_soft is the SOFT SLOT count; _mixed5 replaces n_clear of them
            # with clearance items — do not subtract n_clear here again
            items, prs = self._mixed5(i, n_audio=0, n_elec=1 if has_elec else 0,
                                      n_soft=n_tot - (1 if has_elec else 0),
                                      n_clear=n_clear)
            total = round(sum(x["price"] for x in items), 2)
            oid = self._order(u, items, "delivered", [(card, total)], win_ago)
            gid = self.gift(u)
            spec["order_id"] = oid
            full = family == "gl_agg"
            req_ix = (list(range(len(items))) if full else
                      sorted([k for k, pr in enumerate(prs)
                              if pr and pr["category"] != "electronics"][:2] +
                             [k for k, pr in enumerate(prs) if pr is None][:1]))
            def _is_clear(it):
                pv = self.data["products"][it["product_id"]]["variants"]
                return bool(pv[it["item_id"]].get("clearance"))
            bad = [j for j, k in enumerate(req_ix) if _is_clear(items[k])]
            spec["request"] = [{
                "op": "return", "item_ids": [items[k]["item_id"] for k in req_ix],
                "dest": gid, "doc_bad_idx": bad,
                "ctx": {"dest_original_card": False, "card_brand": None,
                        "order_multi": True, "partial": not full,
                        "order_has_elec": has_elec}}]
            spec["ask"] = {"prices": [items[k]["price"] for k in req_ix],
                           "fee": self._fee(u)}
            spec["pool"] = "agg"
            cl = ", ".join(f"the {items[k]['name']} (item {items[k]['item_id']})"
                           for k in req_ix)
            spec["fmt"] = {"oid": oid, "clauses": cl,
                           "dest": f"my gift card ({gid})"}

        # ---------- C: ordered-preference chains -------------------------------
        elif family in ("chn_au", "chn_av", "chn_cam", "chn_s", "chn_tx",
                        "gl_chn"):
            grp = {"chn_au": self.audio_groups, "chn_av": self.audio_groups,
                   "chn_cam": self.nonaudio_groups}.get(family, self.soft_groups)
            pr = pair(grp, off=1)
            it = item_of(pr)
            p = self.data["products"][pr["product_id"]]
            variants = [v for v in sorted(p["variants"])
                        if v != it["item_id"]]
            avail = [v for v in variants if p["variants"][v]["available"]
                     and not p["variants"][v].get("clearance")]
            unav = [v for v in variants if not p["variants"][v]["available"]]
            first_unav = family in ("gl_chn", "chn_av", "chn_s")
            va = (p["variants"][unav[0]] if (first_unav and unav)
                  else p["variants"][avail[0]])
            vb = p["variants"][avail[-1] if va["item_id"] != avail[-1]
                               else avail[0]]
            brand_dest = self.card_of_brand(
                u, "amex" if i % 3 == 0 else "non_amex")
            brand = u["payment_methods"][brand_dest]["brand"]
            oid = self._order(u, [it], "delivered",
                              [(brand_dest, it["price"])], win_ago,
                              address=geo("tx") if family == "chn_tx" else None)
            gid = self.gift(u)
            spec["order_id"] = oid
            ex_ctx = {"addr_state": self.data["orders"][oid]["address"].get("state"),
                      "pobox": False, "order_multi": False, "partial": False}
            it_ctx = [{"category": pr["category"], "product_name": pr["name"]}]
            spec["request"] = [
                {"op": "exchange", "item_ids": [it["item_id"]],
                 "new_item_ids": [va["item_id"]], "pay": brand_dest,
                 "mech_block": ("unavailable" if not va["available"] else None),
                 "ctx": ex_ctx, "items_ctx": it_ctx},
                {"op": "exchange", "item_ids": [it["item_id"]],
                 "new_item_ids": [vb["item_id"]], "pay": brand_dest,
                 "ctx": ex_ctx, "items_ctx": it_ctx},
                {"op": "return", "item_ids": [it["item_id"]], "dest": brand_dest,
                 "ctx": {"dest_original_card": True, "card_brand": brand,
                         "order_multi": False, "partial": False,
                         "order_has_elec": pr["category"] == "electronics"}},
                {"op": "return", "item_ids": [it["item_id"]], "dest": gid,
                 "ctx": {"dest_original_card": False, "card_brand": None,
                         "order_multi": False, "partial": False,
                         "order_has_elec": pr["category"] == "electronics"}},
            ]
            spec["pool"] = "chain"
            spec["fmt"] = {"oid": oid, "item_name": pr["name"],
                           "item_id": it["item_id"],
                           "id_a": va["item_id"], "opts_a": _opts(va["options"]),
                           "id_b": vb["item_id"], "opts_b": _opts(vb["options"]),
                           "pay_id": brand_dest, "gift_id": gid}

        # ---------- E: nothing-works refusals ----------------------------------
        elif family == "ret2":
            # soft multi-item PARTIAL return to an ORIGINAL AMEX card: outcome
            # = f(R6 state x R3 state) — execute / invalid_status / dna
            its = []
            for k2 in range(3):
                g_ = self.soft_groups[(i * 3 + k2 + 8) % len(self.soft_groups)]
                its.append(g_[(i // len(self.soft_groups)) % len(g_)])
            items = [ {"name": pr["name"], "product_id": pr["product_id"],
                       "item_id": pr["old"]["item_id"], "price": pr["old"]["price"],
                       "options": pr["old"]["options"]} for pr in its ]
            assert len({x["item_id"] for x in items}) == 3
            dest = self.card_of_brand(u, "amex")
            total = round(sum(x["price"] for x in items), 2)
            oid = self._order(u, items, "delivered", [(dest, total)], win_ago)
            tgt = items[i % 3]
            spec.update({"order_id": oid, "pool": "return_one"})
            spec["request"] = [{
                "op": "return", "item_ids": [tgt["item_id"]], "dest": dest,
                "ctx": {"dest_original_card": True, "card_brand": "amex",
                        "order_multi": True, "partial": True,
                        "order_has_elec": False}}]
            spec["fmt"] = {"oid": oid, "item_name": tgt["name"],
                           "item_id": tgt["item_id"],
                           "dest": f"my original credit card ({dest}) — the card I paid with"}

        elif family == "exch2":
            # audio item + TX-bound order: R1 x R4 joint refusal shape
            pr = pair(self.audio_groups, off=1)
            it = item_of(pr)
            oid = self._order(u, [it], "delivered", [(card, it["price"])],
                              win_ago, address=geo("fl"))
            spec.update({"order_id": oid, "pool": "exchange", "tier": "L2"})
            spec["request"] = [{
                "op": "exchange", "item_ids": [it["item_id"]],
                "new_item_ids": [pr["new"]["item_id"]], "pay": card,
                "ctx": {"addr_state": self.data["orders"][oid]["address"].get("state"),
                        "pobox": False, "order_multi": False, "partial": False},
                "items_ctx": [{"category": pr["category"],
                               "product_name": pr["name"]}]}]
            spec["fmt"] = {"oid": oid, "item_name": pr["name"],
                           "item_id": it["item_id"],
                           "new_item_id": pr["new"]["item_id"],
                           "new_opts": _opts(pr["new"]["options"]), "pay_id": card}

        elif family == "mxa":
            # whole-order exchange EXECUTION: customer lists every item with a
            # target variant; GT executes the R1-eligible subset (partial=False
            # keeps R7 out) — the executed set IS the boundary knowledge
            n_tot = self._size(i)
            n_el = 1 if n_tot < 5 else 1 + (i % 2)
            items, prs = self._mixed5(i, n_audio=1, n_elec=n_el,
                                      n_soft=n_tot - 1 - n_el)
            total = round(sum(x["price"] for x in items), 2)
            oid = self._order(u, items, "delivered", [(card, total)], win_ago)
            spec["order_id"] = oid
            spec["request"] = [{
                "op": "exchange",
                "item_ids": [x["item_id"] for x in items],
                "new_item_ids": [pr["new"]["item_id"] for pr in prs],
                "pay": card,
                "ctx": {"addr_state": self.data["orders"][oid]["address"].get("state"),
                        "pobox": False, "order_multi": True, "partial": False},
                "items_ctx": [{"category": pr["category"],
                               "product_name": pr["name"]} for pr in prs]}]
            spec["pool"] = "exch_multi"
            cl = ", and ".join(
                f"the {pr['name']} (item {items[k]['item_id']}) for the variant "
                f"with {_opts(pr['new']['options'])} (item {pr['new']['item_id']})"
                for k, pr in enumerate(prs))
            spec["fmt"] = {"oid": oid, "clauses": cl, "pay_id": card}

        elif family == "agg_x":
            # whole-order exchange-diff question: 1 audio + 1 non-audio elec +
            # 3 soft (all pair-drawn, upgrade targets stated); the eligible
            # subset — hence the AMOUNT — moves with R1's scope
            n_tot = self._size(i)
            items, prs = self._mixed5(i, n_audio=1, n_elec=1 if n_tot > 3 else 0,
                                      n_soft=n_tot - 1 - (1 if n_tot > 3 else 0))
            soft_diff = sum(round(pr["new"]["price"] - pr["old"]["price"], 2)
                            for pr in prs if pr["category"] != "electronics")
            assert soft_diff > 0, "agg_x: zero soft diff (answer could be 0)"
            total = round(sum(x["price"] for x in items), 2)
            oid = self._order(u, items, "delivered", [(card, total)], win_ago)
            spec["order_id"] = oid
            spec["request"] = [{
                "op": "exchange",
                "item_ids": [x["item_id"] for x in items],
                "new_item_ids": [pr["new"]["item_id"] for pr in prs],
                "pay": card,
                "ctx": {"addr_state": self.data["orders"][oid]["address"].get("state"),
                        "pobox": False, "order_multi": True, "partial": False},
                "items_ctx": [{"category": pr["category"],
                               "product_name": pr["name"]} for pr in prs]}]
            spec["ask"] = {"kind": "diff",
                           "diffs": [round(pr["new"]["price"] - pr["old"]["price"], 2)
                                     for pr in prs]}
            spec["pool"] = "agg_x"
            cl = ", ".join(
                f"the {pr['name']} (item {items[k]['item_id']}) to the variant "
                f"with {_opts(pr['new']['options'])} (item {pr['new']['item_id']})"
                for k, pr in enumerate(prs))
            spec["fmt"] = {"oid": oid, "clauses": cl}

        elif family == "rfz":
            pr = pair(self.audio_groups, off=2)
            it = item_of(pr)
            dest = self.card_of_brand(u, "amex")
            addr = geo("po")
            oid = self._order(u, [it], "delivered", [(dest, it["price"])],
                              win_ago, address=addr)
            spec["order_id"] = oid
            spec["request"] = [
                {"op": "exchange", "item_ids": [it["item_id"]],
                 "new_item_ids": [pr["new"]["item_id"]], "pay": dest,
                 "ctx": {"addr_state": self.data["orders"][oid]["address"].get("state"),
                         "pobox": bool(addr), "order_multi": False,
                         "partial": False},
                 "items_ctx": [{"category": pr["category"],
                                "product_name": pr["name"]}]},
                {"op": "return", "item_ids": [it["item_id"]], "dest": dest,
                 "ctx": {"dest_original_card": True, "card_brand": "amex",
                         "order_multi": False, "partial": False,
                         "order_has_elec": True}},
            ]
            spec["pool"] = "rfz"
            spec["fmt"] = {"oid": oid, "item_name": pr["name"],
                           "item_id": it["item_id"],
                           "new_item_id": pr["new"]["item_id"],
                           "new_opts": _opts(pr["new"]["options"]),
                           "pay_id": dest}

        # multi-host variants
        elif family == "rfz_m":
            n_tot = self._size(i)
            items, prs = self._mixed5(i, n_audio=0, n_elec=0, n_soft=n_tot,
                                      n_clear=0)
            g = self.gift(u)
            total = round(sum(x["price"] for x in items), 2)
            a = round(total * 0.6, 2)
            pays = [(card, a), (g, round(total - a, 2))]
            oid = self._order(u, items, "pending", pays, None)
            spec["order_id"] = oid
            k0 = next(k for k, pr in enumerate(prs) if pr is not None)
            pr = prs[k0]
            spec["request"] = [{
                "op": "modify", "item_ids": [items[k0]["item_id"]],
                "new_item_ids": [pr["new"]["item_id"]], "pay": card,
                "ctx": {"pay_kind": "split"}}]
            spec["pool"] = "rfz_m2"
            spec["fmt"] = {"oid": oid, "item_name": pr["name"],
                           "item_id": items[k0]["item_id"],
                           "new_item_id": pr["new"]["item_id"],
                           "new_opts": _opts(pr["new"]["options"]),
                           "pay_id": card}

        elif family == "rfz_x":
            n_tot = self._size(i)
            items, prs = self._mixed5(i, n_audio=0, n_elec=0, n_soft=n_tot,
                                      n_clear=0)
            addr = geo("po")
            total = round(sum(x["price"] for x in items), 2)
            oid = self._order(u, items, "delivered", [(card, total)], win_ago,
                              address=addr)
            spec["order_id"] = oid
            cand = [k for k, pr in enumerate(prs) if pr is not None]
            spec["request"] = [{
                "op": "exchange",
                "item_ids": [items[k]["item_id"] for k in cand],
                "new_item_ids": [prs[k]["new"]["item_id"] for k in cand],
                "pay": card,
                "ctx": {"addr_state": self.data["orders"][oid]["address"].get("state"),
                        "pobox": True, "order_multi": True, "partial": False},
                "items_ctx": [{"category": prs[k]["category"],
                               "product_name": prs[k]["name"]} for k in cand]}]
            spec["pool"] = "rfz_x2"
            cl = ", and ".join(
                f"the {prs[k]['name']} (item {items[k]['item_id']}) for the "
                f"variant with {_opts(prs[k]['new']['options'])} "
                f"(item {prs[k]['new']['item_id']})" for k in cand)
            spec["fmt"] = {"oid": oid, "clauses": cl, "pay_id": card}

        # ---------- whole-order return to the original card (Z3) --------------
        elif family in ("ret3", "ret3_vx", "ret3_ax", "ret3_q"):
            # MULTI-item delivered order; WHOLE-ORDER return (R6-immune) with
            # the refund insisted onto the ORIGINAL card. Permissive axis:
            # ret3 (W2-3, truth ON) = execute-form adapt on any brand;
            # ret3_vx (W6-7, on_visa) = visa hosts, execute-form adapt;
            # ret3_ax = amex hosts — the docs-consistent refusal CONTRAST in
            # the same windows/pool (docs forbid AND truth forbids non-visa).
            n_tot = self._size(i)
            # ret3_vx/ret3_q sit in the on_visa windows where Z4 is OFF: the
            # host carries ONE clearance item, so truth = destination OPEN but
            # clearance DROPPED (modified execute / reduced amount) — the
            # Z3xZ4 interaction that a blind-execute bot cannot replicate
            n_cl = 1 if family in ("ret3_vx", "ret3_q") else 0
            items, prs = self._mixed5(i, n_audio=0, n_elec=0,
                                      n_soft=n_tot - n_cl, n_clear=n_cl)
            dest = (self.card_of_brand(u, "amex") if family == "ret3_ax"
                    else self.card_of_brand(u, "visa")
                    if family in ("ret3_vx", "ret3_q")
                    else self.card_of_brand(u, "amex" if i % 2 else "non_amex"))
            brand = u["payment_methods"][dest]["brand"]
            total = round(sum(x["price"] for x in items), 2)
            oid = self._order(u, items, "delivered", [(dest, total)], win_ago)
            spec["order_id"] = oid
            spec["request"] = [{
                "op": "return", "item_ids": [x["item_id"] for x in items],
                "dest": dest,
                "ctx": {"dest_original_card": True, "card_brand": brand,
                        "order_multi": True, "partial": False,
                        "order_has_elec": False},
                "items_ctx": [{"clearance": bool(
                    self.data["products"][x["product_id"]]["variants"]
                    [x["item_id"]].get("clearance"))} for x in items]}]
            cl = ", ".join(f"the {x['name']} (item {x['item_id']})"
                           for x in items)
            if family == "ret3_q":
                # VALUE form: "how much back if refunded to my original card?"
                # docs say 0 (destination not offered); truth prices it in the
                # open windows — nonzero window-dependent amounts (agg pool
                # gains adapt members: pool spans slices)
                spec["ask"] = {"prices": [x["price"] for x in items],
                               "fee": self._fee(u)}
                spec["pool"] = "agg"
                spec["fmt"] = {"oid": oid, "clauses": cl,
                               "dest": f"my original credit card ({dest}) — "
                                       f"the card I paid with"}
            else:
                spec["pool"] = "ret3_i"
                spec["fmt"] = {"oid": oid, "clauses": cl, "pay_id": dest}

        elif family in ("exg_fl", "exg_tx"):
            # MULTI-item WHOLE-ORDER exchange (R7-immune) on an FL (W3-4) /
            # TX (W5-6) host; soft products keep R1's scope out of the way.
            n_tot = self._size(i)
            items, prs = self._mixed5(i, n_audio=0, n_elec=0, n_soft=n_tot,
                                      n_clear=0)
            addr = geo("fl" if family == "exg_fl" else "tx")
            total = round(sum(x["price"] for x in items), 2)
            oid = self._order(u, items, "delivered", [(card, total)], win_ago,
                              address=addr)
            spec["order_id"] = oid
            cand = [k for k, pr in enumerate(prs) if pr is not None]
            spec["request"] = [{
                "op": "exchange",
                "item_ids": [items[k]["item_id"] for k in cand],
                "new_item_ids": [prs[k]["new"]["item_id"] for k in cand],
                "pay": card,
                "ctx": {"addr_state": self.data["orders"][oid]["address"].get("state"),
                        "pobox": False, "order_multi": True, "partial": False},
                "items_ctx": [{"category": prs[k]["category"],
                               "product_name": prs[k]["name"]} for k in cand]}]
            spec["pool"] = "exg_i"
            cl = ", and ".join(
                f"the {prs[k]['name']} (item {items[k]['item_id']}) for the "
                f"variant with {_opts(prs[k]['new']['options'])} "
                f"(item {prs[k]['new']['item_id']})" for k in cand)
            spec["fmt"] = {"oid": oid, "clauses": cl, "pay_id": card}

        # ---------- L3 families -------------------------------------------------
        elif family in ("cxl_g", "cxl_s"):
            # PENDING order, payment form = the currently blocked side
            pr = pair(self.cheap_soft_groups, off=13)
            it = item_of(pr)
            if family == "cxl_g":
                g = self.gift(u)
                pays, kind = [(g, it["price"])], "gift_only"
            else:
                g = self.gift(u)
                a = round(it["price"] * 0.6, 2)
                pays, kind = [(card, a), (g, round(it["price"] - a, 2))], "split"
            oid = self._order(u, [it], "pending", pays, None)
            spec["order_id"] = oid
            spec["request"] = [{"op": "cancel", "ctx": {"pay_kind": kind}}]
            spec["pool"] = "cxl"
            spec["fmt"] = {"oid": oid}

        elif family in ("clr", "clr_v"):
            # delivered multi-item soft order with 1-2 CLEARANCE items;
            # WHOLE-ORDER return to gift card. items_ctx carries clearance
            # flags: docs drop clearance (final sale), truth keeps them in
            # permissive windows — different item sets / amounts per state.
            n_tot = self._size(i)
            n_cl = 1 + (i % 2)
            items, prs = self._mixed5(i, n_audio=0, n_elec=0, n_soft=n_tot,
                                      n_clear=n_cl)
            total = round(sum(x["price"] for x in items), 2)
            oid = self._order(u, items, "delivered", [(card, total)], win_ago)
            gid = self.gift(u)
            spec["order_id"] = oid
            def _is_clear(it_):
                pv = self.data["products"][it_["product_id"]]["variants"]
                return bool(pv[it_["item_id"]].get("clearance"))
            spec["request"] = [{
                "op": "return", "item_ids": [x["item_id"] for x in items],
                "dest": gid,
                "ctx": {"dest_original_card": False, "card_brand": None,
                        "order_multi": True, "partial": False,
                        "order_has_elec": False},
                "items_ctx": [{"clearance": bool(
                    self.data["products"][x["product_id"]]["variants"]
                    [x["item_id"]].get("clearance"))} for x in items]}]
            if family == "clr_v":
                spec["ask"] = {"prices": [x["price"] for x in items],
                               "fee": self._fee(u)}
            spec["pool"] = "clr_v" if family == "clr_v" else "clr"
            cl = ", ".join(f"the {x['name']} (item {x['item_id']})"
                           for x in items)
            spec["fmt"] = {"oid": oid, "clauses": cl, "gift_id": gid}

        elif family in ("p_clear_val", "p_dest_val"):
            # serving-only VALUE probes: "how much would I get back?" —
            # the amount tracks the drift (open window: price x (1-fee);
            # closed: 0). Never appears in the test suite.
            if family == "p_clear_val":
                items, prs = self._mixed5(i, n_audio=0, n_elec=0, n_soft=1,
                                          n_clear=1)
                # _mixed5 appends the clearance item last — it is the subject
                it = items[-1]
                dest = self.gift(u)
                ctx = {"dest_original_card": False, "card_brand": None,
                       "order_multi": False, "partial": False,
                       "order_has_elec": False}
                dphrase = f"my gift card ({dest})"
                items_ctx = [{"clearance": True}]
            else:
                it = item_of(pair(self.soft_groups, off=11))
                dest = self.card_of_brand(u, "visa")
                brand = u["payment_methods"][dest]["brand"]
                ctx = {"dest_original_card": True, "card_brand": brand,
                       "order_multi": False, "partial": False,
                       "order_has_elec": False}
                dphrase = (f"my original credit card ({dest}) — the card I "
                           f"paid with")
                items_ctx = None
            paysrc = dest if family == "p_dest_val" else card
            oid = self._order(u, [it], "delivered", [(paysrc, it["price"])],
                              win_ago)
            spec.update({"order_id": oid, "pool": "val_q", "tier": "L1"})
            leg = {"op": "return", "item_ids": [it["item_id"]], "dest": dest,
                   "ctx": ctx}
            if items_ctx:
                leg["items_ctx"] = items_ctx
            spec["request"] = [leg]
            spec["ask"] = {"prices": [it["price"]], "fee": self._fee(u)}
            spec["fmt"] = {"oid": oid, "item_name": it["name"],
                           "item_id": it["item_id"], "dest": dphrase}

        elif family == "p_clear_ret":
            # single CLEARANCE item; return to gift card. Permissive windows:
            # execute (adapt); restrictive windows: docs-correct refusal
            # final_sale (general) — the blind-execute trap.
            items, prs = self._mixed5(i, n_audio=0, n_elec=0, n_soft=1,
                                      n_clear=1)
            it = items[0]
            oid = self._order(u, [it], "delivered", [(card, it["price"])],
                              win_ago)
            gid = self.gift(u)
            spec["order_id"] = oid
            spec.update({"pool": "clear_ret", "tier": "L1"})
            spec["request"] = [{
                "op": "return", "item_ids": [it["item_id"]], "dest": gid,
                "ctx": {"dest_original_card": False, "card_brand": None,
                        "order_multi": False, "partial": False,
                        "order_has_elec": False},
                "items_ctx": [{"clearance": True}]}]
            spec["fmt"] = {"oid": oid, "item_name": it["name"],
                           "item_id": it["item_id"], "gift_id": gid}

        elif family == "rfz_mt":
            # TWIN of rfz_m (same pool rfz_m2): CARD-paid pending multi order
            # -> the modification-only insistence is fine, modify EXECUTES
            n_tot = self._size(i)
            items, prs = self._mixed5(i, n_audio=0, n_elec=0, n_soft=n_tot,
                                      n_clear=0)
            total = round(sum(x["price"] for x in items), 2)
            oid = self._order(u, items, "pending", [(card, total)], None)
            spec["order_id"] = oid
            k0 = next(k for k, pr in enumerate(prs) if pr is not None)
            pr = prs[k0]
            spec["request"] = [{
                "op": "modify", "item_ids": [items[k0]["item_id"]],
                "new_item_ids": [pr["new"]["item_id"]], "pay": card,
                "ctx": {"pay_kind": "card"}}]
            spec["pool"] = "rfz_m2"
            spec["fmt"] = {"oid": oid, "item_name": pr["name"],
                           "item_id": items[k0]["item_id"],
                           "new_item_id": pr["new"]["item_id"],
                           "new_opts": _opts(pr["new"]["options"]),
                           "pay_id": card}

        elif family == "rfz_xt":
            # TWIN of rfz_x (same pool rfz_x2): NORMAL address -> the
            # exchange-only whole-order request EXECUTES
            n_tot = self._size(i)
            items, prs = self._mixed5(i, n_audio=0, n_elec=0, n_soft=n_tot,
                                      n_clear=0)
            total = round(sum(x["price"] for x in items), 2)
            oid = self._order(u, items, "delivered", [(card, total)], win_ago)
            spec["order_id"] = oid
            cand = [k for k, pr in enumerate(prs) if pr is not None]
            spec["request"] = [{
                "op": "exchange",
                "item_ids": [items[k]["item_id"] for k in cand],
                "new_item_ids": [prs[k]["new"]["item_id"] for k in cand],
                "pay": card,
                "ctx": {"addr_state": self.data["orders"][oid]["address"].get("state"),
                        "pobox": False, "order_multi": True, "partial": False},
                "items_ctx": [{"category": prs[k]["category"],
                               "product_name": prs[k]["name"]} for k in cand]}]
            spec["pool"] = "rfz_x2"
            cl = ", and ".join(
                f"the {prs[k]['name']} (item {items[k]['item_id']}) for the "
                f"variant with {_opts(prs[k]['new']['options'])} "
                f"(item {prs[k]['new']['item_id']})" for k in cand)
            spec["fmt"] = {"oid": oid, "clauses": cl, "pay_id": card}

        # ---------- General-only forms ----------------------------------------
        elif family == "gl_limit":
            a = item_of(pair(self.soft_groups, off=9))
            b = item_of(pair(self.soft_groups, off=11))
            assert a["item_id"] != b["item_id"]
            total = round(a["price"] + b["price"], 2)
            oid = self._order(u, [a, b], "delivered", [(card, total)], win_ago)
            o = self.data["orders"][oid]
            amt = round(a["price"] * 0.9, 2)
            o["status"] = "return requested"
            o["return_items"] = [a["item_id"]]
            o["return_reason"] = "no longer needed"
            o["return_refund_amount"] = amt
            o["payment_history"].append({"transaction_type": "refund",
                                         "amount": amt,
                                         "payment_method_id": card})
            gid = self.gift(u)
            spec.update({"order_id": oid, "pool": "return_one"})
            spec["request"] = [{
                "op": "return", "item_ids": [b["item_id"]], "dest": gid,
                "mech_block": "limit", "ctx": {}}]
            spec["fmt"] = {"oid": oid, "item_name": b["name"],
                           "item_id": b["item_id"],
                           "dest": f"my gift card ({gid})"}

        elif family in ("gl_advice", "agg_cnt"):
            if family == "agg_cnt":
                items, prs = self._mixed5(i, n_audio=1, n_elec=1, n_soft=3,
                                          n_clear=1)
            else:
                # composition cycles with i: a fixed basket would give every
                # question the same answer, learnable from the phrasing alone
                n_soft, n_clear = [(2, 1), (3, 1), (4, 1), (4, 0), (3, 2)][i % 5]
                items, prs = self._mixed5(i, n_audio=0, n_elec=0,
                                          n_soft=n_soft, n_clear=n_clear)
            total = round(sum(x["price"] for x in items), 2)
            oid = self._order(u, items, "delivered", [(card, total)], win_ago)
            spec["order_id"] = oid
            good = [k for k, pr in enumerate(prs) if pr is not None]
            clear_bad = [j for j, k in enumerate(range(len(items)))
                         if prs[k] is None]
            spec["request"] = [{
                "op": "exchange",
                "item_ids": [x["item_id"] for x in items],
                "new_item_ids": [prs[k]["new"]["item_id"] if prs[k] else
                                 items[k]["item_id"] for k in range(len(items))],
                "pay": card, "doc_bad_idx": clear_bad,
                "ctx": {"addr_state": self.data["orders"][oid]["address"].get("state"),
                        "pobox": False, "order_multi": True, "partial": False},
                "items_ctx": [{"category": (prs[k]["category"] if prs[k]
                                            else "clearance"),
                               "product_name": items[k]["name"]}
                              for k in range(len(items))]}]
            spec["ask"] = {"kind": "count"}
            spec["pool"] = "advice_cnt"
            spec["fmt"] = {"oid": oid}
        else:
            raise AssertionError(family)
        return spec


def build_world() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    data = copy.deepcopy(_load_base())
    _add_users(data, 180)
    # per-episode order view (split hygiene): every task order generated below
    # is invisible outside its own episode — the engine prunes the DB down to
    # the base world + the episode's own order when this marker is present
    data["_episode_view"] = {"base_orders": sorted(data["orders"].keys())}
    b = _Builder(data)

    counters: Dict[str, int] = {}

    def make(family, split):
        i = counters.get(family, 0)
        counters[family] = i + 1
        spec = b.make(family, i, split=split)
        spec["split"] = split
        return spec

    train = []
    for t, fam in slot_sequence():
        spec = make(fam, "train")
        spec["t"] = t
        train.append(spec)

    test: Dict[int, List[Dict[str, Any]]] = {}
    plan_rng = random.Random(31337)   # decorrelates creation order from family blocks
    for w in range(len(rules.WINDOW_STARTS)):
        test[w] = []
        plan = list(test_plan(w))
        plan_rng.shuffle(plan)
        for fam, sl in plan:
            spec = make(fam, "test")
            spec["slice"] = sl
            test[w].append(spec)

    # ---- clearance POOL SPLIT: inject ~20 SERVING-ONLY clearance
    # variants into the catalog (deterministic ids, injected AFTER the test
    # build so no builder draw shifts), then repoint the builder's clearance
    # source at the NEW variants only. The base clearance ids are
    # test-exclusive: memorizing stream ids earns nothing on the test — only
    # the "look up the clearance flag" behavior generalizes.
    all_ids = {iid for p in data["products"].values() for iid in p["variants"]}
    new_clear = []
    for pid in sorted(data["products"]):
        pr = data["products"][pid]
        base_cl = [v for v in sorted(pr["variants"])
                   if pr["variants"][v].get("clearance")]
        if not base_cl:
            continue
        src = pr["variants"][base_cl[0]]
        digest = hashlib.md5(f"svclear|{pid}".encode()).hexdigest()
        nid = str(int(digest, 16) % 10**10).zfill(10)
        while nid in all_ids:
            digest = hashlib.md5(digest.encode()).hexdigest()
            nid = str(int(digest, 16) % 10**10).zfill(10)
        all_ids.add(nid)
        cents = int(digest[:4], 16) % 90
        pr["variants"][nid] = {"item_id": nid, "available": True,
                               "clearance": True,
                               "price": round(src["price"] * 0.85 + cents / 100, 2),
                               "options": dict(src["options"])}
        new_clear.append({"product_id": pid, "name": pr["name"],
                          "variant": pr["variants"][nid]})
    b.clear = new_clear   # every LATER pass draws clearance hosts from here
    # test episodes must not even SEE the serving-only variants: record them
    # in the view marker so the engine prunes them from test-split episodes —
    # a test episode's tool observations are unaffected by the injection
    data["_episode_view"]["serving_only_variants"] = sorted(
        e["variant"]["item_id"] for e in new_clear)

    # ---- pass 3 (appended after every prior pass): serving value probes.
    # Built after the test orders so the shared id-rng prefix — and therefore
    # every test order id and instruction — is unaffected. t values interleave
    # into the timeline; the train list is then t-sorted.
    extra = []
    # value probes are WINDOW-TARGETED, not uniform: a uniform spread would
    # make most answers 0 (closed windows dominate the timeline) while the
    # test's value questions are nonzero — a reverse signal. Most probes land
    # in open windows (nonzero amounts); a floor of closed-window zeros keeps
    # the "closed => 0" mapping learnable.
    def _win_ticks(w, n, salt):
        t0, t1 = rules.window_range(w)
        return [t0 + 3 + ((salt + 7 * j) % (t1 - t0 - 5)) for j in range(n)]
    val_plan = []
    # p_clear_val: Z4 open W2,3,8,9 -> 5 each (nonzero); closed windows
    # W0,1,4,5,6,7 -> 1 zero each
    for w in (2, 3, 8, 9):
        val_plan += [("p_clear_val", t) for t in _win_ticks(w, 5, 11)]
    for w in (0, 1, 4, 5, 6, 7):
        val_plan += [("p_clear_val", t) for t in _win_ticks(w, 1, 29)]
    # p_dest_val: visa hosts; Z3 open W2,3 (all brands) + W6,7 (visa) -> 5
    # each (nonzero); off_all windows W0,1,4,5,8,9 -> 1 zero each
    for w in (2, 3, 6, 7):
        val_plan += [("p_dest_val", t) for t in _win_ticks(w, 5, 17)]
    for w in (0, 1, 4, 5, 8, 9):
        val_plan += [("p_dest_val", t) for t in _win_ticks(w, 1, 41)]
    for fam, t in val_plan:
        spec = make(fam, "train")
        spec["t"] = t
        extra.append(spec)

    # ---- pass 4 (appended): COMPOSITE serving — the stream's form mix
    # matches the test's (multi-item / whole-order / chains / aggregates)
    # instead of being only single-item probes. Counters continue past the
    # test builds, so the id-rng prefix and the test orders are unaffected.
    COMPOSITES = [("p9_up", 40), ("mxa", 90), ("ret3", 85), ("gl_mx", 70), ("clr", 60),
                  ("exg_fl", 35), ("exg_tx", 35), ("gl_chn", 35), ("gl_agg", 50),
                  ("cxl_g", 28), ("cxl_s", 27)]
    for fam, n in COMPOSITES:
        for k in range(n):
            spec = make(fam, "train")
            salt = int(hashlib.md5(fam.encode()).hexdigest(), 16) % 89
            spec["t"] = 1 + ((k * 900) // n + salt) % 900
            extra.append(spec)
    # p_clear_ret rebuilt on the serving-only clearance pool (12, spread)
    for k in range(12):
        spec = make("p_clear_ret", "train")
        spec["t"] = 75 * k + 40
        extra.append(spec)

    # W9 top-up: Z2 re-opens for its FINAL span at t810; these keep its
    # flip evidence at the 8-episode rule-span minimum
    for k in range(4):
        spec = make("cxl_g" if k % 2 == 0 else "cxl_s", "train")
        spec["t"] = 815 + 20 * k
        extra.append(spec)

    # pass 3b: non-audio-electronics exchange probes — WITHOUT these, Z1's
    # off_audio and off_full states produce IDENTICAL serving evidence
    # (audio refused, soft allowed) while test GT flips on the distinction:
    # the stream must carry the discriminating observations (camera exchange
    # OK in off_audio windows, refused in off_full)
    for k in range(40):
        spec = make("p_cam_exch", "train")
        spec["t"] = 22 * k + 8
        extra.append(spec)

    # ---- thinning (LAST assembly step, single merge): keep the total
    # roughly constant by serving only every EIGHTH pass-1 single-item probe.
    # All pass-1 specs are still BUILT (the rng stream and the test orders
    # are unchanged); unserved orders are invisible under the episode view.
    kept, seen = [], {}
    for sp in train:
        i = seen.get(sp["family"], 0)
        seen[sp["family"]] = i + 1
        # pass-1 p_clear_ret instances host test-exclusive clearance ids —
        # dropped here; serving p_clear_ret probes are built on the serving-only
        # pool above
        if i % 8 == 0 and sp["family"] != "p_clear_ret":
            kept.append(sp)
    train = sorted(kept + extra, key=lambda sp: (sp["t"], sp["family"]))

    return data, {"train": train, "test": test}
