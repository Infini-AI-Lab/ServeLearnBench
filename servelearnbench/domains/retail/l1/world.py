"""L1 world: per-window independent order pools, one order per question.

720 training orders (one per serving episode) + 384 test orders (one per test
question; adapt per ADAPT_TOTALS = 20/40/40/40/40/60 across W0-W5, plus 24
general per window) + the 50-order base world. No order is shared between
windows, or between training and test. ~60 extra users are added
deterministically (globally unique full names) to spread ownership.
"""

from __future__ import annotations

import copy
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
        # information holdout for D5 (same discipline as the FL address
        # pools): serving and test draw DISJOINT clearance names, so "this
        # name was returnable" recall cannot answer a test ticket — only the
        # generalization "clearance items are returnable" can.
        _clr = sorted(self.clear, key=lambda e: e["name"])
        self.clear_test = _clr[::3]           # every third name -> test
        self.clear_train = [e for e in _clr if e not in self.clear_test]
        self.big = sorted(_plain_items(p), key=lambda e: -e["variant"]["price"])
        # pool-sufficiency check: every task family must have generous supply
        assert len({e["name"] for e in self.elec_pairs}) >= 10, "elec pair names thin"
        assert len({e["name"] for e in self.soft_pairs}) >= 20, "soft pair names thin"
        assert len(self.cheap) >= 40, "cheap pool thin"
        assert len(self.mid) >= 20, "mid pool thin"
        assert len(self.clear) >= 8, "clearance pool thin"
        assert len(self.clear_train) >= 12, "d5 serving clearance pool thin"
        assert len(self.clear_test) >= 6, "d5 test clearance pool thin"
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

    def _order(self, u, items, status, payments, ago, address=None):
        while True:
            oid = "#W" + "".join(str(self.idrng.randint(0, 9)) for _ in range(7))
            if oid not in self.data["orders"]:
                break
        self.seq += 1
        addr = dict(address or u["address"])
        if address is None and addr.get("state") == "FL":
            addr = dict(SAFE_ADDRESSES[self.seq % len(SAFE_ADDRESSES)])
        self.data["orders"][oid] = {
            "order_id": oid, "user_id": u["user_id"], "address": addr, "items": items,
            "status": status, "placed_at": _date_offset((ago or 1) + 4),
            **({"delivered_at": _date_offset(ago)} if status == "delivered" else {}),
            "payment_history": [{"transaction_type": "payment", "amount": round(a, 2),
                                 "payment_method_id": pid} for pid, a in payments]}
        u["orders"].append(oid)
        return oid

    def make(self, family: str, i: int, split: str = "train") -> Dict[str, Any]:
        """One dedicated order + spec for one question of `family`."""
        u = self.user()
        card = self.card(u)
        win_ago = [4, 6, 8, 10][i % 4]
        item_of = lambda e: {"name": e["name"], "product_id": e["product_id"],
                             "item_id": (e.get("variant") or e["old"])["item_id"],
                             "price": (e.get("variant") or e["old"])["price"],
                             "options": (e.get("variant") or e["old"])["options"]}
        spec = {"family": family, "user_id": u["user_id"],
                "first": u["name"]["first_name"], "last": u["name"]["last_name"],
                "zip": u["address"]["zip"]}

        if family in ("d1_affected", "d1_control"):
            it = item_of(self.cheap[(i * 7 + len(family)) % len(self.cheap)])
            pay = self.gift(u) if family == "d1_affected" else card
            # incidental FL draws honor the train/test address split too
            _flp = FL_ADDRESSES_TEST if split == "test" else FL_ADDRESSES_TRAIN
            fl = _flp[i % len(_flp)] if i % 3 == 0 else None
            spec["order_id"] = self._order(u, [it], "pending", [(pay, it["price"])],
                                           None, address=fl)
            spec["pay_id"] = pay
        elif family == "f_thr":
            items = [item_of(self.big[0]), item_of(self.big[1]), item_of(self.big[2 + i % 3])]
            total = round(sum(x["price"] for x in items), 2)
            assert total > 520.0
            spec["order_id"] = self._order(u, items, "pending", [(card, total)], None)
        elif family in ("d2_affected", "d23_control", "d3_affected"):
            groups = self.elec_groups if family == "d2_affected" else self.soft_groups
            g_ = groups[(i + len(family)) % len(groups)]
            pr = g_[(i // len(groups)) % len(g_)]
            it = item_of(pr)
            fl_pool = FL_ADDRESSES_TEST if split == "test" else FL_ADDRESSES_TRAIN
            addr = fl_pool[i % len(fl_pool)] if family == "d3_affected" else None
            spec["order_id"] = self._order(u, [it], "delivered", [(card, it["price"])],
                                           win_ago, address=addr)
            spec.update({"item_id": pr["old"]["item_id"], "item_name": pr["name"],
                         "new_item_id": pr["new"]["item_id"],
                         "new_opts": _opts(pr["new"]["options"]), "pay_id": card})
        elif family in ("d4_affected", "d4_control", "f_win", "f_clr",
                        "d5_affected", "d6_affected"):
            # d5: CLEARANCE item, delivered 4-10 days ago (inside every
            #     category window, so window rules can't co-cause), refund to
            #     gift card (so D4 can't co-cause) -> only D5 decides.
            # d6: soft item (30-day window), non-clearance, delivered 35-55
            #     days ago (outside the window), gift-card destination
            #     -> only D6 decides.
            if family in ("f_clr", "d5_affected"):
                pool = self.clear_test if split == "test" else self.clear_train
                entry = pool[i % len(pool)]
            else:
                entry = self.soft[(i * 13 + len(family)) % len(self.soft)]
            it = item_of(entry)
            ago = ([35, 45, 55][i % 3] if family in ("f_win", "d6_affected")
                   else win_ago)
            _flp = FL_ADDRESSES_TEST if split == "test" else FL_ADDRESSES_TRAIN
            fl = (_flp[i % len(_flp)]
                  if (i % 3 == 1 and family in ("d4_affected", "d4_control", "f_win"))
                  else None)
            g = self.gift(u)
            # some d4_control orders are SPLIT-PAID and still execute —
            # without them every split-paid order in the suite is an h1
            # refusal, so "split payment => refuse" is behaviorally identical
            # to learning H1; the control punishes that overgeneralization.
            pays = ([(card, round(it["price"] * 0.6, 2)),
                     (g, round(it["price"] - round(it["price"] * 0.6, 2), 2))]
                    if family == "d4_control" and i % 3 == 2
                    else [(card, it["price"])])
            spec["order_id"] = self._order(u, [it], "delivered", pays,
                                           ago, address=fl)
            spec.update({"item_id": it["item_id"], "item_name": it["name"],
                         "dest_id": card if family == "d4_affected" else g,
                         "card_id": card, "gift_id": g})
        elif family == "g_status":
            # documented invalid_status: cancelling a PROCESSED order. Paid by
            # card (never gift-only, so hidden D1 can't co-cause in any window)
            # and cheap (under every tier's cancel cap, so no over_threshold).
            it = item_of(self.cheap[(i * 7 + 3) % len(self.cheap)])
            spec["order_id"] = self._order(u, [it], "processed", [(card, it["price"])], None)
        elif family == "g_balance":
            # documented insufficient_balance: exchange upcharge pinned to a
            # gift card whose balance cannot cover the difference. Soft goods
            # (never electronics -> D2 can't co-cause) and non-FL address.
            g_ = self.soft_groups[(i + 3) % len(self.soft_groups)]
            pr = next((x for x in g_ if x["new"]["price"] - x["old"]["price"] >= 25.0), None)
            if pr is None:  # fall back to the widest upcharge in the group
                pr = max(g_, key=lambda x: x["new"]["price"] - x["old"]["price"])
            diff = round(pr["new"]["price"] - pr["old"]["price"], 2)
            assert diff >= 10.0, f"g_balance: upcharge too small ({diff})"
            it = item_of(pr)
            spec["order_id"] = self._order(u, [it], "delivered", [(card, it["price"])],
                                           win_ago)
            low = f"gift_card_{''.join(str(self.idrng.randint(0, 9)) for _ in range(7))}"
            self.gseq += 1
            bal = round(max(1.0, diff * 0.25), 2)
            u["payment_methods"][low] = {"source": "gift_card", "id": low, "balance": bal}
            spec.update({"item_id": pr["old"]["item_id"], "item_name": pr["name"],
                         "new_item_id": pr["new"]["item_id"],
                         "new_opts": _opts(pr["new"]["options"]), "pay_id": low})
        elif family in ("f_thr_mid", "g_thr_mid"):
            u = self.user_of_tier("bronze" if family == "f_thr_mid" else "gold")
            card = self.card(u)
            spec.update({"user_id": u["user_id"], "first": u["name"]["first_name"],
                         "last": u["name"]["last_name"], "zip": u["address"]["zip"]})
            it = item_of(self.mid[(i * 5) % len(self.mid)])
            assert 200.0 < it["price"] < 300.0
            spec["order_id"] = self._order(u, [it], "pending", [(card, it["price"])], None)
        elif family in ("f_win_elec", "g_win_home"):
            pool = self.elec if family == "f_win_elec" else self.soft
            it = item_of(pool[(i * 9 + 3) % len(pool)])
            ago = [18, 21, 25][i % 3]     # inside 30d (home/apparel), outside 14d (electronics)
            spec["order_id"] = self._order(u, [it], "delivered", [(card, it["price"])], ago)
            g = self.gift(u)
            spec.update({"item_id": it["item_id"], "item_name": it["name"],
                         "dest_id": g, "card_id": card, "gift_id": g})
        elif family in ("h1_affected", "h1_control"):
            g_ = self.soft_groups[(i + 5) % len(self.soft_groups)]
            pr = g_[(i // len(self.soft_groups)) % len(g_)]
            it = item_of(pr)
            if family == "h1_affected":
                g = self.gift(u)
                a = round(it["price"] * 0.6, 2)
                pays = [(card, a), (g, round(it["price"] - a, 2))]
            else:
                pays = [(card, it["price"])]
            spec["order_id"] = self._order(u, [it], "pending", pays, None)
            spec.update({"item_id": pr["old"]["item_id"], "item_name": pr["name"],
                         "new_item_id": pr["new"]["item_id"],
                         "new_opts": _opts(pr["new"]["options"]), "pay_id": card})
        else:
            raise AssertionError(family)
        return spec


def build_world() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    data = copy.deepcopy(_load_base())
    _add_users(data, 180)
    # split hygiene: each episode sees the base world plus its own order only
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

    return data, {"train": train, "test": test}
