"""Deterministic retail data generator.

Fixed seed -> reproducible {users, products, orders} with strong referential
integrity. Code-based construction (seed lists composed by code), following the
tau-bench data-gen philosophy: schema is human-decided, values are code-built.

Guarantees (checked by validate() in load.py):
  - every order references an existing user and real item variants
  - user.orders is the exact reverse index of the orders table
  - each order's payment total equals the sum of its item prices
  - a mixed status distribution with enough `pending` and `delivered` orders so
    that cancel (pending-only) and return (delivered-only) tasks are feasible
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List

FOLDER = os.path.dirname(__file__)
DATA_DIR = os.path.join(FOLDER, "data")

TODAY = "2026-07-15"  # keep in sync with policy_config.TODAY


def _date_offset(days_ago: int) -> str:
    from datetime import date, timedelta
    return (date.fromisoformat(TODAY) - timedelta(days=days_ago)).isoformat()

# ---- seed vocab (fixed lists) ----------------------------------------------
FIRST_NAMES = [
    "noah", "yusuf", "mei", "liam", "olivia", "raj", "sofia", "james",
    "amara", "chen", "lucas", "ava", "omar", "isabella", "kenji", "fatima",
    "diego", "hannah", "sven", "priya", "elena", "marcus", "aisha", "tomas",
    "ingrid", "jamal", "lucia", "viktor", "naomi", "andre", "zara", "felix",
    "carmen", "dmitri", "leila", "oscar", "freya", "mateo", "anika", "ravi",
]
LAST_NAMES = [
    "brown", "rossi", "lin", "smith", "patel", "garcia", "kim", "li",
    "okafor", "wang", "muller", "davis", "hassan", "nguyen", "silva", "cohen",
    "andersen", "moreau", "tanaka", "novak", "reyes", "haddad", "eriksson",
    "costa", "yamamoto", "kowalski", "adeyemi", "petrov", "santos", "fischer",
]
CITIES = [
    ("Denver", "CO", "80"), ("Austin", "TX", "78"), ("New York", "NY", "10"),
    ("Seattle", "WA", "98"), ("Chicago", "IL", "60"), ("Boston", "MA", "02"),
    ("Phoenix", "AZ", "85"), ("Miami", "FL", "33"),
    ("Portland", "OR", "97"), ("Atlanta", "GA", "30"), ("Nashville", "TN", "37"),
    ("Minneapolis", "MN", "55"), ("Salt Lake City", "UT", "84"), ("Columbus", "OH", "43"),
    ("Raleigh", "NC", "27"), ("Albuquerque", "NM", "87"), ("Madison", "WI", "53"),
    ("Baltimore", "MD", "21"), ("Omaha", "NE", "68"), ("Pittsburgh", "PA", "15"),
]
STREETS = ["Sunset Drive", "River Road", "Oak Avenue", "Maple Street", "Pine Lane",
           "Elm Court", "Cedar Boulevard", "Willow Way", "Birch Street", "Aspen Circle",
           "Juniper Lane", "Magnolia Drive", "Sycamore Road", "Chestnut Avenue",
           "Poplar Street", "Laurel Court"]

# ---- product catalog --------------------------------------------------------
# 48 uniquely-named products, 16 per category. Each entry:
#   (name, category, tier, {option_axis: [choices]})
# tier drives the price band; bands are engineered against the policy tables:
#   budget  : $25-140   (cheap pool <$150 for D1/cancel tasks under the bronze cap)
#   core    : $95-200
#   mid     : $212-288  (f_thr_mid / g_thr_mid need 200 < price < 300)
#   premium : $310-495  (big pool: any three premium items total > $520)
_AX_COLOR6 = ["black", "white", "blue", "red", "green", "grey"]
_AX_SIZE = ["S", "M", "L", "XL"]
CATALOG = [
    # -- electronics (16) --
    ("Mechanical Keyboard", "electronics", "mid",
     {"switch": ["linear", "clicky", "tactile"], "backlight": ["RGB", "white", "none"], "size": ["full", "tenkeyless", "compact"]}),
    ("Wireless Mouse", "electronics", "budget",
     {"color": ["black", "white", "grey"], "dpi": ["8000", "16000"], "grip": ["standard", "ergonomic"]}),
    ("Noise-Cancelling Headphones", "electronics", "premium",
     {"color": ["black", "silver", "navy"], "type": ["over-ear", "on-ear"], "wireless": ["yes", "no"]}),
    ("Bluetooth Speaker", "electronics", "core",
     {"color": ["black", "blue", "red"], "waterproof": ["yes", "no"], "size": ["mini", "standard"]}),
    ("Smart Thermostat", "electronics", "mid",
     {"compatibility": ["Google Home", "Apple HomeKit", "Amazon Alexa"], "color": ["white", "black"], "display": ["touch", "dial"]}),
    ("Fitness Tracker", "electronics", "core",
     {"color": ["black", "teal", "coral"], "band": ["silicone", "fabric"], "gps": ["yes", "no"]}),
    ("Webcam", "electronics", "budget",
     {"resolution": ["1080p", "4K"], "fov": ["78deg", "90deg"], "color": ["black", "silver"]}),
    ("USB-C Hub", "electronics", "budget",
     {"ports": ["6-in-1", "8-in-1", "11-in-1"], "color": ["silver", "grey"], "cable": ["short", "long"]}),
    ("Portable Charger", "electronics", "budget",
     {"capacity": ["10000mAh", "20000mAh", "26800mAh"], "color": ["black", "white"], "ports": ["single", "dual"]}),
    ("E-Reader", "electronics", "mid",
     {"storage": ["8GB", "16GB", "32GB"], "screen": ["6in", "7in"], "color": ["black", "white"]}),
    ("Action Camera", "electronics", "premium",
     {"resolution": ["4K", "5K"], "bundle": ["camera only", "with accessories"], "color": ["black", "silver"]}),
    ("Wireless Earbuds", "electronics", "core",
     {"color": ["black", "white", "green"], "anc": ["yes", "no"], "case": ["wired", "wireless charging"]}),
    ("Gaming Controller", "electronics", "core",
     {"color": ["black", "white", "red"], "connectivity": ["wireless", "wired"], "edition": ["standard", "pro"]}),
    ("WiFi Router", "electronics", "premium",
     {"speed": ["AX3000", "AX5400", "AX6600"], "bands": ["dual-band", "tri-band"], "color": ["black", "white"]}),
    ("Digital Photo Frame", "electronics", "core",
     {"size": ["8in", "10in"], "color": ["black", "wood"], "storage": ["16GB", "32GB"]}),
    ("Smart Doorbell", "electronics", "premium",
     {"power": ["battery", "wired"], "resolution": ["1080p", "2K"], "color": ["black", "bronze"]}),
    # -- apparel (16) --
    ("T-Shirt", "apparel", "budget",
     {"color": ["blue", "red", "black", "purple"], "size": _AX_SIZE, "material": ["cotton", "polyester"]}),
    ("Running Shoes", "apparel", "core",
     {"color": ["white", "black", "green", "volt"], "size": ["8", "9", "10", "11"]}),
    ("Backpack", "apparel", "core",
     {"color": ["grey", "navy", "black"], "size": ["small", "medium", "large"], "material": ["nylon", "canvas"]}),
    ("Rain Jacket", "apparel", "core",
     {"color": ["yellow", "navy", "black"], "size": _AX_SIZE, "hood": ["fixed", "detachable"]}),
    ("Denim Jeans", "apparel", "core",
     {"color": ["indigo", "black", "stonewash"], "size": ["28", "30", "32", "34", "36"], "fit": ["slim", "regular"]}),
    ("Wool Sweater", "apparel", "mid",
     {"color": ["charcoal", "cream", "forest"], "size": _AX_SIZE, "knit": ["cable", "ribbed"]}),
    ("Baseball Cap", "apparel", "budget",
     {"color": _AX_COLOR6, "closure": ["snapback", "strapback"]}),
    ("Hiking Boots", "apparel", "premium",
     {"color": ["brown", "black", "grey"], "size": ["8", "9", "10", "11", "12"], "waterproof": ["yes", "no"]}),
    ("Leather Belt", "apparel", "budget",
     {"color": ["brown", "black"], "size": ["32", "34", "36", "38"], "buckle": ["silver", "brass"]}),
    ("Cotton Hoodie", "apparel", "core",
     {"color": ["heather grey", "black", "navy", "maroon"], "size": _AX_SIZE, "zip": ["pullover", "full-zip"]}),
    ("Ankle Socks Pack", "apparel", "budget",
     {"color": ["white", "black", "mixed"], "size": ["M", "L"], "count": ["3-pack", "6-pack"]}),
    ("Winter Gloves", "apparel", "budget",
     {"color": ["black", "grey", "red"], "size": ["S/M", "L/XL"], "touchscreen": ["yes", "no"]}),
    ("Puffer Vest", "apparel", "mid",
     {"color": ["black", "olive", "navy"], "size": _AX_SIZE, "fill": ["down", "synthetic"]}),
    ("Yoga Leggings", "apparel", "core",
     {"color": ["black", "plum", "slate"], "size": _AX_SIZE, "length": ["full", "7/8"]}),
    ("Canvas Sneakers", "apparel", "budget",
     {"color": ["white", "black", "red", "navy"], "size": ["7", "8", "9", "10", "11"]}),
    ("Flannel Shirt", "apparel", "core",
     {"color": ["red plaid", "green plaid", "grey plaid"], "size": _AX_SIZE, "fit": ["slim", "relaxed"]}),
    # -- home (16) --
    ("Water Bottle", "home", "budget",
     {"capacity": ["500ml", "750ml", "1000ml"], "material": ["glass", "steel", "plastic"], "color": ["blue", "black"]}),
    ("Office Chair", "home", "premium",
     {"material": ["fabric", "leather", "mesh"], "color": ["black", "grey"], "armrest": ["fixed", "adjustable"]}),
    ("Coffee Maker", "home", "mid",
     {"type": ["drip", "espresso", "french press"], "color": ["silver", "black"], "capacity": ["4-cup", "12-cup"]}),
    ("Desk Lamp", "home", "budget",
     {"color": ["white", "black", "brass"], "brightness": ["standard", "bright"], "power": ["corded", "rechargeable"]}),
    ("Yoga Mat", "home", "budget",
     {"color": ["purple", "blue", "green"], "thickness": ["4mm", "6mm", "8mm"]}),
    ("Wristwatch", "home", "premium",
     {"color": ["silver", "gold", "black"], "band": ["leather", "metal", "silicone"]}),
    ("Standing Desk", "home", "premium",
     {"width": ["48in", "55in", "60in"], "finish": ["oak", "walnut", "white"]}),
    ("Electric Kettle", "home", "budget",
     {"capacity": ["1.0L", "1.7L"], "color": ["silver", "black", "white"], "control": ["basic", "variable temp"]}),
    ("Throw Blanket", "home", "budget",
     {"color": ["cream", "charcoal", "rust"], "material": ["fleece", "knit"], "size": ["50x60", "60x80"]}),
    ("Cookware Set", "home", "premium",
     {"pieces": ["8-piece", "10-piece", "12-piece"], "material": ["stainless", "nonstick"], "finish": ["silver", "black"]}),
    ("Air Purifier", "home", "mid",
     {"coverage": ["small room", "large room"], "color": ["white", "grey"], "filter": ["HEPA", "HEPA+carbon"]}),
    ("Wall Clock", "home", "budget",
     {"diameter": ["10in", "12in", "14in"], "color": ["black", "white", "walnut"]}),
    ("Storage Ottoman", "home", "core",
     {"color": ["grey", "navy", "beige"], "size": ["small", "large"], "material": ["linen", "faux leather"]}),
    ("Scented Candle Set", "home", "budget",
     {"scent": ["vanilla", "cedar", "citrus"], "count": ["2-pack", "4-pack"], "size": ["small jar", "large jar"]}),
    ("Bookshelf", "home", "core",
     {"shelves": ["3-tier", "5-tier"], "finish": ["oak", "black", "white"], "width": ["24in", "31in"]}),
    ("Weighted Blanket", "home", "mid",
     {"weight": ["12lb", "15lb", "20lb"], "size": ["twin", "queen"], "color": ["grey", "navy"]}),
]

PRICE_BANDS = {"budget": (25.0, 140.0), "core": (95.0, 200.0),
               "mid": (212.0, 288.0), "premium": (310.0, 495.0)}


def _digits(rng: random.Random, n: int) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(n))


def _round2(x: float) -> float:
    return round(x, 2)


def _axis_combos(axes: Dict[str, List[str]]) -> List[Dict[str, str]]:
    """All option combinations, deterministic order."""
    keys = sorted(axes)
    combos: List[Dict[str, str]] = [{}]
    for k in keys:
        combos = [dict(c, **{k: v}) for c in combos for v in axes[k]]
    return combos


def generate(seed: int = 7, n_users: int = 20, n_orders: int = 50) -> Dict[str, Any]:
    rng = random.Random(seed)

    # ---- products ------------------------------------------------------------
    # One entry per CATALOG row (names globally unique). 8-12 variants each with
    # engineered guarantees per product:
    #   - >= 6 available, non-clearance variants (exchange/modify pair supply)
    #   - exactly one clearance variant on ~40% of products (final-sale traps)
    #   - exactly one unavailable variant on ~35% of products
    products: Dict[str, Any] = {}
    item_index: List[Dict[str, Any]] = []
    for pi, (name, category, tier, axes) in enumerate(CATALOG):
        product_id = _digits(rng, 10)
        combos = _axis_combos(axes)
        rng.shuffle(combos)
        n_variants = min(rng.randint(8, 12), len(combos))
        lo, hi = PRICE_BANDS[tier]
        variants: Dict[str, Any] = {}
        flags = ["plain"] * n_variants
        if pi % 5 in (0, 2):      # 40% of products carry one clearance variant
            flags[-1] = "clearance"
        if pi % 3 == 1:           # ~33% carry one unavailable variant
            flags[-2 if len(flags) > 1 else 0] = "unavailable"
        for vi in range(n_variants):
            opts = combos[vi]
            item_id = _digits(rng, 10)
            price = _round2(rng.uniform(lo, hi))
            variants[item_id] = {
                "item_id": item_id,
                "options": opts,
                "available": flags[vi] != "unavailable",
                "clearance": flags[vi] == "clearance",
                "price": price,
            }
            item_index.append({"product_id": product_id, "name": name, "item_id": item_id,
                               "price": price, "options": opts})
        usable = [v for v in variants.values() if v["available"] and not v["clearance"]]
        assert len(usable) >= 6, f"{name}: only {len(usable)} usable variants"
        assert len({v["price"] for v in usable}) >= 4, f"{name}: price ladder too flat"
        products[product_id] = {"name": name, "product_id": product_id,
                                "category": category, "variants": variants}
    assert len({p["name"] for p in products.values()}) == len(CATALOG), "duplicate product name"

    # ---- users ---------------------------------------------------------------
    users: Dict[str, Any] = {}
    user_ids: List[str] = []
    taken_names = set()
    while len(user_ids) < n_users:
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        if (first, last) in taken_names:   # full names globally unique
            continue
        user_id = f"{first}_{last}_{_digits(rng, 4)}"
        if user_id in users:
            continue
        taken_names.add((first, last))
        city, state, zip_prefix = rng.choice(CITIES)
        zip_code = zip_prefix + _digits(rng, 3)
        payment_methods: Dict[str, Any] = {}
        # 1-3 credit cards: extra cards are realistic distractors (the paying
        # card is always pinned explicitly by order construction / instructions)
        n_cards = rng.choices([1, 2, 3], weights=[45, 40, 15])[0]
        for _ in range(n_cards):
            cc_id = f"credit_card_{_digits(rng, 7)}"
            payment_methods[cc_id] = {
                "source": "credit_card", "id": cc_id,
                "brand": rng.choice(["visa", "mastercard", "amex"]), "last_four": _digits(rng, 4),
            }
        # ~75% also hold a gift card (task builders add one on demand otherwise)
        if rng.random() < 0.75:
            gc_id = f"gift_card_{_digits(rng, 7)}"
            payment_methods[gc_id] = {
                "source": "gift_card", "id": gc_id, "balance": _round2(rng.uniform(5, 140)),
            }
        tier = rng.choices(["bronze", "silver", "gold"], weights=[40, 40, 20])[0]
        users[user_id] = {
            "user_id": user_id,
            "membership": tier,
            "name": {"first_name": first.capitalize(), "last_name": last.capitalize()},
            "address": {
                "address1": f"{rng.randint(100, 999)} {rng.choice(STREETS)}",
                "address2": f"Suite {rng.randint(100, 999)}",
                "city": city, "state": state, "country": "USA", "zip": zip_code,
            },
            "email": f"{first}.{last}{_digits(rng, 4)}@example.com",
            "payment_methods": payment_methods,
            "orders": [],
        }
        user_ids.append(user_id)

    # ---- orders --------------------------------------------------------------
    # Force a mixed status distribution with enough pending & delivered.
    statuses = (
        ["pending"] * max(8, n_orders // 4)
        + ["delivered"] * max(8, n_orders // 4)
        + ["processed"] * (n_orders // 4)
        + ["cancelled"] * (n_orders // 8)
    )
    while len(statuses) < n_orders:
        statuses.append("delivered")
    statuses = statuses[:n_orders]
    rng.shuffle(statuses)

    orders: Dict[str, Any] = {}
    for i in range(n_orders):
        order_id = "#W" + _digits(rng, 7)
        if order_id in orders:
            order_id = "#W" + _digits(rng, 7)
        user_id = rng.choice(user_ids)
        user = users[user_id]
        n_items = rng.randint(1, 3)
        items = []
        for _ in range(n_items):
            src = rng.choice(item_index)
            items.append({
                "name": src["name"], "product_id": src["product_id"],
                "item_id": src["item_id"], "price": src["price"], "options": src["options"],
            })
        total = _round2(sum(it["price"] for it in items))
        pay_id = rng.choice(list(user["payment_methods"].keys()))
        # thirds: fresh (inside 14d window) / mid (inside 30d, outside 14d) / stale
        bucket = rng.random()
        if bucket < 0.34:
            delivered_ago = rng.randint(3, 12)
        elif bucket < 0.67:
            delivered_ago = rng.randint(16, 28)
        else:
            delivered_ago = rng.randint(32, 60)
        placed_ago = delivered_ago + rng.randint(2, 6)
        order = {
            "order_id": order_id,
            "user_id": user_id,
            "address": dict(user["address"]),
            "items": items,
            "status": statuses[i],
            "placed_at": _date_offset(placed_ago),
            **({"delivered_at": _date_offset(delivered_ago)} if statuses[i] == "delivered" else {}),
            "payment_history": [
                {"transaction_type": "payment", "amount": total, "payment_method_id": pay_id}
            ],
        }
        orders[order_id] = order
        user["orders"].append(order_id)

    # guarantee: owners of the first few multi-item delivered orders hold a gift
    # card (keeps mixed-eligibility partial-return tasks constructible)
    fixed = 0
    for oid in sorted(orders):
        o = orders[oid]
        if o["status"] != "delivered" or len(o["items"]) < 2:
            continue
        u = users[o["user_id"]]
        if not any(pm["source"] == "gift_card" for pm in u["payment_methods"].values()):
            gc_id = f"gift_card_99000{fixed:02d}"
            u["payment_methods"][gc_id] = {"source": "gift_card", "id": gc_id, "balance": 75.0}
        fixed += 1
        if fixed >= 4:
            break

    return {"users": users, "products": products, "orders": orders}


def write(data: Dict[str, Any], data_dir: str = DATA_DIR) -> None:
    os.makedirs(data_dir, exist_ok=True)
    for table in ("users", "products", "orders"):
        with open(os.path.join(data_dir, f"{table}.json"), "w") as f:
            json.dump(data[table], f, indent=1, sort_keys=True)


if __name__ == "__main__":
    data = generate()
    write(data)
    print(f"generated: {len(data['users'])} users, {len(data['products'])} products, "
          f"{len(data['orders'])} orders -> {DATA_DIR}")
