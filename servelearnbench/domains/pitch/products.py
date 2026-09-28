"""Deterministic product generator for the pitch benchmark.

A product = one value per attribute dimension + a plausible name. Generation is
hash-deterministic (reproducible, byte-freezable) and de-duplicated on the
attribute tuple so the pool spreads across the attribute space with little
repetition. Attributes are generic enough to apply to any of the product
categories below, so the same schema describes every product.

What the agent is shown = the attribute sheet (render_sheet). It must then
decide which attributes to foreground / suppress for the hidden customer.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List

from .attributes import ATTRIBUTES, FULL_DIMS, phrase

# Product categories the schema fits. The category is flavor only (the hidden
# preference is over attributes, not category), but it makes the sheet feel real.
CATEGORIES = [
    "water bottle", "backpack", "wireless earbuds", "desk lamp", "sneakers",
    "coffee maker", "notebook", "office chair", "yoga mat", "phone case",
    "travel mug", "bluetooth speaker", "cutting board", "duffel bag",
    "fountain pen", "table fan", "raincoat", "cookware set", "sunglasses",
    "standing desk",
]
# Two-part brandable name descriptors (+ series token when a two-part name
# would collide — names must be GLOBALLY UNIQUE within a pool, so that a
# learner's note about a product name always refers to one sheet).
BRAND_A = ["Nord", "Alto", "Vela", "Kió", "Muse", "Terra", "Loom", "Fenn",
           "Orin", "Wisp", "Halo", "Cove", "Peak", "Drift", "Ember"]
BRAND_B = ["One", "Go", "Pro", "Lite", "Edit", "Studio", "Field", "Daily",
           "Core", "Works", "Co.", "Labs"]
BRAND_C = ["S", "X", "2", "Max", "Mini", "Air", "Neo", "Plus", "Prime",
           "Ultra", "Nova", "Zen"]


def _h(*parts) -> int:
    return int(hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest(), 16)


def product_dims(product: Dict[str, str]) -> List[str]:
    """The attribute dims this product carries, in canonical order."""
    return [d for d in ATTRIBUTES if d in product]


def _make_product(idx: int, seed: str, dims: List[str]) -> Dict[str, str]:
    """Deterministic product at index idx: a value per dim in dims + id + name."""
    attrs = {dim: ATTRIBUTES[dim][_h(seed, idx, dim) % len(ATTRIBUTES[dim])]
             for dim in dims}
    cat = CATEGORIES[_h(seed, idx, "cat") % len(CATEGORIES)]
    name = (f"{BRAND_A[_h(seed, idx, 'a') % len(BRAND_A)]} "
            f"{BRAND_B[_h(seed, idx, 'b') % len(BRAND_B)]}")
    return {"product_id": f"prod_{idx:04d}", "name": name, "category": cat, **attrs}


def generate_pool(n: int = 60, seed: str = "pitch_v0",
                  dims: List[str] | None = None) -> List[Dict[str, str]]:
    """A de-duplicated pool of n products. Dedup key = attribute tuple +
    category (category included so small dim sets — 5 dims has only 48 attr
    combos — can still fill a pool)."""
    dims = dims or FULL_DIMS
    pool, seen, names, idx = [], set(), set(), 0
    while len(pool) < n and idx < n * 40:
        p = _make_product(idx, seed, dims)
        key = tuple(p[d] for d in dims) + (p["category"],)
        if key not in seen:
            # global name uniqueness: the 15x12 two-part space (180) is
            # smaller than a 240 pool — on collision append a deterministic
            # series token (then bump it) until the name is unique.
            if p["name"] in names:
                base, k = p["name"], _h(seed, idx, "c")
                for step in range(len(BRAND_C) * 3):
                    cand = f"{base} {BRAND_C[(k + step) % len(BRAND_C)]}"
                    if step >= len(BRAND_C):  # 2nd lap: double token
                        cand = f"{base} {BRAND_C[(k + step) % len(BRAND_C)]}{step // len(BRAND_C)}"
                    if cand not in names:
                        p["name"] = cand
                        break
            if p["name"] in names:
                idx += 1
                continue  # give up on this idx rather than ship a duplicate
            seen.add(key)
            names.add(p["name"])
            pool.append(p)
        idx += 1
    return pool


def render_sheet(product: Dict[str, str]) -> str:
    """The attribute sheet shown to the agent (neutral facts, no spin)."""
    lines = [f"Product: {product['name']} ({product['category']})", "Facts:"]
    for dim in product_dims(product):
        lines.append(f"  - {phrase(dim, product[dim])}")
    return "\n".join(lines)


def attr_pairs(product: Dict[str, str]):
    """The (dimension, value) pairs of a product (for valence lookups)."""
    return [(dim, product[dim]) for dim in product_dims(product)]


def kitchen_sink(product: Dict[str, str]) -> str:
    """The FROZEN crowd-pleaser baseline pitch for relative judging: it
    enthusiastically names every fact (including red-lines), so a well-targeted
    pitch beats it for a specific customer while a mis-targeted one does not.
    Deterministic per product."""
    facts = ", ".join(phrase(dim, product[dim]) for dim in product_dims(product))
    return (f"Introducing the {product['name']} — you'll love it! It's {facts}. "
            f"Don't miss out on this amazing product!")
