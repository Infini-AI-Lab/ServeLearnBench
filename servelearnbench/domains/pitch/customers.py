"""Customer taxonomy — the hidden preference the agent must infer.

Each customer is a *taste*, expressed two ways:
  - `persona`: a natural-language description fed to the judge model so it scores
    a pitch AS this customer. The agent never sees it, except in the oracle
    setting.
  - `valence`: a structured (dimension, value) -> score map. The rubric
    verifier uses it to weight each sheet attribute (liked / disliked /
    red-line); it also generates the oracle bulletin and slices tasks.

Valence scale (mentioning a value in the pitch is worth this to this customer):
  +2 strong-love, +1 like, 0 don't-care, -1 dislike, -3 red-line (visceral).

Litmus test (dimension level): every DIMENSION discriminates across customers.
NOT every value sign-flips: five values are globally one-sided (price=mid,
features=minimal, warranty=long_warranty are nowhere penalized; eco=regular,
warranty=final_sale are nowhere rewarded) — the warranty dim in particular is
a universal good/bad axis, not a flipped one.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .attributes import ATTRIBUTES, phrase

STRONG_POS, POS, NEU, NEG, REDLINE = 2, 1, 0, -1, -3

# key -> profile. `valence` lists only non-zero (dim, value) pairs; anything
# unlisted is 0 (don't-care).
CUSTOMERS: Dict[str, dict] = {
    "risk_averse": {
        "abbrev": "稳",
        "name": "the risk-averse buyer",
        "persona": (
            "You are a cautious, risk-averse shopper. You trust things that are "
            "proven and established, and you are wary of anything new or unusual. "
            "You warm to products that many people already rely on, that have "
            "been around for years, and that look understated and dependable. "
            "Flashy, brand-new, or 'trending' pitches make you nervous rather "
            "than excited. You prefer a sensible mid-range price. A long "
            "warranty with easy returns is decisive for you, and 'sold as-is, "
            "final sale' is disqualifying. You prefer factory-standard products "
            "that are in stock everywhere over handmade or limited runs."
        ),
        "valence": {
            ("launch", "classic"): POS, ("launch", "just_launched"): NEG,
            ("popularity", "many_users"): POS, ("popularity", "niche"): NEG,
            ("price", "mid"): POS,
            ("design", "classic"): POS, ("design", "unique"): NEG,
            ("features", "full"): POS,
            ("buzz", "viral"): NEG,
            ("warranty", "long_warranty"): STRONG_POS, ("warranty", "final_sale"): REDLINE,
            ("origin", "mass_produced"): POS, ("origin", "handcrafted"): NEG,
            ("availability", "everywhere"): POS, ("availability", "limited"): NEG,
        },
    },
    "identity": {
        "abbrev": "潮",
        "name": "the identity-expressive buyer",
        "persona": (
            "You are a style-conscious buyer who uses what you own to express who "
            "you are. You are drawn to the distinctive, the exclusive, the "
            "under-the-radar find that not everyone has. Bold design and being an "
            "early adopter appeal to you. Being told 'everyone already owns this' "
            "is a turn-off — you do not want to blend in. A cheap or budget framing "
            "actively cheapens the appeal; you associate premium with identity. "
            "Small-batch, handcrafted origins and limited runs strongly appeal "
            "to you; being factory-made at scale kills the appeal entirely."
        ),
        "valence": {
            ("launch", "just_launched"): POS, ("launch", "classic"): NEG,
            ("popularity", "niche"): STRONG_POS, ("popularity", "many_users"): NEG,
            ("price", "high"): POS, ("price", "low"): REDLINE,
            ("design", "unique"): STRONG_POS, ("design", "classic"): NEG,
            ("eco", "eco"): POS,
            ("features", "minimal"): POS,
            ("buzz", "viral"): NEG,
            ("origin", "handcrafted"): STRONG_POS, ("origin", "mass_produced"): REDLINE,
            ("availability", "limited"): POS, ("availability", "everywhere"): NEG,
        },
    },
    "anti_marketing": {
        "abbrev": "醒",
        "name": "the anti-marketing rationalist",
        "persona": (
            "You are a skeptical, rational buyer who distrusts marketing. You "
            "value substance, honesty, and restraint. Hard-sell cues repel you: "
            "'everyone is buying it' and 'trending / viral' read as manipulation "
            "and are strong turn-offs. You are suspicious of eco claims unless "
            "concretely substantiated. You like a sensible price, a pared-down "
            "product that does not pay for gimmicks, and a pitch that just states "
            "facts without hype. You value practical assurances like a solid "
            "warranty, and practical availability; artificial scarcity ('limited "
            "run, hard to find') reads as a manufactured marketing trick."
        ),
        "valence": {
            ("launch", "just_launched"): NEG, ("launch", "classic"): POS,
            ("popularity", "many_users"): REDLINE,
            ("price", "mid"): POS, ("price", "low"): POS, ("price", "high"): NEG,
            ("design", "unique"): NEG,
            ("eco", "eco"): NEG,
            ("features", "minimal"): STRONG_POS, ("features", "full"): NEG,
            ("buzz", "viral"): REDLINE, ("buzz", "under_the_radar"): POS,
            ("warranty", "long_warranty"): POS, ("warranty", "final_sale"): NEG,
            ("availability", "everywhere"): POS, ("availability", "limited"): NEG,
        },
    },
    "eco": {
        "abbrev": "绿",
        "name": "the eco-conscious buyer",
        "persona": (
            "You care first and foremost about sustainability and environmental "
            "impact. Genuinely eco / sustainable materials delight you; standard "
            "materials disappoint. You also lean toward pared-down products that "
            "waste less, and you find loud, trending, 'viral' framing shallow. "
            "Beyond the environmental angle you are fairly easygoing. Small-batch "
            "production appeals to you as lower-impact."
        ),
        "valence": {
            ("eco", "eco"): STRONG_POS, ("eco", "regular"): NEG,
            ("features", "minimal"): POS,
            ("buzz", "viral"): NEG,
            ("origin", "handcrafted"): POS, ("origin", "mass_produced"): NEG,
        },
    },
    "conformist": {
        "abbrev": "群",
        "name": "the go-with-the-crowd buyer",
        "persona": (
            "You feel safest buying what everyone else is buying. Popularity and "
            "buzz reassure you: a large user base and a trending, much-talked-"
            "about product are big pluses. Niche or under-the-radar picks feel "
            "risky. You like mainstream, full-featured products that clearly do a "
            "lot. You want what everyone can get: 'in stock everywhere' is "
            "reassuring, while a 'limited run, hard to find' item defeats the "
            "point of buying it."
        ),
        "valence": {
            ("popularity", "many_users"): STRONG_POS, ("popularity", "niche"): NEG,
            ("buzz", "viral"): STRONG_POS, ("buzz", "under_the_radar"): NEG,
            ("features", "full"): POS,
            ("design", "unique"): NEG,
            ("launch", "classic"): POS,
            ("availability", "everywhere"): STRONG_POS, ("availability", "limited"): REDLINE,
            ("warranty", "long_warranty"): POS, ("warranty", "final_sale"): NEG,
            ("origin", "mass_produced"): POS, ("origin", "handcrafted"): NEG,
        },
    },
    "value": {
        "abbrev": "省",
        "name": "the value-for-money buyer",
        "persona": (
            "You want the most for your money. A low or mid price and getting a "
            "lot of features for it are what win you over. You resent paying a "
            "premium for things you see as fluff — eco materials, distinctive "
            "design, or hype/trendiness all read as 'paying extra for nothing.' "
            "You prefer mature, established products over untested new ones. A "
            "warranty protects your money and matters to you; handmade or "
            "limited-edition framing usually just means paying more."
        ),
        "valence": {
            ("price", "low"): STRONG_POS, ("price", "mid"): POS, ("price", "high"): NEG,
            ("eco", "eco"): NEG,
            ("design", "unique"): NEG, ("design", "classic"): POS,
            ("features", "full"): POS,
            ("buzz", "viral"): NEG,
            ("launch", "classic"): POS,
            ("warranty", "long_warranty"): POS, ("warranty", "final_sale"): NEG,
            ("origin", "mass_produced"): POS, ("origin", "handcrafted"): NEG,
            ("availability", "everywhere"): POS, ("availability", "limited"): NEG,
        },
    },
}

CUSTOMER_KEYS: List[str] = list(CUSTOMERS)


def valence(customer_key: str, dim: str, value: str) -> int:
    """Signed value of mentioning (dim, value) to this customer; 0 if unlisted."""
    return CUSTOMERS[customer_key]["valence"].get((dim, value), NEU)


def feature_score(customer_key: str, mentioned: List[Tuple[str, str]]) -> int:
    """Ground-truth score of a pitch that emphasizes exactly `mentioned` pairs.

    Sum of per-pair valence; an analysis helper, not the agent's reward.
    """
    return sum(valence(customer_key, d, v) for d, v in mentioned)


def positives(customer_key: str, product: Dict[str, str]) -> List[Tuple[str, str]]:
    """(dim, value) pairs of this product that this customer likes (>0)."""
    return [(d, v) for d, v in product.items() if valence(customer_key, d, v) > 0]


def redlines(customer_key: str, product: Dict[str, str]) -> List[Tuple[str, str]]:
    """(dim, value) pairs of this product that are red-lines for this customer."""
    return [(d, v) for d, v in product.items()
            if valence(customer_key, d, v) <= NEG]


def straddles(customer_key: str, product: Dict[str, str]) -> bool:
    """True if the product gives this customer BOTH something to praise AND
    something to stay silent about (a >0 pair and a <0 pair). These are the
    interesting 'what to say / what not to say' tasks (adapt slice)."""
    return bool(positives(customer_key, product)) and bool(redlines(customer_key, product))


def oracle_bulletin(customer_key: str, dims=None) -> str:
    """Natural-language 'what this customer wants' — used by the oracle setting.
    dims: restrict to the tier's dim set (L1 sheets omit some dims; the
    bulletin must not reference attributes the sheet cannot carry)."""
    c = CUSTOMERS[customer_key]
    likes, avoids = [], []
    for (d, v), s in c["valence"].items():
        if dims is not None and d not in dims:
            continue
        if s > 0:
            likes.append(f"{phrase(d, v)}")
        elif s <= NEG:
            tag = " (do NOT bring this up)" if s == REDLINE else ""
            avoids.append(f"{phrase(d, v)}{tag}")
    # The persona paragraph is free prose written against the FULL dimension
    # set, so on a restricted tier it names attributes the sheet can never
    # carry (L1 omits launch/design/warranty/origin/availability). The
    # structured lists above are filtered; the prose cannot be, and since
    # inventing an unlisted attribute is penalised, pointing the oracle at
    # those dimensions would cost it points. Drop the prose when the tier is
    # restricted — the lists carry the same information in the only form the
    # sheet can support.
    from .attributes import ATTRIBUTES
    restricted = dims is not None and set(dims) != set(ATTRIBUTES)
    persona = "" if restricted else f" {c['persona']}"
    return (f"The customer is {c['name']}.{persona}\n"
            f"They like: {'; '.join(likes)}.\n"
            f"They dislike: {'; '.join(avoids)}.\n"
            "These lists describe the customer's TASTE in general. Use ONLY "
            "facts that actually appear on this product's sheet: praise the "
            "sheet facts the customer likes, and do not bring up the ones "
            "they dislike AT ALL. Two things count as false and are scored "
            "against you: stating the opposite of a sheet fact, and claiming "
            "anything the sheet does not state at all. A liked trait that is "
            "not on this sheet is not available to you. If few likable facts "
            "apply, a shorter pitch is better than touching a disliked topic.")
