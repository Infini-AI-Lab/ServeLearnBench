"""Product attribute schema for the pitch benchmark.

Per-item difficulty is set by attribute dimensionality. L1 uses 5 dims (the
full sheet fits the word budget); L2/L3 use all 10 dims (mentioning every fact
does not fit the budget, so selection is forced). Every dimension discriminates
across customers (see customers.py): a value liked by one audience is disliked
by another. The warranty / origin / availability dims discriminate through
their valence rows (origin and availability sign-flip; warranty is one-sided)
— e.g. "final sale" is a red-line for risk_averse but neutral for identity —
so they are preference signals, not hygiene signals.

Each product is one assignment of a value to every dimension in its tier's
dim set. The agent is shown the product's values (the "attribute sheet") and
must choose which to foreground / suppress given the hidden current customer.
"""

from __future__ import annotations

# dimension -> ordered list of possible values (poles)
ATTRIBUTES = {
    "launch":     ["just_launched", "classic"],
    "popularity": ["many_users", "niche"],
    "price":      ["low", "mid", "high"],
    "design":     ["unique", "classic"],
    "eco":        ["eco", "regular"],
    "features":   ["full", "minimal"],
    "buzz":       ["viral", "under_the_radar"],
    # L2/L3 only
    "warranty":     ["long_warranty", "final_sale"],
    "origin":       ["handcrafted", "mass_produced"],
    "availability": ["limited", "everywhere"],
}

# Tier dim sets. L1 drops launch/design (weak or duplicated signals) and
# warranty/origin/availability; L2/L3 use everything.
L1_DIMS = ["popularity", "price", "eco", "features", "buzz"]
FULL_DIMS = list(ATTRIBUTES)

# Human-readable phrasing for every (dimension, value). Used both to render the
# product sheet shown to the agent and to phrase the oracle bulletin. Kept
# neutral/factual — the *spin* is the agent's job, not baked into the sheet.
PHRASING = {
    ("launch", "just_launched"): "just launched this season",
    ("launch", "classic"):       "on the market for years, well-established",
    ("popularity", "many_users"): "already has a large user base",
    # This pole is about OWNERS; the buzz pole is about MENTIONS, so the two
    # dimensions cannot be credited by the same wording.
    ("popularity", "niche"):      "has only a small number of owners",
    ("price", "low"):  "budget-friendly price",
    ("price", "mid"):  "mid-range price",
    ("price", "high"): "premium price",
    ("design", "unique"):  "bold, distinctive design",
    # Anchored to appearance only; features=minimal is anchored to how many
    # functions the product has. The two carry opposite valence for the
    # identity customer, so their phrasings must not read as one idea.
    ("design", "classic"): "a visually traditional, understated exterior",
    ("eco", "eco"):      "made from sustainable / eco materials",
    ("eco", "regular"):  "standard materials",
    ("features", "full"):    "feature-rich, does a lot",
    # About function count only — see the design=classic note above.
    ("features", "minimal"): "provides only the essential functions, with few extras",
    ("buzz", "viral"):           "a trending, much-talked-about item",
    # About mentions only — see the popularity=niche note.
    ("buzz", "under_the_radar"): "rarely mentioned in the press or on social media",
    ("warranty", "long_warranty"): "backed by a long warranty and easy returns",
    ("warranty", "final_sale"):    "sold as-is, final sale",
    # Purely about production method; availability=limited is purely about
    # how hard the product is to obtain.
    ("origin", "handcrafted"):     "made by hand rather than by machine",
    ("origin", "mass_produced"):   "factory-made at scale",
    ("availability", "limited"):    "stocked in very few places and hard to obtain",
    ("availability", "everywhere"): "in stock everywhere",
}


# What each phrase is ABOUT, as a reader would take it. Two phrases from
# DIFFERENT dimensions must not share a concept: if they do, a pitch that
# addresses one is judged to have addressed both, and when the two carry
# different valence for the active customer the reward comes out wrong.
#
# Word-level overlap does not detect this (conflicting phrases can share no
# word at all), so the mapping is declared by hand.
CONCEPTS = {
    ("launch", "just_launched"):   {"age"},
    ("launch", "classic"):         {"age"},
    ("popularity", "many_users"):  {"owner_count"},
    ("popularity", "niche"):       {"owner_count"},
    ("price", "low"):              {"price"},
    ("price", "mid"):              {"price"},
    ("price", "high"):             {"price"},
    ("design", "unique"):          {"appearance"},
    ("design", "classic"):         {"appearance"},
    ("eco", "eco"):                {"materials"},
    ("eco", "regular"):            {"materials"},
    ("features", "full"):          {"function_count"},
    ("features", "minimal"):       {"function_count"},
    ("buzz", "viral"):             {"mentions"},
    ("buzz", "under_the_radar"):   {"mentions"},
    ("warranty", "long_warranty"): {"after_sale"},
    ("warranty", "final_sale"):    {"after_sale"},
    ("origin", "handcrafted"):     {"production_method"},
    ("origin", "mass_produced"):   {"production_method"},
    ("availability", "limited"):    {"obtainability"},
    ("availability", "everywhere"): {"obtainability"},
}


def phrase(dim: str, value: str) -> str:
    """Neutral one-liner for an (attribute, value)."""
    return PHRASING[(dim, value)]


def all_pairs():
    """Every (dimension, value) pair — used by customers.py to define valence."""
    return [(dim, v) for dim, vals in ATTRIBUTES.items() for v in vals]
