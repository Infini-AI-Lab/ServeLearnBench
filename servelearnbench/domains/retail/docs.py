"""Sectioned reference documents, versioned per stage (REWRITE drift).

Content is defined here directly (single source, aligned with policy_config).
The agent querying read_docs sees only the currently-effective clean text --
no markers, no history. Version history is kept purely for the website.

Sections are stable across stages (same TOC): what changes is the BODY -- e.g.
`returns` says "not offered" at S0 and carries the full program rules from S3.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# {doc: {slug: (title, [(stage, body), ...])}}  -- later stages replace earlier.
_SECTIONS: Dict[str, Dict[str, Any]] = {
    "policy": {
        "cancellation": ("Cancellation", [
            (0,
             "- Only a **pending** order may be cancelled. Processed, delivered, or already\n"
             "  cancelled orders may not be cancelled.\n"
             "- The cancellation reason must be one of: **\"no longer needed\"**, **\"ordered by mistake\"**.\n"
             "- Cancellation is free of charge; the payment is refunded to the payment method(s) used."),
            (4,
             "- Only a **pending** order may be cancelled. Processed, delivered, or already\n"
             "  cancelled orders may not be cancelled.\n"
             "- **Cancellation threshold by membership tier** — orders with a total above the\n"
             "  tier's limit may not be cancelled: bronze **$200**, silver **$300**, gold **$500**.\n"
             "- The cancellation reason must be one of: **\"no longer needed\"**, **\"ordered by mistake\"**.\n"
             "- Cancellation is free of charge; the payment is refunded to the payment method(s) used."),
        ]),
        "returns": ("Returns", [
            (0,
             "- **Returns are not offered** at this store — this service does not exist.\n"
             "- Politely refuse any return request."),
            (3,
             "- Only a **delivered** order may have items returned; items must belong to the order.\n"
             "- A return must include a **reason**, one of: **defective**, **no longer needed**, **wrong item**.\n"
             "- **Return window by product category** (days after delivery): apparel **30**,\n"
             "  electronics **14**, home **30**. Outside the window — refuse.\n"
             "- **Clearance items are final sale** and can never be returned — refuse.\n"
             "- **Restocking fee by membership tier** (withheld from the refund):\n"
             "  bronze **15%**, silver **10%**, gold **5%**.\n"
             "- **Refund destination by tier**: bronze and silver — **gift card only**;\n"
             "  **gold** — gift card **or the original payment method**. A bronze/silver\n"
             "  customer without a gift card cannot receive a return refund — refuse."),
        ]),
        "exchanges": ("Exchanges", [
            (0,
             "- **Exchanges are not offered.**\n"
             "- Politely refuse any exchange request."),
            (4,
             "- Only a **delivered** order may have items exchanged, and only for **available,\n"
             "  non-clearance** variants of the **same product**.\n"
             "- The **return window by category also applies to exchanges** (apparel 30 /\n"
             "  electronics 14 / home 30 days after delivery); **clearance items are final sale**.\n"
             "- If the new variant is **more expensive**, the difference is paid via the\n"
             "  **original payment method or a gift card**; a gift card must have sufficient\n"
             "  balance — refuse if it cannot cover it and the customer declines other methods.\n"
             "- If the new variant is **cheaper**, the refund destination follows the tier rule:\n"
             "  bronze/silver — **gift card only**; **gold** — gift card or the original method."),
        ]),
        "modifications": ("Order modifications", [
            (0,
             "- Only a **pending** order may have items modified, and only to **available,\n"
             "  non-clearance** variants of the **same product**.\n"
             "- The price difference is settled via the **original payment method or a gift\n"
             "  card** (a gift card must have sufficient balance for an upcharge)."),
        ]),
        "placing-orders": ("Placing orders", [
            (0,
             "- New orders are paid by **credit card** on file.\n"
             "- The full amount is charged at order time."),
            (1,
             "- New orders may be paid with any payment method on file, including **gift cards**.\n"
             "- If paid by gift card, the **balance must cover the total**. If it does not, the\n"
             "  remainder may be charged to a **non-gift-card** method only with the customer's\n"
             "  consent; if the customer insists on gift card only, **refuse**."),
        ]),
    },
    "workflow": {
        "cancellation": ("Cancellation", [
            (0,
             "How to cancel an order:\n"
             "1. Authenticate: `find_user_id_by_name_zip(first_name, last_name, zip)`.\n"
             "2. Fetch the order with `get_order_details(order_id)` — order ids start with **'#W'**.\n"
             "   If not found, re-check the id format; the user's order list is in `get_user_details`.\n"
             "3. Check the rules in policy section `cancellation` against the order's status and total.\n"
             "4. Call `cancel_pending_order(order_id, reason)`. The refund is issued automatically.\n"
             "5. Submit `finish {}` — or `finish {\"decision\": \"refuse\", \"reason\": <code>}` if the rules forbid it."),
        ]),
        "modifications": ("Order modifications", [
            (0,
             "How to modify items of a pending order:\n"
             "1. Authenticate, then `get_order_details(order_id)` — order ids start with **'#W'**;\n"
             "   if not found, re-check the id format and the user's order list in `get_user_details`.\n"
             "2. `get_product_details(product_id)` to list variants; match the requested options to\n"
             "   the exact `item_id`; check it is `available` and not `clearance`.\n"
             "3. Check policy section `modifications`; compute the price difference explicitly.\n"
             "4. Call `modify_pending_order_items(order_id, item_ids, new_item_ids, payment_method_id)`.\n"
             "5. Submit `finish {}` — or refuse with the appropriate reason code."),
        ]),
        "returns": ("Returns", [
            (0,
             "Returns are not offered. If a customer asks to return items, check policy section\n"
             "`returns` and refuse accordingly."),
            (3,
             "How to return items from an order:\n"
             "1. Authenticate: `find_user_id_by_name_zip`.\n"
             "2. `get_order_details(order_id)` — order ids start with **'#W'**; if not found,\n"
             "   re-check the id format and the user's order list in `get_user_details`. Verify the\n"
             "   order status and that every item to return belongs to the order (match by `item_id`).\n"
             "3. Check policy section `returns` and gather what its rules depend on: the\n"
             "   customer's **membership tier** (`get_user_details`), each item's **category**\n"
             "   (`get_product_details`), the order's **delivered_at** date vs today, and\n"
             "   whether any item is **clearance**. Determine the allowed refund destination.\n"
             "4. Call `return_delivered_order_items(order_id, item_ids, payment_method_id, reason)`.\n"
             "   The fee and refund are computed automatically.\n"
             "5. Submit `finish {}` — or refuse with the appropriate reason code."),
        ]),
        "exchanges": ("Exchanges", [
            (0,
             "Exchanges are not offered. If a customer asks to exchange items, check policy\n"
             "section `exchanges` and refuse accordingly."),
            (4,
             "How to exchange items for other variants of the same product:\n"
             "1. Authenticate, then `get_order_details(order_id)` — order ids start with **'#W'**;\n"
             "   if not found, re-check the id format and the user's order list in `get_user_details`.\n"
             "2. `get_product_details(product_id)` to list variants; pick the target `item_id`\n"
             "   and check it is `available`.\n"
             "3. Compute the price difference explicitly from retrieved prices — do not estimate.\n"
             "   Check policy section `exchanges` for who pays / where refunds go; check gift\n"
             "   card balances in `get_user_details` when relevant.\n"
             "4. Call `exchange_delivered_order_items(order_id, item_ids, new_item_ids, payment_method_id)`.\n"
             "5. Submit `finish {}` — or refuse with the appropriate reason code."),
        ]),
        "placing-orders": ("Placing orders", [
            (0,
             "How to place a new order:\n"
             "1. Authenticate: `find_user_id_by_name_zip`.\n"
             "2. Resolve the exact `item_id`: from the catalog via `get_product_details`, or from\n"
             "   a past order via `get_order_details` (order ids start with **'#W'**; if not found,\n"
             "   re-check the id format and the user's order list in `get_user_details`).\n"
             "3. Check policy section `placing-orders` for payment rules.\n"
             "4. Call `place_order(user_id, item_ids, payment_method_id)`.\n"
             "5. Submit `finish {}` — or refuse with the appropriate reason code."),
            (1,
             "How to place a new order:\n"
             "1. Authenticate: `find_user_id_by_name_zip`.\n"
             "2. Resolve the exact `item_id`: from the catalog via `get_product_details`, or from\n"
             "   a past order via `get_order_details` (order ids start with **'#W'**; if not found,\n"
             "   re-check the id format and the user's order list in `get_user_details`).\n"
             "3. Compute the total explicitly. If paying by gift card, compare the card balance\n"
             "   (`get_user_details`) with the total and check policy section `placing-orders`\n"
             "   for the shortfall rules (consented fallback vs refuse).\n"
             "4. Call `place_order(user_id, item_ids, payment_method_id[, fallback_payment_method_id])`.\n"
             "5. Submit `finish {}` — or refuse with the appropriate reason code."),
        ]),
        "refusals": ("Refusal reasons", [
            (0,
             "When a request must be refused, cite the reason code matching the ROOT cause:\n"
             "- **service_not_offered** — the requested OPERATION does not exist at this store at all\n"
             "  (e.g., asking to return items when returns are not offered).\n"
             "- **invalid_status** — the operation exists, but the order's current status does not\n"
             "  allow it (e.g., cancelling a processed order; returning a pending order).\n"
             "- **destination_not_allowed** — the operation and order are fine, but the requested\n"
             "  MONEY DESTINATION (refund target or payment instrument) is not permitted by policy\n"
             "  (e.g., demanding a refund to a credit card when policy refunds to gift cards only;\n"
             "  or having no permissible destination at all).\n"
             "- **insufficient_balance** — the chosen payment instrument is permitted but cannot\n"
             "  cover the amount, and the customer declines any alternative.\n"
             "- **over_threshold** — an amount exceeds a policy threshold for the operation\n"
             "  (thresholds may depend on membership tier).\n"
             "- **window_expired** — the request is outside the category's return/exchange\n"
             "  window counted from the delivery date.\n"
             "- **final_sale** — the item is a clearance item and can never be returned or\n"
             "  exchanged.\n"
             "- **not_found** — a referenced order or item does not exist or does not match the\n"
             "  account. Use it ONLY after verifying: re-check the id format (order ids start\n"
             "  with '#W') AND check the user's order list via `get_user_details` for a\n"
             "  plausible match.\n"
             "\n"
             "Pick the MOST SPECIFIC code that applies: prefer a parameter-level cause (status,\n"
             "destination, balance, threshold) over `service_not_offered`, which applies only when\n"
             "the whole operation is unavailable."),
        ]),
        "inquiries": ("Inquiries", [
            (0,
             "How to answer questions about accounts, orders, or products:\n"
             "1. Authenticate first if the question is account-specific.\n"
             "2. Retrieve the records (`get_user_details` / `get_order_details` /\n"
             "   `get_product_details`) and compute the answer explicitly from them — totals,\n"
             "   differences, fees and refund amounts must come from retrieved numbers and the\n"
             "   policy rules, not estimates.\n"
             "3. Submit `finish {\"answer\": <value>}` with exactly the value asked for."),
        ]),
    },
}


def get_sections(doc: str, stage: int = 0) -> List[Dict[str, Any]]:
    """Sections at a stage. `body` = currently effective text; `versions` = full
    history up to the stage (website only)."""
    assert doc in _SECTIONS
    out = []
    for slug, (title, versions) in _SECTIONS[doc].items():
        vs = [{"stage": s, "body": b} for s, b in versions if s <= stage]
        if not vs:
            continue
        out.append({"slug": slug, "title": title, "body": vs[-1]["body"], "versions": vs})
    return out


def section_text(sec: Dict[str, Any]) -> str:
    """The section exactly as the agent sees it: current text, no history."""
    return f"## {sec['title']}\n{sec['body']}"


def toc(doc: str, stage: int = 0) -> str:
    secs = get_sections(doc, stage)
    lines = [f"{doc} document — sections (fetch one with read_docs(doc='{doc}', section='<slug>')):"]
    for s in secs:
        lines.append(f"- {s['slug']} — {s['title']}")
    return "\n".join(lines)


def lookup(doc: str, stage: int, section: Optional[str]) -> str:
    if doc not in _SECTIONS:
        return "Error: unknown document (available: policy, workflow)"
    if section is None:
        return toc(doc, stage)
    for s in get_sections(doc, stage):
        if s["slug"] == section:
            return section_text(s)
    slugs = ", ".join(s["slug"] for s in get_sections(doc, stage))
    return f"Error: unknown section '{section}' (available: {slugs})"


def system_prompt_toc(stage: int = 0) -> str:
    lines = []
    for doc in ("policy", "workflow"):
        slugs = ", ".join(s["slug"] for s in get_sections(doc, stage))
        lines.append(f"- {doc}: {slugs}")
    return "\n".join(lines)


def merged_md(doc: str, stage: int = 0) -> str:
    title = {"policy": "# Retail Policy — Rules", "workflow": "# Retail Workflow — Operating Guide"}[doc]
    return title + "\n\n" + "\n\n".join(section_text(s) for s in get_sections(doc, stage))
