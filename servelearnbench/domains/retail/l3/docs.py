"""L3 frozen documents: identical surface to the L2 docs (S4 + overrides).
Two explicit prohibitions are the permissive-drift targets: the FINAL-SALE
clause (Z4) and the GIFT-CARD-ONLY refund-destination clause (Z3) — reality
relaxes them in some windows while the text never changes.

These documents never change over the timeline. Everything they state stays
true forever (F rules); the hidden H/D dimensions are simply never mentioned,
so no sentence ever becomes false ("docs-silent exceptions" constraint).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ....engine.tool import Tool
from .. import docs as docs_mod

_STAGE = 4  # the frozen surface = S4 documents (+ overrides below)

# {(doc, slug): body} — replaces the S4 body wholesale.
_OVERRIDES: Dict[Any, str] = {
    ("policy", "returns"): (
        "- Only a **delivered** order may have items returned; items must belong to the order.\n"
        "- A return must include a **reason**, one of: **defective**, **no longer needed**, **wrong item**.\n"
        "- **Return window by product category** (days after delivery): apparel **30**,\n"
        "  electronics **14**, home **30**. Outside the window — refuse.\n"
        "- **Clearance items are final sale** and can never be returned — refuse.\n"
        "- **Restocking fee by membership tier** (withheld from the refund):\n"
        "  bronze **15%**, silver **10%**, gold **5%**.\n"
        "- The refund destination is a **gift card on the customer's account** — this\n"
        "  is the **only** destination offered; refunds to the **original payment\n"
        "  method(s) are not available**. If the customer insists on an original-card\n"
        "  refund and declines a gift-card destination, **refuse**.\n"
        "- A return request may cover **any subset** of the order's items. Each requested\n"
        "  item is assessed **independently** against the rules above: ineligible items are\n"
        "  excluded and the eligible remainder proceeds as one return. If **no** requested\n"
        "  item is eligible, refuse the request.\n"
        "- **Each order may have at most ONE post-delivery service request** accepted\n"
        "  over its lifetime — one return OR one exchange, not both.\n"
        "- Where a refund is directed to the **original payment** of an order paid with\n"
        "  **multiple payment methods**, it is **split across those methods in\n"
        "  proportion to the amount each paid**. A gift-card destination always receives\n"
        "  the full amount.\n"
        "- **Refund amount** = the sum of the eligible returned items' prices multiplied\n"
        "  by (1 − the membership restocking fee above).\n"
        "- A return or exchange request counts as **whole-order** when it LISTS every item\n"
        "  of the order, and as **partial** otherwise — items excluded by the rules above\n"
        "  do not change this classification.\n"
        "- The refund is issued when the return is registered."),
    ("policy", "exchanges"): (
        "- Only a **delivered** order may have items exchanged, and only for **available,\n"
        "  non-clearance** variants of the **same product**.\n"
        "- The **return window by category also applies to exchanges** (apparel 30 /\n"
        "  electronics 14 / home 30 days after delivery); **clearance items are final sale**.\n"
        "- If the new variant is **more expensive**, the difference must be settled with a\n"
        "  **credit card on file. Gift cards cannot fund exchange upcharges** — refuse such\n"
        "  requests (service_not_offered) unless the customer offers a credit card instead.\n"
        "- If the new variant is **cheaper**, the difference is refunded.\n"
        "- An exchange request may cover **any subset** of the order's items. Each requested\n"
        "  item is assessed **independently** against the rules above: ineligible items are\n"
        "  excluded and the eligible remainder proceeds as one exchange. If **no** requested\n"
        "  item is eligible, refuse the request.\n"
        "- **Each order may have at most ONE post-delivery service request** accepted\n"
        "  over its lifetime — one exchange OR one return, not both."),
    ("workflow", "refusals"): (
        "When a request must be refused, cite the reason code matching the ROOT cause:\n"
        "- **service_not_offered** — the requested OPERATION is not available at this store\n"
        "  for the request as made.\n"
        "- **invalid_status** — the operation exists, but the order's current status or\n"
        "  composition does not allow it (e.g., cancelling a processed order).\n"
        "- **destination_not_allowed** — the operation and order are fine, but the requested\n"
        "  MONEY DESTINATION (refund target or payment instrument) is not permitted\n"
        "  (e.g., requesting a payout to a payment method that does not belong to the\n"
        "  account, or having no permissible destination at all).\n"
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
        "the whole operation is unavailable.\n"
        "\n"
        "Partial fulfilment and customer-stated alternatives:\n"
        "- If only SOME of the requested items are eligible, carry out the eligible part\n"
        "  and close normally with `finish {}` — a refusal is only correct when NO part of\n"
        "  the request can be carried out.\n"
        "- If the customer states an ORDERED list of preferences (\"first ...; if that's\n"
        "  not possible, then ...; failing that, ...\"), carry out the FIRST option that\n"
        "  is permitted, in the customer's stated order. Refuse only when none of the\n"
        "  stated options can be carried out."),
    ("workflow", "returns"): (
        "How to return items from an order:\n"
        "1. Authenticate: `find_user_id_by_name_zip`.\n"
        "2. `get_order_details(order_id)` — order ids start with **'#W'**; if not found,\n"
        "   re-check the id format and the user's order list in `get_user_details`. Verify the\n"
        "   order status and that every item to return belongs to the order (match by `item_id`).\n"
        "3. Check policy section `returns` and gather what its rules depend on: the\n"
        "   customer's **membership tier** (`get_user_details`), each item's **category**\n"
        "   (`get_product_details`), the order's **delivered_at** date vs today, and\n"
        "   whether any item is **clearance**.\n"
        "4. Call `return_delivered_order_items(order_id, item_ids, payment_method_id, reason)`.\n"
        "   The fee and refund are computed automatically.\n"
        "5. Submit `finish {}` — or refuse with the appropriate reason code."),
}


def get_sections(doc: str) -> List[Dict[str, Any]]:
    secs = docs_mod.get_sections(doc, _STAGE)
    out = []
    for s in secs:
        body = _OVERRIDES.get((doc, s["slug"]), s["body"])
        out.append({"slug": s["slug"], "title": s["title"], "body": body})
    return out


def section_text(sec: Dict[str, Any]) -> str:
    return f"## {sec['title']}\n{sec['body']}"


def toc(doc: str) -> str:
    lines = [f"{doc} document — sections (fetch one with read_docs(doc='{doc}', section='<slug>')):"]
    for s in get_sections(doc):
        lines.append(f"- {s['slug']} — {s['title']}")
    return "\n".join(lines)


def lookup(doc: str, section: Optional[str]) -> str:
    if doc not in ("policy", "workflow"):
        return "Error: unknown document (available: policy, workflow)"
    if section is None:
        return toc(doc)
    for s in get_sections(doc):
        if s["slug"] == section:
            return section_text(s)
    slugs = ", ".join(s["slug"] for s in get_sections(doc))
    return f"Error: unknown section '{section}' (available: {slugs})"


def system_prompt_toc() -> str:
    lines = []
    for doc in ("policy", "workflow"):
        slugs = ", ".join(s["slug"] for s in get_sections(doc))
        lines.append(f"- {doc}: {slugs}")
    return "\n".join(lines)


def merged_md(doc: str) -> str:
    title = {"policy": "# Retail Policy — Rules", "workflow": "# Retail Workflow — Operating Guide"}[doc]
    return title + "\n\n" + "\n\n".join(section_text(s) for s in get_sections(doc))


class ReadDocsV2(Tool):
    """Section-level retrieval over the frozen L3 documents."""

    @classmethod
    def invoke(cls, data: Dict[str, Any], doc: str, section: Optional[str] = None) -> str:  # type: ignore[override]
        return lookup(doc, section)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "read_docs",
                "description": (
                    "Read the store's reference documents at SECTION granularity. Call with only `doc` "
                    "to get the table of contents (section slugs); call again with `section` to read that "
                    "section. 'policy' holds the rules (what may / may not be done); 'workflow' holds "
                    "operating procedures and domain reference. Consult the relevant policy section "
                    "before any write action or refusal."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc": {"type": "string", "enum": ["policy", "workflow"],
                                "description": "Which document."},
                        "section": {"type": "string",
                                    "description": "Section slug from the table of contents, e.g. 'cancellation'. Omit to list sections."},
                    },
                    "required": ["doc"],
                },
            },
        }
