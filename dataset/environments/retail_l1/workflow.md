# Retail Workflow — Operating Guide

## Cancellation
How to cancel an order:
1. Authenticate: `find_user_id_by_name_zip(first_name, last_name, zip)`.
2. Fetch the order with `get_order_details(order_id)` — order ids start with **'#W'**.
   If not found, re-check the id format; the user's order list is in `get_user_details`.
3. Check the rules in policy section `cancellation` against the order's status and total.
4. Call `cancel_pending_order(order_id, reason)`. The refund is issued automatically.
5. Submit `finish {}` — or `finish {"decision": "refuse", "reason": <code>}` if the rules forbid it.

## Order modifications
How to modify items of a pending order:
1. Authenticate, then `get_order_details(order_id)` — order ids start with **'#W'**;
   if not found, re-check the id format and the user's order list in `get_user_details`.
2. `get_product_details(product_id)` to list variants; match the requested options to
   the exact `item_id`; check it is `available` and not `clearance`.
3. Check policy section `modifications`; compute the price difference explicitly.
4. Call `modify_pending_order_items(order_id, item_ids, new_item_ids, payment_method_id)`.
5. Submit `finish {}` — or refuse with the appropriate reason code.

## Returns
How to return items from an order:
1. Authenticate: `find_user_id_by_name_zip`.
2. `get_order_details(order_id)` — order ids start with **'#W'**; if not found,
   re-check the id format and the user's order list in `get_user_details`. Verify the
   order status and that every item to return belongs to the order (match by `item_id`).
3. Check policy section `returns` and gather what its rules depend on: the
   customer's **membership tier** (`get_user_details`), each item's **category**
   (`get_product_details`), the order's **delivered_at** date vs today, and
   whether any item is **clearance**.
4. Call `return_delivered_order_items(order_id, item_ids, payment_method_id, reason)`.
   The fee and refund are computed automatically.
5. Submit `finish {}` — or refuse with the appropriate reason code.

## Exchanges
How to exchange items for other variants of the same product:
1. Authenticate, then `get_order_details(order_id)` — order ids start with **'#W'**;
   if not found, re-check the id format and the user's order list in `get_user_details`.
2. `get_product_details(product_id)` to list variants; pick the target `item_id`
   and check it is `available`.
3. Compute the price difference explicitly from retrieved prices — do not estimate.
   Check policy section `exchanges` for who pays / where refunds go; check gift
   card balances in `get_user_details` when relevant.
4. Call `exchange_delivered_order_items(order_id, item_ids, new_item_ids, payment_method_id)`.
5. Submit `finish {}` — or refuse with the appropriate reason code.

## Placing orders
How to place a new order:
1. Authenticate: `find_user_id_by_name_zip`.
2. Resolve the exact `item_id`: from the catalog via `get_product_details`, or from
   a past order via `get_order_details` (order ids start with **'#W'**; if not found,
   re-check the id format and the user's order list in `get_user_details`).
3. Compute the total explicitly. If paying by gift card, compare the card balance
   (`get_user_details`) with the total and check policy section `placing-orders`
   for the shortfall rules (consented fallback vs refuse).
4. Call `place_order(user_id, item_ids, payment_method_id[, fallback_payment_method_id])`.
5. Submit `finish {}` — or refuse with the appropriate reason code.

## Refusal reasons
When a request must be refused, cite the reason code matching the ROOT cause:
- **service_not_offered** — the requested OPERATION is not available at this store
  for the request as made.
- **invalid_status** — the operation exists, but the order's current status or
  composition does not allow it (e.g., cancelling a processed order).
- **destination_not_allowed** — the operation and order are fine, but a requested
  DESTINATION — a money destination (refund target or payment instrument) or a
  shipping destination (delivery address/region) — is not permitted
  (e.g., requesting a payout to a payment method that does not belong to the
  account, or having no permissible destination at all).
- **insufficient_balance** — the chosen payment instrument is permitted but cannot
  cover the amount, and the customer declines any alternative.
- **over_threshold** — an amount exceeds a policy threshold for the operation
  (thresholds may depend on membership tier).
- **window_expired** — the request is outside the category's return/exchange
  window counted from the delivery date.
- **final_sale** — the item is a clearance item and can never be returned or
  exchanged.
- **not_found** — a referenced order or item does not exist or does not match the
  account. Use it ONLY after verifying: re-check the id format (order ids start
  with '#W') AND check the user's order list via `get_user_details` for a
  plausible match.

Pick the MOST SPECIFIC code that applies: prefer a parameter-level cause (status,
destination, balance, threshold) over `service_not_offered`, which applies only when
the whole operation is unavailable.

## Inquiries
How to answer questions about accounts, orders, or products:
1. Authenticate first if the question is account-specific.
2. Retrieve the records (`get_user_details` / `get_order_details` /
   `get_product_details`) and compute the answer explicitly from them — totals,
   differences, fees and refund amounts must come from retrieved numbers and the
   policy rules, not estimates.
3. Submit `finish {"answer": <value>}` with exactly the value asked for.