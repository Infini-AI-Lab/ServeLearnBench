## decisions
Every pending case — card transaction, limit-increase request or outbound transfer — receives exactly one final decision: approve, deny, or escalate to manual review. Decisions are final and cannot be revised. Deny and escalate must cite a reason code.

## transactions
Card transactions are checked against the category instant-approve limits: electronics $500.00. Categories without a listed limit have no amount restriction. A transaction above its limit is escalated with reason over_limit. Groceries, utilities, dining, fitness, electronics, travel, jewelry and gift cards are all serviceable; gift cards are available to all account tiers, and no category is restricted by merchant region.

## limits
A limit-increase request from an account with a recent chargeback is denied with reason account_standing, whatever the amount. Otherwise the request is approved when the requested limit is within the approvable cap of $5000.00; above the cap, deny with reason cap_exceeded.

## transfers
Outbound transfers are screened against the restricted-payee roster (maintained by the compliance desk; entries are not published) and against the reporting threshold of $8000.00. A transfer to a restricted payee is denied with reason restricted_payee; a transfer above the threshold is escalated with reason reportable_amount.

## accounts
Accounts carry a verification level (basic or verified), a standing flag, tenure in years, and a current credit limit.