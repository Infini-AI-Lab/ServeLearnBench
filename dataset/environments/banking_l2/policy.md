## decisions
Every pending case — card transaction, limit-increase request or outbound transfer — receives exactly one final decision: approve, deny, or escalate to manual review. Decisions are final and cannot be revised. Deny and escalate must cite a reason code.

## case_forms
A case takes one of three forms.
1) A single request. Decide it with approve_case, deny_case or escalate_case.
2) A case listing RANKED ALTERNATIVES under `legs`, most preferred first. The customer will accept any of them. Carry out the FIRST alternative that current practice approves automatically, using execute_leg. An alternative that would be denied or escalated is not carried out — move on to the next one. If NO alternative can be approved automatically, the case takes the disposition of leg 1 (deny_case or escalate_case as that leg requires); the reason code may be the code of any blocked alternative that carries the same disposition as leg 1.
3) A PAYMENT BATCH listing several payment lines under `lines`. Settle it with a single decide_batch call that gives EVERY line its own action and, unless the action is approve, its own reason code. There is no partial submission and no way to revise a settled batch.

## transactions
Card transactions are checked against the category instant-approve limits: electronics $500.00, travel $1200.00. Each pending transaction lists the account's same-day approved total alongside the charge. Categories without a listed limit have no amount restriction. A transaction above its limit is escalated with reason over_limit. Groceries, utilities, dining, fitness, electronics, travel, jewelry and gift cards are all serviceable; gift cards are available to all account tiers, and no category is restricted by merchant region.

## limits
A limit-increase request from an account with a recent chargeback is denied with reason account_standing, whatever the amount. Otherwise the request is approved when the requested limit is within the approvable cap of $4200.00; above the cap, deny with reason cap_exceeded.

## transfers
Outbound transfers to payees added within the last 30 days must be escalated for review with reason new_payee_review. All transfers are additionally screened against the reporting threshold of $8000.00: a transfer above the threshold is escalated with reason reportable_amount. If both transfer checks apply, use new_payee_review.

## accounts
Accounts carry a verification level (basic or verified), a standing flag, tenure in years, a current credit limit, and a running total of today's approved card activity.