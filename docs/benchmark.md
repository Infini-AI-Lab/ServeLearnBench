# Benchmark

## Evolving-environment streaming datasets

A scenario is a chronological stream of tasks divided into environment windows,
`D = W_0 ‖ W_1 ‖ … ‖ W_{S-1}`. A latent environment state governs each window: the store's actual rules, the bank's
actual thresholds, or the current customer's taste. The state is fixed within a window and may change at a window
boundary. Across windows the environment introduces new policies, revises earlier ones, removes them, or returns to
a policy seen before. Neither the state nor its changes are disclosed to the agent.

Each window holds **serving** tasks followed by **test** tasks. Both cover the same active policy components with
disjoint task instances, so a test measures transfer to new instances rather than memorization.

Orthogonal to that split, each task is **Hidden-Dependent** if the visible information (task, documents, tool
results) cannot determine the correct behavior, and **Fully Specified** otherwise. A good learner raises its
Hidden score without losing its Fully Specified score.

## Domains

**Retail.** A multi-turn customer-support environment over an order database: cancel or modify pending orders,
return or exchange delivered items, answer questions. Harder tiers compose several items or fallback alternatives
in one request. The agent reads the store's policy and workflow documents with `read_docs`, which may be incomplete
or out of date. A task is correct when the final database state is the intended one, or when the request is refused
with the correct reason code.

**Banking.** A case-review environment: authorize pending card transactions, review credit-limit increases, screen
outbound transfers. L2 and L3 add ranked alternatives (carry out the first admissible one) and payment batches (a
verdict per line). Hidden policies are numeric: category and travel limits, daily totals, account standing, limit
caps, merchant corridors, payee-age thresholds. A case is correct when the decision and its reason code match.

**Pitch.** A single-turn generation environment: write a 40–80 word sales pitch from a factual product sheet. The
hidden state is the active customer's preference profile. A judge classifies every sheet attribute as mentioned,
contradicted or not addressed and lists invented claims; the score rewards covering liked attributes and penalizes
touching disliked ones, contradictions and invented claims. Every Pitch task is Hidden-Dependent.

## Tiers

| Tier | Environment change | Task composition |
|---|---|---|
| L1 · acquisition | previously unseen states | mostly single-policy requests; non-repeating Pitch profiles |
| L2 · revision and composition | earlier states are revised | multi-policy Retail requests, ranked or batched Banking decisions, recurring Pitch profiles over more attributes |
| L3 · non-monotonic evolution | reversals, removals and re-entry of earlier states | as in L2; Pitch cycles back to earlier profiles |

## Scenarios

| Scenario | Windows | Serving | Test | Latent structure |
|---|---:|---:|---:|---|
| `retail_l1` | 6 | 720 | 384 | categorical rules |
| `retail_l2` | 7 | 810 | 765 | categorical rules |
| `retail_l3` | 8 | 624 | 537 | categorical rules with reversals |
| `banking_l1` | 4 | 336 | 178 | numeric boundaries |
| `banking_l2` | 5 | 500 | 387 | composed numeric boundaries |
| `banking_l3` | 6 | 600 | 551 | numeric boundaries with reversals |
| `pitch_l1` | 4 | 216 | 96 | customer preferences |
| `pitch_l2` | 6 | 324 | 144 | preference drift |
| `pitch_l3` | 7 | 378 | 168 | preferences with reversals |
| **Total** | **53** | **4,508** | **3,210** | |
