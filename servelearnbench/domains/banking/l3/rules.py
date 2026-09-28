"""Banking L3 hidden state: L2-hard single policies + temporal change.

L3 deliberately starts from the same inference primitives as L2: scoped
corridors, two category limits, a same-day sum cap, standing/cap precedence,
and a recent-payee age x amount rule. Later windows alter those SAME rules;
L3 difficulty must come from maintaining and revising an L2-strength policy
model, not from swapping in easier entity-memory tasks.

  policy                    W0     W1     W2      W3     W4      W5
  electronics limit         650    460    460     650    650     650
  travel limit              900    900    900     1150   1150    900
  corridor scope            JEW    JEW    ELEC    ELEC   JEW     -
  daily sum cap             2000   2000   2000    2000   1700    1700
  approvable cap            4200   3800   3800    3800   4200    4200
  recent-payee threshold    6500   6500   7200    7200   7200    7200

Chargeback blocking and the $8000 reporting threshold stay stable.  W3/W4
contain genuine reopenings; W5 removes the corridor restriction while the
travel limit tightens again.  ``n_firing`` records
strict current co-firing, while ``multi_path`` also admits a currently-false
but counterfactually relevant second policy path.
"""

from __future__ import annotations

PER_WINDOW_SERVING = 100
# Scored test size per window after coverage-preserving pruning.
BASELINE_PER_WINDOW_TEST = (72, 90, 89, 97, 105, 98)
PER_WINDOW_TEST = BASELINE_PER_WINDOW_TEST
N_WINDOWS = 6

REASONS = ["corridor_blocked", "over_limit", "daily_cap",
           "account_standing", "cap_exceeded", "new_payee_review",
           "reportable_amount"]

EVENTS = [
    (1, "A_T1_electronics_limit", 460.0),
    (1, "B_cap", 3800.0),
    (2, "A_R1_corridor_scope", "electronics"),
    (2, "C_new_payee_threshold", 7200.0),
    (3, "A_T1_electronics_limit", 650.0),       # reopen
    (3, "A_T2_travel_limit", 1150.0),
    (4, "B_cap", 4200.0),                       # reopen
    (4, "A_R1_corridor_scope", "jewelry"),      # reopen/rescope
    (4, "A_D_daily_cap", 1700.0),
    (5, "A_R1_corridor_scope", None),           # reopen/remove restriction
    (5, "A_T2_travel_limit", 900.0),            # reclose
]

_DEFAULTS = {
    "A_R1_corridor_scope": "jewelry",
    "A_T1_electronics_limit": 650.0,
    "A_T2_travel_limit": 900.0,
    "A_D_daily_cap": 2000.0,
    "B_R3_chargeback_blocks": True,
    "B_cap": 4200.0,
    "C_reporting_threshold": 8000.0,
    "C_new_payee_threshold": 6500.0,
}

DOCS = {"electronics_limit": 500.0, "travel_limit": 1200.0,
        "limit_cap": 4200.0, "reporting_threshold": 8000.0,
        "new_payee_days": 30}

# The frozen manual as a POLICY STATE, so a docs-follower can be evaluated by
# the same state-parameterised engine that evaluates truth. Multi-part cases
# need one engine for both, so the slice label and counterfactual checks
# derive "what the manual would do" the same way.
DOCS_STATE = {"A_R1_corridor_scope": None,
              "A_T1_electronics_limit": DOCS["electronics_limit"],
              "A_T2_travel_limit": DOCS["travel_limit"],
              "A_D_daily_cap": float("inf"),
              "B_R3_chargeback_blocks": True,
              "B_cap": DOCS["limit_cap"],
              "C_reporting_threshold": DOCS["reporting_threshold"],
              "C_new_payee_threshold": 0.0}


# Compatibility only: L3 does not score roster membership.
RESTRICTED_ROSTER: list = []


def truth(window: int) -> dict:
    state = dict(_DEFAULTS)
    for w, key, value in EVENTS:
        if window >= w:
            state[key] = value
    return state


def restricted_set(window: int) -> set:
    return set()


# ---------- ground truth per request type ----------

def decide_txn(account, merchant, amount, window):
    """A: card transaction. Precedence: corridor > daily sum > category."""
    st = truth(window)
    if (st["A_R1_corridor_scope"] is not None
            and merchant["region"] == "overseas"
            and merchant["category"] == st["A_R1_corridor_scope"]):
        return "deny", "corridor_blocked"
    if account.get("today_total", 0.0) + amount > st["A_D_daily_cap"]:
        return "escalate", "daily_cap"
    limit = {"electronics": st["A_T1_electronics_limit"],
             "travel": st["A_T2_travel_limit"]}.get(merchant["category"])
    if limit is not None and amount > limit:
        return "escalate", "over_limit"
    return "approve", None


def decide_limit(account, requested, window):
    st = truth(window)
    if st["B_R3_chargeback_blocks"] and account["standing"] == "recent_chargeback":
        return "deny", "account_standing"
    if requested > st["B_cap"]:
        return "deny", "cap_exceeded"
    return "approve", None


def decide_transfer(account, payee, amount, window):
    st = truth(window)
    if (payee.get("added_days_ago", 999) < DOCS["new_payee_days"]
            and amount > st["C_new_payee_threshold"]):
        return "escalate", "new_payee_review"
    if amount > st["C_reporting_threshold"]:
        return "escalate", "reportable_amount"
    return "approve", None


# ---------- ordered fallback: one case, several ranked requests ------
#
# Same contract as L2 (the tiers are deliberately equal on the single-ticket
# axis; L3's difference is the oscillation schedule). Correct handling is to
# carry out the FIRST ranked request current practice auto-approves. When
# nothing is auto-approvable the case takes LEG 1's disposition, and the
# reason code may be that of any blocked leg sharing that disposition.

def decide_leg(kind: str, account, leg, window):
    if kind == "txn":
        return decide_txn(account, leg["merchant"], leg["amount"], window)
    if kind == "limit":
        return decide_limit(account, leg["requested"], window)
    return decide_transfer(account, leg["payee"], leg["amount"], window)


def docs_leg(kind: str, account, leg, window):
    if kind == "txn":
        return docs_txn(account, leg["merchant"], leg["amount"], window)
    if kind == "limit":
        return docs_limit(account, leg["requested"], window)
    return docs_transfer(account, leg["payee"], leg["amount"], window)


def _resolve_legs(decider, kind, account, legs, window):
    outcomes = [decider(kind, account, leg, window) for leg in legs]
    for i, (action, _code) in enumerate(outcomes):
        if action == "approve":
            return i, "execute", None, []
    lead_action, lead_code = outcomes[0]
    accepted = sorted({code for action, code in outcomes
                       if action == lead_action and code is not None})
    return None, lead_action, lead_code, accepted


def decide_legs(kind: str, account, legs, window):
    """-> (chosen_leg | None, action, reason, accepted_codes)."""
    return _resolve_legs(decide_leg, kind, account, legs, window)


def docs_legs(kind: str, account, legs, window):
    return _resolve_legs(docs_leg, kind, account, legs, window)


# ---------- what a perfect docs-follower would do ----------

def docs_txn(account, merchant, amount, window):
    limit = {"electronics": DOCS["electronics_limit"],
             "travel": DOCS["travel_limit"]}.get(merchant["category"])
    if limit is not None and amount > limit:
        return "escalate", "over_limit"
    return "approve", None


def docs_limit(account, requested, window):
    if account["standing"] == "recent_chargeback":
        return "deny", "account_standing"
    if requested > DOCS["limit_cap"]:
        return "deny", "cap_exceeded"
    return "approve", None


def docs_transfer(account, payee, amount, window):
    if payee.get("added_days_ago", 999) < DOCS["new_payee_days"]:
        return "escalate", "new_payee_review"
    if amount > DOCS["reporting_threshold"]:
        return "escalate", "reportable_amount"
    return "approve", None


DECIDERS = {"txn": (decide_txn, docs_txn),
            "limit": (decide_limit, docs_limit),
            "transfer": (decide_transfer, docs_transfer)}


def slice_of(kind: str, *args) -> str:
    truth_fn, docs_fn = DECIDERS[kind]
    return "adapt" if truth_fn(*args) != docs_fn(*args) else "general"


WINDOW_STARTS = [w * PER_WINDOW_SERVING + 1 for w in range(N_WINDOWS)]


def window_range(w: int):
    """(first, last) serving t of window w — the runner API shared across
    domains (oracle notes are fetched at a window's first serving tick)."""
    return (WINDOW_STARTS[w], WINDOW_STARTS[w] + PER_WINDOW_SERVING - 1)


def n_windows() -> int:
    return N_WINDOWS


def window_of(t: int) -> int:
    return min((t - 1) // PER_WINDOW_SERVING, N_WINDOWS - 1)


def oracle_note(t: int) -> str:
    w = window_of(t)
    st = truth(w)
    lines = ["CURRENT ACTUAL POLICY (authoritative; overrides the manual "
             "where they differ):"]
    if st["A_R1_corridor_scope"] is None:
        lines.append("[card transactions] No overseas corridor restriction "
                     "is in force: region alone never decides a transaction.")
    else:
        lines.append(f"[card transactions] Overseas "
                     f"{st['A_R1_corridor_scope']} merchants: DENY "
                     f"(corridor_blocked). Other categories have no region "
                     f"restriction.")
    lines.append(f"[card transactions] Same-day approved total plus the "
                 f"pending charge above ${st['A_D_daily_cap']:.2f}: "
                 "ESCALATE (daily_cap).")
    lines.append(f"[card transactions] Electronics limit: "
                 f"${st['A_T1_electronics_limit']:.2f}; travel limit: "
                 f"${st['A_T2_travel_limit']:.2f}. Above either limit "
                 "ESCALATE (over_limit).")
    lines.append("[limit increases] Accounts with a recent chargeback: DENY "
                 "(account_standing), whatever the amount.")
    lines.append(f"[limit increases] Approvable cap: ${st['B_cap']:.2f}; "
                 f"above the cap DENY (cap_exceeded).")
    lines.append(f"[transfers] Payees added fewer than "
                 f"{DOCS['new_payee_days']} days ago require review only "
                 f"above ${st['C_new_payee_threshold']:.2f}: ESCALATE "
                 "(new_payee_review).")
    lines.append(f"[transfers] Otherwise, reporting threshold: "
                 f"${st['C_reporting_threshold']:.2f}; above it ESCALATE "
                 f"(reportable_amount).")
    lines.append("Anything not restricted above: APPROVE. When two rules "
                 "apply, the first matching line above wins.")
    # Full policy disclosure has to include how a multi-part case resolves,
    # or an oracle-informed agent is docked for something it cannot infer.
    lines.append("[ranked alternatives] Carry out the FIRST listed "
                 "alternative that the lines above APPROVE; alternatives that "
                 "would be denied or escalated are skipped. If none can be "
                 "approved, the case takes leg 1's own disposition, citing "
                 "the code of any blocked alternative with that same "
                 "disposition.")
    lines.append("[payment batches] Every line is screened ON ITS OWN "
                 "against the transfer lines above — the batch total is not "
                 "screened and does not affect any line. Each line receives "
                 "its own action and reason code.")
    return "\n".join(lines)
