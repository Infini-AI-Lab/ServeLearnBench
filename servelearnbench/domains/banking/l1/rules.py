"""Banking L1 hidden state: DISCOVERY tier — 2 statics + 2 one-way
drifts, few policies with thick per-window data.

Three request types share ONE output protocol (approve / deny(code) /
escalate(code)); they differ in scenario and in which policies apply:

  A  transaction authorization   pending card transaction
  B  limit-increase review       customer asks for a higher credit limit
  C  compliance screening        outbound transfer to a payee

Hidden policy is never a formula the agent must COMPUTE — every numeric
policy is a BOUNDARY the agent brackets from pass/fail feedback (amount
ladders straddle it). The frozen manual states the structure (what is
checked); exactly FOUR policies diverge from it, everything else is
documented correctly (the cap, chargeback rule and reporting threshold are
printed with their true values, so those families are pure general-slice
texture).

L1 schedule (every change only TIGHTENS — a conservative learner never
loses points; W2 is a consolidation window with no event):

  policy                          W0      W1      W2      W3     role
  S1 gift card needs verified     DENY    DENY    DENY    DENY   unwritten static
  S2 electronics instant limit    650     650     650     650    stale print (500)
  D1 overseas-jewelry corridor    -       DENY    DENY    DENY   rule birth (E1)
  D2 restricted payees            3       3       3       5      list growth (E2)
  (documented true: cap 5000, chargeback deny, reporting 8000)

Beat thickening: the changing policy's supply peaks in its event window
(D1: 10 hot serving at W1; D2: 12 live draws at W3) — quotas vary by a
DECLARED per-window table.

Precedence: standing > cap (limit requests) and restricted > reporting
(transfers) are exercised by co-firing tickets. Within card transactions the
rules key on DISJOINT categories, so txn precedence is vacuous by
construction (declared, not tested).
"""

from __future__ import annotations

PER_WINDOW_SERVING = 84
# Scored test after change-sensitive adapt balancing. L1 retains two scored
# positives for each documented numeric boundary and two penalties for each
# off-scope over-generalization; W2 remains the compact retention window.
PER_WINDOW_TEST = (41, 47, 42, 48)
N_WINDOWS = 4

REASONS = ["corridor_blocked", "verification_required", "over_limit",
           "account_standing", "cap_exceeded", "restricted_payee",
           "reportable_amount"]

EVENTS = [
    (1, "A_R1_overseas_jewelry_blocked", True),   # E1 rule birth
    (3, "C_restricted_payees", 5),                # E2 list growth 3 -> 5
]

_DEFAULTS = {
    # A — transaction authorization
    "A_R1_overseas_jewelry_blocked": False,   # D1: born at W1 (unwritten)
    "A_R2_giftcard_needs_verified": True,     # S1: unwritten; docs say all ok
    "A_T1_electronics_limit": 650.0,          # S2: docs print 500
    # B — limit-increase review (both DOCUMENTED correctly)
    "B_R3_chargeback_blocks": True,           # documented -> general
    "B_cap": 5000.0,                          # docs print 5000 -> general
    # C — compliance screening
    "C_reporting_threshold": 8000.0,          # docs print 8000 -> general
    "C_restricted_payees": 3,                 # D2: first N of the roster are
                                              # live; docs list none
}

DOCS = {"electronics_limit": 500.0, "limit_cap": 5000.0,
        "reporting_threshold": 8000.0}

# payee ids that CAN be restricted, in activation order (first N live).
# Filled by world._build_payees() from the SHUFFLED id assignment, so the
# roster cannot be read off the id alone.
RESTRICTED_ROSTER: list = []


def truth(window: int) -> dict:
    state = dict(_DEFAULTS)
    for w, key, value in EVENTS:
        if window >= w:
            state[key] = value
    return state


def restricted_set(window: int) -> set:
    return set(RESTRICTED_ROSTER[:truth(window)["C_restricted_payees"]])


# ---------- ground truth per request type ----------

def decide_txn(account, merchant, amount, window):
    """A: card transaction. Precedence: corridor > verification > limit."""
    st = truth(window)
    if (st["A_R1_overseas_jewelry_blocked"] and merchant["category"] == "jewelry"
            and merchant["region"] == "overseas"):
        return "deny", "corridor_blocked"
    if (st["A_R2_giftcard_needs_verified"] and merchant["category"] == "gift_cards"
            and account["verification"] == "basic"):
        return "deny", "verification_required"
    if merchant["category"] == "electronics" and amount > st["A_T1_electronics_limit"]:
        return "escalate", "over_limit"
    return "approve", None


def decide_limit(account, requested, window):
    """B: limit increase. Precedence: standing > cap."""
    st = truth(window)
    if st["B_R3_chargeback_blocks"] and account["standing"] == "recent_chargeback":
        return "deny", "account_standing"
    if requested > st["B_cap"]:
        return "deny", "cap_exceeded"
    return "approve", None


def decide_transfer(account, payee, amount, window):
    """C: outbound transfer. Precedence: restricted payee > reporting."""
    st = truth(window)
    if payee["payee_id"] in restricted_set(window):
        return "deny", "restricted_payee"
    if amount > st["C_reporting_threshold"]:
        return "escalate", "reportable_amount"
    return "approve", None


# ---------- what a perfect docs-follower would do ----------

def docs_txn(account, merchant, amount, window):
    if merchant["category"] == "electronics" and amount > DOCS["electronics_limit"]:
        return "escalate", "over_limit"
    return "approve", None


def docs_limit(account, requested, window):
    # chargeback blocking is DOCUMENTED: a docs-follower
    # applies it too, so B_standing hot tickets are general-slice
    if account["standing"] == "recent_chargeback":
        return "deny", "account_standing"
    if requested > DOCS["limit_cap"]:
        return "deny", "cap_exceeded"
    return "approve", None


def docs_transfer(account, payee, amount, window):
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
    live = ", ".join(sorted(restricted_set(w)))
    # only constraints IN FORCE are stated (listing an inactive axis — "no
    # restriction on overseas jewelry" — pre-announced where drift would
    # land); the closing line makes the bulletin complete without it
    lines = ["CURRENT ACTUAL POLICY (authoritative; overrides the manual "
             "where they differ):"]
    if st["A_R1_overseas_jewelry_blocked"]:
        lines.append("[card transactions] Jewelry from overseas merchants: "
                     "DENY (corridor_blocked).")
    if st["A_R2_giftcard_needs_verified"]:
        lines.append("[card transactions] Gift cards on basic-verification "
                     "accounts: DENY (verification_required); verified "
                     "accounts unrestricted.")
    lines.append(f"[card transactions] Electronics instant-approve limit: "
                 f"${st['A_T1_electronics_limit']:.2f}; above it ESCALATE "
                 f"(over_limit).")
    if st["B_R3_chargeback_blocks"]:
        lines.append("[limit increases] Accounts with a recent chargeback: "
                     "DENY (account_standing), whatever the amount.")
    lines.append(f"[limit increases] Approvable cap: ${st['B_cap']:.2f}; "
                 f"above the cap DENY (cap_exceeded).")
    lines.append(f"[transfers] Restricted payees (DENY, restricted_payee): "
                 f"{live}.")
    lines.append(f"[transfers] Reporting threshold: "
                 f"${st['C_reporting_threshold']:.2f}; above it ESCALATE "
                 f"(reportable_amount).")
    lines.append("Anything not restricted above: APPROVE. When two rules "
                 "apply, the first matching line above wins.")
    return "\n".join(lines)
