"""Banking L2 hidden state: COMPOSITION + REVISION tier.

Every scored window mixes single-policy anchors with tickets that require
two live rules and their precedence.  One component changes at each event.

The six divergent policies (everything else is documented TRUE):

  policy                     W0      W1      W2      W3      W4     role
  S1 daily cap $2000         static ESCALATE (unwritten; needs today total)
  S2 new-payee threshold     6500 static; docs incorrectly say every new
                             payee requires review regardless of amount
  D1 electronics limit       650     460     460     460     460    E1 tighten
                             (docs print 500 — the event CROSSES the print:
                              the adapt band moves from (500,650) approve to
                              (460,500) escalate, so the revision axis stays
                              in the adapt slice)
  D2 corridor scope          jewelry jewelry ELEC    ELEC    ELEC   E2 migrate
  D3 travel limit            900     900     900     1150    1150   E3 RELAX
  D4 approvable cap          4200    4200    4200    4200    3800   E4 tighten
                             (docs print 4200 — CORRECT until W4, then
                              stale: trusting the docs itself must be
                              revised)

Multi-policy tickets carry TWO stamped metrics:
z["n_firing"] = how many policy conditions STRICTLY hold under current
truth, while z["multi_path"] is a broader counterfactual-path flag. It can
include a currently false condition when changing that policy would alter
the applicable decision path. Therefore multi_path does NOT imply
n_firing >= 2; reports that mean current co-firing must use n_firing.
"""

from __future__ import annotations

# 84 single-request tickets PLUS 28 ordered-fallback tickets per window. The
# fallback supply is ADDED rather than reallocated, so the boundary ladders
# still pin each threshold from same-window serving evidence.
PER_WINDOW_SERVING = 100
# Scored test after change-sensitive adapt balancing.  The event windows are
# smaller because repeated persistent-adapt rows are capped by flip probes.
# General may exceed 40% because each window retains non-redundant policy
# controls and both scored sides of the core numeric boundaries.
PER_WINDOW_TEST = (74, 77, 76, 80, 80)
N_WINDOWS = 5

REASONS = ["corridor_blocked", "over_limit", "account_standing",
           "cap_exceeded", "reportable_amount", "daily_cap",
           "new_payee_review"]

EVENTS = [
    (1, "A_T1_electronics_limit", 460.0),        # E1 tighten ACROSS the print
    (2, "A_R1_corridor_scope", "electronics"),   # E2 scope migration
    (3, "A_T2_travel_limit", 1150.0),            # E3 numeric RELAXATION
    (4, "B_cap", 3800.0),                        # E4 docs-goes-stale tighten
]

_DEFAULTS = {
    "A_R1_corridor_scope": "jewelry",     # overseas x <scope> -> deny
    "A_T1_electronics_limit": 650.0,
    "A_T2_travel_limit": 900.0,
    "A_D_daily_cap": 2000.0,              # unwritten static; needs today total
    "B_R3_chargeback_blocks": True,       # documented -> general
    "B_cap": 4200.0,                      # docs print 4200; stale from W4
    "C_reporting_threshold": 8000.0,      # docs print 8000 -> general
    # Recent payees are reviewed only above this amount.  Combined with the
    # reporting threshold this creates a real new-payee x amount interaction:
    # above 8000 both rules fire and new_payee_review has precedence.
    "C_new_payee_threshold": 6500.0,
}

DOCS = {"electronics_limit": 500.0, "travel_limit": 1200.0,
        "limit_cap": 4200.0, "reporting_threshold": 8000.0,
        "new_payee_days": 30}

# The frozen manual expressed as a POLICY STATE, so a docs-follower can be
# evaluated by the same state-parameterised engine that evaluates truth. The
# manual knows no corridor and no daily cap, and its unconditional new-payee
# clause is an amount threshold of zero. Single-request slices are still
# derived from docs_* directly; multi-part cases use this, because walking
# legs needs one engine, not two.
DOCS_STATE = {"A_R1_corridor_scope": None,
              "A_T1_electronics_limit": DOCS["electronics_limit"],
              "A_T2_travel_limit": DOCS["travel_limit"],
              "A_D_daily_cap": float("inf"),
              "B_R3_chargeback_blocks": True,
              "B_cap": DOCS["limit_cap"],
              "C_reporting_threshold": DOCS["reporting_threshold"],
              "C_new_payee_threshold": 0.0}




def truth(window: int) -> dict:
    state = dict(_DEFAULTS)
    for w, key, value in EVENTS:
        if window >= w:
            state[key] = value
    return state


# ---------- ground truth per request type ----------

def decide_txn(account, merchant, amount, window):
    """A: card transaction. Precedence: corridor > daily > limit.
    The precedence chain is exercised deliberately: corridor x limit
    (overseas electronics above every limit), corridor x daily (overseas
    jewelry with a high same-day total), daily x limit (electronics/travel
    over their limit with a high total) — the multi-policy tickets."""
    st = truth(window)
    if (merchant["region"] == "overseas"
            and merchant["category"] == st["A_R1_corridor_scope"]):
        return "deny", "corridor_blocked"
    if account.get("today_total", 0.0) + amount > st["A_D_daily_cap"]:
        return "escalate", "daily_cap"
    lim = {"electronics": st["A_T1_electronics_limit"],
           "travel": st["A_T2_travel_limit"]}.get(merchant["category"])
    if lim is not None and amount > lim:
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
    # New-payee review takes precedence over ordinary amount reporting when
    # both conditions hold.  There is no restricted roster in L2.
    if (payee.get("added_days_ago", 999) < DOCS["new_payee_days"]
            and amount > st["C_new_payee_threshold"]):
        return "escalate", "new_payee_review"
    if amount > st["C_reporting_threshold"]:
        return "escalate", "reportable_amount"
    return "approve", None


# ---------- ordered fallback: one case, several ranked requests -------
#
# The customer ranks up to three ways of getting what they want; correct
# handling is to carry out the FIRST one current practice auto-approves.
# Answering therefore requires knowing the state of every policy that gates a
# leg above the chosen one, not just the one policy that decides a single
# request — that is the whole point of the form.
#
# Disposition when NOTHING is auto-approvable (frozen contract, printed in the
# manual and in the oracle bulletin):
#   1. the case takes LEG 1's disposition (deny or escalate) — the action is
#      pinned because status is part of the graded database state;
#   2. the reason code may be that of ANY blocked leg that shares leg 1's
#      disposition. Mixing dispositions would produce incoherent pairs such as
#      "escalate + corridor_blocked", so the accepted set is filtered by action
#      rather than being a plain union over legs (retail unions across legs
#      because its refusals live in the answer channel, not in the state).

def decide_leg(kind: str, account, leg, window):
    """One leg's verdict under current truth."""
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
    """-> (chosen_leg | None, action, reason, accepted_codes).

    action is "execute" when a leg is carried out; otherwise it is leg 1's
    own disposition. accepted_codes is empty for executed cases and is the
    tool-level normalisation set otherwise.
    """
    return _resolve_legs(decide_leg, kind, account, legs, window)


def docs_legs(kind: str, account, legs, window):
    return _resolve_legs(docs_leg, kind, account, legs, window)


def slice_of_legs(kind: str, account, legs, window) -> str:
    """A fallback case is adapt when the frozen manual would land on a
    DIFFERENT leg or a different disposition. The accepted-code set is
    deliberately NOT part of this comparison: a manual reader that stops on
    the same leg with the same action is following the documents correctly
    even if it cites another admissible code."""
    truth_v = decide_legs(kind, account, legs, window)[:3]
    docs_v = docs_legs(kind, account, legs, window)[:3]
    return "adapt" if truth_v != docs_v else "general"


# ---------- what a perfect docs-follower would do ----------

def docs_txn(account, merchant, amount, window):
    if merchant["category"] == "electronics" and amount > DOCS["electronics_limit"]:
        return "escalate", "over_limit"
    if merchant["category"] == "travel" and amount > DOCS["travel_limit"]:
        return "escalate", "over_limit"
    return "approve", None


def docs_limit(account, requested, window):
    # chargeback blocking is DOCUMENTED in L2: a docs-follower
    # applies it too, so B_standing hot tickets are general-slice
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
    lines.append(f"[card transactions] Overseas {st['A_R1_corridor_scope']} "
                 f"merchants: DENY (corridor_blocked). Other categories have "
                 f"no region restriction.")
    lines.append(f"[card transactions] Same-day approved total plus this "
                 f"charge above ${st['A_D_daily_cap']:.2f}: ESCALATE "
                 f"(daily_cap).")
    lines.append(f"[card transactions] Instant-approve limits: electronics "
                 f"${st['A_T1_electronics_limit']:.2f}, travel "
                 f"${st['A_T2_travel_limit']:.2f}; above the limit ESCALATE "
                 f"(over_limit).")
    lines.append("[limit increases] Accounts with a recent chargeback: DENY "
                 "(account_standing), whatever the amount.")
    lines.append(f"[limit increases] Approvable cap: ${st['B_cap']:.2f}; "
                 f"above the cap DENY (cap_exceeded).")
    lines.append(f"[transfers] Payees added fewer than "
                 f"{DOCS['new_payee_days']} days ago require review only "
                 f"above ${st['C_new_payee_threshold']:.2f}: ESCALATE "
                 f"(new_payee_review).")
    lines.append(f"[transfers] Otherwise, the reporting threshold is "
                 f"${st['C_reporting_threshold']:.2f}; above it ESCALATE "
                 f"(reportable_amount).")
    lines.append("Anything not restricted above: APPROVE. When two rules "
                 "apply, the first matching line above wins.")
    # the bulletin is FULL policy disclosure, so it must also state the
    # two facts a multi-part case needs; otherwise an oracle-informed agent
    # would be docked for something it cannot infer.
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
