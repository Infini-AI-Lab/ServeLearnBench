"""Assemble the pitch benchmark bundle for a tier (L1/L2/L3).

Reuses the scenario-agnostic harness: the returned bundle carries a serving
stream, per-window test sets, a per-tier Rules object, a system prompt, and the
RubricVerifier (judge-based reward). No tools, no DB — the agent writes a pitch
and submits it via finish({"pitch": ...}).
"""

from __future__ import annotations

import random as _random

from ...engine.types import Task
from ...prompts import compose_system_prompt
from . import customers as C
from . import products as P
from . import rules as R
from .attributes import FULL_DIMS, L1_DIMS
from .verifier import RubricVerifier

# Per-item difficulty via dimensionality. L1 = 5-dim sheets (fit the word
# budget). L2/L3 = 10-dim sheets (full mention does not fit the 40-80 word
# budget; selection forced). Each dim set has its own product pool.
# Each window holds 54 serving / 24 test items. Pools are sized so NO PRODUCT
# IS EVER REUSED across a tier's windows (L1 needs 4x78=312, 10-dim needs
# 7x78=546); reuse would grow with the window index and confound learning
# curves. On L1 the 48-combo space still repeats at the information level
# across windows (unavoidable at 5 dims).
TIER_DIMS = {"L1": L1_DIMS, "L2": FULL_DIMS, "L3": FULL_DIMS}
# combos held back from serving per window so the test adapt quota always has
# serving-fresh combos (5-dim: ~4 products/combo -> 8 combos suffice; 10-dim:
# ~1 product/combo -> the walk extends until half-test products are reserved)
RESERVE_COMBOS = 8
# ...but never more than 1/RESERVE_FRACTION of the combos that exist, and the
# walk below picks WHICH combos by value balance rather than by size alone.
# Reservation runs on EFFECTIVE combos, and a customer scoring only 4 dims has
# ~12 of them holding 28-60 products each; reserving a fixed 8 there could
# leave serving with a value distribution the test does not share (e.g. a
# test value never seen in serving), which the cap and balance walk prevent.
RESERVE_FRACTION = 2
# Max |value share - its natural share among this window's straddlers| that
# either side of the split may carry, per scored (dimension, value). What
# actually balances the split is the greedy's objective (it picks the combo
# leaving the reserved side closest to natural); this band is the fail-closed
# guard on top, so a change that breaks the balance stops the build. Every
# realized window sits well inside it.
MARGIN_BAND = 0.15
# Same check on the REALIZED draws (54 serving / 24 test), which carry ordinary
# sampling noise on top of the pool-level band above. Fail-closed in build.
MAX_DRAW_GAP = 0.35
# Fixed per-window slice quota (exact): every window of a tier has IDENTICAL
# test composition. There is no general slice — a no-conflict product is won
# by praise-everything play, which measures nothing this benchmark is about.
# All test items are adapt/reopen: the product carries both liked and
# disliked facts for the active customer, so every item requires selection.
GENERAL_QUOTA = {"L1": 0, "L2": 0, "L3": 0}
TIER_POOL = {"L1": ("pitch_v2_L1", 480), "L2": ("pitch_v2", 640), "L3": ("pitch_v2", 640)}

SYSTEM_PROMPT_HEAD = (
    "You are an expert copywriter. For each product you are shown, write ONE "
    "persuasive sales pitch of 40-80 words.\n\n"
    "You are pitching to a customer whose taste you must infer "
    "from feedback over time — you are NOT told who they are. The SAME product "
    "fact can win one customer over and turn another away, so choose carefully "
    "which facts to emphasize and which to leave out.\n\n"
    # Drift disclosure: every acting agent is told that the target can
    # change. The wording deliberately does NOT say whether it is one person
    # whose taste moves or a succession of customers: L3 measures
    # re-recognising a taste that returns rather than overwriting it, and
    # "one person who changed" argues for overwriting. It says THAT the taste
    # changes and in both directions; never when, how often, or that tastes
    # recur — that is the object of learning.
    "# Environment\n"
    "The taste you are pitching to can CHANGE over time: a preference you "
    "have correctly inferred may stop applying — in either direction (a fact "
    "that was winning them over may start turning them away, and one you "
    "learned to leave out may become the one that lands). You are judged "
    "against the CURRENT taste, and judged outcomes are your only ground "
    "truth about it.\n\n"
    "BE TRUTHFUL. Describe the product ONLY as the sheet states it. Never "
    "claim or imply a fact the sheet does not support, and never state the "
    "opposite of a sheet fact (calling a mass-produced item handcrafted, a "
    "trending item unknown, a final-sale item returnable). A false or "
    "reversed claim still counts as bringing that topic up, and is scored "
    "against you. If a fact would put the customer off, LEAVE IT OUT — "
    "silence is allowed, lying is not.\n\n"
    "HARD LIMIT: a pitch longer than 90 words scores at most 25/100."
)
SYSTEM_PROMPT = compose_system_prompt(SYSTEM_PROMPT_HEAD, "pitch")


def effective_dims(tier: str, customer_key: str):
    """The dims this customer scores on (valence-nonzero) within the tier's
    dim set — the resolution at which holdout must operate."""
    from .attributes import ATTRIBUTES
    return [d for d in TIER_DIMS[tier]
            if any(C.valence(customer_key, d, v) != 0 for v in ATTRIBUTES[d])]


def _slice_of(tier: str, w: int, product, customer_key: str) -> str:
    if C.straddles(customer_key, product):
        return "reopen" if R.is_reopen_window(tier, w) else "adapt"
    return "general"


def _task(tier, w, t, product, customer_key, phase):
    sl = _slice_of(tier, w, product, customer_key)
    tid = f"pitch_{tier}_w{w}_{sl}_{product['product_id']}_{phase}"
    instruction = "Product to pitch:\n" + P.render_sheet(product) + "\n\nWrite your pitch."
    return Task(task_id=tid, user_id="pitch", instruction=instruction, actions=[],
                answer=None,
                z={"t": t, "window": w, "tier": tier, "slice": sl,
                   "family": customer_key, "customer_key": customer_key,
                   "product": product})


def build_bundle(tier: str):
    rules = R.Rules(tier)
    seed, n = TIER_POOL[tier]
    pool = P.generate_pool(n, seed=seed, dims=TIER_DIMS[tier])
    nw = R.n_windows(tier)
    serving, test_by_w = [], {}
    # per-window reservation stats: the ACHIEVABLE test-combo count (the cap
    # moves with the customer's effective combo space — see RESERVE_FRACTION)
    combo_stats: dict = {}
    # Global no-reuse: a product appears in exactly ONE window across the
    # whole tier, across customers too — pools are sized for it (78 per
    # window).
    used_ids: set = set()
    # combos this customer has already been SERVED (feedback given) — later
    # test windows of the same customer must not test those combos
    # (information-level holdout across windows)
    served_combos: dict = {}
    # PASS 1 — reserve each window's EXACT general quota globally, scarcest
    # customers first. general candidates carry no red/neg conflict for the
    # window's customer but MUST have >=1 positive (a zero-positive item's
    # optimal pitch is silence, so it cannot discriminate). Reserving
    # up front keeps earlier windows from consuming later windows' scarce
    # general capacity (10-dim customers have only 4-12 candidates).
    #
    # GENERAL_QUOTA is 0 on every tier, so this pass is skipped; it is kept
    # behind the guard so a general slice can be enabled by setting a quota.
    q_gen = GENERAL_QUOTA[tier]
    gdims = TIER_DIMS[tier]
    gen_by_w = {w: [] for w in range(nw)}
    if q_gen:
        gen_cap = {}
        for ck in set(R.SCHEDULES[tier]):
            lst = [p for p in pool
                   if not C.straddles(ck, p) and C.positives(ck, p)]
            _random.Random(f"pitch_v2_gen:{tier}:{ck}").shuffle(lst)
            gen_cap[ck] = lst
        # products usable by fewer customers go first, so shared candidates are
        # left for the customers that have no alternative
        flex = {}
        for ck, lst in gen_cap.items():
            for p in lst:
                flex[p["product_id"]] = flex.get(p["product_id"], 0) + 1
        for ck in gen_cap:
            gen_cap[ck] = sorted(gen_cap[ck],
                                 key=lambda p: flex[p["product_id"]])
        for w in sorted(range(nw),
                        key=lambda w: len(gen_cap[R.customer_of_window(tier, w)])):
            ck = R.customer_of_window(tier, w)
            cands = [p for p in gen_cap[ck] if p["product_id"] not in used_ids]
            # round-robin across attribute combos so a window does not fill
            # with copies of ONE combo (same sheet, different names). Combo
            # diversity is capped by structure (anti_marketing has only 2
            # no-conflict combos).
            groups: dict = {}
            for p in cands:
                groups.setdefault(tuple(p[d] for d in gdims), []).append(p)
            order = list(groups)
            pick, i = [], 0
            while len(pick) < q_gen and any(groups[c] for c in order):
                c = order[i % len(order)]
                if groups[c]:
                    pick.append(groups[c].pop(0))
                i += 1
            if len(pick) < q_gen:
                raise ValueError(f"pitch {tier} W{w}: general quota short "
                                 f"({len(pick)}/{q_gen})")
            gen_by_w[w] = pick
            used_ids.update(p["product_id"] for p in pick)

    # PASS 2 — serving + adapt test per window, never touching reserved
    # generals (they are in used_ids already)
    for w in range(nw):
        ck = R.customer_of_window(tier, w)
        rnd = _random.Random(f"pitch_v2:{tier}:{ck}")
        order = list(pool)
        rnd.shuffle(order)
        avail = [p for p in order if p["product_id"] not in used_ids]
        straddle = [p for p in avail if C.straddles(ck, p)]
        # Test is held out at the INFORMATION level: no test item may share
        # its attribute combination with any serving item of the same window
        # (5-dim has only 48 combos — id-level holdout is not holdout there).
        # To keep the adapt quota from being starved on small combo spaces
        # (eco's ~24 straddle combos vs 54 serving items), RESERVE straddle
        # combos for test FIRST: walk the shuffled list collecting the first
        # distinct combos until >= RESERVE_COMBOS combos and >= half-test
        # products are reserved; serving then fills its 54 from the remaining
        # combos (serving may repeat combos — it always could). Combos this
        # customer was already served in an earlier window are skipped.
        # Holdout operates on EFFECTIVE combos — the values of the dims this
        # customer actually scores on (valence-nonzero). Two products
        # differing only in scored-zero dims are the SAME item to the rubric,
        # so full-dim holdout would allow testing combos already served.
        nz = effective_dims(tier, ck)
        combo = lambda p: tuple(p[d] for d in nz)
        h = R.PER_WINDOW_TEST - q_gen  # exact adapt/reopen quota
        old = served_combos.get(ck, set())
        # Reserve effective combos for test. Reserving SMALLEST first keeps
        # serving fed (effective combos are few and product-heavy —
        # risk_averse on 5-dim: 14 combos, ~20 products each — so reserving in
        # list order starves serving), but size order alone is blind to WHICH
        # values end up on which side. The walk therefore prefers small
        # combos, and additionally keeps both sides' scored-value shares
        # inside MARGIN_BAND of their natural share; among the combos that
        # qualify it takes the one that leaves the reserved side most
        # balanced.
        by_combo: dict = {}
        for p in straddle:
            c = combo(p)
            if c not in old:
                by_combo.setdefault(c, []).append(p)
        total = sum(len(v) for v in by_combo.values())
        # natural share of every scored (dim, value) among this window's
        # straddlers — the target both sides are held to
        nat = {}
        for d in nz:
            cnt: dict = {}
            for lst in by_combo.values():
                for p in lst:
                    cnt[p[d]] = cnt.get(p[d], 0) + 1
            for v, k in cnt.items():
                nat[(d, v)] = k / total

        # per-combo value tallies, so the greedy below scores a candidate split
        # from combo-sized vectors instead of re-walking every product
        keys = sorted(nat)
        vec = {c: tuple(sum(1 for p in lst if p[d] == v) for d, v in keys)
               for c, lst in by_combo.items()}
        full = tuple(sum(vec[c][i] for c in vec) for i in range(len(keys)))

        def _dev(agg, n):
            """max |share - natural share| over scored values, for a tally."""
            if not n:
                return 1.0
            return max(abs(agg[i] / n - nat[k]) for i, k in enumerate(keys))

        # never reserve more than a third of what exists: past that the serve
        # side has too few combos left to span the value space at all
        k_max = min(RESERVE_COMBOS, max(2, len(by_combo) // RESERVE_FRACTION))
        order = sorted(by_combo, key=lambda c: (len(by_combo[c]), c))
        reserved, n_res = [], 0
        agg = tuple(0 for _ in keys)
        while len(reserved) < k_max or n_res < h:
            best = None
            for c in order:
                if c in reserved:
                    continue
                sz = len(by_combo[c])
                if total - n_res - sz < R.PER_WINDOW_SERVING:
                    continue  # reserving this combo would starve serving
                a = tuple(agg[i] + vec[c][i] for i in range(len(keys)))
                rest = tuple(full[i] - a[i] for i in range(len(keys)))
                # the serve side must always stay in band; the reserved side
                # only once it is big enough for its share to mean anything
                if _dev(rest, total - n_res - sz) > MARGIN_BAND:
                    continue
                d_cand = _dev(a, n_res + sz)
                if len(reserved) + 1 >= k_max and d_cand > MARGIN_BAND:
                    continue
                score = (d_cand, sz, c)
                if best is None or score < best[0]:
                    best = (score, c, a)
            if best is None:
                break
            reserved.append(best[1])
            n_res += len(by_combo[best[1]])
            agg = best[2]
        combo_stats[w] = {"available": len(by_combo), "cap": k_max,
                          "reserved": len(reserved)}
        reserved = set(reserved)
        if len(reserved) < 2 or n_res < h:
            raise ValueError(f"pitch {tier} W{w}: balanced reservation short "
                             f"({len(reserved)} combos / {n_res} products, "
                             f"need >=2 and >={h})")
        test_pool = [p for p in straddle if combo(p) in reserved]
        serve_pool = [p for p in straddle if combo(p) not in reserved]

        sv = serve_pool[:R.PER_WINDOW_SERVING]
        if len(sv) < R.PER_WINDOW_SERVING:
            raise ValueError(f"pitch {tier} W{w}: serving short after combo "
                             f"reservation ({len(sv)}/{R.PER_WINDOW_SERVING})")
        for i, p in enumerate(sv):
            t = w * R.PER_WINDOW_SERVING + i + 1
            serving.append(_task(tier, w, t, p, ck, "sv"))
        sv_combos = {combo(p) for p in sv}

        def draw(cands, k, used, allow_dup=False):
            out = []
            for p in cands:
                if len(out) == k:
                    break
                if not allow_dup and combo(p) in used:
                    continue
                out.append(p)
                used.add(combo(p))
            return out

        used: set = set()
        # adapt/reopen quota (EXACT h): reserved combos first (combo-unique,
        # then repeats), then fresh straddlers whose combo never entered this
        # window's serving nor this customer's earlier serving
        te = draw(test_pool, h, used)
        te += draw([p for p in test_pool if p not in te],
                   h - len(te), used, allow_dup=True)
        if len(te) < h:
            spare = [p for p in serve_pool[R.PER_WINDOW_SERVING:]
                     if combo(p) not in sv_combos and combo(p) not in old]
            te += draw(spare, h - len(te), used, allow_dup=True)
        if len(te) < h:
            raise ValueError(f"pitch {tier} W{w}: adapt quota short "
                             f"({len(te)}/{h})")
        # Fail-closed check on the REALIZED draws, not just the pools.
        # Two things must hold for a window's feedback to be about what its
        # test asks: (1) every scored value the test carries was priced at
        # least once in serving — a value tested but never served cannot be
        # learned from feedback at all; (2) the two distributions are within
        # MAX_DRAW_GAP, so no window is structurally easier or harder than the
        # stream that precedes it.
        for d in nz:
            for v in {p[d] for p in te} | {p[d] for p in sv}:
                ns = sum(1 for p in sv if p[d] == v)
                nt = sum(1 for p in te if p[d] == v)
                if nt and not ns:
                    raise ValueError(
                        f"pitch {tier} W{w} ({ck}): {d}={v} on {nt}/{len(te)} "
                        f"test items but 0/{len(sv)} serving items — tested "
                        f"but never priced by feedback")
                gap = abs(ns / len(sv) - nt / len(te))
                if gap > MAX_DRAW_GAP:
                    raise ValueError(
                        f"pitch {tier} W{w} ({ck}): {d}={v} serving "
                        f"{ns/len(sv)*100:.1f}% vs test {nt/len(te)*100:.1f}% "
                        f"— gap {gap*100:.1f}pp over {MAX_DRAW_GAP*100:.0f}pp")
        # general quota: the pre-reserved exact set from pass 1
        te += gen_by_w[w]
        t0 = w * R.PER_WINDOW_SERVING + 1
        test_by_w[w] = [_task(tier, w, t0, p, ck, "te") for p in te]
        used_ids.update(p["product_id"] for p in sv)
        used_ids.update(p["product_id"] for p in te)
        served_combos.setdefault(ck, set()).update(sv_combos)

    meta = {"scenario": f"pitch_{tier.lower()}", "domain": "pitch", "tier": tier,
            "system_prompt": SYSTEM_PROMPT, "combo_stats": combo_stats}
    return {
        "meta": meta, "id": f"pitch_{tier}",
        "load_data": lambda: {},
        "tools": [],
        "policy_md": "", "workflow_md": "", "docs_toc": "",
        "serving": serving,
        "test_set": lambda w: test_by_w.get(w, []),
        "rules": rules,
        # rubric checklist scored in code, no baseline pitch
        "verifier": RubricVerifier(),
    }
