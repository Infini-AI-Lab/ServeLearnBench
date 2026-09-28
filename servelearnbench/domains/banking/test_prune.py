"""Deterministically remove redundant banking test tickets.

The banking generators intentionally build thick raw grids for boundary,
scope, and leakage checks. Exposing every raw control in the scored test set,
however, would let both persistent adapt rows and the general slice dominate
the headline denominator.  This module keeps every true change/reopen probe,
caps repeated persistent-adapt rows, and downsamples semantically repeated
general controls.

After this pass, ``manual_slice`` (and compatibility alias ``source_slice``)
records current-truth disagreement with the frozen-manual baseline.  Final
``slice=adapt`` is broader: it denotes a scored temporal-learning probe
(manual disagreement, current-window change, reopen, or reclose).  Final
``slice=general`` means both manual agreement and no current-window policy
transition.
"""

from __future__ import annotations

import hashlib
import itertools
import math

from collections import Counter, defaultdict
from functools import reduce
from math import gcd

# Upper bound on the starting-point search in balanced_shell_assignment.
MAX_SHELL_SEARCH = 50_000


_SEMANTIC_FIELDS = (
    "family",
    "kind",
    "decision",
    "reason",
    "category",
    "region",
    "standing",
)


def _stable_key(seed: str, task) -> str:
    return hashlib.sha256(f"{seed}|{task.task_id}".encode()).hexdigest()


def _semantic_key(task):
    return tuple(task.z.get(field) for field in _SEMANTIC_FIELDS)


def _stratified_keep(
    tasks,
    budget: int,
    *,
    seed: str,
    protected_ids=None,
    prefer_composite: bool = False,
):
    """Select a deterministic, family-balanced subset of ``tasks``."""
    if budget >= len(tasks):
        return list(tasks)
    if budget <= 0:
        return []
    groups = defaultdict(list)
    for task in tasks:
        groups[_semantic_key(task)].append(task)
    for group in groups.values():
        # Prefer single-cause rows when thinning persistent adapt. Broad
        # multi_path rows can be labelled for one family while a higher
        # precedence rule actually decides the verdict.
        group.sort(key=lambda task: (
            (-bool(task.z.get("multi_path")) if prefer_composite
             else bool(task.z.get("multi_path"))),
            (-task.z.get("n_firing", 0) if prefer_composite
             else task.z.get("n_firing", 0)),
            _stable_key(seed, task),
        ))
    ordered = sorted(groups.values(), key=lambda group: (
        (-bool(group[0].z.get("multi_path")) if prefer_composite
         else bool(group[0].z.get("multi_path"))),
        (-group[0].z.get("n_firing", 0) if prefer_composite
         else group[0].z.get("n_firing", 0)),
        _stable_key(seed, group[0]),
    ))

    protected_ids = protected_ids or set()
    selected = [task for task in tasks if task.task_id in protected_ids]
    selected_ids = {task.task_id for task in selected}
    if len(selected) > budget:
        raise AssertionError(
            f"persistent witnesses need {len(selected)} slots, budget {budget}"
        )
    # Preserve family coverage before adding repeated semantic cells.
    for family in sorted({task.z.get("family") for task in tasks}, key=str):
        if len(selected) == budget:
            break
        if any(task.z.get("family") == family for task in selected):
            continue
        group = next(group for group in ordered
                     if group[0].z.get("family") == family)
        task = next((task for task in group if task.task_id not in selected_ids), None)
        if task is not None:
            selected.append(task)
            selected_ids.add(task.task_id)
    remaining_groups = [
        [task for task in group if task.task_id not in selected_ids]
        for group in ordered
    ]
    depth = 0
    while len(selected) < budget:
        added = False
        for group in remaining_groups:
            if depth < len(group):
                task = group[depth]
                selected.append(task)
                selected_ids.add(task.task_id)
                added = True
                if len(selected) == budget:
                    break
        if not added:
            break
        depth += 1
    if len(selected) != budget:
        raise AssertionError((len(selected), budget))
    return selected


def _stratified_keep_w(window, *args, **kwargs):
    """`_stratified_keep` with the window in the message: the adapt targets
    are per window and a bare failure does not say which one to move."""
    try:
        return _stratified_keep(*args, **kwargs)
    except AssertionError as exc:
        raise AssertionError(f"W{window}: {exc}") from exc


def balance_change_adapt(
    test_by_w,
    *,
    rules,
    data,
    verdict_fn,
    seed: str,
    retention_budget_by_window=None,
    protected_persistent_task_ids_by_window=None,
    target_adapt_by_window=None,
    prefer_composite: bool = False,
):
    """Make the scored adapt slice measure policy learning over time.

    Classification is counterfactual:

    - W0 docs-divergent rows are ``discovery``;
    - a W>0 row whose verdict changes relative to W-1 is ``change``;
    - rows already labelled reopen are folded into adapt as ``reopen``;
    - docs-divergent rows unchanged from W-1 are ``persistent``.

    Every change/reopen row is retained.  In event windows, persistent rows
    are capped at the hard-row count, guaranteeing that at least half of the
    exposed post-W0 adapt slice is change-sensitive.  Explicit consolidation
    windows may retain a small, declared retention sample.
    """
    retention_budget_by_window = retention_budget_by_window or {}
    protected_persistent_task_ids_by_window = (
        protected_persistent_task_ids_by_window or {}
    )
    target_adapt_by_window = target_adapt_by_window or {}
    out = {}
    for window, tasks in sorted(test_by_w.items()):
        discovery, hard, persistent, general = [], [], [], []
        current = rules.truth(window)
        previous = rules.truth(window - 1) if window else None
        for task in tasks:
            original_slice = task.z.get("slice")
            task.z["source_slice"] = task.z.get(
                "manual_slice", original_slice
            )
            task.z["adapt_kind"] = "general"
            if window == 0:
                if original_slice == "adapt":
                    task.z["adapt_kind"] = "discovery"
                    discovery.append(task)
                else:
                    general.append(task)
                continue

            changed = verdict_fn(task, current, data) != verdict_fn(
                task, previous, data
            )
            if original_slice == "reopen":
                task.z["adapt_kind"] = "reopen"
                hard.append(task)
            elif changed:
                task.z["adapt_kind"] = "change"
                hard.append(task)
            elif original_slice == "adapt":
                task.z["adapt_kind"] = "persistent"
                persistent.append(task)
            else:
                general.append(task)

        target = target_adapt_by_window.get(window)
        if window == 0:
            budget = len(discovery) if target is None else target
            if budget > len(discovery):
                raise AssertionError(
                    f"W{window}: adapt target {budget} exceeds "
                    f"{len(discovery)} discovery rows"
                )
            kept_adapt = _stratified_keep(
                discovery,
                budget,
                seed=f"{seed}:discovery:W{window}",
                protected_ids=protected_persistent_task_ids_by_window.get(
                    window, set()
                ),
                prefer_composite=prefer_composite,
            )
        elif hard:
            budget = (min(len(persistent), len(hard)) if target is None
                      else target - len(hard))
            if budget < 0 or budget > len(persistent):
                raise AssertionError(
                    f"W{window}: adapt target {target} cannot retain "
                    f"{len(hard)} change rows from {len(persistent)} "
                    f"persistent rows"
                )
            kept_adapt = hard + _stratified_keep_w(
                window,
                persistent, budget,
                seed=f"{seed}:persistent:W{window}",
                protected_ids=protected_persistent_task_ids_by_window.get(
                    window, set()
                ),
                prefer_composite=prefer_composite,
            )
        else:
            budget = min(len(persistent), retention_budget_by_window.get(window, 0))
            kept_adapt = _stratified_keep(
                persistent, budget, seed=f"{seed}:retention:W{window}",
                protected_ids=protected_persistent_task_ids_by_window.get(
                    window, set()
                ),
                prefer_composite=prefer_composite,
            )
            for task in kept_adapt:
                task.z["adapt_kind"] = "retention"

        for task in kept_adapt:
            task.z["slice"] = "adapt"
        for task in general:
            task.z["slice"] = "general"
        out[window] = kept_adapt + general
    return out


def prune_general_repetitions(
    test_by_w,
    *,
    target_per_window,
    seed: str,
    protected_families_by_window=None,
    preferred_families_by_window=None,
    protected_task_ids_by_window=None,
    redundant_only: bool = False,
    prefer_composite: bool = False,
    numeric_specs=None,
    numeric_min_each: int = 2,
    numeric_min_fraction: float = 0.20,
    mark_controls: bool = False,
):
    """Keep every adapt/reopen ticket and stratify the remaining general set.

    Spare slots are filled round-robin across coarse semantic cells,
    preventing a large family from consuming the entire general budget.
    With ``redundant_only=True``, every semantic cell present in the input is
    mandatory and the function fails closed rather than delete unique
    coverage.  This mode is intended as a second pass over an already pruned
    scored set.
    """
    out = {}
    protected_families_by_window = protected_families_by_window or {}
    preferred_families_by_window = preferred_families_by_window or {}
    protected_task_ids_by_window = protected_task_ids_by_window or {}
    for window, tasks in sorted(test_by_w.items()):
        target = (target_per_window if isinstance(target_per_window, int)
                  else target_per_window[window])
        fixed = [task for task in tasks if task.z.get("slice") != "general"]
        general = [task for task in tasks if task.z.get("slice") == "general"]
        general_budget = target - len(fixed)
        if general_budget < 0:
            raise AssertionError(
                f"W{window}: target {target} is smaller than the "
                f"{len(fixed)} adapt/reopen tickets"
            )

        groups = defaultdict(list)
        for task in general:
            groups[_semantic_key(task)].append(task)
        for group in groups.values():
            group.sort(key=lambda task: (
                -task.z.get("n_firing", 0) if prefer_composite else 0,
                -bool(task.z.get("multi_path")) if prefer_composite else 0,
                _stable_key(seed, task),
            ))

        if len({key[0] for key in groups}) > general_budget:
            raise AssertionError(
                f"W{window}: general budget {general_budget} cannot retain "
                "every family"
            )

        ordered_groups = sorted(
            groups.values(), key=lambda group: (
                -group[0].z.get("n_firing", 0) if prefer_composite else 0,
                -bool(group[0].z.get("multi_path")) if prefer_composite else 0,
                _stable_key(seed, group[0]),
            )
        )
        protected_families = protected_families_by_window.get(window, set())
        protected_ids = protected_task_ids_by_window.get(window, set())
        selected = [
            task for task in general
            if (task.z.get("family") in protected_families
                or task.task_id in protected_ids)
        ]
        selected_ids = {task.task_id for task in selected}
        mandatory_ids = set(selected_ids)
        if len(selected) > general_budget:
            raise AssertionError(
                f"W{window}: protected controls need {len(selected)} slots "
                f"but the general budget is {general_budget}"
            )
        # A numeric family whose minority side is simply small cannot absorb
        # an unbounded majority side: past the cap the filler would push the
        # 20% contract under water with rows nothing else needs.
        fill_cap = {}
        for family, value_fn, threshold_fn, applies_fn in (numeric_specs or ()):
            threshold = threshold_fn(window)
            rows = [task for task in tasks
                    if task.z.get("family") == family
                    and applies_fn(task, window)]
            possible = Counter(value_fn(task, window) > threshold
                               for task in rows)
            if len(possible) < 2:
                continue
            for side in (False, True):
                fill_cap[(family, side)] = max(
                    numeric_min_each,
                    int(possible[not side] * (1.0 - numeric_min_fraction)
                        / numeric_min_fraction),
                )

        def _over_cap(task, exposed):
            for family, value_fn, threshold_fn, applies_fn in (
                    numeric_specs or ()):
                if task.z.get("family") != family:
                    continue
                if not applies_fn(task, window):
                    continue
                side = value_fn(task, window) > threshold_fn(window)
                cap = fill_cap.get((family, side))
                if cap is not None and exposed[(family, side)] >= cap:
                    return True
            return False

        def _count_exposed(chosen_ids):
            seen = Counter()
            for family, value_fn, threshold_fn, applies_fn in (
                    numeric_specs or ()):
                threshold = threshold_fn(window)
                for task in tasks:
                    if (task.z.get("family") != family
                            or not applies_fn(task, window)):
                        continue
                    if (task.z.get("slice") == "general"
                            and task.task_id not in chosen_ids):
                        continue
                    seen[(family, value_fn(task, window) > threshold)] += 1
            return seen

        # The first pass only guarantees family coverage.
        # A second, redundant-only pass can additionally require every
        # semantic cell in its input to survive.
        families = sorted({task.z.get("family") for task in general})
        for family in families:
            if any(task.z.get("family") == family for task in selected):
                continue
            group = next(
                group for group in ordered_groups
                if group[0].z.get("family") == family
            )
            selected.append(group[0])
            selected_ids.add(group[0].task_id)
        exposed = _count_exposed(selected_ids) if numeric_specs else Counter()
        for group in ordered_groups:
            if ((redundant_only or len(selected) < general_budget)
                    and not any(task.task_id in selected_ids for task in group)):
                pick = (next((task for task in group
                              if not _over_cap(task, exposed)), None)
                        if numeric_specs else group[0])
                if pick is None:
                    continue
                selected.append(pick)
                selected_ids.add(pick.task_id)
                exposed = _count_exposed(selected_ids) if numeric_specs else exposed

        if redundant_only and len(selected) > general_budget:
            raise AssertionError(
                f"W{window}: non-redundant coverage needs {len(selected)} "
                f"general slots but the budget is {general_budget}"
            )

        remaining_groups = [
            [task for task in group if task.task_id not in selected_ids]
            for group in ordered_groups
        ]
        preferred_families = preferred_families_by_window.get(window, set())
        remaining_groups.sort(
            key=lambda group: (
                0 if group and group[0].z.get("family") in preferred_families else 1,
                _stable_key(seed, group[0]) if group else "",
            )
        )
        depth = 0
        while len(selected) < general_budget:
            added = False
            for group in remaining_groups:
                if depth < len(group):
                    task = group[depth]
                    if numeric_specs and (task.task_id in selected_ids
                                          or _over_cap(task, exposed)):
                        continue
                    selected.append(task)
                    selected_ids.add(task.task_id)
                    if numeric_specs:
                        exposed = _count_exposed(selected_ids)
                    added = True
                    if len(selected) == general_budget:
                        break
            if not added:
                break
            depth += 1

        if len(selected) != general_budget:
            raise AssertionError(
                f"W{window}: only {len(selected)} general tickets available "
                f"for budget {general_budget}"
            )

        selected_ids = {task.task_id for task in selected}

        # Filling the spare slots draws from the raw distribution, so a
        # numeric family that was balanced among the protected rows can be
        # skewed back below the 20% contract by the filler.  Repair by
        # swapping filler rows on the majority side for unselected rows on
        # the minority side; both sides keep the same slot count, so the
        # window total is unchanged.
        for family, value_fn, threshold_fn, applies_fn in (numeric_specs or ()):
            threshold = threshold_fn(window)
            rows = [task for task in tasks
                    if task.z.get("family") == family
                    and applies_fn(task, window)]
            for _ in range(len(rows)):
                shown = [task for task in rows
                         if task.z.get("slice") != "general"
                         or task.task_id in selected_ids]
                counts = Counter(value_fn(task, window) > threshold
                                 for task in shown)
                total = counts[False] + counts[True]
                if (counts[False] >= numeric_min_each
                        and counts[True] >= numeric_min_each
                        and min(counts.values()) / max(total, 1)
                            >= numeric_min_fraction):
                    break
                minority = counts[True] <= counts[False]
                add = [task for task in rows
                       if task.z.get("slice") == "general"
                       and task.task_id not in selected_ids
                       and (value_fn(task, window) > threshold) == minority]
                cells = Counter(_semantic_key(task) for task in selected)
                drop = [task for task in selected
                        if task.task_id not in mandatory_ids
                        and cells[_semantic_key(task)] > 1
                        and (task.z.get("family") != family
                             or (value_fn(task, window) > threshold)
                             != minority)]
                if not add or not drop:
                    break
                gained = min(add, key=lambda task: _stable_key(
                    f"{seed}:{family}:W{window}:repair", task))
                lost = max(drop, key=lambda task: _stable_key(
                    f"{seed}:{family}:W{window}:repair", task))
                selected = [task for task in selected
                            if task.task_id != lost.task_id] + [gained]
                selected_ids = {task.task_id for task in selected}

        if mark_controls:
            # Record WHICH general rows were required, so the general share
            # can be split into coverage and repetition.
            for task in tasks:
                if task.z.get("slice") == "general":
                    task.z["control"] = task.task_id in mandatory_ids
        exposed = [
            task for task in tasks
            if task.z.get("slice") != "general" or task.task_id in selected_ids
        ]
        if len(exposed) != target:
            raise AssertionError((window, len(exposed), target))
        if {task.task_id for task in fixed} - {task.task_id for task in exposed}:
            raise AssertionError(f"W{window}: non-general ticket was removed")
        out[window] = exposed
    return out


def event_general_probe_ids(test_by_w, *, rules, data, verdict_fn):
    """General tickets that are required to measure a drift event.

    An event comparison stays live until the next event on the same policy.
    These tickets may be general relative to the printed docs while still
    being indispensable pre/post-event counterfactual probes.
    """
    protected = defaultdict(set)
    for event_window, key, _ in rules.EVENTS:
        end = min(
            (
                window
                for window, later_key, _ in rules.EVENTS
                if later_key == key and window > event_window
            ),
            default=rules.N_WINDOWS,
        )
        for window in range(event_window, end):
            post = rules.truth(window)
            pre = dict(post)
            pre[key] = rules.truth(event_window - 1)[key]
            for task in test_by_w[window]:
                if task.z.get("slice") != "general":
                    continue
                if verdict_fn(task, pre, data) != verdict_fn(task, post, data):
                    protected[window].add(task.task_id)
    return protected


def categorical_alternation_probe_ids(test_by_w, *, specs, seed: str):
    """Smallest general subsets that preserve two hidden-field rank flips.

    ``specs`` entries are ``(family, field, band_predicates)``. A ``None``
    band list means the whole family; otherwise every band is protected
    independently. Adapt/reopen rows are treated as already mandatory.
    """
    protected = defaultdict(set)
    for window, tasks in sorted(test_by_w.items()):
        for family, field, band_predicates in specs:
            predicates = band_predicates or (lambda task: True,)
            for in_band in predicates:
                raw = [
                    task for task in tasks
                    if task.z.get("family") == family and in_band(task)
                ]
                if len({task.z.get(field) for task in raw}) < 2:
                    continue
                fixed = [task for task in raw if task.z.get("slice") != "general"]
                candidates = sorted(
                    (task for task in raw if task.z.get("slice") == "general"),
                    key=lambda task: _stable_key(seed, task),
                )

                def enough(extra):
                    rows = sorted(fixed + list(extra), key=lambda task: task.z["amount"])
                    values = [task.z.get(field) for task in rows]
                    if not values:
                        return False
                    flips_ok = sum(
                        values[i] != values[i - 1] for i in range(1, len(values))
                    ) >= 2
                    return flips_ok

                chosen = None
                for size in range(len(candidates) + 1):
                    for combo in itertools.combinations(candidates, size):
                        if enough(combo):
                            chosen = combo
                            break
                    if chosen is not None:
                        break
                if chosen is None:
                    raise AssertionError(
                        f"W{window} {family}.{field}: raw test cannot preserve "
                        "two amount-rank alternations"
                    )
                protected[window].update(task.task_id for task in chosen)
    return protected


def alternation_witness_ids(test_by_w, *, specs):
    """Return minimal all-slice witnesses for two rank alternations.

    Unlike ``categorical_alternation_probe_ids``, this identifies both adapt
    and general rows that must survive earlier adapt thinning.
    """
    witnesses = defaultdict(set)
    for window, tasks in sorted(test_by_w.items()):
        for family, field, band_predicates in specs:
            predicates = band_predicates or (lambda task: True,)
            for in_band in predicates:
                rows = sorted(
                    (task for task in tasks
                     if task.z.get("family") == family and in_band(task)),
                    key=lambda task: task.z["amount"],
                )

                def enough(combo):
                    values = [task.z.get(field) for task in combo]
                    return (len(set(values)) >= 2
                            and sum(values[i] != values[i - 1]
                                    for i in range(1, len(values))) >= 2)

                chosen = None
                for size in range(3, len(rows) + 1):
                    for combo in itertools.combinations(rows, size):
                        if enough(combo):
                            chosen = combo
                            break
                    if chosen is not None:
                        break
                if chosen is None:
                    # A uniform band carries no condition/amount alias.
                    if len({task.z.get(field) for task in rows}) < 2:
                        continue
                    raise AssertionError(
                        f"W{window} {family}.{field}: no alternation witness"
                    )
                witnesses[window].update(task.task_id for task in chosen)
    return witnesses


def value_coverage_probe_ids(test_by_w, *, specs, seed: str):
    """Protect the smallest general set that gives each family two values.

    ``specs`` entries are ``(family, field)``. Existing adapt rows count
    toward coverage; general rows are added only for missing values.
    """
    protected = defaultdict(set)
    for window, tasks in sorted(test_by_w.items()):
        for family, field in specs:
            rows = [task for task in tasks if task.z.get("family") == family]
            values = {task.z.get(field) for task in rows}
            if len(values) < 2:
                raise AssertionError(
                    f"W{window} {family}.{field}: raw test has one value {values}"
                )
            fixed_values = {
                task.z.get(field) for task in rows
                if task.z.get("slice") != "general"
            }
            chosen_values = set(fixed_values)
            candidates = sorted(
                (task for task in rows if task.z.get("slice") == "general"),
                key=lambda task: _stable_key(seed, task),
            )
            for task in candidates:
                value = task.z.get(field)
                if value in chosen_values:
                    continue
                protected[window].add(task.task_id)
                chosen_values.add(value)
                if len(chosen_values) >= 2:
                    break
            if len(chosen_values) < 2:
                raise AssertionError(
                    f"W{window} {family}.{field}: cannot preserve two values"
                )
    return protected


def constant_answer_balance_ids(
    test_by_w,
    *,
    families,
    seed: str,
    max_rate: float = 0.70,
    preselected_by_window=None,
):
    """Protect general counterexamples against family-constant answers.

    Adapt/change rows are mandatory and can be strongly one-sided.  A plain
    semantic-cell sampler can then discard nearly every opposite outcome,
    making an over-general rule cheap.  For each requested family, retain the
    smallest deterministic set of general rows needed to bring the best
    constant answer below ``max_rate`` before spare slots are filled.

    Selection prefers the currently least represented outcome and spreads
    additions across windows.  This helper is a construction aid; it does
    not validate the final exposed set.
    """
    if not 0.5 < max_rate < 1.0:
        raise ValueError("max_rate must be between 0.5 and 1.0")
    protected = defaultdict(set)
    preselected_by_window = preselected_by_window or {}
    all_tasks = [task for tasks in test_by_w.values() for task in tasks]
    for family in families:
        rows = [task for task in all_tasks
                if task.z.get("family") == family]
        possible_answers = {
            (task.z.get("decision"), task.z.get("reason")) for task in rows
        }
        # A policy that is genuinely inactive in this window can make the
        # whole family one-outcome. There is no opposite answer to preserve.
        if len(possible_answers) < 2:
            continue
        preselected = {
            task_id
            for window, task_ids in preselected_by_window.items()
            for task_id in task_ids
        }
        counts = Counter(
            (task.z.get("decision"), task.z.get("reason"))
            for task in rows
            if (task.z.get("slice") != "general"
                or task.task_id in preselected)
        )
        candidates = [task for task in rows
                      if (task.z.get("slice") == "general"
                          and task.task_id not in preselected)]
        chosen_by_window = Counter()

        def rate():
            total = sum(counts.values())
            return max(counts.values(), default=0) / max(total, 1)

        while rate() > max_rate and candidates:
            candidates.sort(key=lambda task: (
                counts[(task.z.get("decision"), task.z.get("reason"))],
                chosen_by_window[task.z.get("window")],
                _stable_key(f"{seed}:{family}:constant", task),
            ))
            task = candidates.pop(0)
            answer = (task.z.get("decision"), task.z.get("reason"))
            counts[answer] += 1
            chosen_by_window[task.z.get("window")] += 1
            protected[task.z.get("window")].add(task.task_id)
        if rate() > max_rate:
            raise AssertionError(
                f"{family}: raw general controls cannot cap the best "
                f"constant answer at {max_rate:.0%}"
            )
    return protected


def per_window_constant_answer_balance_ids(
    test_by_w,
    *,
    families,
    seed: str,
    max_rate: float = 0.75,
    preselected_by_window=None,
    active_windows_by_family=None,
):
    """Window-local counterpart of :func:`constant_answer_balance_ids`."""
    out = defaultdict(set)
    preselected_by_window = preselected_by_window or {}
    active_windows_by_family = active_windows_by_family or {}
    for window, tasks in sorted(test_by_w.items()):
        active_families = tuple(
            family for family in families
            if (family not in active_windows_by_family
                or window in active_windows_by_family[family])
        )
        if not active_families:
            continue
        try:
            chosen = constant_answer_balance_ids(
                {window: tasks},
                families=active_families,
                seed=f"{seed}:W{window}",
                max_rate=max_rate,
                preselected_by_window={
                    window: preselected_by_window.get(window, set())
                },
            )
        except AssertionError as exc:
            raise AssertionError(f"W{window}: {exc}") from exc
        out[window].update(chosen[window])
    return out


def categorical_value_adapt_ids(
    test_by_w,
    *,
    specs,
    seed: str,
    min_each: int = 2,
):
    """Adapt rows the categorical-coverage contract cannot afford to lose.

    The counterpart of :func:`numeric_side_adapt_ids` for value coverage and
    alternation: when a field value is carried only by docs-divergent rows
    (W3 overseas electronics, say), thinning the adapt slice can erase the
    value entirely, and the probe builders that run afterwards have nothing
    left to select from.  ``specs`` entries are ``(family, field)``.
    """
    protected = defaultdict(set)
    for window, tasks in sorted(test_by_w.items()):
        for family, field in specs:
            by_value = defaultdict(list)
            for task in tasks:
                if task.z.get("family") == family:
                    by_value[task.z.get(field)].append(task)
            for group in by_value.values():
                have = sum(1 for task in group
                           if task.z.get("slice") == "general")
                reserve = sorted(
                    (task for task in group
                     if task.z.get("slice") != "general"),
                    key=lambda task: _stable_key(
                        f"{seed}:{family}:{field}:W{window}:value", task
                    ),
                )[:max(0, min_each - have)]
                protected[window].update(task.task_id for task in reserve)
    return protected


def multipart_control_ids(test_by_w, *, rules, families, min_probes=3,
                          min_multi_code=1):
    """The multi-part GENERAL rows that are still needed by name.

    Multi-part general rows carry execution length and no hidden information,
    so they are prunable as a class -- but a few of them are load-bearing:
    batches need line-vs-total probes in every window, every stop position
    must be present, and the accepted-set contract needs cases whose blocked
    legs disagree on the code.  Protect those by id and
    let the rest go, instead of protecting the whole family and spending the
    general budget on tickets nothing asks for.
    """
    protected = defaultdict(set)
    for window, tasks in sorted(test_by_w.items()):
        general = [task for task in tasks
                   if task.z.get("slice") == "general"
                   and task.z.get("family") in families]
        threshold = rules.truth(window)["C_reporting_threshold"]
        probes = [task for task in general if task.z.get("batch")
                  and math.fsum(item["amount"] for item in task.z["lines"]) > threshold
                  and all(item["amount"] <= threshold
                          for item in task.z["lines"])]
        protected[window].update(t.task_id for t in probes[:min_probes])
        seen = set()
        for task in general:
            if not task.z.get("fallback"):
                continue
            stop = task.z.get("chosen_leg")
            if stop not in seen:
                seen.add(stop)
                protected[window].add(task.task_id)
        codes = [t for t in general
                 if t.z.get("fallback") and len(t.z.get("accepted_reasons") or []) > 1]
        protected[window].update(t.task_id for t in codes[:min_multi_code])
    return protected


def decisive_values(task, value_of, applies):
    """The amounts on a ticket that actually decide its graded answer.

    A single-request ticket contributes its own amount.  A ranked-alternative
    case contributes the leg that was carried out plus every leg ahead of it
    -- those were refused, and each refusal is a threshold comparison the
    answer depends on; legs BEHIND the executed one are never evaluated and
    prove nothing.  A payment batch contributes every line, since each line
    enters the graded answer independently.

    A multi-part case has no `z["amount"]` or `z["family"]`; counting its
    decisive amounts here keeps multi-part forms visible to every balance
    requirement where their legs straddle the boundary.
    """
    z = task.z
    if z.get("batch"):
        return [value_of(item) for item in z["lines"] if applies(item)]
    if z.get("fallback"):
        chosen = z.get("chosen_leg")
        legs = z["legs"][:chosen] if chosen else z["legs"]
        return [value_of(item) for item in legs if applies(item)]
    return []


def numeric_side_adapt_ids(
    test_by_w,
    *,
    specs,
    seed: str,
    min_each: int = 2,
    construction_min_fraction: float = 0.20,
):
    """Adapt rows the numeric-side contract cannot afford to lose.

    ``balance_change_adapt`` thins persistent adapt rows blind to numeric
    balance.  When one side of a threshold lives almost entirely in the adapt
    slice -- W3 travel, where every above-limit row is docs-divergent -- that
    thinning can leave the minority side too small for
    ``numeric_side_balance_ids`` to repair out of the general pool, and the
    only symptom is a raw-grid failure several stages later.  Reserve the
    minority side up front, sized off the raw majority so the 20% contract
    still has room once pruning is done.
    """
    protected = defaultdict(set)
    frac = construction_min_fraction
    for window, tasks in sorted(test_by_w.items()):
        for family, value_fn, threshold_fn, applies_fn in specs:
            rows = [task for task in tasks
                    if task.z.get("family") == family
                    and applies_fn(task, window)]
            threshold = threshold_fn(window)
            sides = defaultdict(list)
            for task in rows:
                sides[value_fn(task, window) > threshold].append(task)
            if len(sides) < 2:
                continue
            # Both sides, not just the smaller one: which side is scarce
            # after thinning is not knowable here, and a tie would otherwise
            # pick arbitrarily.
            for side in (False, True):
                need = max(min_each, math.ceil(
                    len(sides[not side]) * frac / (1.0 - frac)))
                have = sum(1 for task in sides[side]
                           if task.z.get("slice") == "general")
                reserve = sorted(
                    (task for task in sides[side]
                     if task.z.get("slice") != "general"),
                    key=lambda task: _stable_key(
                        f"{seed}:{family}:W{window}:side", task
                    ),
                )[:max(0, need - have)]
                protected[window].update(task.task_id for task in reserve)
    return protected


def numeric_side_balance_ids(
    test_by_w,
    *,
    specs,
    seed: str,
    preselected_by_window=None,
    min_each: int = 2,
    construction_min_fraction: float = 0.20,
    proportional: bool = False,
):
    """Protect both sides of each scored numeric threshold per window.

    ``specs`` entries are ``(family, value_fn, threshold_fn, applies_fn)``.
    The optional applicability predicate excludes rows that do not exercise
    the threshold (for example an old payee in the recent-payee family).
    Construction uses the public 20% contract.
    """
    preselected_by_window = preselected_by_window or {}
    protected = defaultdict(set)
    for window, tasks in sorted(test_by_w.items()):
        preselected = set(preselected_by_window.get(window, set()))
        for family, value_fn, threshold_fn, applies_fn in specs:
            rows = [task for task in tasks
                    if task.z.get("family") == family
                    and applies_fn(task, window)]
            threshold = threshold_fn(window)
            counts = Counter()
            for task in rows:
                if (task.z.get("slice") != "general"
                        or task.task_id in preselected):
                    counts[value_fn(task, window) > threshold] += 1
            candidates = [task for task in rows
                          if task.z.get("slice") == "general"
                          and task.task_id not in preselected]

            # Size each side against the RAW opposite side, not against
            # what is protected so far: the general filler later draws from
            # the raw distribution, so a minimal 2-and-2 protection is
            # diluted straight back below the contract.
            frac = construction_min_fraction
            raw = Counter(value_fn(task, window) > threshold for task in rows)

            def need(side):
                return max(min_each,
                           math.ceil(raw[not side] * frac / (1.0 - frac)))

            def complete():
                total = counts[False] + counts[True]
                return (counts[False] >= (min(need(False), raw[False])
                                          if proportional else min_each)
                        and counts[True] >= (min(need(True), raw[True])
                                             if proportional else min_each)
                        and counts[False] >= min_each
                        and counts[True] >= min_each
                        and min(counts.values()) / max(total, 1) >= frac)

            for side in (False, True):
                quota = (min(need(side), raw[side]) if proportional
                         else min_each)
                side_rows = sorted(
                    (task for task in candidates
                     if (value_fn(task, window) > threshold) == side),
                    key=lambda item: _stable_key(
                        f"{seed}:{family}:W{window}:numeric", item),
                )
                for task in side_rows[:max(0, quota - counts[side])]:
                    candidates.remove(task)
                    counts[side] += 1
                    protected[window].add(task.task_id)
                    preselected.add(task.task_id)
            if not complete():
                raise AssertionError(
                    f"W{window} {family}: raw grid cannot preserve numeric "
                    f"sides at {dict(counts)}"
                )
    return protected


def align_clean_payee_anchors(
    serving,
    test_by_w,
    *,
    transfer_cases,
    payees,
    clean_payees,
    reporting_threshold_by_window,
    seed: str,
):
    """Give every scored clean test payee a direct same-window anchor.

    Restricted-payee membership is entity-level and is intentionally absent
    from the public payee tool.  A clean test entity must therefore occur in
    serving, below the reporting threshold, or its label is only identifiable
    through the generator shortcut "unseen payee means clean".

    This is a label-preserving reassignment among clean serving transfers.
    Required anchors are never displaced; when the target already occurs on
    a high-amount ticket, the two payees are swapped so entity coverage is
    preserved. Ticket counts, families, amounts, slices, and gold actions
    remain unchanged.
    """
    clean_payees = set(clean_payees)
    serving_by_w = defaultdict(list)
    for task in serving:
        serving_by_w[task.z["window"]].append(task)

    changed = []
    for window, tests in sorted(test_by_w.items()):
        required = {
            task.z.get("payee_id")
            for task in tests
            if task.z.get("kind") == "transfer"
            and task.z.get("payee_id") in clean_payees
        }
        threshold = reporting_threshold_by_window(window)
        anchors = [
            task for task in serving_by_w[window]
            if task.z.get("kind") == "transfer"
            and task.z.get("payee_id") in clean_payees
            and task.z.get("decision") == "approve"
            and task.z.get("amount") < threshold
        ]
        counts = Counter(task.z["payee_id"] for task in anchors)
        all_counts = Counter(
            task.z["payee_id"]
            for task in serving_by_w[window]
            if task.z.get("kind") == "transfer"
            and task.z.get("payee_id") in clean_payees
        )
        missing = sorted(required - set(counts))
        for new_payee_id in missing:
            donors = [
                task for task in anchors
                if (counts[task.z["payee_id"]] > 1
                    or task.z["payee_id"] not in required)
            ]
            if not donors:
                raise AssertionError(
                    f"W{window}: no redundant clean serving anchor available "
                    f"for {new_payee_id}"
                )
            donor = min(
                donors,
                key=lambda task: _stable_key(
                    f"{seed}:clean-anchor:{new_payee_id}", task
                ),
            )
            old_payee_id = donor.z["payee_id"]

            # If the missing low anchor already occurs elsewhere in this
            # window, swap rather than collapse two serving entities into one.
            counterpart = next(
                (
                    task for task in serving_by_w[window]
                    if task.z.get("kind") == "transfer"
                    and task.z.get("payee_id") == new_payee_id
                    and task is not donor
                ),
                None,
            )
            if counterpart is not None:
                transfer_cases[counterpart.z["case_id"]]["payee_id"] = old_payee_id
                counterpart.z["payee_id"] = old_payee_id
                counterpart.instruction = counterpart.instruction.replace(
                    payees[new_payee_id]["name"], payees[old_payee_id]["name"]
                ).replace(new_payee_id, old_payee_id)

            transfer_cases[donor.z["case_id"]]["payee_id"] = new_payee_id
            donor.z["payee_id"] = new_payee_id
            donor.instruction = donor.instruction.replace(
                payees[old_payee_id]["name"], payees[new_payee_id]["name"]
            ).replace(old_payee_id, new_payee_id)
            counts[old_payee_id] -= 1
            counts[new_payee_id] += 1
            if counterpart is None:
                all_counts[old_payee_id] -= 1
                all_counts[new_payee_id] += 1
            changed.append((window, donor.task_id, old_payee_id, new_payee_id))

        if required - set(counts):
            raise AssertionError(
                f"W{window}: clean payees still lack serving anchors: "
                f"{sorted(required - set(counts))}"
            )
    return changed


def balanced_shell_assignment(tasks, *, shell_count_by_kind, seed: str,
                              key_field: str = "kind",
                              decision_field: str = "decision"):
    """Assign shells so each one approximates its pool's verdict mix.

    Start with even per-verdict placement, consider proportional uneven shell
    sizes when integer ratios require them, then improve deterministically.

    ``key_field`` selects which z field names the shell pool. It defaults to
    "kind" (L1 behaviour); tiers whose ordered-fallback cases
    need their own sentences pass "shell_pool" so a fallback shell can never
    become a marker for the case form.

    ``decision_field`` selects what counts as the verdict being balanced. It
    defaults to "decision", but a multi-part case has a CONSTANT decision
    ("batch", or "execute" for any carried-out fallback) while its graded
    answer varies, so those tiers pass a field holding the full answer —
    balancing a constant would leave the real answer free to concentrate in
    one shell.
    """
    assignment = {}
    kinds = sorted({task.z[key_field] for task in tasks})
    for kind in kinds:
        pool = [task for task in tasks if task.z[key_field] == kind]
        decisions = sorted({task.z[decision_field] for task in pool}, key=str)
        n_shells = shell_count_by_kind[kind]
        counts = {
            decision: sum(task.z[decision_field] == decision for task in pool)
            for decision in decisions
        }
        choices = []
        for decision in decisions:
            remainder = counts[decision] % n_shells
            choices.append(list(itertools.combinations(range(n_shells), remainder)))
        # The exhaustive product below is only a SEARCH FOR A STARTING POINT;
        # the proportional allocation and the local search after it reach a
        # good assignment on their own. Its size is the product of the
        # per-verdict remainder placements, so it explodes once a pool has
        # many distinct verdicts — a multi-line batch has one verdict per
        # per-line outcome tuple, which can reach 35**12 combinations. Bound
        # it and fall back to the even allocation.
        product_size = 1
        for options in choices:
            product_size *= max(len(options), 1)
            if product_size > MAX_SHELL_SEARCH:
                break
        if product_size > MAX_SHELL_SEARCH:
            # one fixed placement, not the empty one: the remainder rows still
            # have to land somewhere or those tasks get no shell at all
            choices = [[tuple(range(counts[decision] % n_shells))]
                       for decision in decisions]

        def matrix_score(matrix):
            totals = [sum(matrix[d][s] for d in decisions)
                      for s in range(n_shells)]
            missing_mix = sum(
                sum(matrix[d][s] > 0 for d in decisions) < 2
                for s in range(n_shells)
            )
            majority = [
                (max(matrix[d][s] for d in decisions) / totals[s]
                 if totals[s] else 1.0)
                for s in range(n_shells)
            ]
            # math.fsum, not sum: this tuple is compared with `<`, and fsum
            # is correctly rounded, so ties between candidates do not depend
            # on summation order or the interpreter version.
            # The sum term lets local search cross a plateau where moving
            # one row improves one 3:1 shell while another 3:1 shell still
            # determines the same maximum.  Without it, the proportional
            # solution can be two individually useful moves away.
            return (missing_mix, max(majority), math.fsum(majority),
                    max(totals) - min(totals), tuple(totals))

        best = None
        for extra_sets in itertools.product(*choices):
            matrix = {decision: [counts[decision] // n_shells] * n_shells
                      for decision in decisions}
            for decision, extra in zip(decisions, extra_sets):
                for shell in extra:
                    matrix[decision][shell] += 1
            score = matrix_score(matrix)
            if best is None or score < best[0]:
                best = (score, matrix)

        matrix = best[1]
        # Use the smallest verdict as proportional shell units when it can
        # place at least one unit in every shell.  Largest-remainder
        # apportionment then mirrors ratios such as 15:7 as
        # 5:2, 4:2, 2:1, 2:1, 2:1 without an expensive integer search.
        anchor = min(decisions, key=lambda decision: counts[decision])
        anchor_total = counts[anchor]
        if anchor_total >= n_shells:
            loads = [anchor_total // n_shells] * n_shells
            for shell in range(anchor_total % n_shells):
                loads[shell] += 1
            proportional = {anchor: loads}
            for decision in decisions:
                if decision == anchor:
                    continue
                raw = [counts[decision] * load / anchor_total for load in loads]
                row = [int(value) for value in raw]
                remaining = counts[decision] - sum(row)
                order = sorted(range(n_shells),
                               key=lambda shell: (-(raw[shell] - row[shell]), shell))
                for shell in order[:remaining]:
                    row[shell] += 1
                proportional[decision] = row
            if matrix_score(proportional) < matrix_score(matrix):
                matrix = proportional
        # If the verdict totals share enough exact ratio units, distribute
        # whole units instead of individual rows.  For example 14:7 over
        # five shells becomes unit loads 2,2,1,1,1, giving every shell the
        # true 2:1 base rate; row-wise balancing would create 3:1 shells.
        ratio_units = reduce(gcd, counts.values())
        if ratio_units >= n_shells:
            loads = [ratio_units // n_shells] * n_shells
            for shell in range(ratio_units % n_shells):
                loads[shell] += 1
            ratio_matrix = {
                decision: [counts[decision] // ratio_units * load
                           for load in loads]
                for decision in decisions
            }
            if matrix_score(ratio_matrix) < matrix_score(matrix):
                matrix = ratio_matrix
        # Equal-per-verdict allocation is a good starting point, but small
        # two-verdict kinds can require a 2-count swing to avoid a 3:1 shell.
        # Deterministic local moves find that better integer allocation.
        while True:
            current_score = matrix_score(matrix)
            improvement = None
            for decision in decisions:
                for source in range(n_shells):
                    if matrix[decision][source] == 0:
                        continue
                    for dest in range(n_shells):
                        if source == dest:
                            continue
                        candidate = {d: list(row) for d, row in matrix.items()}
                        candidate[decision][source] -= 1
                        candidate[decision][dest] += 1
                        score = matrix_score(candidate)
                        if score < current_score and (
                                improvement is None or score < improvement[0]):
                            improvement = (score, candidate)
            if improvement is None:
                break
            matrix = improvement[1]
        for decision in decisions:
            group = sorted(
                (task for task in pool if task.z[decision_field] == decision),
                key=lambda task: _stable_key(seed, task),
            )
            offset = 0
            for shell in range(n_shells):
                take = matrix[decision][shell]
                for task in group[offset:offset + take]:
                    assignment[task.task_id] = shell
                offset += take
    return assignment
