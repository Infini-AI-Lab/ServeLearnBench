"""Replaying a task's ground-truth actions scores 1; doing nothing scores 0
whenever the correct outcome is not "do nothing"."""

import random

import pytest

from servelearnbench.agent import new_env
from servelearnbench.engine.types import Action
from servelearnbench.scenarios import get_bundle, n_windows

SCENARIOS = [f"{d}_l{t}" for d in ("retail", "banking") for t in (1, 2, 3)]


def gt_finish(task) -> Action:
    a = task.answer or {}
    if a.get("kind") == "refuse":
        return Action(name="finish", kwargs={"decision": "refuse", "reason": a["reasons"][0]})
    if a.get("kind") == "value":
        return Action(name="finish", kwargs={"answer": a["expect"]})
    return Action(name="finish", kwargs={})


def sample(bundle, k=40, seed=0):
    tasks = list(bundle["serving"])
    for w in range(n_windows(bundle)):
        tasks += bundle["test_set"](w)
    return random.Random(seed).sample(tasks, min(k, len(tasks)))


@pytest.mark.parametrize("name", SCENARIOS)
def test_ground_truth_scores_one(name):
    b = get_bundle(name)
    for task in sample(b):
        env = new_env(b)
        env.reset(task)
        for a in task.actions:
            env.step(a)
        env.step(gt_finish(task))
        r = env.compute_reward()
        assert r.reward == 1.0, (task.task_id, r.answer_detail)


@pytest.mark.parametrize("name", SCENARIOS)
def test_doing_nothing_scores_zero(name):
    b = get_bundle(name)
    for task in sample(b, seed=1):
        if not task.actions and not task.answer:
            continue   # the correct outcome is to do nothing
        env = new_env(b)
        env.reset(task)
        env.step(Action(name="finish", kwargs={}))
        assert env.compute_reward().reward == 0.0, task.task_id
