"""The generated scenarios are exactly the ones evaluated in the paper."""

import json
from pathlib import Path

import pytest

import servelearnbench
from servelearnbench.fingerprint import fingerprint
from servelearnbench.scenarios import SCENARIOS

PINNED = json.loads((Path(servelearnbench.__file__).parent / "fingerprints.json").read_text())

SCALE = {  # (windows, serving tasks, test tasks), as in the paper
    "retail_l1": (6, 720, 384), "retail_l2": (7, 810, 765), "retail_l3": (8, 624, 537),
    "banking_l1": (4, 336, 178), "banking_l2": (5, 500, 387), "banking_l3": (6, 600, 551),
    "pitch_l1": (4, 216, 96), "pitch_l2": (6, 324, 144), "pitch_l3": (7, 378, 168),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_fingerprint(name):
    fp = fingerprint(name)
    assert (fp["n_windows"], fp["n_serving"], fp["n_test"]) == SCALE[name]
    assert fp == PINNED[name]
