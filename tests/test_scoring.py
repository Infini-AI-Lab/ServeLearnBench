from servelearnbench.domains.pitch.verifier import RubricVerifier
from servelearnbench.scoring import category, setting_scores


def test_categories():
    assert category("retail", "adapt") == "hidden"
    assert category("banking", "reopen") == "fully_specified"
    assert category("pitch", "reopen") == "hidden"


def test_setting_scores_use_test_rows_only():
    rows = [{"phase": "test", "category": "hidden", "reward": 1.0},
            {"phase": "test", "category": "hidden", "reward": 0.0},
            {"phase": "test", "category": "fully_specified", "reward": 1.0},
            {"phase": "serving", "category": "hidden", "reward": 1.0}]
    s = setting_scores(rows)
    assert s == {"hidden": 50.0, "fully_specified": 100.0, "n_test": 3}


def test_pitch_rubric_formula():
    items = {"a": "strong", "b": "pos", "c": "neg", "d": "red", "e": "neu"}
    score, det = RubricVerifier._score({"a": "M", "b": "M", "c": "N", "d": "N", "e": "N"}, items)
    assert score == 100.0
    score, _ = RubricVerifier._score({"a": "M", "b": "N", "c": "M", "d": "N", "e": "C"}, items, ["x"])
    assert abs(score - 100 * (2 / 3) * 0.5 * 0.5 * 0.5) < 1e-9
    assert RubricVerifier._word_gate(90.0, 91) == 25.0
