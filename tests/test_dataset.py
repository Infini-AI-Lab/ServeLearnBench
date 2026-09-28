"""The dataset files in dataset/ are exactly what the code exports."""

import filecmp
import json
from pathlib import Path

import pytest

import servelearnbench
from servelearnbench.export import export

ROOT = Path(__file__).resolve().parents[1] / "dataset"


def test_dataset_fingerprints_match_the_package():
    pinned = json.loads((Path(servelearnbench.__file__).parent / "fingerprints.json").read_text())
    assert json.loads((ROOT / "fingerprints.json").read_text()) == pinned


@pytest.mark.parametrize("name", ["banking_l1", "pitch_l1", "retail_l1"])
def test_dataset_files_match_a_fresh_export(tmp_path, name):
    export(str(tmp_path), scenarios=[name])
    for sub in ("data", "environments"):
        fresh = sorted(p.relative_to(tmp_path) for p in (tmp_path / sub / name).rglob("*") if p.is_file())
        kept = sorted(p.relative_to(ROOT) for p in (ROOT / sub / name).rglob("*") if p.is_file())
        assert fresh == kept
        for rel in fresh:
            assert filecmp.cmp(tmp_path / rel, ROOT / rel, shallow=False), rel
