"""Run with: python -m pytest tests/test_stage1.py"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "simulation"))

from generate_ieee13 import CIRCUIT, run_one, scenario, split_for_group


def test_source_and_reference_circuit():
    assert CIRCUIT.is_file()
    r = run_one(scenario(0, 0, 20261102))
    assert r["solver"]["converged"]
    assert len(r["buses"]) == 41
    assert len(r["lines"]) == 12
    assert r["kcl_residual"]["max_relative"] < 1e-3
    assert any(e["name"] == "transformer.xfm1" for e in r["elements"])


def test_grouped_variants_share_base_conditions():
    a = scenario(5, 0, 20261102)
    b = scenario(5, 1, 20261102)
    assert a["group_id"] == b["group_id"]
    assert a["phase_factor_abc"] == b["phase_factor_abc"]
    assert a["load_scale"] != b["load_scale"]
    assert split_for_group(5) == "train"
    assert split_for_group(250) == "val"
    assert split_for_group(275) == split_for_group(290) == "test"
