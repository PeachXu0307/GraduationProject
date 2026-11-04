"""Run with: python -m pytest tests/test_stage1.py"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "simulation"))

from generate_ieee13 import CIRCUIT, FIXED_REGULATOR_TAPS, run_one, scenario, split_for_group


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


def test_fixed_tap_policy_is_declared_and_controls_off():
    params = scenario(0, 0, 20261102)
    fixed = run_one(params, tap_policy="fixed")
    assert fixed["solver"]["converged"]
    assert fixed["tap_policy"]["control_mode"] == "OFF"
    assert fixed["tap_policy"]["declared_regulator_taps"] == FIXED_REGULATOR_TAPS
    # OpenDSS may count one control-pass bookkeeping step even with controls off.
    for name, tap in FIXED_REGULATOR_TAPS.items():
        assert abs(fixed["transformer_taps_after_solve"][name][1] - tap) < 1e-10
    assert fixed["kcl_residual"]["max_relative"] < 1e-3
