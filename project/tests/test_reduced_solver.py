"""Verify the independent AC corrector against a fixed-tap reference case."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "physics"))
from audit_partial_ac import solved_input
from full_ac import build_fixture
from reduced_solver import flat_start, solve_ac, unknown_indices


def test_damped_newton_flat_start_reaches_full_kcl():
    topology = json.loads((ROOT / "data/graphs/ieee13_fixed_taps/topology.json").read_text())
    sample = json.loads((ROOT / "data/raw/ieee13_fixed_taps/samples.jsonl").read_text().splitlines()[0])
    _, bases = solved_input(sample, topology)
    fixture = build_fixture(sample, topology)
    assert len(unknown_indices(topology)) == 29
    result = solve_ac(flat_start(topology), sample, topology, bases, fixture,
                      max_iterations=10, tolerance_a=0.01)
    assert result["converged"]
    assert result["final_kcl_max_a"] < 0.01
    assert result["iterations"] <= 5
