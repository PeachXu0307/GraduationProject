"""Focused tests for fixed-tap complete AC residual and gradient."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "physics"))
from audit_partial_ac import solved_input
from full_ac import build_fixture, full_ac_residual


def test_complete_residual_at_solved_voltage_and_gradient():
    topology = json.loads((ROOT / "data/graphs/ieee13/topology.json").read_text(encoding="utf-8"))
    rows = (ROOT / "data/raw/ieee13_fixed_taps/samples.jsonl").read_text(encoding="utf-8").splitlines()
    stress_low_generator = next(row for row in rows if '"sample_id": "g290_v2"' in row)
    for row in (rows[0], rows[-1], stress_low_generator):
        sample = json.loads(row)
        fixture = build_fixture(sample, topology)
        assert fixture["device_counts"]["transformer"] == 5
        pred, bases = solved_input(sample, topology)
        pred.requires_grad_(True)
        result = full_ac_residual(pred, sample, topology, bases, fixture)
        assert result["residual_a"].shape == (41,)
        assert result["residual_a"].abs().max().item() < 0.5
        result["residual_rms_a"].backward()
        assert pred.grad is not None and torch.isfinite(pred.grad).all()
        with pytest.raises(TypeError):
            full_ac_residual(pred.float(), sample, topology, bases, fixture)
