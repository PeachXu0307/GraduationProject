"""The closed-switch projection must preserve differentiability and KCL scope."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "physics"))
from audit_partial_ac import solved_input
from full_ac import build_fixture, full_ac_residual
from switch_projection import project_671692


def test_projection_retains_solved_state_and_gradient():
    topology = json.loads((ROOT / "data/graphs/ieee13_fixed_taps/topology.json").read_text())
    sample = json.loads((ROOT / "data/raw/ieee13_fixed_taps/samples.jsonl").read_text().splitlines()[0])
    fixture = build_fixture(sample, topology)
    pred, bases = solved_input(sample, topology)
    pred.requires_grad_(True)
    projected = project_671692(pred, sample, topology, bases, fixture)
    assert projected.shape == pred.shape
    assert full_ac_residual(projected, sample, topology, bases, fixture)["residual_a"].abs().max() < 1
    projected.square().mean().backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
