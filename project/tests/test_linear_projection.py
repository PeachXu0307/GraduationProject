"""Check local linear KCL correction without claiming a network solution."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "physics"))
from audit_partial_ac import solved_input
from full_ac import build_fixture, full_ac_residual
from linear_projection import LINEAR_BUSES, project_source_regulator
from switch_projection import project_671692


def test_linear_projection_zeroes_selected_kcl_and_keeps_gradient():
    topology = json.loads((ROOT / "data/graphs/ieee13_fixed_taps/topology.json").read_text())
    sample = json.loads((ROOT / "data/raw/ieee13_fixed_taps/samples.jsonl").read_text().splitlines()[0])
    fixture = build_fixture(sample, topology)
    pred, bases = solved_input(sample, topology)
    pred = (pred + 0.002 * torch.randn_like(pred)).requires_grad_(True)
    switch = project_671692(pred, sample, topology, bases, fixture)
    corrected = project_source_regulator(switch, topology, bases, fixture)
    residual = full_ac_residual(corrected, sample, topology, bases, fixture)["residual_a"]
    order = topology["phase_node_order"]
    selected = [order.index(f"{bus}.{phase}") for bus in LINEAR_BUSES for phase in (1, 2, 3)]
    assert residual[selected].abs().max().item() < 1e-5
    residual.abs().square().mean().backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
