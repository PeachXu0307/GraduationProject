"""Evaluate voltage accuracy and complete KCL on the same output state.

Reports raw and switch-projected states separately for the fixed-tap IEEE-13
supervised and physics-trained checkpoints. No checkpoint selection occurs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "physics"))
from baselines import EdgeAwareGNN  # noqa: E402
from full_ac import build_fixture, full_ac_residual  # noqa: E402
from switch_projection import project_671692  # noqa: E402
from linear_projection import project_source_regulator  # noqa: E402
from train_baselines import load_data  # noqa: E402


def state_errors(state: torch.Tensor, truth: torch.Tensor,
                 mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Voltage-magnitude error in p.u. and wrapped angle error in degrees."""
    p = torch.view_as_complex(state.reshape(-1, 3, 2).contiguous())[mask.bool()]
    y = torch.view_as_complex(truth.reshape(-1, 3, 2).contiguous())[mask.bool()]
    return (p.abs() - y.abs()).abs(), torch.angle(p * y.conj()).abs() * 180 / torch.pi


@torch.no_grad()
def evaluate_states(model, graphs: list, samples: dict, topology: dict,
                    bases: torch.Tensor, fixtures: dict) -> dict:
    model.eval()
    n = len(topology["bus_names"])
    accum = {kind: {"voltage": [], "angle": [], "kcl": []}
             for kind in ("raw", "switch_projected", "switch_and_linear_projected")}
    node_squares = {kind: torch.zeros(len(topology["phase_node_order"]), dtype=torch.float64)
                    for kind in accum}
    node_counts = 0
    for batch in DataLoader(graphs, batch_size=16, shuffle=False):
        prediction = model(batch)
        for i, sample_id in enumerate(batch.sample_id):
            raw = prediction[i*n:(i+1)*n]
            projected = project_671692(raw, samples[sample_id], topology, bases,
                                        fixtures[sample_id])
            linear_projected = project_source_regulator(projected, topology, bases,
                                                         fixtures[sample_id])
            truth = batch.y[i*n:(i+1)*n]
            mask = batch.phase_mask[i*n:(i+1)*n]
            for kind, state in (("raw", raw), ("switch_projected", projected),
                                ("switch_and_linear_projected", linear_projected)):
                voltage, angle = state_errors(state, truth, mask)
                kcl = full_ac_residual(state, samples[sample_id], topology, bases,
                                       fixtures[sample_id])["residual_a"].abs()
                accum[kind]["voltage"].append(voltage.cpu())
                accum[kind]["angle"].append(angle.cpu())
                accum[kind]["kcl"].append(kcl.cpu())
                node_squares[kind] += kcl.cpu().square()
            node_counts += 1
    out = {}
    for kind, values in accum.items():
        voltage = torch.cat(values["voltage"])
        angle = torch.cat(values["angle"])
        kcl = torch.cat(values["kcl"])
        per_node = (node_squares[kind] / node_counts).sqrt()
        ranking = sorted(zip(topology["phase_node_order"], per_node.tolist()),
                         key=lambda item: -item[1])
        out[kind] = {"voltage_mae_pu": float(voltage.mean()),
                     "voltage_rmse_pu": float(voltage.square().mean().sqrt()),
                     "angle_mae_deg": float(angle.mean()),
                     "kcl_rms_a": float(kcl.square().mean().sqrt()),
                     "kcl_mean_abs_a": float(kcl.mean()),
                     "kcl_max_a": float(kcl.max()),
                     "worst_nodes_kcl_rms_a": [
                         {"node": node, "rms_a": value} for node, value in ranking[:10]],
                     "samples": node_counts}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results/stage3_fixed_physics_projection/consistent_metrics.json")
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    raw_dir = ROOT / "data/raw/ieee13_fixed_taps"
    graph_dir = ROOT / "data/graphs/ieee13_fixed_taps"
    split, topology, bases = load_data(graph_dir, raw_dir)
    samples = {row["sample_id"]: row for row in
               (json.loads(line) for line in (raw_dir / "samples.jsonl").open(encoding="utf-8"))}
    graphs = split["val"] + split["test_ordinary"] + split["test_stress"]
    fixtures = {g.sample_id: build_fixture(samples[g.sample_id], topology) for g in graphs}
    checkpoints = {
        "supervised": ROOT / "results/stage3_fixed_supervised/edge_gnn_best.pt",
        "physics": ROOT / "results/stage3_fixed_physics_projection/edge_gnn_physics_best.pt",
    }
    report = {"scope": "Fixed-tap IEEE-13; each voltage and KCL pair uses the same raw or projected state",
              "models": {}}
    for name, path in checkpoints.items():
        model = EdgeAwareGNN().double()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])
        report["models"][name] = {
            key: evaluate_states(model, split[key], samples, topology, bases, fixtures)
            for key in ("val", "test_ordinary", "test_stress")}
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    compact = {model: {split: {state: {
        "voltage_mae_pu": metrics["voltage_mae_pu"],
        "kcl_rms_a": metrics["kcl_rms_a"],
        "worst_node": metrics["worst_nodes_kcl_rms_a"][0]}
        for state, metrics in states.items()} for split, states in splits.items()}
        for model, splits in report["models"].items()}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
