"""Compare flat and GNN starts for the independent fixed-tap AC solver.

Tests are never used to select a checkpoint. Each method uses the same
declared-input fixture, convergence tolerance, and complete 41-node KCL check.
Per-case end-to-end time starts before inference/projection and includes the
nonlinear solver; fixture construction time is recorded separately.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "training"))
from baselines import EdgeAwareGNN  # noqa: E402
from train_baselines import load_data  # noqa: E402
from full_ac import build_fixture  # noqa: E402
from reduced_solver import flat_start, solve_ac  # noqa: E402
from switch_projection import project_671692  # noqa: E402
from linear_projection import project_source_regulator  # noqa: E402


def solved_voltage_errors(final_pu: torch.Tensor, target_pu: torch.Tensor,
                          phase_mask: torch.Tensor) -> tuple[float, float]:
    mask = phase_mask.bool()
    predicted = torch.view_as_complex(final_pu.reshape(-1, 3, 2).contiguous())[mask]
    target = torch.view_as_complex(target_pu.reshape(-1, 3, 2).contiguous())[mask]
    return float((predicted.abs() - target.abs()).abs().mean()), float((predicted - target).abs().mean())


def model_from(path: Path) -> EdgeAwareGNN:
    model = EdgeAwareGNN().double().eval()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-split", type=int, default=45)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--tolerance-a", type=float, default=0.01)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results/stage3_solver/benchmark.json")
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    raw_dir = ROOT / "data/raw/ieee13_fixed_taps"
    graph_dir = ROOT / "data/graphs/ieee13_fixed_taps"
    split, topology, bases = load_data(graph_dir, raw_dir)
    samples = {row["sample_id"]: row for row in
               (json.loads(line) for line in (raw_dir / "samples.jsonl").open(encoding="utf-8"))}
    models = {
        "supervised_gnn": model_from(ROOT / "results/stage3_fixed_supervised/edge_gnn_best.pt"),
        "physics_gnn": model_from(ROOT / "results/stage3_fixed_physics_projection/edge_gnn_physics_best.pt"),
    }
    records = []
    for split_name in ("test_ordinary", "test_stress"):
        for graph in split[split_name][:args.per_split]:
            sample = samples[graph.sample_id]
            started = time.perf_counter()
            fixture = build_fixture(sample, topology)
            fixture_seconds = time.perf_counter() - started
            starts = {"flat": flat_start(topology)}
            start_seconds = {"flat": 0.0}
            batch = next(iter(DataLoader([graph], batch_size=1)))
            for name, model in models.items():
                started = time.perf_counter()
                with torch.no_grad():
                    raw = model(batch)
                    projected = project_671692(raw, sample, topology, bases, fixture)
                    starts[name] = project_source_regulator(projected, topology, bases, fixture)
                start_seconds[name] = time.perf_counter() - started
            for name, state in starts.items():
                started = time.perf_counter()
                result = solve_ac(state, sample, topology, bases, fixture,
                                  max_iterations=args.max_iterations,
                                  tolerance_a=args.tolerance_a)
                solver_seconds = time.perf_counter() - started
                voltage_mae, complex_mae = solved_voltage_errors(
                    result["voltage_pu"], graph.y, graph.phase_mask)
                records.append({"sample_id": graph.sample_id, "split": split_name,
                                "method": name, "converged": result["converged"],
                                "status": result["status"],
                                "iterations": result["iterations"],
                                "function_evaluations": result["function_evaluations"],
                                "initial_kcl_rms_a": result["initial_kcl_rms_a"],
                                "final_kcl_rms_a": result["final_kcl_rms_a"],
                                "final_kcl_max_a": result["final_kcl_max_a"],
                                "final_voltage_mae_vs_opendss_pu": voltage_mae,
                                "final_complex_mae_vs_opendss_pu": complex_mae,
                                "fixture_seconds": fixture_seconds,
                                "inference_projection_seconds": start_seconds[name],
                                "solver_seconds": solver_seconds,
                                "execution_seconds_with_cached_fixture": start_seconds[name] + solver_seconds,
                                "full_pipeline_seconds": fixture_seconds + start_seconds[name] + solver_seconds})
        print(split_name, "completed", min(args.per_split, len(split[split_name])), flush=True)
    summary = {}
    for split_name in ("test_ordinary", "test_stress"):
        summary[split_name] = {}
        for method in ("flat", "supervised_gnn", "physics_gnn"):
            subset = [row for row in records if row["split"] == split_name and row["method"] == method]
            summary[split_name][method] = {
                "cases": len(subset), "successes": sum(row["converged"] for row in subset),
                "mean_iterations": sum(row["iterations"] for row in subset) / len(subset),
                "mean_function_evaluations": sum(row["function_evaluations"] for row in subset) / len(subset),
                "mean_solver_seconds": sum(row["solver_seconds"] for row in subset) / len(subset),
                "mean_fixture_seconds": sum(row["fixture_seconds"] for row in subset) / len(subset),
                "mean_cached_execution_seconds": sum(row["execution_seconds_with_cached_fixture"] for row in subset) / len(subset),
                "mean_full_pipeline_seconds": sum(row["full_pipeline_seconds"] for row in subset) / len(subset),
                "max_final_kcl_a": max(row["final_kcl_max_a"] for row in subset),
                "max_final_voltage_mae_vs_opendss_pu": max(row["final_voltage_mae_vs_opendss_pu"] for row in subset),
            }
    report = {"config": {"per_split": args.per_split,
                         "max_iterations": args.max_iterations,
                         "tolerance_a": args.tolerance_a,
                         "timing_scope": "cached execution includes inference/projection plus solver; full pipeline also includes fixture construction",
                         "fixture_is_declared_input": True},
              "summary": summary, "records": records}
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
