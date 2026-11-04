"""Fine-tune the fixed-tap double-precision GNN with complete AC KCL.

Run with the saved supervised checkpoint. Test sets are evaluated only after
validation selection. The physics penalty is robust to early large residuals;
its scale and weight are declared in the saved configuration.
"""
from __future__ import annotations

import argparse
import csv
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
from train_baselines import load_data, masked_complex_mse, evaluate, set_seed  # noqa: E402


def physics_loss(pred: torch.Tensor, batch, samples: dict, topology: dict,
                 bases: torch.Tensor, fixtures: dict, current_scale_a: float,
                 project_switch: bool = False) -> torch.Tensor:
    """Mean log(1+|KCL|/Ibase), over all phase nodes and graphs."""
    n = len(topology["bus_names"])
    pieces = []
    for i, sample_id in enumerate(batch.sample_id):
        state = pred[i*n:(i+1)*n]
        if project_switch:
            state = project_671692(state, samples[sample_id], topology, bases,
                                    fixtures[sample_id])
        result = full_ac_residual(state, samples[sample_id], topology,
                                  bases, fixtures[sample_id])
        pieces.append(torch.log1p(result["residual_a"].abs() / current_scale_a).mean())
    return torch.stack(pieces).mean()


@torch.no_grad()
def physics_metrics(model, graphs: list, samples: dict, topology: dict,
                    bases: torch.Tensor, fixtures: dict, device: torch.device,
                    project_switch: bool = False) -> dict:
    model.eval()
    values = []
    for batch in DataLoader(graphs, batch_size=16):
        batch = batch.to(device)
        pred = model(batch)
        n = len(topology["bus_names"])
        for i, sample_id in enumerate(batch.sample_id):
            state = pred[i*n:(i+1)*n]
            if project_switch:
                state = project_671692(state, samples[sample_id], topology, bases,
                                        fixtures[sample_id])
            result = full_ac_residual(state, samples[sample_id], topology,
                                      bases, fixtures[sample_id])
            values.append(result["residual_a"].abs().cpu())
    a = torch.stack(values)
    return {"kcl_rms_a": float(torch.sqrt((a**2).mean())),
            "kcl_mean_abs_a": float(a.mean()),
            "kcl_max_a": float(a.max()), "samples": len(values)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "data/graphs/ieee13_fixed_taps")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/ieee13_fixed_taps")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "results/stage3_fixed_supervised/edge_gnn_best.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "results/stage3_fixed_physics")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lambda-pf", type=float, default=1e-4)
    parser.add_argument("--current-scale-a", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--project-switch", action="store_true")
    args = parser.parse_args()
    if args.lambda_pf <= 0 or args.current_scale_a <= 0:
        parser.error("Physics weight and current scale must be positive")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    split, topology, bases = load_data(args.graph_dir, args.raw_dir)
    if split["train"][0].x.dtype != torch.float64:
        raise TypeError("Full AC training requires float64 graph tensors")
    samples = {row["sample_id"]: row for row in
               (json.loads(line) for line in (args.raw_dir / "samples.jsonl").open(encoding="utf-8"))}
    # Build declared-input fixtures once. OpenDSS is never called during the
    # differentiable optimization steps, and no reference outcome enters Y.
    required = split["train"] + split["val"] + split["test_ordinary"] + split["test_stress"]
    fixtures = {graph.sample_id: build_fixture(samples[graph.sample_id], topology)
                for graph in required}
    bases = bases.to(device)
    model = EdgeAwareGNN().to(device=device, dtype=torch.float64)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    baseline = {}
    for key in ("val",):
        baseline[key] = {**evaluate(model, DataLoader(split[key], batch_size=16), device),
                         **physics_metrics(model, split[key], samples, topology, bases, fixtures,
                                           device, args.project_switch)}
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = float("inf")
    best_state = None
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = [0.0, 0.0]
        count = 0
        for batch in DataLoader(split["train"], batch_size=args.batch_size, shuffle=True):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            data = masked_complex_mse(pred, batch)
            pf = physics_loss(pred, batch, samples, topology, bases, fixtures,
                              args.current_scale_a, args.project_switch)
            total = data + args.lambda_pf * pf
            if not torch.isfinite(total):
                raise FloatingPointError("Nonfinite training loss")
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sums[0] += float(data.detach()) * batch.num_graphs
            sums[1] += float(pf.detach()) * batch.num_graphs
            count += batch.num_graphs
        val_voltage = evaluate(model, DataLoader(split["val"], batch_size=16), device)
        val_physics = physics_metrics(model, split["val"], samples, topology,
                                      bases, fixtures, device, args.project_switch)
        # Validation selection uses both objectives; no test result is used.
        score = val_voltage["voltage_mae_pu"] + args.lambda_pf * \
            (val_physics["kcl_mean_abs_a"] / args.current_scale_a)
        history.append({"epoch": epoch, "train_data_mse": sums[0]/count,
                        "train_pf_log_loss": sums[1]/count,
                        "val_voltage_mae_pu": val_voltage["voltage_mae_pu"],
                        "val_kcl_rms_a": val_physics["kcl_rms_a"],
                        "selection_score": score})
        if score < best:
            best, best_epoch = score, epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        print("epoch", epoch, "val_voltage_mae", f"{val_voltage['voltage_mae_pu']:.6g}",
              "val_kcl_rms_a", f"{val_physics['kcl_rms_a']:.6g}", flush=True)
    model.load_state_dict(best_state)
    final = {}
    for key in ("val", "test_ordinary", "test_stress"):
        final[key] = {**evaluate(model, DataLoader(split[key], batch_size=16), device),
                      **physics_metrics(model, split[key], samples, topology, bases, fixtures,
                                        device, args.project_switch)}
    baseline_model = EdgeAwareGNN().to(device=device, dtype=torch.float64)
    baseline_model.load_state_dict(checkpoint["state_dict"])
    for key in ("test_ordinary", "test_stress"):
        baseline[key] = {**evaluate(baseline_model, DataLoader(split[key], batch_size=16), device),
                         **physics_metrics(baseline_model, split[key], samples, topology,
                                           bases, fixtures, device, args.project_switch)}
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "best_epoch": best_epoch,
                "source_checkpoint": str(args.checkpoint)}, args.out / "edge_gnn_physics_best.pt")
    with (args.out / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    report = {"config": {"seed": args.seed, "epochs": args.epochs, "lr": args.lr,
                         "batch_size": args.batch_size, "lambda_pf": args.lambda_pf,
                         "current_scale_a": args.current_scale_a, "device": str(device),
                         "dtype": "float64", "physics_loss": "mean log1p(abs(KCL_A)/Ibase)"},
              "projection": "671692 one-step" if args.project_switch else "none",
              "best_epoch": best_epoch, "baseline": baseline, "physics_model": final}
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "baseline": baseline,
                      "physics_model": final}, indent=2))


if __name__ == "__main__":
    main()
