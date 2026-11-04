"""Train MLP/GCN baselines on grouped IEEE 13 graph splits.

Run from project/: .venv/Scripts/python.exe src/training/train_baselines.py
"""
from __future__ import annotations

import argparse, csv, json, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
from baselines import make_model


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_data(graph_dir: Path, raw_dir: Path):
    graphs = torch.load(graph_dir / "graphs.pt", weights_only=False)
    top = json.loads((graph_dir / "topology.json").read_text(encoding="utf-8"))
    index = pd.read_csv(raw_dir / "index.csv").set_index("sample_id")
    assert len(graphs) == len(index) == 900
    assert index.groupby("group_id").split.nunique().max() == 1
    rows = [json.loads(s) for s in (raw_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["sample_id"] for r in rows] == [g.sample_id for g in graphs]
    base = {}
    for b in rows[0]["buses"]:
        if b["bus"] in base: assert abs(base[b["bus"]] - b["v_base_ln_kv"]) < 1e-10
        base[b["bus"]] = b["v_base_ln_kv"]
    bases = torch.tensor([base[name] * 1000 for name in top["bus_names"]], dtype=graphs[0].y.dtype)
    # Graphs share a fixed bus order and voltage bases. Scaling each bus by its
    # line-neutral base prevents 115-kV and 480-V buses from dominating loss.
    for g in graphs:
        assert g.x.shape == (len(bases), 13) and g.y.shape == (len(bases), 6)
        g.y = g.y / bases[:, None].repeat(1, 6)
        assert torch.isfinite(g.y).all()
    split = {k: [] for k in ("train", "val", "test_ordinary", "test_stress")}
    for g in graphs:
        item = index.loc[g.sample_id]
        key = item["split"]
        if key == "test": key = "test_stress" if item["regime"] == "stress_holdout" else "test_ordinary"
        split[key].append(g)
    assert [len(split[k]) for k in split] == [720, 90, 45, 45]
    return split, top, bases


def masked_complex_mse(pred: torch.Tensor, batch) -> torch.Tensor:
    m = batch.phase_mask.repeat_interleave(2, dim=1)
    return (((pred - batch.y) ** 2) * m).sum() / m.sum()


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> dict:
    model.eval()
    voltage_errors, angle_errors, complex_errors = [], [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch).reshape(-1, 3, 2)
        truth = batch.y.reshape(-1, 3, 2)
        mask = batch.phase_mask.bool()
        p = torch.view_as_complex(pred.contiguous())[mask]
        y = torch.view_as_complex(truth.contiguous())[mask]
        voltage_errors.append((p.abs() - y.abs()).cpu())
        complex_errors.append((p - y).abs().cpu())
        angle = torch.angle(p * y.conj()) * 180 / torch.pi
        angle_errors.append(angle.abs().cpu())
    dv = torch.cat(voltage_errors); dc = torch.cat(complex_errors); da = torch.cat(angle_errors)
    return {"voltage_mae_pu": float(dv.abs().mean()),
            "voltage_rmse_pu": float(torch.sqrt((dv ** 2).mean())),
            "complex_mae_pu": float(dc.mean()),
            "angle_mae_deg": float(da.mean()), "phase_observations": int(dv.numel())}


def train_one(name: str, split: dict, num_buses: int, args, device: torch.device) -> dict:
    floating_dtype = split["train"][0].x.dtype
    model = make_model(name, num_buses).to(device=device, dtype=floating_dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(split["train"], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(split["val"], batch_size=args.batch_size, shuffle=False)
    history, best, best_state, patience = [], float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter(); model.train(); loss_sum = count = 0
        for batch in train_loader:
            batch = batch.to(device); optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            loss = masked_complex_mse(pred, batch)
            loss.backward(); optimizer.step()
            loss_sum += float(loss) * batch.num_graphs; count += batch.num_graphs
        val = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "train_masked_mse": loss_sum/count,
                  "val_voltage_mae_pu": val["voltage_mae_pu"], "seconds": time.perf_counter()-start}
        history.append(record)
        if val["voltage_mae_pu"] < best - 1e-6:
            best = val["voltage_mae_pu"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch; patience = 0
        else: patience += 1
        if epoch == 1 or epoch % 10 == 0:
            print(name, "epoch", epoch, "train_mse", f"{record['train_masked_mse']:.6g}", "val_mae", f"{best:.6g}")
        if patience >= args.patience: break
    model.load_state_dict(best_state)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": name, "state_dict": best_state, "best_epoch": best_epoch,
                "seed": args.seed, "num_buses": num_buses}, args.out / f"{name}_best.pt")
    with (args.out / f"{name}_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    metrics = {"model": name, "best_epoch": best_epoch}
    for key in ("val", "test_ordinary", "test_stress"):
        metrics[key] = evaluate(model, DataLoader(split[key], batch_size=args.batch_size), device)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--graph-dir", type=Path, default=ROOT/"data/graphs/ieee13")
    p.add_argument("--raw-dir", type=Path, default=ROOT/"data/raw/ieee13")
    p.add_argument("--out", type=Path, default=ROOT/"results/stage3_baselines")
    p.add_argument("--models", nargs="+", choices=["mlp", "gcn", "edge_gnn"], default=["mlp", "gcn", "edge_gnn"])
    p.add_argument("--seed", type=int, default=20261102)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    a = p.parse_args(); a.out = a.out.resolve()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if a.device == "cuda" or (a.device == "auto" and torch.cuda.is_available()) else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    split, top, bases = load_data(a.graph_dir, a.raw_dir)
    results = []
    for name in a.models:
        set_seed(a.seed)
        results.append(train_one(name, split, len(top["bus_names"]), a, device))
    a.out.mkdir(parents=True, exist_ok=True)
    report = {"config": {"seed": a.seed, "epochs": a.epochs, "patience": a.patience,
                         "batch_size": a.batch_size, "lr": a.lr, "weight_decay": a.weight_decay,
                         "device": str(device), "torch": torch.__version__, "voltage_bases_v": bases.tolist()},
              "graph_dtype": str(split["train"][0].x.dtype),
              "results": results}
    (a.out/"metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__": main()
