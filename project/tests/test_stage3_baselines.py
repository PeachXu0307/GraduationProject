"""Focused checks of Stage 3a masking and model output dimensions."""
import sys
from pathlib import Path
import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "training"))
from baselines import make_model
from train_baselines import load_data, masked_complex_mse, evaluate
from torch_geometric.loader import DataLoader


def test_masks_and_group_splits():
    split, top, _ = load_data(ROOT / "data/graphs/ieee13", ROOT / "data/raw/ieee13")
    assert [len(split[k]) for k in split] == [720, 90, 45, 45]
    batch = Batch.from_data_list([split["train"][0], split["train"][1]])
    assert tuple(batch.x.shape) == (32, 13)
    assert tuple(batch.y.shape) == (32, 6)
    assert tuple(batch.phase_mask.shape) == (32, 3)
    for name in ("mlp", "gcn", "edge_gnn"):
        model = make_model(name, len(top["bus_names"]))
        output = model(batch)
        assert tuple(output.shape) == tuple(batch.y.shape)
        assert torch.isfinite(masked_complex_mse(output, batch))
        # Missing-phase predictions must have no effect on the loss.
        changed = output.clone()
        missing = (1 - batch.phase_mask).repeat_interleave(2, dim=1).bool()
        changed[missing] = 1e6
        assert torch.allclose(masked_complex_mse(output, batch), masked_complex_mse(changed, batch))
        metrics = evaluate(model, DataLoader(split["val"][:2], batch_size=2), torch.device("cpu"))
        assert metrics["phase_observations"] == 82


def test_fixed_tap_graphs_preserve_double_precision():
    split, top, bases = load_data(ROOT / "data/graphs/ieee13_fixed_taps",
                                  ROOT / "data/raw/ieee13_fixed_taps")
    assert [len(split[k]) for k in split] == [720, 90, 45, 45]
    assert split["train"][0].x.dtype == torch.float64
    assert split["train"][0].y.dtype == torch.float64
    assert bases.dtype == torch.float64
    batch = Batch.from_data_list(split["train"][:2])
    model = make_model("edge_gnn", len(top["bus_names"])).double()
    pred = model(batch)
    assert pred.dtype == torch.float64 and pred.shape == batch.y.shape
    assert torch.isfinite(masked_complex_mse(pred, batch))
