# Stage 3: supervised baselines and edge-aware model

This is the first small task of Stage 3. It builds a **supervised-only**
voltage-prediction baseline before adding edge-aware messages and physical
losses. It does not claim to solve or physically validate an AC power flow.

## Models and tensors

- `NodeMLP`: each bus receives its own 13 inputs plus a fixed bus identity
  vector of length 16; layers `29 -> 64 -> 64 -> 6`. There is no message
  passing. Its identity feature is appropriate only for this fixed topology.
- `BusGCN`: two standard `GCNConv` layers `13 -> 64 -> 64`, then a `64 -> 6`
  output head. It uses the 34 directed topology edges but does not use the
  28 edge features or transformer physics. This is intentionally an ordinary
  GNN baseline, not the final electric-aware model.
- `EdgeAwareGNN`: three residual message-passing layers. Each message uses
  both node states and 28 edge features, including line mutual R/X entries.
  It remains supervised only and does not yet enforce AC equations.
- The 6 outputs are `Re/Im` for phases A/B/C in line-to-neutral per-unit.
  The target voltage in volts is divided by each bus's OpenDSS `kVBase`.
  Masked phase slots never contribute to loss or metrics.

Training minimizes masked mean squared error of the six real channels.
Validation selection uses voltage-magnitude MAE (p.u.). Reported metrics
are magnitude MAE/RMSE, complex-voltage MAE, and wrapped angle MAE; each
observed phase has equal weight. No test set is used for model selection.
The test is separately reported for 45 ordinary and 45 stress scenarios.

## Run and verify

From `E:\GraduationProject\project`:

```powershell
.venv\Scripts\python.exe src\training\train_baselines.py
.venv\Scripts\python.exe -m pytest tests\test_stage3_baselines.py -q -p no:faulthandler -p no:cacheprovider
```

MLP/GCN results are saved in `results/stage3_baselines/`. The edge-aware
run is saved in `results/stage3_edge/`:

- `mlp_best.pt`, `gcn_best.pt`: selected state dictionary and seed;
- `mlp_history.csv`, `gcn_history.csv`: per-epoch train loss, validation MAE,
  and elapsed time;
- `metrics.json`: configuration, voltage bases, and validation/test metrics.

The first run uses seed 20261102, CPU, Adam, batch size 32, learning rate
0.001, weight decay 1e-5, up to 80 epochs, and patience 20. The current
dataset and graph have only one topology. Results do not establish
generalization to another feeder, missing measurements, or improved solver
convergence. In particular, a low voltage error is not proof of AC
equation feasibility.
