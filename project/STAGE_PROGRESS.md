# Graduation project progress

## Stage 0 — route agreed

First implementation: IEEE 13 feeder; OpenDSS reference data; bus-level graph
with three-phase matrix edge features; edge-aware GNN voltage prediction;
full AC residual audit; branch states from predicted voltages; independent
three-phase damped NR initial-state comparison if OpenDSS cannot accept
per-node voltage initialization.

## Stage 1 — complete for the initial fixed-topology regression dataset

- Source circuit and source hashes: `networks/ieee13/SOURCE.md`.
- Environment: Python 3.12, dss-python 0.15.7 / DSS C-API 0.14.5,
  NumPy 2.5.4, Pandas 2.3.3, Matplotlib 3.11.2, pytest 8.4.2.
- Seed: 20261102. 300 scenario groups × 3 nearby variants = 900 samples.
- Split: train 720, validation 90, test 90, by complete group.
  Test has 45 ordinary and 45 stress samples.
- Solver: 900/900 converged. Its final iteration and control-iteration counts
  are stored. OpenDSS does not expose a final mismatch value through the
  interface used here; terminal-current KCL is computed after the solve.
- Network per sample: 16 bus names, 41 energized phase nodes, 12 lines,
  5 transformer/regulator elements; 15 source loads plus optional DG.
- Largest relative KCL postprocessing residual: 6.696947137529763e-05.
- Repeated default generation produced byte-for-byte identical output files.
- Quality audit passed. Tests: 2 passed.

### Important limits

The dataset has one line topology and synthetic load, DG, and capacitor
operating conditions. There are no reliable numerical-failure labels, no
proof of physical infeasibility, and no line-rating audit. The voltage range
is observational only; safety limits have not yet been formalized. The
failure-classification dataset remains a separate stage 4 task.

## Stage 2 — complete for the initial graph representation

- Installed PyTorch 2.5.1 and PyTorch Geometric 2.7.0 in the new directory's
  `.venv`.
- Recommended representation: one physical bus per graph node, A/B/C slots
  with a phase mask, and two directed message edges per physical branch.
- Generated 900 PyG graphs under `data/graphs/ieee13/`.
- Tensor shapes: `x=(16,13)`, `phase_mask=(16,3)`, `y=(16,6)`,
  `edge_index=(2,34)`, `edge_attr=(34,28)`, `branch_id=(34,)`; incidence is
  `(17,16)`.
- Graph audit passed for all samples: 41 valid phase nodes, 17 physical
  line/transformer branches, 34 directed message edges.
- Line pi-model current reconstruction agrees with OpenDSS terminal current
  with maximum relative error `2.71e-8`.

### Stage 2 limitations

Transformer/regulator edge matrices are intentionally not treated as line
impedances. The exact complete AC residual still needs source, load,
capacitor, generator, transformer and regulator port models. The P/Q node
features for multi-phase and delta loads are equal-allocation input proxies;
the raw exact terminal quantities remain available for Stage 3 physics.

### Next task

Stage 3: implement MLP and ordinary GNN baselines first, then an edge-aware
physical GNN; use exact device equations and phase masks in the AC residual;
do not claim a full physical constraint until transformer and regulator
models are included.

## Stage 3a — ordinary baselines complete

- Models: fixed-topology node MLP with bus identity and two-layer GCN.
- Inputs: Stage 2 graph node features; targets: masked A/B/C complex
  line-to-neutral voltage normalized by each bus's voltage base.
- Training: seed 20261102; Adam; CPU; 80 epochs maximum; select minimum
  validation voltage-magnitude MAE. The ordinary and stress test splits
  remain unseen until final evaluation.
- MLP best epoch 75: ordinary test MAE 0.00603 p.u., stress test MAE
  0.01128 p.u.; angle MAE 0.395 and 0.799 degrees.
- GCN best epoch 77: ordinary test MAE 0.00565 p.u., stress test MAE
  0.01108 p.u.; angle MAE 0.314 and 0.645 degrees.
- These are first-run supervised results, not a physics-constrained model or
  a claim of speedup. The two-layer GCN does not use edge impedance features.

### Next Stage 3 task

Implement a branch-feature-aware message passing model and a complete,
independently checked three-phase AC residual. Determine how to include
transformers, tap positions, delta loads, capacitors and source terms from
known scenario inputs without using solved output quantities as model inputs.

## Stage 3b — edge-aware supervised model complete

- Added three residual edge-conditioned message layers using the verified
  28-dimensional line/device edge features.
- Same seed, grouped splits, target normalization, optimizer and stopping
  rule as the MLP/GCN comparison.
- Best epoch 80: ordinary test voltage MAE `0.005095 p.u.`, stress test MAE
  `0.010486 p.u.`; angle MAE `0.306` and `0.694` degrees.
- The model is still supervised only; these are prediction errors, not AC
  feasibility or solver convergence results.
- Stage 1 was regenerated to record load connection flags and solved tap
  positions. Solved tap positions are audit data, not predictor inputs.

### Remaining Stage 3 work

Implement and independently validate complete device equations before
training with a physical loss. Do not feed solved terminal currents, powers,
or regulator taps to the predictor as hidden inputs.

### Stage 3b port-audit evidence

- Reproducible audit script: `src/physics/audit_ports.py`.
- Twelve sampled operating points yielded 224 line/transformer/capacitor
  `Yprim` comparisons; maximum relative current error `7.68e-12`.
- Load audit found OpenDSS voltage-band model switching. Applying a first
  fallback reduced, but did not eliminate, the low-voltage discrepancy:
  Model 1 wye maximum `0.00357` relative, Model 5 wye maximum `0.00203`.
- Therefore no full AC physics loss or physics-informed model is yet claimed.
  The next action is to reconcile the remaining load-model discrepancy and
  validate generator/source current treatment across the full scenario set.
