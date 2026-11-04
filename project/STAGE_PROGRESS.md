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

The fixed-PQ generator phase-current relation matches the base DG scenario at
about `5.3e-5` relative error. The source is voltage-controlled and is not a
PQ load; its current will be handled through the complete network model rather
than a guessed source-power formula. The residual implementation remains
deferred until these declared input/device rules are complete.

## Stage 3c — expanded port audit, 30 scenarios

- Added declared-parameter source Norton current and fixed-PQ generator
  current to `src/physics/audit_ports.py`.
- 562 linear-device port comparisons: maximum relative error `9.50e-12`.
- 20 generator comparisons: maximum relative error `9.91e-5`.
- 30 source comparisons: maximum relative error `2.29e-6`.
- Focused port test passed for ordinary and stress cases.
- Low-voltage Model 1/5 load differences remain up to `0.00357`/`0.00203`
  relative. These are not acceptable as a claimed exact AC physics loss.
- Next: identify OpenDSS's precise low-voltage load rule; choose a declared
  regulator-tap treatment, then implement differentiable complete residual.

The public OpenDSS `Load.pas` source identifies the low-voltage current
interpolation. Model 1 and Model 5 use different upper-boundary currents.
The complete 900-scenario audit now finds Model 1 wye maximum `1.42e-4`
and Model 5 wye maximum `1.13e-4` relative current error. Linear-device
`Yprim` comparisons remain at `1.18e-11` maximum; source maximum is
`2.56e-6`; generator maximum is `3.49e-4`. No full-network physics loss
has been trained yet. The next requirement is a declared tap-state policy
and an independently checked nodal KCL residual with predicted voltages.

Added `src/physics/partial_ac.py` as a differentiable tensor prototype. It
returns line/load residuals, a covered-node mask, omitted-device metadata and
an RMS value with gradients. The gradient and finite-value test passes. A
solved-voltage numerical comparison still has large residuals at some
load/branch junctions, so the module is explicitly **not** used for training
and is not called a complete AC physics loss. The next step is to reconcile
phase orientation and line/load device composition before any physics-aware
training.

Added `src/physics/audit_nodal_kcl.py`. Independent device-current summation
over all 900 solved scenarios gives maximum relative nodal KCL residual
`1.00e-4` and maximum absolute residual `0.213 A`. This validates device
composition at solved voltages only; it does not yet justify a differentiable
training loss or a claim that the GNN output is a solved AC state.

## Stage 3d — covered partial AC residual checked

- Corrected the KCL sign: line and load port currents both enter their
  respective devices, so their contributions add at a bus.
- The `671--692` switch has a `1e-7 ohm` per-phase series resistance. Its
  terminal current reconstructed in double precision matches the recorded
  OpenDSS current to about `8.3e-9` relative on the first sample.
- The differentiable residual now evaluates its device equations in
  `float64/complex128` and excludes every phase node incident to an omitted
  source, transformer/regulator, capacitor, or generator.
- `src/physics/audit_partial_ac.py` checked all 900 solved operating points:
  largest covered-node residual `0.038573 A`, largest sample covered-node RMS
  `0.011012 A`; report in `results/stage3_edge/partial_ac_audit_900.json`.
- All eight project tests pass. The supervised GNN has **not** been retrained
  with this partial residual. Float32 model outputs have already lost precision
  before the internal double cast and are unsuitable near the low-impedance
  switch without an explicitly validated precision or switch policy.
- Next: define regulator tap treatment and implement the missing device
  equations before claiming a complete AC loss. Keep prediction, equation
  residual, solver convergence, and operating safety as separate metrics.

## Stage 3e — tap-state policy decision and pilot

- `src/physics/audit_tap_policy.py` audited all 900 existing regulated
  scenarios. The three regulators have 9, 8, and 10 distinct solved tap
  states, respectively. A single joint state made from the training modes
  matches only `14.44%` of test cases. Solved taps remain audit labels and
  must not be used as model inputs or an undeclared training-time parameter.
- The canonical IEEE-13 DSS file documents an alternate fixed-tap solution:
  Reg1 `1.0625`, Reg2 `1.0500`, Reg3 `1.06875`, with controls off.
  `src/physics/audit_fixed_taps.py` tested that policy on 30 spread-out
  scenarios. All 30 converged. The largest phase-voltage magnitude difference
  from the original regulated solution was `0.027243 p.u.` This is a distinct
  operating policy and requires its own labels and reported comparison.
- Numerically, casting the first solved sample's voltage from float64 to
  float32 before partial KCL evaluation changes the covered-node RMS from
  `0.00398 A` to `371 A`, because the explicit low-impedance 671--692 switch
  magnifies rounding error. An internal float64 cast cannot restore lost
  input precision.
- Decision for the first **complete physics-loss experiment**: build a
  separate fixed-tap, controls-off dataset with the same scenario-group split;
  calculate and train the electrical residual in float64/complex128 from a
  float64 prediction path. Preserve the regulated dataset as the supervised
  benchmark. Compare the two operating policies explicitly; never score
  fixed-tap predictions against regulated labels as if they were identical.
- Remaining work: generate the separate fixed-tap labels, construct declared
  linear-device port matrices and nonlinear device currents, validate complete
  KCL at solved voltages, then train and evaluate the physical-loss model.

## Stage 3f — separate fixed-tap dataset generated

- Extended `src/simulation/generate_ieee13.py` with `--tap-policy fixed`.
  The fixed policy applies the canonical IEEE-13 alternate Reg1/2/3 taps
  `1.0625/1.0500/1.06875` and sets `Controlmode=OFF` before the scenario solve.
  The original default `regulated` path remains the existing benchmark.
- Generated `data/raw/ieee13_fixed_taps/`: 900/900 converged, 720 train,
  90 validation, 90 test, with all 900 sample IDs, scenario inputs and group
  splits exactly matching the regulated counterpart. Every saved regulator
  tap equals its declared fixed value. Maximum postprocessed KCL relative
  residual is `6.98e-5`.
- Repeating generation yielded byte-for-byte identical `samples.jsonl`,
  `index.csv`, `bus_voltages.csv`, `branch_terminals.csv`, and `manifest.json`.
- The 900-sample covered partial AC audit on this dataset reports maximum
  phase-node residual `0.051265 A` and maximum sample RMS `0.012378 A`.
  Report: `results/stage3_edge/partial_ac_audit_fixed_900.json`.
- All nine project tests pass. The complete device residual and physical-loss
  training remain pending. This fixed-tap dataset must be identified as a
  separate operating policy in any model comparison or thesis result.

## Stage 3g — differentiable complete fixed-tap AC residual audited

- Added `src/physics/full_ac.py`. It compiles the declared IEEE-13 circuit,
  applies each scenario and the fixed tap policy, then calls OpenDSS
  `BuildYMatrix(1, 1)` to refresh primitive admittances **before** the scenario
  solve. The fixture contains the line, transformer, capacitor and source
  `Yprim` contributions plus the source Norton injection. Load and generator
  currents are differentiable functions of the predicted complex voltages.
  No solved voltage, port current, power, or post-solve tap enters the fixture.
- Device coverage is all 41 energized phase nodes: 12 lines, 5 transformers,
  1 or 2 enabled capacitors, 1 source, 15 loads and optional DG. The source
  term is a declared three-phase Norton model; RegControl objects have no
  electrical port and their fixed transformer taps are already represented.
- Two debugging findings were essential: transformer `Yprim` is stale after
  a tap edit until `BuildYMatrix`; and the Model 1 generator switches to an
  equivalent admittance when a phase falls below `Vminpu=0.9`. The stress
  case `g290_v2` has phase 675.3 at `0.873 p.u.`.
- Across all 900 fixed-tap solved samples, maximum complete KCL phase-node
  residual is `0.201215 A`; maximum sample RMS is `0.037151 A`.
  Report: `results/stage3_edge/full_ac_audit_fixed_900.json`. The remaining
  largest error is at 634.1, consistent with small load-model mismatch.
- The complete residual requires a float64 prediction tensor; the 671--692
  low-impedance switch makes a float32 voltage path numerically invalid.
  Gradient, low-voltage generator, and reference-voltage tests pass. All ten
  project tests pass.
- This establishes an independently checked equation residual at solved
  voltages. It does not yet establish model prediction accuracy, physical
  feasibility of predictions, solver convergence, or speedup. Next: create
  fixed-tap graph tensors and train a double-precision physics-embedded GNN
  with a supervised baseline under the same fixed-tap dataset and split.

## Stage 3h — double-precision supervised and physical-loss experiments

- Rebuilt the 900 fixed-tap graphs at `data/graphs/ieee13_fixed_taps/` in
  float64. Topology, sample IDs and all input features exactly match the
  original regulated graph set; all 900 voltage targets differ. Raw reference
  complex voltages are preserved exactly in the graph targets.
- The fixed-tap double-precision edge-aware supervised baseline was trained
  with seed 20261102 and the existing 720/90/45/45 grouped split. Its
  ordinary and stress test voltage-magnitude MAE are `0.006569` and
  `0.015750 p.u.`. The validation-selected checkpoint and metrics are in
  `results/stage3_fixed_supervised/`.
- Direct substitution of the supervised GNN voltage into complete KCL gives
  enormous current residuals, mainly across the explicit 671--692 switch:
  its 1e-7-ohm impedance converts modest independently predicted endpoint
  voltage differences into tens of millions of amperes. This is a model
  failure, not an AC solution, despite small voltage-magnitude MAE.
- Added `src/physics/switch_projection.py`: a differentiable one-step voltage
  projection enforcing a near-equal 671/692 voltage and using the other-device
  node-692 KCL to estimate the switch's small physical drop. The switch is
  retained in the complete residual; this is a declared numerical
  initialization, not a solver or proof of feasibility.
- Fine-tuned the same edge-aware GNN in float64 for 20 epochs using the full
  KCL penalty `mean(log1p(|I_KCL|/100 A))` with weight `1e-4`, plus the
  original masked complex-voltage MSE. The best checkpoint was selected on
  validation only at epoch 17. Reproducible code, history and metrics are in
  `src/training/train_physics.py` and
  `results/stage3_fixed_physics_projection/`.
- With the **same switch projection applied during KCL evaluation** for both
  models, ordinary-test KCL RMS decreased from `11865.19 A` to `930.76 A`,
  while raw-output voltage-magnitude MAE changed from `0.006569` to
  `0.006528 p.u.`. Stress-test KCL RMS decreased from `11850.48 A` to
  `1017.01 A`; raw-output voltage MAE changed from `0.015750` to
  `0.015095 p.u.`. The voltage metrics refer to unprojected GNN output;
  KCL metrics refer to projected output. They must not be presented as if
  evaluated on the same voltage state.
- The KCL residual remains orders of magnitude above the solved-reference
  audit maximum sample RMS `0.037151 A`. No predicted state is claimed to
  satisfy the full AC equations or operating limits. Stage 3 next needs
  output-consistent metrics, stronger physics-aware architecture/training,
  and eventually exact-solver correction before any feasibility claim.
- All 12 project tests pass.

## Stage 3i — output-consistent evaluation and local linear correction

- Added `src/training/evaluate_consistent.py`. For each checkpoint and split,
  voltage MAE and complete KCL RMS are now evaluated on the **same** voltage
  state: raw prediction, switch-projected prediction, or switch plus local
  source/regulator correction. Report:
  `results/stage3_fixed_physics_projection/consistent_metrics.json`.
- Raw supervised ordinary-test output has voltage MAE `0.006569 p.u.` but
  KCL RMS `6.107e7 A`, concentrated at the 671--692 low-impedance switch.
  Switch projection changes those same-state metrics to `0.006699 p.u.` and
  `11865 A`. The physical-loss model's switch-projected metrics are
  `0.006667 p.u.` and `930.76 A`.
- Added `src/physics/linear_projection.py`. It solves the nine linear KCL
  equations for sourcebus, 650 and rg60 phase nodes from the declared-input
  Y/Norton fixture while holding the remaining voltage estimates fixed.
  It is differentiable and is a local algebraic correction, not an AC power-
  flow solution. A test verifies its selected-node KCL residual is below
  `1e-5 A` after correction.
- On the final **same corrected state**, ordinary-test supervised metrics are
  voltage MAE `0.006390 p.u.`, KCL RMS `263.54 A`; physical-loss metrics are
  `0.006520 p.u.`, `237.47 A`. Stress-test supervised metrics are
  `0.015622 p.u.`, `515.72 A`; physical-loss metrics are `0.015173 p.u.`,
  `467.05 A`. Thus physical-loss training reduces corrected-state KCL but
  does not improve every voltage-MAE comparison.
- The largest remaining node RMS is now 632.3. All corrected-state KCL
  values remain far above the solved-reference level, so no network solution,
  safety feasibility or solver speedup is claimed. Next: investigate a
  complete solver correction from these initial states with explicit
  convergence and residual checks.
- All 13 project tests pass.

## Stage 3j — independent nonlinear AC correction benchmark

- Added `src/physics/reduced_solver.py`: a double-precision damped Newton
  corrector for the fixed-tap complete AC equations. It eliminates nine
  source/regulator phase voltages by linear KCL and the three 692 switch
  phase voltages by local KCL, then solves 29 complex remaining unknowns.
  A result is accepted only if the **full 41-node** maximum KCL residual is
  at most `0.01 A`. This is an independent research solver, not OpenDSS with
  a user-supplied voltage initialization.
- Added `src/physics/benchmark_solver.py`. On 45 ordinary and 45 stress
  held-out fixed-tap scenarios, all three starts (balanced flat, supervised
  GNN plus projections, physics-loss GNN plus projections) succeeded:
  45/45 on each split and method. The flat start took 4.0 iterations on
  average; supervised GNN took 3.0 ordinary and 3.067 stress; physics-loss
  GNN took 3.0 ordinary and 3.044 stress. Thus the two GNN starts reduce
  iterations, but this dataset does **not** demonstrate a convergence-rate
  advantage or nonconvergence recovery.
- Mean full pipeline wall time, including declared-input fixture extraction,
  inference/projection, and correction: ordinary flat `0.18594 s`, supervised
  `0.13151 s`, physics-loss `0.13009 s`; stress flat `0.18518 s`, supervised
  `0.13470 s`, physics-loss `0.13256 s`. These are one CPU run and not yet a
  statistical speed study. Shared fixture cost is charged to every method.
- Every accepted state passed the full KCL tolerance. The largest final
  voltage-magnitude MAE against the OpenDSS reference, computed per case,
  was about `6.21e-6 p.u.` across the benchmark. This compares two solvers
  on the same declared fixed-tap circuit; it is not the raw GNN accuracy.
- Result records include status, iteration count, function evaluations,
  initial/final residual, fixture time, inference/projection time, solver
  time, full pipeline time and reference-voltage error in
  `results/stage3_solver/benchmark.json`. All 14 project tests pass.
- Next: independently check branch currents/P/Q and operating limits on the
  corrected state, repeat timing runs if a performance claim is needed,
  then decide whether the stage-3 core is ready before starting the separate
  stage-4 convergence-classification dataset.
