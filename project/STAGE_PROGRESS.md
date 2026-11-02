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

### Next task, after stage 1 review

Stage 2: compare bus-level matrix edges with phase-node expansion; define
bus/phase and conductor ordering, adjacency and incidence matrices, convert
the raw records to graph tensors, and check graph-device mapping against
OpenDSS. Do not silently treat transformer connections as ordinary lines.
