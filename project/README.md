# Stage 1: IEEE 13 unbalanced three-phase data

## Goal and prerequisites

Generate a fixed-topology, grouped-scenario dataset with OpenDSS reference
voltages and element terminal quantities. This stage does not train a GNN.
The source feeder is described in `networks/ieee13/SOURCE.md`.

On Windows, from this directory:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-stage1.txt
.venv\Scripts\python.exe src\simulation\generate_ieee13.py
.venv\Scripts\python.exe src\simulation\check_dataset.py data\raw\ieee13
.venv\Scripts\python.exe src\visualization\plot_dataset.py data\raw\ieee13 --out results\stage1
.venv\Scripts\python.exe -m pytest tests\test_stage1.py -q
```

If `py -3.12` is unavailable, use an installed Python 3.12 executable to
create `.venv`. `dss-python==0.15.7` provides DSS C-API 0.14.5 and does
not require the OpenDSS GUI. On this machine the project environment was
created from the Codex bundled Python 3.12 executable. The stage 1 dependencies
are pinned in `requirements-stage1.txt`.

## Source and scenario definition

The unmodified source DSS circuit contains a 115/4.16 kV transformer, a
4.16/0.48 kV transformer, three single-phase regulators, 12 lines, 15 loads,
two capacitor definitions, wye and delta connections, and mixed one-, two-,
and three-phase branches. OpenDSS exposes 16 buses including `sourcebus` and
`rg60`, and 41 energized bus-phase voltage entries. These counts are from
the exact downloaded circuit, not the IEEE feeder's short name.

Each scenario group fixes its base load multiplier, phase factors, DG choice,
and capacitor state. Three small load/DG variants share that group. The new
DG is a fixed three-phase PQ generator at bus 675; it is a study perturbation,
not a component of the source IEEE case. The capacitor switch is the only
discrete operating-state change. Line topology and impedances remain fixed.

Groups 0-239 are train (720 samples), 240-269 validation (90), 270-284
ordinary test (45), and 285-299 stress test (45). The stress range uses
larger load and phase skew; related variants never cross splits. All 900
generated cases converged, so this dataset supplies regression targets only.
It does **not** contain valid nonconvergence-class labels. The simulated
imbalances are scenario perturbations, not measured real-world profiles.

## Files and fields

`data/raw/ieee13/` is generated and ignored by Git. It contains:

| File | Content and dimensions for default run |
|---|---|
| `samples.jsonl` | 900 complete sample objects. Each has scenario, solver status, 41 bus-phase states, enabled element terminals, 12 line metadata records, and KCL audit. |
| `index.csv` | 900 rows: sample/group IDs, split, scenario regime, load scale, DG kW, capacitor state, convergence, iteration counts, KCL residual. |
| `bus_voltages.csv` | 36,900 rows = 900 × 41. Bus, phase, real/imag volts, voltage p.u., angle degrees, line-neutral base kV. |
| `branch_terminals.csv` | 77,400 rows = 900 × 86 terminal conductors for lines and transformers. Real/imag amps; terminal P kW and Q kvar. Terminal 1 and 2 use the OpenDSS sign convention, so do not sum them as though both point in the same direction. |
| `manifest.json` | Seed, source hash, DSS version, split policy. |

In JSONL, `lines[*]` includes phase-ordered R/X/C matrices as exposed by
OpenDSS. These are **per unit length in OpenDSS line Units**, not total
branch impedance; `length` and `units_code` must be applied correctly in
later physics work. Transformer/regulator matrices require separate device
models and are not inferred from line matrices. `elements[*]` gives every
enabled element's terminal current and power, including loads, capacitors,
generators, and sources. For a two-phase branch, conductor order follows
`terminal_node_order`, which need not be A/B/C order.

`iterations` and `control_iterations` are OpenDSS-reported values from the
final solve. `kcl_residual` is **postprocessed from all enabled element terminal
currents**; it is not an OpenDSS-reported solver residual. A converged flag
does not imply that operating limits are satisfied. No line ampacity claims
are made because the supplied case does not establish a consistent set of
engineering ratings for every device.

## Quality checks and current results

`check_dataset.py` verifies IDs, group leakage, fixed bus-phase and branch
topology, finite values, split regime, convergence, and normalized KCL
residual below 1e-3. The default run passed: 900/900 converged; 41 phase
nodes and 17 line/transformer elements per sample; largest normalized KCL
residual 6.70e-5. The plots under `results/stage1/` show voltage p.u.
distribution and iterations versus load scale. The observed voltage range
is roughly 0.896-1.100 p.u.; these are observations, not yet a formal
voltage-violation rate because the limits must be stated explicitly.

## Limitations and next stage

The initial dataset has one physical topology, synthetic operating-point
perturbations, and no solver-failure samples. It is adequate to verify the
stage 2 graph construction and train an initial fixed-topology predictor.
Stage 2 must check the mapping from buses and terminal conductors to graph
nodes and edge matrices. Before stage 4, a separate and explicitly labeled
failure dataset must be developed with real solver traces; a `Converged=False`
flag alone is not a proof of physical infeasibility.
