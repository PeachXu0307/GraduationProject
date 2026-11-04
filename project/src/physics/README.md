# Stage 3 physical-residual audit

The Stage 1 records now preserve each load's wye/delta connection and the
transformer/regulator tap positions after each solve. The tap position is an
**outcome of the regulator control process**. It can be used to audit a solved
operating point, but must not be passed to a predictor as though it were a
known input. A future physics-informed model needs a declared tap policy,
fixed taps, or a separate tap predictor.

The repeatable 30-scenario audit in `audit_ports.py` shows that `Yprim @ V`
reproduces 562 line, transformer, and capacitor terminal-current records with
maximum relative error `9.50e-12`. For loads,
the model must distinguish constant PQ (`Model=1`), constant impedance
(`Model=2`), and constant current (`Model=5`), including line-to-line voltage
and conductor-current mapping for delta connections. A simple calculation
using the recorded load setpoints agrees with terminal currents to roughly
1e-4 relative in ordinary conditions. The audit detected low-voltage model
switching: a naive nominal-load formula reached 7% relative error on one
stress sample. The corrected OpenDSS low-voltage interpolation is now
implemented separately for Model 1 and Model 5. Across all 900 scenarios,
the largest relative current errors are `1.42e-4` for Model 1 wye and
`1.13e-4` for Model 5 wye. The remaining differences are small numerical
or model-parameter effects and must be stated when setting a physics-loss
tolerance.

The OpenDSS terminal currents are audit targets, never prediction inputs.
The current edge-aware GNN is supervised only. A full AC physical loss is
pending an independent device-by-device residual audit for loads, generator,
capacitors, transformers, regulators, and source treatment.

The generator audit uses the declared fixed-PQ setpoint and terminal voltage;
20 DG scenarios had maximum relative current error `9.91e-5`. For the
three-phase source, the declared kV, pu and angle define an internal EMF
vector. `Yprim @ V` minus the corresponding Norton injection reproduces its
terminal current with maximum relative error `2.29e-6` in 30 scenarios.
This source formula is specific to the present three-phase source connection.

Run the updated audit and focused test from `project/`:

```powershell
.venv\Scripts\python.exe src\physics\audit_ports.py --sample-count 30 --out results\stage3_edge\port_audit_30.json
.venv\Scripts\python.exe -m pytest tests\test_stage3_ports.py -q -p no:faulthandler -p no:cacheprovider
```

The stress example `g299_v2` has a Model 1 wye load at bus 675 phase C
around 0.920 of its 2.4 kV nominal voltage. The EPRI OpenDSS documentation
states that the load leaves its nominal model below `Vminpu`. The public
OpenDSS `Load.pas` source shows a current interpolation between `Vlowpu`
and `Vminpu`. Model 1 joins the constant-PQ current at the upper boundary;
Model 5 joins its constant-current relation. The full audit report is
`results/stage3_edge/port_audit_900.json`. A complete physics loss still
needs a declared regulator state policy and an independent nodal KCL audit.

The independent device-current KCL audit is implemented in
`audit_nodal_kcl.py`. Across all 900 solved scenarios, summing the declared
device equations at each non-ground phase node gives maximum relative KCL
residual `1.00e-4` and maximum absolute residual `0.213 A`. This is an audit
at solved voltages with solved regulator states; it is not yet the
differentiable prediction residual used during GNN training.

The prototype differentiable implementation is in `partial_ac.py`. It is
currently a tensor/gradient prototype only: it includes the line Ybus and
load equations, returns an explicit covered-node mask, and reports omitted
source/transformer/regulator/capacitor/generator devices. A direct solved-
voltage numerical check still shows large residuals at some load/branch
junctions, so this module must not be used as a physics-loss term yet. The
remaining issue is being isolated against the exact OpenDSS phase and device
current orientation before training.
