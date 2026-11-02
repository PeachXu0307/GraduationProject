# Stage 3 physical-residual audit

The Stage 1 records now preserve each load's wye/delta connection and the
transformer/regulator tap positions after each solve. The tap position is an
**outcome of the regulator control process**. It can be used to audit a solved
operating point, but must not be passed to a predictor as though it were a
known input. A future physics-informed model needs a declared tap policy,
fixed taps, or a separate tap predictor.

The repeatable 12-scenario audit in `audit_ports.py` shows that `Yprim @ V`
reproduces 224 line, transformer, and capacitor terminal-current records with
maximum relative error `7.68e-12`. For loads,
the model must distinguish constant PQ (`Model=1`), constant impedance
(`Model=2`), and constant current (`Model=5`), including line-to-line voltage
and conductor-current mapping for delta connections. A simple calculation
using the recorded load setpoints agrees with terminal currents to roughly
1e-4 relative in ordinary conditions. The audit detected low-voltage model
switching: a naive nominal-load formula reached 7% relative error on one
stress sample. A first Vmin/Vmax fallback reduced the maximum to about
0.36% for Model 1 wye and 0.20% for Model 5 wye; the remaining difference
must be resolved before calling that relation an exact training constraint.

The OpenDSS terminal currents are audit targets, never prediction inputs.
The current edge-aware GNN is supervised only. A full AC physical loss is
pending an independent device-by-device residual audit for loads, generator,
capacitors, transformers, regulators, and source treatment.
