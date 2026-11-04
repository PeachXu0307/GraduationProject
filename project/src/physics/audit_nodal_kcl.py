"""Audit nodal KCL from independently calculated device port currents.

This validates the device composition at solved OpenDSS voltages. Regulator
tap positions are the solved operating state and are used only for audit;
the script does not define a training-time tap policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import dss

from audit_ports import (ROOT, CIRCUIT, apply_scenario, base_loads,
                         yprim_current, load_current, generator_current,
                         source_current)


def audit_one(params: dict) -> dict:
    dss.DSS.Text.Command = f'Compile "{CIRCUIT}"'
    circuit = dss.DSS.ActiveCircuit
    apply_scenario(circuit, params, base_loads(circuit))
    circuit.Solution.Solve()
    if not circuit.Solution.Converged:
        raise RuntimeError("Scenario did not converge")
    currents: dict[str, complex] = {}
    through: dict[str, float] = {}
    for name in list(circuit.AllElementNames):
        circuit.SetActiveElement(name)
        element = circuit.ActiveCktElement
        if not element.Enabled:
            continue
        kind, short_name = name.split('.', 1)
        kind = kind.lower()
        if kind in {'line', 'transformer', 'capacitor'}:
            calc, _ = yprim_current(circuit, name)
        elif kind == 'load':
            calc, _, _, _ = load_current(circuit, short_name)
            # Load audit returns the first terminal's phase conductors.
            # Ground return conductor, when present, balances their sum.
            circuit.SetActiveElement(name)
            element = circuit.ActiveCktElement
            n = int(element.NumConductors * element.NumTerminals)
            if len(calc) < n:
                calc = np.concatenate([calc, np.full(n-len(calc), -calc.sum())])
        elif kind == 'generator':
            calc, _ = generator_current(circuit, short_name)
        elif kind == 'vsource':
            calc, _ = source_current(circuit, short_name)
        elif kind == 'regcontrol':
            # Control objects have no electrical terminal current. Their
            # effect is already reflected in the transformer tap/Yprim.
            continue
        else:
            raise ValueError(f"Unhandled enabled device {name}")
        circuit.SetActiveElement(name)
        element = circuit.ActiveCktElement
        nc = int(element.NumConductors)
        if len(calc) != nc * int(element.NumTerminals):
            raise ValueError(f"Terminal size mismatch for {name}")
        for t, bus_spec in enumerate(element.BusNames):
            bus = bus_spec.split('.')[0].lower()
            for j in range(nc):
                node = int(element.NodeOrder[t*nc+j])
                if node == 0:
                    continue
                key = f'{bus}.{node}'
                value = calc[t*nc+j]
                currents[key] = currents.get(key, 0j) + value
                through[key] = through.get(key, 0.0) + abs(value)
    abs_err = {key: abs(value) for key, value in currents.items()}
    rel_err = {key: abs(value)/max(through[key], 1.0) for key, value in currents.items()}
    return {'max_abs_a': float(max(abs_err.values(), default=0)),
            'max_relative': float(max(rel_err.values(), default=0)),
            'worst_node': max(rel_err, key=rel_err.get),
            'node_count': len(currents)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--raw', type=Path, default=ROOT/'data/raw/ieee13')
    p.add_argument('--sample-count', type=int, default=30)
    p.add_argument('--out', type=Path, default=ROOT/'results/stage3_edge/nodal_kcl_audit.json')
    args = p.parse_args()
    rows = [json.loads(s) for s in (args.raw/'samples.jsonl').open(encoding='utf-8')]
    ids = np.linspace(0, len(rows)-1, args.sample_count, dtype=int)
    results = [{'sample_id': rows[i]['sample_id'], **audit_one(rows[i]['scenario'])} for i in ids]
    report = {'samples': len(results), 'max_abs_a': max(x['max_abs_a'] for x in results),
              'max_relative': max(x['max_relative'] for x in results), 'records': results,
              'scope': 'Solved voltages and solved regulator taps used only for audit'}
    args.out = args.out.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
