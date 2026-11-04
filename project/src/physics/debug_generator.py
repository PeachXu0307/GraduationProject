"""Inspect generator current mismatch at an identified stress scenario."""
from __future__ import annotations

import json
import numpy as np
import dss

from audit_ports import ROOT, CIRCUIT, apply_scenario, base_loads, complex_pairs, element_voltages
from full_ac import FIXED_REGULATOR_TAPS


def main() -> None:
    sample = next(json.loads(line) for line in (ROOT / "data/raw/ieee13_fixed_taps/samples.jsonl").open(
        encoding="utf-8") if '"sample_id": "g290_v2"' in line)
    dss.DSS.Text.Command = f'Compile "{CIRCUIT}"'
    circuit = dss.DSS.ActiveCircuit
    apply_scenario(circuit, sample["scenario"], base_loads(circuit))
    for name, tap in FIXED_REGULATOR_TAPS.items():
        circuit.Transformers.Name = name
        circuit.Transformers.Wdg = 2
        circuit.Transformers.Tap = tap
    dss.DSS.Text.Command = "Set Controlmode=OFF"
    circuit.Solution.Solve()
    circuit.SetActiveElement("Generator.DG675")
    element = circuit.ActiveCktElement
    v = element_voltages(circuit, element)[:3]
    obs = complex_pairs(element.Currents)[:3]
    s = sample["scenario"]["dg_kw"] * 1000 / 3
    calc = -np.conj(s / v)
    print("voltage magnitude", np.abs(v).tolist())
    print("observed", obs.tolist())
    print("constant PQ", calc.tolist())
    print("absolute error", np.abs(calc - obs).tolist())
    print("generator model", circuit.Generators.Model,
          "Vminpu", circuit.Generators.Vminpu, "Vmaxpu", circuit.Generators.Vmaxpu)


if __name__ == "__main__":
    main()
