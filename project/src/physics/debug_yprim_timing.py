"""Check whether OpenDSS exposes fresh regulator Yprim before a solve."""
from __future__ import annotations

import json
import numpy as np
import dss

from audit_ports import ROOT, CIRCUIT, apply_scenario, base_loads
from full_ac import FIXED_REGULATOR_TAPS


def main() -> None:
    sample = json.loads((ROOT / "data/raw/ieee13_fixed_taps/samples.jsonl").read_text(
        encoding="utf-8").splitlines()[0])
    dss.DSS.Text.Command = f'Compile "{CIRCUIT}"'
    circuit = dss.DSS.ActiveCircuit
    apply_scenario(circuit, sample["scenario"], base_loads(circuit))
    for name, tap in FIXED_REGULATOR_TAPS.items():
        circuit.Transformers.Name = name
        circuit.Transformers.Wdg = 2
        circuit.Transformers.Tap = tap
    dss.DSS.Text.Command = "Set Controlmode=OFF"
    before = {}
    for name in FIXED_REGULATOR_TAPS:
        circuit.SetActiveElement(f"Transformer.{name}")
        before[name] = np.asarray(circuit.ActiveCktElement.Yprim, dtype=float)
    circuit.Solution.BuildYMatrix(1, 1)
    built = {}
    for name in FIXED_REGULATOR_TAPS:
        circuit.SetActiveElement(f"Transformer.{name}")
        built[name] = np.asarray(circuit.ActiveCktElement.Yprim, dtype=float)
    circuit.Solution.Solve()
    for name in FIXED_REGULATOR_TAPS:
        circuit.SetActiveElement(f"Transformer.{name}")
        after = np.asarray(circuit.ActiveCktElement.Yprim, dtype=float)
        print(name, "before/post", float(np.max(np.abs(before[name] - after))),
              "built/post", float(np.max(np.abs(built[name] - after))))


if __name__ == "__main__":
    main()
