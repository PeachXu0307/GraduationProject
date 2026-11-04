"""Pilot a declared fixed-tap, controls-off IEEE-13 operating policy.

This is a distinct physical scenario from the existing regulated dataset.
The published circuit's commented alternate solution supplies the three
fixed taps. No solved tap labels or solved voltages are used as solver inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import dss
import numpy as np

from audit_ports import ROOT, CIRCUIT, apply_scenario, base_loads


FIXED_TAPS = {"reg1": 1.0625, "reg2": 1.05, "reg3": 1.06875}


def solve_fixed(params: dict) -> dict:
    dss.DSS.Text.Command = f'Compile "{CIRCUIT}"'
    circuit = dss.DSS.ActiveCircuit
    apply_scenario(circuit, params, base_loads(circuit))
    for name, tap in FIXED_TAPS.items():
        circuit.Transformers.Name = name
        circuit.Transformers.Wdg = 2
        circuit.Transformers.Tap = tap
    dss.DSS.Text.Command = "Set Controlmode=OFF"
    circuit.Solution.Solve()
    voltages = {}
    for name in circuit.AllBusNames:
        circuit.SetActiveBus(name)
        bus = circuit.ActiveBus
        pairs = np.asarray(bus.puVmagAngle, dtype=float).reshape(-1, 2)
        for node, (mag, _) in zip(bus.Nodes, pairs):
            if node in (1, 2, 3):
                voltages[f"{name.lower()}.{node}"] = float(mag)
    return {"converged": bool(circuit.Solution.Converged),
            "iterations": int(circuit.Solution.Iterations),
            "voltage_mag_pu": voltages}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=30)
    args = parser.parse_args()
    samples = [json.loads(line) for line in (ROOT / "data/raw/ieee13/samples.jsonl").open(encoding="utf-8")]
    indices = np.linspace(0, len(samples)-1, args.count, dtype=int)
    records = []
    for index in indices:
        sample = samples[int(index)]
        fixed = solve_fixed(sample["scenario"])
        ref = {f"{row['bus']}.{row['node']}": row["v_mag_pu"] for row in sample["buses"]}
        shared = ref.keys() & fixed["voltage_mag_pu"].keys()
        deviations = [abs(ref[key] - fixed["voltage_mag_pu"][key]) for key in shared]
        records.append({"sample_id": sample["sample_id"], "split": sample["split"],
                        "converged": fixed["converged"], "iterations": fixed["iterations"],
                        "compared_phase_nodes": len(shared),
                        "max_voltage_difference_pu": max(deviations)})
    summary = {"samples": len(records), "fixed_taps": FIXED_TAPS,
               "converged_count": sum(row["converged"] for row in records),
               "max_voltage_difference_from_regulated_pu": max(
                   row["max_voltage_difference_pu"] for row in records),
               "scope": "Fixed taps with controls off; a distinct scenario, not a replacement for regulated labels",
               "records": records}
    out = ROOT / "results/stage3_edge/fixed_tap_pilot.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
