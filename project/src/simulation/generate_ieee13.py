"""Generate reproducible IEEE-13 three-phase power-flow samples.

Run from project root: python src/simulation/generate_ieee13.py --out data/raw/ieee13
The script never changes the source DSS files. Raw sample records are JSONL; tabular
CSV files are views for inspection, not a replacement for the complex quantities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import dss
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CIRCUIT = ROOT / "networks" / "ieee13" / "IEEE13Nodeckt.dss"
PHASES = {1: "A", 2: "B", 3: "C"}


def pairs(values: Any) -> list[list[float]]:
    """Convert OpenDSS interleaved real/imag or P/Q arrays to N x 2."""
    a = np.asarray(values, dtype=np.float64)
    if a.size % 2:
        raise ValueError("Odd-sized complex/power array from OpenDSS")
    return a.reshape(-1, 2).tolist()


def finite_tree(obj: Any) -> bool:
    if isinstance(obj, dict):
        return all(finite_tree(v) for v in obj.values())
    if isinstance(obj, list):
        return all(finite_tree(v) for v in obj)
    if isinstance(obj, (float, int)):
        return math.isfinite(obj)
    return True


def scenario(group: int, variant: int, seed: int) -> dict[str, Any]:
    """Create grouped scenarios; related variants share a base operating point."""
    rng = np.random.default_rng(seed + 1009 * group)
    # Test-only stress groups are kept outside the ordinary training range.
    if group >= 285:
        load = rng.uniform(1.30, 1.65)
        skew = rng.uniform(0.25, 0.45)
        regime = "stress_holdout"
    else:
        load = rng.uniform(0.65, 1.30)
        skew = rng.uniform(0.03, 0.25)
        regime = "ordinary"
    phase_factor = rng.uniform(1 - skew, 1 + skew, size=3)
    phase_factor /= phase_factor.mean()
    return {
        "group_id": f"g{group:03d}",
        "variant": variant,
        "regime": regime,
        "load_scale": float(load * (1 + (variant - 1) * 0.015)),
        "phase_factor_abc": phase_factor.tolist(),
        "dg_kw": float(rng.choice([0, 150, 300]) * (1 + (variant - 1) * 0.02)),
        "cap1_enabled": bool(group % 4 != 0),
        "base_seed": seed + 1009 * group,
    }


def split_for_group(group: int) -> str:
    if group < 240:
        return "train"
    if group < 270:
        return "val"
    return "test"


def load_phase_factor(bus: str, factors: list[float]) -> float:
    """For delta loads, average the two terminal-phase factors."""
    nodes = [int(x) for x in bus.split(".")[1:] if x.isdigit()]
    nodes = [n for n in nodes if n in PHASES]
    return float(np.mean([factors[n - 1] for n in nodes])) if nodes else 1.0


def apply_scenario(circuit: Any, params: dict[str, Any], base_loads: dict[str, Any]) -> dict[str, Any]:
    loads = circuit.Loads
    applied = {}
    for name, base in base_loads.items():
        loads.Name = name
        factor = params["load_scale"] * load_phase_factor(base["bus"], params["phase_factor_abc"])
        loads.kW = base["kw"] * factor
        loads.kvar = base["kvar"] * factor
        applied[name] = {"bus": base["bus"], "model": base["model"], "phases": base["phases"],
                         "kw": float(loads.kW), "kvar": float(loads.kvar)}
    if params["dg_kw"] > 0:
        # Fixed three-phase PQ injection; no generator exists in the source circuit.
        dss.DSS.Text.Command = (
            f"New Generator.DG675 phases=3 bus1=675.1.2.3 kv=4.16 "
            f"kW={params['dg_kw']:.8f} kvar=0 model=1"
        )
    if not params["cap1_enabled"]:
        dss.DSS.Text.Command = "Disable Capacitor.Cap1"
    return applied


def element_record(circuit: Any, name: str) -> dict[str, Any]:
    circuit.SetActiveElement(name)
    e = circuit.ActiveCktElement
    nc, nt = int(e.NumConductors), int(e.NumTerminals)
    node_order = [int(x) for x in e.NodeOrder]
    if len(node_order) != nc * nt:
        raise ValueError(f"Bad node order for {name}")
    currents = pairs(e.Currents)
    powers = pairs(e.Powers)
    if len(currents) != nc * nt or len(powers) != nc * nt:
        raise ValueError(f"Bad terminal array size for {name}")
    terminals = []
    for t in range(nt):
        terminals.append({
            "bus": e.BusNames[t].split(".")[0].lower(),
            "bus_spec": e.BusNames[t],
            "nodes": node_order[t * nc:(t + 1) * nc],
            "current_re_im_a": currents[t * nc:(t + 1) * nc],
            "power_p_q_kw_kvar": powers[t * nc:(t + 1) * nc],
        })
    return {
        "name": name.lower(), "kind": name.split(".")[0].lower(),
        "num_conductors": nc, "num_phases": int(e.NumPhases),
        "num_terminals": nt, "terminals": terminals,
    }


def bus_records(circuit: Any) -> list[dict[str, Any]]:
    out = []
    for name in circuit.AllBusNames:
        circuit.SetActiveBus(name)
        bus = circuit.ActiveBus
        nodes = [int(x) for x in bus.Nodes]
        vi = pairs(bus.Voltages)
        pu = np.asarray(bus.puVmagAngle, dtype=np.float64).reshape(-1, 2)
        for k, node in enumerate(nodes):
            if node not in PHASES:
                continue
            out.append({
                "bus": name.lower(), "phase": PHASES[node], "node": node,
                "v_re_im_v": vi[k], "v_mag_pu": float(pu[k, 0]),
                "v_angle_deg": float(pu[k, 1]), "v_base_ln_kv": float(bus.kVBase),
            })
    return out


def kcl_residual(elements: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum all enabled element terminal currents at every non-ground node."""
    sums: dict[str, complex] = {}
    through: dict[str, float] = {}
    for e in elements:
        for t in e["terminals"]:
            for n, iq in zip(t["nodes"], t["current_re_im_a"]):
                if n == 0:
                    continue
                key = f"{t['bus']}.{n}"
                z = complex(*iq)
                sums[key] = sums.get(key, 0j) + z
                through[key] = through.get(key, 0.0) + abs(z)
    abs_values = [abs(x) for x in sums.values()]
    rel_values = [abs(sums[k]) / max(through[k], 1.0) for k in sums]
    return {
        "max_abs_a": float(max(abs_values, default=0)),
        "max_relative": float(max(rel_values, default=0)),
        "by_node_abs_a": {k: float(abs(v)) for k, v in sums.items()},
    }


def line_metadata(circuit: Any) -> list[dict[str, Any]]:
    """Record line geometry and impedance matrices as exposed by OpenDSS.

    Rmatrix/Xmatrix/Cmatrix are OpenDSS per-length values in the line Units,
    not multiplied by Length. Phase order follows the terminal node order.
    """
    out = []
    lines = circuit.Lines
    for name in lines.AllNames:
        lines.Name = name
        circuit.SetActiveElement(f"Line.{name}")
        e = circuit.ActiveCktElement
        n = int(lines.Phases)
        out.append({
            "name": f"line.{name}".lower(), "bus_names": list(e.BusNames),
            "terminal_node_order": [int(x) for x in e.NodeOrder],
            "phases": n, "length": float(lines.Length), "units_code": int(lines.Units),
            "line_code": str(lines.LineCode),
            "r_matrix_per_length_ohm": np.asarray(lines.Rmatrix, dtype=float).reshape(n, n).tolist(),
            "x_matrix_per_length_ohm": np.asarray(lines.Xmatrix, dtype=float).reshape(n, n).tolist(),
            "c_matrix_per_length_nf": np.asarray(lines.Cmatrix, dtype=float).reshape(n, n).tolist(),
        })
    return out


def run_one(params: dict[str, Any]) -> dict[str, Any]:
    dss.DSS.Text.Command = f'Compile "{CIRCUIT}"'
    circuit = dss.DSS.ActiveCircuit
    # The source circuit contains a Solve and establishes baseline regulator taps.
    if not circuit.Solution.Converged:
        raise RuntimeError("Canonical IEEE13 base circuit did not converge")
    base_loads = {}
    for name in circuit.Loads.AllNames:
        circuit.Loads.Name = name
        circuit.SetActiveElement(f"Load.{name}")
        base_loads[name] = {
            "kw": float(circuit.Loads.kW), "kvar": float(circuit.Loads.kvar),
            "bus": circuit.ActiveCktElement.BusNames[0],
            "model": int(circuit.Loads.Model), "phases": int(circuit.Loads.Phases),
        }
    applied_loads = apply_scenario(circuit, params, base_loads)
    circuit.Solution.Solve()
    solution = circuit.Solution
    converged = bool(solution.Converged)
    names = list(circuit.AllElementNames)
    elements = []
    for name in names:
        circuit.SetActiveElement(name)
        if circuit.ActiveCktElement.Enabled:
            elements.append(element_record(circuit, name))
    buses = bus_records(circuit)
    lines = line_metadata(circuit)
    residual = kcl_residual(elements)
    record = {
        "sample_id": f"{params['group_id']}_v{params['variant']}",
        "split": split_for_group(int(params["group_id"][1:])),
        "scenario": params,
        "solver": {"converged": converged, "iterations": int(solution.Iterations),
                   "control_iterations": int(solution.ControlIterations),
                   "max_iterations": int(solution.MaxIterations),
                   "residual_source": "postprocessed_element_terminal_kcl"},
        "loads_base": base_loads,
        "loads_applied": applied_loads,
        "buses": buses,
        "elements": elements,
        "lines": lines,
        "kcl_residual": residual,
    }
    if not finite_tree(record):
        raise ValueError(f"Nonfinite data in {record['sample_id']}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "raw" / "ieee13")
    parser.add_argument("--groups", type=int, default=300)
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    if not 286 <= args.groups <= 300 or args.variants < 1:
        parser.error("groups must be 286..300 to preserve all predefined splits")
    # OpenDSS Compile changes the process working directory to the DSS folder.
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for group in range(args.groups):
        for variant in range(args.variants):
            records.append(run_one(scenario(group, variant, args.seed)))
    with (args.out / "samples.jsonl").open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    index = pd.DataFrame([{
        "sample_id": r["sample_id"], "group_id": r["scenario"]["group_id"],
        "split": r["split"], "regime": r["scenario"]["regime"],
        "load_scale": r["scenario"]["load_scale"], "dg_kw": r["scenario"]["dg_kw"],
        "cap1_enabled": r["scenario"]["cap1_enabled"],
        "converged": r["solver"]["converged"],
        "iterations": r["solver"]["iterations"],
        "control_iterations": r["solver"]["control_iterations"],
        "kcl_max_abs_a": r["kcl_residual"]["max_abs_a"],
        "kcl_max_relative": r["kcl_residual"]["max_relative"],
    } for r in records])
    index.to_csv(args.out / "index.csv", index=False)
    bus_rows = []
    for r in records:
        for b in r["buses"]:
            bus_rows.append({"sample_id": r["sample_id"], "bus": b["bus"], "phase": b["phase"],
                             "node": b["node"], "v_re_v": b["v_re_im_v"][0],
                             "v_im_v": b["v_re_im_v"][1], "v_mag_pu": b["v_mag_pu"],
                             "v_angle_deg": b["v_angle_deg"], "v_base_ln_kv": b["v_base_ln_kv"]})
    pd.DataFrame(bus_rows).to_csv(args.out / "bus_voltages.csv", index=False)
    branch_rows = []
    for r in records:
        for e in r["elements"]:
            if e["kind"] not in {"line", "transformer"}:
                continue
            for ti, t in enumerate(e["terminals"]):
                for n, iq, pq in zip(t["nodes"], t["current_re_im_a"], t["power_p_q_kw_kvar"]):
                    branch_rows.append({
                        "sample_id": r["sample_id"], "element": e["name"],
                        "terminal": ti + 1, "bus": t["bus"], "node": n,
                        "i_re_a": iq[0], "i_im_a": iq[1],
                        "p_kw": pq[0], "q_kvar": pq[1],
                    })
    pd.DataFrame(branch_rows).to_csv(args.out / "branch_terminals.csv", index=False)
    manifest = {
        "seed": args.seed, "groups": args.groups, "variants": args.variants,
        "samples": len(records), "source_dss_sha256": hashlib.sha256(CIRCUIT.read_bytes()).hexdigest(),
        "dss_version": dss.DSS.Version.strip(),
        "split_rule": "groups 0-239 train, 240-269 val, 270-284 ordinary test, 285-299 stress test",
        "notes": "KCL residual is postprocessed, not a solver-reported residual. No topology changes or parameter missingness in version 1.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(index.groupby(["split", "converged"]).size().to_string())
    print(f"samples={len(records)} buses={len(records[0]['buses'])} branch_rows={len(branch_rows)}")
    print(f"max KCL relative residual={index.kcl_max_relative.max():.3g}")
    print(f"saved to {args.out.resolve()}")


if __name__ == "__main__":
    main()
