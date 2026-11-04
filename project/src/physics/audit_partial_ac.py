"""Audit the covered partial AC residual at solved voltages for all scenarios.

This is a validation of the line/load equations, not a training script.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import torch

from partial_ac import partial_ac_residual


ROOT = Path(__file__).resolve().parents[2]


def solved_input(sample: dict, topology: dict) -> tuple[torch.Tensor, torch.Tensor]:
    bases = {row["bus"]: row["v_base_ln_kv"] * 1000 for row in sample["buses"]}
    volts = {(row["bus"], row["node"]): complex(*row["v_re_im_v"])
             for row in sample["buses"]}
    bus_names = topology["bus_names"]
    voltage_base = torch.tensor([bases[name] for name in bus_names], dtype=torch.float64)
    pred_pu = torch.tensor(
        [[[volts.get((name, phase), 0j).real / bases[name],
           volts.get((name, phase), 0j).imag / bases[name]]
          for phase in (1, 2, 3)] for name in bus_names], dtype=torch.float64
    ).reshape(-1, 6)
    return pred_pu, voltage_base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/ieee13")
    parser.add_argument("--out", type=Path, default=ROOT / "results/stage3_edge/partial_ac_audit_900.json")
    args = parser.parse_args()
    topology = json.loads((ROOT / "data/graphs/ieee13/topology.json").read_text(encoding="utf-8"))
    input_path = args.raw / "samples.jsonl"
    records = []
    for line in input_path.open(encoding="utf-8"):
        sample = json.loads(line)
        pred_pu, bases = solved_input(sample, topology)
        with torch.no_grad():
            result = partial_ac_residual(pred_pu, sample, topology, bases)
        residual = result["residual_a"][result["covered_mask"]].abs()
        records.append({"sample_id": sample["sample_id"],
                        "covered_phase_nodes": int(residual.numel()),
                        "rms_a": float(result["residual_rms_a"]),
                        "max_a": float(residual.max())})
    report = {"sample_count": len(records),
              "max_rms_a": max(row["rms_a"] for row in records),
              "max_a": max(row["max_a"] for row in records),
              "worst_sample": max(records, key=lambda row: row["max_a"])["sample_id"],
              "scope": "Line and load KCL at covered conductors only, using solved voltages",
              "records": records}
    path = args.out.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
