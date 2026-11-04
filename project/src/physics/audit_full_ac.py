"""Audit complete fixed-tap AC KCL on all solved IEEE-13 scenarios."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from audit_partial_ac import ROOT, solved_input
from full_ac import build_fixture, full_ac_residual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/ieee13_fixed_taps")
    parser.add_argument("--out", type=Path, default=ROOT / "results/stage3_edge/full_ac_audit_fixed_900.json")
    parser.add_argument("--count", type=int, default=0,
                        help="Zero audits all rows; positive values audit the first N")
    args = parser.parse_args()
    topology = json.loads((ROOT / "data/graphs/ieee13/topology.json").read_text(encoding="utf-8"))
    records = []
    with (args.raw / "samples.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            sample = json.loads(line)
            fixture = build_fixture(sample, topology)
            pred, bases = solved_input(sample, topology)
            with torch.no_grad():
                result = full_ac_residual(pred, sample, topology, bases, fixture)
            magnitude = result["residual_a"].abs()
            worst = int(magnitude.argmax())
            records.append({"sample_id": sample["sample_id"], "split": sample["split"],
                            "rms_a": float(result["residual_rms_a"]),
                            "max_a": float(magnitude[worst]),
                            "worst_node": topology["phase_node_order"][worst]})
            if args.count and len(records) >= args.count:
                break
    worst = max(records, key=lambda row: row["max_a"])
    report = {"sample_count": len(records), "phase_nodes_per_sample": len(topology["phase_node_order"]),
              "max_rms_a": max(row["rms_a"] for row in records),
              "max_a": worst["max_a"], "worst_sample": worst["sample_id"],
              "worst_node": worst["worst_node"],
              "scope": "Full 41-node KCL evaluated at solved voltages, with pre-solve input Yprim fixture",
              "records": records}
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
