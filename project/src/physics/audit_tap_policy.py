"""Quantify regulator tap variation and a train-only fixed-tap baseline.

Solved taps are used here strictly as audit labels. The script does not
provide them to the voltage predictor or a training-time AC residual.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REGULATORS = ("reg1", "reg2", "reg3")


def tap_step(sample: dict, name: str) -> int:
    tap = sample["transformer_taps_after_solve"][name][1]
    # OpenDSS regulator taps in this circuit use 0.00625 p.u. increments.
    step = round((tap - 1.0) / 0.00625)
    if abs(tap - (1.0 + step * 0.00625)) > 1e-8:
        raise ValueError(f"Unexpected {name} tap {tap} in {sample['sample_id']}")
    return step


def main() -> None:
    path = ROOT / "data/raw/ieee13/samples.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    train = [row for row in rows if row["split"] == "train"]
    modes = {name: Counter(tap_step(row, name) for row in train).most_common(1)[0][0]
             for name in REGULATORS}
    report = {"samples": len(rows), "tap_step_pu": 0.00625,
              "train_mode_steps": modes, "by_split": {}}
    for split in ("train", "val", "test"):
        subset = [row for row in rows if row["split"] == split]
        report["by_split"][split] = {
            "count": len(subset),
            "unique_joint_states": len({tuple(tap_step(row, name) for name in REGULATORS)
                                        for row in subset}),
            "per_regulator": {
                name: {
                    "observed_steps": sorted({tap_step(row, name) for row in subset}),
                    "fixed_mode_accuracy": sum(tap_step(row, name) == modes[name]
                                               for row in subset) / len(subset),
                    "mean_absolute_step_error": sum(abs(tap_step(row, name) - modes[name])
                                                    for row in subset) / len(subset),
                } for name in REGULATORS
            },
            "joint_fixed_mode_accuracy": sum(
                all(tap_step(row, name) == modes[name] for name in REGULATORS)
                for row in subset) / len(subset),
        }
    out = ROOT / "results/stage3_edge/tap_policy_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
