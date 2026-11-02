"""Audit the raw IEEE-13 dataset; fails with a nonzero status on violations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def check(folder: Path) -> dict:
    index = pd.read_csv(folder / "index.csv")
    rows = [json.loads(x) for x in (folder / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(index) == index.sample_id.nunique(), "Sample count or ID mismatch"
    assert set(index.split) <= {"train", "val", "test"}
    assert index.groupby("group_id").split.nunique().max() == 1, "Scenario group leaks across splits"
    assert not index.isna().any().any(), "Missing index field"
    assert np.isfinite(index[["load_scale", "dg_kw", "iterations", "kcl_max_abs_a", "kcl_max_relative"]]).all().all()
    assert all(r["solver"]["converged"] for r in rows), "Nonconverged sample needs a separate unlabeled partition"
    assert index.kcl_max_relative.max() < 1e-3, "Large postprocessed KCL residual"
    keys0 = {(b["bus"], b["phase"]) for b in rows[0]["buses"]}
    branches0 = {e["name"] for e in rows[0]["elements"] if e["kind"] in {"line", "transformer"}}
    for r in rows:
        assert len(keys0) == len(r["buses"]), "Duplicate or missing phase node"
        assert {(b["bus"], b["phase"]) for b in r["buses"]} == keys0, "Topology/phase-node drift"
        assert {e["name"] for e in r["elements"] if e["kind"] in {"line", "transformer"}} == branches0
        for b in r["buses"]:
            assert np.isfinite(b["v_re_im_v"]).all()
            assert b["v_base_ln_kv"] > 0
    assert set(index.loc[index.split == "test", "regime"]) == {"ordinary", "stress_holdout"}
    assert index.loc[index.split != "test", "regime"].eq("ordinary").all()
    return {
        "samples": len(rows), "split_counts": index.split.value_counts().to_dict(),
        "converged": int(index.converged.sum()), "phase_nodes": len(keys0),
        "branch_elements": len(branches0),
        "max_kcl_relative": float(index.kcl_max_relative.max()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    a = p.parse_args()
    print(json.dumps(check(a.folder), indent=2))


if __name__ == "__main__":
    main()
