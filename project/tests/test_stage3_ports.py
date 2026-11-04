"""Cross-device checks for the independently calculated terminal currents."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "physics"))
from audit_ports import audit_one
from audit_nodal_kcl import audit_one as nodal_audit_one


def test_source_generator_and_linear_device_ports():
    # A DG scenario and a high-load scenario exercise both source and
    # generator treatment, regulator taps, capacitors and transformer ports.
    import json
    rows = [json.loads(line) for line in (ROOT / "data/raw/ieee13/samples.jsonl").open(encoding="utf-8")]
    chosen = [next(r for r in rows if r["sample_id"] == key)
              for key in ("g000_v0", "g299_v2")]
    checked = [item for r in chosen for item in audit_one(r["scenario"])]
    by_method = {}
    for row in checked:
        by_method.setdefault(row["method"], []).append(row["relative_error"])
    assert max(by_method["Yprim"]) < 1e-8
    assert max(by_method["source_Norton"]) < 1e-5
    assert max(by_method["generator_PQ_wye"]) < 2e-4
    # Current low-voltage load approximation is monitored, not certified as
    # an exact physics loss. Its largest error is discussed in README.
    assert max(by_method["model1_wye"]) < 5e-3
    assert max(by_method["model5_wye"]) < 5e-3


def test_independent_device_kcl_at_solved_voltage():
    import json
    rows = [json.loads(line) for line in (ROOT / "data/raw/ieee13/samples.jsonl").open(encoding="utf-8")]
    for key in ("g000_v0", "g299_v2"):
        params = next(r["scenario"] for r in rows if r["sample_id"] == key)
        result = nodal_audit_one(params)
        assert result["node_count"] == 41
        assert result["max_relative"] < 2e-4
