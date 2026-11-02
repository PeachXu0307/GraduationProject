"""Build phase-aware bus graphs from Stage 1 JSONL. Run from project root."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[2]
PHASES = {1: 0, 2: 1, 3: 2}


def topology(sample: dict) -> dict:
    buses = list(dict.fromkeys(b["bus"] for b in sample["buses"]))
    bus_index = {name: i for i, name in enumerate(buses)}
    lines = {line["name"]: line for line in sample["lines"]}
    branches = []
    for e in sample["elements"]:
        if e["kind"] not in {"line", "transformer"}:
            continue
        if len(e["terminals"]) != 2:
            raise ValueError(f"Expected two terminals for {e['name']}")
        a, b = e["terminals"]
        u, v = a["bus"], b["bus"]
        if u not in bus_index or v not in bus_index or u == v:
            raise ValueError(f"Invalid endpoint for {e['name']}")
        branch = {"id": len(branches), "name": e["name"], "kind": e["kind"],
                  "u": u, "v": v, "u_nodes": a["nodes"], "v_nodes": b["nodes"]}
        if e["kind"] == "line":
            line = lines[e["name"]]
            branch.update({"length": line["length"], "units_code": line["units_code"],
                           "r_matrix": line["r_matrix_per_length_ohm"],
                           "x_matrix": line["x_matrix_per_length_ohm"],
                           "c_matrix": line["c_matrix_per_length_nf"]})
        branches.append(branch)
    incidence = np.zeros((len(branches), len(buses)), dtype=np.float32)
    adjacency = np.zeros((len(buses), len(buses)), dtype=np.int8)
    for e in branches:
        incidence[e["id"], bus_index[e["u"]]] = -1
        incidence[e["id"], bus_index[e["v"]]] = 1
        adjacency[bus_index[e["u"]],bus_index[e["v"]]] = 1
        adjacency[bus_index[e["v"]],bus_index[e["u"]]] = 1
    phase_nodes = [(b["bus"], b["node"]) for b in sample["buses"]]
    phase_index = {key:i for i,key in enumerate(phase_nodes)}
    yline = np.zeros((len(phase_nodes),len(phase_nodes)),dtype=np.complex128)
    for e in branches:
        if e["kind"] != "line": continue
        order_u=[p for p in e["u_nodes"] if p in PHASES]
        order_v=[p for p in e["v_nodes"] if p in PHASES]
        if len(order_u)!=len(order_v): raise ValueError(f"Line phase mismatch: {e['name']}")
        zu=(np.asarray(e["r_matrix"])+1j*np.asarray(e["x_matrix"]))*e["length"]
        ys=np.linalg.inv(zu)
        ysh=0.5j*2*np.pi*60*np.asarray(e["c_matrix"])*e["length"]*1e-9
        iu=[phase_index[(e["u"],p)] for p in order_u]
        iv=[phase_index[(e["v"],p)] for p in order_v]
        for i in range(len(iu)):
            for j in range(len(iu)):
                yline[iu[i],iu[j]]+=ys[i,j]+ysh[i,j]
                yline[iv[i],iv[j]]+=ys[i,j]+ysh[i,j]
                yline[iu[i],iv[j]]-=ys[i,j]
                yline[iv[i],iu[j]]-=ys[i,j]
    return {"bus_names": buses, "branches": branches, "incidence": incidence.tolist(),
            "adjacency": adjacency.tolist(),
            "phase_node_order": [f"{b}.{p}" for b,p in phase_nodes],
            "line_ybus_re_s": yline.real.tolist(), "line_ybus_im_s": yline.imag.tolist()}


def node_features(sample: dict, buses: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = {name: i for i, name in enumerate(buses)}
    # x: mask(3), approximate injection P(3) MW, Q(3) MVAr,
    # source flag, load flag, DG flag, capacitor flag = 13 columns.
    x = np.zeros((len(buses), 13), dtype=np.float32)
    y = np.zeros((len(buses), 6), dtype=np.float32)  # Re/Im volts for A/B/C
    mask = np.zeros((len(buses), 3), dtype=np.float32)
    for b in sample["buses"]:
        i, p = idx[b["bus"]], PHASES[b["node"]]
        mask[i, p] = x[i, p] = 1
        y[i, 2*p:2*p+2] = b["v_re_im_v"]
    for load in sample["loads_applied"].values():
        pieces = load["bus"].lower().split(".")
        bus = pieces[0]
        phases = [int(z) for z in pieces[1:] if z in {"1", "2", "3"}]
        if not phases or bus not in idx:
            raise ValueError(f"Cannot map load {load['bus']}")
        # Delta allocation is a graph input proxy, not an exact per-phase
        # injection. Stage 3 physics must use the true device connection.
        for p in phases:
            x[idx[bus], 3+PHASES[p]] -= load["kw"] / len(phases) / 1000
            x[idx[bus], 6+PHASES[p]] -= load["kvar"] / len(phases) / 1000
        x[idx[bus], 10] = 1
    for e in sample["elements"]:
        if e["kind"] == "generator":
            bus = e["terminals"][0]["bus"]
            x[idx[bus], 11] = 1
            # Setpoint is in scenario, not the solved generator terminal power.
            for p in (0, 1, 2):
                if mask[idx[bus], p]: x[idx[bus], 3+p] += sample["scenario"]["dg_kw"] / 3000
        if e["kind"] == "capacitor":
            x[idx[e["terminals"][0]["bus"]], 12] = 1
    x[idx["sourcebus"], 9] = 1
    return x, y, mask


def branch_features(top: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bus_idx = {name: i for i, name in enumerate(top["bus_names"])}
    edges, attrs, ids = [], [], []
    # edge_attr 28: type(2), phase mask(3), length(1), R(9), X(9), C diagonal(3), units code(1)
    for e in top["branches"]:
        f = np.zeros(28, dtype=np.float32)
        f[0 if e["kind"] == "line" else 1] = 1
        for n in set(e["u_nodes"]) & set(e["v_nodes"]):
            if n in PHASES: f[2+PHASES[n]] = 1
        if e["kind"] == "line":
            order = [n for n in e["u_nodes"] if n in PHASES]
            m = len(order)
            assert m == len(e["r_matrix"]) == len(e["x_matrix"]) == len(e["c_matrix"])
            f[5], f[27] = e["length"], e["units_code"]
            for i, pi in enumerate(order):
                for j, pj in enumerate(order):
                    f[6+3*PHASES[pi]+PHASES[pj]] = e["r_matrix"][i][j]
                    f[15+3*PHASES[pi]+PHASES[pj]] = e["x_matrix"][i][j]
                f[24+PHASES[pi]] = e["c_matrix"][i][i]
        u, v = bus_idx[e["u"]], bus_idx[e["v"]]
        edges.extend([(u, v), (v, u)])
        attrs.extend([f, f.copy()])
        ids.extend([e["id"], e["id"]])
    return np.asarray(edges, np.int64).T, np.asarray(attrs, np.float32), np.asarray(ids, np.int64)


def build(sample: dict, top: dict) -> Data:
    x, y, mask = node_features(sample, top["bus_names"])
    ei, ea, ids = branch_features(top)
    return Data(x=torch.from_numpy(x), y=torch.from_numpy(y), phase_mask=torch.from_numpy(mask),
                edge_index=torch.from_numpy(ei), edge_attr=torch.from_numpy(ea),
                branch_id=torch.from_numpy(ids), sample_id=sample["sample_id"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=ROOT/"data/raw/ieee13")
    p.add_argument("--out", type=Path, default=ROOT/"data/graphs/ieee13")
    a = p.parse_args(); a.out = a.out.resolve(); a.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(s) for s in (a.raw/"samples.jsonl").read_text(encoding="utf-8").splitlines()]
    top = topology(rows[0])
    graphs = [build(r, top) for r in rows]
    torch.save(graphs, a.out/"graphs.pt")
    (a.out/"topology.json").write_text(json.dumps(top, indent=2), encoding="utf-8")
    print(f"graphs={len(graphs)} x={tuple(graphs[0].x.shape)} edge_index={tuple(graphs[0].edge_index.shape)} edge_attr={tuple(graphs[0].edge_attr.shape)}")

if __name__ == "__main__": main()
