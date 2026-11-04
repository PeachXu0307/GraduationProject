"""Differentiable, explicitly partial AC residual for the Stage 3 prototype.

Included: line pi-model mutual R/X/C and declared load models 1/2/5,
including delta/wye connection and OpenDSS low-voltage interpolation.
Excluded: source, transformers, regulators, capacitors and generators.
The returned coverage mask excludes buses incident to excluded elements, so
the result must not be called a complete AC power-flow residual.
"""
from __future__ import annotations

from typing import Any
import math
import torch


def _complex(re_im: torch.Tensor) -> torch.Tensor:
    return torch.complex(re_im[..., 0], re_im[..., 1])


def _line_ybus(topology: dict[str, Any], bus_names: list[str], phase_order: list[str], *, device, dtype):
    n = len(phase_order)
    y = torch.zeros((n, n), dtype=torch.complex64 if dtype == torch.float32 else torch.complex128, device=device)
    pidx = {key: i for i, key in enumerate(phase_order)}
    for edge in topology["branches"]:
        if edge["kind"] != "line":
            continue
        order_u = [p for p in edge["u_nodes"] if p in {1, 2, 3}]
        order_v = [p for p in edge["v_nodes"] if p in {1, 2, 3}]
        z = torch.as_tensor(edge["r_matrix"], dtype=dtype, device=device) + 1j * torch.as_tensor(edge["x_matrix"], dtype=dtype, device=device)
        z = z * float(edge["length"])
        ys = torch.linalg.inv(z.to(y.dtype))
        c = torch.as_tensor(edge["c_matrix"], dtype=dtype, device=device) * float(edge["length"]) * 1e-9
        ysh = 0.5j * 2 * math.pi * 60 * c.to(y.dtype)
        iu = [pidx[f"{edge['u']}.{p}"] for p in order_u]
        iv = [pidx[f"{edge['v']}.{p}"] for p in order_v]
        for i, ui in enumerate(iu):
            for j, uj in enumerate(iu):
                vi, vj = iv[i], iv[j]
                y[ui, uj] += ys[i, j] + ysh[i, j]
                y[vi, vj] += ys[i, j] + ysh[i, j]
                y[ui, vj] -= ys[i, j]
                y[vi, uj] -= ys[i, j]
    return y


def _load_current(sample: dict[str, Any], load: dict[str, Any], v: torch.Tensor, voltage_by_node: dict[tuple[str, int], torch.Tensor]):
    """Return current leaving the load's bus conductors, in OpenDSS order."""
    bus, *parts = load["bus"].lower().split(".")
    nodes = [int(p) for p in parts if p in {"1", "2", "3"}]
    phases = len(nodes)
    branch_v = torch.stack([voltage_by_node[(bus, p)] for p in nodes])
    delta = bool(load["is_delta"])
    if delta and phases == 3:
        branch_v = torch.stack([branch_v[0] - branch_v[1], branch_v[1] - branch_v[2], branch_v[2] - branch_v[0]])
        mapping = torch.tensor([[1, 0, -1], [-1, 1, 0], [0, -1, 1]], dtype=v.real.dtype, device=v.device)
    elif delta and phases == 2:
        branch_v = branch_v[:1] - branch_v[1:2]
        mapping = torch.tensor([[1], [-1]], dtype=v.real.dtype, device=v.device)
    else:
        mapping = torch.eye(phases, dtype=v.real.dtype, device=v.device)
    s = torch.tensor(complex(float(load["kw"]), float(load["kvar"])) * 1000 / len(branch_v), dtype=branch_v.dtype, device=v.device)
    # OpenDSS uses the bus line-neutral base for wye, and line-line base for delta.
    nominal = float(sample.get("nominal_load_voltage_v", {}).get(load["bus"], 0.0))
    if nominal <= 0:
        # The generator stores only the load kV; this fallback is valid for the
        # present feeder's load definitions and is checked against OpenDSS.
        nominal = float(load.get("kv", 0.0)) * 1000
    if nominal <= 0:
        raise ValueError(f"Missing load nominal voltage for {load['bus']}")
    vm = branch_v.abs()
    if int(load["model"]) == 1:
        current = torch.conj(s / branch_v)
    elif int(load["model"]) == 2:
        current = torch.conj(s) / nominal**2 * branch_v
    elif int(load["model"]) == 5:
        current = torch.conj(s) / nominal * branch_v / vm.clamp_min(1e-6)
    else:
        raise ValueError(f"Unsupported load model {load['model']}")
    # Use the documented Vlowpu=0.5 interpolation. Model 1 joins PQ; Model 5
    # joins nominal-magnitude constant-current behavior at Vminpu.
    if int(load["model"]) in {1, 5}:
        lower = 0.95 * nominal
        low = 0.50 * nominal
        yeq = torch.conj(s) / nominal**2
        yeq_min = yeq / (0.95**2 if int(load["model"]) == 1 else 0.95)
        ilow, imin = yeq * low, yeq_min * lower
        interp = (ilow + (imin - ilow) * (vm - low) / (lower - low)) / vm.clamp_min(1e-6) * branch_v
        current = torch.where(vm <= low, yeq * branch_v, current)
        current = torch.where((vm > low) & (vm <= lower), interp, current)
        current = torch.where(vm > 1.05 * nominal, yeq / (1.05**2) * branch_v, current)
    return mapping.to(branch_v.dtype) @ current


def partial_ac_residual(pred_pu: torch.Tensor, sample: dict[str, Any], topology: dict[str, Any], voltage_bases_v: torch.Tensor) -> dict[str, torch.Tensor | list[str]]:
    """Return differentiable partial KCL residual and explicit coverage metadata.

    `pred_pu` shape is `(num_buses, 6)` in Re/Im A/B/C p.u.; the returned
    `residual_a` shape is `(num_phase_nodes,)` in amperes. `covered_mask` is
    false at buses touching transformers or the source, and at nodes without
    a line connection. The omitted device list is always returned.
    """
    names = topology["bus_names"]
    phase_order = topology["phase_node_order"]
    device = pred_pu.device
    # The 671--692 switch has a 1e-7 ohm series resistance. Complex64
    # cancellation of nearly equal endpoint voltages creates kiloamp errors.
    # Cast before converting p.u. to volts. A float32 prediction has already
    # lost voltage precision, however; this cast cannot recover those bits.
    # Physics-loss training needs a double-precision prediction path or a
    # separately justified switch-equivalent model.
    pred_pu = pred_pu.to(torch.float64)
    dtype = pred_pu.dtype
    # pred_pu is normalized line-neutral voltage. Convert each bus back to V.
    v = _complex(pred_pu.reshape(len(names), 3, 2)) * voltage_bases_v.to(device=device, dtype=dtype).reshape(-1, 1)
    voltage = {(bus, p): v[i, p-1] for i, bus in enumerate(names) for p in (1, 2, 3)}
    full_v = torch.stack([voltage[key] for key in [(x.rsplit('.', 1)[0], int(x.rsplit('.', 1)[1])) for x in phase_order]])
    residual = _line_ybus(topology, names, phase_order, device=device, dtype=dtype) @ full_v
    pidx = {key: i for i, key in enumerate(phase_order)}
    # OpenDSS reports each device terminal current entering that device.
    # Line YV and load terminal currents therefore have the same sign in KCL.
    for load in sample["loads_applied"].values():
        current = _load_current(sample, load, v, voltage)
        bus, *parts = load["bus"].lower().split(".")
        for node, value in zip([int(p) for p in parts if p in {"1", "2", "3"}], current):
            residual[pidx[f"{bus}.{node}"]] += value
    excluded = set()
    omitted = ["vsource", "transformer", "regulator", "capacitor", "generator"]
    # Exclude every conductor incident to an omitted electrical element.
    # A regulator control object itself has no port current; its transformer
    # does, so the transformer terminals are already excluded here.
    for element in sample["elements"]:
        if element["name"].split(".", 1)[0].lower() in {"vsource", "transformer", "capacitor", "generator"}:
            for terminal in element["terminals"]:
                excluded.update(f"{terminal['bus'].lower()}.{node}" for node in terminal["nodes"] if node in {1, 2, 3})
    connected = set()
    for branch in topology["branches"]:
        if branch["kind"] == "line":
            connected.update(f"{branch['u']}.{p}" for p in branch["u_nodes"] if p in {1, 2, 3})
            connected.update(f"{branch['v']}.{p}" for p in branch["v_nodes"] if p in {1, 2, 3})
    covered = torch.tensor([x not in excluded and x in connected for x in phase_order], dtype=torch.bool, device=device)
    if not bool(covered.any()):
        raise ValueError("No phase nodes covered by the partial AC residual")
    return {"residual_a": residual, "covered_mask": covered, "omitted_devices": omitted,
            "residual_rms_a": torch.sqrt((residual[covered].abs() ** 2).mean().clamp_min(1e-18))}
