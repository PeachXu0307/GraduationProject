"""Differentiable complete nodal AC current residual for fixed-tap IEEE-13.

Linear-device Yprim matrices are extracted from a fresh OpenDSS compilation
after applying declared scenario settings and fixed regulator taps, BEFORE
the scenario solve. No solved voltage, current, power, or tap is an input.
The source Norton injection and nonlinear load/generator terms complete KCL.
This implementation is intentionally specific to the current IEEE-13 devices.
"""
from __future__ import annotations

import sys
from pathlib import Path

import dss
import numpy as np
import torch

from partial_ac import _load_current, _complex

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "simulation"))
from generate_ieee13 import CIRCUIT, FIXED_REGULATOR_TAPS, apply_scenario  # noqa: E402
from audit_ports import base_loads  # noqa: E402


def _pairs(values) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64).reshape(-1, 2)
    return a[:, 0] + 1j * a[:, 1]


def build_fixture(sample: dict, topology: dict) -> dict:
    """Build a declared-input linear network fixture for one scenario.

    The returned matrix is in amperes per volt, indexed by the 41 physical
    phase nodes. Ground conductors (node 0) are held at zero volts.
    """
    if sample.get("tap_policy", {}).get("name") != "fixed":
        raise ValueError("Full AC fixture requires a fixed-tap sample")
    if sample["tap_policy"]["declared_regulator_taps"] != FIXED_REGULATOR_TAPS:
        raise ValueError("Sample fixed taps differ from the declared IEEE-13 policy")
    dss.DSS.Text.Command = f'Compile "{CIRCUIT}"'
    circuit = dss.DSS.ActiveCircuit
    # Base settings are read from the canonical circuit, never from solved
    # sample terminal quantities or post-solve regulator outcomes.
    apply_scenario(circuit, sample["scenario"], base_loads(circuit))
    for name, tap in FIXED_REGULATOR_TAPS.items():
        circuit.Transformers.Name = name
        circuit.Transformers.Wdg = 2
        circuit.Transformers.Tap = tap
    dss.DSS.Text.Command = "Set Controlmode=OFF"
    # Tap edits invalidate transformer Yprim. BuildYMatrix refreshes every
    # primitive admittance without solving for bus voltages.
    circuit.Solution.BuildYMatrix(1, 1)
    order = topology["phase_node_order"]
    index = {name: i for i, name in enumerate(order)}
    y = np.zeros((len(order), len(order)), dtype=np.complex128)
    injection = np.zeros(len(order), dtype=np.complex128)
    counts = {kind: 0 for kind in ("line", "transformer", "capacitor", "vsource")}
    for element_name in list(circuit.AllElementNames):
        circuit.SetActiveElement(element_name)
        element = circuit.ActiveCktElement
        if not element.Enabled:
            continue
        kind = element_name.split(".", 1)[0].lower()
        if kind in counts:
            nc = int(element.NumConductors)
            slots = []
            for terminal, bus_spec in enumerate(element.BusNames):
                bus = bus_spec.split(".")[0].lower()
                for node in element.NodeOrder[terminal * nc:(terminal + 1) * nc]:
                    slots.append(index.get(f"{bus}.{int(node)}") if node != 0 else None)
            yp = _pairs(element.Yprim).reshape(len(slots), len(slots))
            for i, row in enumerate(slots):
                if row is None:
                    continue
                for j, col in enumerate(slots):
                    if col is not None:
                        y[row, col] += yp[i, j]
            if kind == "vsource":
                if nc != 3 or int(element.NumTerminals) != 2 or len(slots) != 6:
                    raise ValueError("Unsupported voltage-source connection")
                circuit.Vsources.Name = element_name.split(".", 1)[1]
                angle = np.deg2rad(circuit.Vsources.AngleDeg - 120 * np.arange(3))
                emf = circuit.Vsources.BasekV * 1000 * circuit.Vsources.pu / np.sqrt(3) * np.exp(1j * angle)
                norton = yp[:3, :3] @ emf
                for i, row in enumerate(slots):
                    if row is not None:
                        injection[row] += norton[i] if i < 3 else -norton[i - 3]
            counts[kind] += 1
        elif kind not in {"load", "generator", "regcontrol"}:
            raise ValueError(f"Unmodeled enabled device: {element_name}")
    if counts["vsource"] != 1 or counts["line"] != 12 or counts["transformer"] != 5:
        raise ValueError(f"Unexpected IEEE-13 device counts: {counts}")
    return {"y_si": torch.from_numpy(y), "norton_a": torch.from_numpy(injection),
            "device_counts": counts, "phase_order": order}


def full_ac_residual(pred_pu: torch.Tensor, sample: dict, topology: dict,
                     voltage_bases_v: torch.Tensor, fixture: dict) -> dict:
    """Return all 41 complex phase-node KCL residuals in amperes.

    `pred_pu` must be float64 [16,6], ordered Re/Im for A/B/C. The direct
    float64 path is required by the 671--692 low-impedance switch. The loss
    can use mean(abs(residual_a)**2), scaled by a declared current base.
    """
    if pred_pu.dtype != torch.float64:
        raise TypeError("Full AC residual requires float64 predictions")
    names = topology["bus_names"]
    if pred_pu.shape != (len(names), 6):
        raise ValueError(f"Expected {(len(names), 6)} predictions")
    if fixture["phase_order"] != topology["phase_node_order"]:
        raise ValueError("Fixture phase order differs from topology")
    v = _complex(pred_pu.reshape(len(names), 3, 2)) * voltage_bases_v.to(
        device=pred_pu.device, dtype=torch.float64).reshape(-1, 1)
    voltage = {(name, phase): v[i, phase - 1]
               for i, name in enumerate(names) for phase in (1, 2, 3)}
    order = topology["phase_node_order"]
    full_v = torch.stack([voltage[(node.rsplit(".", 1)[0], int(node.rsplit(".", 1)[1]))]
                          for node in order])
    y = fixture["y_si"].to(device=pred_pu.device)
    norton = fixture["norton_a"].to(device=pred_pu.device)
    residual = y @ full_v - norton
    index = {node: i for i, node in enumerate(order)}
    for load in sample["loads_applied"].values():
        current = _load_current(sample, load, v, voltage)
        bus, *parts = load["bus"].lower().split(".")
        nodes = [int(part) for part in parts if part in {"1", "2", "3"}]
        residual = residual.index_add(0, torch.tensor([index[f"{bus}.{node}"] for node in nodes],
                                                       device=pred_pu.device), current)
    dg_kw = float(sample["scenario"]["dg_kw"])
    if dg_kw > 0:
        # Present generator is Model 1, 3-phase wye at 675, kvar=0.
        # OpenDSS changes to a constant-admittance relation below Vminpu
        # and above Vmaxpu, matched to the PQ current at each boundary.
        nodes = [index[f"675.{phase}"] for phase in (1, 2, 3)]
        generator_v = v[names.index("675")]
        nominal = 4160 / np.sqrt(3)
        minimum, maximum = 0.9 * nominal, 1.1 * nominal
        pq = -torch.conj((dg_kw * 1000 / 3) / generator_v)
        low = -(dg_kw * 1000 / 3) / minimum**2 * generator_v
        high = -(dg_kw * 1000 / 3) / maximum**2 * generator_v
        current = torch.where(generator_v.abs() < minimum, low, pq)
        current = torch.where(generator_v.abs() > maximum, high, current)
        residual = residual.index_add(0, torch.tensor(nodes, device=pred_pu.device), current)
    return {"residual_a": residual,
            "residual_rms_a": torch.sqrt((residual.abs() ** 2).mean().clamp_min(1e-18))}
