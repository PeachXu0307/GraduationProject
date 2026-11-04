"""Independent fixed-tap IEEE-13 nonlinear AC correction experiment.

Eliminate source/regulator phase voltages by exact linear KCL and the three
692 switch phase voltages by their local KCL. Solve the remaining 29 complex
node equations with a damped Newton method in PyTorch. Every returned state is checked against all
41 complex KCL equations; a solver success flag alone is never accepted.
"""
from __future__ import annotations

import numpy as np
import torch

from full_ac import full_ac_residual
from switch_projection import project_671692
from linear_projection import LINEAR_BUSES, project_source_regulator


def unknown_indices(topology: dict) -> list[int]:
    order = topology["phase_node_order"]
    excluded = {f"{bus}.{phase}" for bus in LINEAR_BUSES for phase in (1, 2, 3)}
    excluded.update(f"692.{phase}" for phase in (1, 2, 3))
    return [i for i, key in enumerate(order) if key not in excluded]


def flat_start(topology: dict) -> torch.Tensor:
    """Balanced 1 p.u. phase phasors using the declared source angle 30 deg."""
    angles = np.deg2rad([30.0, -90.0, 150.0])
    phasors = np.exp(1j * angles)
    row = np.stack([phasors.real, phasors.imag], axis=1).reshape(6)
    return torch.from_numpy(np.tile(row, (len(topology["bus_names"]), 1)))


def solve_ac(initial_pu: torch.Tensor, sample: dict, topology: dict,
             voltage_bases_v: torch.Tensor, fixture: dict,
             max_iterations: int = 20, tolerance_a: float = 1.0) -> dict:
    if initial_pu.dtype != torch.float64:
        raise TypeError("AC solver initial state must be float64")
    names = topology["bus_names"]
    order = topology["phase_node_order"]
    unknown = unknown_indices(topology)
    phase_to_bus = [(names.index(key.rsplit(".", 1)[0]), int(key.rsplit(".", 1)[1])-1)
                    for key in order]
    def pack(state: torch.Tensor) -> torch.Tensor:
        return torch.stack([state[bus, 2*phase+k]
                           for node in unknown for bus, phase in [phase_to_bus[node]]
                           for k in (0, 1)])
    def unpack(values: torch.Tensor) -> torch.Tensor:
        state = initial_pu.clone()
        for pos, node in enumerate(unknown):
            bus, phase = phase_to_bus[node]
            state[bus, 2*phase:2*phase+2] = values[2*pos:2*pos+2]
        # Local linear equations use the independent node state. The switch
        # projection then solves node 692 KCL, including delta-load current.
        state = project_source_regulator(state, topology, voltage_bases_v, fixture)
        state = project_671692(state, sample, topology, voltage_bases_v, fixture)
        return state
    evaluations = 0
    def equation(values: torch.Tensor) -> torch.Tensor:
        nonlocal evaluations
        evaluations += 1
        state = unpack(values)
        current = full_ac_residual(state, sample, topology, voltage_bases_v,
                                   fixture)["residual_a"][unknown]
        return torch.view_as_real(current).reshape(-1) / 100.0
    x = pack(initial_pu).detach()
    before = full_ac_residual(initial_pu, sample, topology, voltage_bases_v,
                              fixture)["residual_a"].abs()
    status = "maximum_iterations"
    iterations = 0
    for iteration in range(max_iterations):
        iterations = iteration + 1
        fx = equation(x).detach()
        if float(fx.abs().max()) * 100 <= tolerance_a:
            status = "converged"
            break
        jac = torch.autograd.functional.jacobian(equation, x, vectorize=True)
        try:
            step = torch.linalg.solve(jac, -fx)
        except torch.linalg.LinAlgError:
            status = "singular_jacobian"
            break
        accepted = False
        old_norm = float(torch.linalg.vector_norm(fx))
        for damping in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            trial = (x + damping * step).detach()
            trial_norm = float(torch.linalg.vector_norm(equation(trial).detach()))
            if np.isfinite(trial_norm) and trial_norm < old_norm:
                x = trial
                accepted = True
                break
        if not accepted:
            status = "line_search_failed"
            break
    final = unpack(x).detach()
    residual = full_ac_residual(final, sample, topology, voltage_bases_v,
                                fixture)["residual_a"].abs()
    maximum = float(residual.max())
    return {"converged": bool(maximum <= tolerance_a),
            "status": status, "iterations": iterations,
            "function_evaluations": evaluations,
            "initial_kcl_rms_a": float(before.square().mean().sqrt()),
            "final_kcl_rms_a": float(residual.square().mean().sqrt()),
            "final_kcl_max_a": maximum, "voltage_pu": final}
