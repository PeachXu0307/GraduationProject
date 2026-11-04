"""Differentiable one-step voltage projection for the closed 671--692 switch.

The switch is retained in the complete AC residual. Its declared per-phase
series resistance is 1e-7 ohm. The projection uses the other-device KCL at
692 to compute the small voltage drop that carries switch current. It is a
numerical initialization, not an exact power-flow solution.
"""
from __future__ import annotations

import torch

from full_ac import full_ac_residual


def project_671692(pred_pu: torch.Tensor, sample: dict, topology: dict,
                   voltage_bases_v: torch.Tensor, fixture: dict) -> torch.Tensor:
    if pred_pu.dtype != torch.float64:
        raise TypeError("Switch projection requires float64")
    names = topology["bus_names"]
    order = topology["phase_node_order"]
    i671, i692 = names.index("671"), names.index("692")
    if not torch.isclose(voltage_bases_v[i671], voltage_bases_v[i692]):
        raise ValueError("Switch endpoint voltage bases differ")
    # Start with equal terminal voltages. This annihilates only the switch
    # current; all other device contributions remain in the full residual.
    equal = pred_pu.clone()
    equal[i692] = equal[i671]
    residual = full_ac_residual(equal, sample, topology, voltage_bases_v, fixture)["residual_a"]
    # OpenDSS Rmatrix=1e-4 ohm per length and Length=0.001 for this switch.
    # At node 692, I_switch=(V692-V671)/Z. Make it oppose the other currents.
    z_ohm = 1e-7
    correction = torch.stack([residual[order.index(f"692.{p}")] for p in (1, 2, 3)])
    delta_pu = -z_ohm * correction / voltage_bases_v[i692]
    projected = equal.clone()
    projected[i692] = equal[i671] + torch.view_as_real(delta_pu).reshape(6)
    return projected
