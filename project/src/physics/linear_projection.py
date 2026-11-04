"""Exact local KCL projection for source and regulator connection nodes.

The nine sourcebus/650/rg60 phase nodes have no nonlinear devices in the
fixed-tap IEEE-13 circuit. Holding all other predicted voltages fixed, solve
their linear KCL block from the declared-input Y matrix and Norton current.
This is a local algebraic correction, not a complete network power-flow solve.
"""
from __future__ import annotations

import torch


LINEAR_BUSES = ("sourcebus", "650", "rg60")


def project_source_regulator(pred_pu: torch.Tensor, topology: dict,
                             voltage_bases_v: torch.Tensor, fixture: dict) -> torch.Tensor:
    if pred_pu.dtype != torch.float64:
        raise TypeError("Linear projection requires float64 predictions")
    names = topology["bus_names"]
    order = topology["phase_node_order"]
    local_names = [f"{bus}.{phase}" for bus in LINEAR_BUSES for phase in (1, 2, 3)]
    local = torch.tensor([order.index(name) for name in local_names],
                         dtype=torch.long, device=pred_pu.device)
    other = torch.tensor([i for i in range(len(order)) if i not in set(local.tolist())],
                         dtype=torch.long, device=pred_pu.device)
    base = voltage_bases_v.to(device=pred_pu.device, dtype=torch.float64)
    voltage = torch.view_as_complex(pred_pu.reshape(-1, 3, 2).contiguous()) * base[:, None]
    by_node = {(bus, phase): voltage[i, phase - 1]
               for i, bus in enumerate(names) for phase in (1, 2, 3)}
    full = torch.stack([by_node[(key.rsplit(".", 1)[0], int(key.rsplit(".", 1)[1]))]
                        for key in order])
    y = fixture["y_si"].to(pred_pu.device)
    injection = fixture["norton_a"].to(pred_pu.device)
    local_y = y[local][:, local]
    rhs = injection[local] - y[local][:, other] @ full[other]
    solved = torch.linalg.solve(local_y, rhs)
    output = pred_pu.clone()
    for index, value in zip(local.tolist(), solved):
        bus, phase = order[index].rsplit(".", 1)
        output[names.index(bus), 2*(int(phase)-1):2*int(phase)] = torch.view_as_real(value / base[names.index(bus)])
    return output
