"""Stage 3a voltage baselines for the fixed-topology IEEE 13 feeder.

Both models predict complex line-to-neutral voltage in per-unit, stored as
[Re(A), Im(A), Re(B), Im(B), Re(C), Im(C)] at each physical bus. Missing phases
are masked by the training and evaluation code.
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing


class NodeMLP(nn.Module):
    """Local input baseline with a fixed-bus identity feature (no message passing)."""

    def __init__(self, num_buses: int, input_dim: int = 13, hidden: int = 64):
        super().__init__()
        self.num_buses = num_buses
        self.net = nn.Sequential(
            nn.Linear(input_dim + num_buses, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 6),
        )

    def forward(self, batch):
        # Every graph uses the same verified 16-bus order. PyG concatenates
        # graphs in that order, so modulo recovers the bus index per graph.
        assert batch.x.shape[0] % self.num_buses == 0
        bus_id = torch.arange(batch.x.shape[0], device=batch.x.device) % self.num_buses
        onehot = torch.nn.functional.one_hot(bus_id, self.num_buses).to(batch.x.dtype)
        return self.net(torch.cat([batch.x, onehot], dim=1))


class BusGCN(nn.Module):
    """Two-layer ordinary GCN baseline; topology used, edge features unused."""

    def __init__(self, input_dim: int = 13, hidden: int = 64):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.head = nn.Linear(hidden, 6)

    def forward(self, batch):
        h = self.conv1(batch.x, batch.edge_index).relu()
        h = self.conv2(h, batch.edge_index).relu()
        return self.head(h)


class ElectricMessageLayer(MessagePassing):
    """Mean aggregation with an edge-conditioned message and residual update.

    h: [total buses, hidden], edge_attr: [directed edges, 28]. The edge
    encoder sees both directions' identical physical parameters; direction
    is supplied by the source and destination hidden states.
    """

    def __init__(self, hidden: int):
        super().__init__(aggr="mean")
        self.message_net = nn.Sequential(
            nn.Linear(2 * hidden + 28, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.update_net = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, h, edge_index, edge_attr):
        messages = self.propagate(edge_index, h=h, edge_attr=edge_attr)
        return torch.relu(h + self.update_net(torch.cat([h, messages], dim=-1)))

    def message(self, h_i, h_j, edge_attr):
        return self.message_net(torch.cat([h_i, h_j, edge_attr], dim=-1))


class EdgeAwareGNN(nn.Module):
    """Three message layers using line R/X mutual terms and device types.

    This model is supervised only. Its edge features do not yet constitute a
    physics loss or a complete transformer model.
    """

    def __init__(self, input_dim: int = 13, hidden: int = 64):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU())
        self.layers = nn.ModuleList([ElectricMessageLayer(hidden) for _ in range(3)])
        self.head = nn.Linear(hidden, 6)

    @staticmethod
    def scale_edges(edge_attr):
        e = edge_attr.clone()
        # OpenDSS R/X/C matrices are per-unit-length. Convert R/X to total
        # ohms, C diagonal to a small dimensionless scale, and length to kft.
        length = e[:, 5:6]
        e[:, 6:24] *= length
        e[:, 24:27] *= length * 1e-3
        e[:, 5:6] = length / 1000.0
        e[:, 27:28] /= 10.0
        return e

    def forward(self, batch):
        h = self.input(batch.x)
        edge = self.scale_edges(batch.edge_attr)
        for layer in self.layers:
            h = layer(h, batch.edge_index, edge)
        return self.head(h)


def make_model(name: str, num_buses: int) -> nn.Module:
    if name == "mlp": return NodeMLP(num_buses)
    if name == "gcn": return BusGCN()
    if name == "edge_gnn": return EdgeAwareGNN()
    raise ValueError(f"Unknown model {name}")
