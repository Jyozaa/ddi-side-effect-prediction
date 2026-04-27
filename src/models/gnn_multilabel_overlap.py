from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.molgraph import Graph


def scatter_mean(x: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
    out = torch.zeros((n_graphs, x.size(1)), device=x.device, dtype=x.dtype)
    cnt = torch.zeros((n_graphs, 1), device=x.device, dtype=x.dtype)
    out.index_add_(0, batch, x)
    cnt.index_add_(0, batch, torch.ones((x.size(0), 1), device=x.device, dtype=x.dtype))
    return out / cnt.clamp_min(1.0)


def scatter_max(x: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
    out = torch.full((n_graphs, x.size(1)), -1e9, device=x.device, dtype=x.dtype)
    for g in range(n_graphs):
        mask = (batch == g)
        if mask.any():
            out[g] = x[mask].max(dim=0).values
        else:
            out[g] = 0.0
    return out


class StrongMPNNLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return h

        src, dst = edge_index[0], edge_index[1]
        m_in = torch.cat([h[src], edge_attr], dim=1)
        m = self.msg(m_in)

        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m)

        h_new = self.gru(agg, h)
        h_out = h + F.dropout(self.bn(h_new), p=self.dropout, training=self.training)
        return h_out


class DrugEncoder(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lin_in = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [StrongMPNNLayer(hidden_dim, edge_dim, dropout) for _ in range(num_layers)]
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, g: Graph, n_graphs: int) -> torch.Tensor:
        h = F.relu(self.lin_in(g.x))
        for layer in self.layers:
            h = layer(h, g.edge_index, g.edge_attr)

        z_mean = scatter_mean(h, g.batch, n_graphs=n_graphs)
        z_max = scatter_max(h, g.batch, n_graphs=n_graphs)
        z = self.readout(torch.cat([z_mean, z_max], dim=1))
        return z


class AuxEncoder(nn.Module):
    def __init__(self, aux_in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(aux_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedFusion(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, z_graph: torch.Tensor, z_aux: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([z_graph, z_aux], dim=1))
        z = g * z_graph + (1.0 - g) * z_aux
        return self.norm(z)


class CrossDrugGate(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.g12 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.g21 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate12 = self.g12(torch.cat([z1, z2], dim=1))
        gate21 = self.g21(torch.cat([z2, z1], dim=1))

        z1_new = self.norm1(z1 + gate12 * z2)
        z2_new = self.norm2(z2 + gate21 * z1)
        return z1_new, z2_new


def compute_target_overlap_features(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:

    eps = 1e-8

    shared = (t1 * t2).sum(dim=1, keepdim=True)
    count1 = t1.sum(dim=1, keepdim=True)
    count2 = t2.sum(dim=1, keepdim=True)
    union = ((t1 + t2) > 0).float().sum(dim=1, keepdim=True)

    jaccard = shared / (union + eps)

    dot = (t1 * t2).sum(dim=1, keepdim=True)
    norm1 = torch.sqrt((t1 * t1).sum(dim=1, keepdim=True) + eps)
    norm2 = torch.sqrt((t2 * t2).sum(dim=1, keepdim=True) + eps)
    cosine = dot / (norm1 * norm2 + eps)

    has_shared = (shared > 0).float()

    # mild scaling for counts
    shared_log = torch.log1p(shared)
    union_log = torch.log1p(union)
    count1_log = torch.log1p(count1)
    count2_log = torch.log1p(count2)

    return torch.cat(
        [shared_log, union_log, jaccard, cosine, has_shared, count1_log, count2_log],
        dim=1,
    )


class PairHead(nn.Module):
    def __init__(self, hidden_dim: int, overlap_dim: int, n_labels: int, dropout: float):
        super().__init__()
        in_dim = hidden_dim * 4 + overlap_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_labels),
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, overlap_feat: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([z1, z2, torch.abs(z1 - z2), z1 * z2, overlap_feat], dim=1)
        return self.mlp(feat)


class StrongGNNMultiLabelOverlap(nn.Module):
    def __init__(
        self,
        node_in: int,
        edge_in: int,
        morgan_in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        n_labels: int,
    ):
        super().__init__()
        self.graph_encoder = DrugEncoder(node_in, edge_in, hidden_dim, num_layers, dropout)
        self.morgan_encoder = AuxEncoder(morgan_in_dim, hidden_dim, dropout)
        self.fusion = GatedFusion(hidden_dim, dropout)
        self.cross_gate = CrossDrugGate(hidden_dim, dropout)
        self.head = PairHead(hidden_dim, overlap_dim=7, n_labels=n_labels, dropout=dropout)

    def forward(
        self,
        g1: Graph,
        g2: Graph,
        morgan1: torch.Tensor,
        morgan2: torch.Tensor,
        target1: torch.Tensor,
        target2: torch.Tensor,
        n_graphs: int,
    ) -> torch.Tensor:
        z1_graph = self.graph_encoder(g1, n_graphs=n_graphs)
        z2_graph = self.graph_encoder(g2, n_graphs=n_graphs)

        z1_aux = self.morgan_encoder(morgan1)
        z2_aux = self.morgan_encoder(morgan2)

        z1 = self.fusion(z1_graph, z1_aux)
        z2 = self.fusion(z2_graph, z2_aux)

        z1, z2 = self.cross_gate(z1, z2)

        overlap_feat = compute_target_overlap_features(target1, target2)

        return self.head(z1, z2, overlap_feat)