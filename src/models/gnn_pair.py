import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class DrugGNN(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.lin_in = nn.Linear(1, hidden_dim)  # atom feature = atomic number
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.lin_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, batch):
        x = self.lin_in(x)
        x = F.relu(x)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        g = global_mean_pool(x, batch)
        g = self.lin_out(g)
        g = F.relu(g)
        return g

class PairClassifier(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.encoder = DrugGNN(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch_a, batch_b):
        za = self.encoder(batch_a.x, batch_a.edge_index, batch_a.batch)
        zb = self.encoder(batch_b.x, batch_b.edge_index, batch_b.batch)
        feats = torch.cat([za, zb, torch.abs(za - zb), za * zb], dim=-1)
        logits = self.mlp(feats).squeeze(-1)
        return logits
