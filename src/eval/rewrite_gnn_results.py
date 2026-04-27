from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.io import load_yaml
from src.models.molgraph import smiles_to_graph, batch_graphs, Graph
from src.eval.multilabel_metrics import compute_multilabel_metrics, pick_key_labels


def load_smiles_map(path: str) -> dict[str, str]:
    df = pd.read_csv(path)
    return dict(zip(df["drug_id"].astype(str), df["smiles"].astype(str)))


def build_graph_cache(smiles_map: dict[str, str]) -> dict[str, tuple]:
    cache = {}
    for did, smi in smiles_map.items():
        g = smiles_to_graph(smi)
        if g is not None:
            cache[str(did)] = g
    return cache


def precompute_graph_only_samples(
    df: pd.DataFrame,
    label_cols: list[str],
    graph_cache: dict,
):
    samples = []
    for _, r in df.iterrows():
        d1 = str(r["drug1_id"])
        d2 = str(r["drug2_id"])

        if d1 not in graph_cache or d2 not in graph_cache:
            continue

        samples.append(
            {
                "g1": graph_cache[d1],
                "g2": graph_cache[d2],
                "y": r[label_cols].values.astype(np.float32),
            }
        )
    return samples


def iter_index_minibatches(n_items: int, batch_size: int):
    idx = np.arange(n_items)
    for s in range(0, n_items, batch_size):
        yield idx[s:s + batch_size]


def build_batch_from_samples(samples: list[dict], batch_idx: np.ndarray, device: torch.device):
    g1_list = [samples[i]["g1"] for i in batch_idx]
    g2_list = [samples[i]["g2"] for i in batch_idx]
    y_np = np.stack([samples[i]["y"] for i in batch_idx])

    g1 = batch_graphs(g1_list)
    g2 = batch_graphs(g2_list)

    for g in (g1, g2):
        g.x = g.x.to(device)
        g.edge_index = g.edge_index.to(device)
        g.edge_attr = g.edge_attr.to(device)
        g.batch = g.batch.to(device)

    y = torch.tensor(y_np, dtype=torch.float32, device=device)
    return g1, g2, y


@torch.no_grad()
def predict_probs_graph_only(model, samples: list[dict], n_labels: int, device, batch_size: int):
    model.eval()
    all_probs, all_y = [], []

    for batch_idx in iter_index_minibatches(len(samples), batch_size):
        g1, g2, y = build_batch_from_samples(samples, batch_idx, device)
        logits = model(g1, g2, n_graphs=y.size(0))
        probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.append(probs)
        all_y.append(y.cpu().numpy())

    P = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, n_labels), dtype=np.float32)
    Y = np.concatenate(all_y, axis=0) if all_y else np.zeros((0, n_labels), dtype=np.float32)
    return Y, P

def scatter_mean(x: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
    out = torch.zeros((n_graphs, x.size(1)), device=x.device, dtype=x.dtype)
    cnt = torch.zeros((n_graphs, 1), device=x.device, dtype=x.dtype)
    out.index_add_(0, batch, x)
    cnt.index_add_(0, batch, torch.ones((x.size(0), 1), device=x.device, dtype=x.dtype))
    return out / cnt.clamp_min(1.0)


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

    def forward(self, g: Graph, n_graphs: int) -> torch.Tensor:
        h = F.relu(self.lin_in(g.x))
        for layer in self.layers:
            h = layer(h, g.edge_index, g.edge_attr)
        z = scatter_mean(h, g.batch, n_graphs=n_graphs)
        return z


class PairHead(nn.Module):
    def __init__(self, hidden_dim: int, n_labels: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_labels),
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([z1, z2, torch.abs(z1 - z2), z1 * z2], dim=1)
        return self.mlp(feat)


class GraphOnlyStrongGNNMultiLabel(nn.Module):
    def __init__(self, node_in: int, edge_in: int, hidden_dim: int, num_layers: int, dropout: float, n_labels: int):
        super().__init__()
        self.encoder = DrugEncoder(node_in, edge_in, hidden_dim, num_layers, dropout)
        self.head = PairHead(hidden_dim, n_labels, dropout)

    def forward(self, g1: Graph, g2: Graph, n_graphs: int) -> torch.Tensor:
        z1 = self.encoder(g1, n_graphs=n_graphs)
        z2 = self.encoder(g2, n_graphs=n_graphs)
        return self.head(z1, z2)


def main(cfg_path: str = "configs/default.yaml") -> None:
    cfg = load_yaml(cfg_path)

    split_type = cfg["split"]["type"]
    top_k = int(cfg["multilabel"]["top_k_effects"])
    run_dir = Path(cfg["data"]["output_dir"]) / f"multilabel_top{top_k}_{split_type}"

    train_df = pd.read_csv(run_dir / "pairs_train.csv")
    val_df = pd.read_csv(run_dir / "pairs_val.csv")
    test_df = pd.read_csv(run_dir / "pairs_test.csv")

    label_cols = [c for c in train_df.columns if c not in ["drug1_id", "drug2_id"]]
    n_labels = len(label_cols)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Device:", device)

    smiles_map = load_smiles_map(cfg["smiles"]["smiles_cache_csv"])
    graph_cache = build_graph_cache(smiles_map)
    print("Graph cache size:", len(graph_cache))

    print("Precomputing graph-only samples...")
    train_samples = precompute_graph_only_samples(train_df, label_cols, graph_cache)
    val_samples = precompute_graph_only_samples(val_df, label_cols, graph_cache)
    test_samples = precompute_graph_only_samples(test_df, label_cols, graph_cache)
    print(f"Valid samples: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    ckpt_path = run_dir / "gnn_multilabel" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    best = torch.load(ckpt_path, map_location=device, weights_only=True)

    cfg_gnn = best["cfg_gnn"]
    thresholds = np.array(best["thresholds"], dtype=np.float32)
    label_cols_ckpt = best["label_cols"]

    model = GraphOnlyStrongGNNMultiLabel(
        node_in=int(cfg_gnn["node_in"]),
        edge_in=int(cfg_gnn["edge_in"]),
        hidden_dim=int(cfg_gnn["hidden_dim"]),
        num_layers=int(cfg_gnn["num_layers"]),
        dropout=float(cfg_gnn["dropout"]),
        n_labels=len(label_cols_ckpt),
    ).to(device)

    model.load_state_dict(best["model_state_dict"])

    batch_size = int(cfg["gnn"]["batch_size"])

    Yv, Pv = predict_probs_graph_only(model, val_samples, n_labels, device, batch_size)
    val_metrics = compute_multilabel_metrics(Yv, Pv, thresholds=thresholds)

    Yt, Pt = predict_probs_graph_only(model, test_samples, n_labels, device, batch_size)
    test_metrics = compute_multilabel_metrics(Yt, Pt, thresholds=thresholds)

    train_y = np.stack([s["y"] for s in train_samples]).astype(np.float32)
    key_idx = pick_key_labels(label_cols_ckpt, train_y, top_n=5)

    key_labels = []
    for ki in key_idx:
        key_labels.append(
            {
                "label": label_cols_ckpt[ki],
                "test_auc": test_metrics["per_label_auc"][ki],
                "test_f1@tuned": test_metrics["per_label_f1@tuned"][ki],
                "threshold": float(thresholds[ki]) if thresholds is not None else 0.5,
                "train_pos": int(train_y[:, ki].sum()),
            }
        )

    old_results_path = run_dir / "gnn_multilabel" / "results.json"
    old_results = {}
    if old_results_path.exists():
        old_results = json.loads(old_results_path.read_text(encoding="utf-8"))

    results = {
        "val_best_micro_auc": float(old_results.get("val_best_micro_auc", val_metrics["micro_auc"])),

        "val_micro_auc": float(val_metrics["micro_auc"]),
        "val_macro_auc": float(val_metrics["macro_auc"]),
        "val_micro_f1@0.5": float(val_metrics["micro_f1@0.5"]),
        "val_micro_f1@tuned": float(val_metrics["micro_f1@tuned"]),

        "test_micro_auc": float(test_metrics["micro_auc"]),
        "test_macro_auc": float(test_metrics["macro_auc"]),
        "test_micro_f1@0.5": float(test_metrics["micro_f1@0.5"]),
        "test_micro_f1@tuned": float(test_metrics["micro_f1@tuned"]),

        "n_labels": int(n_labels),
        "split_dir": str(run_dir),
        "checkpoint": str(ckpt_path),

        "train_used": int(len(train_samples)),
        "val_used": int(len(val_samples)),
        "test_used": int(len(test_samples)),

        "device": str(device),
        "key_label_results": key_labels,
        "feature_dims": {
            "node_in": int(cfg_gnn["node_in"]),
            "edge_in": int(cfg_gnn["edge_in"]),
        },
        "model_variant": "graph_only_gnn_rewritten_eval",
    }

    out_path = run_dir / "gnn_multilabel" / "results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Rewrote:", out_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()