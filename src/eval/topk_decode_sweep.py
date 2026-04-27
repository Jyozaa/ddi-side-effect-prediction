from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
import torch

from src.utils.io import load_yaml
from src.models.molgraph import smiles_to_graph, batch_graphs
from src.models.molfeatures import build_morgan_cache
from src.models.gnn_multilabel import StrongGNNMultiLabel


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


def iter_index_minibatches(n_items: int, batch_size: int):
    idx = np.arange(n_items)
    for s in range(0, n_items, batch_size):
        yield idx[s:s + batch_size]


def precompute_samples(df, label_cols, graph_cache, aux_cache):
    samples = []
    for _, r in df.iterrows():
        d1 = str(r["drug1_id"])
        d2 = str(r["drug2_id"])
        if d1 not in graph_cache or d2 not in graph_cache:
            continue
        if d1 not in aux_cache or d2 not in aux_cache:
            continue

        samples.append(
            {
                "g1": graph_cache[d1],
                "g2": graph_cache[d2],
                "aux1": aux_cache[d1],
                "aux2": aux_cache[d2],
                "y": r[label_cols].values.astype(np.float32),
            }
        )
    return samples


def build_batch_from_samples(samples, batch_idx, device):
    g1_list = [samples[i]["g1"] for i in batch_idx]
    g2_list = [samples[i]["g2"] for i in batch_idx]
    aux1_np = np.stack([samples[i]["aux1"] for i in batch_idx])
    aux2_np = np.stack([samples[i]["aux2"] for i in batch_idx])
    y_np = np.stack([samples[i]["y"] for i in batch_idx])

    g1 = batch_graphs(g1_list)
    g2 = batch_graphs(g2_list)

    for g in (g1, g2):
        g.x = g.x.to(device)
        g.edge_index = g.edge_index.to(device)
        g.edge_attr = g.edge_attr.to(device)
        g.batch = g.batch.to(device)

    aux1 = torch.tensor(aux1_np, dtype=torch.float32, device=device)
    aux2 = torch.tensor(aux2_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.float32, device=device)
    return g1, g2, aux1, aux2, y


@torch.no_grad()
def predict_probs_from_samples(model, samples, n_labels, device, batch_size):
    model.eval()
    all_probs, all_y = [], []

    for batch_idx in iter_index_minibatches(len(samples), batch_size):
        g1, g2, aux1, aux2, y = build_batch_from_samples(samples, batch_idx, device)
        logits = model(g1, g2, aux1, aux2, n_graphs=y.size(0))
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_y.append(y.cpu().numpy())

    P = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, n_labels), dtype=np.float32)
    Y = np.concatenate(all_y, axis=0) if all_y else np.zeros((0, n_labels), dtype=np.float32)
    return Y, P


def topk_predictions(probs: np.ndarray, k: int) -> np.ndarray:
    n, L = probs.shape
    pred = np.zeros_like(probs, dtype=np.int32)
    if k <= 0:
        return pred
    k = min(k, L)
    idx = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(n)[:, None]
    pred[rows, idx] = 1
    return pred


def sweep_topk_micro_f1(y_true: np.ndarray, probs: np.ndarray, max_k: int = 20):
    best_k = 1
    best_f1 = -1.0

    for k in range(1, max_k + 1):
        pred = topk_predictions(probs, k)
        f1 = f1_score(y_true, pred, average="micro", zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_k = k

    return best_k, best_f1


def main(cfg_path: str = "configs/default.yaml") -> None:
    cfg = load_yaml(cfg_path)

    split_type = cfg["split"]["type"]
    kcfg = int(cfg["multilabel"]["top_k_effects"])

    if kcfg <= 0:
        raw = pd.read_csv(cfg["data"]["smiles_pairs_path"])
        counts = raw["Side Effect Name"].value_counts()
        min_pos = int(cfg["multilabel"]["min_pos_per_label"])
        n_effects = sum(int(c) >= min_pos for c in counts.values)
        run_dir = Path(cfg["data"]["output_dir"]) / f"multilabel_top{n_effects}_{split_type}"
    else:
        run_dir = Path(cfg["data"]["output_dir"]) / f"multilabel_top{kcfg}_{split_type}"

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

    smiles_map = load_smiles_map(cfg["smiles"]["smiles_cache_csv"])
    graph_cache = build_graph_cache(smiles_map)
    aux_cache = build_morgan_cache(
        smiles_map=smiles_map,
        radius=int(cfg["fusion"]["morgan_radius"]),
        nbits=int(cfg["fusion"]["morgan_nbits"]),
    )

    val_samples = precompute_samples(val_df, label_cols, graph_cache, aux_cache)
    test_samples = precompute_samples(test_df, label_cols, graph_cache, aux_cache)

    ckpt_path = run_dir / "gnn_multilabel_fusion" / "best.pt"
    best = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg_gnn = best["cfg_gnn"]

    model = StrongGNNMultiLabel(
        node_in=int(cfg_gnn["node_in"]),
        edge_in=int(cfg_gnn["edge_in"]),
        aux_in_dim=int(cfg_gnn["aux_in_dim"]),
        hidden_dim=int(cfg_gnn["hidden_dim"]),
        num_layers=int(cfg_gnn["num_layers"]),
        dropout=float(cfg_gnn["dropout"]),
        n_labels=n_labels,
    ).to(device)
    model.load_state_dict(best["model_state_dict"])

    batch_size = int(cfg["gnn"]["batch_size"])

    Yv, Pv = predict_probs_from_samples(model, val_samples, n_labels, device, batch_size)
    Yt, Pt = predict_probs_from_samples(model, test_samples, n_labels, device, batch_size)

    avg_cardinality = float(Yv.sum(axis=1).mean())
    print(f"Average val positive labels per sample: {avg_cardinality:.2f}")

    max_k = max(5, min(50, int(round(avg_cardinality * 3))))
    best_k, val_best_micro_f1 = sweep_topk_micro_f1(Yv, Pv, max_k=max_k)

    print(f"Best top-k on val: {best_k}")
    print(f"Best val micro-F1@topk: {val_best_micro_f1:.6f}")

    pred_val = topk_predictions(Pv, best_k)
    pred_test = topk_predictions(Pt, best_k)

    val_micro_f1 = f1_score(Yv, pred_val, average="micro", zero_division=0)
    test_micro_f1 = f1_score(Yt, pred_test, average="micro", zero_division=0)

    results = {
        "best_top_k": int(best_k),
        "avg_val_cardinality": float(avg_cardinality),
        "val_micro_f1@topk": float(val_micro_f1),
        "test_micro_f1@topk": float(test_micro_f1),
        "n_labels": int(n_labels),
        "split_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
    }

    out_path = run_dir / "gnn_multilabel_fusion" / "topk_decode_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved:", out_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()