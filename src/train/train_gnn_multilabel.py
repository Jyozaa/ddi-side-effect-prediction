from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import WeightedRandomSampler

from src.utils.io import load_yaml, ensure_dir
from src.models.molgraph import smiles_to_graph, batch_graphs
from src.models.molfeatures import build_morgan_cache
from src.models.gnn_multilabel import StrongGNNMultiLabel
from src.eval.multilabel_metrics import (
    tune_thresholds_per_label,
    compute_multilabel_metrics,
    pick_key_labels,
)

class AsymmetricLossMultiLabel(nn.Module):
    def __init__(
        self,
        gamma_pos: float = 1.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # positive and negative probabilities
        xs_pos = probs
        xs_neg = 1.0 - probs

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        loss_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        # asymmetric focusing
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt_pos = xs_pos * targets
            pt_neg = xs_neg * (1.0 - targets)
            pt = pt_pos + pt_neg

            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            focal_weight = torch.pow(1.0 - pt, gamma)

            loss = loss * focal_weight

        loss = -loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
    

def build_loss_fn(cfg: dict, y_train: np.ndarray, device: torch.device) -> tuple[nn.Module, str, bool]:
    loss_name = str(cfg["gnn"].get("loss_name", "bce")).lower()

    if loss_name == "bce":
        pos_weight = compute_pos_weight_from_y(y_train).to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return loss_fn, loss_name, True

    if loss_name == "asl":
        loss_fn = AsymmetricLossMultiLabel(
            gamma_pos=float(cfg["gnn"].get("asl_gamma_pos", 1.0)),
            gamma_neg=float(cfg["gnn"].get("asl_gamma_neg", 4.0)),
            clip=float(cfg["gnn"].get("asl_clip", 0.05)),
            reduction="mean",
        )
        return loss_fn, loss_name, False

    raise ValueError(f"Unsupported loss_name: {loss_name}. Expected 'bce' or 'asl'.")


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


def make_sample_weights_from_y(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.float32)
    return 1.0 + y.sum(axis=1)


def iter_index_minibatches(n_items: int, batch_size: int, shuffle: bool, seed: int, sampler=None):
    if sampler is not None:
        rng = np.random.default_rng(seed)
        weights = np.asarray(sampler.weights, dtype=np.float64)
        weights = weights / weights.sum()
        idx = rng.choice(n_items, size=n_items, replace=True, p=weights)
    else:
        idx = np.arange(n_items)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(idx)

    for s in range(0, len(idx), batch_size):
        yield idx[s:s + batch_size]


def precompute_samples(
    df: pd.DataFrame,
    label_cols: list[str],
    graph_cache: dict,
    aux_cache: dict,
):
    samples = []

    for _, r in df.iterrows():
        d1 = str(r["drug1_id"])
        d2 = str(r["drug2_id"])

        if d1 not in graph_cache or d2 not in graph_cache:
            continue
        if d1 not in aux_cache or d2 not in aux_cache:
            continue

        y = r[label_cols].values.astype(np.float32)

        samples.append(
            {
                "g1": graph_cache[d1],
                "g2": graph_cache[d2],
                "aux1": aux_cache[d1],
                "aux2": aux_cache[d2],
                "y": y,
            }
        )

    return samples


def build_batch_from_samples(samples: list[dict], batch_idx: np.ndarray, device: torch.device):
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
def predict_probs_from_samples(model, samples: list[dict], n_labels: int, device, batch_size: int):
    model.eval()
    all_probs, all_y = [], []

    for batch_idx in iter_index_minibatches(
        n_items=len(samples),
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        sampler=None,
    ):
        g1, g2, aux1, aux2, y = build_batch_from_samples(samples, batch_idx, device)
        logits = model(g1, g2, aux1, aux2, n_graphs=y.size(0))
        probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.append(probs)
        all_y.append(y.cpu().numpy())

    P = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, n_labels), dtype=np.float32)
    Y = np.concatenate(all_y, axis=0) if all_y else np.zeros((0, n_labels), dtype=np.float32)
    return Y, P


def compute_pos_weight_from_y(y: np.ndarray) -> torch.Tensor:
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    w = (neg / np.clip(pos, 1.0, None)).astype(np.float32)
    return torch.tensor(w, dtype=torch.float32)


def main(cfg_path: str = "configs/default.yaml") -> None:
    cfg = load_yaml(cfg_path)
    seed = int(cfg["data"]["seed"])

    split_type = cfg["split"]["type"]
    k = int(cfg["multilabel"]["top_k_effects"])

    if k <= 0:
        raw = pd.read_csv(cfg["data"]["smiles_pairs_path"])
        counts = raw["Side Effect Name"].value_counts()
        min_pos = int(cfg["multilabel"]["min_pos_per_label"])
        n_effects = sum(int(c) >= min_pos for c in counts.values)
        run_dir = Path(cfg["data"]["output_dir"]) / f"multilabel_top{n_effects}_{split_type}"
    else:
        run_dir = Path(cfg["data"]["output_dir"]) / f"multilabel_top{k}_{split_type}"

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

    aux_in_dim = int(cfg["fusion"]["morgan_nbits"])
    aux_cache = build_morgan_cache(
        smiles_map=smiles_map,
        radius=int(cfg["fusion"]["morgan_radius"]),
        nbits=aux_in_dim,
    )
    print("Aux feature cache size:", len(aux_cache))

    limit_pairs = int(cfg["gnn"].get("limit_pairs", 0) or 0)
    if limit_pairs > 0 and len(train_df) > limit_pairs:
        train_df = train_df.sample(n=limit_pairs, random_state=seed).reset_index(drop=True)

    print("Precomputing valid samples...")
    train_samples = precompute_samples(train_df, label_cols, graph_cache, aux_cache)
    val_samples = precompute_samples(val_df, label_cols, graph_cache, aux_cache)
    test_samples = precompute_samples(test_df, label_cols, graph_cache, aux_cache)

    print(f"Valid samples: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
    if len(train_samples) == 0:
        raise RuntimeError("No valid training samples after filtering.")

    train_y = np.stack([s["y"] for s in train_samples]).astype(np.float32)

    train_weights = make_sample_weights_from_y(train_y)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(train_weights, dtype=torch.double),
        num_samples=len(train_samples),
        replacement=True,
    )

    node_in = 19
    edge_in = 12

    model = StrongGNNMultiLabel(
        node_in=node_in,
        edge_in=edge_in,
        aux_in_dim=aux_in_dim,
        hidden_dim=int(cfg["gnn"]["hidden_dim"]),
        num_layers=int(cfg["gnn"]["num_layers"]),
        dropout=float(cfg["gnn"]["dropout"]),
        n_labels=n_labels,
    ).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["gnn"]["lr"]),
        weight_decay=float(cfg["gnn"]["weight_decay"]),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=3, threshold=1e-4
    )

    y_train_np = np.stack([s["y"] for s in train_samples], axis=0)
    loss_fn, loss_name, pos_weight_used = build_loss_fn(cfg, y_train_np, device)

    batch_size = int(cfg["gnn"]["batch_size"])
    epochs = int(cfg["gnn"]["epochs"])
    early_stop_patience = int(cfg["gnn"].get("early_stop_patience", 5))

    out_dir = run_dir / f"gnn_multilabel_fusion_{loss_name}"
    ensure_dir(str(out_dir))
    ckpt_path = out_dir / "best.pt"

    best_val_micro_auc = -1.0
    best_thresholds = None
    epochs_without_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0

        for batch_idx in iter_index_minibatches(
            n_items=len(train_samples),
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch,
            sampler=sampler,
        ):
            g1, g2, aux1, aux2, y = build_batch_from_samples(train_samples, batch_idx, device)

            opt.zero_grad()
            logits = model(g1, g2, aux1, aux2, n_graphs=y.size(0))
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * y.size(0)
            n_seen += y.size(0)

        avg_loss = total_loss / max(1, n_seen)

        Yv, Pv = predict_probs_from_samples(
            model=model,
            samples=val_samples,
            n_labels=n_labels,
            device=device,
            batch_size=batch_size,
        )

        if len(Yv) == 0:
            print(f"Epoch {epoch:02d}/{epochs} | loss={avg_loss:.4f} | val empty after filtering")
            continue

        ths = tune_thresholds_per_label(Yv, Pv)
        val_metrics = compute_multilabel_metrics(Yv, Pv, thresholds=ths)

        scheduler.step(val_metrics["micro_auc"])

        print(
            f"Epoch {epoch:02d}/{epochs} | loss={avg_loss:.4f} | "
            f"val_micro_auc={val_metrics['micro_auc']:.4f} | "
            f"val_macro_auc={val_metrics['macro_auc']:.4f} | "
            f"val_micro_f1@tuned={val_metrics['micro_f1@tuned']:.4f}"
        )

        if val_metrics["micro_auc"] > best_val_micro_auc:
            best_val_micro_auc = float(val_metrics["micro_auc"])
            best_thresholds = np.array(val_metrics["thresholds"], dtype=np.float32)
            epochs_without_improve = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_cols": label_cols,
                    "thresholds": best_thresholds.tolist(),
                    "cfg_gnn": {
                        "hidden_dim": int(cfg["gnn"]["hidden_dim"]),
                        "num_layers": int(cfg["gnn"]["num_layers"]),
                        "dropout": float(cfg["gnn"]["dropout"]),
                        "node_in": node_in,
                        "edge_in": edge_in,
                        "aux_in_dim": aux_in_dim,
                    },
                },
                ckpt_path,
            )
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    best = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(best["model_state_dict"])
    label_cols = best["label_cols"]
    thresholds = np.array(best["thresholds"], dtype=np.float32)

    Yv_best, Pv_best = predict_probs_from_samples(
        model=model,
        samples=val_samples,
        n_labels=n_labels,
        device=device,
        batch_size=batch_size,
    )
    val_metrics_best = compute_multilabel_metrics(Yv_best, Pv_best, thresholds=thresholds)

    Yt, Pt = predict_probs_from_samples(
        model=model,
        samples=test_samples,
        n_labels=n_labels,
        device=device,
        batch_size=batch_size,
    )
    test_metrics = compute_multilabel_metrics(Yt, Pt, thresholds=thresholds)

    key_idx = pick_key_labels(label_cols, train_y, top_n=5)

    key_labels = []
    for ki in key_idx:
        key_labels.append(
            {
                "label": label_cols[ki],
                "test_auc": test_metrics["per_label_auc"][ki],
                "test_f1@tuned": test_metrics["per_label_f1@tuned"][ki],
                "threshold": float(thresholds[ki]) if thresholds is not None else 0.5,
                "train_pos": int(train_y[:, ki].sum()),
            }
        )

    results = {
        "val_best_micro_auc": float(best_val_micro_auc),
        "val_micro_auc": float(val_metrics_best["micro_auc"]),
        "val_macro_auc": float(val_metrics_best["macro_auc"]),
        "val_micro_f1@0.5": float(val_metrics_best["micro_f1@0.5"]),
        "val_micro_f1@tuned": float(val_metrics_best["micro_f1@tuned"]),
        "test_micro_auc": float(test_metrics["micro_auc"]),
        "test_macro_auc": float(test_metrics["macro_auc"]),
        "test_micro_f1@0.5": float(test_metrics["micro_f1@0.5"]),
        "test_micro_f1@tuned": float(test_metrics["micro_f1@tuned"]),
        "loss_name": loss_name,
        "asl_gamma_pos": float(cfg["gnn"].get("asl_gamma_pos", 1.0)) if loss_name == "asl" else None,
        "asl_gamma_neg": float(cfg["gnn"].get("asl_gamma_neg", 4.0)) if loss_name == "asl" else None,
        "asl_clip": float(cfg["gnn"].get("asl_clip", 0.05)) if loss_name == "asl" else None,
        "n_labels": int(n_labels),
        "split_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "train_used": int(len(train_samples)),
        "val_used": int(len(val_samples)),
        "test_used": int(len(test_samples)),
        "device": str(device),
        "key_label_results": key_labels,
        "lr_last": float(opt.param_groups[0]["lr"]),
        "sampling": "WeightedRandomSampler(weight=1+num_positive_labels)",
        "pos_weight_used": bool(pos_weight_used),
        "feature_dims": {
            "node_in": node_in,
            "edge_in": edge_in,
            "aux_in_dim": aux_in_dim,
        },
        "scheduler": "ReduceLROnPlateau(monitor=val_micro_auc,factor=0.5,patience=3)",
        "early_stop_patience": early_stop_patience,
        "model_variant": f"graph_plus_morgan_gated_fusion_meanmax_crossgate_{loss_name}"
    }

    out_json = out_dir / "results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved:", out_json)
    print(results)


if __name__ == "__main__":
    main()