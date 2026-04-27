from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def tune_thresholds_per_label(y_true: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """
    Tune per-label thresholds on validation set to maximize per-label F1.
    Returns thresholds shape [K].
    """
    y_true = y_true.astype(np.int32)
    probs = probs.astype(np.float32)

    K = y_true.shape[1]
    ths = np.zeros(K, dtype=np.float32)

    grid = np.linspace(0.01, 0.99, 99)

    for k in range(K):
        yk = y_true[:, k]

        if len(np.unique(yk)) < 2:
            ths[k] = 0.5
            continue

        best_f1, best_t = -1.0, 0.5
        for t in grid:
            pred = (probs[:, k] >= t).astype(int)
            f1 = f1_score(yk, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        ths[k] = best_t

    return ths


def compute_multilabel_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict:
    """
    Computes:
      - micro_auc
      - macro_auc (nanmean over per-label AUCs)
      - micro_f1@0.5
      - micro_f1@tuned (using per-label thresholds)
      - per_label_auc
      - per_label_f1@tuned
      - thresholds (list)
    """
    y_true = y_true.astype(np.int32)
    probs = probs.astype(np.float32)

    K = y_true.shape[1]

    # AUCs
    micro_auc = roc_auc_score(y_true, probs, average="micro")

    per_label_auc = []
    for k in range(K):
        yk = y_true[:, k]
        pk = probs[:, k]
        if len(np.unique(yk)) < 2:
            per_label_auc.append(float("nan"))
        else:
            per_label_auc.append(float(roc_auc_score(yk, pk)))

    macro_auc = float(np.nanmean(per_label_auc))

    # F1 0.5
    pred_05 = (probs >= 0.5).astype(int)
    micro_f1_05 = f1_score(y_true, pred_05, average="micro", zero_division=0)

    # F1 tuned thresholds
    if thresholds is None:
        thresholds = np.full((K,), 0.5, dtype=np.float32)
    else:
        thresholds = np.asarray(thresholds, dtype=np.float32)

    pred_th = (probs >= thresholds[None, :]).astype(int)
    micro_f1_th = f1_score(y_true, pred_th, average="micro", zero_division=0)

    per_label_f1 = []
    for k in range(K):
        per_label_f1.append(float(f1_score(y_true[:, k], pred_th[:, k], zero_division=0)))

    return {
        "micro_auc": float(micro_auc),
        "macro_auc": float(macro_auc),
        "micro_f1@0.5": float(micro_f1_05),
        "micro_f1@tuned": float(micro_f1_th),
        "per_label_auc": per_label_auc,
        "per_label_f1@tuned": per_label_f1,
        "thresholds": thresholds.astype(float).tolist(),
    }


def pick_key_labels(label_names: list[str], y_train: np.ndarray, top_n: int = 5) -> list[int]:
    """
    Pick key labels for reporting: top_n most prevalent positives in training.
    """
    y_train = y_train.astype(np.int32)
    pos_counts = y_train.sum(axis=0)
    idx = np.argsort(-pos_counts)[:top_n]
    return idx.tolist()