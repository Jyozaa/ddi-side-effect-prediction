from __future__ import annotations

import numpy as np
import pandas as pd


def load_target_mapping(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = ["drug_id", "target_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in target file: {missing}. Have: {list(df.columns)}")

    df["drug_id"] = df["drug_id"].astype(str)
    df["target_id"] = df["target_id"].astype(str)
    df = df.dropna(subset=["drug_id", "target_id"]).drop_duplicates()

    return df


def build_target_vocab(df: pd.DataFrame, min_freq: int = 1, max_targets: int | None = None) -> list[str]:
    counts = df["target_id"].value_counts()
    targets = [t for t, c in counts.items() if c >= min_freq]

    if max_targets is not None and len(targets) > max_targets:
        targets = targets[:max_targets]

    return targets


def build_target_cache(
    df: pd.DataFrame,
    target_vocab: list[str],
) -> dict[str, np.ndarray]:
    target_to_idx = {t: i for i, t in enumerate(target_vocab)}
    dim = len(target_vocab)

    cache: dict[str, np.ndarray] = {}

    for drug_id, sub in df.groupby("drug_id"):
        vec = np.zeros((dim,), dtype=np.float32)
        for target_id in sub["target_id"].tolist():
            idx = target_to_idx.get(target_id)
            if idx is not None:
                vec[idx] = 1.0
        cache[str(drug_id)] = vec

    return cache