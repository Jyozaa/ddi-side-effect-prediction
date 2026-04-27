from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from scipy.sparse import csr_matrix

from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.exceptions import ConvergenceWarning
import warnings

from src.utils.io import load_yaml
from src.eval.multilabel_metrics import (
    tune_thresholds_per_label,
    compute_multilabel_metrics,
    pick_key_labels,
)


def load_run_dir(output_dir: str, top_k: int, split_type: str) -> Path:
    run_dir = Path(output_dir) / f"multilabel_top{top_k}_{split_type}"
    if not run_dir.exists():
        raise FileNotFoundError(f"Cannot find run dir: {run_dir} (did you run make_multilabel_dataset?)")
    return run_dir


def load_smiles_map(path: str) -> dict[str, str]:
    df = pd.read_csv(path)
    return dict(zip(df["drug_id"].astype(str), df["smiles"].astype(str)))


def morgan_onbits(smiles: str, gen) -> list[int] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = gen.GetFingerprint(mol)
    return list(fp.GetOnBits())


def build_sparse_X_concat(
    pairs: pd.DataFrame,
    smiles_map: dict[str, str],
    radius: int,
    nbits: int,
    verbose_every: int = 5000
):
    gen = GetMorganGenerator(radius=radius, fpSize=nbits)

    rows, cols, data = [], [], []
    keep_idx = []

    for i, r in pairs.iterrows():
        d1 = str(r["drug1_id"])
        d2 = str(r["drug2_id"])
        s1 = smiles_map.get(d1)
        s2 = smiles_map.get(d2)
        if not s1 or not s2:
            continue

        b1 = morgan_onbits(s1, gen)
        b2 = morgan_onbits(s2, gen)
        if b1 is None or b2 is None:
            continue

        row_id = len(keep_idx)

        # fp1 bits [0..nbits-1]
        for c in b1:
            rows.append(row_id)
            cols.append(c)
            data.append(1.0)

        # fp2 bits [nbits..2*nbits-1]
        off = nbits
        for c in b2:
            rows.append(row_id)
            cols.append(off + c)
            data.append(1.0)

        keep_idx.append(i)

        if verbose_every and len(keep_idx) % verbose_every == 0:
            print(f"  featurized {len(keep_idx)} pairs...")

    X = csr_matrix((data, (rows, cols)), shape=(len(keep_idx), 2 * nbits), dtype=np.float32)
    pairs2 = pairs.loc[keep_idx].reset_index(drop=True)
    return X, pairs2


def main(cfg_path: str = "configs/default.yaml") -> None:
    cfg = load_yaml(cfg_path)

    seed = int(cfg["data"]["seed"])
    split_type = cfg["split"]["type"]
    top_k = int(cfg["multilabel"]["top_k_effects"])

    run_dir = load_run_dir(cfg["data"]["output_dir"], top_k=top_k, split_type=split_type)

    train_df = pd.read_csv(run_dir / "pairs_train.csv")
    val_df = pd.read_csv(run_dir / "pairs_val.csv")
    test_df = pd.read_csv(run_dir / "pairs_test.csv")

    label_cols = [c for c in train_df.columns if c not in ["drug1_id", "drug2_id"]]
    n_labels = len(label_cols)
    if n_labels < 2:
        raise RuntimeError("This does not look like a multi-label dataset (need multiple label columns).")

    smiles_map = load_smiles_map(cfg["smiles"]["smiles_cache_csv"])

    radius = int(cfg["baseline"]["morgan_radius"])
    nbits = int(cfg["baseline"]["morgan_nbits"])

    print("Building sparse features (train)...")
    X_train, train_df2 = build_sparse_X_concat(train_df, smiles_map, radius, nbits, verbose_every=5000)
    y_train = train_df2[label_cols].values.astype(np.int32)

    print("Building sparse features (val)...")
    X_val, val_df2 = build_sparse_X_concat(val_df, smiles_map, radius, nbits, verbose_every=0)
    y_val = val_df2[label_cols].values.astype(np.int32)

    print("Building sparse features (test)...")
    X_test, test_df2 = build_sparse_X_concat(test_df, smiles_map, radius, nbits, verbose_every=0)
    y_test = test_df2[label_cols].values.astype(np.int32)

    print(f"Shapes: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"        X_val={X_val.shape}, y_val={y_val.shape}")
    print(f"        X_test={X_test.shape}, y_test={y_test.shape}")

    # Multilabel logistic regression
    base = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        max_iter=1000,
        tol=1e-4,
        random_state=seed,
    )
    clf = OneVsRestClassifier(base, n_jobs=-1)

    print("Fitting OneVsRest SGD logistic regression...")
    clf.fit(X_train, y_train)

    print("Predicting probabilities...")
    val_probs = clf.predict_proba(X_val).astype(np.float32)
    test_probs = clf.predict_proba(X_test).astype(np.float32)

    # Tune thresholds using validation set
    ths = tune_thresholds_per_label(y_val, val_probs)

    # Metrics
    val_metrics = compute_multilabel_metrics(y_val, val_probs, thresholds=ths)
    test_metrics = compute_multilabel_metrics(y_test, test_probs, thresholds=ths)

    # Key per-label reporting
    key_idx = pick_key_labels(label_cols, y_train, top_n=5)
    key_labels = []
    for ki in key_idx:
        key_labels.append({
            "label": label_cols[ki],
            "test_auc": test_metrics["per_label_auc"][ki],
            "test_f1@tuned": test_metrics["per_label_f1@tuned"][ki],
            "threshold": float(ths[ki]),
            "train_pos": int(y_train[:, ki].sum()),
        })

    results = {
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

        "n_train_used": int(len(train_df2)),
        "n_val_used": int(len(val_df2)),
        "n_test_used": int(len(test_df2)),
        "feature_dim": int(X_train.shape[1]),

        "key_label_results": key_labels,
    }

    out_path = run_dir / "baseline_multilabel_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved:", out_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()