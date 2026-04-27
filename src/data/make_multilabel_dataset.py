import json
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.io import load_yaml, ensure_dir
from src.utils.seed import set_seed

def canonical_pair(a, b):
    a = str(a); b = str(b)
    return (a, b) if a <= b else (b, a)

def build_pair_labels(df: pd.DataFrame, effects: list[str]) -> pd.DataFrame:
    # Keep only top effects
    df = df[df["Side Effect Name"].isin(effects)].copy()

    # Canonicalize pairs so (A,B) == (B,A)
    pairs = df.apply(lambda r: canonical_pair(r["ID1"], r["ID2"]), axis=1)
    df["p1"] = [p[0] for p in pairs]
    df["p2"] = [p[1] for p in pairs]

    df["value"] = 1
    pivot = (
        df.pivot_table(
            index=["p1", "p2"],
            columns="Side Effect Name",
            values="value",
            aggfunc="max",
            fill_value=0,
        )
        .reset_index()
    )

    pivot.rename(columns={"p1": "drug1_id", "p2": "drug2_id"}, inplace=True)
    return pivot

def make_random_split(pairs_df: pd.DataFrame, seed: int, train: float, val: float):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(pairs_df))
    rng.shuffle(idx)

    n = len(idx)
    n_train = int(n * train)
    n_val = int(n * val)

    tr = pairs_df.iloc[idx[:n_train]].reset_index(drop=True)
    va = pairs_df.iloc[idx[n_train:n_train+n_val]].reset_index(drop=True)
    te = pairs_df.iloc[idx[n_train+n_val:]].reset_index(drop=True)
    return tr, va, te

def make_cold_start_split(pairs_df: pd.DataFrame, seed: int, holdout_drug_frac: float):
    """
    Cold-start (both drugs held out):
    - sample holdout drugs
    - test pairs are those where BOTH drugs are in holdout set
    - remaining pairs are split into train/val randomly
    """
    rng = np.random.default_rng(seed)
    drugs = pd.unique(pd.concat([pairs_df["drug1_id"], pairs_df["drug2_id"]], axis=0))
    drugs = np.array([str(d) for d in drugs])

    n_hold = max(1, int(len(drugs) * holdout_drug_frac))
    holdout = set(rng.choice(drugs, size=n_hold, replace=False))

    is_test = pairs_df["drug1_id"].isin(holdout) & pairs_df["drug2_id"].isin(holdout)
    test_df = pairs_df[is_test].reset_index(drop=True)

    remaining = pairs_df[~is_test].reset_index(drop=True)

    # Split remaining into train/val
    train_df, val_df, _ = make_random_split(remaining, seed=seed, train=0.85, val=0.15)
    return train_df, val_df, test_df, holdout

def main(cfg_path: str = "configs/default.yaml") -> None:
    cfg = load_yaml(cfg_path)
    seed = int(cfg["data"]["seed"])
    set_seed(seed)

    in_path = cfg["data"]["smiles_pairs_path"]
    out_dir = Path(cfg["data"]["output_dir"])
    ensure_dir(str(out_dir))

    raw = pd.read_csv(in_path)
    required = ["ID1", "ID2", "Side Effect Name", "X1", "X2"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}. Have: {list(raw.columns)}")

    # Choose all eligible effects above a minimum positive-count threshold
    counts = raw["Side Effect Name"].value_counts()
    min_pos = int(cfg["multilabel"]["min_pos_per_label"])

    candidate_effects = [e for e, c in counts.items() if c >= min_pos]

    top_k = cfg["multilabel"].get("top_k_effects", None)
    if top_k is None or int(top_k) <= 0:
        effects = candidate_effects
    else:
        effects = candidate_effects[:int(top_k)]

    if len(effects) < 2:
        raise RuntimeError("Too few side effects after filtering. Lower min_pos_per_label or increase dataset.")

    pairs_df = build_pair_labels(raw, effects)

    max_pairs = cfg["multilabel"].get("max_pairs", None)
    if max_pairs is not None and len(pairs_df) > int(max_pairs):
        pairs_df = pairs_df.sample(n=int(max_pairs), random_state=seed).reset_index(drop=True)

    # Decide split
    split_type = cfg["split"]["type"]
    if split_type == "random":
        train_df, val_df, test_df = make_random_split(
            pairs_df, seed=seed, train=float(cfg["split"]["train"]), val=float(cfg["split"]["val"])
        )
        holdout = None
    elif split_type == "cold_start":
        train_df, val_df, test_df, holdout = make_cold_start_split(
            pairs_df, seed=seed, holdout_drug_frac=float(cfg["split"]["holdout_drug_frac"])
        )
    else:
        raise ValueError("split.type must be 'random' or 'cold_start'")

    # Save
    run_name = f"multilabel_top{len(effects)}_{split_type}"
    out_run = out_dir / run_name
    ensure_dir(str(out_run))

    train_df.to_csv(out_run / "pairs_train.csv", index=False)
    val_df.to_csv(out_run / "pairs_val.csv", index=False)
    test_df.to_csv(out_run / "pairs_test.csv", index=False)

    meta = {
        "source": "smiles_pairs_path",
        "input_csv": in_path,
        "split_type": split_type,
        "n_pairs_total": int(len(pairs_df)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "n_effects": int(len(effects)),
        "effects": effects,
        "min_pos_per_label": min_pos,
        "max_pairs": max_pairs,
        "holdout_drug_frac": cfg["split"].get("holdout_drug_frac", None),
        "holdout_drugs": sorted(list(holdout)) if holdout is not None else None,
        "label_columns": [c for c in train_df.columns if c not in ["drug1_id", "drug2_id"]],
    }
    (out_run / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Saved multilabel dataset to:", out_run)
    print("Meta summary:", {k: meta[k] for k in ["split_type","n_pairs_total","n_train","n_val","n_test","n_effects"]})

if __name__ == "__main__":
    main()
