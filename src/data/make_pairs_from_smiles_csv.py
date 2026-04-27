import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from src.utils.io import load_yaml, ensure_dir
from src.utils.seed import set_seed

def canonical_pair(a, b):
    return (a, b) if str(a) <= str(b) else (b, a)

def main(cfg_path: str = "configs/default.yaml") -> None:
    cfg = load_yaml(cfg_path)
    set_seed(int(cfg["data"]["seed"]))

    in_path = cfg["data"]["smiles_pairs_path"]
    out_dir = Path(cfg["data"]["output_dir"])
    ensure_dir(str(out_dir))

    df = pd.read_csv(in_path)

    required = ["ID1", "ID2", "Side Effect Name", "X1", "X2"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in smiles_pairs_path CSV: {missing_cols}. Have: {list(df.columns)}")

    effect_name = cfg["data"]["effect_name"]
    if effect_name is None:
        counts = Counter(df["Side Effect Name"].astype(str).tolist())
        effect_name = counts.most_common(1)[0][0]
        print(f"[Auto-picked effect] {effect_name}")

    # Positives for chosen effect
    pos = df[df["Side Effect Name"].astype(str) == str(effect_name)].copy()

    # Unique positive pairs
    pos_pairs = set()
    pos_rows = []
    for _, r in pos.iterrows():
        a, b = canonical_pair(str(r["ID1"]), str(r["ID2"]))
        if (a, b) in pos_pairs:
            continue
        pos_pairs.add((a, b))
        pos_rows.append((a, b, 1))

    pos_df = pd.DataFrame(pos_rows, columns=["drug1_id", "drug2_id", "label"])
    print("Unique positive pairs:", len(pos_df))

    # Negatives, sample from all unique drugs
    drugs = pd.unique(pd.concat([df["ID1"].astype(str), df["ID2"].astype(str)], axis=0))
    rng = np.random.default_rng(int(cfg["data"]["seed"]))

    neg_ratio = int(cfg["data"]["negative_ratio"])
    n_neg = len(pos_df) * neg_ratio
    pos_lookup = set((str(a), str(b)) for (a, b) in pos_pairs)

    neg_pairs = set()
    neg_rows = []
    tries = 0
    max_tries = n_neg * 50 + 1000

    while len(neg_rows) < n_neg and tries < max_tries:
        tries += 1
        a = drugs[int(rng.integers(0, len(drugs)))]
        b = drugs[int(rng.integers(0, len(drugs)))]
        if str(a) == str(b):
            continue
        a2, b2 = canonical_pair(a, b)
        key = (str(a2), str(b2))
        if key in pos_lookup or key in neg_pairs:
            continue
        neg_pairs.add(key)
        neg_rows.append((a2, b2, 0))

    neg_df = pd.DataFrame(neg_rows, columns=["drug1_id", "drug2_id", "label"])
    print("Sampled negatives:", len(neg_df))

    all_df = pd.concat([pos_df, neg_df], ignore_index=True).sample(
        frac=1.0, random_state=int(cfg["data"]["seed"])
    ).reset_index(drop=True)

    # Split
    n = len(all_df)
    tr = float(cfg["data"]["split"]["train"])
    va = float(cfg["data"]["split"]["val"])
    n_train = int(n * tr)
    n_val = int(n * va)

    train_df = all_df.iloc[:n_train]
    val_df = all_df.iloc[n_train:n_train + n_val]
    test_df = all_df.iloc[n_train + n_val:]

    effect_slug = str(effect_name).strip().lower().replace(" ", "_").replace("/", "_")
    out_effect_dir = out_dir / effect_slug
    ensure_dir(str(out_effect_dir))

    train_df.to_csv(out_effect_dir / "pairs_train.csv", index=False)
    val_df.to_csv(out_effect_dir / "pairs_val.csv", index=False)
    test_df.to_csv(out_effect_dir / "pairs_test.csv", index=False)

    meta = {
        "effect_name": effect_name,
        "n_total": int(n),
        "n_pos": int(all_df["label"].sum()),
        "n_neg": int((all_df["label"] == 0).sum()),
        "unique_drugs": int(pd.unique(pd.concat([all_df["drug1_id"], all_df["drug2_id"]])).shape[0]),
        "source": "smiles_pairs_path",
        "input_csv": in_path,
    }
    pd.Series(meta).to_json(out_effect_dir / "meta.json")
    print("Saved to:", out_effect_dir)
    print("Meta:", meta)

if __name__ == "__main__":
    main()
