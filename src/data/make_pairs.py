import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from src.utils.io import load_yaml, ensure_dir
from src.utils.seed import set_seed

DRUG1_ID_COLS = ["drug_1_rxnorn_id", "drug_1_rxnorm_id", "drug1_rxnorm_id", "drug_1_concept_id", "drug1_concept_id", "drug1_id"]
DRUG2_ID_COLS = ["drug_2_rxnorm_id", "drug2_rxnorm_id", "drug_2_concept_id", "drug2_concept_id", "drug2_id"]
DRUG1_NAME_COLS = ["drug_1_concept_name", "drug1_concept_name", "drug1_name"]
DRUG2_NAME_COLS = ["drug_2_concept_name", "drug2_concept_name", "drug2_name"]
EFFECT_NAME_COLS = ["condition_concept_name"]

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Missing columns. Tried {candidates}. Have {list(df.columns)[:40]} ...")

def canonical_pair(a, b):
    return (a, b) if str(a) <= str(b) else (b, a)

def main(cfg_path: str = "configs/default.yaml") -> None:
    cfg = load_yaml(cfg_path)
    set_seed(cfg["data"]["seed"])

    twosides_path = cfg["data"]["twosides_path"]
    out_dir = Path(cfg["data"]["output_dir"])
    ensure_dir(str(out_dir))

    usecols = [
    "drug_1_rxnorn_id", 
    "drug_2_rxnorm_id",
    "drug_1_concept_name",
    "drug_2_concept_name",
    "condition_concept_name",
    ]

    df = pd.read_csv(
        twosides_path,
        compression="infer",
        usecols=usecols,
        low_memory=False,
        dtype={
            "drug_1_rxnorn_id": "string",
            "drug_2_rxnorm_id": "string",
            "drug_1_concept_name": "string",
            "drug_2_concept_name": "string",
            "condition_concept_name": "string",
        }
    )

    d1_id = pick_col(df, DRUG1_ID_COLS)
    d2_id = pick_col(df, DRUG2_ID_COLS)
    d1_name = pick_col(df, DRUG1_NAME_COLS)
    d2_name = pick_col(df, DRUG2_NAME_COLS)
    eff_col = pick_col(df, EFFECT_NAME_COLS)

    # Choose effect
    effect_name = cfg["data"]["effect_name"]
    if effect_name is None:
        counts = Counter(df[eff_col].astype(str).tolist())
        effect_name = counts.most_common(1)[0][0]
        print(f"[Auto-picked effect] {effect_name}")

    # Positives for chosen effect
    pos = df[df[eff_col].astype(str) == str(effect_name)].copy()
    print("Pos rows:", len(pos))

    # Canonicalize pairs
    pos_pairs = set()
    pos_rows = []
    for _, r in pos.iterrows():
        a, b = canonical_pair(r[d1_id], r[d2_id])
        key = (a, b)
        if key in pos_pairs:
            continue
        pos_pairs.add(key)
        pos_rows.append((a, b, r[d1_name], r[d2_name], 1))

    pos_df = pd.DataFrame(pos_rows, columns=["drug1_id", "drug2_id", "drug1_name", "drug2_name", "label"])
    print("Unique positive pairs:", len(pos_df))

    # Build negatives by sampling from observed drugs
    drugs = pd.unique(pd.concat([df[d1_id], df[d2_id]], axis=0))
    drugs = [d for d in drugs if pd.notna(d)]
    rng = np.random.default_rng(cfg["data"]["seed"])

    neg_ratio = int(cfg["data"]["negative_ratio"])
    n_neg = len(pos_df) * neg_ratio
    print("Target negatives:", n_neg)

    neg_pairs = set()
    neg_rows = []

    pos_lookup = set((str(a), str(b)) for (a, b) in pos_pairs)

    # name lookup
    name_map = {}
    for _, r in df[[d1_id, d1_name]].dropna().drop_duplicates(subset=[d1_id]).iterrows():
        name_map[str(r[d1_id])] = str(r[d1_name])
    for _, r in df[[d2_id, d2_name]].dropna().drop_duplicates(subset=[d2_id]).iterrows():
        name_map.setdefault(str(r[d2_id]), str(r[d2_name]))

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
        if key in pos_lookup:
            continue
        if key in neg_pairs:
            continue
        neg_pairs.add(key)
        neg_rows.append((a2, b2, name_map.get(str(a2), ""), name_map.get(str(b2), ""), 0))

    neg_df = pd.DataFrame(neg_rows, columns=["drug1_id", "drug2_id", "drug1_name", "drug2_name", "label"])
    print("Sampled negatives:", len(neg_df))

    # Combine
    all_df = pd.concat([pos_df, neg_df], ignore_index=True)

    max_pairs = cfg["data"]["max_pairs"]
    if max_pairs is not None and len(all_df) > int(max_pairs):
        all_df = all_df.sample(n=int(max_pairs), random_state=cfg["data"]["seed"]).reset_index(drop=True)

    # Shuffle
    all_df = all_df.sample(frac=1.0, random_state=cfg["data"]["seed"]).reset_index(drop=True)

    # Split
    n = len(all_df)
    tr = cfg["data"]["split"]["train"]
    va = cfg["data"]["split"]["val"]
    n_train = int(n * tr)
    n_val = int(n * va)
    train_df = all_df.iloc[:n_train]
    val_df = all_df.iloc[n_train:n_train + n_val]
    test_df = all_df.iloc[n_train + n_val:]

    # Save
    effect_slug = str(effect_name).strip().lower().replace(" ", "_").replace("/", "_")
    out_effect_dir = out_dir / effect_slug
    ensure_dir(str(out_effect_dir))

    train_df.to_csv(out_effect_dir / "pairs_train.csv", index=False)
    val_df.to_csv(out_effect_dir / "pairs_val.csv", index=False)
    test_df.to_csv(out_effect_dir / "pairs_test.csv", index=False)

    meta = {
        "effect_name": effect_name,
        "n_total": n,
        "n_pos": int(all_df["label"].sum()),
        "n_neg": int((all_df["label"] == 0).sum()),
        "unique_drugs": int(pd.unique(pd.concat([all_df["drug1_id"], all_df["drug2_id"]])).shape[0]),
    }
    pd.Series(meta).to_json(out_effect_dir / "meta.json")
    print("Saved to:", out_effect_dir)
    print("Meta:", meta)

if __name__ == "__main__":
    main()
