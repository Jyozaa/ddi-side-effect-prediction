import pandas as pd
from pathlib import Path
from src.utils.io import load_yaml, ensure_dir

def main(cfg_path: str = "configs/default.yaml") -> None:
    cfg = load_yaml(cfg_path)

    in_path = cfg["data"]["smiles_pairs_path"]
    out_csv = Path(cfg["smiles"]["smiles_cache_csv"])
    ensure_dir(str(out_csv.parent))

    df = pd.read_csv(in_path)

    required = ["ID1", "ID2", "X1", "X2"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in smiles_pairs_path CSV: {missing_cols}. Have: {list(df.columns)}")

    # Build mapping from ID -> SMILES using both columns
    a = df[["ID1", "X1"]].rename(columns={"ID1": "drug_id", "X1": "smiles"})
    b = df[["ID2", "X2"]].rename(columns={"ID2": "drug_id", "X2": "smiles"})
    m = pd.concat([a, b], ignore_index=True)

    m["drug_id"] = m["drug_id"].astype(str)
    m["smiles"] = m["smiles"].astype(str)

    # Drop empty / invalid entries
    m = m.dropna(subset=["drug_id", "smiles"])
    m = m[m["smiles"].str.len() > 0]

    # Keep last occurrence per drug_id
    m = m.drop_duplicates(subset=["drug_id"], keep="last")

    # Add columns expected by baseline
    m["drug_name"] = ""
    m["source"] = "local_smiles_pairs"

    m = m[["drug_id", "drug_name", "smiles", "source"]]
    m.to_csv(out_csv, index=False)

    print("SMILES cache saved:", out_csv)
    print("Mapped drugs:", len(m))

if __name__ == "__main__":
    main()
