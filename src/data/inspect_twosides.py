import pandas as pd
from collections import Counter
from src.utils.io import load_yaml

# Column candidates
DRUG1_ID_COLS = ["drug_1_rxnorn_id", "drug_1_rxnorm_id", "drug1_rxnorm_id", "drug_1_concept_id", "drug1_id"]
DRUG2_ID_COLS = ["drug_2_rxnorm_id", "drug2_rxnorm_id", "drug_2_concept_id", "drug2_id"]

DRUG1_NAME_COLS = ["drug_1_concept_name"]
DRUG2_NAME_COLS = ["drug_2_concept_name"]

EFFECT_NAME_COLS = ["condition_concept_name"]


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns found: {candidates}\nAvailable: {list(df.columns)[:40]} ...")

def main(cfg_path: str = "configs/default.yaml") -> None:
    cfg = load_yaml(cfg_path)
    path = cfg["data"]["twosides_path"]
    
    usecols = [
    "drug_1_rxnorn_id",
    "drug_2_rxnorm_id",
    "condition_concept_name",
    ]
    df = pd.read_csv(
        path,
        compression="infer",
        usecols=usecols,
        low_memory=False,
        dtype={
            "drug_1_rxnorn_id": "string",
            "drug_2_rxnorm_id": "string",
            "condition_concept_name": "string",
        }
    )

    print("Loaded:", path)
    print("Rows:", len(df))
    print("Cols:", len(df.columns))
    print("First columns:", list(df.columns)[:30])

    effect_col = pick_col(df, EFFECT_NAME_COLS)
    counts = Counter(df[effect_col].astype(str).tolist())
    print("\nTop 15 side effects:")
    for name, n in counts.most_common(15):
        print(f"{n:>8}  {name}")

if __name__ == "__main__":
    main()
