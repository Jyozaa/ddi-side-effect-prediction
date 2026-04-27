from __future__ import annotations

import re
import pandas as pd
from pathlib import Path


CID_TO_NAME_CSV = "data/processed/cid_to_name.csv"
DGIDB_INTERACTIONS_TSV = "data/raw/dgidb/interactions.tsv"
OUTPUT_CSV = "data/raw/drug_targets.csv"


def normalize_name(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s


def main() -> None:
    cid_df = pd.read_csv(CID_TO_NAME_CSV)
    dgidb_df = pd.read_csv(DGIDB_INTERACTIONS_TSV, sep="\t", low_memory=False)

    cid_required = ["drug_id", "drug_name"]
    dgidb_required = ["drug_name", "gene_name"]

    missing_cid = [c for c in cid_required if c not in cid_df.columns]
    missing_dgidb = [c for c in dgidb_required if c not in dgidb_df.columns]

    if missing_cid:
        raise KeyError(f"Missing columns in {CID_TO_NAME_CSV}: {missing_cid}")
    if missing_dgidb:
        raise KeyError(f"Missing columns in {DGIDB_INTERACTIONS_TSV}: {missing_dgidb}")

    # Keep only rows with non-empty names
    cid_df["drug_name"] = cid_df["drug_name"].fillna("").astype(str).str.strip()
    cid_df = cid_df[cid_df["drug_name"] != ""].copy()

    dgidb_df["drug_name"] = dgidb_df["drug_name"].fillna("").astype(str).str.strip()
    dgidb_df["gene_name"] = dgidb_df["gene_name"].fillna("").astype(str).str.strip()
    dgidb_df = dgidb_df[(dgidb_df["drug_name"] != "") & (dgidb_df["gene_name"] != "")].copy()

    # Normalize names for exact matching
    cid_df["drug_name_norm"] = cid_df["drug_name"].map(normalize_name)
    dgidb_df["drug_name_norm"] = dgidb_df["drug_name"].map(normalize_name)

    # Drop empty normalized names
    cid_df = cid_df[cid_df["drug_name_norm"] != ""].copy()
    dgidb_df = dgidb_df[dgidb_df["drug_name_norm"] != ""].copy()

    # Join CID names to DGIdb by normalized drug name
    merged = cid_df.merge(
        dgidb_df[["drug_name", "drug_name_norm", "gene_name"]],
        on="drug_name_norm",
        how="inner",
        suffixes=("_cid", "_dgidb"),
    )

    # Build final target mapping
    out = merged[["drug_id", "gene_name"]].rename(columns={"gene_name": "target_id"}).copy()
    out["drug_id"] = out["drug_id"].astype(str)
    out["target_id"] = out["target_id"].astype(str)

    out = out.dropna(subset=["drug_id", "target_id"]).drop_duplicates().reset_index(drop=True)

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)

    # Stats
    matched_drugs = merged["drug_id"].nunique()
    total_named_drugs = cid_df["drug_id"].nunique()
    unmatched_drugs = total_named_drugs - matched_drugs

    print("Saved:", OUTPUT_CSV)
    print("Total named CID drugs:", total_named_drugs)
    print("Matched CID drugs:", matched_drugs)
    print("Unmatched CID drugs:", unmatched_drugs)
    print("Drug-target rows:", len(out))

    matched_names = merged[["drug_id", "drug_name_cid", "drug_name_dgidb"]].drop_duplicates()
    matched_names.to_csv("data/processed/dgidb_name_matches.csv", index=False)

    unmatched = cid_df[~cid_df["drug_id"].isin(merged["drug_id"].unique())][["drug_id", "drug_name"]].drop_duplicates()
    unmatched.to_csv("data/processed/dgidb_unmatched_cid_names.csv", index=False)

    print("Saved debug match file: data/processed/dgidb_name_matches.csv")
    print("Saved unmatched file: data/processed/dgidb_unmatched_cid_names.csv")


if __name__ == "__main__":
    main()