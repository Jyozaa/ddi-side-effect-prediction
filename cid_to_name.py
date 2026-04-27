from __future__ import annotations

import time
import requests
import pandas as pd
from pathlib import Path

INPUT = "data/processed/smiles_map.csv"
OUTPUT = "data/processed/cid_to_name.csv"

def cid_string_to_int(s: str) -> int | None:
    s = str(s).strip()
    if not s.startswith("CID"):
        return None
    num = s[3:].lstrip("0")
    if num == "":
        num = "0"
    try:
        return int(num)
    except ValueError:
        return None

def get_pubchem_title(cid: int) -> str | None:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/Title/JSON"
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return None
    try:
        data = r.json()
        props = data["PropertyTable"]["Properties"]
        if props and "Title" in props[0]:
            return str(props[0]["Title"]).strip()
    except Exception:
        return None
    return None

def get_pubchem_synonyms(cid: int) -> list[str]:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return []
    try:
        data = r.json()
        info = data["InformationList"]["Information"]
        if info and "Synonym" in info[0]:
            return [str(x).strip() for x in info[0]["Synonym"] if str(x).strip()]
    except Exception:
        return []
    return []

def choose_name(title: str | None, synonyms: list[str]) -> str:
    if title:
        return title
    for s in synonyms:
        # crude filter to avoid very ugly long systematic strings if possible
        if len(s) <= 80 and not s.startswith("SCHEMBL") and not s.startswith("CHEMBL"):
            return s
    return synonyms[0] if synonyms else ""

def main():
    df = pd.read_csv(INPUT)
    rows = []

    for drug_id in df["drug_id"].astype(str).unique():
        cid = cid_string_to_int(drug_id)
        if cid is None:
            rows.append({"drug_id": drug_id, "pubchem_cid": None, "drug_name": ""})
            continue

        title = get_pubchem_title(cid)
        if not title:
            syns = get_pubchem_synonyms(cid)
        else:
            syns = []

        name = choose_name(title, syns)

        rows.append({
            "drug_id": drug_id,
            "pubchem_cid": cid,
            "drug_name": name,
        })

        time.sleep(0.15)  

    out = pd.DataFrame(rows)
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)
    print(f"Saved {OUTPUT} with {len(out)} rows")

if __name__ == "__main__":
    main()