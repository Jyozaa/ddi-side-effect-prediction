import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from src.utils.io import load_yaml

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

def morgan_fp(smiles: str, radius: int, nbits: int):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def load_latest_effect_dir(processed_dir: str) -> Path:
    p = Path(processed_dir)
    cands = sorted([d for d in p.glob("*") if (d / "pairs_train.csv").exists()])
    if not cands:
        raise FileNotFoundError("No processed pairs found. Run make_pairs.py first.")
    return cands[-1]

def main(cfg_path: str = "configs/default.yaml") -> None:
    cfg = load_yaml(cfg_path)
    effect_dir = load_latest_effect_dir(cfg["data"]["output_dir"])

    train_df = pd.read_csv(effect_dir / "pairs_train.csv")
    val_df = pd.read_csv(effect_dir / "pairs_val.csv")
    test_df = pd.read_csv(effect_dir / "pairs_test.csv")

    smiles_csv = Path(cfg["smiles"]["smiles_cache_csv"])
    if not smiles_csv.exists():
        raise FileNotFoundError("smiles_map.csv not found. Run: python -m src.features.smiles_map")

    sm = pd.read_csv(smiles_csv).dropna(subset=["smiles"])
    sm["drug_id"] = sm["drug_id"].astype(str)
    sm_map = dict(zip(sm["drug_id"], sm["smiles"]))

    radius = int(cfg["baseline"]["morgan_radius"])
    nbits = int(cfg["baseline"]["morgan_nbits"])

    # Cache fingerprints
    fp_cache = {}

    def pair_to_vec(row):
        a = str(row["drug1_id"]); b = str(row["drug2_id"])
        sa = sm_map.get(a); sb = sm_map.get(b)
        if not sa or not sb:
            return None
        if a not in fp_cache:
            fp_cache[a] = morgan_fp(sa, radius, nbits)
        if b not in fp_cache:
            fp_cache[b] = morgan_fp(sb, radius, nbits)
        if fp_cache[a] is None or fp_cache[b] is None:
            return None
        return np.concatenate([fp_cache[a], fp_cache[b]], axis=0)

    def build_xy(df):
        X, y = [], []
        for _, r in tqdm(df.iterrows(), total=len(df), desc="Vectorizing"):
            v = pair_to_vec(r)
            if v is None:
                continue
            X.append(v)
            y.append(int(r["label"]))
        if not X:
            raise RuntimeError("No samples with SMILES available. Improve SMILES mapping coverage.")
        return np.stack(X), np.array(y, dtype=np.int64)

    Xtr, ytr = build_xy(train_df)
    Xva, yva = build_xy(val_df)
    Xte, yte = build_xy(test_df)

    clf = LogisticRegression(
        C=float(cfg["baseline"]["C"]),
        max_iter=int(cfg["baseline"]["max_iter"]),
        class_weight="balanced",
        n_jobs=-1
    )
    clf.fit(Xtr, ytr)

    def eval_split(X, y):
        probs = clf.predict_proba(X)[:, 1]
        pred = (probs >= 0.5).astype(int)
        return {
            "roc_auc": float(roc_auc_score(y, probs)),
            "f1": float(f1_score(y, pred)),
            "accuracy": float(accuracy_score(y, pred)),
            "n": int(len(y)),
            "pos_rate": float(y.mean()),
        }

    metrics = {
        "train": eval_split(Xtr, ytr),
        "val": eval_split(Xva, yva),
        "test": eval_split(Xte, yte),
        "effect_dir": str(effect_dir),
        "mapped_drugs": int(len(sm_map)),
        "vector_dim": int(Xtr.shape[1]),
    }

    out = effect_dir / "baseline_lr_metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Saved:", out)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
