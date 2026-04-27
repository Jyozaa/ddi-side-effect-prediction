import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from rdkit import Chem
from pathlib import Path
from tqdm import tqdm

from src.utils.io import load_yaml
from src.utils.seed import set_seed
from src.models.gnn_pair import PairClassifier

def mol_to_graph(smiles: str) -> Data | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = []
    for atom in mol.GetAtoms():
        x.append([float(atom.GetAtomicNum())])
    x = torch.tensor(x, dtype=torch.float)

    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append((i, j))
        edges.append((j, i))

    if not edges:
        return None

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)

class PairGraphDataset(Dataset):
    def __init__(self, pairs_df: pd.DataFrame, smiles_map: dict[str, str]):
        self.df = pairs_df.reset_index(drop=True)
        self.smiles_map = smiles_map
        self.graph_cache = {}

        keep = []
        for i, r in self.df.iterrows():
            a = str(r["drug1_id"]); b = str(r["drug2_id"])
            if a in smiles_map and b in smiles_map and smiles_map[a] and smiles_map[b]:
                keep.append(i)
        self.df = self.df.loc[keep].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def get_graph(self, drug_id: str) -> Data | None:
        if drug_id in self.graph_cache:
            return self.graph_cache[drug_id]
        g = mol_to_graph(self.smiles_map[drug_id])
        self.graph_cache[drug_id] = g
        return g

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        a = str(r["drug1_id"]); b = str(r["drug2_id"])
        y = torch.tensor(float(r["label"]), dtype=torch.float)

        ga = self.get_graph(a)
        gb = self.get_graph(b)
        if ga is None or gb is None:
            return None
        return ga, gb, y

def collate_fn(batch):
    batch = [x for x in batch if x is not None]
    ga_list, gb_list, y_list = zip(*batch)
    ba = Batch.from_data_list(list(ga_list))
    bb = Batch.from_data_list(list(gb_list))
    y = torch.stack(list(y_list), dim=0)
    return ba, bb, y

def load_latest_effect_dir(processed_dir: str) -> Path:
    p = Path(processed_dir)
    cands = sorted([d for d in p.glob("*") if (d / "pairs_train.csv").exists()])
    if not cands:
        raise FileNotFoundError("No processed pairs found. Run make_pairs.py first.")
    return cands[-1]

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits = []
    all_y = []
    for ba, bb, y in loader:
        ba = ba.to(device); bb = bb.to(device); y = y.to(device)
        logits = model(ba, bb)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())
    logits = torch.cat(all_logits)
    y = torch.cat(all_y)

    probs = torch.sigmoid(logits).numpy()
    pred = (probs >= 0.5).astype(int)
    y_np = y.numpy().astype(int)

    acc = float((pred == y_np).mean())

    # f1
    tp = int(((pred == 1) & (y_np == 1)).sum())
    fp = int(((pred == 1) & (y_np == 0)).sum())
    fn = int(((pred == 0) & (y_np == 1)).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = float(2 * precision * recall / (precision + recall + 1e-9))

    return {"accuracy": acc, "f1": f1, "n": int(len(y_np)), "pos_rate": float(y_np.mean())}

def main(cfg_path: str = "configs/default.yaml") -> None:
    cfg = load_yaml(cfg_path)
    set_seed(int(cfg["data"]["seed"]))

    effect_dir = load_latest_effect_dir(cfg["data"]["output_dir"])
    train_df = pd.read_csv(effect_dir / "pairs_train.csv")
    val_df = pd.read_csv(effect_dir / "pairs_val.csv")
    test_df = pd.read_csv(effect_dir / "pairs_test.csv")

    limit = cfg["gnn"]["limit_pairs"]
    if limit is not None:
        limit = int(limit)
        train_df = train_df.head(limit)

    smiles_csv = Path(cfg["smiles"]["smiles_cache_csv"])
    if not smiles_csv.exists():
        raise FileNotFoundError("smiles_map.csv not found. Run: python -m src.features.smiles_map")

    sm = pd.read_csv(smiles_csv).dropna(subset=["smiles"])
    sm["drug_id"] = sm["drug_id"].astype(str)
    smiles_map = dict(zip(sm["drug_id"], sm["smiles"]))

    train_ds = PairGraphDataset(train_df, smiles_map)
    val_ds = PairGraphDataset(val_df, smiles_map)
    test_ds = PairGraphDataset(test_df, smiles_map)

    bs = int(cfg["gnn"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PairClassifier(
        hidden_dim=int(cfg["gnn"]["hidden_dim"]),
        num_layers=int(cfg["gnn"]["num_layers"]),
        dropout=float(cfg["gnn"]["dropout"])
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["gnn"]["lr"]), weight_decay=float(cfg["gnn"]["weight_decay"]))
    loss_fn = torch.nn.BCEWithLogitsLoss()

    epochs = int(cfg["gnn"]["epochs"])
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for ba, bb, y in tqdm(train_loader, desc=f"Epoch {ep}/{epochs}"):
            ba = ba.to(device); bb = bb.to(device); y = y.to(device)
            opt.zero_grad()
            logits = model(ba, bb)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1

        val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": ep, "train_loss": total_loss / max(n_batches, 1), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(row)

    test_metrics = evaluate(model, test_loader, device)

    out = {
        "effect_dir": str(effect_dir),
        "train_size_used": len(train_ds),
        "val_size_used": len(val_ds),
        "test_size_used": len(test_ds),
        "history": history,
        "test": test_metrics
    }

    out_path = effect_dir / "gnn_metrics.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Saved:", out_path)
    print("Test:", test_metrics)

if __name__ == "__main__":
    main()
