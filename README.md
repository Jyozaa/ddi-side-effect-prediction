# Drug-Drug Interaction Prediction

This project builds drug-drug interaction (DDI) prediction models from TwoSIDES-style side-effect data. It supports both simple binary experiments for a single side effect and multi-label experiments that predict many side effects for each drug pair.

The current pipeline uses drug-pair SMILES data, Morgan fingerprints, molecular graphs, and optional drug-target features. Models include logistic regression baselines and graph neural networks built with PyTorch Geometric.

## Project Structure

```text
.
├── configs/default.yaml          # Main configuration file
├── data/
│   ├── raw/                      # Source CSV files
│   └── processed/                # Generated datasets, caches, metrics, checkpoints
├── src/
│   ├── data/                     # Dataset builders and inspection scripts
│   ├── eval/                     # Thresholding and decoding evaluation utilities
│   ├── features/                 # SMILES and target feature preparation
│   ├── models/                   # Baseline and GNN model definitions
│   ├── train/                    # Training entry points
│   └── utils/                    # Config, IO, and seed helpers
├── cid_to_name.py                # PubChem CID name lookup utility
└── requirements.txt
```

## Data

The main configured input is:

```text
data/raw/smiles_pairs.csv
```

Expected columns:

| Column | Description |
| --- | --- |
| `ID1` | First drug identifier |
| `ID2` | Second drug identifier |
| `Y` | Side-effect identifier |
| `Side Effect Name` | Side-effect label |
| `X1` | SMILES string for the first drug |
| `X2` | SMILES string for the second drug |

Optional target-feature input:

```text
data/raw/drug_targets.csv
```

Expected columns:

| Column | Description |
| --- | --- |
| `drug_id` | Drug identifier matching `ID1` / `ID2` |
| `target_id` | Protein, gene, or target identifier |

The repository also contains `data/raw/TWOSIDES.csv` and `data/raw/TWOSIDES.csv.gz`. The included `data/raw/README.md` describes the original TwoSIDES column definitions.

## Setup

Create and activate a Python environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Most settings live in `configs/default.yaml`.

Important options include:

| Section | Key | Purpose |
| --- | --- | --- |
| `data` | `smiles_pairs_path` | Input CSV used by the SMILES-based pipelines |
| `data` | `output_dir` | Directory for generated datasets and results |
| `multilabel` | `top_k_effects` | Number of most common labels to keep; `0` keeps all labels above the minimum count |
| `multilabel` | `min_pos_per_label` | Minimum positive examples required per side-effect label |
| `split` | `type` | `random` or `cold_start` |
| `split` | `holdout_drug_frac` | Fraction of drugs reserved for cold-start testing |
| `gnn` | `epochs`, `batch_size`, `lr` | Main training hyperparameters |
| `gnn` | `limit_pairs` | Optional cap on training pairs for quick experiments |
| `targets` | `csv_path` | Target-feature CSV for overlap models |

The default GNN config is intentionally small (`epochs: 1`, `limit_pairs: 256`) so a smoke test can run quickly. Increase these values for real training runs.

## Multi-Label Workflow

Generate the SMILES cache:

```bash
python -m src.features.smiles_map
```

Build the multi-label train/validation/test splits:

```bash
python -m src.data.make_multilabel_dataset
```

This creates a directory like:

```text
data/processed/multilabel_top<N>_<split_type>/
```

Run the sparse Morgan fingerprint logistic regression baseline:

```bash
python -m src.models.baseline_multilabel_lr
```

Train the graph-plus-Morgan multi-label GNN:

```bash
python -m src.train.train_gnn_multilabel
```

Train the cardinality-aware multi-label GNN:

```bash
python -m src.train.train_gnn_multilabel_cardinality
```

Train the GNN variant that also uses target-overlap features:

```bash
python -m src.train.train_gnn_multilabel_overlap
```

The overlap model requires `data/raw/drug_targets.csv`.

## Evaluation Utilities

After training a multi-label GNN, use these scripts to evaluate alternate decoding strategies:

```bash
python -m src.eval.global_threshold_sweep
python -m src.eval.topk_decode_sweep
python -m src.eval.cardinality_decode_eval
```

Result JSON files are written under the corresponding `data/processed/...` run directory.

Note: `global_threshold_sweep.py` and `topk_decode_sweep.py` look for checkpoints under `gnn_multilabel_fusion/best.pt`. The current `train_gnn_multilabel.py` writes loss-specific directories such as `gnn_multilabel_fusion_bce/best.pt`, so adjust the checkpoint path in those evaluation scripts or copy the checkpoint into the expected directory before running them.

## Binary Workflow

For a single side-effect binary classification experiment, use the SMILES-pair dataset builder:

```bash
python -m src.data.make_pairs_from_smiles_csv
```

Then run either baseline logistic regression or a pairwise GNN:

```bash
python -m src.models.baseline_lr
python -m src.train.train_gnn
```

If `data.effect_name` is `null`, the dataset builder automatically picks the most frequent side effect. To choose a specific label, set `data.effect_name` in `configs/default.yaml`.

Note: the binary dataset builders are older than the current multi-label config layout. If you use them, make sure the config contains the split values they expect, or update the scripts to read the top-level `split` section used by the multi-label pipeline.

## Outputs

Common generated files include:

| File | Description |
| --- | --- |
| `pairs_train.csv` | Training split |
| `pairs_val.csv` | Validation split |
| `pairs_test.csv` | Test split |
| `meta.json` | Dataset metadata and label list |
| `baseline_multilabel_results.json` | Multi-label baseline metrics |
| `gnn_metrics.json` | Binary GNN metrics |
| `gnn_multilabel_fusion_*/best.pt` | Best multi-label GNN checkpoint |
| `gnn_multilabel_fusion_*/results.json` | Multi-label GNN metrics |

## Notes

- Always run scripts from the repository root.
- The project uses RDKit for SMILES parsing. Invalid or missing SMILES strings are skipped during feature generation.
- For cold-start splits, test pairs contain drugs selected from a held-out drug set.
- GPU, Apple MPS, or CPU will be selected automatically by the GNN training scripts when supported.
