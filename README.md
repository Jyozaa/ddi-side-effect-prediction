# DDI Side-Effect Prediction

This repository contains the code for the final-year project:

**Graph Neural Network Fusion for Multi-label Drug-Drug Interaction Side-Effect Prediction**

The project studies multi-label prediction of adverse side effects arising from drug-drug interactions (DDIs) using molecularly derived drug representations. It includes:

- a Morgan fingerprint one-vs-rest logistic baseline
- a graph-only GNN
- a graph-fingerprint fusion GNN
- improved graph readout and cross-drug interaction refinement
- exploratory variants including target-based features, asymmetric loss (ASL), and alternative decoding strategies

The main reported experiments use **multi-label datasets** derived from a TwoSIDES-style drug-pair CSV with SMILES strings.

## Repository Structure

```text
.
├── configs/
│   └── default.yaml                 # Main configuration file
├── data/
│   ├── raw/                         # Input CSV files
│   └── processed/                   # Generated datasets, caches, checkpoints, metrics
├── src/
│   ├── data/                        # Dataset builders and inspection scripts
│   ├── eval/                        # Thresholding / decoding evaluation utilities
│   ├── features/                    # SMILES and target feature preparation
│   ├── models/                      # Baseline and GNN model definitions
│   ├── train/                       # Training entry points
│   └── utils/                       # Config, IO, and seed helpers
├── cid_to_name.py                   # PubChem CID name lookup utility
├── requirements.txt
└── README.md
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

The full `data/` folder is not stored directly in GitHub because it is too large for normal repository storage. Download the compressed data archive from Google Drive:

**Data archive:**  
https://drive.google.com/file/d/1dg7CLVzi9XxiUQ2TeM7muRdfv6xxSGMb/view?usp=sharing

Original source datasets are also available from:

- **TwoSIDES:** https://tatonettilab-resources.s3.amazonaws.com/nsides/TWOSIDES.csv.gz
- **DGIdb:** download `interactions.tsv`, `genes.tsv`, and `drugs.tsv` from https://dgidb.org/downloads

After downloading, unzip the archive into the project root so the layout is:

```text
data/
├── raw/
└── processed/
```

## Environment Setup

Create and activate a Python environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Always run commands from the repository root.

## Main Configuration

Most experiment settings are controlled through:

```text
configs/default.yaml
```

Important configuration groups include:

| Section | Key | Purpose |
| --- | --- | --- |
| `data` | `smiles_pairs_path` | Main SMILES-pair CSV input |
| `data` | `output_dir` | Base directory for generated outputs |
| `multilabel` | `top_k_effects` | Number of most frequent labels to keep; `0` keeps all labels above the minimum count |
| `multilabel` | `min_pos_per_label` | Minimum positive count required per label |
| `multilabel` | `max_pairs` | Optional cap on the constructed multi-label dataset |
| `split` | `type` | `random` or `cold_start` |
| `split` | `holdout_drug_frac` | Fraction of drugs reserved for cold-start testing |
| `baseline` | `morgan_radius`, `morgan_nbits` | Fingerprint settings for the logistic baseline |
| `fusion` | `morgan_radius`, `morgan_nbits` | Fingerprint settings for the neural fusion branch |
| `gnn` | `hidden_dim`, `num_layers`, `dropout` | Core GNN architecture settings |
| `gnn` | `batch_size`, `lr`, `weight_decay`, `epochs` | Training hyperparameters |
| `gnn` | `loss_name` | `bce` or `asl` |
| `gnn` | `asl_gamma_pos`, `asl_gamma_neg`, `asl_clip` | ASL loss settings |
| `targets` | `csv_path` | Target-feature CSV for overlap / target-based variants |

## Recommended Reproduction Path

This is the main path for reproducing the dissertation-style multi-label workflow.

### 1. Build the SMILES cache

```bash
python -m src.features.smiles_map
```

### 2. Build the multi-label dataset

```bash
python -m src.data.make_multilabel_dataset
```

This creates a processed directory such as:

```text
data/processed/multilabel_top20_cold_start/
data/processed/multilabel_top1211_random/
data/processed/multilabel_top1211_cold_start/
```

The exact directory name depends on the configuration.

### 3. Run the multi-label baseline

```bash
python -m src.models.baseline_multilabel_lr
```

### 4. Train the main multi-label GNN

```bash
python -m src.train.train_gnn_multilabel
```

This is the primary training entry point used for the retained graph-only / fusion-based multi-label experiments.

## Switching Between Top-20 and All-Eligible Labels

The label regime is controlled in `configs/default.yaml` through the `multilabel` section:

- **Focused benchmark:** set `top_k_effects` to a positive value such as `20`
- **Large-label benchmark:** set `top_k_effects: 0` and keep an appropriate `min_pos_per_label`

Example:

```yaml
multilabel:
  top_k_effects: 20
  min_pos_per_label: 100
```

or:

```yaml
multilabel:
  top_k_effects: 0
  min_pos_per_label: 100
```

## Switching Between BCE and ASL

The main trainer supports both BCE and asymmetric loss through configuration:

```yaml
gnn:
  loss_name: "bce"
```

or:

```yaml
gnn:
  loss_name: "asl"
  asl_gamma_pos: 1.0
  asl_gamma_neg: 4.0
  asl_clip: 0.05
```

This allows the same training pipeline to be reused for loss comparisons.

## Additional Training Variants

The repository also includes exploratory multi-label training scripts.

### Cardinality-aware variant

```bash
python -m src.train.train_gnn_multilabel_cardinality
```

### Target-overlap / target-feature variant

```bash
python -m src.train.train_gnn_multilabel_overlap
```

This requires:

```text
data/raw/drug_targets.csv
```

These variants were exploratory and are not the main retained pipeline.

## Evaluation Utilities

After training a multi-label model, the repository includes utilities for alternate decoding and thresholding experiments:

```bash
python -m src.eval.global_threshold_sweep
python -m src.eval.topk_decode_sweep
python -m src.eval.cardinality_decode_eval
```

These scripts write result JSON files under the corresponding processed run directory.

If custom checkpoint naming or loss-specific output folders are used, check the expected checkpoint path inside the evaluation script before running it.

## Optional Binary Workflow

The repository also contains older binary single-effect scripts. These are optional and were not the main retained path in the final report.

Build a binary dataset for a single effect:

```bash
python -m src.data.make_pairs_from_smiles_csv
```

Then run either:

```bash
python -m src.models.baseline_lr
python -m src.train.train_gnn
```

If `data.effect_name` is `null`, the dataset builder selects the most frequent side effect automatically.

## Common Outputs

Typical generated files include:

| File | Description |
| --- | --- |
| `pairs_train.csv` | Training split |
| `pairs_val.csv` | Validation split |
| `pairs_test.csv` | Test split |
| `meta.json` | Dataset metadata and label list |
| `baseline_multilabel_results.json` | Multi-label baseline metrics |
| `best.pt` | Best saved checkpoint for a run |
| `results.json` | Saved run metrics |
| `smiles_map.csv` | Cached drug-to-SMILES mapping |

## Notes

- Invalid or missing SMILES strings are skipped during feature generation.
- For cold-start splits, test pairs contain drugs drawn from a held-out drug set.
- Device selection is automatic where supported: GPU, Apple MPS, or CPU.
- The main code path for the dissertation is the **multi-label workflow**, not the older binary scripts.
