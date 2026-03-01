# TNBC Drug Cytotoxicity Prediction

Three-phase transfer learning pipeline for predicting drug cytotoxicity (LN_IC50) in Triple-Negative Breast Cancer (TNBC) cell lines using Graph Neural Networks.

## Method

A hybrid GIN + TransformerConv GNN encodes drug molecular graphs, while an MLP encodes cell line features (GSVA pathway scores, somatic mutations, proteomics). A bilinear attention layer fuses both representations, feeding into multi-task heads for IC50 regression, sensitivity classification, and reconstruction regularisation.

Training proceeds in three phases with cell-line-level splitting to prevent data leakage:

| Phase | Data | Purpose |
|-------|------|---------|
| 1 | Pan-cancer (GDSC2, all TNBC cells excluded) | Learn general drug–cell interactions |
| 2 | Breast cancer (all TNBC cells excluded) | Adapt to breast-specific patterns |
| 3 | TNBC only, LOCO CV (21 folds) | Specialise for TNBC; one cell line held out per fold |

Phases 1 and 2 are trained once. Phase 3 is retrained per fold with Leave-One-Cell-Line-Out (LOCO) cross-validation. Protein feature selection is done per fold on train/val cells only to avoid leakage.

## Repository Structure

```
├── tnbc.ipynb              # Training pipeline (run this first)
├── visualisations.ipynb    # Evaluation, baselines, and all figures
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                # Input data (see below)
│   └── processed/          # Cached drug graphs
├── results/
│   ├── models/tnbc/        # LOCO fold checkpoints + shared Phase 1/2 models
│   ├── viz/                # Prediction CSVs from visualisations.ipynb
│   └── *.csv / *.pdf/png   # Metrics and figures
├── data_splits/            # Saved split indices
├── prebatched_data/        # Pre-batched tensors for faster training
└── models/                 # Legacy directory
```

## Required Data

Place these in `data/raw/`:

| File | Description |
|------|-------------|
| `GDSC2 Fitted Dose Response Oct 27 2023.xlsx` | Drug response data (LN_IC50) |
| `DepMap Model Data.csv` | Cell line metadata (lineage, subtype) |
| `drugs_with_smiles.csv` | Drug names and SMILES strings |
| `cell_ge.csv` | GSVA pathway enrichment scores (pathways × cell lines) |
| `Omics Somatic Mutations.csv` | Somatic mutation calls per cell line |
| `Breast Cancer Proteomic Dynamics (2).csv` | Breast cancer proteomics |
| `ReactomePathways.gmt` | Reactome pathway gene sets |

## Setup

```bash
pip install -r requirements.txt
```

Requires PyTorch, PyTorch Geometric, torch-scatter, RDKit, scikit-learn, scipy, pandas, numpy, matplotlib.

## Usage

1. **Train the model** — run all cells in `tnbc.ipynb`. This produces:
   - Shared Phase 1/2 checkpoints (`results/models/tnbc/loco_shared/`)
   - Per-fold Phase 3 checkpoints (`results/models/tnbc/loco_fold{1..21}/`)
   - LOCO CV results pickle (`results/loco_cv_results_cell_line.pkl`)
   - Pre-batched data cache (`prebatched_data/tnbc/`)

2. **Generate figures and baselines** — run all cells in `visualisations.ipynb`. This produces:
   - GNN predictions per fold (DrEval-style evaluation)
   - Random Forest and ElasticNet LOCO baselines
   - All publication figures (PDF + PNG)
   - Statistical tests (Friedman + pairwise Wilcoxon)

## Configuration

Key parameters are set in `tnbc.ipynb` cell 2 (`cfg` dictionary):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `phase1_lr` | 1e-3 | Phase 1 learning rate |
| `phase2_lr` | 8e-5 | Phase 2 learning rate |
| `phase3_lr` | 1e-4 | Phase 3 learning rate |
| `phase{1,2,3}_epochs` | 100, 50, 50 | Max epochs per phase |
| `phase{1,2,3}_batch` | 256, 64, 64 | Batch sizes |

Device is auto-detected (CUDA → MPS → CPU).

## License

MIT
