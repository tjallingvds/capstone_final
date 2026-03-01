# GARNET — TNBC Drug Response Prediction

Three-phase transfer learning pipeline for predicting drug response (LN IC50) in Triple-Negative Breast Cancer (TNBC) cell lines using a Graph Attention Network.

## Method

A hybrid GIN + TransformerConv GNN encodes drug molecular graphs, while a residual feedforward network encodes cell-line features (GSVA pathway scores, somatic mutations, proteomics). Scaled dot-product cross-attention fuses both representations, feeding into multi-task heads for IC50 regression, sensitivity classification, and reconstruction regularisation.

Training proceeds in three phases with cell-line-level splitting to prevent data leakage:

| Phase | Data | Purpose |
|-------|------|---------|
| 1 | Pan-cancer (GDSC2, all TNBC cells excluded) | Learn general drug–cell interactions |
| 2 | Breast cancer (all TNBC cells excluded) | Adapt to breast-specific patterns |
| 3 | TNBC only, LOCO CV (21 folds) | Specialise for TNBC; one cell line held out per fold |

Phases 1 and 2 are trained once with the drug encoder frozen in Phase 2. Phase 3 is retrained per fold with Leave-One-Cell-Line-Out (LOCO) cross-validation. Protein feature selection is done per fold on train/val cells only to avoid leakage.

## Repository Structure

```
├── tnbc.ipynb              # Training pipeline (run this first)
├── visualisations.ipynb    # Evaluation, baselines, and all figures
├── requirements.txt
├── README.md
├── LICENSE
├── data/
│   ├── raw/                # Input data (see Data Acquisition below)
│   └── processed/          # Cached drug molecular graphs
├── results/
│   ├── models/tnbc/
│   │   ├── loco_shared/    # Shared Phase 1 & 2 checkpoints
│   │   └── loco_fold{1..21}/  # Per-fold Phase 3 checkpoints
│   ├── viz/                # GNN predictions and DrEval metrics CSVs
│   ├── *.csv               # Baseline metrics and per-drug Spearman scores
│   └── *.pdf / *.png       # Publication figures
├── prebatched_data/tnbc/   # Pre-batched tensors (Phase 1–3, per fold)
└── results/loco_cv_results_cell_line.pkl  # Aggregated LOCO CV results
```

## Data Acquisition

All raw data files should be placed in `data/raw/`. The pipeline integrates five primary sources from the Genomics of Drug Sensitivity in Cancer 2 (GDSC2) and Cancer Dependency Map (DepMap) projects.

| File | Source | How to obtain |
|------|--------|---------------|
| `GDSC2 Fitted Dose Response Oct 27 2023.xlsx` | GDSC2 (Yang et al., 2012) | Download from [cancerrxgene.org/downloads/bulk_download](https://www.cancerrxgene.org/downloads/bulk_download) under *Dose Response Data* → *GDSC2*. Only pairs with RMSE < 0.3 are used. |
| `drugs_with_smiles.csv` | GDSC2 | Download the compound list with SMILES from the same GDSC bulk download page. Compounds with invalid SMILES are excluded automatically. |
| `DepMap Model Data.csv` | DepMap (Broad, 2023) | Download `Model.csv` from [depmap.org/portal/data_page](https://depmap.org/portal/data_page/?tab=allData) (DepMap Public 23Q4). Provides lineage and subtype annotations used to define pan-cancer, breast, and TNBC subsets. |
| `Omics Somatic Mutations.csv` | DepMap (Broad, 2023) | Download `OmicsSomaticMutations.csv` from the same DepMap data page. Binary mutation status is extracted for 18 breast cancer driver genes. |
| `cell_ge.csv` | MSigDB + GSVA | Computed by running Gene Set Variation Analysis (Hänzelmann et al., 2013) in R on cell line expression data across 1,707 MSigDB pathways (Subramanian et al., 2005). The resulting cell-line × pathway matrix should be saved as `cell_ge.csv`. |
| `Breast Cancer Proteomic Dynamics (2).csv` | Sun et al. (2023) | Download from the supplementary data of [Sun et al. (2023)](https://doi.org/10.1016/j.mcpro.2023.100602). Contains mass spectrometry proteomics for breast cancer cell lines. |
| `ReactomePathways.gmt` | MSigDB / Reactome | Download from [gsea-msigdb.org/gsea/msigdb](https://www.gsea-msigdb.org/gsea/msigdb/) under *Reactome* gene sets in GMT format. |

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
   - GNN predictions per fold (DrEval-style residualized evaluation)
   - Random Forest and ElasticNet LOCO baselines
   - All publication figures (PDF + PNG)
   - Statistical tests (Friedman + pairwise Wilcoxon)

## Configuration

Training hyperparameters are hardcoded inline in `tnbc.ipynb`:

| Hyperparameter | Phase 1 (Pan-cancer) | Phase 2 (Breast) | Phase 3 (TNBC) |
|----------------|----------------------|-------------------|----------------|
| Learning rate | 1e-3 | 8e-5 | 1e-4 |
| Max epochs | 100 | 50 | 50 |
| Batch size | 256 | 64 | 64 |
| Drug encoder | Trainable | Frozen | Frozen |

Device is auto-detected (CUDA → MPS → CPU).

## License

MIT
