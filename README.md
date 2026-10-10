# Hybrid Credit Card Fraud Detection using GraphSAGE + MLP

A final-year project that detects credit card fraud by combining **graph structure** (how transactions are linked through shared cards, addresses and emails) with **tabular transaction features**. A GraphSAGE network learns relationship-aware embeddings, which are fused with the original features and classified by an MLP.

---

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Dataset](#dataset)
4. [Architecture](#architecture)
5. [Project Structure](#project-structure)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Results](#results)
9. [Explainability](#explainability)
10. [Dashboard (Optional)](#dashboard-optional)
11. [Troubleshooting](#troubleshooting)
12. [Limitations and Future Work](#limitations-and-future-work)
13. [References](#references)

---

## Overview

Most fraud detectors score each transaction in isolation. Fraud, however, is often organised: the same card, billing address or email domain shows up across many suspicious transactions. This project models that by building a transaction graph and letting a Graph Neural Network learn from neighbouring transactions, while an MLP keeps the strength of classic tabular features.

**Pipeline:** Data loading → Cleaning → Feature engineering → Graph construction → GraphSAGE embeddings → Feature fusion → MLP classifier → Evaluation → Explainability

## Key Features

- **Hybrid architecture:** GraphSAGE embeddings (64-d) concatenated with original features, then classified by an MLP
- **Entity-based graph:** edges between transactions sharing `card1`, `card2`, `addr1` or `P_emaildomain`; entity groups limited to size < 1000 for memory; self-loops added
- **Class imbalance handling:** weighted cross-entropy (fraud weight ≈ 14.29×), stratified 70/15/15 split, evaluation with PR-AUC
- **Feature engineering:** cyclical time encoding, log/binned amounts, per-card aggregations and deviations, C/D feature aggregates
- **Explainability:** GNN-vs-feature contribution analysis, SHAP, graph connectivity analysis, t-SNE embedding plot
- **CPU-only:** no GPU required
- **Reproducible:** fixed random seed (42), saved checkpoints

## Dataset

**IEEE-CIS Fraud Detection** (Kaggle)

| Item | Value |
|---|---|
| Transactions | 590,540 |
| Original features | 434 (after merging transaction + identity tables) |
| Fraud rate | ~3.5% (about 27.6 : 1 imbalance) |
| Files used | `train_transaction.csv`, `train_identity.csv` |

Download from: https://www.kaggle.com/c/ieee-fraud-detection/data and place both CSV files in the project folder (or update the path in Section 2 of the script).

## Architecture

```
Transaction features ─┬──────────────────────────────┐
                      │                              │
                      ▼                              │
        GraphSAGE (3 layers, mean aggregation)       │
        in → 128 → 128 → 64                          │
        BatchNorm + ReLU + Dropout(0.3)              │
                      │                              │
                64-d embedding                       │
                      └──────── concatenate ◄────────┘
                                   │
                                   ▼
                    MLP: fused → 256 → 128 → 64 → 2
                                   │
                          Fraud / Non-fraud
```

| Setting | Value |
|---|---|
| Parameters | 319,874 |
| Loss | Weighted CrossEntropyLoss (fraud ≈ 14.29×) |
| Optimizer | Adam (lr = 0.001, weight decay = 1e-5) |
| Scheduler | ReduceLROnPlateau |
| Epochs | up to 50, early stopping (patience 10) |
| Split | 70% train / 15% validation / 15% test (stratified) |

## Project Structure

```
archive (1)/
├── fraud_detection_project.py     # Full pipeline: data → graph → train → evaluate → explain
├── fraud_dashboard_enhanced.py    # Optional Streamlit dashboard (demo/mock data by default)
├── train_transaction.csv          # Dataset (download separately)
├── train_identity.csv             # Dataset (download separately)
├── fraud_detection_venv/          # Virtual environment
│
│  Generated when you run the pipeline:
├── fraud_graph_data.pt            # Saved graph
├── best_model.pt                  # Best model checkpoint
├── evaluation_results.png         # Confusion matrix, ROC, PR curve, training history
└── tsne_embeddings.png            # Embedding visualisation
```

## Installation

**Requirements:** Windows, Python 3.13 (tested with 3.13.2), VS Code recommended.

```powershell
# 1. Go to the project folder
cd "C:\Users\yoges\Downloads\archive (1)"

# 2. Create and activate a virtual environment
python -m venv fraud_detection_venv
.\fraud_detection_venv\Scripts\Activate.ps1

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install PyTorch (CPU) and PyTorch Geometric
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric

# 5. Install the remaining packages
pip install pandas numpy scipy scikit-learn imbalanced-learn matplotlib seaborn shap networkx tqdm
pip install streamlit plotly        # only needed for the dashboard
```

Versions used during development: PyTorch 2.10.0 (CPU), PyTorch Geometric 2.7.0, Pandas 3.0.0, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, imbalanced-learn 0.14.1, Matplotlib 3.10.8, Seaborn 0.13.2, SHAP 0.50.0, NetworkX 3.6.1.

If PowerShell blocks activation, run once:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Usage

### Run the full pipeline
```powershell
python fraud_detection_project.py
```
or open the file in VS Code and press **F5**.

Approximate runtime on CPU: **30–60 minutes** (graph construction 10–20 min, training 15–30 min, SHAP a few minutes).

### What the script does
| Section | Description |
|---|---|
| 1 | Library imports, device and seed setup |
| 2 | Load and merge transaction + identity data |
| 3 | Cleaning: drop >90% missing columns, encode categoricals, impute, variance filter |
| 4 | Feature engineering |
| 5 | Graph construction and saving |
| 6 | Model definition, data split, class weights |
| 7 | Training loop with early stopping |
| 8 | Evaluation on the test set, plots |
| 9 | Explainability (contribution analysis, SHAP, connectivity, t-SNE) |
| 10 | Final summary printout |

## Results

Final numbers are produced by your own run. Copy them from the Section 8/10 output into this table.

| Metric | Value |
|---|---|
| ROC-AUC | _fill in after running_ |
| PR-AUC | _fill in after running_ |
| F1-score | _fill in after running_ |
| Precision | _fill in after running_ |
| Recall (fraud detection rate) | _fill in after running_ |
| False alarm rate | _fill in after running_ |

> **Why PR-AUC and recall matter:** with only ~3.5% fraud, plain accuracy is misleading (predicting "not fraud" every time already gives ~96.5%).

## Explainability

- **GNN vs feature contribution:** compares predictions using embeddings only, features only, and both
- **SHAP (KernelExplainer):** global and per-transaction feature importance on a sampled subset
- **Graph connectivity:** compares the average degree of fraud and non-fraud nodes
- **t-SNE:** 2-D view of the learned embeddings, coloured by label

Report the contribution percentages and connectivity ratio that **your run** produces rather than fixed numbers.

## Dashboard (Optional)

```powershell
streamlit run fraud_dashboard_enhanced.py
```

> **Note:** the dashboard currently shows **demo/mock values** (sample metrics, mock predictions and attention weights). It is a UI prototype. To make it real, load `best_model.pt` and `fraud_graph_data.pt` and replace the mock sections with model outputs.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: imblearn` | `pip install imbalanced-learn` |
| Venv won't activate in PowerShell | Set execution policy (see Installation) or use CMD: `fraud_detection_venv\Scripts\activate.bat` |
| `verbose` error in `ReduceLROnPlateau` | Remove `verbose=True` (not supported in recent PyTorch) |
| Unicode / encoding errors in Terminal | Avoid special characters in print statements |
| Out-of-memory during graph building | Lower the entity group limit or sample fewer transactions |
| SHAP is very slow | Reduce the number of samples and `nsamples` in Section 9 |
| Output paths like `/mnt/user-data/outputs/` fail on Windows | Change them to local paths, e.g. `evaluation_results.png` |

## Limitations and Future Work

**Limitations**
- Static graph (no temporal dynamics)
- Anonymised features limit interpretability
- Graph construction is expensive on very large data
- Single dataset, single time window

**Future work**
- Temporal / dynamic GNNs
- Heterogeneous graphs (separate node types for cards, devices, emails)
- Real-time inference service
- Comparison with XGBoost / Random Forest baselines and ablations (GNN-only vs MLP-only vs hybrid)
- Adversarial robustness and fairness auditing

## References

1. W. L. Hamilton, R. Ying, J. Leskovec, "Inductive representation learning on large graphs," NeurIPS, 2017.
2. T. N. Kipf, M. Welling, "Semi-supervised classification with graph convolutional networks," ICLR, 2017.
3. S. M. Lundberg, S.-I. Lee, "A unified approach to interpreting model predictions," NeurIPS, 2017.
4. M. Fey, J. E. Lenssen, "Fast graph representation learning with PyTorch Geometric," 2019.
5. A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," NeurIPS, 2019.
6. IEEE-CIS Fraud Detection, Kaggle, 2019.

Full reference list (55 citations) is in the project report.

## Author

Yogesh — Final-year project, Department of Computer Science / Data Science
Guide: _add name_ · Institution: _add name_ · Year: _add year_

## License

For academic use. The IEEE-CIS dataset is subject to Kaggle's competition terms.
