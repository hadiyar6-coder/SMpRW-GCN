# SMpRW-GCN: Structure-aware Multi-perspective Random Walks Graph Convolutional Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

Official PyTorch implementation of the paper: **"Structure-aware Multi-perspective Random Walks Graph Convolutional Networks for Adaptive Graph Embedding"**.

---

## 📌 Overview
Graph Convolutional Networks (GCNs) often suffer from performance degradation on heterophilic graphs due to their strict reliance on local neighborhood homophily. **SMpRW-GCN** decouples structural similarity modeling from representation learning through a two-stage architecture:
* **Stage $S_1$ (Offline Structural Graph Construction):** Extracts multi-perspective random walks (degree-aware, behavioral, and topological roles) and aligns degree sequences via Dynamic Time Warping (DTW) to construct a structural adjacency matrix $A'$.
* **Stage $S_2$ (Adaptive Multi-View Representation Learning):** Integrates ego features ($I$), local neighborhood ($A$), and structural topology ($A'$) via a dimension-wise trainable softmax gating mechanism across three architectural variants (`110`, `101`, and `011`), using a strict **Validation-Selected (`Val-Selected`)** protocol across 10 independent 60%/20%/20% splits.

---

## 📂 Repository & Directory Structure

Ensure your root project directory follows the structure below. 
> **Note on Compressed Folders:** If `data.rar` / `data.zip` or `output.rar` / `output.zip` are provided as compressed archives (to comply with GitHub's 100MB file limit), **extract them directly into the root directory** before running the code.

```text
SMpRW-GCN/
│
├── data/                        # Raw benchmark datasets
│   ├── 0/                       # Planetoid datasets (cora, citeseer, pubmed) + Feature/ cache
│   ├── 1/                       # Air-traffic datasets (Category/ & Edgelist/)
│   └── 2/                       # WebKB, Wikipedia & Actor datasets (chameleon, squirrel, actor, cornell, texas, wisconsin) + Feature/
│
├── output/                      # Cached Stage-S1 structural matrices (role_adj_*.pkl, weigh_list_*.pkl)
│   └── <dataset>/               # Subfolders per dataset (e.g., cora, chameleon, squirrel, etc.)
│
├── SMpRWGCN/
│   └── tvtsplit/                # Fixed 10 random Train/Val/Test (60/20/20) split indices
│       └── <dataset>/0.6/       # Contains <dataset>_tvt_index0.npy to <dataset>_tvt_index9.npy
│
├── result/                      # Detailed per-run execution logs and learned gate coefficients
│
├── main3.py                     # Main training, validation-based selection, and evaluation pipeline
├── model.py                     # Aggregator, Encoder (trainable softmax gates), and SMpRWGNN definitions
├── utils_data.py                # Dataset loaders, Stage-S1 caching, and 60/20/20 TVT split handler
├── utils.py                     # Seed initialization, graph partitioning, and evaluation metrics
├── alias.py                     # Alias table sampling for random walk generation
├── create_config.py             # Script to generate tab-separated hyperparameter configuration files
├── new test.txt                 # Generated tab-separated configuration file for batch experiments
├── end.txt                      # Final summary table (with ★ Val-Selected marker) and 10-run JSON logs
└── requirements.txt             # Required Python packages
```

---

## 🛠️ Installation & Requirements

1. Clone the repository and navigate to the project root:
```bash
git clone [https://github.com/YOUR_USERNAME/SMpRW-GCN.git](https://github.com/YOUR_USERNAME/SMpRW-GCN.git)
cd SMpRW-GCN
```

2. Install all required dependencies using `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## ⚠️ Crucial Step: Path Configuration Before Running

Before executing the pipeline on a new machine, you **must** update the root directory paths in **`utils_data.py`** and **`main3.py`** so the scripts can properly locate the `data/`, `output/`, `SMpRWGCN/tvtsplit/`, and `result/` directories:

### 1. Update Paths in `utils_data.py`
* **Line 17 (`BASE_PATH` for `data/` and `output/`):**
  Change the hardcoded path to your local project root directory (or set it dynamically):
  ```python
  # Replace 'C:/Users/hy/PycharmProjects/pythonProject6/bk3/SMpRW-GCN' with your path, or use:
  BASE_PATH = os.path.dirname(os.path.abspath(__file__))
  ```
* **Lines 188–190 (inside `def prepare_dataset(args, labels, i)` for `tvtsplit`):**
  Update `base_path` so it points to your local root folder containing `SMpRWGCN/tvtsplit/`:
  ```python
  base_path = os.path.dirname(os.path.abspath(__file__))
  split_dir = os.path.join(base_path, "SMpRWGCN", "tvtsplit", args.dataset, str(args.train_ra / 100))
  ```

### 2. Update Output Log Path in `main3.py`
* **Lines 141–145 (inside `if __name__ == "__main__":`):**
  Update the `result/` folder path to your local directory (or relative path):
  ```python
  os.makedirs("./result", exist_ok=True)
  path = os.path.join("./result", args.dataset + "_" + str(args_runs[k]['id']) + '.txt')
  ```

---

## ⚙️ Configuration File Setup (`create_config.py`)

The training pipeline reads experiment hyperparameters from a **tab-separated (`.txt`) configuration file** specified via `--config_path`. Each row defines a complete experiment with the following columns:

`id | N | train_ra | val_ra | dataset | feat_flag | backbone | normalize | aggr | alpha | beta | gama | lr | dr | h_dim | weight_decay | role_generate | feat_generate | seed | topk | force_regen`

* **Architectural Variants (`alpha`, `beta`, `gama`):**
  Exactly one coefficient must be `0` and the active pair must sum to `1.0` (enforced in `model.py`):
  * **`110` Variant (`gama = 0`):** Gates Self ($I$) and Local Neighborhood ($A$), concatenates Structural ($A'$).
  * **`101` Variant (`beta = 0`):** Gates Self ($I$) and Structural ($A'$), concatenates Local Neighborhood ($A$).
  * **`011` Variant (`alpha = 0`):** Gates Local Neighborhood ($A$) and Structural ($A'$), concatenates Self ($I$).
* **`feat_flag`:** `True` for *With Node Features*, `False` for *Featureless (Without Features)*.
* **`force_regen`:** Set to `False` to load precomputed Stage-$S_1$ structural matrices from `output/<dataset>/`, or `True` to regenerate them from scratch.

### Generating a Custom Config File
You can uncomment your desired dataset/variant combinations in `create_config.py` and run:
```bash
python create_config.py
```
This will generate **`new test.txt`** (or any custom config filename you specify).

---

## 🚀 Running Experiments (`main3.py`)

To run the training and evaluation pipeline, pass the exact name of your configuration file to `main3.py` using `--config_path`:

```bash
python main3.py --config_path "new test.txt"
```
*(Or if your default config path is set inside `param_parser.py`, simply run `python main3.py`)*.

### What Happens During Execution?
1. **Data & Structural Matrix Loading:** Loads the graph and node attributes from `data/` and fetches (or generates) the structural similarity graph $A'$ from `output/<dataset>/`.
2. **10-Split Evaluation (60/20/20):** Loads the exact pre-saved split indices (`<dataset>_tvt_index0.npy` to `9.npy`) from `SMpRWGCN/tvtsplit/<dataset>/0.6/`.
3. **Adaptive Gating & Early Stopping:** Trains `SMpRWGNN` for `N=10` runs, selects the best checkpoint per split based strictly on **Validation Accuracy**, logs the learned Layer-1 and Layer-2 softmax gate weights ($\alpha, \beta, \gamma$), and evaluates on the isolated Test split.
4. **Automated `Val-Selected` Summary (`end.txt`):**
   At the end of execution, `main3.py` automatically groups variants by dataset, identifies the variant with the **highest mean Validation Accuracy** (`★ = Best Validation Selected Model`), and saves the complete formatted summary table along with all 10 validation and test accuracy arrays (for paired $t$-test and Wilcoxon statistical analysis) into **`end.txt`** and individual run logs into **`result/`**.