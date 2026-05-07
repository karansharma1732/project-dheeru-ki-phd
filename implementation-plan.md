# Implementation Plan — FedTB Phase 1

## Step 1: VS Code Setup

### 1.1 Install Tools
- Python 3.10+ → https://python.org
- VS Code → https://code.visualstudio.com
- Git → https://git-scm.com

### 1.2 VS Code Extensions to Install
- Python (Microsoft)
- Pylance
- Jupyter
- GitLens
- Rainbow CSV (for metadata files)

---

## Step 2: Project Folder Structure

```
fedtb/
├── data/
│   ├── raw/                  # downloaded datasets
│   │   ├── montgomery/
│   │   ├── shenzhen/
│   │   └── kaggle_tb/
│   └── processed/            # resized, normalized images
├── clients/                  # one folder per FL client
│   ├── client_1/             # Montgomery data
│   ├── client_2/             # Shenzhen data
│   └── client_3/             # Kaggle data
├── models/
│   ├── cnn.py                # local model definition
│   └── checkpoints/          # saved weights
├── fl/
│   ├── server.py             # aggregation logic
│   ├── client.py             # local training logic
│   └── aggregators.py        # FedAvg, FedProx
├── xai/
│   └── gradcam.py            # Grad-CAM logic (Phase 4)
├── fairness/
│   └── metrics.py            # bias metrics (Phase 3)
├── experiments/
│   └── run_fedavg.py         # main experiment script
├── notebooks/
│   └── eda.ipynb             # data exploration
├── results/
│   └── logs/                 # training metrics, plots
├── requirements.txt
└── README.md
```

---

## Step 3: Python Environment

```bash
# In VS Code terminal (Ctrl + `)
python -m venv venv
venv\Scripts\activate          # Windows

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install flwr                # Flower — FL framework
pip install scikit-learn pandas numpy matplotlib seaborn
pip install grad-cam            # pytorch-grad-cam
pip install shap fairlearn
pip install mlflow              # experiment tracking
pip install kaggle              # dataset download
pip install Pillow tqdm
pip freeze > requirements.txt
```

---

## Step 4: Download Datasets

### 4.1 Kaggle Setup
```bash
# Put kaggle.json API key in C:\Users\<you>\.kaggle\kaggle.json
kaggle datasets download -d jtiptj/chest-xray-pneumoniacovid19tuberculosis -p data/raw/kaggle_tb --unzip
kaggle datasets download -d saife245/tuberculosis-image-datasets -p data/raw/kaggle_phone --unzip
```

### 4.2 Montgomery + Shenzhen (NIH)
- Download from: https://lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayImageDataSets.html
- Place in `data/raw/montgomery/` and `data/raw/shenzhen/`

---

## Step 5: Data Preprocessing Script

**File:** `data/preprocess.py`

```python
# Resize all images to 224x224
# Normalize using ImageNet mean/std
# Split each dataset into train/val/test (70/15/15)
# Save processed images + labels as CSV per client
```

Output per client:
```
clients/client_1/train.csv   # columns: image_path, label (0=normal, 1=TB)
clients/client_1/val.csv
clients/client_1/test.csv
```

---

## Step 6: Local Model (CNN)

**File:** `models/cnn.py`

- Base: DenseNet-121 pretrained on ImageNet
- Replace final FC layer: 2 classes (TB / Normal)
- Loss: Binary Cross-Entropy
- Optimizer: Adam (lr=1e-4)

---

## Step 7: FL Training Loop (Flower Framework)

### 7.1 Client — `fl/client.py`
- Load local dataset
- Train local model for N epochs
- Return updated weights to server

### 7.2 Server — `fl/server.py`
- Coordinate clients
- Aggregate weights (FedAvg first)
- Send global model back
- Repeat for R rounds

### 7.3 Run experiment
```bash
# Terminal 1 — start server
python fl/server.py

# Terminal 2, 3, 4 — start clients
python fl/client.py --client_id 1
python fl/client.py --client_id 2
python fl/client.py --client_id 3
```

Or simulate all in one script:
```bash
python experiments/run_fedavg.py
```

---

## Step 8: Metrics to Track

| Metric | Tool |
|--------|------|
| AUC-ROC, F1, Accuracy | scikit-learn |
| Per-round loss/acc | MLflow |
| Confusion matrix | matplotlib |
| Communication rounds | logged in server |

---

## Step 9: Experiment Order

| # | Experiment | Purpose |
|---|-----------|---------|
| E1 | Centralized baseline | Upper bound benchmark |
| E2 | FedAvg (IID split) | Sanity check FL works |
| E3 | FedAvg (real non-IID split) | See performance drop |
| E4 | FedProx | Fix non-IID degradation |
| E5 | SCAFFOLD | Compare to FedProx |
| E6 | Ablation: rounds vs accuracy | Communication cost analysis |

---

## Step 10: First Week Checklist

- [ ] VS Code + Python env working
- [ ] Datasets downloaded and verified
- [ ] `eda.ipynb` — look at class distribution per dataset
- [ ] `preprocess.py` — images resized, CSVs created
- [ ] `models/cnn.py` — DenseNet-121 loads without error
- [ ] Centralized training runs (E1)
- [ ] FedAvg simulation runs (E2)
- [ ] MLflow dashboard shows training curves

---

## Immediate Next Action

1. Create `fedtb/` folder in VS Code
2. Open terminal → `python -m venv venv` → activate
3. `pip install torch flwr scikit-learn pandas matplotlib`
4. Download one dataset (Kaggle TB) to verify setup
5. Run `eda.ipynb` to explore data before any training
