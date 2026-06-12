# Code — Phase 1 (centralized baseline)

Centralized TB classifier on Shenzhen + Montgomery. This is the upper-bound baseline and
the **metric harness** ([metrics.py](metrics.py)) reused by all later federated experiments.

## Files
| File | Purpose |
|------|---------|
| [config.yaml](config.yaml) | All paths + hyperparameters (Colab/Drive paths by default) |
| [parse_demographics.py](parse_demographics.py) | Parse ClinicalReadings → `demographics.csv` (no GPU) |
| [data_loader.py](data_loader.py) | Dataset, transforms, stratified splits, class weights |
| [models.py](models.py) | DenseNet121 / ResNet50 factory + Grad-CAM target layer |
| [metrics.py](metrics.py) | `evaluate()` — overall + subgroup fairness metrics |
| [train.py](train.py) | Training loop, early stopping, test report |
| [gradcam.py](gradcam.py) | TB saliency sanity check |

## Run order (Colab)
```bash
pip install -r requirements.txt
python parse_demographics.py --config config.yaml   # 1. build demographics.csv
python train.py             --config config.yaml    # 2. train + evaluate
python gradcam.py --config config.yaml --checkpoint <out_dir>/best.pt   # 3. sanity check
```

See [../notebooks/phase1_colab.ipynb](../notebooks/phase1_colab.ipynb) for the Colab driver.

## Data expectations
`config.yaml` points to dataset roots that each contain a `ClinicalReadings/` folder and
the CXR images (the parser searches recursively). Labels come from the filename suffix
(`_0` normal, `_1` TB). Download instructions: [../data/README.md](../data/README.md).

## Next (Phase 2)
Swap the centralized loop for a Flower simulation (FedAvg → FedProx). `metrics.evaluate`
stays identical so results remain directly comparable.
