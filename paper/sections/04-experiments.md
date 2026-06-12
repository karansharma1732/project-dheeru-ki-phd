# 4. Experimental Setup

## 4.1 Datasets
Shenzhen, Montgomery (fairness backbone); optionally TBX11K / Qatar subset as extra clients.
See [../../docs/datasets.md](../../docs/datasets.md). Preprocessing: resize to 224/256,
normalization, augmentation.

## 4.2 Federated configuration
Flower simulation; 5–10 clients; fractional GPU; rounds, local epochs, optimizer, LR.

## 4.3 Baselines
Centralized (upper bound), local-only (lower bound), FedAvg.

## 4.4 Protocol
≥3 seeds; mean ± 95% CI; leave-one-dataset-out external validation.

## 4.5 Implementation
PyTorch + Flower; AIF360 / Fairlearn for fairness metrics; pytorch-grad-cam for saliency.
Hardware (Colab/Kaggle GPU). Configs in [../../experiments/](../../experiments/).
