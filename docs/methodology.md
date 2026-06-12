# Methodology

Condensed methods reference. Full rationale and citations are in
[implementation-plan.md](implementation-plan.md).

## 1. Non-IID / heterogeneity simulation
- **Source-based partitioning** (most defensible): each public dataset = one client/region
  → realistic covariate/feature shift.
- **Dirichlet label-skew** (Hsu et al. 2019): class proportions `q ~ Dir(α)`,
  with **α ∈ {0.1, 0.5, 1.0, ∞}** (severe → IID). Reuse NIID-Bench partitioning.
- **Client counts:** 5–10 virtual clients on a single Colab/Kaggle GPU (fractional GPU per
  client in Flower).

## 2. Heterogeneity-robust FL
| Method | Idea | Priority |
|--------|------|----------|
| **FedAvg** | weighted parameter averaging | baseline |
| **FedProx** | proximal term `μ/2‖w−wᵗ‖²`; tune μ ∈ {0.001,0.01,0.1,0.5,1} | **primary** |
| **SCAFFOLD** | control variates correct client drift; ~2× comms | comparison |
| **Clustered / personalized** (pFedMe, FedRep, Per-FedAvg) | per-client models | stretch |

## 3. Fairness methods & metrics
- **FairFed** (Ezzeldin et al. 2023) — server-side, group-fairness-aware aggregation
  (e.g., sex). **Primary fairness algorithm**; benefit grows under heterogeneity.
- **q-FFL / q-FedAvg** (Li et al. 2020) — performance/client fairness (uniform accuracy).
- **Local reweighting / oversampling** — cheap client-side complement.
- **Metrics:** subgroup sensitivity/specificity/AUC gaps (sex, age band), equal opportunity
  difference, equalized odds difference, demographic parity difference, worst-group &
  worst-client performance, Jain's fairness index. Compute with **AIF360 / Fairlearn**.

## 4. Models & multimodal fusion
- **CXR encoder:** DenseNet121 or ResNet (downsample to 224/256).
- **Tabular encoder:** MLP.
- **Fusion:** late (logit) fusion preferred — most robust to missing modalities;
  intermediate (feature-concat) as comparison.
- **Modality-incomplete FL (Objective 2):**
  - **ClusMFL** — FINCH clustering + supervised-contrastive alignment + modality-aware
    aggregation. Strongest template.
  - **Feature Imputation Network** (Poudel et al. 2025) — lightweight bottleneck-feature
    reconstruction; Colab-friendly.
  - **FedPIA** — adapter permutation via Wasserstein barycenters; heavy, optional.
  - **Modality-dropout** — simple strong baseline.
- **Sputum stream:** bacilli object detection (YOLO / Faster R-CNN) as a modality-client.

## 5. Tooling
- **FL framework:** Flower (`flwr`) simulation mode (Ray Virtual Client Engine).
- **Fairness:** AIF360, Fairlearn.
- **XAI:** pytorch-grad-cam (CXR lesion saliency, bacilli sanity check).
- **DP (optional):** Opacus — only with an explicit privacy claim.

## 6. Experimental design
- **Baselines:** centralized (upper bound), local-only (lower bound), FedAvg.
- **Proposed:** FedProx/SCAFFOLD × FairFed/q-FFL (Obj. 1); ClusMFL / feature-imputation
  (Obj. 2).
- **Metrics:** accuracy, sensitivity, specificity, AUC, F1 + fairness metrics +
  communication cost (rounds to target accuracy, MB/round).
- **Ablations:** α sweep; with/without fairness module; modality-availability ratios;
  image-encoder choice.
- **Rigor:** ≥3 seeds; mean ± 95% CI; **leave-one-dataset-out external validation**.
