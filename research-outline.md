# PhD Research Project Outline

## Title
Privacy-Preserving Federated Learning for Tuberculosis Detection with Fairness, Explainability, and Multimodal Integration

---

## Core Problem
Centralized TB detection models require pooling sensitive patient data — infeasible due to privacy laws (HIPAA, GDPR) and cross-institution data silos. Existing FL approaches ignore: (1) demographic bias, (2) model opacity, and (3) single-modality limitations.

---

## Research Objectives

| # | Objective | Method | Output |
|---|-----------|--------|--------|
| O1 | Build federated CNN for TB detection | FedAvg, FedProx, SCAFFOLD | Baseline FL model |
| O2 | Handle non-IID / heterogeneous data | Data distribution analysis, client weighting | Robust aggregation strategy |
| O3 | Ensure fairness across demographics | Demographic parity, equal opportunity metrics | Bias audit report + mitigation |
| O4 | Explain model decisions | Grad-CAM, SHAP/LIME for metadata | Interpretable prediction maps |
| O5 | Integrate multimodal inputs | Early/late fusion (X-ray + patient metadata) | Multimodal FL model |

---

## Research Questions

1. Can FL achieve competitive TB detection accuracy without centralizing patient data?
2. How do non-IID distributions affect convergence and fairness in federated TB models?
3. Do FL models exhibit demographic bias, and can it be mitigated without accuracy loss?
4. Can XAI techniques produce clinically meaningful explanations in a federated setting?
5. Does multimodal fusion improve diagnostic performance over image-only FL models?

---

## Datasets

| Dataset | Source | Modality | Notes |
|---------|--------|----------|-------|
| Montgomery County | NIH | Chest X-ray | 138 images, TB/normal |
| Shenzhen Hospital | NIH | Chest X-ray | 662 images, TB/normal |
| NIH Chest X-ray 14 | NIH | Chest X-ray | 112,000 images, 14 diseases |
| Kaggle TB Dataset | Kaggle | Chest X-ray + phone camera | Multi-source heterogeneity |
| Patient Metadata | Simulated/real | Age, gender, symptoms | For multimodal fusion |

Datasets used as **separate federated clients** to simulate real-world data silos.

---

## Methodology

### Phase 1 — FL Baseline (Months 1–6)
- Implement CNN (ResNet-50 / DenseNet-121) as local model
- Federated setup: 3–5 clients (datasets as clients)
- Algorithms: FedAvg → FedProx → SCAFFOLD
- Metrics: Accuracy, AUC-ROC, F1, communication rounds

### Phase 2 — Heterogeneity Handling (Months 4–9)
- Quantify non-IID degree (label skew, quantity skew)
- Strategies: FedProx proximal term, personalized FL (per-FedAvg)
- Ablation: performance vs. heterogeneity level

### Phase 3 — Fairness Analysis (Months 7–12)
- Compute per-demographic metrics: gender, age group, dataset origin
- Fairness metrics: demographic parity difference, equalized odds
- Mitigation: re-weighting, fairness-aware aggregation
- Compare: fairness vs. accuracy trade-off curves

### Phase 4 — XAI Integration (Months 10–15)
- Grad-CAM on CNN layers → heatmaps over X-ray regions
- SHAP for metadata feature importance
- Evaluate: clinician validation of explanation quality
- Consistency check: same explanation across clients?

### Phase 5 — Multimodal FL (Months 13–18)
- Fusion strategies: early (input concat), late (decision-level), hybrid
- Metadata: age, gender, symptom flags
- Compare unimodal vs. multimodal FL performance
- Privacy: ensure metadata protected same as images

### Phase 6 — Integration & Evaluation (Months 16–24)
- Full system: FL + fairness + XAI + multimodal
- Comprehensive benchmark against centralized baselines
- Ablation study: each component's contribution
- Thesis writing + paper submissions

---

## Technical Stack

- **Framework:** PyTorch + PySyft / Flower (FL simulation)
- **Models:** ResNet-50, DenseNet-121, MobileNetV2
- **XAI:** pytorch-grad-cam, SHAP
- **Fairness:** Fairlearn, AIF360
- **Experiment tracking:** MLflow / Weights & Biases

---

## Expected Contributions

1. **Novel FL framework** handling non-IID TB data across distributed clients
2. **Fairness-aware FL aggregation** for demographically balanced TB screening
3. **XAI pipeline for federated medical imaging** — consistent explanations without raw data access
4. **Multimodal FL architecture** combining X-ray + metadata without privacy leakage
5. **Benchmark suite** for federated TB detection with heterogeneity and fairness metrics

---

## Publication Plan

| Paper | Target Venue | Timeline |
|-------|-------------|----------|
| P1: FL baseline + non-IID handling | IEEE JBHI / Medical Image Analysis | Month 9 |
| P2: Fairness in federated medical AI | NeurIPS / FAccT | Month 14 |
| P3: XAI for federated TB detection | MICCAI | Month 17 |
| P4: Full multimodal FL system | Nature Digital Medicine / Patterns | Month 22 |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Scope too broad | Prioritize: FL baseline → fairness → XAI → multimodal (in order) |
| Small dataset size | Data augmentation, transfer learning from ImageNet |
| No real multi-institution data | Simulate clients using existing public datasets |
| XAI inconsistency across clients | Aggregate explanations; use consensus metrics |
