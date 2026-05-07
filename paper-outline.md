# Research Paper Outline

## Paper 1 (Core / First Paper)

**Title:** FedTB: A Privacy-Preserving Federated Learning Framework for Tuberculosis Detection on Heterogeneous Non-IID Data

**Target:** IEEE Journal of Biomedical and Health Informatics (JBHI) or Medical Image Analysis

---

### Abstract (structure)
- Problem: TB detection needs multi-site data; privacy prevents centralization
- Gap: Existing FL ignores non-IID distribution in medical imaging
- Method: FedTB — federated CNN with heterogeneity-aware aggregation
- Result: [X]% AUC on distributed TB datasets, outperforms FedAvg baseline
- Impact: Enables privacy-safe TB screening across institutions

---

### 1. Introduction
- TB burden globally (WHO stats)
- AI for chest X-ray: promise vs. data privacy barrier
- Federated learning: concept + why it fits healthcare
- Problem: non-IID data degrades FL performance
- Contributions (bullet list — 3–4 points)
- Paper structure

### 2. Related Work
- 2.1 Deep learning for TB / chest X-ray diagnosis
- 2.2 Federated learning in healthcare
- 2.3 Non-IID challenges in FL
- 2.4 Gap: no work addresses all three (FL + TB + non-IID fairness)

### 3. Datasets and Federated Setup
- 3.1 Datasets: Montgomery, Shenzhen, NIH, Kaggle
- 3.2 Simulating federated clients (how datasets map to clients)
- 3.3 Non-IID quantification (label distribution, quantity skew)
- 3.4 Data preprocessing and augmentation

### 4. Proposed Method: FedTB
- 4.1 Local model architecture (DenseNet-121 / ResNet-50)
- 4.2 Federated training protocol
- 4.3 Heterogeneity-aware aggregation strategy
  - FedProx proximal term
  - Client weighting by data quality/distribution
- 4.4 Communication efficiency
- 4.5 Privacy guarantees (differential privacy if applicable)

### 5. Experiments
- 5.1 Baselines: centralized, FedAvg, FedProx, SCAFFOLD
- 5.2 Evaluation metrics: AUC-ROC, F1, sensitivity, specificity
- 5.3 Main results table
- 5.4 Ablation: effect of heterogeneity degree
- 5.5 Convergence analysis
- 5.6 Communication cost analysis

### 6. Discussion
- Why proposed method outperforms
- Limitations: simulated federation, no real differential privacy
- Clinical relevance

### 7. Conclusion
- Summary of contributions
- Future: fairness, XAI, multimodal (tease next papers)

### References
- ~35–50 references (IEEE format)

---

## Paper 2

**Title:** Fairness-Aware Federated Learning for Equitable Tuberculosis Screening Across Demographics

**Target:** NeurIPS ML4H workshop → extended to FAccT or AAAI

---

### Key Sections
1. Introduction — demographic bias risk in medical AI
2. Fairness metrics in federated context (demographic parity, equalized odds, per-group F1)
3. Method: fairness-aware aggregation (weighted by demographic balance)
4. Experiments: per-gender, per-age-group, per-dataset performance
5. Fairness vs. accuracy trade-off analysis
6. Discussion: clinical safety implications

---

## Paper 3

**Title:** Federated Grad-CAM: Consistent Explainable AI for TB Detection Without Raw Data Access

**Target:** MICCAI (Medical Image Computing and Computer Assisted Intervention)

---

### Key Sections
1. Introduction — black-box problem in federated medical AI
2. Background: Grad-CAM, federated XAI challenges
3. Method: aggregating gradient maps across clients without sharing images
4. Clinician validation protocol
5. Consistency analysis: do explanations agree across clients?
6. Results: heatmap quality vs. centralized Grad-CAM baseline

---

## Paper 4 (Full System / Journal)

**Title:** MultiModal-FedTB: Integrating Chest X-Rays and Clinical Metadata in a Fair, Explainable Federated Learning System for Tuberculosis Detection

**Target:** Nature Digital Medicine or Cell Patterns

---

### Key Sections
1. Introduction — full problem statement, all 4 contributions
2. System architecture (FL + fairness + XAI + multimodal)
3. Multimodal fusion: early vs. late vs. hybrid
4. Comprehensive experiments
5. Ablation: each component's contribution
6. Limitations and future work
7. Conclusion

---

## General Writing Notes

- Each paper = one clear contribution (don't cram all into one)
- Always include: (1) problem, (2) gap, (3) method, (4) experiment, (5) result number
- Figures every paper needs:
  - System architecture diagram
  - Results comparison table
  - Convergence curves
  - (Paper 3) Grad-CAM heatmap examples
- Code + data release → reproducibility → stronger acceptance
