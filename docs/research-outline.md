# Research Outline & Timeline

Phased build order from [implementation-plan.md](implementation-plan.md) §8, with
deliverables and decision gates.

## Phase 1 — Centralized baseline (months 1–2)
- Centralized DenseNet121 TB classifier on Shenzhen + Montgomery.
- Parse & harmonize demographics from `ClinicalReadings` .txt files.
- Lock the metric suite; integrate Grad-CAM.
- **Deliverable:** reproducible centralized baseline + metric harness.

## Phase 2 — Federated heterogeneity core (months 2–4)
- Flower simulation; FedAvg over source-based and Dirichlet partitions.
- Add FedProx, then SCAFFOLD; α sweep {0.1, 0.5, 1.0, ∞}.
- **Deliverable:** Objective 1 heterogeneity results.

## Phase 3 — Fairness (months 4–6)
- Add FairFed + q-FFL; full subgroup evaluation by sex/age.
- **Deliverable:** Objective 1 complete → **first paper**.

## Phase 4 — Fusion module (months 6–9)
- Tabular MLP + CXR late fusion (TB Portals if DUA granted; else simulated clinical proxy).
- **Deliverable:** fusion baseline.

## Phase 5 — Modality-heterogeneous FL (months 9–14)
- CXR clients + clinical-tabular clients + sputum bacilli-detection client.
- Implement ClusMFL and/or feature-imputation; sweep modality-availability levels.
- **Deliverable:** Objective 2 → **second/third paper**.

## Decision gates
- **TB Portals DUA:** apply now. If denied/delayed past **month 8**, fall back to a fully
  open CXR(+demographics) + simulated-clinical-tabular + sputum-detection modality-incomplete
  benchmark (still novel/publishable).
- **Compute:** downsample CXR to 224/256, subsample TBX11K, fractional-GPU clients. Avoid
  FedPIA-scale models unless more compute is secured.

## Feasibility verdicts
- Objective 1 — **high feasibility**.
- Objective 2 — **feasible only in modality-heterogeneous framing**; do not promise
  per-patient tri-modal fusion.
- Sputum — **feasible as a separate detection modality-client**.

## Target venues (to refine)
- Paper 1 (fairness + heterogeneity FL for TB): medical-imaging / ML-for-health venue.
- Paper 2 (modality-incomplete TB benchmark): FL or multimodal medical-AI venue.
