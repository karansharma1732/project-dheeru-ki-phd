# Implementation Guide: Heterogeneity-Aware & Fairness-Aware Federated Learning for TB Detection (Two PhD Objectives)

## TL;DR
- **Objective 1 (heterogeneity- + fairness-aware FL on chest X-rays) is fully feasible** on Colab/Kaggle-tier compute using public TB CXR datasets partitioned across virtual clients in Flower. Build it first — it is the backbone pipeline and yields a publishable paper within ~6 months.
- **Objective 2 as originally framed (per-patient CXR + clinical + sputum fusion) is essentially impossible with public data** — no public dataset pairs all three modalities for the same patients, and the sputum images are a different cohort. Reframe Objective 2 as **modality-heterogeneous / missing-modality federated learning**, where different FL clients hold different TB evidence types. This is genuine, publishable novelty for TB.
- The ~1000 sputum images with XML are a real asset, but they belong to a **separate cohort** from any public CXR set. Use them as a distinct **bacilli-detection modality-client**, NOT as a fused per-patient feature.

## Key Findings
- **Montgomery (138 CXR) and Shenzhen (662 CXR) are the only major public TB CXR sets that carry per-image age/sex** (in their `ClinicalReadings` `.txt` files). They are therefore the indispensable fairness backbone. Other large TB CXR sets (TBX11K, Qatar/Dhaka) do not release per-image demographics.
- **No freely downloadable public TB dataset pairs CXR + clinical/tabular for the same patients.** CODA TB has cough+clinical but no CXR; TB Portals has imaging+clinical+genomic but is gated behind a Data Use Agreement and is drug-resistance-focused. This single fact forces the reframing of Objective 2.
- **The modality-incomplete / missing-modality federated learning frontier (2024–2025) is well-defined** (ClusMFL, FedPIA, feature-imputation networks) but has **not** been applied to the CXR + clinical + sputum TB triad. That gap is your central novelty.

## Details

### 1. DATASETS

**(a) Public TB CXR datasets**

- **Montgomery County (MC):** 138 posteroanterior CXRs — 80 normal, 58 TB. 12-bit PNG, ~4020×4892 px, plus left/right lung masks. Each image has a `ClinicalReadings` `.txt` file containing **patient age, sex, and abnormality**. Collected from the Montgomery County, Maryland TB screening program (NLM). *Best fairness substrate alongside Shenzhen.* (Jaeger et al., *Quant Imaging Med Surg* 2014; PMC4256233)
- **Shenzhen (SH):** 662 frontal CXRs — 326 normal, 336 TB-manifesting. PNG ~3000×3000 px. Each image has a `ClinicalReadings` `.txt` file with **age, sex, and findings**. From Shenzhen No. 3 People's Hospital, China. Before TBX11K, this was "the existing largest public TB dataset" with image-level annotations. (PMC4256233; Liu et al. CVPR 2020)
- **TBX11K:** 11,200 CXRs at 512×512 with **bounding-box annotations for TB regions**. Five categories: healthy (5,000), sick-but-non-TB (5,000), and active/latent/uncertain TB (1,200); train/val/test splits of 6,600 / 1,800 / 2,800. Per Liu et al. (CVPR 2020; mmcheng.net/tb), TBX11K itself "combined data from four smaller TB datasets: DA, DB, Montgomery, and Shenzhen … 156, 150, 138, and 662 X-ray images, respectively." Largest TB CXR set, but **no per-image demographics are released**, so it cannot anchor fairness work. (arXiv 2307.02848)
- **Kaggle TB Chest Radiography Database (Qatar/Dhaka, Rahman et al.):** current release 3,500 normal + 3,500 TB (earlier release 700 TB / 3,500 normal), PNG, 512×512. Aggregated from NLM, Belarus, NIAID, and RSNA sources. **No demographics.** (IEEE Access, DOI 10.1109/ACCESS.2020.3031384)
- **VinDr-CXR:** 18,000 CXRs (15,000 train / 3,000 test), DICOM, radiologist annotations for 28 findings; **patient age/sex are in the DICOM tags**. Not TB-specific, but valuable as a non-TB / domain-shift client and for demographic analysis. (PhysioNet; arXiv 2012.15029)
- **IN-CXR (ICMR-NIRT, India):** Adult CXRs from India's National TB Prevalence Survey (2019–21), grouped normal/abnormal. Access via request to NIRT (nirt.res.in). Use as an India-specific client for leave-one-dataset-out external validation.

**(b) Paired CXR + clinical/tabular**

- **CODA TB DREAM Challenge:** "733,756 cough sounds from 2,143 patients across 7 countries" — India, Madagascar, the Philippines, South Africa, Tanzania, Uganda, and Vietnam — with demographic, clinical, and microbiologic data (PMC11489852). But it is **cough + clinical, not CXR**, so it cannot serve CXR+clinical fusion. Notably, its post-hoc **clinical-data-only model achieved AUROC 0.817 (95% CI 0.778–0.850)** and the best cough+clinical model reached AUROC 0.832 (PMC12502651) — useful evidence that tabular clinical features alone carry strong TB signal.
- **TB Portals (NIAID):** Per NIAID, "linked socioeconomic/geographic, clinical, laboratory, radiological, and genomic data from over 33,000 international TB patient cases" (re3data lists the curated core as 7,525 cases / 3,305 genomic sequences / 9,020 chest X-ray and CT images). This is **genuinely paired CXR+clinical(+genomic) TB data — the only realistic such source** — but: (i) it requires a signed **Data Use Agreement**; (ii) clinical vs. imaging/genomic data now sit behind **two separate access requests** (AccessClinicalData@NIAID for clinical; TB Portals for imaging/genomic); and (iii) ~75% of cases are multi-/extensively drug-resistant, so it skews toward DR-TB rather than general detection.
- **Honest verdict:** Genuinely paired, freely-downloadable CXR+clinical public TB data essentially **does not exist**. Either (a) apply for the TB Portals DUA, or (b) adopt the modality-heterogeneous framing (recommended).

**(c) Sputum smear microscopy datasets with XML/bbox** (to contextualize your ~1000-image asset)

- **Kaggle "Tuberculosis Image Dataset" (saife245):** **928 images, 3,734 bacilli instances, with bounding-box annotations** — the closest public match to your "~1000 images + XML." **Action: open one XML file; if it is `<annotation><object><bndbox>…`, it is PASCAL VOC.** This is very likely your asset's origin or twin.
- **ZNSM-iDB:** ~2,000 Ziehl-Neelsen images across 7 categories (autofocus stacks, autostitching, manually segmented viewfields, with/without bacilli, occluded, over-stained), from 3 bright-field microscopes. Annotations are segmentation-style viewfields, **not** native VOC. (DOI 10.1117/1.JMI.4.2.027503; PMC5492794)
- **Costa/TBimages (IEEE DataPort):** 120 annotated detection images (+ autofocus stacks), Kinyoun acid-fast, from 12 patients; annotations are geometric shapes drawn on-image. (DOI 10.21227/ya12-j913)
- **Gomide et al. 2025:** 502 annotated ZN images + positive/negative patches with per-bacillus positions and Python code. (DOI 10.1117/1.JMI.12.3.034505)
- **CRITICAL CONFIRMATION:** Sputum smear datasets share **no patients** with any CXR dataset. They differ in modality (slide microscopy vs. thoracic radiograph), institution/geography (India/Brazil/Tanzania/Asia vs. USA/China), and even in what a "bounding box" means (individual bacilli vs. lung-lesion regions). TBX11K's composition is exclusively CXR sub-datasets. **Per-patient CXR+sputum fusion is impossible with public data — but combining them across clients is legitimate, not patient leakage.**

### 2. NON-IID / HETEROGENEITY SIMULATION
- **Source-based partitioning (most defensible):** treat each public dataset (Montgomery, Shenzhen, a TBX11K subset, Qatar/Dhaka, IN-CXR) as one client/"region." This produces realistic covariate/feature shift (scanner, geography, TB prevalence) — the strongest non-IID story for medical FL.
- **Dirichlet label-skew (Hsu et al. 2019, arXiv 1909.06335):** sample class proportions q ~ Dir(α). Use **α ∈ {0.1 (severe), 0.5 (moderate), 1.0 (mild), ∞ (IID)}**, consistent with Hsu et al. and the NIID-Bench study (Li et al., ICDE 2022, arXiv 2102.02079). Reuse the NIID-Bench partitioning code (`partition=noniid-labeldir`, `--beta`) directly; it is also packaged as a Flower baseline.
- **Client counts feasible on Colab:** 5–10 clients. Source-based gives ~5 natural clients; use 10 for the Dirichlet sweep. Flower's simulation engine emulates these as virtual clients on a single GPU.

### 3. FAIRNESS METHODS & METRICS
- **q-FFL / q-FedAvg (Li et al., ICLR 2020, arXiv 1905.10497):** server-side reweighting of the objective toward a more uniform accuracy distribution across clients; cheap, tunable via q. Code: github.com/litian96/fair_flearn. This is *client/performance* fairness, not demographic-group fairness.
- **FairFed (Ezzeldin et al., AAAI 2023, arXiv 2110.00857):** **server-side, fairness-aware aggregation for GROUP fairness** (e.g., sex), agnostic to the local debiasing method, built on secure aggregation. Best fit for demographic fairness; its advantage grows under high heterogeneity. This is your primary fairness algorithm.
- **AFL / min-max (Agnostic FL):** worst-client optimization; more expensive, lower priority.
- **Local reweighting / oversampling:** client-side, cheapest; combine with FairFed.
- **Metrics to report:** subgroup sensitivity/specificity/AUC gaps (by sex, age band), equal opportunity difference, equalized odds difference, demographic parity difference, worst-group and worst-client performance, and Jain's fairness index. Compute group metrics with **AIF360 / Fairlearn**.

### 4. HETEROGENEITY-ROBUST FL METHODS
- **FedProx (Li et al., MLSys 2020, arXiv 1812.06127):** adds a proximal term μ/2‖w−wᵗ‖² to the local objective; trivial to implement. Per the paper and the FedProx repo, "in highly heterogeneous settings, FedProx demonstrates significantly more stable and accurate convergence behavior relative to FedAvg — improving absolute test accuracy by 22% on average." Tune μ over {0.001, 0.01, 0.1, 0.5, 1}. **Best first robustness choice.**
- **SCAFFOLD (Karimireddy et al. 2020):** control variates correct client drift; can converge faster but roughly doubles communication and (per NIID-Bench) "cannot work effectively" under partial client participation. Moderate complexity — include as a comparison, not your workhorse.
- **Clustered FL & personalized FL (pFedMe, FedRep, Per-FedAvg):** per-cluster/per-client models for severe heterogeneity. TB-specific precedents already exist (HPFL hyper-network personalized FL; FedARC adaptive contrastive FL for multi-center TB CXR), which both validates the direction and means you must differentiate clearly.
- **Realistic for a solo student:** FedAvg → FedProx → optionally SCAFFOLD and one personalized method. All are available as Flower strategies or short custom loops.

### 5. MULTIMODAL FUSION ARCHITECTURE
- **CXR + clinical/tabular (where paired data exists, e.g., TB Portals):** DenseNet121 or ResNet image encoder + MLP tabular encoder, with **intermediate (feature-concatenation) or late (logit) fusion**. Precedent: DeepCOVID-Fuse (Ye et al., *Bioengineering* 2023; arXiv 2301.08798) fused CXR+clinical to **AUC 0.842 vs. 0.807 (CXR-only) and 0.502 (clinical-only)** on 1,657 patients; also MDF-Net and the RSNA transformer-fusion study. **Late fusion is the most robust to missing modalities** and is your safest default.
- **Modality-heterogeneous federated setting (the reframed Objective 2):**
  - **ClusMFL (Wang et al., arXiv 2502.12180):** FINCH feature clustering + supervised-contrastive alignment + modality-aware aggregation; explicitly handles **client-level and instance-level modality incompleteness** (demonstrated on ADNI MRI/PET). Directly analogous to your CXR/clinical/sputum-across-clients case — your strongest template.
  - **Feature Imputation Network (Poudel et al., MIUA 2025, arXiv 2505.20232):** a lightweight, low-dimensional translator reconstructs the bottleneck features of missing modalities; validated on MIMIC-CXR, NIH Open-I, and CheXpert. Computationally cheap — well suited to Colab.
  - **FedPIA (Saha et al., AAAI 2025, arXiv 2412.14424):** adapter permutation/integration via Wasserstein barycenters; handles statistical + modality + task heterogeneity for vision-language foundation models. **Heavier compute — treat as optional/stretch.**
  - **Modality-dropout training:** include as a simple, strong baseline.

### 6. PLATFORMS & TOOLS
- **FL framework — Flower (`flwr`), recommended.** It runs a **simulation mode** (Virtual Client Engine via Ray) on a single Colab/Kaggle GPU, with identical code for simulation and real deployment, and ships with FedAvg, FedProx, and QFedAvg strategies. You can assign **fractional GPU per client** (e.g., `--client-resources-num-gpus=0.2` → ~5 clients sharing one GPU). TensorFlow Federated is more rigid; FedML is heavier; hand-written PyTorch FedAvg/FedProx loops are fine but reinvent infrastructure. A 2025 benchmarking study (arXiv 2511.00037) positions Flower as the lightweight, research-friendly choice for medical-imaging FL (vs. NVIDIA FLARE for production scale).
- **Fairness:** AIF360, Fairlearn. **Differential privacy:** Opacus — **only if you make an explicit privacy claim**, since DP noise typically hurts accuracy on small TB-positive sets. **XAI:** pytorch-grad-cam, for CXR lesion saliency and as a sanity check on the sputum bacilli detector.

### 7. EXPERIMENTAL DESIGN & EVALUATION
- **Baselines:** centralized training (upper bound), local-only per-client models (lower bound), and FedAvg.
- **Proposed methods:** FedProx/SCAFFOLD (heterogeneity) combined with FairFed/q-FFL (fairness) for Objective 1; ClusMFL / feature-imputation for the modality-incomplete Objective 2.
- **Metrics:** accuracy, sensitivity, specificity, AUC, F1; **plus** the fairness metrics in §3; **plus** communication cost (rounds to target accuracy, MB transferred per round).
- **Ablations:** α sweep (0.1 / 0.5 / 1.0); with vs. without the fairness module; varying modality-availability ratios across clients; image-encoder choice.
- **Statistical rigor:** ≥3 random seeds; report mean ± 95% CI; perform **leave-one-dataset-out external validation** (e.g., train on Shenzhen + TBX11K, test on Montgomery or IN-CXR) to demonstrate generalization under domain shift.

### 8. REALISTIC BUILD ORDER & FEASIBILITY VERDICT
- **Phase 1 (months 1–2):** Centralized DenseNet121 TB classifier on Shenzhen + Montgomery; parse and harmonize demographics from `ClinicalReadings`. Lock in the metric suite and Grad-CAM.
- **Phase 2 (months 2–4):** Stand up Flower simulation; FedAvg across both source-based and Dirichlet partitions; add FedProx (then SCAFFOLD). → **Objective 1 heterogeneity core.**
- **Phase 3 (months 4–6):** Add fairness (FairFed + q-FFL); full subgroup evaluation by sex/age. → **Objective 1 complete; first paper.**
- **Phase 4 (months 6–9):** Build the tabular MLP + CXR fusion module (on TB Portals if the DUA is granted; otherwise on a clinical-tabular proxy / simulated clinical features).
- **Phase 5 (months 9–14):** Modality-heterogeneous FL — CXR clients + clinical-tabular clients + a sputum bacilli-detection client — implementing ClusMFL and/or the feature-imputation network; evaluate across modality-incompleteness levels. → **Objective 2; second/third paper.**
- **Verdicts:**
  - **Objective 1:** **High feasibility.** Standard datasets, modest compute, mature tooling.
  - **Objective 2:** **Feasible only in the modality-heterogeneous framing.** Per-patient tri-modal fusion on public data is **infeasible**; do not promise it.
  - **Sputum incorporation:** **Feasible as a separate modality-client** (object detection of bacilli with YOLO/Faster R-CNN), **not** as fused per-patient features.
  - **Compute risks:** full TBX11K plus native-resolution CXRs (≈3K×3K) and bacilli-scale microscopy can exhaust Colab RAM/VRAM. Downsample CXRs to 224/256, subsample TBX11K, and use fractional-GPU clients. Avoid FedPIA-scale foundation models unless you secure more compute.

### 9. BETTER ALTERNATIVE FRAMING
Frame the whole thesis around **"federated learning across heterogeneous TB evidence types."** In Objective 1, clients differ in data *distribution* (statistical heterogeneity). In Objective 2, clients differ in *which modality they possess* (modality heterogeneity): some hold CXR + demographics, some hold clinical/tabular records, and one holds sputum-smear microscopy. The ~1000 sputum images then become a **legitimate third modality-client running bacilli object detection**, unified under a federated multi-task / missing-modality framework. This turns a "forced add-on" into the **central contribution: the first TB-specific modality-incomplete federated benchmark** — which does not yet exist in the literature, even though the underlying machinery (ClusMFL, feature-imputation, FedPIA) is established in other domains.

## Recommendations
1. **Start Objective 1 now** on Shenzhen + Montgomery in Flower simulation. It de-risks the entire program and produces a paper within ~6 months. Benchmark FedAvg → FedProx → FairFed/q-FFL, sweeping α ∈ {0.1, 0.5, 1.0}.
2. **Apply for the TB Portals Data Use Agreement immediately, in parallel** — it is the only realistic source of genuinely paired CXR+clinical TB data, access takes time, and you may need both the clinical and imaging requests.
3. **Reframe Objective 2** as modality-heterogeneous / missing-modality FL. Do not commit to per-patient tri-modal fusion in your proposal or papers.
4. **Verify the sputum XML schema** (PASCAL VOC vs. other) and design the sputum stream as a bacilli-detection modality-client.
5. **Decision threshold:** if the TB Portals DUA is denied or delayed past month 8, fall back to a fully-open **CXR(+demographics) + simulated-clinical-tabular + sputum-detection** modality-incomplete benchmark — still novel, still publishable.
6. **Differentiate from prior TB FL work** (HPFL, FedARC) explicitly: your contributions are the *fairness* axis (Objective 1) and the *modality-incomplete* axis (Objective 2), neither of which those papers address.

## Caveats
- TBX11K and the Qatar/Dhaka database lack per-image demographics; **restrict fairness analysis to Shenzhen and Montgomery** (and to VinDr/IN-CXR only if their demographics are confirmed usable).
- TB-positive counts are small (58 in Montgomery, 336 in Shenzhen), so subgroup fairness estimates will be noisy — **always report confidence intervals and avoid over-claiming** equalized-odds improvements on tiny strata.
- Differential privacy (Opacus) and foundation-model fine-tuning (FedPIA) can exceed Colab budgets and degrade small-data accuracy; treat both as optional stretch goals, not core deliverables.
- The CODA clinical-only AUROC (0.817) and DeepCOVID-Fuse gains are from non-TB-detection-on-CXR settings (cough triage and COVID risk, respectively); cite them as *motivating evidence* for tabular signal and fusion, not as TB-detection benchmarks.
- Existing TB FL papers (HPFL, FedARC) are accessible mainly via paywalled/secondary sources; confirm their exact methods from the primary venues before positioning your novelty against them.