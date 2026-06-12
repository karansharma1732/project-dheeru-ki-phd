# 3. Methods

<!-- Condense docs/methodology.md to what this paper uses. -->

## 3.1 Problem formulation
Federated TB binary classification across K clients; notation for local objectives and the
global model.

## 3.2 Data partitioning / heterogeneity
- Source-based partitioning (each dataset = client).
- Dirichlet label-skew, `q ~ Dir(α)`, α ∈ {0.1, 0.5, 1.0, ∞}.

## 3.3 Model
DenseNet121 CXR encoder; input 224/256; training details.

## 3.4 Aggregation methods
FedAvg (baseline), FedProx (proximal μ), SCAFFOLD (control variates).

## 3.5 Fairness-aware aggregation
FairFed (group fairness, sex/age); q-FFL (performance fairness, q). Local reweighting.

## 3.6 Metrics
Accuracy, sensitivity, specificity, AUC, F1; subgroup gaps, equalized-odds difference,
demographic-parity difference, worst-group / worst-client, Jain's index; communication cost.
