Privacy-Preserving Federated Learning for Smart Grid Outage Prediction

Federated feeder-level outage prediction across utilities — without sharing raw operational data.

This repository contains the implementation and supporting material for the white paper:

Privacy-Preserving Federated Learning for Feeder-Level Outage Prediction in Distribution Grids

The project investigates whether electricity utilities can train a shared outage-prediction model while keeping all raw feeder data local. The system combines federated learning, differential privacy, secure aggregation, and graph-based modelling.

Why this project matters

Utilities hold valuable data on feeder loading, weather exposure, asset condition, maintenance history, and outages. However, legal, regulatory, and commercial constraints often prevent this information from being pooled across organisations.

This project takes a different approach:

Raw data remains within each utility.

Each utility trains a local model.

Only protected model updates are shared.

A global model is created through federated aggregation.

The aim is to improve outage prediction without requiring any utility to surrender data sovereignty.

Key results

Experimental condition

AUC-ROC

ECE

Privacy budget

Centralised model

0.76–0.78

~0.22

N/A

Local-only MLP

0.53–0.63

~0.30

N/A

Local-only GraphSAGE

0.85–0.86

~0.40

N/A

Federated MLP, no DP

0.918

~0.010

N/A

Federated MLP + DP

0.918

~0.007

ε ≈ 1.0

Federated MLP + DP

0.890

~0.009

ε ≈ 2.0

Federated GraphSAGE + DP + SecAgg

0.38–0.39

~0.009

ε ≈ 1.0

Main findings

Federation substantially improved prediction quality.Local-only MLP models achieved AUC values between 0.53 and 0.63, while the federated MLP reached 0.918.

Differential privacy had almost no measurable cost at ε ≈ 1.0.The AUC difference between the non-private federated model and the private federated model was below 0.002.

Graph models failed under topology-heterogeneous federation.GraphSAGE performed strongly when trained locally, but collapsed when graph parameters were aggregated across utilities with structurally different network topologies.

Core insight

The strongest improvement came from cross-utility data diversity, not from using a more complex model.

A tabular MLP performed strongly under federated learning because feeder features such as loading, temperature, asset age, and outage history have comparable meaning across utilities.

Graph models behaved differently. Their learned filters depended heavily on each utility's local topology. When those filters were averaged across structurally different networks, the shared graph representation became incoherent.

This motivates a hybrid design:

Federated tabular encoder for transferable feeder representations

Local graph head for utility-specific topology learning

System architecture

Layer

Component

Purpose

Data

Synthetic feeder-level grid dataset

Controlled cross-utility experiments

Features

Rolling 7-day and 30-day statistics, weather, DER and asset indicators

Capture temporal and static risk

Local models

Tabular MLP, GraphRiskModel, SAIDIGraphModel

Feeder-level risk estimation

Federation

FedAvg + FedProx

Shared learning under non-IID data

Privacy

Opacus DP-SGD

Protect individual training records

Aggregation

Bonawitz-style secure aggregation

Hide individual client updates

Evaluation

AUC-ROC, ECE, top-10% capture, epsilon

Measure discrimination, calibration, operational utility, and privacy

Privacy and security design

Differential privacy

The project uses DP-SGD through Opacus.

Each client:

Clips per-sample gradients to a maximum norm.

Adds calibrated Gaussian noise.

Tracks cumulative privacy loss across all federated rounds.

A single persistent PrivacyEngine is used per client so that epsilon accounting is not reset between rounds.

Secure aggregation

The implementation simulates Bonawitz-style secure aggregation.

Each client:

Applies its FedAvg weight before masking.

Generates pairwise masks from shared secrets.

Sends only the masked weighted update.

The server receives only the aggregate sum, not any individual client's update.

The current secure aggregation implementation is an algorithmic simulation. Production use would require X25519, authenticated TLS, and dropout recovery.

Dataset

The experiments use a synthetic dataset calibrated to published engineering and reliability benchmarks.

Dataset characteristics

500 feeders

3 years of daily observations

Approximately 547,500 feeder-day records

3 simulated utility clients

Temporal split: 80% train, 10% validation, 10% test

Multiple climate zones

Asset age, loading, cable, vegetation, DER, and outage-history features

Binary target: outage within the next 7 days

The data is intentionally non-IID across utilities to reflect differences in:

climate

asset age

outage prevalence

feeder condition

loading behaviour

The dataset is synthetic and does not replace validation on real SCADA or ADMS utility records.

Experimental conditions

The repository supports six main comparisons:

Centralised model

Local-only MLP

Local-only graph model

Federated MLP with FedProx

Federated MLP with differential privacy

Federated graph model with differential privacy and secure aggregation

Repository structure

.
├── README.md
├── LICENSE
├── requirements.txt
├── run.sh
├── federated_final.py
│
├── data/
│   ├── raw/
│   │   ├── model_dataset.csv
│   │   └── assets.csv
│   └── generated/
│
├── src/
│   ├── clients/
│   ├── models/
│   ├── privacy/
│   ├── aggregation/
│   ├── topology/
│   └── evaluation/
│
├── experiments/
│   ├── configs/
│   └── logs/
│
├── results/
│   ├── tables/
│   ├── plots/
│   └── metrics/
│
└── paper/
    └── SmartGrid_WhitePaper_FINAL.pdf

Adjust the folder names above to match your final repository layout.

Installation

1. Clone the repository

git clone https://github.com/Zishanisme/<YOUR-REPOSITORY>.git
cd <YOUR-REPOSITORY>

2. Create a virtual environment

Windows

python -m venv .venv
.venv\Scripts\activate

macOS / Linux

python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

Running the federated experiment

Differential privacy with ε ≈ 1.0

python federated_final.py \
  --data data/raw/model_dataset.csv \
  --assets data/raw/assets.csv \
  --rounds 10 \
  --epsilon 1.0

Differential privacy with ε ≈ 2.0

python federated_final.py \
  --data data/raw/model_dataset.csv \
  --assets data/raw/assets.csv \
  --rounds 10 \
  --epsilon 2.0

Windows PowerShell

python federated_final.py `
  --data data/raw/model_dataset.csv `
  --assets data/raw/assets.csv `
  --rounds 10 `
  --epsilon 1.0

Expected outputs

Depending on configuration, the pipeline produces:

AUC-ROC

Expected Calibration Error

Top-10% capture rate

Cumulative epsilon

Per-client metrics

Federated-round logs

Privacy-accounting output

Graph-filter alignment measurements

Model checkpoints

Result tables and plots

Graph federation failure

One of the main findings is that locally strong graph models do not necessarily federate well.

Observed result:

Local GraphSAGE: AUC 0.85–0.86

Federated GraphSAGE: AUC 0.38–0.39

The likely reason is topology heterogeneity.

Graph convolutional filters learn expectations about local neighbourhood structure. When clients operate different grid topologies, averaging those filters can produce a global model that is well-matched to none of them.

Measured cosine similarity:

Graph filter alignment: 0.3–0.5

MLP weight alignment: 0.8–0.95

Threat model

Threat

Mitigation

Honest-but-curious server

DP-SGD + secure aggregation

Passive network eavesdropping

TLS in production deployment

Membership inference

Differential privacy

Maliciously scaled client updates

Gradient clipping

Not yet fully addressed:

Sybil attacks

Malicious server deviation

Data poisoning

Full Byzantine robustness

Inference attacks against the released global model

Regulatory relevance

The architecture is designed to support privacy-by-design and data-minimisation objectives under:

GDPR Article 5

GDPR Article 25

EU AI Act requirements for critical infrastructure

Saudi Arabia PDPL

UAE PDPL / ADGM frameworks

This repository demonstrates technical controls. It does not by itself establish legal compliance.

Future work

Federated encoder with local graph heads

Validation using real SCADA or ADMS data

Full privacy-budget sweep across ε ∈ {0.5, 1.0, 2.0, 4.0}

SAIDI-weighted outage optimisation

Byzantine-robust aggregation

Production-grade secure aggregation

Larger federations with more utility clients

Formal multi-seed evaluation

White paper

The full white paper is available in the paper/ directory.

Suggested citation:

@techreport{khan2025smartgridfl,
  title={Privacy-Preserving Federated Learning for Feeder-Level Outage Prediction in Distribution Grids},
  author={Khan, Zishan},
  year={2025},
  institution={Independent Research}
}

Author

Zishan KhanEnergy AI ResearcherFederated Learning and Privacy Engineering

GitHub: Zishanisme

LinkedIn: Add your LinkedIn URL

Email: Add your preferred public email

License

This project is licensed under the MIT License. See LICENSE for details.# smart-grid-fl-dp
