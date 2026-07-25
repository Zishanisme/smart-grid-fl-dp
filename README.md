# ⚡ Privacy-Preserving Federated Learning for Feeder-Level Outage Prediction in Distribution Grids

> **A privacy-preserving AI framework enabling electricity utilities to collaboratively predict feeder-level outages without sharing sensitive operational data.**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)]()
[![Flower](https://img.shields.io/badge/Federated-Learning-green.svg)]()
[![Opacus](https://img.shields.io/badge/Differential-Privacy-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

---

# Overview

Electricity distribution companies generate enormous volumes of operational data from feeders, substations, weather systems, and asset monitoring platforms. While combining this information across utilities could significantly improve outage prediction, regulatory requirements and commercial constraints prevent raw data from being shared.

This project demonstrates how **Federated Learning**, **Differential Privacy**, and **Secure Aggregation** can be combined to build a collaborative machine learning framework where:

- Each utility trains a model locally.
- Raw operational data never leaves the organisation.
- Only privacy-preserving model updates are exchanged.
- A global model is created without exposing sensitive infrastructure data.

The repository accompanies the research white paper:

> **Privacy-Preserving Federated Learning for Feeder-Level Outage Prediction in Distribution Grids**

---

# Key Contributions

- Privacy-preserving federated learning pipeline for power distribution networks
- Differential Privacy (DP-SGD) integration using Opacus
- Secure Aggregation simulation based on Bonawitz et al.
- Comparison of Local, Centralised and Federated learning
- Evaluation of Tabular Neural Networks and Graph Neural Networks
- Analysis of graph learning under topology heterogeneity
- Synthetic smart grid dataset for reproducible experimentation

---

# Architecture

```text
                +----------------------+
                | Utility A            |
                | Local Training       |
                +----------+-----------+
                           |
                Protected Model Updates
                           |
                +----------v-----------+
                | Federated Server     |
                | FedAvg / FedProx     |
                +----------+-----------+
                           |
                Global Model Parameters
                           |
      -----------------------------------------
      |                 |                    |
+-----v-----+     +-----v-----+      +------v------+
| Utility A |     | Utility B |      | Utility C   |
| Local Data|     | Local Data|      | Local Data  |
+-----------+     +-----------+      +-------------+
```

---

# Experimental Results

| Model | AUC-ROC |
|---------|---------|
| Centralised Model | 0.76–0.78 |
| Local MLP | 0.53–0.63 |
| Local GraphSAGE | 0.85–0.86 |
| Federated MLP | **0.918** |
| Federated MLP + Differential Privacy (ε≈1.0) | **0.918** |
| Federated MLP + Differential Privacy (ε≈2.0) | 0.890 |
| Federated GraphSAGE + DP + Secure Aggregation | 0.38–0.39 |

---

# Key Findings

## 1. Federated Learning improves performance

Local models trained independently achieved relatively modest performance.

Collaborative federated learning increased predictive performance to:

**AUC = 0.918**

without exchanging raw operational data.

---

## 2. Privacy is nearly free

Adding Differential Privacy produced almost no measurable reduction in predictive performance.

| Configuration | AUC |
|--------------|------|
| Federated Learning | 0.918 |
| Federated + DP (ε≈1.0) | 0.918 |

This demonstrates that strong privacy guarantees can be achieved with negligible utility loss for this application.

---

## 3. Graph models failed under federation

Graph Neural Networks achieved strong performance when trained locally.

However, performance collapsed after federated aggregation.

Reason:

Different utilities possess different network topologies.

Graph convolution filters therefore learn topology-specific representations that cannot simply be averaged across clients.

---

# Dataset

The experiments use a synthetic feeder-level dataset representing multiple electricity distribution utilities.

Dataset characteristics:

- 500 feeders
- 3 years of daily observations
- ~547,500 records
- 3 simulated utility clients
- Weather variables
- Asset condition
- Feeder loading
- Outage history
- Distributed Energy Resource penetration

Target:

Predict whether a feeder will experience an outage within the next seven days.

---

# Technology Stack

- Python
- PyTorch
- Flower Federated Learning
- Opacus Differential Privacy
- Scikit-learn
- NumPy
- Pandas
- NetworkX
- Matplotlib

---

# Repository Structure

```text
├── data/
│
├── models/
│
├── privacy/
│
├── aggregation/
│
├── experiments/
│
├── results/
│
├── paper/
│
├── figures/
│
├── requirements.txt
│
├── federated_final.py
│
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/Zishanisme/<repository-name>.git
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

Example:

```bash
python federated_final.py \
    --epsilon 1.0 \
    --rounds 10
```

---

# Privacy Mechanisms

The project incorporates:

## Differential Privacy

- DP-SGD
- Gradient clipping
- Gaussian noise
- Privacy accounting using ε (epsilon)

## Secure Aggregation

- Pairwise masking
- Weighted FedAvg updates
- Protected client updates

---

# Threat Model

The framework considers protection against:

- Membership inference attacks
- Honest-but-curious federated servers
- Passive communication interception

Future work includes:

- Byzantine robustness
- Model poisoning defence
- Sybil attack protection

---

# Future Work

- Real SCADA integration
- ADMS integration
- Graph-aware federated optimisation
- Federated Graph Transformers
- Adaptive privacy budgets
- Production secure aggregation
- Multi-utility validation

---

# White Paper

The complete research white paper is available in the **paper/** directory.

Suggested citation:

```bibtex
@techreport{khan2025,
title={Privacy-Preserving Federated Learning for Feeder-Level Outage Prediction in Distribution Grids},
author={Zishan Khan},
year={2025}
}
```

---

# Author

**Zishan Ali Khan**

Energy AI | Machine Learning | Federated Learning | Smart Grids

- GitHub: https://github.com/Zishanisme
- LinkedIn: *(Add your LinkedIn profile)*

---

# License

Released under the MIT License.

---

# Disclaimer

This repository is intended for research and educational purposes. The experiments are conducted using synthetic data and should not be deployed in production power systems without additional validation using real utility datasets.
