# IoT + AI for Secure Remote Healthcare Monitoring

This repository contains a research prototype that integrates **Reinforcement Learning (Q‑learning + an LSTM‑based component)** with a **synthetic patient vitals dataset** and a **proof‑of‑concept blockchain encryption/ledger notebook** for secure, intelligent remote patient monitoring. The learning agent decides when to **monitor** vs **administer medication** using time‑series vitals while the security layer illustrates how health records can be **encrypted, hashed, and chained** for integrity and auditability.

> ⚠️ **Research prototype only. Not medical advice.** Do not use to make clinical decisions.

---

## Repository Structure
- `Qlearning.ipynb` — Train/evaluate tabular Q‑learning on the synthetic vitals environment (plots for reward trends and policy behavior).
- `qlearned_lstm.py` — Q‑learning pipeline with an LSTM‑based sequence component (e.g., forecaster/feature extractor) for temporal signals.
- `dataset.ipynb` — Synthetic data generation and preprocessing for patient vitals and labels/actions.
- `blockchain_encryption.ipynb` — Prototype of encryption + lightweight ledgering (hash links) for integrity/auditability of vitals records.
- `Indaba_poster_Presentation.pdf` — A project overview poster with architecture and outcomes.

---

## Key Ideas
- **Task:** Given time‑series vitals (e.g., Temp, BP, HR, RR, SpO₂, etc.), learn a policy that balances *monitoring* vs *medication*.
- **RL Core:** Tabular Q‑learning for action‑value updates; LSTM module used for temporal pattern capture (e.g., risk estimation or value shaping).
- **Security Layer:** Demonstrates content hashing, symmetric encryption, and simple chained‑record logic as a blueprint toward a permissioned ledger (e.g., Hyperledger Fabric in future work).
- **Ethics & Privacy:** Uses **synthetic** data for rapid iteration without PHI.

---

## Getting Started

### 1) Environment
- Python 3.10+ recommended

Install core dependencies (adapt as needed):
```bash
pip install numpy pandas scikit-learn matplotlib torch jupyter ipykernel
```

### 2) Data
Open and run **`dataset.ipynb`** to generate/inspect the synthetic vitals. You can export CSV/Parquet if you wish to feed scripts directly.

### 3) Training & Evaluation
- **Notebook workflow:** Open **`Qlearning.ipynb`** and run cells end‑to‑end to:
  - Train the RL agent
  - Plot **cumulative reward** and **rolling average reward**
  - Inspect the **Monitor vs Medicate** action distribution and confusion matrix
- **Script workflow:** Run the LSTM‑augmented pipeline (edit in‑file hyperparameters as needed):
  ```bash
  python qlearned_lstm.py
  ```

### 4) Security Prototype
Open **`blockchain_encryption.ipynb`** to see how a vitals record is:
1. Serialized and encrypted
2. Hashed and appended to a simple chained structure
3. Verified for tamper‑evidence

> This is a minimal, educational prototype—*not* a full blockchain node. Future work targets a permissioned BFT ledger (e.g., Hyperledger Fabric).

---

## Metrics You’ll See
- **Cumulative Reward** per episode  
- **Rolling Average Reward** (e.g., 10‑episode window)  
- **Success Rate** (task/goal completion, scenario‑dependent)  
- **Sample Efficiency** (first episode where rolling action accuracy surpasses a set threshold and remains stable)  
- **Per‑step Action Accuracy** and **Confusion Matrix** (Monitor vs Medicate)  
- **Precision/Recall/F1** on intervention decisions (where applicable)

---

## Typical Results (Example)
- Increasing cumulative/rolling rewards as the policy stabilizes.
- Action distribution often skews **Monitor > Medicate** under conservative reward shaping (penalizing unnecessary interventions).
- Stable policy behavior over longer horizons with the LSTM component improving temporal signal understanding.

For the research poster summarizing the layered architecture (IoT → Blockchain → RL) and example outcomes, see **`Indaba_poster_Presentation.pdf`**.

---

## Roadmap
- [ ] Replace the toy ledger with a real **permissioned blockchain** deployment (e.g., Fabric).
- [ ] Add **hyperparameter CLI** for `qlearned_lstm.py` (episodes, α, γ, ε, seeds).
- [ ] Support **real wearable streams** and plug‑in simulators.
- [ ] Broaden **reward functions** (adverse‑event penalties, resource/cost terms).
- [ ] Add evaluation suites (stress tests, off‑policy evaluation, confidence intervals).

---

## Citing / Attribution
If you build on this repository, please cite and acknowledge the authors.

---

## License
 'MIT LICENSE`.

---

## Contact
Lukman Kunveng
+233248653219
lukmankunveng@gmail.com
