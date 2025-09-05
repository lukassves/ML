# Quantitative ML Trading Framework

![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/license-All%20rights%20reserved-red)
![Status](https://img.shields.io/badge/status-Active-brightgreen)

A full end-to-end machine learning pipeline for **cryptocurrency trading signals**.  
It combines **high-frequency OHLCV + order-flow (OF)** features, multi-stage data processing, supervised model training (**XGBoost 2D**), and realistic trade simulation with leverage and fees.

---

## 📐 Architecture

```text
          ┌───────────────────────────┐
          │ Stage 1: Data Acquisition │
          │  looperis_data.py         │
          └───────────────┬───────────┘
                          │
         WITH_BTC_USDT_1m_part*.csv
                          │
          ┌───────────────▼───────────┐
          │ Stage 2: Labeling & ML    │
          │  AI2D.py                  │
          └───────────────┬───────────┘
                          │
       ┌──────────────────┴───────────────────┐
       │ Labels  │ Models │ Mastery │ Simulation│
       └───────────────────────────────────────┘
Stage 1 (looperis_data.py)
Fetches OHLCV via CCXT + parses Bybit OF archives.
Produces feature-enriched CSVs (51+ engineered features).

Stage 2 (AI2D.py)

Label generation (TP/SL/lookahead grid)

Model training & checkpointing (XGBoost)

Mastery & feature importance reports

Black-box opinion (predict_proba + gates)

Stage-2 simulation with leverage/fees

## ⚡ Features
Parallelized ingestion: full CPU utilization via multiprocessing + joblib (loky)

Memory guards: RAM estimation, chunked I/O, safe multiprocessing (freeze_support)

AEA filters: accumulation/distribution, RSI/ADX, spikes, order-flow windows (OF-5/OF-15)

Checkpointed models: per-chunk .joblib with early stopping & feature summaries

Explainability: SHAP attribution with noise filtering

Simulation realism: leverage, round-trip fees, capped position sizing

Feedback loops: per-bar win/loss boost bank (1.2 / 0.8)

Reproducibility: deterministic seeds, consistent feature schema

📦 Requirements
Python ≥ 3.11

PostgreSQL (for persistence of large datasets)

Install dependencies:

pip install -r requirements.txt
Core packages:
ccxt, pandas, numpy, xgboost, scikit-learn,
joblib, tqdm, psutil, pympler, scipy, shap, matplotlib.

▶️ Usage
1. Stage 1 — Data Acquisition
Collects OHLCV + OF data and generates enriched feature files.
(Default loop: 8-month chunks from 2022-11-10 → today)

python looperis_data.py
Outputs:

COMBINED_BTC_USDT_part*.csv — merged raw data

WITH_BTC_USDT_1m_part*.csv — enriched features (for Stage 2)

2. Stage 2 — Labeling, Training & Simulation

python AI2D.py
Configurable parameters (in AI2D.py):

| Param            | Description                        | Example |
|------------------|------------------------------------|---------|
| tp_values        | Take-profit targets (% move)       | [0.10]  |
| sl_values        | Stop-loss targets (% move)         | [0.10]  |
| lookahead_values | Bars to look ahead for TP/SL       | [20]    |
| initial_balance  | Starting balance in simulation     | 300     |
| leverage         | Leverage factor                    | 25      |
| fee              | Trading fee per side (decimal)     | 0.0006  |
| max_used_balance | Cap for position sizing            | 450     |
| n_jobs           | Parallel workers                   | 16      |
| batch_size       | Training batch size                | 1024    |


Outputs (by category):

Labels: labeliai/*.csv

Models: checkpoints/model_part*.joblib, encoder.joblib

Reports: mastery_*.csv, black_box_opinion_*.csv

Simulation: preds_log_*.csv, simuliacijos_outputas_*.csv, feedback_bank_*.csv

📊 Example Run
# Stage 1: Generate features
python looperis_data.py

# Stage 2: Train and simulate with TP=10%, SL=10%, lookahead=20
python AI2D.py
Result:

Models saved in checkpoints/

Simulation report in simuliacijos_outputas_tp10_sl10.csv

Mastery insights in mastery_tp10_sl10_look20.csv

🧩 Tips
Excel row limits: CSV itself has no row cap; Excel has a 1,048,576-row limit. Use PostgreSQL for stability.

Visualization: set SHOW_BOX_PREVIEW=True to overlay entries/exits on charts.

Aggregation: run mastery across multiple parts for stronger signal insights.

Out-of-sample checks: set OOS_FILE for validation preview.

📂 Repo Structure
.
├── looperis_data.py         # Stage 1: CCXT + OF → features
├── AI2D.py                  # Stage 2: Labels, training, simulation
├── checkpoints/             # Model checkpoints (.joblib)
├── labeliai/                # Label chunks
├── outputs/                 # Mastery, black-box, simulations
├── docs/                    # Extended documentation (.docx, diagrams)
├── requirements.txt
└── README.md
📜 License & Authorship
© 2025 Lukas Svešnikovas
All rights reserved.

For academic/research use only.
Commercial use requires explicit permission.