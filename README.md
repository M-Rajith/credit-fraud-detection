# Credit Card Fraud Detection
## Per-User Transaction History Analysis — LSTM + Isolation Forest

---

## Project Overview

This project detects credit card fraud by **analysing each cardholder's full
transaction history** to identify deviations from their personal spending
patterns. Unlike transaction-at-a-time approaches, this system understands
*context* — it knows what is normal for YOU before deciding if something is
suspicious.

---

## Architecture

```
Transaction History (per user)
          │
          ▼
  ┌─────────────────┐
  │  Feature Eng.   │  amount, hour, day, region, txn_gap,
  │  + Scaling      │  rolling stats, amt_vs_avg, cat_code
  └────────┬────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌────────┐   ┌──────────────────┐
│  LSTM  │   │ Isolation Forest │
│Sliding │   │  Per-transaction │
│Windows │   │  Anomaly Score   │
└────┬───┘   └────────┬─────────┘
     │                │
     └──────┬─────────┘
            ▼
     Weighted Ensemble
     (60% LSTM + 40% IsoForest)
            │
            ▼
    Fraud Alert Report
```

---

## How Each Model Works

### LSTM (Bidirectional)
- Treats each user's last N transactions as a **sequence**
- Learns what a normal spending pattern looks like over time
- A sliding window moves through the user's history
- If a window's pattern is anomalous → fraud alert
- **Catches**: slow behavioural drift, unusual sequences, pattern breaks

### Isolation Forest
- Trained exclusively on **legitimate transactions**
- Scores each transaction independently based on how hard it is to isolate
- Unusual transactions (far from the normal cluster) get high anomaly scores
- **Catches**: sudden one-off anomalies — big spikes, foreign transactions

---

## Fraud Patterns Detected

| Pattern           | Description                                          |
|-------------------|------------------------------------------------------|
| `foreign_txn`     | Transaction in a region the user has never used      |
| `amount_spike`    | Single transaction 8-20× the user's rolling average  |
| `velocity_burst`  | 4-6 transactions within 30 minutes                  |
| `odd_hour`        | Transaction at 1-4 AM for a daytime-only user        |
| `category_anomaly`| Merchant category the user has never visited before  |

---

## File Structure

```
fraud_detection/
├── main.py                    ← Run this to execute the full pipeline
├── generate_data.py           ← Synthetic per-user transaction history
├── preprocess.py              ← Feature engineering + sequence building
├── lstm_model.py              ← Bidirectional LSTM model
├── isolation_forest_model.py  ← Isolation Forest anomaly detector
├── history_analyzer.py        ← Per-user history scanning (core logic)
├── evaluate.py                ← Sequence + user-level metrics
├── fraud_reporter.py          ← Alert report generation
├── visualize.py               ← All plots
├── requirements.txt
│
├── data/
│   └── transactions.csv       ← Generated dataset
│
├── outputs/                   ← Saved models + plots
│   ├── lstm_model.keras
│   ├── isolation_forest.pkl
│   ├── scaler.pkl
│   ├── 01_lstm_training.png
│   ├── 02_roc_pr.png
│   ├── 03_confusion_matrices.png
│   ├── 04_score_distribution.png
│   ├── 05_metrics_summary.png
│   └── user_timelines/        ← Per-user score timeline plots
│
└── reports/
    ├── fraud_alerts.txt        ← Full text alert report
    ├── flagged_transactions.csv
    └── user_risk_summary.csv
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python main.py
```

---

## Key Design Decisions

**User-level train/test split** — All transactions for a user stay in either
train or test, preventing data leakage.

**LSTM trained on sequences, IsoForest on transactions** — Each model operates
at its natural granularity; the ensemble combines both signals.

**IsoForest trained on legitimate only** — The model builds a tight boundary
around normal behaviour; anything outside it is suspicious.

**Sliding window analysis** — Every position in a user's history is examined.
The fraud score at each transaction is the *maximum* window score that
included it, meaning a single anomalous window raises the alarm for all
transactions in that window.

**Weighted ensemble** — LSTM (60%) + IsoForest (40%). The LSTM has higher
weight because it is supervised and has access to labelled training data.

---

## Output Example (fraud_alerts.txt)

```
USER ALERT  ──  USER_0042
  Risk Level          : 🔴 CRITICAL
  Max Hybrid Score    : 0.8731
  Max LSTM Score      : 0.9124
  Max IsoForest Score : 0.7643
  Total Transactions  : 67
  Flagged Transactions: 5
  Ground Truth Fraud  : YES
  Fraud Type(s)       : amount_spike, velocity_burst

  SUSPICIOUS TRANSACTIONS
  ─────────────────────────────────────────────────────────────────
  ▶ TXN_0031482
    Timestamp   : 2024-04-15 02:13:44
    Amount      : $2,847.50
    Category    : luxury
    Region      : west  ⚠ FOREIGN
    Time        : Mon 02:00
    Txn Gap     : 3.2 min from previous
    Amt/Avg     : 14.23× (5-txn rolling avg: $52.10)
    LSTM Score  : 0.9124
    ISO  Score  : 0.8812
    Hybrid Score: 0.8997
    Reason      : Foreign region | Amount 14.2× avg | Unusual hour (02:00)
    True Label  : FRAUD
```
