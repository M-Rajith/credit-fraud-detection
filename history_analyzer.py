"""
history_analyzer.py
════════════════════
Per-user transaction history analyser.

For every user in the test set this module:

  1. Slides the LSTM window across their full transaction history
     to produce a time-series of "suspicion scores" — one per window.
  2. Scores every individual transaction with Isolation Forest.
  3. Fuses the two scores (weighted ensemble).
  4. Identifies the EXACT transactions that triggered an alert.
  5. Returns a structured report per flagged user.

Output
──────
A list of UserFraudReport dicts, one per flagged user:
  {
    user_id          : str
    total_txns       : int
    flagged_txns     : list of txn records with scores
    max_lstm_score   : float    (worst window score)
    max_iso_score    : float    (worst single-txn score)
    max_hybrid_score : float    (worst combined score)
    fraud_types_found: list[str]  (ground-truth labels for evaluation)
    is_true_fraud    : bool       (ground truth)
    alert_triggered  : bool
  }
"""

import numpy as np
import pandas as pd
from tensorflow.keras import Model
from sklearn.ensemble import IsolationForest

from preprocess import FEATURE_COLS, SEQ_LEN
from isolation_forest_model import score_user_transactions


# ─────────────────────────────────────────────────────────────────────────────
def _build_user_sequences(user_scaled: np.ndarray, seq_len: int) -> np.ndarray:
    """Return all sliding windows for one user's scaled features."""
    if len(user_scaled) < seq_len:
        return np.empty((0, seq_len, user_scaled.shape[1]), dtype=np.float32)
    seqs = []
    for i in range(len(user_scaled) - seq_len + 1):
        seqs.append(user_scaled[i : i + seq_len])
    return np.array(seqs, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
def analyze_user_history(
    lstm_model: Model,
    iso_model:  IsolationForest,
    scaler,
    user_df: pd.DataFrame,
    seq_len: int       = SEQ_LEN,
    lstm_threshold: float   = 0.45,
    iso_threshold:  float   = 0.55,
    hybrid_threshold: float = 0.48,
    w_lstm: float  = 0.60,
    w_iso:  float  = 0.40,
) -> dict:
    """
    Analyse a single user's transaction history end-to-end.

    Parameters
    ----------
    lstm_model       : Trained BiLSTM model
    iso_model        : Trained Isolation Forest
    scaler           : Fitted StandardScaler
    user_df          : Raw transactions for ONE user, sorted by timestamp
    lstm_threshold   : Per-window LSTM fraud probability cut-off
    iso_threshold    : Per-transaction IsoForest anomaly score cut-off
    hybrid_threshold : Combined score cut-off for final alert
    w_lstm / w_iso   : Ensemble weights (must sum to 1.0)

    Returns
    -------
    UserFraudReport dict (see module docstring)
    """
    user_id = user_df["user_id"].iloc[0]
    user_df = user_df.sort_values("timestamp").reset_index(drop=True)
    n_txns  = len(user_df)

    # ── Scale features ────────────────────────────────────────────────────
    X_flat = scaler.transform(
        user_df[FEATURE_COLS].values.astype(np.float32)
    )

    # ── LSTM : sliding window scores ──────────────────────────────────────
    seqs = _build_user_sequences(X_flat, seq_len)
    if len(seqs) > 0:
        lstm_window_scores = lstm_model.predict(seqs, verbose=0, batch_size=64).flatten()
    else:
        lstm_window_scores = np.zeros(1)

    # Map each transaction to the MAXIMUM LSTM window score that included it
    txn_lstm_scores = np.zeros(n_txns, dtype=np.float32)
    for win_idx, score in enumerate(lstm_window_scores):
        start = win_idx
        end   = win_idx + seq_len
        txn_lstm_scores[start:end] = np.maximum(txn_lstm_scores[start:end], score)

    # ── IsoForest : per-transaction scores ────────────────────────────────
    iso_scores, iso_flags = score_user_transactions(iso_model, X_flat, iso_threshold)

    # ── Hybrid fusion ─────────────────────────────────────────────────────
    hybrid_scores = w_lstm * txn_lstm_scores + w_iso * iso_scores.astype(np.float32)
    hybrid_flags  = (hybrid_scores >= hybrid_threshold).astype(int)

    # ── Identify flagged transactions ─────────────────────────────────────
    flagged_indices = np.where(hybrid_flags == 1)[0]
    flagged_txns = []
    for idx in flagged_indices:
        row = user_df.iloc[idx]
        flagged_txns.append({
            "txn_id":            row.get("txn_id", f"TXN_{idx}"),
            "timestamp":         str(row["timestamp"]),
            "amount":            round(float(row["amount"]), 2),
            "merchant_category": row["merchant_category"],
            "hour":              int(row["hour"]),
            "day_of_week":       int(row["day_of_week"]),
            "region":            row["region"],
            "is_foreign":        int(row["is_foreign"]),
            "txn_gap_min":       round(float(row.get("txn_gap_min", 0)), 1),
            "rolling_5_amt":     round(float(row.get("rolling_5_amt", 0)), 2),
            "amt_vs_rolling5":   round(float(row.get("amt_vs_rolling5", 0)), 3),
            "lstm_score":        round(float(txn_lstm_scores[idx]), 4),
            "iso_score":         round(float(iso_scores[idx]), 4),
            "hybrid_score":      round(float(hybrid_scores[idx]), 4),
            "true_label":        int(row["Class"]),
            "fraud_type":        row.get("fraud_type", "unknown"),
        })

    # ── Ground truth ──────────────────────────────────────────────────────
    is_true_fraud     = bool(user_df["Class"].any())
    fraud_types_found = list(user_df[user_df["Class"] == 1]["fraud_type"].unique())
    alert_triggered   = len(flagged_txns) > 0

    return {
        "user_id":           user_id,
        "total_txns":        n_txns,
        "flagged_txns":      flagged_txns,
        "n_flagged":         len(flagged_txns),
        "max_lstm_score":    round(float(txn_lstm_scores.max()), 4),
        "max_iso_score":     round(float(iso_scores.max()), 4),
        "max_hybrid_score":  round(float(hybrid_scores.max()), 4),
        "fraud_types_found": fraud_types_found,
        "is_true_fraud":     is_true_fraud,
        "alert_triggered":   alert_triggered,
        "lstm_window_scores": lstm_window_scores.tolist(),
        "hybrid_txn_scores":  hybrid_scores.tolist(),
        "timestamps":         user_df["timestamp"].astype(str).tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
def analyze_all_users(
    lstm_model,
    iso_model,
    scaler,
    test_df: pd.DataFrame,
    **kwargs,
) -> list[dict]:
    """
    Run analysis for every user in the test set.

    Returns a list of UserFraudReport dicts sorted by max_hybrid_score desc.
    """
    users   = test_df["user_id"].unique()
    reports = []

    print(f"\n[Analyzer]  Scanning {len(users)} users ...")
    for i, uid in enumerate(users, 1):
        user_df = test_df[test_df["user_id"] == uid].copy()
        report  = analyze_user_history(
            lstm_model, iso_model, scaler, user_df, **kwargs
        )
        reports.append(report)
        if i % 10 == 0:
            print(f"  {i}/{len(users)} users processed …", end="\r")

    reports.sort(key=lambda r: r["max_hybrid_score"], reverse=True)
    n_alerted    = sum(1 for r in reports if r["alert_triggered"])
    n_true_fraud = sum(1 for r in reports if r["is_true_fraud"])
    print(f"\n[Analyzer]  Users scanned        : {len(users)}")
    print(f"[Analyzer]  Alerts triggered      : {n_alerted}")
    print(f"[Analyzer]  True fraud users      : {n_true_fraud}")
    return reports
