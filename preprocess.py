"""
preprocess.py
═════════════
Turns raw per-user transaction history into model-ready tensors.

Pipeline
────────
1. Select numerical feature columns
2. StandardScale  (fit on train, transform test)
3. Build FIXED-LENGTH sequences per user
   • Each sequence = the last SEQ_LEN transactions of one user
   • Label = 1 if ANY transaction in the window is fraudulent
   • This teaches the LSTM to recognise *behavioural drift* over time
4. Return 2-D flat arrays for Isolation Forest (per-transaction level)

Feature columns used
────────────────────
amount, hour, day_of_week, is_foreign, txn_gap_min,
rolling_3_amt, rolling_5_amt, rolling_3_std, rolling_5_std,
amt_vs_rolling5, cat_code, reg_code
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib, os

SEQ_LEN = 10          # transactions per LSTM window

FEATURE_COLS = [
    "amount", "hour", "day_of_week", "is_foreign",
    "txn_gap_min", "rolling_3_amt", "rolling_5_amt",
    "rolling_3_std", "rolling_5_std", "amt_vs_rolling5",
    "cat_code", "reg_code",
]


# ─────────────────────────────────────────────────────────────────────────────
def _build_user_sequences(
    user_df: pd.DataFrame,
    X_scaled: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window of `seq_len` across one user's sorted transactions.

    Returns
    -------
    seqs   : (n_windows, seq_len, n_features)
    labels : (n_windows,)  1 if window contains a fraud, else 0
    """
    y = user_df["Class"].values
    seqs, labels = [], []

    for i in range(len(X_scaled) - seq_len + 1):
        seqs.append(X_scaled[i : i + seq_len])
        labels.append(int(y[i : i + seq_len].max()))   # window-level label

    return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
def build_pipeline(
    df: pd.DataFrame,
    seq_len: int = SEQ_LEN,
    test_size: float = 0.20,
    scaler_path: str = "outputs/scaler.pkl",
) -> dict:
    """
    Full preprocessing pipeline.

    Splits at the USER level (all transactions of a user stay together),
    preventing data leakage between train and test sets.

    Returns
    -------
    dict with:
        X_train_lstm   (n, seq_len, n_features)   – for LSTM training
        X_test_lstm    (n, seq_len, n_features)   – for LSTM evaluation
        X_train_flat   (n, n_features)            – for IsoForest training
        X_test_flat    (n, n_features)            – for IsoForest evaluation
        y_train / y_test                          – sequence-level labels
        y_train_txn / y_test_txn                  – transaction-level labels
        feature_cols                              – list of feature names
        scaler                                    – fitted StandardScaler
        test_user_ids                             – user IDs in test split
        test_df                                   – raw test transactions
    """
    # ── User-level train/test split ───────────────────────────────────────
    all_users = df["user_id"].unique()
    train_users, test_users = train_test_split(
        all_users, test_size=test_size, random_state=42
    )
    train_df = df[df["user_id"].isin(train_users)].copy()
    test_df  = df[df["user_id"].isin(test_users)].copy()

    print(f"[Preprocess]  Train users : {len(train_users):,}  "
          f"| txns : {len(train_df):,}")
    print(f"[Preprocess]  Test  users : {len(test_users):,}  "
          f"| txns : {len(test_df):,}")

    # ── Scale features ────────────────────────────────────────────────────
    scaler    = StandardScaler()
    X_tr_flat = scaler.fit_transform(train_df[FEATURE_COLS].values.astype(np.float32))
    X_te_flat = scaler.transform(test_df[FEATURE_COLS].values.astype(np.float32))

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"[Preprocess]  Scaler saved → {scaler_path}")

    # ── Build per-user sequences ──────────────────────────────────────────
    def _sequences_for_split(split_df, scaled_X):
        all_seqs, all_labels = [], []
        offset = 0
        for uid in split_df["user_id"].unique():
            mask   = split_df["user_id"] == uid
            n_txns = mask.sum()
            if n_txns < seq_len:
                offset += n_txns
                continue
            u_scaled = scaled_X[offset : offset + n_txns]
            u_df     = split_df[mask]
            seqs, labels = _build_user_sequences(u_df, u_scaled, seq_len)
            all_seqs.append(seqs)
            all_labels.append(labels)
            offset += n_txns
        return (np.concatenate(all_seqs,  axis=0),
                np.concatenate(all_labels, axis=0))

    X_tr_lstm, y_train = _sequences_for_split(train_df, X_tr_flat)
    X_te_lstm, y_test  = _sequences_for_split(test_df,  X_te_flat)

    print(f"[Preprocess]  LSTM train sequences : {len(y_train):,}  "
          f"(fraud={y_train.sum():,})")
    print(f"[Preprocess]  LSTM test  sequences : {len(y_test):,}  "
          f"(fraud={y_test.sum():,})")

    return {
        "X_train_lstm":  X_tr_lstm,
        "X_test_lstm":   X_te_lstm,
        "X_train_flat":  X_tr_flat,
        "X_test_flat":   X_te_flat,
        "y_train":       y_train,
        "y_test":        y_test,
        "y_train_txn":   train_df["Class"].values,
        "y_test_txn":    test_df["Class"].values,
        "feature_cols":  FEATURE_COLS,
        "scaler":        scaler,
        "train_df":      train_df,
        "test_df":       test_df,
        "train_users":   train_users,
        "test_users":    test_users,
    }
