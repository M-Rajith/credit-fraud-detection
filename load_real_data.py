"""
load_real_data.py
=================
Adapts the Sparkov Credit Card Transactions dataset into the exact format
expected by the existing pipeline (preprocess.py, lstm_model.py, ...).

Dataset
-------
  Name    : Credit Card Transactions Fraud Detection Dataset (Sparkov)
  Kaggle  : https://www.kaggle.com/datasets/kartik2112/fraud-detection
  Files   : fraudTrain.csv  +  fraudTest.csv  (combine both)
  Size    : ~1.3 M rows  .  1,000 customers  .  800 merchants  .  2 years
"""

import os
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  Geo-distance helper
# ─────────────────────────────────────────────────────────────────────────────

def _compute_is_foreign(df: pd.DataFrame, distance_km: float = 50.0) -> pd.Series:
    R = 6371.0
    lat1 = np.radians(df["lat"].values)
    lat2 = np.radians(df["merch_lat"].values)
    lon1 = np.radians(df["long"].values)
    lon2 = np.radians(df["merch_long"].values)
    dphi = lat2 - lat1
    dlam = lon2 - lon1
    a = np.sin(dphi / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlam / 2)**2
    dist = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return (dist > distance_km).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
#  Rolling / sequence features
# ─────────────────────────────────────────────────────────────────────────────

def _add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    df["txn_gap_min"] = (
        df.groupby("user_id")["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(60)
        .fillna(0)
    )

    for w in [3, 5]:
        df[f"rolling_{w}_amt"] = (
            df.groupby("user_id")["amount"]
            .transform(lambda x: x.rolling(w, min_periods=1).mean())
        )
        df[f"rolling_{w}_std"] = (
            df.groupby("user_id")["amount"]
            .transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
        )

    df["amt_vs_rolling5"] = df["amount"] / (df["rolling_5_amt"] + 1e-6)
    df["cat_code"] = df["merchant_category"].astype("category").cat.codes
    df["reg_code"] = df["region"].astype("category").cat.codes

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Main loader
# ─────────────────────────────────────────────────────────────────────────────

def load_sparkov(
    csv_path: str,
    min_txns: int = 15,
    max_users: int = 1000,
    foreign_distance_km: float = 50.0,
    save_path: str = "data/transactions.csv",
    random_seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)

    print(f"[RealData]  Reading  {csv_path} ...")
    raw = pd.read_csv(csv_path, low_memory=False)
    print(f"[RealData]  Raw rows : {len(raw):,}  |  columns : {list(raw.columns)}")

    # ── Validate columns ──────────────────────────────────────────────────
    required = {
        "trans_date_trans_time", "cc_num", "category", "amt",
        "lat", "long", "merch_lat", "merch_long", "state", "is_fraud"
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required Sparkov columns: {missing}\n"
            "Make sure you are using fraudTrain.csv / fraudTest.csv from\n"
            "https://www.kaggle.com/datasets/kartik2112/fraud-detection"
        )

    # ── Column mapping ────────────────────────────────────────────────────
    raw["user_id"]           = raw["cc_num"].astype(str)
    raw["timestamp"]         = pd.to_datetime(raw["trans_date_trans_time"])
    raw["amount"]            = raw["amt"].astype(float)
    raw["merchant_category"] = raw["category"].str.strip()
    raw["hour"]              = raw["timestamp"].dt.hour
    raw["day_of_week"]       = raw["timestamp"].dt.dayofweek
    raw["region"]            = raw["state"].str.strip().str.upper()
    raw["is_foreign"]        = _compute_is_foreign(raw, distance_km=foreign_distance_km)
    raw["Class"]             = raw["is_fraud"].astype(int)

    # fraud_type: required by history_analyzer.py
    # For legitimate transactions it is "none".
    # For fraud transactions it mirrors the merchant category so reports
    # show exactly what kind of transaction the fraud occurred in.
    raw["fraud_type"] = "none"
    raw.loc[raw["Class"] == 1, "fraud_type"] = (
        raw.loc[raw["Class"] == 1, "category"].str.strip()
    )

    if "trans_num" in raw.columns:
        raw["txn_id"] = raw["trans_num"].astype(str)
    else:
        raw["txn_id"] = [f"TXN_{i:07d}" for i in range(len(raw))]

    # ── Filter users with enough history ─────────────────────────────────
    txn_counts  = raw["user_id"].value_counts()
    valid_users = txn_counts[txn_counts >= min_txns].index
    raw = raw[raw["user_id"].isin(valid_users)].copy()
    print(f"[RealData]  Users with >= {min_txns} txns : {len(valid_users):,}")

    # ── Fraud / legitimate user split ─────────────────────────────────────
    fraud_users = raw[raw["Class"] == 1]["user_id"].unique()
    legit_users = raw[raw["Class"] == 0]["user_id"].unique()
    print(f"[RealData]  Fraud users : {len(fraud_users):,}  |  Legit users : {len(legit_users):,}")

    max_legit = max_users - len(fraud_users)
    if len(legit_users) > max_legit and max_legit > 0:
        legit_users = rng.choice(legit_users, size=max_legit, replace=False)
        print(f"[RealData]  Capped to {max_users:,} total users")

    keep_users = set(fraud_users) | set(legit_users)

    # ── Build final dataframe (fraud_type already on raw) ─────────────────
    df = raw[raw["user_id"].isin(keep_users)].copy()

    keep_cols = [
        "user_id", "txn_id", "timestamp", "amount",
        "merchant_category", "hour", "day_of_week",
        "region", "is_foreign", "Class", "fraud_type",
    ]
    df = df[keep_cols].copy()

    # ── Sequence features ─────────────────────────────────────────────────
    print(f"[RealData]  Computing sequence features ...")
    df = _add_sequence_features(df)

    # ── Summary ───────────────────────────────────────────────────────────
    total         = len(df)
    n_fraud       = int(df["Class"].sum())
    n_users       = df["user_id"].nunique()
    n_fraud_users = df[df["Class"] == 1]["user_id"].nunique()

    print(f"[RealData]  Users               : {n_users:,}")
    print(f"[RealData]  Total Transactions  : {total:,}")
    print(f"[RealData]  Fraudulent txns     : {n_fraud:,}  ({n_fraud / total * 100:.2f}%)")
    print(f"[RealData]  Users with fraud    : {n_fraud_users:,}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[RealData]  Saved --> {save_path}")

    return df