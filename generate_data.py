"""
generate_data.py
════════════════
Generates a realistic per-user transaction history dataset.

Each cardholder has a unique spending profile:
  • Preferred merchant categories  (groceries, fuel, dining, online, travel …)
  • Typical spend range            (low / medium / high income tier)
  • Active hours                   (morning shopper vs night owl)
  • Home region                    (location fingerprint)

Fraud patterns injected per user
  1. Foreign transaction  – purchase far from home region
  2. Category anomaly     – merchant type never used before
  3. Amount spike         – single transaction >> user's 99th percentile
  4. Velocity burst       – 5+ transactions within 30 minutes
  5. Odd-hour access      – purchase at 2-4 AM for a day-only user

Output columns
──────────────
user_id, txn_id, timestamp, amount, merchant_category,
hour, day_of_week, region, is_foreign, txn_gap_minutes,
rolling_3_amt, rolling_5_amt, rolling_3_std, Class (0/1)
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

CATEGORIES   = ["groceries", "fuel", "dining", "online", "travel",
                 "entertainment", "pharmacy", "utilities", "atm", "luxury"]
REGIONS      = ["north", "south", "east", "west", "central"]
FRAUD_LABELS = {
    "foreign_txn": 1, "category_anomaly": 1,
    "amount_spike": 1, "velocity_burst": 1, "odd_hour": 1,
}

# ─────────────────────────────────────────────────────────────────────────────
class CardholderProfile:
    """Encodes a synthetic user's normal spending behaviour."""

    def __init__(self, user_id: int):
        self.user_id      = f"USER_{user_id:04d}"
        tier              = RNG.choice(["low", "mid", "high"], p=[0.5, 0.35, 0.15])
        self.tier         = tier
        self.amount_mu    = {"low": 30, "mid": 80, "high": 250}[tier]
        self.amount_sigma = {"low": 20, "mid": 60, "high": 180}[tier]

        # 2-4 preferred categories
        n_cats            = RNG.integers(2, 5)
        self.fav_cats     = list(RNG.choice(CATEGORIES, size=n_cats, replace=False))
        self.home_region  = RNG.choice(REGIONS)

        # Active hour window
        profile           = RNG.choice(["morning", "daytime", "evening"])
        self.hour_mu      = {"morning": 9, "daytime": 13, "evening": 19}[profile]
        self.hour_std     = 2.5


# ─────────────────────────────────────────────────────────────────────────────
def _normal_txns(profile: CardholderProfile, n: int,
                 start_ts: datetime) -> list[dict]:
    records = []
    ts = start_ts
    for _ in range(n):
        gap = RNG.exponential(scale=360)          # avg 6 h between transactions
        ts += timedelta(minutes=gap)
        hour = int(np.clip(RNG.normal(profile.hour_mu, profile.hour_std), 0, 23))
        amount = max(1.0, RNG.normal(profile.amount_mu, profile.amount_sigma))
        records.append({
            "user_id":           profile.user_id,
            "timestamp":         ts,
            "amount":            round(amount, 2),
            "merchant_category": RNG.choice(profile.fav_cats),
            "hour":              hour,
            "day_of_week":       ts.weekday(),
            "region":            profile.home_region,
            "is_foreign":        0,
            "fraud_type":        "normal",
            "Class":             0,
        })
    return records


def _inject_fraud(profile: CardholderProfile, base_ts: datetime) -> list[dict]:
    """Return 1-6 fraudulent transactions for this user."""
    records = []
    fraud_type = RNG.choice(
        ["foreign_txn", "category_anomaly", "amount_spike",
         "velocity_burst", "odd_hour"]
    )

    ts = base_ts + timedelta(minutes=int(RNG.integers(30, 300)))

    if fraud_type == "foreign_txn":
        foreign = RNG.choice([r for r in REGIONS if r != profile.home_region])
        records.append({
            "user_id": profile.user_id, "timestamp": ts,
            "amount": round(RNG.uniform(50, 500), 2),
            "merchant_category": RNG.choice(CATEGORIES),
            "hour": int(RNG.integers(8, 22)), "day_of_week": ts.weekday(),
            "region": foreign, "is_foreign": 1,
            "fraud_type": fraud_type, "Class": 1,
        })

    elif fraud_type == "category_anomaly":
        rare_cat = RNG.choice([c for c in CATEGORIES if c not in profile.fav_cats])
        records.append({
            "user_id": profile.user_id, "timestamp": ts,
            "amount": round(RNG.uniform(100, 800), 2),
            "merchant_category": rare_cat,
            "hour": profile.hour_mu, "day_of_week": ts.weekday(),
            "region": profile.home_region, "is_foreign": 0,
            "fraud_type": fraud_type, "Class": 1,
        })

    elif fraud_type == "amount_spike":
        spike = profile.amount_mu * RNG.uniform(8, 20)
        records.append({
            "user_id": profile.user_id, "timestamp": ts,
            "amount": round(spike, 2),
            "merchant_category": RNG.choice(CATEGORIES),
            "hour": profile.hour_mu, "day_of_week": ts.weekday(),
            "region": profile.home_region, "is_foreign": 0,
            "fraud_type": fraud_type, "Class": 1,
        })

    elif fraud_type == "velocity_burst":
        for i in range(RNG.integers(4, 7)):
            records.append({
                "user_id": profile.user_id,
                "timestamp": ts + timedelta(minutes=int(i * 5)),
                "amount": round(RNG.uniform(10, 60), 2),
                "merchant_category": RNG.choice(CATEGORIES),
                "hour": ts.hour, "day_of_week": ts.weekday(),
                "region": profile.home_region, "is_foreign": 0,
                "fraud_type": fraud_type, "Class": 1,
            })

    elif fraud_type == "odd_hour":
        records.append({
            "user_id": profile.user_id,
            "timestamp": ts.replace(hour=int(RNG.integers(1, 4))),
            "amount": round(RNG.uniform(50, 300), 2),
            "merchant_category": RNG.choice(CATEGORIES),
            "hour": int(RNG.integers(1, 4)), "day_of_week": ts.weekday(),
            "region": profile.home_region, "is_foreign": 0,
            "fraud_type": fraud_type, "Class": 1,
        })

    return records


# ─────────────────────────────────────────────────────────────────────────────
def _add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal & rolling features computed within each user's history.
    These are what the LSTM and IsoForest will use.
    """
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Time gap between consecutive user transactions (minutes)
    df["txn_gap_min"] = (
        df.groupby("user_id")["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(60)
        .fillna(0)
    )

    # Rolling statistics over last 3 and 5 transactions per user
    for w in [3, 5]:
        df[f"rolling_{w}_amt"] = (
            df.groupby("user_id")["amount"]
            .transform(lambda x: x.rolling(w, min_periods=1).mean())
        )
        df[f"rolling_{w}_std"] = (
            df.groupby("user_id")["amount"]
            .transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
        )

    # Ratio of current amount to user rolling mean
    df["amt_vs_rolling5"] = df["amount"] / (df["rolling_5_amt"] + 1e-6)

    # Encode categoricals
    df["cat_code"] = df["merchant_category"].astype("category").cat.codes
    df["reg_code"] = df["region"].astype("category").cat.codes

    return df


# ─────────────────────────────────────────────────────────────────────────────
def generate_dataset(
    n_users: int = 200,
    txns_per_user: int = 60,
    fraud_rate: float = 0.12,
    save_path: str = "data/transactions.csv",
) -> pd.DataFrame:
    """
    Build the full per-user transaction history dataset.

    Parameters
    ----------
    n_users        : Number of synthetic cardholders
    txns_per_user  : Avg normal transactions per user
    fraud_rate     : Fraction of users who experience at least one fraud event
    save_path      : CSV destination

    Returns
    -------
    pd.DataFrame with all transactions, sorted by user and timestamp.
    """
    start_date = datetime(2024, 1, 1)
    all_records = []

    n_fraud_users = int(n_users * fraud_rate)
    fraud_user_ids = set(RNG.choice(n_users, size=n_fraud_users, replace=False))

    for uid in range(n_users):
        profile = CardholderProfile(uid)
        n_txns  = int(RNG.integers(txns_per_user - 10, txns_per_user + 20))
        records = _normal_txns(profile, n_txns, start_date)

        if uid in fraud_user_ids:
            # Inject fraud mid-history
            inject_at = records[len(records) // 2]["timestamp"]
            fraud_records = _inject_fraud(profile, inject_at)
            records.extend(fraud_records)

        all_records.extend(records)

    df = pd.DataFrame(all_records)
    df["txn_id"] = [f"TXN_{i:07d}" for i in range(len(df))]
    df = _add_sequence_features(df)

    # Final sort
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    total   = len(df)
    n_fraud = df["Class"].sum()
    n_users_w_fraud = df[df["Class"]==1]["user_id"].nunique()
    print(f"[DataGen]  Users               : {df['user_id'].nunique():,}")
    print(f"[DataGen]  Total Transactions  : {total:,}")
    print(f"[DataGen]  Fraudulent txns     : {n_fraud:,}  ({n_fraud/total*100:.2f}%)")
    print(f"[DataGen]  Users with fraud    : {n_users_w_fraud}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[DataGen]  Saved → {save_path}")
    return df


if __name__ == "__main__":
    generate_dataset()
