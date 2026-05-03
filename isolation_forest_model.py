"""
isolation_forest_model.py
══════════════════════════
Isolation Forest for per-transaction anomaly detection within user histories.

Role in the pipeline
────────────────────
• Trained ONLY on legitimate transactions → learns the normal boundary.
• At inference time every transaction in a user's history is scored.
• Transactions whose anomaly score exceeds a threshold are flagged.
• The IsoForest score is combined with the LSTM sequence score in the
  hybrid detector for a final fraud decision.

Why use both LSTM + IsoForest?
──────────────────────────────
• LSTM    : catches PATTERN anomalies (sequence-level drift over time)
• IsoForest: catches POINT anomalies  (a single wildly unusual transaction)
  Together they cover both slow behavioural drift and sudden one-off attacks.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
import joblib, os


# ─────────────────────────────────────────────────────────────────────────────
def train_isolation_forest(
    X_train: np.ndarray,
    y_train_txn: np.ndarray,
    contamination: float = 0.05,
    n_estimators: int = 300,
    model_path: str = "outputs/isolation_forest.pkl",
) -> IsolationForest:
    """
    Fit Isolation Forest exclusively on NORMAL transactions.

    Training on only legitimate data means the model builds a tight
    boundary around normal behaviour; fraud falls outside that boundary.

    Parameters
    ----------
    X_train      : All training transaction features (scaled)
    y_train_txn  : Transaction-level labels (0/1) — used to filter
    contamination: Expected fraction of outliers in new data
    n_estimators : Number of isolation trees
    model_path   : Where to persist the fitted model
    """
    X_legit = X_train[y_train_txn == 0]
    print(f"[IsoForest]  Training on {len(X_legit):,} legitimate transactions "
          f"(n_estimators={n_estimators}, contamination={contamination})")

    iso = IsolationForest(
        n_estimators  = n_estimators,
        contamination = contamination,
        max_samples   = "auto",
        random_state  = 42,
        n_jobs        = -1,
    )
    iso.fit(X_legit)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(iso, model_path)
    print(f"[IsoForest]  Saved → {model_path}")
    return iso


# ─────────────────────────────────────────────────────────────────────────────
def _normalise_scores(raw: np.ndarray) -> np.ndarray:
    """
    Convert IsolationForest.score_samples() output to [0,1] where
    1 = most anomalous.

    score_samples returns negative values; more negative = more anomalous.
    """
    inverted = -raw
    return (inverted - inverted.min()) / (inverted.max() - inverted.min() + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
def predict_isolation_forest(
    iso: IsolationForest,
    X: np.ndarray,
    threshold: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Score a batch of transactions.

    Returns
    -------
    probs : anomaly probability per transaction (float32, range [0,1])
    preds : binary flag  1 = anomalous
    """
    raw   = iso.score_samples(X)
    probs = _normalise_scores(raw).astype(np.float32)
    preds = (probs >= threshold).astype(np.int32)
    return probs, preds


# ─────────────────────────────────────────────────────────────────────────────
def score_user_transactions(
    iso: IsolationForest,
    user_X: np.ndarray,
    threshold: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Score every individual transaction in a user's history.

    Returns
    -------
    txn_scores : anomaly probability for each transaction
    txn_flags  : 1 if transaction is anomalous
    """
    if len(user_X) == 0:
        return np.array([]), np.array([])
    return predict_isolation_forest(iso, user_X, threshold)
