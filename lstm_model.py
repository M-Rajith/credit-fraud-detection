"""
lstm_model.py
═════════════
Bidirectional LSTM trained on per-user transaction sequences.

The model learns what a NORMAL sequence of transactions looks like
for a given user's behavioural window. Any sequence that deviates
significantly from the learned pattern is flagged as suspicious.

Architecture
────────────
Input  : (batch, SEQ_LEN, n_features)
         Each row = one transaction; each column = one feature.
         The sequence captures the temporal pattern of behaviour.

Layers
──────
  BiLSTM(64, return_sequences=True)
  → Dropout(0.3)
  → BiLSTM(32, return_sequences=False)
  → Dropout(0.3)
  → Dense(32, ReLU)
  → BatchNorm
  → Dropout(0.2)
  → Dense(1, Sigmoid)           ← fraud probability

Output : probability that the sequence window contains fraud
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks, regularizers
from sklearn.utils.class_weight import compute_class_weight


# ─────────────────────────────────────────────────────────────────────────────
def build_lstm(seq_len: int, n_features: int, lr: float = 1e-3) -> Model:
    """
    Build and compile the BiLSTM fraud detection model.

    Parameters
    ----------
    seq_len    : Number of transactions per sequence window
    n_features : Number of features per transaction
    lr         : Adam learning rate
    """
    inp = layers.Input(shape=(seq_len, n_features), name="txn_sequence")

    # ── Recurrent layers ──────────────────────────────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True,
                    kernel_regularizer=regularizers.l2(1e-4)),
        name="bilstm_1"
    )(inp)
    x = layers.Dropout(0.3, name="drop_1")(x)

    x = layers.Bidirectional(
        layers.LSTM(32, return_sequences=False,
                    kernel_regularizer=regularizers.l2(1e-4)),
        name="bilstm_2"
    )(x)
    x = layers.Dropout(0.3, name="drop_2")(x)

    # ── Dense head ────────────────────────────────────────────────────────
    x = layers.Dense(32, activation="relu", name="dense_1")(x)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(0.2, name="drop_3")(x)

    out = layers.Dense(1, activation="sigmoid", name="fraud_prob")(x)

    model = Model(inputs=inp, outputs=out, name="BiLSTM_PatternDetector")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
def train_lstm(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 40,
    batch_size: int = 128,
    model_path: str = "outputs/lstm_model.keras",
) -> dict:
    """
    Fit the LSTM with early stopping, LR scheduling and class-weight balancing.

    Returns the Keras history dict.
    """
    classes   = np.unique(y_train)
    cw_vals   = compute_class_weight("balanced", classes=classes, y=y_train)
    cw        = dict(zip(classes.tolist(), cw_vals.tolist()))
    print(f"[LSTM]  Class weights → {cw}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_auc", patience=7,
            restore_best_weights=True, mode="max"
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-6, verbose=1
        ),
        callbacks.ModelCheckpoint(
            model_path, monitor="val_auc",
            save_best_only=True, mode="max", verbose=0
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=cw,
        callbacks=cb_list,
        verbose=1,
    )
    print(f"[LSTM]  Best model saved → {model_path}")
    return history.history


# ─────────────────────────────────────────────────────────────────────────────
def predict_lstm(
    model: Model,
    X: np.ndarray,
    threshold: float = 0.50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Score transaction sequences.

    Returns
    -------
    probs : fraud probability per sequence  (float32 array)
    preds : binary label 0/1               (int32 array)
    """
    probs = model.predict(X, verbose=0, batch_size=256).flatten().astype(np.float32)
    preds = (probs >= threshold).astype(np.int32)
    return probs, preds


# ─────────────────────────────────────────────────────────────────────────────
def score_user_history(
    model: Model,
    user_sequences: np.ndarray,
    threshold: float = 0.50,
) -> tuple[np.ndarray, bool]:
    """
    Score ALL windows for a single user and return per-window probabilities
    plus a flag indicating whether any window exceeded the threshold.

    Parameters
    ----------
    user_sequences : (n_windows, seq_len, n_features)
    threshold      : fraud probability cut-off

    Returns
    -------
    window_probs   : fraud probability for each window
    is_flagged     : True if any window was classified as fraud
    """
    if len(user_sequences) == 0:
        return np.array([]), False
    probs = model.predict(user_sequences, verbose=0, batch_size=64).flatten()
    return probs, bool((probs >= threshold).any())
