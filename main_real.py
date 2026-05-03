import os, warnings, joblib
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
import numpy as np
import tensorflow as tf
from load_real_data import load_sparkov
from preprocess import build_pipeline
from lstm_model import build_lstm, train_lstm, predict_lstm
from isolation_forest_model import train_isolation_forest, predict_isolation_forest
from history_analyzer import analyze_all_users
from evaluate import evaluate_model, evaluate_user_level
from fraud_reporter import generate_text_report, generate_csv_reports, print_console_summary
from visualize import plot_training_history, plot_roc_pr, plot_confusion_matrices, plot_score_distribution, plot_metrics_summary, plot_top_user_timelines

LSTM_PATH   = "outputs/lstm_model.keras"
ISO_PATH    = "outputs/isolation_forest.pkl"
SPARKOV_CSV = "data/sparkov.csv"

def main():
    print("\n" + "="*60)
    print("   CREDIT CARD FRAUD DETECTION  [REAL DATA - SPARKOV]")
    print("   Per-User History Analysis  |  LSTM + Isolation Forest")
    print("="*60 + "\n")

    print("-- Step 1 : Loading Real Sparkov Data -------------------")
    df = load_sparkov(csv_path=SPARKOV_CSV, min_txns=15, max_users=1000,
                      foreign_distance_km=50.0, save_path="data/transactions.csv")

    print("\n-- Step 2 : Preprocessing -------------------------------")
    data = build_pipeline(df, seq_len=10, test_size=0.20)
    X_tr_lstm = data["X_train_lstm"]; X_te_lstm = data["X_test_lstm"]
    X_tr_flat = data["X_train_flat"]; X_te_flat = data["X_test_flat"]
    y_train = data["y_train"];        y_test    = data["y_test"]
    scaler  = data["scaler"];         test_df   = data["test_df"]
    seq_len = X_tr_lstm.shape[1];     n_feat    = X_tr_lstm.shape[2]

    if os.path.exists(LSTM_PATH):
        print("\n-- Step 3 : Loading saved LSTM (skipping training) ------")
        lstm = tf.keras.models.load_model(LSTM_PATH)
        lstm.summary()
        print("[LSTM]  Loaded successfully.")
    else:
        print("\n-- Step 3 : Training LSTM --------------------------------")
        lstm = build_lstm(seq_len=seq_len, n_features=n_feat, lr=1e-3)
        history = train_lstm(lstm, X_tr_lstm, y_train, X_te_lstm, y_test,
                             epochs=40, batch_size=128, model_path=LSTM_PATH)
        plot_training_history(history)

    if os.path.exists(ISO_PATH):
        print("\n-- Step 4 : Loading saved IsoForest (skipping training) -")
        iso = joblib.load(ISO_PATH)
        print("[IsoForest]  Loaded successfully.")
    else:
        print("\n-- Step 4 : Training Isolation Forest -------------------")
        iso = train_isolation_forest(X_tr_flat, data["y_train_txn"],
                                     contamination=0.05, n_estimators=300,
                                     model_path=ISO_PATH)

    print("\n-- Step 5 : Sequence-Level Model Evaluation -------------")
    lstm_probs, lstm_preds = predict_lstm(lstm, X_te_lstm, threshold=0.45)
    iso_probs,  iso_preds  = predict_isolation_forest(iso, X_te_flat, threshold=0.55)

    n_seqs    = len(lstm_probs)
    n_te_txns = len(X_te_flat)
    iso_window = np.zeros(n_seqs, dtype=np.float32)
    for i in range(n_seqs):
        end = min(i + seq_len, n_te_txns)
        iso_window[i] = iso_probs[i:end].max() if i < n_te_txns else 0.0

    hybrid_probs = 0.60 * lstm_probs + 0.40 * iso_window
    hybrid_preds = (hybrid_probs >= 0.48).astype(int)

    m_lstm   = evaluate_model(y_test, lstm_probs,   lstm_preds,   "LSTM (sequences)")
    m_iso    = evaluate_model(y_test, iso_window,   (iso_window >= 0.55).astype(int), "IsoForest (windows)")
    m_hybrid = evaluate_model(y_test, hybrid_probs, hybrid_preds, "Hybrid")

    print("\n-- Step 6 : Per-User History Analysis -------------------")
    reports = analyze_all_users(
        lstm_model=lstm, iso_model=iso, scaler=scaler, test_df=test_df,
        seq_len=seq_len, lstm_threshold=0.45, iso_threshold=0.55,
        hybrid_threshold=0.48, w_lstm=0.60, w_iso=0.40)
    user_metrics = evaluate_user_level(reports)

    print("\n-- Step 7 : Generating Reports --------------------------")
    generate_text_report(reports)
    generate_csv_reports(reports)
    print_console_summary(reports)

    print("-- Step 8 : Saving Plots --------------------------------")
    model_results = {
        "LSTM":      (lstm_probs,   lstm_preds),
        "IsoForest": (iso_window,   (iso_window >= 0.55).astype(int)),
        "Hybrid":    (hybrid_probs, hybrid_preds)
    }
    all_metrics = {"LSTM": m_lstm, "IsoForest": m_iso, "Hybrid": m_hybrid}
    plot_roc_pr(y_test, model_results)
    plot_confusion_matrices(y_test, model_results)
    plot_score_distribution(y_test, hybrid_probs)
    plot_metrics_summary(all_metrics)
    plot_top_user_timelines(reports, top_n=6)

    print("\n" + "="*60)
    print("  FINAL SUMMARY")
    print("="*60)
    for name, m in all_metrics.items():
        print(f"  {name:<14}  ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}"
              f"  F1={m['f1']:.4f}  Recall={m['recall']:.4f}")
    print(f"\n  User-Level Detection Rate : {user_metrics['user_recall']*100:.1f}%"
          f"  (F1={user_metrics['user_f1']:.4f})")
    print("="*60)
    print("  Reports --> reports/")
    print("  Plots   --> outputs/\n")

if __name__ == "__main__":
    main()
