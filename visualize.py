"""
visualize.py
════════════
All evaluation and diagnostic plots saved to outputs/.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import os

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")


# ─────────────────────────────────────────────────────────────────────────────
def plot_training_history(history: dict, path: str = f"{OUT}/01_lstm_training.png"):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle("LSTM Training History", fontsize=14, fontweight="bold")

    pairs = [("loss","Loss"),("auc","AUC"),("recall","Recall"),("precision","Precision")]
    for ax, (key, title) in zip(axes, pairs):
        ax.plot(history[key],            label="Train", color="#4F8EF7", lw=2)
        ax.plot(history[f"val_{key}"],   label="Val",   color="#F76B4F", lw=2, ls="--")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[Plot]  {path}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_roc_pr(y_true, model_results: dict, path: str = f"{OUT}/02_roc_pr.png"):
    """model_results = { 'ModelName': (probs, preds), … }"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ROC & Precision-Recall Curves", fontsize=14, fontweight="bold")

    colors = ["#4F8EF7", "#F76B4F", "#2DCE89", "#F7A24F"]
    for (name, (probs, _)), color in zip(model_results.items(), colors):
        fpr, tpr, _  = roc_curve(y_true, probs)
        ax1.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.3f})", color=color, lw=2)

        prec, rec, _ = precision_recall_curve(y_true, probs)
        ax2.plot(rec, prec, label=f"{name} (AP={auc(rec,prec):.3f})", color=color, lw=2)

    ax1.plot([0,1],[0,1],"k--",alpha=0.4); ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
    ax1.set_title("ROC Curve"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[Plot]  {path}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrices(y_true, model_results: dict,
                             path: str = f"{OUT}/03_confusion_matrices.png"):
    n = len(model_results)
    fig, axes = plt.subplots(1, n, figsize=(n * 5, 4))
    if n == 1: axes = [axes]
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")

    cmaps = ["Blues","Oranges","Greens","Purples"]
    for ax, (name, (_, preds)), cmap in zip(axes, model_results.items(), cmaps):
        cm = confusion_matrix(y_true, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                    xticklabels=["Legit","Fraud"],
                    yticklabels=["Legit","Fraud"])
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[Plot]  {path}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_score_distribution(y_true, hybrid_probs: np.ndarray,
                             path: str = f"{OUT}/04_score_distribution.png"):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(hybrid_probs[y_true==0], bins=60, alpha=0.6,
            color="#4F8EF7", label="Legitimate", density=True)
    ax.hist(hybrid_probs[y_true==1], bins=60, alpha=0.75,
            color="#F76B4F", label="Fraudulent", density=True)
    ax.axvline(0.48, color="black", ls="--", lw=1.5, label="Threshold = 0.48")

    ax.set_title("Hybrid Score Distribution by Class",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Hybrid Fraud Score"); ax.set_ylabel("Density")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[Plot]  {path}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_metrics_summary(metrics_dict: dict, path: str = f"{OUT}/05_metrics_summary.png"):
    models   = list(metrics_dict.keys())
    roc_vals = [metrics_dict[m]["roc_auc"]   for m in models]
    pr_vals  = [metrics_dict[m]["pr_auc"]    for m in models]
    f1_vals  = [metrics_dict[m]["f1"]        for m in models]
    rec_vals = [metrics_dict[m]["recall"]    for m in models]

    x, w = np.arange(len(models)), 0.20
    fig, ax = plt.subplots(figsize=(11, 5))

    b1 = ax.bar(x - 1.5*w, roc_vals, w, label="ROC-AUC",  color="#4F8EF7")
    b2 = ax.bar(x - 0.5*w, pr_vals,  w, label="PR-AUC",   color="#F76B4F")
    b3 = ax.bar(x + 0.5*w, f1_vals,  w, label="F1 Score", color="#2DCE89")
    b4 = ax.bar(x + 1.5*w, rec_vals, w, label="Recall",   color="#F7A24F")

    for bars in [b1, b2, b3, b4]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Key Metrics", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[Plot]  {path}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_user_timeline(report: dict, path: str | None = None):
    """
    Plot a single user's transaction history with fraud score overlay.
    Saves to outputs/user_timelines/<user_id>.png
    """
    uid      = report["user_id"]
    scores   = np.array(report["hybrid_txn_scores"])
    ts_idx   = np.arange(len(scores))

    out_dir = f"{OUT}/user_timelines"
    os.makedirs(out_dir, exist_ok=True)
    save_to = path or f"{out_dir}/{uid}.png"

    flagged_idx = [i for i, s in enumerate(scores) if s >= 0.48]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(f"Transaction History — {uid}  "
                 f"({'⚠ FRAUD USER' if report['is_true_fraud'] else 'Legitimate'})",
                 fontsize=13, fontweight="bold")

    # Hybrid score over time
    ax1.plot(ts_idx, scores, color="#4F8EF7", lw=1.5, label="Hybrid Score")
    ax1.axhline(0.48, color="red", ls="--", lw=1, label="Threshold")
    if flagged_idx:
        ax1.scatter(flagged_idx, scores[flagged_idx], color="#F76B4F",
                    zorder=5, s=60, label="Alert")
    ax1.set_ylabel("Fraud Score"); ax1.set_ylim(0, 1.05)
    ax1.legend(loc="upper left"); ax1.grid(alpha=0.3)

    # LSTM vs IsoForest breakdown
    lstm_s = np.array(report["lstm_window_scores"])
    # lstm_s might be shorter than scores; pad left with its first value
    if len(lstm_s) < len(scores):
        lstm_s = np.concatenate([np.full(len(scores)-len(lstm_s), lstm_s[0] if len(lstm_s)>0 else 0), lstm_s])

    ax2.fill_between(ts_idx, lstm_s[:len(scores)], alpha=0.4,
                     color="#4F8EF7", label="LSTM (window)")
    ax2.fill_between(ts_idx, scores,  alpha=0.3,
                     color="#F76B4F", label="Hybrid")
    ax2.axhline(0.48, color="red", ls="--", lw=1)
    ax2.set_xlabel("Transaction Index"); ax2.set_ylabel("Score")
    ax2.legend(loc="upper left"); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_to, dpi=130, bbox_inches="tight"); plt.close()


def plot_top_user_timelines(reports: list[dict], top_n: int = 6):
    """Save timeline plots for the top-N highest-risk users."""
    top = sorted(reports, key=lambda r: r["max_hybrid_score"], reverse=True)[:top_n]
    for r in top:
        plot_user_timeline(r)
    print(f"[Plot]  User timelines → {OUT}/user_timelines/")
