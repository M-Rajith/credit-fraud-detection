"""
evaluate.py
════════════
Model evaluation: computes and prints all fraud-detection metrics.

Metrics reported
────────────────
• ROC-AUC          – overall discrimination
• PR-AUC           – precision-recall area (best for imbalanced data)
• F1 Score         – harmonic mean of precision & recall
• Recall           – fraction of actual frauds caught   (critical!)
• Precision        – fraction of alerts that are real fraud
• Confusion Matrix – TP, FP, TN, FN counts
• User-level detection rate – fraction of fraud users correctly flagged
"""

import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, f1_score,
)


def evaluate_model(
    y_true: np.ndarray,
    probs:  np.ndarray,
    preds:  np.ndarray,
    name:   str = "Model",
) -> dict:
    roc_auc = roc_auc_score(y_true, probs)
    pr_auc  = average_precision_score(y_true, probs)
    f1      = f1_score(y_true, preds, zero_division=0)
    cm      = confusion_matrix(y_true, preds)

    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    recall    = tp / (tp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)

    print(f"\n{'═'*58}")
    print(f"  {name} — Evaluation")
    print(f"{'═'*58}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print(f"  PR-AUC   : {pr_auc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  Recall   : {recall:.4f}  (fraud caught / all fraud)")
    print(f"  Precision: {precision:.4f}")
    print(f"\n  Confusion Matrix")
    print(f"  {'':12}  Pred-Legit  Pred-Fraud")
    print(f"  {'True-Legit':12}  {tn:10d}  {fp:10d}")
    print(f"  {'True-Fraud':12}  {fn:10d}  {tp:10d}")
    print(f"\n  Classification Report")
    print(classification_report(y_true, preds,
                                target_names=["Legitimate", "Fraud"],
                                zero_division=0, digits=4))

    return dict(roc_auc=roc_auc, pr_auc=pr_auc, f1=f1,
                recall=recall, precision=precision,
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


def evaluate_user_level(reports: list[dict]) -> dict:
    """
    User-level detection: did we flag any transaction for a fraud user?
    Also measures false-positive rate for legitimate users.
    """
    tp = sum(1 for r in reports if r["is_true_fraud"]     and r["alert_triggered"])
    fn = sum(1 for r in reports if r["is_true_fraud"]     and not r["alert_triggered"])
    fp = sum(1 for r in reports if not r["is_true_fraud"] and r["alert_triggered"])
    tn = sum(1 for r in reports if not r["is_true_fraud"] and not r["alert_triggered"])

    user_recall    = tp / (tp + fn + 1e-9)
    user_precision = tp / (tp + fp + 1e-9)
    user_f1        = 2 * user_precision * user_recall / (user_precision + user_recall + 1e-9)

    print(f"\n{'═'*58}")
    print("  USER-LEVEL Detection Results")
    print(f"{'═'*58}")
    print(f"  Fraud users caught (Recall)   : {tp}/{tp+fn}  ({user_recall*100:.1f}%)")
    print(f"  False alarms (FP rate)        : {fp}/{fp+tn}  ({fp/(fp+tn+1e-9)*100:.1f}%)")
    print(f"  User-level Precision          : {user_precision:.4f}")
    print(f"  User-level F1                 : {user_f1:.4f}")
    print(f"{'═'*58}")

    return dict(user_recall=user_recall, user_precision=user_precision,
                user_f1=user_f1, tp=tp, fp=fp, tn=tn, fn=fn)
