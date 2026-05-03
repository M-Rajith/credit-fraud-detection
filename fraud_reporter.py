"""
fraud_reporter.py
══════════════════
Generates human-readable fraud alert reports from analysis results.

Outputs
───────
• reports/fraud_alerts.txt        – full text report of all flagged users
• reports/flagged_transactions.csv – every suspicious transaction with scores
• reports/user_risk_summary.csv    – one row per user with risk metrics
• Console summary table
"""

import os
import pandas as pd
from datetime import datetime

OUT = "reports"
os.makedirs(OUT, exist_ok=True)

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ─────────────────────────────────────────────────────────────────────────────
def _risk_level(score: float) -> str:
    if score >= 0.80: return "🔴 CRITICAL"
    if score >= 0.65: return "🟠 HIGH"
    if score >= 0.48: return "🟡 MEDIUM"
    return "🟢 LOW"


def _flag_reason(txn: dict) -> str:
    """Human-readable reason for flagging this transaction."""
    reasons = []
    if txn["is_foreign"]:
        reasons.append("Foreign region transaction")
    if txn["amt_vs_rolling5"] > 5:
        reasons.append(f"Amount {txn['amt_vs_rolling5']:.1f}× user average")
    if txn["hour"] in list(range(0, 5)) + [23]:
        reasons.append(f"Unusual hour ({txn['hour']:02d}:00)")
    if txn["txn_gap_min"] < 5 and txn["txn_gap_min"] > 0:
        reasons.append(f"Rapid sequence (gap {txn['txn_gap_min']:.1f} min)")
    if txn["iso_score"] > 0.70:
        reasons.append("Statistical outlier (IsoForest)")
    if txn["lstm_score"] > 0.70:
        reasons.append("Pattern deviation (LSTM)")
    return " | ".join(reasons) if reasons else "Combined score exceeded threshold"


# ─────────────────────────────────────────────────────────────────────────────
def generate_text_report(
    reports: list[dict],
    save_path: str = f"{OUT}/fraud_alerts.txt",
) -> None:
    """Write a detailed text alert report for every flagged user."""
    flagged = [r for r in reports if r["alert_triggered"]]

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("   CREDIT CARD FRAUD DETECTION — ALERT REPORT\n")
        f.write(f"   Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"  Total users analysed : {len(reports)}\n")
        f.write(f"  Users flagged        : {len(flagged)}\n")
        f.write(f"  True fraud users     : {sum(r['is_true_fraud'] for r in reports)}\n\n")
        f.write("=" * 70 + "\n\n")

        for r in flagged:
            risk = _risk_level(r["max_hybrid_score"])
            f.write(f"USER ALERT  ──  {r['user_id']}\n")
            f.write(f"  Risk Level          : {risk}\n")
            f.write(f"  Max Hybrid Score    : {r['max_hybrid_score']:.4f}\n")
            f.write(f"  Max LSTM Score      : {r['max_lstm_score']:.4f}\n")
            f.write(f"  Max IsoForest Score : {r['max_iso_score']:.4f}\n")
            f.write(f"  Total Transactions  : {r['total_txns']}\n")
            f.write(f"  Flagged Transactions: {r['n_flagged']}\n")
            f.write(f"  Ground Truth Fraud  : {'YES' if r['is_true_fraud'] else 'NO'}\n")
            if r["fraud_types_found"]:
                f.write(f"  Fraud Type(s)       : {', '.join(r['fraud_types_found'])}\n")
            f.write("\n  SUSPICIOUS TRANSACTIONS\n")
            f.write("  " + "-" * 65 + "\n")

            for txn in r["flagged_txns"]:
                day = DAY_NAMES[txn["day_of_week"]]
                f.write(f"  ▶ {txn['txn_id']}\n")
                f.write(f"    Timestamp   : {txn['timestamp']}\n")
                f.write(f"    Amount      : ${txn['amount']:,.2f}\n")
                f.write(f"    Category    : {txn['merchant_category']}\n")
                f.write(f"    Region      : {txn['region']}"
                        f"{'  ⚠ FOREIGN' if txn['is_foreign'] else ''}\n")
                f.write(f"    Time        : {day} {txn['hour']:02d}:00\n")
                f.write(f"    Txn Gap     : {txn['txn_gap_min']:.1f} min "
                        f"from previous\n")
                f.write(f"    Amt/Avg     : {txn['amt_vs_rolling5']:.2f}× "
                        f"(5-txn rolling avg: ${txn['rolling_5_amt']:.2f})\n")
                f.write(f"    LSTM Score  : {txn['lstm_score']:.4f}\n")
                f.write(f"    ISO  Score  : {txn['iso_score']:.4f}\n")
                f.write(f"    Hybrid Score: {txn['hybrid_score']:.4f}\n")
                f.write(f"    Reason      : {_flag_reason(txn)}\n")
                f.write(f"    True Label  : {'FRAUD' if txn['true_label'] else 'LEGIT'}\n")
                f.write("\n")

            f.write("=" * 70 + "\n\n")

    print(f"[Reporter]  Text report saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
def generate_csv_reports(
    reports: list[dict],
    txn_path: str     = f"{OUT}/flagged_transactions.csv",
    summary_path: str = f"{OUT}/user_risk_summary.csv",
) -> None:
    """Export flagged transactions and user-level risk summary to CSV."""

    # ── Flagged transactions ──────────────────────────────────────────────
    rows = []
    for r in reports:
        for txn in r["flagged_txns"]:
            rows.append({
                "user_id":           r["user_id"],
                **txn,
                "flag_reason":       _flag_reason(txn),
                "risk_level":        _risk_level(r["max_hybrid_score"]),
            })

    if rows:
        pd.DataFrame(rows).to_csv(txn_path, index=False)
        print(f"[Reporter]  Flagged txns CSV  → {txn_path}")
    else:
        print("[Reporter]  No flagged transactions.")

    # ── User risk summary ─────────────────────────────────────────────────
    summary_rows = []
    for r in reports:
        summary_rows.append({
            "user_id":           r["user_id"],
            "total_txns":        r["total_txns"],
            "n_flagged":         r["n_flagged"],
            "max_hybrid_score":  r["max_hybrid_score"],
            "max_lstm_score":    r["max_lstm_score"],
            "max_iso_score":     r["max_iso_score"],
            "alert_triggered":   r["alert_triggered"],
            "is_true_fraud":     r["is_true_fraud"],
            "fraud_types":       "|".join(r["fraud_types_found"]),
            "risk_level":        _risk_level(r["max_hybrid_score"]),
        })

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"[Reporter]  User summary CSV   → {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
def print_console_summary(reports: list[dict]) -> None:
    """Print a ranked console table of the top-10 highest-risk users."""
    flagged = sorted(
        [r for r in reports if r["alert_triggered"]],
        key=lambda r: r["max_hybrid_score"], reverse=True
    )[:10]

    if not flagged:
        print("[Reporter]  No users flagged.")
        return

    print(f"\n{'═'*72}")
    print("  TOP FLAGGED USERS  (ranked by hybrid score)")
    print(f"{'═'*72}")
    hdr = f"  {'User':<14} {'HybridScore':>11} {'LSTM':>7} {'ISO':>7} " \
          f"{'Flagged':>8} {'TrueFraud':>10} {'Risk':<14}"
    print(hdr)
    print(f"  {'-'*67}")
    for r in flagged:
        tf  = "YES ✓" if r["is_true_fraud"] else "NO"
        lvl = _risk_level(r["max_hybrid_score"]).replace("🔴","[CRIT]")\
                                                  .replace("🟠","[HIGH]")\
                                                  .replace("🟡","[MED ]")\
                                                  .replace("🟢","[LOW ]")
        print(f"  {r['user_id']:<14} {r['max_hybrid_score']:>11.4f} "
              f"{r['max_lstm_score']:>7.4f} {r['max_iso_score']:>7.4f} "
              f"{r['n_flagged']:>8}  {tf:>9}  {lvl}")
    print(f"{'═'*72}\n")
