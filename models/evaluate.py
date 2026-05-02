"""
Model evaluation utilities — classification metrics and probability calibration checks.
"""
import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix,
)

logger = logging.getLogger(__name__)


def evaluate_model(
    y_true: pd.Series,
    y_proba: pd.Series,
    threshold: float = 0.55,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true    : True binary labels
    y_proba   : Predicted probabilities
    threshold : Decision boundary for precision/recall/F1
    """
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "auc_roc": roc_auc_score(y_true, y_proba),
        "avg_precision": average_precision_score(y_true, y_proba),
        "brier_score": brier_score_loss(y_true, y_proba),
        "base_rate": float(y_true.mean()),
        "pred_positive_rate": float(y_pred.mean()),
    }

    # Precision at top-decile probabilities
    top_decile = y_proba >= y_proba.quantile(0.90)
    if top_decile.sum() > 0:
        metrics["precision_top10pct"] = float(y_true[top_decile].mean())

    if verbose:
        print("\n=== Model Evaluation ===")
        print(f"AUC-ROC:             {metrics['auc_roc']:.4f}")
        print(f"Avg Precision (AP):  {metrics['avg_precision']:.4f}")
        print(f"Brier Score:         {metrics['brier_score']:.4f}  (lower=better, 0.25=random)")
        print(f"Base rate:           {metrics['base_rate']:.3f}")
        print(f"Precision top 10%:   {metrics.get('precision_top10pct', 'N/A'):.3f}")
        print(f"\nClassification @ threshold={threshold}:")
        print(classification_report(y_true, y_pred, target_names=["Down", "Up"]))

    return metrics


def calibration_summary(
    y_true: pd.Series, y_proba: pd.Series, n_bins: int = 10
) -> pd.DataFrame:
    """
    Binned calibration table: compare predicted probability to actual win rate.

    A well-calibrated model should have predicted ≈ actual in each bin.
    """
    bins = pd.cut(y_proba, bins=n_bins)
    tbl = pd.DataFrame({"predicted": y_proba, "actual": y_true}).groupby(bins, observed=True)
    summary = tbl.agg(
        mean_pred=("predicted", "mean"),
        mean_actual=("actual", "mean"),
        count=("actual", "count"),
    )
    summary["calibration_error"] = (summary["mean_pred"] - summary["mean_actual"]).abs()
    return summary


def probability_distribution_report(y_proba: pd.Series) -> None:
    """Print decile distribution of predicted probabilities."""
    deciles = y_proba.quantile(np.arange(0.1, 1.1, 0.1))
    print("\n=== Probability Deciles ===")
    for q, v in deciles.items():
        print(f"  {int(q*100):3d}th pct: {v:.3f}")
