"""
Model training with walk-forward cross-validation.

Walk-forward validation simulates real-world deployment:
- Train on years 0–N, validate on year N+1
- Slide the window forward; never peek at future data
- Report mean/std of AUC across all splits

Supported model types:
  gradient_boosting  – GradientBoostingClassifier (default)
  random_forest      – RandomForestClassifier
  logistic_regression – LogisticRegression (fast baseline)
"""
import logging
import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline

from config import ModelConfig

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train, validate, and persist the signal prediction model."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.model: Optional[Pipeline] = None
        self.feature_names: List[str] = []
        self.val_scores: List[Dict] = []
        os.makedirs(cfg.model_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self, X: pd.DataFrame, y: pd.Series, final_fit: bool = True
    ) -> Dict[str, float]:
        """
        Walk-forward cross-validation, then optionally fit on all data.

        Parameters
        ----------
        X           : Feature DataFrame with DatetimeIndex
        y           : Binary target series (same index)
        final_fit   : If True, fit final model on full dataset after CV

        Returns
        -------
        Dictionary of validation metrics (mean AUC, mean AP, etc.)
        """
        self.feature_names = X.columns.tolist()
        splits = self._walk_forward_splits(X.index)

        logger.info("Walk-forward CV: %d splits", len(splits))
        all_aucs, all_aps = [], []

        for i, (train_idx, val_idx) in enumerate(splits):
            X_train, y_train = X.loc[train_idx], y.loc[train_idx]
            X_val, y_val = X.loc[val_idx], y.loc[val_idx]

            model = self._build_model()
            model.fit(X_train, y_train)

            proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, proba)
            ap = average_precision_score(y_val, proba)

            all_aucs.append(auc)
            all_aps.append(ap)
            self.val_scores.append({"split": i, "auc": auc, "ap": ap,
                                    "n_train": len(X_train), "n_val": len(X_val)})
            logger.info(
                "Split %d: AUC=%.4f  AP=%.4f  (train=%d, val=%d)",
                i, auc, ap, len(X_train), len(X_val)
            )

        metrics = {
            "auc_mean": float(np.mean(all_aucs)),
            "auc_std": float(np.std(all_aucs)),
            "ap_mean": float(np.mean(all_aps)),
            "ap_std": float(np.std(all_aps)),
        }
        logger.info("CV summary: AUC=%.4f±%.4f  AP=%.4f±%.4f",
                    metrics["auc_mean"], metrics["auc_std"],
                    metrics["ap_mean"], metrics["ap_std"])

        if final_fit:
            logger.info("Fitting final model on full dataset (%d rows)...", len(X))
            self.model = self._build_model()
            self.model.fit(X, y)
            logger.info("Final model fitted.")

        return metrics

    def save(self, path: Optional[str] = None) -> str:
        """Persist model + metadata to disk."""
        if self.model is None:
            raise RuntimeError("No model to save. Call train() first.")
        if path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.cfg.model_dir, f"model_{ts}.pkl")
        payload = {
            "model": self.model,
            "feature_names": self.feature_names,
            "val_scores": self.val_scores,
            "model_type": self.cfg.model_type,
            "saved_at": datetime.now().isoformat(),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        # Also write a "latest" pointer
        latest = os.path.join(self.cfg.model_dir, "latest.pkl")
        with open(latest, "wb") as f:
            pickle.dump(payload, f)
        logger.info("Model saved to %s", path)
        return path

    @classmethod
    def load(cls, cfg: ModelConfig, path: Optional[str] = None) -> "ModelTrainer":
        """Load a previously saved model."""
        if path is None:
            path = os.path.join(cfg.model_dir, "latest.pkl")
        with open(path, "rb") as f:
            payload = pickle.load(f)
        instance = cls(cfg)
        instance.model = payload["model"]
        instance.feature_names = payload["feature_names"]
        instance.val_scores = payload.get("val_scores", [])
        logger.info("Model loaded from %s (saved %s)", path, payload.get("saved_at", "?"))
        return instance

    def feature_importance(self, top_n: int = 20) -> pd.Series:
        """Return top feature importances from the trained model."""
        if self.model is None:
            raise RuntimeError("Model not trained yet.")
        estimator = self.model.named_steps.get("clf", self.model)
        # Unwrap calibration wrapper if present
        if hasattr(estimator, "calibrated_classifiers_"):
            estimator = estimator.calibrated_classifiers_[0].estimator

        if hasattr(estimator, "feature_importances_"):
            imp = pd.Series(
                estimator.feature_importances_, index=self.feature_names
            ).sort_values(ascending=False)
        elif hasattr(estimator, "coef_"):
            imp = pd.Series(
                np.abs(estimator.coef_[0]), index=self.feature_names
            ).sort_values(ascending=False)
        else:
            return pd.Series(dtype=float)
        return imp.head(top_n)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_model(self) -> Pipeline:
        mtype = self.cfg.model_type
        if mtype == "gradient_boosting":
            clf = GradientBoostingClassifier(
                n_estimators=self.cfg.n_estimators,
                max_depth=self.cfg.max_depth,
                learning_rate=self.cfg.learning_rate,
                min_samples_leaf=self.cfg.min_samples_leaf,
                subsample=self.cfg.subsample,
                random_state=42,
            )
        elif mtype == "random_forest":
            clf = RandomForestClassifier(
                n_estimators=self.cfg.n_estimators,
                max_depth=self.cfg.max_depth,
                min_samples_leaf=self.cfg.min_samples_leaf,
                n_jobs=-1,
                random_state=42,
            )
        elif mtype == "logistic_regression":
            clf = LogisticRegression(
                C=0.1, max_iter=1000, random_state=42, solver="lbfgs"
            )
        else:
            raise ValueError(f"Unknown model_type: {mtype}")

        # Isotonic calibration improves probability estimates
        calibrated = CalibratedClassifierCV(clf, method="isotonic", cv=3)
        return Pipeline([("clf", calibrated)])

    def _walk_forward_splits(
        self, index: pd.DatetimeIndex
    ) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        Generate (train_idx, val_idx) pairs for walk-forward CV.

        Uses expanding train window (anchored start) to maximise training data.
        """
        dates = index.sort_values().unique()
        train_days = self.cfg.train_years * 252
        val_days = self.cfg.val_years * 252
        n_splits = self.cfg.n_walk_forward_splits

        splits = []
        total_needed = train_days + n_splits * val_days

        if len(dates) < total_needed:
            # Reduce splits if we don't have enough data
            n_splits = max(1, (len(dates) - train_days) // val_days)

        for i in range(n_splits):
            val_end_pos = len(dates) - (n_splits - 1 - i) * val_days
            val_start_pos = val_end_pos - val_days
            train_end_pos = val_start_pos

            if train_end_pos < train_days:
                continue

            train_dates = dates[:train_end_pos]
            val_dates = dates[val_start_pos:val_end_pos]

            train_mask = index.isin(train_dates)
            val_mask = index.isin(val_dates)

            splits.append((index[train_mask], index[val_mask]))

        return splits
