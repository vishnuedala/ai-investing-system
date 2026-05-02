"""
Inference wrapper — loads a trained model and produces buy probabilities.
"""
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import ModelConfig
from models.trainer import ModelTrainer

logger = logging.getLogger(__name__)


class Predictor:
    """
    Wraps a trained ModelTrainer to produce probability scores for new data.

    predict(feature_dict) → {ticker: float probability in [0, 1]}
    """

    def __init__(self, trainer: ModelTrainer):
        if trainer.model is None:
            raise RuntimeError("Trainer has no fitted model.")
        self.trainer = trainer
        self._feature_names = trainer.feature_names

    @classmethod
    def from_saved(cls, cfg: ModelConfig, path: Optional[str] = None) -> "Predictor":
        trainer = ModelTrainer.load(cfg, path)
        return cls(trainer)

    def predict(
        self, features: Dict[str, pd.Series]
    ) -> Dict[str, float]:
        """
        Parameters
        ----------
        features : {ticker: feature_series} — output of FeaturePipeline.transform_latest

        Returns
        -------
        {ticker: probability of positive 10-day return}
        """
        if not features:
            return {}

        tickers = list(features.keys())
        rows = []
        for ticker in tickers:
            row = features[ticker].reindex(self._feature_names).fillna(0.0)
            rows.append(row.values)

        X = np.array(rows)
        probas = self.trainer.model.predict_proba(X)[:, 1]

        result = {t: float(p) for t, p in zip(tickers, probas)}
        logger.debug("Predicted probabilities for %d tickers", len(result))
        return result

    def predict_history(
        self, X: pd.DataFrame
    ) -> pd.Series:
        """Batch prediction over a full feature DataFrame (for backtesting)."""
        X_aligned = X.reindex(columns=self._feature_names, fill_value=0.0)
        probas = self.trainer.model.predict_proba(X_aligned)[:, 1]
        return pd.Series(probas, index=X.index, name="probability")
