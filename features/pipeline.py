"""
Feature pipeline — assembles per-ticker feature DataFrames into a
unified training matrix and provides inference-time snapshots.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from config import FeatureConfig
from features.technical import compute_features
from features.regime import add_regime_features

logger = logging.getLogger(__name__)

# Columns that must NOT be fed into the model
_META_COLS = {"target", "forward_return"}


class FeaturePipeline:
    """
    Builds and transforms features for training and inference.

    Workflow:
        pipeline = FeaturePipeline(cfg)
        X_train, y_train = pipeline.fit_transform(price_data, spy_df)
        X_live = pipeline.transform_latest(price_data, spy_df)
    """

    def __init__(self, cfg: FeatureConfig):
        self.cfg = cfg
        self.scaler = RobustScaler()
        self.feature_names: List[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Training path
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        price_data: Dict[str, pd.DataFrame],
        spy_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build training matrix across all tickers.

        Returns (X, y) where X is a DataFrame of features and y is the
        binary target (1 = positive forward return).

        Each row has a MultiIndex (ticker, date).
        """
        frames = []
        for ticker, df in price_data.items():
            if ticker == "SPY" or len(df) < self.cfg.min_rows:
                continue
            try:
                feat = self._compute_for_ticker(df, spy_df)
                feat["ticker"] = ticker
                frames.append(feat)
            except Exception as exc:
                logger.warning("Feature computation failed for %s: %s", ticker, exc)

        if not frames:
            raise RuntimeError("No valid feature data computed.")

        combined = pd.concat(frames, axis=0)
        combined.sort_index(inplace=True)

        # Separate target
        y = combined["target"].astype(int)
        X = combined.drop(columns=list(_META_COLS) + ["ticker"], errors="ignore")

        # Record feature names, fit scaler
        self.feature_names = X.columns.tolist()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            index=X.index,
            columns=self.feature_names,
        )
        self._fitted = True
        logger.info("Pipeline fitted: %d rows, %d features", len(X_scaled), len(self.feature_names))
        return X_scaled, y

    # ------------------------------------------------------------------
    # Inference path
    # ------------------------------------------------------------------

    def transform_latest(
        self,
        price_data: Dict[str, pd.DataFrame],
        spy_df: pd.DataFrame,
    ) -> Dict[str, pd.Series]:
        """
        For each ticker, return its most recent feature vector (latest date).

        Returns {ticker: feature_Series}.
        """
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted before transform_latest.")

        result: Dict[str, pd.Series] = {}
        for ticker, df in price_data.items():
            if ticker == "SPY" or len(df) < self.cfg.min_rows:
                continue
            try:
                feat = self._compute_for_ticker(df, spy_df, include_target=False)
                latest_row = feat.drop(columns=list(_META_COLS), errors="ignore").iloc[-1]
                # Align to trained features, filling any new cols with 0
                row_df = latest_row.to_frame().T.reindex(columns=self.feature_names, fill_value=0.0)
                scaled = pd.Series(
                    self.scaler.transform(row_df)[0],
                    index=self.feature_names,
                )
                result[ticker] = scaled
            except Exception as exc:
                logger.debug("transform_latest failed for %s: %s", ticker, exc)

        return result

    def transform_history(
        self,
        price_data: Dict[str, pd.DataFrame],
        spy_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Same as fit_transform but uses already-fitted scaler (for backtesting
        held-out data or walk-forward out-of-sample periods).
        """
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted before transform_history.")

        frames = []
        for ticker, df in price_data.items():
            if ticker == "SPY" or len(df) < self.cfg.min_rows:
                continue
            try:
                feat = self._compute_for_ticker(df, spy_df)
                feat["ticker"] = ticker
                frames.append(feat)
            except Exception as exc:
                logger.warning("Feature computation failed for %s: %s", ticker, exc)

        if not frames:
            raise RuntimeError("No valid feature data computed.")

        combined = pd.concat(frames).sort_index()
        y = combined["target"].astype(int)
        X = combined.drop(columns=list(_META_COLS) + ["ticker"], errors="ignore")
        X = X.reindex(columns=self.feature_names, fill_value=0.0)
        X_scaled = pd.DataFrame(
            self.scaler.transform(X), index=X.index, columns=self.feature_names
        )
        return X_scaled, y

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_for_ticker(
        self, df: pd.DataFrame, spy_df: pd.DataFrame, include_target: bool = True
    ) -> pd.DataFrame:
        cfg = self.cfg
        feat = compute_features(
            df,
            spy_df=spy_df,
            ma_windows=cfg.ma_windows,
            rsi_period=cfg.rsi_period,
            macd_fast=cfg.macd_fast,
            macd_slow=cfg.macd_slow,
            macd_signal=cfg.macd_signal,
            atr_period=cfg.atr_period,
            bb_period=cfg.bb_period,
            bb_std=cfg.bb_std,
            volume_ma_period=cfg.volume_ma_period,
            lookahead_days=cfg.lookahead_days if include_target else 0,
        )
        feat = add_regime_features(feat, spy_df)
        # Drop forward_return from training features (data leak protection)
        if not include_target:
            feat.drop(columns=["forward_return", "target"], errors="ignore", inplace=True)
        return feat
