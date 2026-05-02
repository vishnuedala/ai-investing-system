"""
Market regime detection.

Classifies the overall market environment into:
  +1  Bull  — trending up, low stress
   0  Neutral — unclear / transitioning
  -1  Bear  — trending down, elevated stress

The regime score is used to gate new long entries and size positions.
"""
import numpy as np
import pandas as pd
from typing import Optional


class MarketRegime:
    """
    Compute a continuous regime score in [-1, +1] from SPY price data.

    Score components (equal weight):
    1. Price vs 200-day MA  (above = +1, below = -1)
    2. 50-day MA trend      (rising = +1, falling = -1)
    3. Drawdown from 52-week high (< 5% = +1, < 15% = 0, else = -1)
    4. Short-term momentum  (20-day return positive = +1, else = -1)
    5. Volatility regime    (realised vol < long-term median = +1, else = -1)
    """

    def __init__(self, spy_df: pd.DataFrame):
        self._scores = self._compute(spy_df)

    def score_at(self, date: pd.Timestamp) -> float:
        """Return regime score on or before given date."""
        try:
            scores_before = self._scores[self._scores.index <= date]
            if len(scores_before) == 0:
                return 0.0
            return float(scores_before.iloc[-1])
        except (TypeError, KeyError):
            return 0.0

    def score_series(self) -> pd.Series:
        return self._scores.copy()

    def label_at(self, date: pd.Timestamp) -> str:
        s = self.score_at(date)
        if s >= 0.4:
            return "bull"
        if s <= -0.4:
            return "bear"
        return "neutral"

    # ------------------------------------------------------------------

    @staticmethod
    def _compute(spy: pd.DataFrame) -> pd.Series:
        close = spy["Close"]
        scores = pd.DataFrame(index=close.index)

        # 1. Price vs 200-day MA
        ma200 = close.rolling(200, min_periods=150).mean()
        scores["vs_ma200"] = np.where(close > ma200, 1.0, -1.0)

        # 2. 50-day MA slope (rising vs falling)
        ma50 = close.rolling(50, min_periods=40).mean()
        scores["ma50_slope"] = np.where(ma50.pct_change(10) > 0, 1.0, -1.0)

        # 3. Drawdown from rolling 252-day high
        rolling_high = close.rolling(252, min_periods=100).max()
        drawdown = (close / rolling_high) - 1
        scores["drawdown"] = np.where(
            drawdown > -0.05, 1.0,
            np.where(drawdown > -0.15, 0.0, -1.0)
        )

        # 4. 20-day momentum
        scores["momentum"] = np.where(close.pct_change(20) > 0, 1.0, -1.0)

        # 5. Volatility regime
        daily_vol = close.pct_change().rolling(20).std() * np.sqrt(252)
        median_vol = daily_vol.expanding(min_periods=50).median()
        scores["vol_regime"] = np.where(daily_vol < median_vol, 1.0, -1.0)

        # Composite: simple average of all components
        regime_score = scores.mean(axis=1)
        regime_score.name = "regime_score"
        return regime_score.fillna(0.0)


def add_regime_features(
    features_df: pd.DataFrame,
    spy_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Append regime columns to an existing features DataFrame.

    Added columns:
    - regime_score      : continuous [-1, +1]
    - regime_bull       : 1 if bull
    - regime_bear       : 1 if bear
    """
    regime = MarketRegime(spy_df)
    scores = regime.score_series().reindex(features_df.index).ffill().fillna(0.0)
    features_df = features_df.copy()
    features_df["regime_score"] = scores
    features_df["regime_bull"] = (scores >= 0.4).astype(int)
    features_df["regime_bear"] = (scores <= -0.4).astype(int)
    return features_df
