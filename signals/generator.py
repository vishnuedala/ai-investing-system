"""
Signal generation — converts raw model probabilities into actionable signals
after applying trend alignment, regime, and quality filters.

Signal hierarchy:
  STRONG_BUY   : p >= 0.65, trend aligned, bull/neutral regime
  WEAK_BUY     : p >= 0.55
  HOLD         : 0.45 <= p < 0.55
  WEAK_SELL    : p < 0.45
  STRONG_SELL  : p < 0.35
"""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd

from config import SignalConfig
from features.regime import MarketRegime

logger = logging.getLogger(__name__)


class SignalType(Enum):
    STRONG_BUY = "STRONG_BUY"
    WEAK_BUY = "WEAK_BUY"
    HOLD = "HOLD"
    WEAK_SELL = "WEAK_SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class Signal:
    ticker: str
    signal: SignalType
    probability: float          # model probability
    regime_score: float         # market regime at signal date
    trend_aligned: bool         # price above trend MA
    date: pd.Timestamp
    close_price: float
    notes: str = ""

    @property
    def is_actionable_buy(self) -> bool:
        return self.signal in (SignalType.STRONG_BUY, SignalType.WEAK_BUY)

    @property
    def confidence(self) -> str:
        if self.signal == SignalType.STRONG_BUY:
            return "high"
        if self.signal == SignalType.WEAK_BUY:
            return "medium"
        return "low"


class SignalGenerator:
    """
    Converts model predictions into filtered, regime-aware signals.

    Usage:
        gen = SignalGenerator(cfg, regime)
        signals = gen.generate(probabilities, price_data, as_of_date)
    """

    def __init__(self, cfg: SignalConfig, regime: Optional[MarketRegime] = None):
        self.cfg = cfg
        self.regime = regime

    def generate(
        self,
        probabilities: Dict[str, float],
        price_data: Dict[str, pd.DataFrame],
        as_of_date: Optional[pd.Timestamp] = None,
    ) -> List[Signal]:
        """
        Generate signals for all tickers.

        Parameters
        ----------
        probabilities : {ticker: probability}
        price_data    : {ticker: OHLCV DataFrame}
        as_of_date    : Signal date (defaults to most recent date in data)
        """
        signals = []

        for ticker, prob in probabilities.items():
            df = price_data.get(ticker)
            if df is None or len(df) < 60:
                continue

            date = as_of_date if as_of_date is not None else df.index[-1]
            close = float(df["Close"].iloc[-1])

            # Trend alignment: price must be above trend MA for buys
            trend_ma = df["Close"].rolling(self.cfg.trend_ma, min_periods=40).mean()
            above_trend = bool(df["Close"].iloc[-1] > trend_ma.iloc[-1]) if not pd.isna(trend_ma.iloc[-1]) else True

            # Market regime score
            regime_score = 0.0
            if self.regime is not None:
                regime_score = self.regime.score_at(date)

            signal_type = self._classify(prob)
            notes = self._build_notes(signal_type, above_trend, regime_score, prob)

            # Downgrade buy signals in bear markets or when not trend-aligned
            if signal_type in (SignalType.STRONG_BUY, SignalType.WEAK_BUY):
                if regime_score < self.cfg.min_regime_score:
                    signal_type = SignalType.HOLD
                    notes += " [downgraded: bear regime]"
                elif not above_trend and self.cfg.trend_ma > 0:
                    signal_type = SignalType.WEAK_BUY
                    notes += " [caution: below trend MA]"

            signals.append(Signal(
                ticker=ticker,
                signal=signal_type,
                probability=prob,
                regime_score=regime_score,
                trend_aligned=above_trend,
                date=date,
                close_price=close,
                notes=notes,
            ))

        signals.sort(key=lambda s: s.probability, reverse=True)
        return signals

    def top_buys(
        self,
        signals: List[Signal],
        n: int = 10,
        min_type: SignalType = SignalType.WEAK_BUY,
    ) -> List[Signal]:
        """Return the top-N buy signals above min_type threshold."""
        buy_types = {SignalType.STRONG_BUY, SignalType.WEAK_BUY}
        if min_type == SignalType.STRONG_BUY:
            buy_types = {SignalType.STRONG_BUY}
        filtered = [s for s in signals if s.signal in buy_types]
        return filtered[:n]

    def print_summary(self, signals: List[Signal]) -> None:
        """Print a formatted signal summary table."""
        print(f"\n{'='*70}")
        print(f"  DAILY SIGNAL SUMMARY — {signals[0].date.date() if signals else 'N/A'}")
        print(f"{'='*70}")
        print(f"{'Ticker':<8} {'Signal':<14} {'Prob':>6} {'Regime':>8} {'Trend':>7}  Notes")
        print(f"{'-'*70}")
        for s in signals:
            trend_str = "above" if s.trend_aligned else "below"
            print(
                f"{s.ticker:<8} {s.signal.value:<14} {s.probability:>6.3f} "
                f"{s.regime_score:>8.2f} {trend_str:>7}  {s.notes}"
            )
        print(f"{'='*70}\n")

    # ------------------------------------------------------------------

    def _classify(self, prob: float) -> SignalType:
        cfg = self.cfg
        if prob >= cfg.strong_buy_threshold:
            return SignalType.STRONG_BUY
        if prob >= cfg.weak_buy_threshold:
            return SignalType.WEAK_BUY
        if prob >= cfg.weak_sell_threshold:
            return SignalType.HOLD
        if prob >= cfg.strong_sell_threshold:
            return SignalType.WEAK_SELL
        return SignalType.STRONG_SELL

    @staticmethod
    def _build_notes(
        signal: SignalType, trend: bool, regime: float, prob: float
    ) -> str:
        parts = []
        if regime >= 0.4:
            parts.append("bull mkt")
        elif regime <= -0.4:
            parts.append("bear mkt")
        if not trend:
            parts.append("below MA50")
        return ", ".join(parts)
