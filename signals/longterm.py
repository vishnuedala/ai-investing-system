"""
Long-term buy-and-hold signal generator.

Target hold period: 3–6 months (vs 10-day swing signals)
Strategy: trend-following + insider buying confirmation

Entry criteria (ALL must be true):
  1. Price above 200-day MA  (long-term uptrend)
  2. 50-day MA above 200-day MA  (golden cross zone)
  3. 6-month momentum > 5%  (sustained strength)
  4. Insider buying signal  (management has skin in the game)
  5. Relative strength vs SPY positive over 3 months

Exit criteria:
  - Price crosses below 200-day MA  (trend broken)
  - Hard stop: −15% from entry
  - Profit target: +40% from entry
  - Hold for minimum 60 days before considering exit (avoid overtrading)

Risk:
  - Max 6 long-term positions
  - 12–15% per position (larger than swing — higher conviction)
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from data.insider import get_insider_signals_bulk

logger = logging.getLogger(__name__)


@dataclass
class LongTermSignal:
    ticker: str
    score: float                    # composite 0–100
    close_price: float
    above_200ma: bool
    golden_cross: bool              # 50MA > 200MA
    momentum_6m: float              # 6-month price return
    momentum_3m: float
    rel_strength_3m: float          # vs SPY
    insider_score: float            # 0–1 from Form 4 data
    insider_buys: int
    insider_buy_value: float
    action: str                     # STRONG_BUY / BUY / WATCH / AVOID
    reasons: List[str]

    def __str__(self):
        return (
            f"{self.ticker:<6}  {self.action:<12}  score={self.score:.0f}/100  "
            f"6m={self.momentum_6m*100:+.1f}%  "
            f"insider={'✅' if self.insider_buys > 0 else '—'}"
        )


class LongTermAnalyzer:
    """
    Scores each stock on a 0–100 composite scale and recommends
    long-term positions.

    Score components:
      30 pts — Trend (above 200MA, golden cross)
      25 pts — Momentum (3m and 6m price return)
      25 pts — Relative strength vs market
      20 pts — Insider buying (SEC Form 4)
    """

    def __init__(
        self,
        stop_loss_pct: float = 0.15,
        take_profit_pct: float = 0.40,
        min_hold_days: int = 60,
        fetch_insider: bool = True,
    ):
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.min_hold_days = min_hold_days
        self.fetch_insider = fetch_insider

    def analyze(
        self,
        price_data: Dict[str, pd.DataFrame],
        spy_df: Optional[pd.DataFrame] = None,
    ) -> List[LongTermSignal]:
        """
        Analyze all tickers and return ranked long-term signals.
        """
        tickers = [t for t in price_data if t != "SPY" and len(price_data[t]) >= 200]

        # Fetch insider data in bulk (one SEC request per ticker)
        insider_data: Dict[str, Dict] = {}
        if self.fetch_insider and tickers:
            logger.info("Fetching insider transaction data for %d tickers...", len(tickers))
            print(f"\n  Fetching SEC Form 4 insider data for {len(tickers)} stocks...")
            insider_data = get_insider_signals_bulk(tickers, lookback_days=90)
            logger.info("Insider data fetched.")

        signals = []
        for ticker in tickers:
            df = price_data[ticker]
            spy = spy_df
            try:
                sig = self._score_ticker(ticker, df, spy, insider_data.get(ticker, {}))
                signals.append(sig)
            except Exception as exc:
                logger.debug("Long-term score failed for %s: %s", ticker, exc)

        # Sort by composite score descending
        signals.sort(key=lambda s: s.score, reverse=True)
        return signals

    def print_report(self, signals: List[LongTermSignal], top_n: int = 15) -> None:
        print(f"\n{'='*75}")
        print("  LONG-TERM BUY & HOLD ANALYSIS  (3–6 month horizon)")
        print(f"  Scores: Trend(30) + Momentum(25) + Rel.Strength(25) + Insider(20) = 100")
        print(f"{'='*75}")
        print(f"{'Ticker':<7} {'Action':<14} {'Score':>5}  {'6m Ret':>7}  "
              f"{'3m Ret':>7}  {'vs SPY':>7}  {'Insider':>8}  200MA")
        print(f"{'-'*75}")

        for s in signals[:top_n]:
            trend_str = "✅" if s.above_200ma else "❌"
            insider_str = f"+${s.insider_buy_value/1000:.0f}K" if s.insider_buys > 0 else "  none"
            print(
                f"{s.ticker:<7} {s.action:<14} {s.score:>5.0f}  "
                f"{s.momentum_6m*100:>+7.1f}%  "
                f"{s.momentum_3m*100:>+7.1f}%  "
                f"{s.rel_strength_3m*100:>+7.1f}%  "
                f"{insider_str:>8}  {trend_str}"
            )

        print(f"{'='*75}")
        buys = [s for s in signals if s.action in ("STRONG_BUY", "BUY")]
        if buys:
            print(f"\n  TOP PICKS ({len(buys)} stocks):")
            for s in buys[:5]:
                print(f"  ★  {s.ticker:<6}  ${s.close_price:.2f}")
                print(f"     Stop loss:   ${s.close_price * (1 - self.stop_loss_pct):.2f}  (−{self.stop_loss_pct*100:.0f}%)")
                print(f"     Take profit: ${s.close_price * (1 + self.take_profit_pct):.2f}  (+{self.take_profit_pct*100:.0f}%)")
                print(f"     Hold target: {self.min_hold_days}–180 days")
                for r in s.reasons[:3]:
                    print(f"     → {r}")
                print()

    # ------------------------------------------------------------------

    def _score_ticker(
        self,
        ticker: str,
        df: pd.DataFrame,
        spy_df: Optional[pd.DataFrame],
        insider: Dict,
    ) -> LongTermSignal:
        close = df["Close"]
        current = float(close.iloc[-1])

        # --- Trend component (30 pts) ---
        ma50 = close.rolling(50, min_periods=40).mean().iloc[-1]
        ma200 = close.rolling(200, min_periods=150).mean().iloc[-1]
        above_200 = current > ma200
        golden_cross = ma50 > ma200

        trend_score = 0.0
        if above_200:
            trend_score += 18
            # How far above 200MA?
            pct_above = (current / ma200 - 1)
            trend_score += min(8, pct_above * 100)   # up to 8 pts for distance
        if golden_cross:
            trend_score += 4

        # --- Momentum component (25 pts) ---
        periods = {
            "1m": 21, "3m": 63, "6m": 126, "12m": 252
        }
        returns = {}
        for label, days in periods.items():
            if len(close) > days:
                returns[label] = float(close.iloc[-1] / close.iloc[-days] - 1)
            else:
                returns[label] = 0.0

        mom_6m = returns["6m"]
        mom_3m = returns["3m"]

        mom_score = 0.0
        # 6-month return (15 pts)
        if mom_6m > 0.20:   mom_score += 15
        elif mom_6m > 0.10: mom_score += 10
        elif mom_6m > 0.05: mom_score += 6
        elif mom_6m > 0:    mom_score += 2
        # 3-month return (10 pts)
        if mom_3m > 0.10:   mom_score += 10
        elif mom_3m > 0.05: mom_score += 6
        elif mom_3m > 0:    mom_score += 3

        # --- Relative strength vs SPY (25 pts) ---
        rel_3m = 0.0
        rel_score = 0.0
        if spy_df is not None:
            spy_close = spy_df["Close"].reindex(close.index).ffill()
            if len(spy_close) > 63:
                spy_ret_3m = float(spy_close.iloc[-1] / spy_close.iloc[-63] - 1)
                rel_3m = mom_3m - spy_ret_3m
                if rel_3m > 0.10:   rel_score = 25
                elif rel_3m > 0.05: rel_score = 18
                elif rel_3m > 0:    rel_score = 10
                elif rel_3m > -0.05: rel_score = 4
        else:
            rel_score = 12  # neutral if no benchmark

        # --- Insider buying component (20 pts) ---
        insider_score_raw = insider.get("signal_score", 0.5)
        n_buys = insider.get("n_buys", 0)
        buy_value = insider.get("buy_value", 0.0)

        insider_score = 0.0
        if n_buys >= 3:
            insider_score = 20                        # multiple insiders buying = strong
        elif n_buys == 2:
            insider_score = 14
        elif n_buys == 1:
            insider_score = 8
            if buy_value > 500_000:
                insider_score = 14                    # large single purchase = more weight
        elif insider.get("n_sells", 0) > 2:
            insider_score = -5                        # heavy selling is a warning

        # --- Composite ---
        total_score = trend_score + mom_score + rel_score + insider_score
        total_score = max(0.0, min(100.0, total_score))

        # --- Action label ---
        if total_score >= 70 and above_200:
            action = "STRONG_BUY"
        elif total_score >= 55 and above_200:
            action = "BUY"
        elif total_score >= 40:
            action = "WATCH"
        else:
            action = "AVOID"

        # --- Build reasoning ---
        reasons = []
        if above_200:
            reasons.append(f"Price {(current/ma200-1)*100:+.1f}% above 200-day MA")
        else:
            reasons.append(f"Price {(current/ma200-1)*100:+.1f}% BELOW 200-day MA")
        if golden_cross:
            reasons.append("Golden cross: 50MA above 200MA ✅")
        if mom_6m > 0.05:
            reasons.append(f"Strong 6-month momentum: +{mom_6m*100:.1f}%")
        if rel_3m > 0.03:
            reasons.append(f"Outperforming SPY by {rel_3m*100:+.1f}% over 3 months")
        if n_buys > 0:
            reasons.append(
                f"Insider buying: {n_buys} purchase(s) totalling ${buy_value:,.0f} (SEC Form 4)"
            )

        return LongTermSignal(
            ticker=ticker,
            score=total_score,
            close_price=current,
            above_200ma=above_200,
            golden_cross=golden_cross,
            momentum_6m=mom_6m,
            momentum_3m=mom_3m,
            rel_strength_3m=rel_3m,
            insider_score=insider_score_raw,
            insider_buys=n_buys,
            insider_buy_value=buy_value,
            action=action,
            reasons=reasons,
        )
