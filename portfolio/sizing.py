"""
Position sizing — volatility-adjusted and signal-strength-weighted.

Two sizing methods:
1. Fixed fractional: flat pct of portfolio per position
2. Volatility-targeted: scale size so each position contributes equal volatility
   (target_portfolio_vol / n_positions) / stock_daily_vol
"""
import numpy as np
import pandas as pd
from typing import Optional

from config import PortfolioConfig


class PositionSizer:
    """Compute target position sizes as a fraction of portfolio equity."""

    def __init__(self, cfg: PortfolioConfig):
        self.cfg = cfg

    def size(
        self,
        ticker: str,
        probability: float,
        price_df: pd.DataFrame,
        portfolio_equity: float,
        n_current_positions: int,
    ) -> float:
        """
        Return target position value in dollars.

        Uses volatility targeting when cfg.use_volatility_sizing is True,
        otherwise returns a fixed fraction of equity.

        Scales up for higher-confidence signals (strong_buy gets 1.25x).
        """
        base = self._base_fraction(probability) * portfolio_equity

        if self.cfg.use_volatility_sizing:
            base = self._vol_adjust(base, price_df, n_current_positions)

        # Enforce position limits
        max_val = self.cfg.max_position_size * portfolio_equity
        min_val = self.cfg.min_position_size * portfolio_equity
        result = float(np.clip(base, min_val, max_val))

        # Never allocate if we'd breach the cash reserve
        available = portfolio_equity * (1 - self.cfg.min_cash_reserve)
        return min(result, available)

    def shares(
        self,
        ticker: str,
        probability: float,
        price_df: pd.DataFrame,
        portfolio_equity: float,
        n_current_positions: int,
        current_price: Optional[float] = None,
    ) -> int:
        """Return number of whole shares to buy."""
        if current_price is None:
            current_price = float(price_df["Close"].iloc[-1])
        if current_price <= 0:
            return 0
        target_value = self.size(ticker, probability, price_df, portfolio_equity, n_current_positions)
        return max(0, int(target_value // current_price))

    # ------------------------------------------------------------------

    def _base_fraction(self, probability: float) -> float:
        """Scale base fraction by signal confidence."""
        base = self.cfg.base_position_size
        if probability >= 0.65:
            return base * 1.25    # strong buy gets more
        if probability >= 0.55:
            return base * 1.0
        return base * 0.75

    def _vol_adjust(
        self, base_value: float, price_df: pd.DataFrame, n_positions: int
    ) -> float:
        """
        Adjust size so per-position volatility ≈ target_vol / n_positions.
        Falls back to base_value if we can't compute vol.
        """
        try:
            returns = price_df["Close"].pct_change().dropna()
            if len(returns) < 20:
                return base_value
            daily_vol = returns.rolling(20).std().iloc[-1]
            annual_vol = daily_vol * np.sqrt(252)
            if annual_vol <= 0 or np.isnan(annual_vol):
                return base_value

            n = max(n_positions, 1)
            target_pos_vol = self.cfg.target_portfolio_vol / n

            # vol_scale = target_pos_vol / stock_annual_vol
            vol_scale = target_pos_vol / annual_vol
            adjusted = base_value * vol_scale
            return adjusted
        except Exception:
            return base_value
