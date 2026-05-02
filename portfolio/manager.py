"""
Portfolio state manager — tracks open positions, P&L, and enforces
stop-loss / take-profit / trailing-stop exit rules.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import PortfolioConfig

logger = logging.getLogger(__name__)


@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    stop_loss: float            # absolute price level
    take_profit: float          # absolute price level
    trailing_stop: float        # trailing stop price (updated as price rises)
    probability: float          # signal probability at entry
    notes: str = ""

    @property
    def cost_basis(self) -> float:
        return self.entry_price * self.shares

    def current_value(self, price: float) -> float:
        return price * self.shares

    def unrealised_pnl(self, price: float) -> float:
        return (price - self.entry_price) * self.shares

    def unrealised_pct(self, price: float) -> float:
        if self.entry_price == 0:
            return 0.0
        return (price / self.entry_price) - 1

    def update_trailing_stop(self, current_price: float, trail_pct: float) -> None:
        new_stop = current_price * (1 - trail_pct)
        if new_stop > self.trailing_stop:
            self.trailing_stop = new_stop

    def should_exit(self, current_price: float) -> Tuple[bool, str]:
        """Return (should_exit, reason)."""
        if current_price <= self.stop_loss:
            return True, "stop_loss"
        if current_price >= self.take_profit:
            return True, "take_profit"
        if current_price <= self.trailing_stop:
            return True, "trailing_stop"
        return False, ""


@dataclass
class Trade:
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    exit_reason: str


class PortfolioManager:
    """
    Tracks cash, open positions, and closed trade history.

    Call update_prices() each day, then check for exits.
    Call enter() to add a new position, exit_position() to close one.
    """

    def __init__(self, cfg: PortfolioConfig):
        self.cfg = cfg
        self.cash: float = cfg.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Trade] = []
        self._peak_equity: float = cfg.initial_capital

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @property
    def equity(self) -> float:
        """Current equity = cash + market value of all positions (no price update)."""
        return self.cash + sum(p.cost_basis for p in self.positions.values())

    def mark_to_market(self, prices: Dict[str, float]) -> float:
        """Return portfolio equity at current prices."""
        mkt_value = sum(
            p.shares * prices.get(p.ticker, p.entry_price)
            for p in self.positions.values()
        )
        total = self.cash + mkt_value
        self._peak_equity = max(self._peak_equity, total)
        return total

    def can_open(self, cost: float) -> bool:
        min_cash = self.cfg.initial_capital * self.cfg.min_cash_reserve
        return (
            self.cash - cost >= min_cash
            and len(self.positions) < self.cfg.max_positions
        )

    def enter(
        self,
        ticker: str,
        date: pd.Timestamp,
        price: float,
        shares: int,
        probability: float,
        commission_pct: float = 0.001,
    ) -> Optional[Position]:
        """Open a new long position. Returns None if rejected."""
        if shares <= 0:
            return None
        cost = price * shares
        commission = cost * commission_pct

        if not self.can_open(cost + commission):
            logger.debug("Cannot open %s: insufficient cash or max positions reached", ticker)
            return None

        stop_price = price * (1 - self.cfg.stop_loss_pct)
        target_price = price * (1 + self.cfg.take_profit_pct)
        trail_price = price * (1 - self.cfg.trailing_stop_pct)

        pos = Position(
            ticker=ticker,
            entry_date=date,
            entry_price=price,
            shares=shares,
            stop_loss=stop_price,
            take_profit=target_price,
            trailing_stop=trail_price,
            probability=probability,
        )
        self.positions[ticker] = pos
        self.cash -= cost + commission
        logger.info(
            "ENTER %s: %d sh @ %.2f  stop=%.2f  target=%.2f  cash=%.0f",
            ticker, shares, price, stop_price, target_price, self.cash
        )
        return pos

    def exit_position(
        self,
        ticker: str,
        date: pd.Timestamp,
        price: float,
        reason: str = "signal",
        commission_pct: float = 0.001,
    ) -> Optional[Trade]:
        """Close an open position. Returns Trade record or None."""
        pos = self.positions.pop(ticker, None)
        if pos is None:
            return None

        proceeds = price * pos.shares
        commission = proceeds * commission_pct
        pnl = proceeds - pos.cost_basis - commission * 2  # entry + exit commission
        pnl_pct = pnl / pos.cost_basis

        self.cash += proceeds - commission
        trade = Trade(
            ticker=ticker,
            entry_date=pos.entry_date,
            exit_date=date,
            entry_price=pos.entry_price,
            exit_price=price,
            shares=pos.shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
        )
        self.trade_history.append(trade)
        logger.info(
            "EXIT %s: %d sh @ %.2f  pnl=%.2f (%.1f%%)  reason=%s",
            ticker, pos.shares, price, pnl, pnl_pct * 100, reason
        )
        return trade

    def update_exits(
        self,
        date: pd.Timestamp,
        prices: Dict[str, float],
        commission_pct: float = 0.001,
    ) -> List[Trade]:
        """
        Check all open positions against stop/target rules and close as needed.
        Returns list of trades triggered.
        """
        exits = []
        for ticker in list(self.positions.keys()):
            pos = self.positions[ticker]
            price = prices.get(ticker)
            if price is None:
                continue

            # Update trailing stop
            pos.update_trailing_stop(price, self.cfg.trailing_stop_pct)

            should_exit, reason = pos.should_exit(price)
            if should_exit:
                trade = self.exit_position(ticker, date, price, reason, commission_pct)
                if trade:
                    exits.append(trade)
        return exits

    def trade_summary(self) -> pd.DataFrame:
        """Return a DataFrame summarising all completed trades."""
        if not self.trade_history:
            return pd.DataFrame()
        rows = []
        for t in self.trade_history:
            rows.append({
                "ticker": t.ticker,
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "hold_days": (t.exit_date - t.entry_date).days,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "shares": t.shares,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "exit_reason": t.exit_reason,
            })
        return pd.DataFrame(rows)

    def current_holdings_str(self, prices: Dict[str, float]) -> str:
        if not self.positions:
            return "  (no open positions)"
        lines = []
        for ticker, pos in self.positions.items():
            price = prices.get(ticker, pos.entry_price)
            pnl_pct = pos.unrealised_pct(price) * 100
            lines.append(
                f"  {ticker:<6} {pos.shares:>5} sh  "
                f"cost={pos.entry_price:.2f}  now={price:.2f}  "
                f"{'+'if pnl_pct>=0 else ''}{pnl_pct:.1f}%"
            )
        return "\n".join(lines)
