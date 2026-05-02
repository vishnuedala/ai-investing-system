"""
Paper trading — runs the full signal → portfolio pipeline on live market data
without placing real orders.

Daily workflow (call run_daily()):
1. Fetch latest data
2. Compute features & model predictions
3. Generate signals
4. Update open positions (check stop/target)
5. Enter new positions from top buy signals
6. Log state and save snapshot
"""
import json
import logging
import os
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from config import Config
from data.fetcher import DataFetcher
from data.universe import get_universe
from features.pipeline import FeaturePipeline
from features.regime import MarketRegime
from models.predictor import Predictor
from models.trainer import ModelTrainer
from portfolio.manager import PortfolioManager
from portfolio.sizing import PositionSizer
from signals.generator import SignalGenerator, SignalType

logger = logging.getLogger(__name__)


class PaperTrader:
    """
    End-to-end paper trading system.

    Persists portfolio state to disk so it survives between runs.
    """

    def __init__(self, cfg: Config, state_path: str = "logs/paper_state.json"):
        self.cfg = cfg
        self.state_path = state_path
        os.makedirs(os.path.dirname(state_path), exist_ok=True)

        self.fetcher = DataFetcher(cfg.data)
        self.pipeline = FeaturePipeline(cfg.features)
        self.portfolio = PortfolioManager(cfg.portfolio)
        self.sizer = PositionSizer(cfg.portfolio)
        self._equity_log: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(self, universe_size: str = "default") -> None:
        """
        Initial setup: download data, train model, save state.
        Call once before starting daily trading.
        """
        tickers = get_universe(universe_size)
        logger.info("Setting up paper trader with %d tickers...", len(tickers))

        price_data = self.fetcher.fetch(tickers)
        spy_df = price_data.get(self.cfg.data.spy_ticker)

        # Train model on historical data
        trainer = ModelTrainer(self.cfg.model)
        X, y = self.pipeline.fit_transform(price_data, spy_df)
        metrics = trainer.train(X, y, final_fit=True)
        trainer.save()

        logger.info("Setup complete. CV AUC=%.4f", metrics.get("auc_mean", 0))
        print(f"\nSetup complete. Validation AUC: {metrics.get('auc_mean', 0):.4f}")

    def run_daily(self, universe_size: str = "default") -> None:
        """
        Run one day of paper trading. Call this each market day.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        logger.info("=== Paper trade daily run: %s ===", today)

        tickers = get_universe(universe_size)
        price_data = self.fetcher.fetch(tickers, force_refresh=False)
        spy_df = price_data.get(self.cfg.data.spy_ticker)

        # Load trained model and pipeline
        try:
            trainer = ModelTrainer.load(self.cfg.model)
            predictor = Predictor(trainer)
        except FileNotFoundError:
            logger.error("No trained model found. Run setup() first.")
            return

        # Transform latest features
        live_features = self.pipeline.transform_latest(price_data, spy_df)
        probabilities = predictor.predict(live_features)

        # Market regime
        regime = MarketRegime(spy_df) if spy_df is not None else None
        sig_gen = SignalGenerator(self.cfg.signals, regime)

        # Current prices
        prices = {
            ticker: float(df["Close"].iloc[-1])
            for ticker, df in price_data.items()
            if len(df) > 0
        }
        today_ts = pd.Timestamp(today)

        # Update exits
        exits = self.portfolio.update_exits(today_ts, prices, commission_pct=0.0)
        for trade in exits:
            logger.info("PAPER EXIT: %s @ %.2f (%s)", trade.ticker, trade.exit_price, trade.exit_reason)

        # Generate signals
        signals = sig_gen.generate(probabilities, price_data, today_ts)
        sig_gen.print_summary(signals[:20])

        # Enter top buys
        buy_signals = sig_gen.top_buys(
            signals,
            n=self.cfg.portfolio.max_positions - len(self.portfolio.positions),
            min_type=SignalType.WEAK_BUY,
        )
        for sig in buy_signals:
            if sig.ticker in self.portfolio.positions:
                continue
            price = prices.get(sig.ticker, 0)
            if price <= 0:
                continue
            n_shares = self.sizer.shares(
                sig.ticker, sig.probability,
                price_data[sig.ticker],
                self.portfolio.equity,
                len(self.portfolio.positions),
                current_price=price,
            )
            self.portfolio.enter(sig.ticker, today_ts, price, n_shares, sig.probability, commission_pct=0.0)

        # Record equity
        equity = self.portfolio.mark_to_market(prices)
        self._equity_log[today] = equity

        # Print status
        print(f"\n{'='*50}")
        print(f"  PAPER PORTFOLIO — {today}")
        print(f"  Equity: ${equity:,.0f}  Cash: ${self.portfolio.cash:,.0f}")
        print(f"  Open positions ({len(self.portfolio.positions)}):")
        print(self.portfolio.current_holdings_str(prices))
        print(f"{'='*50}\n")

        self._save_state(today)

    def equity_history(self) -> pd.Series:
        if not self._equity_log:
            return pd.Series(dtype=float)
        return pd.Series(self._equity_log).rename_axis("date")

    # ------------------------------------------------------------------

    def _save_state(self, date: str) -> None:
        state = {
            "date": date,
            "cash": self.portfolio.cash,
            "equity_log": self._equity_log,
            "positions": {
                t: {
                    "ticker": p.ticker,
                    "entry_date": str(p.entry_date),
                    "entry_price": p.entry_price,
                    "shares": p.shares,
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                    "trailing_stop": p.trailing_stop,
                    "probability": p.probability,
                }
                for t, p in self.portfolio.positions.items()
            },
        }
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)
        logger.debug("State saved to %s", self.state_path)
