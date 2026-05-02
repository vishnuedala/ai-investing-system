"""
Backtest report generation — text + matplotlib charts saved to disk.
"""
import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from backtesting.metrics import compute_metrics, format_metrics

logger = logging.getLogger(__name__)


class BacktestReport:
    """Generate and save backtest reports."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def generate(
        self,
        equity_curve: pd.Series,
        trades_df: pd.DataFrame,
        benchmark_equity: Optional[pd.Series] = None,
        save_plots: bool = True,
        tag: str = "",
    ) -> Dict[str, float]:
        """
        Compute metrics, print summary, and save charts.

        Returns the metrics dict.
        """
        metrics = compute_metrics(equity_curve, trades_df, benchmark_equity)

        # Print text report
        print(format_metrics(metrics))

        # Trade breakdown
        if trades_df is not None and len(trades_df) > 0:
            self._print_trade_breakdown(trades_df)

        if save_plots:
            self._save_plots(equity_curve, trades_df, benchmark_equity, tag)

        # Save metrics CSV
        tag_str = f"_{tag}" if tag else ""
        metrics_path = os.path.join(self.results_dir, f"metrics{tag_str}.csv")
        pd.Series(metrics).to_csv(metrics_path)
        logger.info("Metrics saved to %s", metrics_path)

        return metrics

    # ------------------------------------------------------------------

    def _print_trade_breakdown(self, trades_df: pd.DataFrame) -> None:
        print("\n=== Trade Breakdown by Exit Reason ===")
        grouped = trades_df.groupby("exit_reason").agg(
            count=("pnl", "count"),
            total_pnl=("pnl", "sum"),
            avg_pnl_pct=("pnl_pct", "mean"),
            win_rate=("pnl", lambda x: (x > 0).mean()),
        )
        print(grouped.round(4).to_string())

        print("\n=== Top 10 Winners ===")
        top_win = trades_df.nlargest(10, "pnl_pct")[
            ["ticker", "entry_date", "exit_date", "pnl_pct", "exit_reason"]
        ]
        print(top_win.to_string(index=False))

        print("\n=== Top 10 Losers ===")
        top_loss = trades_df.nsmallest(10, "pnl_pct")[
            ["ticker", "entry_date", "exit_date", "pnl_pct", "exit_reason"]
        ]
        print(top_loss.to_string(index=False))

    def _save_plots(
        self,
        equity_curve: pd.Series,
        trades_df: pd.DataFrame,
        benchmark: Optional[pd.Series],
        tag: str,
    ) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")  # headless
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec

            fig = plt.figure(figsize=(14, 10))
            gs = gridspec.GridSpec(3, 2, figure=fig)

            # --- Equity curve ---
            ax1 = fig.add_subplot(gs[0, :])
            norm = equity_curve / equity_curve.iloc[0] * 100
            ax1.plot(norm.index, norm.values, label="Strategy", color="steelblue", linewidth=1.5)
            if benchmark is not None:
                bm_aligned = benchmark.reindex(equity_curve.index).ffill()
                bm_norm = bm_aligned / bm_aligned.iloc[0] * 100
                ax1.plot(bm_norm.index, bm_norm.values, label="SPY", color="gray",
                         linewidth=1.0, linestyle="--", alpha=0.8)
            ax1.set_title("Equity Curve (normalised to 100)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # --- Drawdown ---
            ax2 = fig.add_subplot(gs[1, :])
            rolling_max = equity_curve.cummax()
            drawdown = (equity_curve / rolling_max - 1) * 100
            ax2.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.4)
            ax2.set_title("Drawdown (%)")
            ax2.grid(True, alpha=0.3)

            # --- P&L distribution ---
            ax3 = fig.add_subplot(gs[2, 0])
            if trades_df is not None and len(trades_df) > 0:
                pnl_pcts = trades_df["pnl_pct"] * 100
                ax3.hist(pnl_pcts, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
                ax3.axvline(0, color="red", linestyle="--", linewidth=1)
                ax3.set_title("Trade Return Distribution (%)")
                ax3.set_xlabel("Return %")

            # --- Monthly returns heatmap ---
            ax4 = fig.add_subplot(gs[2, 1])
            monthly = equity_curve.resample("ME").last().pct_change() * 100
            monthly.dropna(inplace=True)
            if len(monthly) > 0:
                ax4.bar(range(len(monthly)), monthly.values,
                        color=["green" if x >= 0 else "red" for x in monthly.values],
                        alpha=0.7)
                ax4.set_title("Monthly Returns (%)")
                ax4.set_xlabel("Month index")
                ax4.axhline(0, color="black", linewidth=0.5)

            plt.tight_layout()
            tag_str = f"_{tag}" if tag else ""
            path = os.path.join(self.results_dir, f"backtest_report{tag_str}.png")
            plt.savefig(path, dpi=120, bbox_inches="tight")
            plt.close()
            print(f"\nChart saved: {path}")

        except Exception as exc:
            logger.warning("Could not generate plots: %s", exc)
