"""
Portfolio performance metrics.

All metrics use daily equity curve values.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional


def compute_metrics(
    equity_curve: pd.Series,
    trades_df: Optional[pd.DataFrame] = None,
    benchmark: Optional[pd.Series] = None,
    risk_free_rate: float = 0.04,
) -> Dict[str, float]:
    """
    Parameters
    ----------
    equity_curve  : Daily portfolio equity (DatetimeIndex)
    trades_df     : Trade history DataFrame (from PortfolioManager.trade_summary())
    benchmark     : Benchmark equity curve (e.g. SPY buy-and-hold)
    risk_free_rate: Annual risk-free rate for Sharpe/Sortino

    Returns
    -------
    Dictionary of metrics.
    """
    returns = equity_curve.pct_change().dropna()
    n_days = len(equity_curve)
    n_years = n_days / 252.0

    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    cagr = (1 + total_return) ** (1 / max(n_years, 1e-6)) - 1

    ann_vol = returns.std() * np.sqrt(252)
    rfr_daily = (1 + risk_free_rate) ** (1 / 252) - 1
    excess = returns - rfr_daily
    sharpe = (excess.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0

    # Sortino (downside deviation only)
    downside = returns[returns < rfr_daily]
    sortino_vol = downside.std() * np.sqrt(252) if len(downside) > 1 else ann_vol
    sortino = (returns.mean() - rfr_daily) / (downside.std() if len(downside) > 1 else 1e-6) * np.sqrt(252)

    # Max drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve / rolling_max) - 1
    max_dd = float(drawdown.min())

    # Calmar ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # Win rate and trade stats
    win_rate = trade_win_rate = avg_win = avg_loss = avg_hold = 0.0
    n_trades = 0
    if trades_df is not None and len(trades_df) > 0:
        n_trades = len(trades_df)
        winners = trades_df[trades_df["pnl"] > 0]
        losers = trades_df[trades_df["pnl"] <= 0]
        win_rate = len(winners) / n_trades
        avg_win = winners["pnl_pct"].mean() if len(winners) > 0 else 0.0
        avg_loss = losers["pnl_pct"].mean() if len(losers) > 0 else 0.0
        avg_hold = trades_df["hold_days"].mean() if "hold_days" in trades_df.columns else 0.0
        # Profit factor
        gross_profit = winners["pnl"].sum()
        gross_loss = abs(losers["pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    else:
        profit_factor = 0.0

    # Alpha / beta vs benchmark
    alpha = beta = 0.0
    if benchmark is not None:
        bm_returns = benchmark.pct_change().dropna()
        aligned = returns.align(bm_returns, join="inner")
        p_ret, b_ret = aligned
        if len(p_ret) > 10:
            cov = np.cov(p_ret, b_ret)
            beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
            alpha = (p_ret.mean() - beta * b_ret.mean()) * 252

    metrics = {
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "ann_vol_pct": ann_vol * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown_pct": max_dd * 100,
        "n_trades": n_trades,
        "win_rate_pct": win_rate * 100,
        "avg_win_pct": avg_win * 100,
        "avg_loss_pct": avg_loss * 100,
        "profit_factor": profit_factor,
        "avg_hold_days": avg_hold,
        "alpha_annual": alpha * 100,
        "beta": beta,
    }
    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    """Pretty-print a metrics dict."""
    lines = [
        "\n╔══════════════════════════════════════╗",
        "║      BACKTEST PERFORMANCE REPORT     ║",
        "╚══════════════════════════════════════╝",
        f"  Total Return:    {metrics['total_return_pct']:>8.1f}%",
        f"  CAGR:            {metrics['cagr_pct']:>8.1f}%",
        f"  Annualised Vol:  {metrics['ann_vol_pct']:>8.1f}%",
        f"  Sharpe Ratio:    {metrics['sharpe']:>8.2f}",
        f"  Sortino Ratio:   {metrics['sortino']:>8.2f}",
        f"  Calmar Ratio:    {metrics['calmar']:>8.2f}",
        f"  Max Drawdown:    {metrics['max_drawdown_pct']:>8.1f}%",
        "  ─────────────────────────────────────",
        f"  Total Trades:    {int(metrics['n_trades']):>8d}",
        f"  Win Rate:        {metrics['win_rate_pct']:>8.1f}%",
        f"  Avg Win:         {metrics['avg_win_pct']:>8.1f}%",
        f"  Avg Loss:        {metrics['avg_loss_pct']:>8.1f}%",
        f"  Profit Factor:   {metrics['profit_factor']:>8.2f}",
        f"  Avg Hold Days:   {metrics['avg_hold_days']:>8.1f}",
        "  ─────────────────────────────────────",
        f"  Alpha (ann):     {metrics['alpha_annual']:>8.1f}%",
        f"  Beta:            {metrics['beta']:>8.2f}",
        "═" * 40,
    ]
    return "\n".join(lines)
