#!/usr/bin/env python3
"""
AI Investing System — CLI entry point.

Modes
-----
  train       Download data, engineer features, train model, save to disk
  backtest    Run historical simulation and generate performance report
  signals     Generate today's buy/sell signals from a saved model
  paper       Run one day of paper trading (call daily via cron)
  setup       Full setup: train + initial paper state

Usage examples
--------------
  python main.py train
  python main.py backtest
  python main.py signals
  python main.py paper
  python main.py setup
  python main.py train --universe small
  python main.py backtest --start 2020-01-01 --end 2024-01-01
"""
import argparse
import logging
import os
import sys

# Ensure local modules are importable regardless of working directory
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

from config import get_config


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join("logs", "investing.log"), mode="a"
            ),
        ],
    )


def cmd_train(args) -> None:
    """Train model on historical data."""
    from data.fetcher import DataFetcher
    from data.universe import get_universe
    from features.pipeline import FeaturePipeline
    from models.trainer import ModelTrainer
    from models.evaluate import evaluate_model

    cfg = get_config()
    if args.start:
        cfg.data.start_date = args.start
    if args.end:
        cfg.data.end_date = args.end

    print(f"\n{'='*60}")
    print("  TRAINING MODE")
    print(f"  Data window: {cfg.data.start_date} → {cfg.data.end_date}")
    print(f"  Model: {cfg.model.model_type}")
    print(f"{'='*60}\n")

    tickers = get_universe(args.universe)
    print(f"Universe: {len(tickers)} tickers")

    fetcher = DataFetcher(cfg.data)
    price_data = fetcher.fetch(tickers)
    spy_df = price_data.get(cfg.data.spy_ticker)

    pipeline = FeaturePipeline(cfg.features)
    X, y = pipeline.fit_transform(price_data, spy_df)
    print(f"Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"Class balance: {y.mean():.3f} (positive label rate)\n")

    trainer = ModelTrainer(cfg.model)
    metrics = trainer.train(X, y, final_fit=True)

    print(f"\nWalk-forward CV Results:")
    print(f"  AUC:  {metrics['auc_mean']:.4f} ± {metrics['auc_std']:.4f}")
    print(f"  AP:   {metrics['ap_mean']:.4f} ± {metrics['ap_std']:.4f}")

    # Evaluate on held-out last-year data
    last_year_start = str(int(cfg.data.end_date[:4]) - 1) + cfg.data.end_date[4:]
    X_val = X[X.index.get_level_values(-1) >= last_year_start] if hasattr(X.index, 'get_level_values') else X.iloc[int(len(X)*0.8):]
    y_val = y.loc[X_val.index]

    from models.predictor import Predictor
    predictor = Predictor(trainer)
    proba_val = predictor.predict_history(X_val)
    evaluate_model(y_val, proba_val, verbose=True)

    # Feature importance
    print("\nTop 15 Most Important Features:")
    imp = trainer.feature_importance(top_n=15)
    for feat, score in imp.items():
        bar = "█" * int(score * 300)
        print(f"  {feat:<30} {score:.4f}  {bar}")

    trainer.save()
    print("\nModel saved. Ready for backtesting and live signals.")


def cmd_backtest(args) -> None:
    """Run vectorised historical backtest."""
    from data.fetcher import DataFetcher
    from data.universe import get_universe
    from features.pipeline import FeaturePipeline
    from models.trainer import ModelTrainer
    from models.predictor import Predictor
    from backtesting.engine import BacktestEngine
    from backtesting.report import BacktestReport

    cfg = get_config()
    if args.start:
        cfg.backtest.start_date = args.start
    if args.end:
        cfg.backtest.end_date = args.end

    print(f"\n{'='*60}")
    print("  BACKTEST MODE")
    print(f"  Backtest window: {cfg.backtest.start_date} → {cfg.backtest.end_date}")
    print(f"{'='*60}\n")

    tickers = get_universe(args.universe)
    fetcher = DataFetcher(cfg.data)

    # Fetch training data (before backtest window for zero lookahead)
    print("Fetching price data...")
    price_data = fetcher.fetch(tickers)
    spy_df = price_data.get(cfg.data.spy_ticker)

    # Train on pre-backtest data
    train_end = cfg.backtest.start_date
    train_data = {t: df[df.index < train_end] for t, df in price_data.items() if len(df[df.index < train_end]) >= cfg.features.min_rows}

    print(f"Training on data before {train_end}...")
    pipeline = FeaturePipeline(cfg.features)
    X_train, y_train = pipeline.fit_transform(train_data, spy_df)

    trainer = ModelTrainer(cfg.model)
    trainer.train(X_train, y_train, final_fit=True)

    predictor = Predictor(trainer)

    # Generate predictions for the backtest window
    print("Generating predictions for backtest window...")
    bt_data = {t: df[(df.index >= cfg.backtest.start_date) & (df.index <= cfg.backtest.end_date)]
               for t, df in price_data.items()}
    spy_bt = spy_df[(spy_df.index >= cfg.backtest.start_date) & (spy_df.index <= cfg.backtest.end_date)] if spy_df is not None else None

    # Build per-ticker probability DataFrames and combine
    prob_frames = {}
    for ticker, df in bt_data.items():
        if ticker == cfg.data.spy_ticker or len(df) < 60:
            continue
        full_df = price_data[ticker]
        try:
            from features.technical import compute_features
            from features.regime import add_regime_features
            feat = compute_features(full_df, spy_df=spy_df,
                                     ma_windows=cfg.features.ma_windows,
                                     rsi_period=cfg.features.rsi_period,
                                     macd_fast=cfg.features.macd_fast,
                                     macd_slow=cfg.features.macd_slow,
                                     macd_signal=cfg.features.macd_signal,
                                     atr_period=cfg.features.atr_period,
                                     bb_period=cfg.features.bb_period,
                                     bb_std=cfg.features.bb_std,
                                     volume_ma_period=cfg.features.volume_ma_period,
                                     lookahead_days=0)
            feat = add_regime_features(feat, spy_df)
            feat.drop(columns=["forward_return", "target"], errors="ignore", inplace=True)
            bt_feat = feat[feat.index >= cfg.backtest.start_date]
            if len(bt_feat) == 0:
                continue
            bt_feat = bt_feat.reindex(columns=pipeline.feature_names, fill_value=0.0)
            import numpy as np
            scaled = pd.DataFrame(
                pipeline.scaler.transform(bt_feat),
                index=bt_feat.index,
                columns=pipeline.feature_names,
            )
            proba = predictor.predict_history(scaled)
            prob_frames[ticker] = proba
        except Exception as e:
            logging.getLogger(__name__).debug("Prediction failed for %s: %s", ticker, e)

    if not prob_frames:
        print("ERROR: No predictions generated. Check that data covers the backtest window.")
        return

    prob_df = pd.DataFrame(prob_frames)
    print(f"Prediction matrix: {prob_df.shape[0]} days × {prob_df.shape[1]} tickers")

    # Run backtest
    engine = BacktestEngine(cfg.backtest, cfg.portfolio, cfg.signals)
    equity_curve, trades_df = engine.run(prob_df, bt_data, spy_bt)

    # Benchmark: SPY buy-and-hold
    benchmark_equity = None
    if spy_bt is not None and len(spy_bt) > 0:
        spy_norm = spy_bt["Close"] / spy_bt["Close"].iloc[0] * cfg.portfolio.initial_capital
        benchmark_equity = spy_norm.reindex(equity_curve.index).ffill()

    # Generate report
    os.makedirs(cfg.results_dir, exist_ok=True)
    report = BacktestReport(cfg.results_dir)
    metrics = report.generate(equity_curve, trades_df, benchmark_equity, save_plots=True)

    print(f"\nResults saved to: {cfg.results_dir}/")
    print("Safe path: review results, then run paper trading before any live use.")


def cmd_signals(args) -> None:
    """Generate today's signals from saved model."""
    from data.fetcher import DataFetcher
    from data.universe import get_universe
    from features.pipeline import FeaturePipeline
    from models.trainer import ModelTrainer
    from models.predictor import Predictor
    from features.regime import MarketRegime
    from signals.generator import SignalGenerator

    cfg = get_config()

    print(f"\n{'='*60}")
    print(f"  DAILY SIGNALS")
    print(f"{'='*60}\n")

    tickers = get_universe(args.universe)
    fetcher = DataFetcher(cfg.data)
    price_data = fetcher.fetch(tickers)
    spy_df = price_data.get(cfg.data.spy_ticker)

    # Load model
    try:
        trainer = ModelTrainer.load(cfg.model)
    except FileNotFoundError:
        print("No trained model found. Run: python main.py train")
        return

    predictor = Predictor(trainer)

    # Load saved pipeline (scaler + feature names from setup)
    try:
        pipeline = FeaturePipeline.load(cfg.features)
    except FileNotFoundError:
        print("No saved pipeline found. Run: python3.10 main.py setup --universe small")
        return

    live_features = pipeline.transform_latest(price_data, spy_df)

    probabilities = predictor.predict(live_features)

    regime = MarketRegime(spy_df) if spy_df is not None else None
    sig_gen = SignalGenerator(cfg.signals, regime)
    signals = sig_gen.generate(probabilities, price_data)

    sig_gen.print_summary(signals)

    # Save to CSV
    os.makedirs("logs", exist_ok=True)
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    rows = []
    for s in signals:
        rows.append({
            "date": today,
            "ticker": s.ticker,
            "signal": s.signal.value,
            "probability": s.probability,
            "regime_score": s.regime_score,
            "trend_aligned": s.trend_aligned,
            "close": s.close_price,
        })
    pd.DataFrame(rows).to_csv(f"logs/signals_{today}.csv", index=False)
    print(f"Signals saved to logs/signals_{today}.csv")


def cmd_paper(args) -> None:
    """Run one day of paper trading."""
    from execution.paper_trader import PaperTrader

    cfg = get_config()
    trader = PaperTrader(cfg)
    trader.run_daily(universe_size=args.universe)


def cmd_setup(args) -> None:
    """Full initial setup: train + save paper state."""
    from execution.paper_trader import PaperTrader

    cfg = get_config()
    trader = PaperTrader(cfg)
    trader.setup(universe_size=args.universe)
    print("\nSetup complete. You can now run:")
    print("  python3.10 main.py paper      # daily swing paper trading")
    print("  python3.10 main.py signals    # view today's swing signals")
    print("  python3.10 main.py longterm   # long-term buy & hold + insider data")
    print("  python3.10 main.py backtest   # run historical backtest")


def cmd_longterm(args) -> None:
    """
    Long-term buy-and-hold analysis with SEC insider data.

    Scores each stock 0–100 across:
      Trend (30pts) + Momentum (25pts) + Relative Strength (25pts) + Insider Buying (20pts)
    """
    from data.fetcher import DataFetcher
    from data.universe import get_universe
    from signals.longterm import LongTermAnalyzer

    cfg = get_config()

    print(f"\n{'='*60}")
    print("  LONG-TERM BUY & HOLD  (3–6 month horizon)")
    print("  Incorporates SEC Form 4 legal insider transactions")
    print(f"{'='*60}\n")

    tickers = get_universe(args.universe)
    fetcher = DataFetcher(cfg.data)
    print("Downloading price data...")
    price_data = fetcher.fetch(tickers)
    spy_df = price_data.get(cfg.data.spy_ticker)

    no_insider = getattr(args, "no_insider", False)
    no_news    = getattr(args, "no_news", False)
    analyzer = LongTermAnalyzer(
        stop_loss_pct=0.15,
        take_profit_pct=0.40,
        fetch_insider=not no_insider,
        fetch_news=not no_news,
    )

    signals = analyzer.analyze(price_data, spy_df)
    analyzer.print_report(signals)

    # Save CSV
    os.makedirs("logs", exist_ok=True)
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    rows = [{
        "date": today,
        "ticker": s.ticker,
        "action": s.action,
        "score": round(s.score, 1),
        "close": round(s.close_price, 2),
        "momentum_6m_pct": round(s.momentum_6m * 100, 2),
        "momentum_3m_pct": round(s.momentum_3m * 100, 2),
        "rel_strength_3m_pct": round(s.rel_strength_3m * 100, 2),
        "above_200ma": s.above_200ma,
        "golden_cross": s.golden_cross,
        "insider_buys": s.insider_buys,
        "insider_buy_value": round(s.insider_buy_value, 0),
    } for s in signals]
    csv_path = f"logs/longterm_{today}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nFull results saved to {csv_path}")
    print("\nNOTE: Long-term positions use wider stops (−15%) and targets (+40%).")
    print("      Plan to hold for 60–180 days. Check signals monthly, not daily.")


def cmd_insider(args) -> None:
    """
    Show detailed SEC Form 4 insider transactions for one or more tickers.
    Tells you exactly WHO bought, HOW MUCH, and WHEN.
    """
    from data.insider import get_insider_signal, print_insider_summary, _get_cik

    tickers = [t.strip().upper() for t in args.ticker.split(",")]
    days = getattr(args, "days", 180)

    print(f"\n{'='*65}")
    print(f"  SEC FORM 4 INSIDER TRANSACTIONS  (last {days} days)")
    print(f"  Source: SEC EDGAR — 100% public, legal information")
    print(f"{'='*65}")
    print(f"  What this shows: when executives/directors/directors")
    print(f"  buy their OWN company stock on the open market.")
    print(f"  They must report it to the SEC within 2 business days.")
    print(f"{'='*65}\n")

    for ticker in tickers:
        print(f"  Looking up {ticker}...")

        # Show the CIK so user can verify on SEC.gov
        cik = _get_cik(ticker)
        if cik:
            print(f"  SEC CIK: {cik}  →  https://www.sec.gov/cgi-bin/browse-edgar?"
                  f"action=getcompany&CIK={cik}&type=4&dateb=&owner=include&count=40")

        signal = get_insider_signal(ticker, lookback_days=days)
        print_insider_summary(ticker, signal)

        if signal["n_buys"] == 0 and signal["n_sells"] == 0:
            print(f"  ℹ No open-market transactions found in last {days} days.")
            print(f"  This is common — insiders often go months without trading.")

        print()

    print("  KEY:")
    print("  BUY  = insider purchased stock on open market (bullish signal)")
    print("  SELL = insider sold stock (not always bearish — could be tax/diversify)")
    print("  Only 'P' (Purchase) and 'S' (Sale) codes = open market trades")
    print("  Excludes: option exercises, gifts, automatic plan sales (10b5-1)")
    print(f"\n  Verify yourself: https://www.sec.gov/cgi-bin/browse-edgar")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    os.makedirs("logs", exist_ok=True)
    parser = argparse.ArgumentParser(
        description="AI Investing System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])

    sub = parser.add_subparsers(dest="command", required=True)

    # Shared arguments
    def add_common(p):
        p.add_argument("--universe", default="default",
                       choices=["default", "small", "sp500"],
                       help="Stock universe to use")
        p.add_argument("--start", default=None, help="Override start date (YYYY-MM-DD)")
        p.add_argument("--end", default=None, help="Override end date (YYYY-MM-DD)")

    p_train = sub.add_parser("train", help="Train model on historical data")
    add_common(p_train)

    p_bt = sub.add_parser("backtest", help="Run historical backtest")
    add_common(p_bt)

    p_sig = sub.add_parser("signals", help="Generate today's signals")
    add_common(p_sig)

    p_paper = sub.add_parser("paper", help="Run daily paper trading")
    add_common(p_paper)

    p_setup = sub.add_parser("setup", help="Full initial setup")
    add_common(p_setup)

    p_lt = sub.add_parser("longterm", help="Long-term buy & hold analysis with insider data")
    add_common(p_lt)
    p_lt.add_argument("--no-insider", action="store_true",
                      help="Skip SEC insider data fetch (faster)")
    p_lt.add_argument("--no-news", action="store_true",
                      help="Skip news headlines fetch (faster)")

    p_ins = sub.add_parser("insider", help="Show SEC Form 4 insider transactions for a stock")
    p_ins.add_argument("--ticker", required=True,
                       help="Ticker(s) to look up, comma-separated e.g. --ticker NVDA,AAPL")
    p_ins.add_argument("--days", type=int, default=180,
                       help="How many days back to search (default 180)")

    args = parser.parse_args()
    setup_logging(args.log_level)

    dispatch = {
        "train": cmd_train,
        "backtest": cmd_backtest,
        "signals": cmd_signals,
        "paper": cmd_paper,
        "setup": cmd_setup,
        "longterm": cmd_longterm,
        "insider": cmd_insider,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
