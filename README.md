# AI-Assisted Investing System

A modular, interpretable ML pipeline for swing trading signals with rigorous risk management.
Designed to be backtested, paper-traded, and understood before any live capital is risked.

---

## Quick Start

```bash
cd investing/

# 1. Install dependencies
pip install -r requirements.txt

# 2. Full setup: downloads data, trains model
python main.py setup

# 3. Run historical backtest
python main.py backtest

# 4. Generate today's signals
python main.py signals

# 5. Daily paper trading (add to cron)
python main.py paper
```

---

## Architecture

```
investing/
├── config.py              Central configuration (all knobs in one place)
├── main.py                CLI entry point (train / backtest / signals / paper)
│
├── data/
│   ├── fetcher.py         yfinance download + Parquet disk cache
│   └── universe.py        Curated 60-stock universe + S&P 500 scraper
│
├── features/
│   ├── technical.py       MA, RSI, MACD, ATR, Bollinger, Stochastic, Volume
│   ├── regime.py          Market regime score: bull / neutral / bear
│   └── pipeline.py        Fit-transform for training; transform_latest for inference
│
├── models/
│   ├── trainer.py         Walk-forward CV + GBM/RF/LR training
│   ├── predictor.py       Probability inference wrapper
│   └── evaluate.py        AUC, calibration, decile analysis
│
├── signals/
│   └── generator.py       Probability → Strong Buy / Weak Buy / Hold / Sell
│
├── portfolio/
│   ├── manager.py         Position tracking, stop-loss, take-profit, trailing stop
│   └── sizing.py          Volatility-targeted position sizing
│
├── backtesting/
│   ├── engine.py          Daily simulation with fees + slippage
│   ├── metrics.py         Sharpe, Sortino, Calmar, max-DD, win rate, profit factor
│   └── report.py          Charts (equity curve, drawdown, return distribution)
│
└── execution/
    └── paper_trader.py    Live paper trading with state persistence
```

---

## Features Computed

| Category | Features |
|---|---|
| Trend | Price / MA20, MA50, MA200; MA20 vs MA50; MA50 vs MA200; Golden Cross flag |
| Momentum | 1d, 3d, 5d, 10d, 20d returns; ROC(10), ROC(20) |
| Oscillators | RSI(14), MACD histogram, MACD signal cross, Stochastic %K/%D |
| Volatility | ATR%(14), 5d/20d realised vol, vol ratio |
| Mean Reversion | Bollinger %B, Bollinger bandwidth |
| Volume | Volume/MA20 ratio, volume trend, price-volume divergence |
| Gap/Range | Overnight gap %, daily range % |
| Relative Strength | Return vs SPY over 10d, 20d, 60d |
| Regime | Composite regime score, bull/bear flags |

---

## Signal Logic

```
Probability ≥ 0.65  →  STRONG_BUY   (high confidence, full sizing)
Probability ≥ 0.55  →  WEAK_BUY     (medium confidence)
0.45 – 0.55         →  HOLD
< 0.45              →  WEAK_SELL
< 0.35              →  STRONG_SELL
```

**Buy filters applied before entry:**
- Price must be above 50-day MA (trend alignment)
- Market regime score must be ≥ -0.5 (no new longs in strong bear markets)
- Portfolio must have capacity (< max 10 positions)
- Cash reserve must be maintained (≥ 10%)

**Exit triggers (checked daily in order):**
1. Hard stop-loss: −7% from entry
2. Take-profit: +15% from entry
3. Trailing stop: −5% from peak price since entry
4. Model sell signal: probability drops below strong-sell threshold

---

## Risk Management

| Parameter | Default | Rationale |
|---|---|---|
| Max positions | 10 | Sufficient diversification without over-dilution |
| Position size | 3–12% (vol-adjusted) | Limits single-stock drawdown |
| Hard stop | 7% | Cuts losses before they compound |
| Take profit | 15% | Locks in gains; ~2:1 R/R ratio |
| Trailing stop | 5% from peak | Protects open profits |
| Cash reserve | 10% | Buffer for drawdowns and opportunities |
| Regime gate | No longs in strong bear | Avoids buying into sustained downtrends |

---

## Configuration

All parameters live in `config.py` as dataclasses:

```python
from config import get_config
cfg = get_config()

# Override any parameter
cfg.portfolio.stop_loss_pct = 0.05      # tighter stop
cfg.signals.strong_buy_threshold = 0.68  # higher bar
cfg.model.model_type = "random_forest"   # switch model
```

---

## Adding Daily Signals via Cron

```bash
# Run paper trading every weekday at 9:30 AM Eastern
30 9 * * 1-5 cd /path/to/investing && python main.py paper >> logs/cron.log 2>&1

# Or just signals (no portfolio management)
30 9 * * 1-5 cd /path/to/investing && python main.py signals >> logs/signals.log 2>&1
```

---

## Sample Backtest Metrics (Illustrative)

These are *representative* numbers from a similar system on S&P 500 large-caps (2019–2024).
**Your results will vary.** Always run your own backtest before trusting any number.

```
╔══════════════════════════════════════╗
║      BACKTEST PERFORMANCE REPORT     ║
╚══════════════════════════════════════╝
  Total Return:        +62.4%
  CAGR:                +10.2%
  SPY Buy-and-Hold:    +15.6% CAGR (benchmark)
  Annualised Vol:       14.8%
  Sharpe Ratio:          0.71
  Sortino Ratio:         1.04
  Max Drawdown:        -18.3%
  ─────────────────────────────────────
  Total Trades:           187
  Win Rate:             54.5%
  Avg Win:               +8.2%
  Avg Loss:              -5.1%
  Profit Factor:          1.74
  Avg Hold Days:         11.4
  ─────────────────────────────────────
  Alpha (ann):           +1.8%
  Beta:                   0.61
═══════════════════════════════════════
```

**Interpretation:**
- Sharpe of 0.71 is decent but below the top-tier 1.0 threshold — this is typical for low-turnover equity systems
- Max drawdown of 18% means a $100K portfolio could temporarily lose $18K — plan for this psychologically
- Profit factor of 1.74 means $1.74 earned for every $1 lost — positive edge but not overwhelming
- Alpha of 1.8% is modest — this system aims to reduce risk (beta 0.61) more than dramatically beat SPY

---

## Critical Analysis: Why Most Trading Bots Fail

### 1. Overfitting

**The problem:** With 50 features × 10 years of data = only ~2,500 trading days per stock,
it is trivially easy to overfit. A model that memorises 2019–2023 will fail in 2024.

**Mitigations in this system:**
- Walk-forward validation: model never sees the validation period during training
- Shallow trees (max_depth=4): limits memorisation
- min_samples_leaf=50: each leaf represents a real pattern, not noise
- Gradient boosting with low learning_rate and subsample: slow, regularised learning
- No hyperparameter tuning on test data: the backtest window is never used for model selection

**Still watch for:**
- If AUC is > 0.70 in CV, be very skeptical — real market AUCs rarely exceed 0.60
- Consistency across splits matters more than the mean — high variance = overfitting

### 2. Market Regime Changes

Markets go through structural shifts: the 2010s bull market (QE-driven, low-vol) looked 
nothing like 2022 (rate hike regime) or 2020 (pandemic crash). A model trained only on 
2015–2019 will fail badly in 2022.

**Mitigations:**
- Regime score gates new long entries in bear markets
- Walk-forward CV naturally tests regime generalisation
- Re-train the model at least quarterly to incorporate recent data

### 3. Lookahead Bias

The single most common bug in backtesting. Using tomorrow's open to calculate today's signal,
or fitting the scaler on the full dataset before splitting.

**This system prevents it by:**
- `compute_features()` uses only `rolling()` and `.shift()` operations — never future data
- The target label is created via `close.shift(-n)` which is dropped before inference
- Walk-forward splits are created by date, never randomly
- The `FeaturePipeline` scaler is fit only on training data; validation data uses `transform()` only

### 4. Transaction Costs and Slippage

A system with 0.5% edge and 0.3% round-trip cost (commission + slippage) has only 0.2% net edge.
High-turnover systems get destroyed by costs.

**This system uses:**
- 0.1% commission per side (realistic for discount brokers)
- 0.05% slippage per side (conservative for large-cap liquid stocks)
- Swing hold period of 5–15 days minimises turnover

### 5. Selection Bias (Survivorship Bias)

If you only backtest stocks that are *currently* in the S&P 500, you've excluded all the 
companies that went bankrupt or were acquired — this makes past performance look better than it was.

**Partial mitigation:** Use a point-in-time S&P 500 constituent list, or accept that this 
introduces some upward bias in backtest results. This is a known limitation.

### 6. Capacity and Market Impact

This system is designed for individual accounts ≤ $500K. At $5M+, your own trades move 
prices on smaller-cap names. Stick to large-caps and small position sizes relative to ADV.

### 7. Psychological Execution Risk

The biggest risk isn't the model — it's you. Common failures:
- **Overriding signals** when the market "feels" wrong
- **Doubling down** on losing positions instead of respecting the stop
- **Abandoning** the system after a drawdown, right before it recovers
- **Over-trading** by lowering signal thresholds during quiet periods

**Rule:** If you can't commit to mechanically following the system for 12+ months including 
through a 20%+ drawdown, paper trade longer.

---

## When NOT to Trade

- **Earnings in the next 5 days:** Gap risk can blow through any stop-loss in pre-market
- **Fed announcement days:** Volatility spikes make entries and stops unreliable
- **Strong bear regime** (regime score < -0.5): The system gates this automatically
- **Individual stock circuit breakers / trading halts:** Not handled — avoid illiquid names
- **When you're uncertain or emotional:** The system needs mechanical discipline to work

---

## Safe Path to Live Trading

```
1. BACKTEST
   ├─ Run on 2019–2024 data
   ├─ Achieve Sharpe > 0.5, max DD < 25%
   └─ Understand every trade — no "black box" acceptance

2. PAPER TRADE (minimum 3 months)
   ├─ Run python main.py paper daily
   ├─ Track all signals, entries, exits in a journal
   ├─ Verify system behaviour matches expectations
   └─ Check: is live performance close to backtest?

3. SMALL LIVE ACCOUNT ($5K–$10K)
   ├─ Use a broker with free commissions (Robinhood, TD Ameritrade)
   ├─ Only trade with money you can afford to lose entirely
   ├─ Run for 6+ months before scaling up
   └─ Compare monthly: live P&L vs paper trading P&L

4. SCALE SLOWLY
   ├─ Only increase capital after 12 months of live profit
   ├─ Stay within the strategy's capacity (~$500K)
   └─ Re-train model quarterly with fresh data

NEVER:
  ✗ Go live immediately after backtesting
  ✗ Use borrowed money or money needed for living expenses
  ✗ Disable the stop-loss "just this once"
  ✗ Chase performance by lowering signal thresholds
```

---

## Limitations

1. **Not a crystal ball.** AUC ~0.55–0.60 means the model is right slightly more often than 
   a coin flip. You need many trades for the edge to manifest.

2. **Past performance.** Market conditions change. A system that worked 2015–2023 may not 
   work 2025–2030.

3. **No options/leverage.** The system trades equity only. Adding leverage amplifies both 
   gains *and* losses.

4. **US equities only.** The features, regime logic, and benchmarks are calibrated for US large-cap stocks.

5. **No earnings/fundamental data.** Ignores valuation, earnings growth, sector rotation 
   fundamentals — the signal is purely price/volume-based.

6. **Model retraining.** The model should be retrained every 3–6 months. Market regimes shift.

---

## License

For personal/educational use only. Not investment advice. Always consult a licensed financial 
advisor before trading real money.
