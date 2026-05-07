"""
Long-term buy-and-hold signal generator with live news.
Target hold: 3–6 months. Entry: trend + momentum + insider confirmation.
"""
import logging
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from data.insider import get_insider_signals_bulk

logger = logging.getLogger(__name__)


@dataclass
class LongTermSignal:
    ticker: str
    score: float
    close_price: float
    above_200ma: bool
    golden_cross: bool
    momentum_6m: float
    momentum_3m: float
    momentum_1m: float
    rel_strength_3m: float
    insider_score: float
    insider_buys: int
    insider_sells: int
    insider_buy_value: float
    insider_sell_value: float
    action: str                 # STRONG_BUY / BUY / WATCH / AVOID
    stop_loss: float
    take_profit: float
    reasons: List[str] = field(default_factory=list)
    news: List[Dict] = field(default_factory=list)


class LongTermAnalyzer:
    """Score stocks 0–100 and generate clean, readable reports with live news."""

    def __init__(
        self,
        stop_loss_pct: float = 0.15,
        take_profit_pct: float = 0.40,
        fetch_insider: bool = True,
        fetch_news: bool = True,
    ):
        self.stop_pct = stop_loss_pct
        self.tp_pct = take_profit_pct
        self.fetch_insider = fetch_insider
        self.fetch_news = fetch_news

    def analyze(
        self,
        price_data: Dict[str, pd.DataFrame],
        spy_df: Optional[pd.DataFrame] = None,
    ) -> List[LongTermSignal]:
        tickers = [t for t in price_data if t != "SPY" and len(price_data[t]) >= 200]

        # Insider data
        insider_data: Dict[str, Dict] = {}
        if self.fetch_insider:
            print(f"  📋 Fetching SEC Form 4 insider filings ({len(tickers)} stocks)...")
            insider_data = get_insider_signals_bulk(tickers, lookback_days=180, delay=1.0)

        # News
        news_data: Dict[str, List] = {}
        if self.fetch_news:
            print(f"  📰 Fetching latest news headlines...")
            news_data = _fetch_news_bulk(tickers)

        signals = []
        for ticker in tickers:
            try:
                sig = self._score(ticker, price_data[ticker], spy_df,
                                  insider_data.get(ticker, {}),
                                  news_data.get(ticker, []))
                signals.append(sig)
            except Exception as e:
                logger.debug("Score failed %s: %s", ticker, e)

        signals.sort(key=lambda s: s.score, reverse=True)
        return signals

    def print_report(self, signals: List[LongTermSignal]) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        buys  = [s for s in signals if s.action in ("STRONG_BUY", "BUY")]
        watch = [s for s in signals if s.action == "WATCH"]
        avoid = [s for s in signals if s.action == "AVOID"]

        print(f"\n{'━'*65}")
        print(f"  📈  LONG-TERM BUY & HOLD REPORT  —  {now}")
        print(f"  Strategy: 3–6 month holds  |  Stop −15%  |  Target +40%")
        print(f"{'━'*65}")

        # ── BUY section ───────────────────────────────────────────────
        if buys:
            print(f"\n  ✅  BUY CANDIDATES  ({len(buys)} stocks)\n")
            for s in buys:
                self._print_card(s)
        else:
            print("\n  ⚠️  No strong buy candidates today.\n")

        # ── WATCH section ─────────────────────────────────────────────
        if watch:
            print(f"  👀  WATCHING  ({len(watch)} stocks — improving but not ready)\n")
            for s in watch:
                trend = "✅ above 200MA" if s.above_200ma else "❌ below 200MA"
                print(f"     {s.ticker:<6}  ${s.close_price:>8.2f}  "
                      f"6m {s.momentum_6m*100:>+6.1f}%  {trend}")
            print()

        # ── AVOID section ─────────────────────────────────────────────
        if avoid:
            print(f"  🚫  AVOID  ({len(avoid)} stocks — downtrending)\n")
            for s in avoid:
                print(f"     {s.ticker:<6}  ${s.close_price:>8.2f}  "
                      f"6m {s.momentum_6m*100:>+6.1f}%  "
                      f"3m {s.momentum_3m*100:>+6.1f}%")
            print()

        print(f"{'━'*65}")
        print(f"  Insider data: SEC Form 4 (openinsider.com) — public legal filings")
        print(f"  News: Yahoo Finance live headlines")
        print(f"{'━'*65}\n")

    # ------------------------------------------------------------------

    def _print_card(self, s: LongTermSignal) -> None:
        action_icon = "🔥" if s.action == "STRONG_BUY" else "📈"
        print(f"  {action_icon}  {s.ticker}  —  ${s.close_price:.2f}  "
              f"(Score: {s.score:.0f}/100)")
        print(f"     {'─'*55}")

        # Price targets
        print(f"     💰 Entry now:   ${s.close_price:.2f}")
        print(f"     🛑 Stop loss:   ${s.stop_loss:.2f}  (−{self.stop_pct*100:.0f}%  "
              f"max loss per share: ${s.close_price - s.stop_loss:.2f})")
        print(f"     🎯 Take profit: ${s.take_profit:.2f}  (+{self.tp_pct*100:.0f}%  "
              f"potential gain: ${s.take_profit - s.close_price:.2f}/share)")

        # Returns
        trend_str = "✅ above 200-day MA" if s.above_200ma else "❌ below 200-day MA"
        gc_str    = "✅ Golden cross" if s.golden_cross else ""
        print(f"\n     📊 Performance:")
        print(f"        1-month:  {s.momentum_1m*100:>+6.1f}%")
        print(f"        3-month:  {s.momentum_3m*100:>+6.1f}%")
        print(f"        6-month:  {s.momentum_6m*100:>+6.1f}%")
        print(f"        vs SPY:   {s.rel_strength_3m*100:>+6.1f}% (3m outperformance)")
        print(f"        Trend:    {trend_str}  {gc_str}")

        # Insider activity
        print(f"\n     🏢 Insider Activity (last 6 months — SEC Form 4):")
        if s.insider_buys == 0 and s.insider_sells == 0:
            print(f"        No open-market transactions filed")
        else:
            if s.insider_buys > 0:
                print(f"        ✅ BUYING:  {s.insider_buys} purchase(s)  "
                      f"total ${s.insider_buy_value:,.0f}")
            if s.insider_sells > 0:
                print(f"        ❌ SELLING: {s.insider_sells} sale(s)  "
                      f"total ${s.insider_sell_value:,.0f}")
            net = s.insider_buy_value - s.insider_sell_value
            if net > 0:
                print(f"        👍 Net insider buying: +${net:,.0f}")
            else:
                print(f"        👎 Net insider selling: −${abs(net):,.0f}")

        # Why we like it
        if s.reasons:
            print(f"\n     💡 Why this stock:")
            for r in s.reasons[:4]:
                print(f"        → {r}")

        # Live news
        if s.news:
            print(f"\n     📰 Latest News:")
            for item in s.news[:3]:
                title = item.get("title", "")[:70]
                age   = item.get("age", "")
                print(f"        [{age}]  {title}")

        print()

    # ------------------------------------------------------------------

    def _score(self, ticker, df, spy_df, insider, news):
        close = df["Close"]
        price = float(close.iloc[-1])

        ma50  = close.rolling(50,  min_periods=40).mean().iloc[-1]
        ma200 = close.rolling(200, min_periods=150).mean().iloc[-1]
        above_200 = price > ma200
        golden    = ma50 > ma200

        # Trend score (30)
        trend_score = 0.0
        if above_200:
            trend_score += 18
            trend_score += min(8, (price / ma200 - 1) * 100)
        if golden:
            trend_score += 4

        # Momentum (25)
        def ret(d):
            return float(close.iloc[-1] / close.iloc[-d] - 1) if len(close) > d else 0.0
        m1, m3, m6 = ret(21), ret(63), ret(126)
        mom_score = 0.0
        mom_score += 15 if m6 > 0.20 else (10 if m6 > 0.10 else (6 if m6 > 0.05 else (2 if m6 > 0 else 0)))
        mom_score += 10 if m3 > 0.10 else (6 if m3 > 0.05 else (3 if m3 > 0 else 0))

        # Relative strength (25)
        rel3 = 0.0
        rel_score = 12.0
        if spy_df is not None:
            spy = spy_df["Close"].reindex(close.index).ffill()
            if len(spy) > 63:
                spy_r3 = float(spy.iloc[-1] / spy.iloc[-63] - 1)
                rel3 = m3 - spy_r3
                rel_score = (25 if rel3 > 0.10 else (18 if rel3 > 0.05 else
                             (10 if rel3 > 0 else (4 if rel3 > -0.05 else 0))))

        # Insider (20)
        n_buys      = insider.get("n_buys", 0)
        n_sells     = insider.get("n_sells", 0)
        buy_val     = insider.get("buy_value", 0.0)
        sell_val    = insider.get("sell_value", 0.0)
        ins_score   = (20 if n_buys >= 3 else (14 if n_buys == 2 else
                       (14 if n_buys == 1 and buy_val > 500_000 else
                        (8 if n_buys == 1 else (-5 if n_sells > 2 else 0)))))

        total = max(0.0, min(100.0, trend_score + mom_score + rel_score + ins_score))

        action = ("STRONG_BUY" if total >= 70 and above_200 else
                  "BUY"        if total >= 55 and above_200 else
                  "WATCH"      if total >= 35 else "AVOID")

        # Build reasons
        reasons = []
        if above_200:  reasons.append(f"Price is {(price/ma200-1)*100:.1f}% above its 200-day average — long-term uptrend intact")
        if golden:     reasons.append("50-day MA crossed above 200-day MA (golden cross) — bullish momentum shift")
        if m6 > 0.05:  reasons.append(f"Up {m6*100:.1f}% over the past 6 months — sustained strength")
        if rel3 > 0.03:reasons.append(f"Outperforming the S&P 500 by {rel3*100:.1f}% over 3 months")
        if n_buys > 0: reasons.append(f"{n_buys} insider purchase(s) totalling ${buy_val:,.0f} — executives buying own stock")

        return LongTermSignal(
            ticker=ticker, score=total, close_price=price,
            above_200ma=above_200, golden_cross=golden,
            momentum_6m=m6, momentum_3m=m3, momentum_1m=m1,
            rel_strength_3m=rel3,
            insider_score=float(insider.get("signal_score", 0.5)),
            insider_buys=n_buys, insider_sells=n_sells,
            insider_buy_value=buy_val, insider_sell_value=sell_val,
            action=action,
            stop_loss=round(price * (1 - self.stop_pct), 2),
            take_profit=round(price * (1 + self.tp_pct), 2),
            reasons=reasons, news=news,
        )


# ------------------------------------------------------------------
# News fetcher (Yahoo Finance via yfinance — free, no API key)
# ------------------------------------------------------------------

def _fetch_news_bulk(tickers: List[str]) -> Dict[str, List]:
    """Fetch the 3 most recent headlines per ticker using yfinance."""
    import yfinance as yf
    result = {}
    for ticker in tickers:
        try:
            raw_news = yf.Ticker(ticker).news or []
            items = []
            for n in raw_news[:4]:
                # yfinance ≥0.2.50 nests everything under 'content'
                content = n.get("content", n)
                title     = content.get("title", n.get("title", ""))
                pub_str   = content.get("pubDate", "")
                publisher = (content.get("provider", {}) or {}).get("displayName",
                             n.get("publisher", ""))
                # Parse ISO date or unix timestamp
                if pub_str:
                    try:
                        pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                        age = _age_str(int(pub_dt.timestamp()))
                    except Exception:
                        age = pub_str[:10]
                else:
                    age = _age_str(int(n.get("providerPublishTime", 0)))

                if title:
                    items.append({
                        "title":     title,
                        "publisher": publisher,
                        "age":       age,
                    })
            result[ticker] = items[:3]
        except Exception:
            result[ticker] = []
    return result


def _age_str(unix_ts: int) -> str:
    """Convert Unix timestamp to human-readable age string."""
    if not unix_ts:
        return "?"
    dt = datetime.fromtimestamp(unix_ts)
    diff = datetime.now() - dt
    if diff.days == 0:
        hours = diff.seconds // 3600
        return f"{hours}h ago" if hours > 0 else "just now"
    if diff.days == 1:
        return "yesterday"
    if diff.days < 7:
        return f"{diff.days}d ago"
    return dt.strftime("%b %d")
