"""
Legal insider transaction data — SEC Form 4 via OpenInsider.com

OpenInsider aggregates the same public SEC EDGAR Form 4 filings into a
clean HTML table that's easy to parse. 100% legal public information.

When executives/directors buy their OWN stock on the open market they must
report it to the SEC within 2 business days via Form 4.
This is a historically strong bullish signal.

NOT insider trading: we only use publicly disclosed data.
Actual trading on non-public material information is a federal crime.
"""
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def get_insider_signal(
    ticker: str,
    lookback_days: int = 180,
) -> Dict:
    """
    Fetch recent Form 4 insider transactions for a ticker via OpenInsider.

    Returns a dict with:
      n_buys, n_sells, buy_value, sell_value,
      net_sentiment, signal_score (0–1),
      latest_buy_date, insiders (list of transactions)
    """
    empty = {
        "n_buys": 0, "n_sells": 0,
        "buy_value": 0.0, "sell_value": 0.0,
        "net_sentiment": 0.0, "signal_score": 0.5,
        "latest_buy_date": None, "insiders": [],
        "source": "none",
    }

    df = _fetch_openinsider(ticker, lookback_days)
    if df is None or df.empty:
        return empty

    n_buys = n_sells = 0
    buy_value = sell_value = 0.0
    latest_buy: Optional[datetime] = None
    insiders = []

    for _, row in df.iterrows():
        txn_type = str(row.get("type", "")).strip().upper()
        value = _safe_float(row.get("value", 0))
        date = row.get("date")

        abs_value = abs(value)                           # always positive magnitude

        if txn_type == "P":                              # open-market purchase
            n_buys += 1
            buy_value += abs_value
            if date and (latest_buy is None or date > latest_buy):
                latest_buy = date
            insiders.append({
                "date": date, "name": row.get("name", "?"),
                "title": row.get("title", ""), "type": "P",
                "shares": abs(_safe_float(row.get("shares", 0))),
                "price": _safe_float(row.get("price", 0)),
                "value": abs_value, "direction": "BUY",
            })
        elif txn_type == "S":                            # open-market sale
            n_sells += 1
            sell_value += abs_value
            insiders.append({
                "date": date, "name": row.get("name", "?"),
                "title": row.get("title", ""), "type": "S",
                "shares": abs(_safe_float(row.get("shares", 0))),
                "price": _safe_float(row.get("price", 0)),
                "value": abs_value, "direction": "SELL",
            })

    # Signal score
    net = buy_value - sell_value
    total = buy_value + sell_value + 1
    raw_score = (net / total + 1) / 2

    recency_boost = 0.0
    if latest_buy:
        days_ago = (datetime.now() - latest_buy).days
        if days_ago < 14:
            recency_boost = 0.15
        elif days_ago < 30:
            recency_boost = 0.08

    signal_score = min(1.0, raw_score + recency_boost) if n_buys > 0 else raw_score

    return {
        "n_buys": n_buys, "n_sells": n_sells,
        "buy_value": buy_value, "sell_value": sell_value,
        "net_sentiment": net,
        "signal_score": float(signal_score),
        "latest_buy_date": latest_buy,
        "insiders": insiders,
        "source": "openinsider",
    }


def get_insider_signals_bulk(
    tickers: List[str],
    lookback_days: int = 180,
    delay: float = 1.0,
) -> Dict[str, Dict]:
    """Fetch insider signals for multiple tickers with polite rate-limiting."""
    results = {}
    for i, ticker in enumerate(tickers):
        try:
            results[ticker] = get_insider_signal(ticker, lookback_days)
            n = results[ticker]["n_buys"]
            status = f"{n} buy(s)" if n > 0 else "none"
            print(f"  [{i+1}/{len(tickers)}] {ticker:<6} insider: {status}")
            time.sleep(delay)
        except Exception as exc:
            logger.warning("Insider fetch failed for %s: %s", ticker, exc)
            results[ticker] = {
                "n_buys": 0, "n_sells": 0, "buy_value": 0.0,
                "net_sentiment": 0.0, "signal_score": 0.5,
                "latest_buy_date": None, "insiders": [],
            }
    return results


def print_insider_summary(ticker: str, signal: Dict) -> None:
    n_buys = signal.get("n_buys", 0)
    n_sells = signal.get("n_sells", 0)
    buy_val = signal.get("buy_value", 0)
    sell_val = signal.get("sell_value", 0)

    print(f"\n  {'='*60}")
    print(f"  Insider Activity — {ticker}  (SEC Form 4, open-market only)")
    print(f"  {'='*60}")
    print(f"  Purchases: {n_buys:>3} transactions   ${buy_val:>12,.0f}")
    print(f"  Sales:     {n_sells:>3} transactions   ${sell_val:>12,.0f}")

    if n_buys == 0 and n_sells == 0:
        print(f"  Result:    No open-market transactions found")
    elif buy_val > sell_val:
        print(f"  Result:    NET BUYING  ✅  ${buy_val - sell_val:,.0f} more bought than sold")
    else:
        print(f"  Result:    NET SELLING ❌  ${sell_val - buy_val:,.0f} more sold than bought")

    print(f"  Signal score: {signal.get('signal_score', 0):.2f}  (>0.6 = bullish)")

    lb = signal.get("latest_buy_date")
    if lb:
        print(f"  Latest buy: {lb.strftime('%Y-%m-%d')}")

    insiders = signal.get("insiders", [])
    if insiders:
        print(f"\n  {'Date':<12} {'Name':<28} {'Title':<20} {'Dir':<5} {'Value':>12}")
        print(f"  {'-'*80}")
        for t in insiders[:10]:
            date_str = t["date"].strftime("%Y-%m-%d") if t.get("date") else "?"
            direction = "BUY ✅" if t["direction"] == "BUY" else "SELL  "
            print(
                f"  {date_str:<12} {str(t.get('name','?')):<28} "
                f"{str(t.get('title','')):<20} {direction:<5} "
                f"${t.get('value', 0):>11,.0f}"
            )


# ------------------------------------------------------------------
# OpenInsider scraper
# ------------------------------------------------------------------

def _fetch_openinsider(ticker: str, lookback_days: int) -> Optional[pd.DataFrame]:
    """
    Scrape Form 4 data from OpenInsider.com for a given ticker.
    Returns open-market purchases (P) and sales (S) only.
    """
    url = (
        f"http://openinsider.com/screener"
        f"?s={ticker}&o=&pl=&ph=&ll=&lh="
        f"&fd={lookback_days}&fdr=&td=0&tdr=&fdlyl=&fdlyh=&daysago="
        f"&xp=1&xs=1"
        f"&vl=&vh=&ocl=&och="
        f"&sic1=-1&sicl=100&sich=9999"
        f"&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h=&oc2l=&oc2h="
        f"&sortcol=0&cnt=40&action=1"
    )
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        from io import StringIO
        tables = pd.read_html(StringIO(resp.text))
    except Exception as exc:
        logger.warning("OpenInsider fetch failed for %s: %s", ticker, exc)
        return None

    if not tables:
        return None

    # Find the transactions table — it has 'Trade Type' and 'Insider Name' columns
    df = None
    for t in tables:
        cols_str = " ".join(str(c) for c in t.columns)
        if "Trade" in cols_str and "Insider" in cols_str and len(t) > 0:
            df = t
            break

    if df is None or df.empty:
        return None

    return _parse_openinsider_table(df)


def _parse_openinsider_table(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the OpenInsider HTML table into a standard DataFrame."""
    # Clean column names: strip whitespace and non-breaking spaces
    df.columns = [str(c).replace("\xa0", " ").strip() for c in df.columns]

    # Map to standard names
    col_map = {
        "Trade Date":    "date",
        "Insider Name":  "name",
        "Title":         "title",
        "Trade Type":    "type",
        "Price":         "price",
        "Qty":           "shares",
        "Value":         "value",
    }
    df = df.rename(columns=col_map)

    # Keep only needed columns that exist
    keep = [c for c in ["date", "name", "title", "type", "price", "shares", "value"]
            if c in df.columns]
    df = df[keep].copy()

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Clean numeric columns: remove $, commas, +, − signs
    for col in ("price", "shares", "value"):
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(r"[\$,+\s]", "", regex=True)
                .str.replace(r"[−\-](?=\d)", "-", regex=True)   # normalise minus
                .pipe(pd.to_numeric, errors="coerce")
                .fillna(0.0)
            )

    # Classify: "P - Purchase" → "P",  "S - Sale" → "S"
    if "type" in df.columns:
        type_str = df["type"].astype(str).str.upper()
        df["type"] = type_str.apply(
            lambda x: "P" if x.startswith("P") else ("S" if x.startswith("S") else x)
        )
        # Keep only open-market P and S (exclude option exercises, gifts, etc.)
        df = df[df["type"].isin(["P", "S"])]

    return df.reset_index(drop=True)


def _safe_float(val) -> float:
    try:
        return float(str(val).replace(",", "").replace("$", "").strip())
    except (ValueError, TypeError):
        return 0.0
