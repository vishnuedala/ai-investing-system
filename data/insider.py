"""
Legal insider transaction data from SEC EDGAR Form 4 filings.

When a corporate insider (CEO, CFO, Director) buys their own company's stock,
they must file a Form 4 with the SEC within 2 business days. This is 100%
public information and a historically strong bullish signal.

Source: SEC EDGAR public API (free, no API key needed)
  https://www.sec.gov/cgi-bin/browse-edgar

NOT insider trading: we only use publicly disclosed information.
Actual insider trading (non-public material info) is a federal crime.
"""
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# SEC requires a descriptive User-Agent header
_HEADERS = {
    "User-Agent": "AI-Investing-System research@example.com",
    "Accept-Encoding": "gzip, deflate",
}

# Map ticker → CIK (loaded once, cached in memory)
_TICKER_TO_CIK: Dict[str, str] = {}


def get_insider_signal(
    ticker: str,
    lookback_days: int = 90,
) -> Dict:
    """
    Fetch recent Form 4 insider BUY transactions for a ticker.

    Returns a dict with:
      n_buys          : number of insider buy transactions in lookback window
      n_sells         : number of insider sell transactions
      buy_value       : total $ value of insider purchases
      sell_value      : total $ value of insider sales
      net_sentiment   : buy_value - sell_value  (positive = net buying)
      signal_score    : 0.0–1.0  (1.0 = strong insider buying)
      latest_buy_date : most recent insider buy date (or None)
      insiders        : list of recent transactions
    """
    empty = {
        "n_buys": 0, "n_sells": 0, "buy_value": 0.0, "sell_value": 0.0,
        "net_sentiment": 0.0, "signal_score": 0.0,
        "latest_buy_date": None, "insiders": [],
    }

    cik = _get_cik(ticker)
    if cik is None:
        logger.debug("CIK not found for %s", ticker)
        return empty

    filings = _fetch_form4_filings(cik, lookback_days)
    if not filings:
        return empty

    n_buys, n_sells = 0, 0
    buy_value, sell_value = 0.0, 0.0
    latest_buy: Optional[datetime] = None
    insiders = []

    for f in filings:
        txn_type = f.get("type", "").upper()
        value = f.get("value", 0.0) or 0.0
        date = f.get("date")

        if txn_type in ("P", "BUY", "PURCHASE"):           # Open-market purchase
            n_buys += 1
            buy_value += value
            if date and (latest_buy is None or date > latest_buy):
                latest_buy = date
            insiders.append({**f, "direction": "BUY"})

        elif txn_type in ("S", "SELL", "SALE"):             # Open-market sale
            n_sells += 1
            sell_value += value
            insiders.append({**f, "direction": "SELL"})

    # Signal score: weighted by recency and net buy dominance
    net = buy_value - sell_value
    total = buy_value + sell_value + 1  # avoid div/0
    raw_score = (net / total + 1) / 2   # map [-1,+1] → [0,1]

    # Boost score if buy is very recent (< 30 days)
    recency_boost = 0.0
    if latest_buy:
        days_ago = (datetime.now() - latest_buy).days
        if days_ago < 14:
            recency_boost = 0.15
        elif days_ago < 30:
            recency_boost = 0.08

    signal_score = min(1.0, raw_score + recency_boost) if n_buys > 0 else raw_score

    return {
        "n_buys": n_buys,
        "n_sells": n_sells,
        "buy_value": buy_value,
        "sell_value": sell_value,
        "net_sentiment": net,
        "signal_score": float(signal_score),
        "latest_buy_date": latest_buy,
        "insiders": insiders[:10],
    }


def get_insider_signals_bulk(
    tickers: List[str],
    lookback_days: int = 90,
    delay: float = 0.3,
) -> Dict[str, Dict]:
    """
    Fetch insider signals for multiple tickers.
    Adds a small delay between requests to respect SEC rate limits.
    """
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = get_insider_signal(ticker, lookback_days)
            time.sleep(delay)
        except Exception as exc:
            logger.warning("Insider fetch failed for %s: %s", ticker, exc)
            results[ticker] = {"signal_score": 0.5, "n_buys": 0, "n_sells": 0,
                               "buy_value": 0.0, "net_sentiment": 0.0, "latest_buy_date": None}
    return results


def print_insider_summary(ticker: str, signal: Dict) -> None:
    print(f"\n  Insider Activity — {ticker} (last 90 days)")
    print(f"  Buys:  {signal['n_buys']} transactions  ${signal['buy_value']:>12,.0f}")
    print(f"  Sells: {signal['n_sells']} transactions  ${signal['sell_value']:>12,.0f}")
    net = signal['net_sentiment']
    direction = "NET BUYING ✅" if net > 0 else "NET SELLING ❌"
    print(f"  Net:   {direction}  ${abs(net):,.0f}")
    print(f"  Signal score: {signal['signal_score']:.2f}")
    if signal.get("latest_buy_date"):
        print(f"  Latest buy: {signal['latest_buy_date'].strftime('%Y-%m-%d')}")
    for txn in signal["insiders"][:5]:
        print(f"    {txn.get('date','?'):%Y-%m-%d}  {txn.get('name','?'):<30} "
              f"{txn['direction']}  ${txn.get('value',0):>10,.0f}")


# ------------------------------------------------------------------
# SEC EDGAR helpers
# ------------------------------------------------------------------

def _get_cik(ticker: str) -> Optional[str]:
    """Map ticker symbol → 10-digit CIK string using SEC company_tickers.json."""
    global _TICKER_TO_CIK

    if not _TICKER_TO_CIK:
        try:
            resp = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers=_HEADERS, timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            _TICKER_TO_CIK = {
                v["ticker"].upper(): str(v["cik_str"]).zfill(10)
                for v in data.values()
            }
        except Exception as exc:
            logger.warning("Could not load SEC ticker map: %s", exc)
            return None

    return _TICKER_TO_CIK.get(ticker.upper())


def _fetch_form4_filings(cik: str, lookback_days: int) -> List[Dict]:
    """
    Fetch recent Form 4 filings for a CIK via SEC EDGAR submissions API.
    Returns list of transaction dicts.
    """
    try:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("EDGAR submissions fetch failed (CIK %s): %s", cik, exc)
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])

    cutoff = datetime.now() - timedelta(days=lookback_days)
    transactions = []

    for form, date_str, acc in zip(forms, dates, accessions):
        if form != "4":
            continue
        try:
            filing_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        if filing_date < cutoff:
            continue

        # Parse the actual Form 4 XML for transaction details
        txns = _parse_form4_xml(cik, acc, filing_date)
        transactions.extend(txns)

    return transactions


def _parse_form4_xml(cik: str, accession: str, filing_date: datetime) -> List[Dict]:
    """Download and parse a Form 4 XML file for transaction details."""
    acc_clean = accession.replace("-", "")
    base_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}"
    index_url = f"{base_url}/{accession}-index.htm"

    try:
        resp = requests.get(index_url, headers=_HEADERS, timeout=10)
        resp.raise_for_status()

        # Find the .xml file link in the index
        import re
        xml_files = re.findall(r'href="([^"]+\.xml)"', resp.text, re.IGNORECASE)
        if not xml_files:
            return []

        xml_url = f"https://www.sec.gov{xml_files[0]}" if xml_files[0].startswith("/") else \
                  f"{base_url}/{xml_files[0]}"

        xml_resp = requests.get(xml_url, headers=_HEADERS, timeout=10)
        xml_resp.raise_for_status()
        return _extract_transactions(xml_resp.text, filing_date)

    except Exception as exc:
        logger.debug("Form 4 XML parse failed (%s): %s", accession, exc)
        return []


def _extract_transactions(xml_text: str, filing_date: datetime) -> List[Dict]:
    """Extract non-derivative transactions from Form 4 XML."""
    import xml.etree.ElementTree as ET
    transactions = []
    try:
        root = ET.fromstring(xml_text)

        # Insider name
        reporter = root.find(".//reportingOwner/reportingOwnerId/rptOwnerName")
        name = reporter.text.strip() if reporter is not None and reporter.text else "Unknown"

        # Title
        title_el = root.find(".//reportingOwner/reportingOwnerRelationship/officerTitle")
        title = title_el.text.strip() if title_el is not None and title_el.text else ""

        for txn in root.findall(".//nonDerivativeTransaction"):
            code_el = txn.find("transactionAmounts/transactionAcquiredDisposedCode/value")
            shares_el = txn.find("transactionAmounts/transactionShares/value")
            price_el = txn.find("transactionAmounts/transactionPricePerShare/value")
            code_type_el = txn.find("transactionCoding/transactionCode")

            if code_el is None or code_type_el is None:
                continue

            ad_code = code_el.text.strip() if code_el.text else ""
            txn_code = code_type_el.text.strip() if code_type_el.text else ""

            # Only open-market buys (P) and sells (S) — skip options exercises, gifts, etc.
            if txn_code not in ("P", "S"):
                continue

            try:
                shares = float(shares_el.text) if shares_el is not None and shares_el.text else 0
                price = float(price_el.text) if price_el is not None and price_el.text else 0
                value = shares * price
            except (ValueError, TypeError):
                shares, price, value = 0, 0, 0

            transactions.append({
                "date": filing_date,
                "name": name,
                "title": title,
                "type": txn_code,
                "shares": shares,
                "price": price,
                "value": value,
            })
    except ET.ParseError as exc:
        logger.debug("XML parse error: %s", exc)

    return transactions
