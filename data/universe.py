"""
Stock universe selection.

Provides a curated list of liquid S&P 500 large-caps as the default universe.
Optionally fetches the live S&P 500 constituent list from Wikipedia.
"""
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Curated 60-stock liquid universe across 11 GICS sectors.
# Large-cap, high-liquidity names — safe for daily swing trading.
DEFAULT_UNIVERSE: List[str] = [
    # Technology
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "AMD", "INTC", "CRM",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "BKNG",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "V", "MA", "AXP",
    # Healthcare
    "UNH", "JNJ", "LLY", "ABBV", "PFE", "MRK", "TMO", "ABT",
    # Industrials
    "CAT", "DE", "HON", "UPS", "BA", "GE", "RTX",
    # Energy
    "XOM", "CVX", "COP", "SLB",
    # Consumer Staples
    "PG", "KO", "PEP", "COST", "WMT",
    # Communication Services
    "NFLX", "DIS", "T", "VZ",
    # Materials
    "LIN", "FCX",
    # Real Estate
    "PLD", "AMT",
    # Utilities
    "NEE", "DUK",
]

# Market benchmark always fetched alongside the universe
BENCHMARK = "SPY"


def get_universe(size: str = "default") -> List[str]:
    """
    Return a list of tickers to trade.

    size:
        "default"  – 60-stock curated liquid universe
        "small"    – 20-stock core (tech/fin/healthcare)
        "sp500"    – attempt to fetch live S&P 500 list from Wikipedia
    """
    if size == "small":
        return DEFAULT_UNIVERSE[:20]
    if size == "sp500":
        return _fetch_sp500()
    return list(DEFAULT_UNIVERSE)


def _fetch_sp500() -> List[str]:
    """Scrape current S&P 500 constituents from Wikipedia."""
    try:
        import pandas as pd
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        tickers = tables[0]["Symbol"].tolist()
        # Normalise dot notation (e.g. BRK.B → BRK-B for yfinance)
        tickers = [t.replace(".", "-") for t in tickers]
        logger.info("Fetched %d S&P 500 tickers from Wikipedia", len(tickers))
        return tickers
    except Exception as exc:
        logger.warning("Could not fetch S&P 500 list (%s). Using default universe.", exc)
        return list(DEFAULT_UNIVERSE)
