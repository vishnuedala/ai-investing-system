"""
OHLCV data fetcher with disk caching.

Downloads daily bars from Yahoo Finance via yfinance and caches them as
Parquet files so repeated runs don't hit the network.
"""
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from config import DataConfig

logger = logging.getLogger(__name__)


class DataFetcher:
    """Download and cache daily OHLCV data."""

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        os.makedirs(cfg.cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self,
        tickers: List[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Return a dict of {ticker: OHLCV DataFrame} for all tickers.

        DataFrames have columns: Open, High, Low, Close, Volume, Adj_Close.
        Index is a DatetimeIndex (UTC-naive).
        """
        start = start or self.cfg.start_date
        end = end or self.cfg.end_date

        data: Dict[str, pd.DataFrame] = {}
        missing: List[str] = []

        for ticker in tickers:
            cached = self._load_cache(ticker)
            if cached is not None and not force_refresh:
                # Extend cache if end date is beyond what we have
                cached = self._maybe_extend(cached, ticker, end)
                df = cached.loc[start:end]
                if len(df) >= 20:
                    data[ticker] = df
                    continue
            missing.append(ticker)

        if missing:
            logger.info("Downloading %d tickers from Yahoo Finance...", len(missing))
            downloaded = self._download_batch(missing, start, end)
            for ticker, df in downloaded.items():
                if df is not None and len(df) >= 20:
                    self._save_cache(ticker, df)
                    data[ticker] = df.loc[start:end]
                else:
                    logger.warning("Skipping %s — insufficient data", ticker)

        # Always fetch benchmark
        spy = self.fetch_single(self.cfg.spy_ticker, start, end)
        if spy is not None:
            data[self.cfg.spy_ticker] = spy

        logger.info("Loaded data for %d tickers", len(data))
        return data

    def fetch_single(
        self, ticker: str, start: Optional[str] = None, end: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Fetch a single ticker, returning None on failure."""
        start = start or self.cfg.start_date
        end = end or self.cfg.end_date
        result = self.fetch([ticker], start, end)
        return result.get(ticker)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_path(self, ticker: str) -> str:
        return os.path.join(self.cfg.cache_dir, f"{ticker}.parquet")

    def _load_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(ticker)
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as exc:
            logger.debug("Cache read failed for %s: %s", ticker, exc)
            return None

    def _save_cache(self, ticker: str, df: pd.DataFrame) -> None:
        try:
            df.to_parquet(self._cache_path(ticker))
        except Exception as exc:
            logger.debug("Cache write failed for %s: %s", ticker, exc)

    def _maybe_extend(
        self, cached: pd.DataFrame, ticker: str, end: str
    ) -> pd.DataFrame:
        """Append newer data if the cache is stale."""
        last_date = cached.index[-1]
        end_dt = pd.Timestamp(end)
        # Leave a 2-day buffer for weekends/holidays
        if last_date >= end_dt - timedelta(days=2):
            return cached

        new_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        logger.debug("Extending cache for %s from %s", ticker, new_start)
        ext = self._download_single(ticker, new_start, end)
        if ext is not None and len(ext) > 0:
            combined = pd.concat([cached, ext])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
            self._save_cache(ticker, combined)
            return combined
        return cached

    def _download_single(
        self, ticker: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            return self._clean(raw) if not raw.empty else None
        except Exception as exc:
            logger.warning("Download failed for %s: %s", ticker, exc)
            return None

    def _download_batch(
        self, tickers: List[str], start: str, end: str
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """Download multiple tickers in one yfinance call."""
        result: Dict[str, Optional[pd.DataFrame]] = {}

        # yfinance handles batches but returns a MultiIndex DataFrame
        # We process in chunks of 50 to avoid timeout issues
        chunk_size = 50
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i : i + chunk_size]
            try:
                raw = yf.download(
                    chunk,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                    group_by="ticker",
                )
            except Exception as exc:
                logger.warning("Batch download failed: %s", exc)
                for t in chunk:
                    result[t] = self._download_single(t, start, end)
                continue

            for ticker in chunk:
                try:
                    if len(chunk) == 1:
                        df = raw
                    else:
                        df = raw[ticker] if ticker in raw.columns.get_level_values(0) else None
                    if df is not None and not df.empty:
                        result[ticker] = self._clean(df)
                    else:
                        result[ticker] = None
                except Exception as exc:
                    logger.debug("Extract failed for %s: %s", ticker, exc)
                    result[ticker] = self._download_single(t, start, end)

        return result

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        """Standardise column names and drop rows with missing OHLCV."""
        df = df.copy()
        df.columns = [c.strip().title().replace(" ", "_") for c in df.columns]
        # Rename common yfinance column variants
        rename = {
            "Adj_Close": "Adj_Close",
            "Adj Close": "Adj_Close",
            "Close": "Close",
        }
        df.rename(columns=rename, inplace=True)

        # Ensure we have the standard columns
        required = ["Open", "High", "Low", "Close", "Volume"]
        df = df[[c for c in required if c in df.columns]]
        df.dropna(subset=["Close"], inplace=True)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.sort_index(inplace=True)
        return df
