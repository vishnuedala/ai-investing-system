"""
OHLCV data fetcher with disk caching.

Downloads daily bars from Yahoo Finance via yfinance and caches them as
Parquet files so repeated runs don't hit the network.

Compatible with yfinance ≥0.2.x which returns MultiIndex columns for all
downloads (single and batch alike).
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
        Return {ticker: OHLCV DataFrame} for all tickers.
        Columns are always: Open, High, Low, Close, Volume.
        Index is a tz-naive DatetimeIndex.
        """
        start = start or self.cfg.start_date
        end = end or self.cfg.end_date

        data: Dict[str, pd.DataFrame] = {}
        missing: List[str] = []

        for ticker in tickers:
            cached = self._load_cache(ticker)
            if cached is not None and not force_refresh:
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

        # Include benchmark (fetch directly to avoid recursion)
        if self.cfg.spy_ticker not in data:
            spy = self.fetch_single(self.cfg.spy_ticker, start, end)
            if spy is not None:
                data[self.cfg.spy_ticker] = spy

        logger.info("Loaded data for %d tickers", len(data))
        return data

    def fetch_single(
        self, ticker: str, start: Optional[str] = None, end: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Fetch one ticker directly (does not trigger the SPY auto-add)."""
        start = start or self.cfg.start_date
        end = end or self.cfg.end_date

        cached = self._load_cache(ticker)
        if cached is not None:
            cached = self._maybe_extend(cached, ticker, end)
            df = cached.loc[start:end]
            if len(df) >= 20:
                return df

        df = self._download_single(ticker, start, end)
        if df is not None and len(df) >= 20:
            self._save_cache(ticker, df)
            return df.loc[start:end]
        return None

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
        last_date = cached.index[-1]
        end_dt = pd.Timestamp(end)

        # If cache is more than 7 days stale, always try to extend
        days_stale = (end_dt - last_date).days
        if days_stale <= 2:
            return cached

        if days_stale > 7:
            logger.warning(
                "%s cache is %d days stale (last: %s) — refreshing...",
                ticker, days_stale, last_date.date()
            )

        new_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        ext = self._download_single(ticker, new_start, end)
        if ext is not None and len(ext) > 0:
            combined = pd.concat([cached, ext])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
            self._save_cache(ticker, combined)
            logger.info("%s updated: %s → %s", ticker,
                        last_date.date(), combined.index[-1].date())
            return combined

        logger.warning("%s cache extension failed — still using data up to %s",
                       ticker, last_date.date())
        return cached

    def _download_single(
        self, ticker: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """Download one ticker and return a clean flat DataFrame."""
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if raw.empty:
                return None
            # yfinance ≥0.2 returns MultiIndex (Price, Ticker) for single downloads
            flat = self._flatten(raw, ticker)
            return self._clean(flat)
        except Exception as exc:
            logger.warning("Download failed for %s: %s", ticker, exc)
            return None

    def _download_batch(
        self, tickers: List[str], start: str, end: str
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """Download multiple tickers in one call, return flat per-ticker DataFrames."""
        result: Dict[str, Optional[pd.DataFrame]] = {}

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
                for ticker in chunk:
                    result[ticker] = self._download_single(ticker, start, end)
                continue

            for ticker in chunk:
                try:
                    flat = self._flatten(raw, ticker)
                    result[ticker] = self._clean(flat) if flat is not None and not flat.empty else None
                except Exception as exc:
                    logger.debug("Extract failed for %s: %s — retrying single", ticker, exc)
                    result[ticker] = self._download_single(ticker, start, end)

        return result

    @staticmethod
    def _flatten(raw: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        """
        Extract a flat (non-MultiIndex) OHLCV DataFrame for one ticker.

        yfinance ≥0.2.x column layouts we handle:
          Single-ticker:  MultiIndex (Price, Ticker)  e.g. ('Close', 'AAPL')
          Multi-ticker:   MultiIndex (Ticker, Price)  e.g. ('AAPL', 'Close')
          Legacy:         flat strings                e.g. 'Close'
        """
        if not isinstance(raw.columns, pd.MultiIndex):
            # Already flat — legacy yfinance or single already extracted
            return raw

        level0_vals = raw.columns.get_level_values(0).unique().tolist()
        level1_vals = raw.columns.get_level_values(1).unique().tolist()

        # Multi-ticker layout: level 0 = Ticker, level 1 = Price field
        if ticker in level0_vals:
            return raw[ticker].copy()

        # Single-ticker layout: level 0 = Price field, level 1 = Ticker
        if ticker in level1_vals:
            df = raw.xs(ticker, axis=1, level=1).copy()
            return df

        logger.warning("Ticker %s not found in columns %s", ticker, raw.columns.tolist()[:6])
        return None

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        """Normalise column names and drop rows with missing Close."""
        df = df.copy()

        # If somehow still MultiIndex, flatten to strings
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(str(p) for p in col if p).strip() for col in df.columns]

        df.columns = [str(c).strip().title().replace(" ", "_") for c in df.columns]

        required = ["Open", "High", "Low", "Close", "Volume"]
        available = [c for c in required if c in df.columns]
        if not available or "Close" not in available:
            raise ValueError(f"Missing required columns. Got: {df.columns.tolist()}")

        df = df[available]
        df.dropna(subset=["Close"], inplace=True)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.sort_index(inplace=True)
        return df
