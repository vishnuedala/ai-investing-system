"""
Technical feature engineering — pure pandas, no external TA libraries.

All indicators are computed with no lookahead bias:
every value at time T uses only data available at or before T.
"""
import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Low-level indicator functions
# ---------------------------------------------------------------------------

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram}
    )


def bollinger_bands(
    close: pd.Series, period: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    mid = sma(close, period)
    std = close.rolling(period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    bandwidth = (upper - lower) / mid.replace(0, np.nan)
    return pd.DataFrame(
        {"bb_upper": upper, "bb_mid": mid, "bb_lower": lower,
         "bb_pct_b": pct_b, "bb_bandwidth": bandwidth}
    )


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series,
    k_period: int = 14, d_period: int = 3
) -> pd.DataFrame:
    lowest_low = low.rolling(k_period, min_periods=k_period).min()
    highest_high = high.rolling(k_period, min_periods=k_period).max()
    range_ = (highest_high - lowest_low).replace(0, np.nan)
    k = 100 * (close - lowest_low) / range_
    d = k.rolling(d_period, min_periods=d_period).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d})


# ---------------------------------------------------------------------------
# Main feature computation
# ---------------------------------------------------------------------------

def compute_features(
    df: pd.DataFrame,
    spy_df: Optional[pd.DataFrame] = None,
    ma_windows: list = None,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    atr_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
    volume_ma_period: int = 20,
    lookahead_days: int = 10,
) -> pd.DataFrame:
    """
    Compute a rich set of technical features from OHLCV data.

    Parameters
    ----------
    df          : OHLCV DataFrame with columns Open, High, Low, Close, Volume
    spy_df      : Optional SPY DataFrame for relative-strength features
    lookahead_days : Forward-return window for the target label

    Returns
    -------
    DataFrame of features + 'target' column (1 if forward return > 0 else 0).
    Rows with NaN features are dropped.
    """
    if ma_windows is None:
        ma_windows = [20, 50, 200]

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    open_ = df["Open"]

    feat = pd.DataFrame(index=df.index)

    # --- Price-based MA features ---
    for w in ma_windows:
        ma = sma(close, w)
        feat[f"close_vs_ma{w}"] = (close / ma) - 1          # distance from MA
        feat[f"ma{w}_slope"] = ma.pct_change(5)               # MA momentum

    # Cross-MA signals
    ma20 = sma(close, 20)
    ma50 = sma(close, 50)
    ma200 = sma(close, 200)
    feat["ma20_vs_ma50"] = (ma20 / ma50) - 1
    feat["ma50_vs_ma200"] = (ma50 / ma200) - 1
    feat["golden_cross"] = (ma50 > ma200).astype(int)         # 1 in uptrend

    # --- Momentum / returns ---
    for period in [1, 3, 5, 10, 20]:
        feat[f"ret_{period}d"] = close.pct_change(period)

    # Rate of change
    feat["roc_10"] = close.pct_change(10)
    feat["roc_20"] = close.pct_change(20)

    # --- RSI ---
    feat["rsi"] = rsi(close, rsi_period)
    feat["rsi_vs_50"] = feat["rsi"] - 50                       # centered RSI

    # RSI regime flags
    feat["rsi_oversold"] = (feat["rsi"] < 30).astype(int)
    feat["rsi_overbought"] = (feat["rsi"] > 70).astype(int)

    # --- MACD ---
    macd_df = macd(close, macd_fast, macd_slow, macd_signal)
    feat["macd_hist"] = macd_df["macd_hist"]
    feat["macd_hist_change"] = macd_df["macd_hist"].diff()
    feat["macd_above_signal"] = (macd_df["macd"] > macd_df["macd_signal"]).astype(int)

    # --- Bollinger Bands ---
    bb = bollinger_bands(close, bb_period, bb_std)
    feat["bb_pct_b"] = bb["bb_pct_b"]
    feat["bb_bandwidth"] = bb["bb_bandwidth"]

    # --- ATR and volatility ---
    atr_series = atr(high, low, close, atr_period)
    feat["atr_pct"] = atr_series / close                       # normalised ATR
    feat["vol_20d"] = close.pct_change().rolling(20).std() * np.sqrt(252)
    feat["vol_5d"] = close.pct_change().rolling(5).std() * np.sqrt(252)
    feat["vol_ratio"] = feat["vol_5d"] / feat["vol_20d"].replace(0, np.nan)

    # --- Volume features ---
    vol_ma = sma(volume, volume_ma_period)
    feat["volume_ratio"] = volume / vol_ma.replace(0, np.nan)
    feat["volume_trend"] = vol_ma.pct_change(5)

    # Price-volume divergence: price up but volume down = weak trend
    price_chg = close.pct_change(5)
    vol_chg = volume.pct_change(5)
    feat["pv_divergence"] = np.sign(price_chg) * np.sign(vol_chg)

    # --- Stochastic oscillator ---
    stoch = stochastic(high, low, close)
    feat["stoch_k"] = stoch["stoch_k"]
    feat["stoch_d"] = stoch["stoch_d"]

    # --- Gap / overnight ---
    feat["gap_pct"] = (open_ / close.shift(1)) - 1

    # --- High-low range ---
    feat["daily_range_pct"] = (high - low) / close

    # --- Relative strength vs benchmark ---
    if spy_df is not None:
        spy_close = spy_df["Close"].reindex(df.index).ffill()
        for period in [10, 20, 60]:
            stock_ret = close.pct_change(period)
            spy_ret = spy_close.pct_change(period)
            feat[f"rel_strength_{period}d"] = stock_ret - spy_ret

    # --- Target label (NO lookahead — shifted at this step) ---
    # forward_return is the return from close[t] to close[t + n]
    forward_return = close.shift(-lookahead_days) / close - 1
    feat["target"] = (forward_return > 0).astype(int)
    feat["forward_return"] = forward_return   # kept for analysis, not for training

    # Drop NaN rows (caused by rolling windows and the forward shift)
    feat.dropna(inplace=True)

    return feat
