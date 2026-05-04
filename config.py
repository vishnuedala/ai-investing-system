"""Central configuration for the AI investing system."""
from dataclasses import dataclass, field
from datetime import date
from typing import List


def _today() -> str:
    return date.today().strftime("%Y-%m-%d")


@dataclass
class DataConfig:
    start_date: str = "2015-01-01"
    end_date: str = field(default_factory=_today)  # always fetch up to today
    cache_dir: str = "data/cache"
    spy_ticker: str = "SPY"
    min_avg_volume: int = 500_000


@dataclass
class FeatureConfig:
    ma_windows: List[int] = field(default_factory=lambda: [20, 50, 200])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    volume_ma_period: int = 20
    # Forward-return target window (trading days)
    lookahead_days: int = 10
    # Minimum history rows required before computing features
    min_rows: int = 250


@dataclass
class ModelConfig:
    # Options: "gradient_boosting", "random_forest", "logistic_regression"
    # random_forest uses all CPU cores (n_jobs=-1) — much faster than gradient_boosting
    model_type: str = "random_forest"
    # RF hyperparameters
    n_estimators: int = 150
    max_depth: int = 6
    learning_rate: float = 0.05    # only used by gradient_boosting
    min_samples_leaf: int = 50
    subsample: float = 0.8         # only used by gradient_boosting
    # Walk-forward validation
    train_years: int = 3
    val_years: int = 1
    n_walk_forward_splits: int = 2
    model_dir: str = "models/saved"


@dataclass
class SignalConfig:
    strong_buy_threshold: float = 0.65
    weak_buy_threshold: float = 0.55
    weak_sell_threshold: float = 0.45
    strong_sell_threshold: float = 0.35
    # Only issue Buy signals when price is above this MA (trend alignment)
    trend_ma: int = 50
    # Minimum market regime score to allow new longs (-1=bear, 0=neutral, 1=bull)
    min_regime_score: float = -0.5


@dataclass
class PortfolioConfig:
    initial_capital: float = 100_000.0
    max_positions: int = 10
    base_position_size: float = 0.08    # 8% per position
    min_position_size: float = 0.03
    max_position_size: float = 0.12
    stop_loss_pct: float = 0.07         # 7% hard stop
    take_profit_pct: float = 0.15       # 15% profit target
    trailing_stop_pct: float = 0.05     # 5% trailing stop from peak
    min_cash_reserve: float = 0.10      # always keep 10% cash
    use_volatility_sizing: bool = True
    target_portfolio_vol: float = 0.15  # annualised vol target


@dataclass
class BacktestConfig:
    commission_pct: float = 0.001       # 0.1% per side
    slippage_pct: float = 0.0005        # 0.05% per side
    start_date: str = "2019-01-01"
    end_date: str = field(default_factory=_today)
    benchmark: str = "SPY"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    log_dir: str = "logs"
    results_dir: str = "results"


def get_config() -> Config:
    return Config()
