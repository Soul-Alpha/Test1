"""
Prometheus Market Analysis System — Configuration
==================================================
Central configuration for all modules. Modify this file to tune engine
behaviour, database paths, API settings, and ML hyper-parameters.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# ─────────────────────────────────────────────
# Directory layout
# ─────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
DATA_DIR     = BASE_DIR / "data"
MODELS_DIR   = BASE_DIR / "models"
CHARTS_DIR   = BASE_DIR / "charts"
OUTPUTS_DIR  = BASE_DIR / "outputs"
STORAGE_DIR  = BASE_DIR / "storage"
PATTERNS_DIR = BASE_DIR / "patterns"

for _d in [DATA_DIR, MODELS_DIR, CHARTS_DIR, OUTPUTS_DIR, STORAGE_DIR, PATTERNS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Sub-configs (dataclasses for type safety)
# ─────────────────────────────────────────────

@dataclass
class MarketStructureConfig:
    """Swing-detection and structure-classification parameters."""
    pivot_sensitivity:      int   = 5      # bars to left/right for pivot confirmation
    min_swing_atr_mult:     float = 0.5    # ignore swings smaller than N × ATR
    bos_confirmation_bars:  int   = 2      # bars closing beyond level to confirm BOS
    atr_period:             int   = 14


@dataclass
class SupportResistanceConfig:
    """Zone-detection and clustering parameters."""
    zone_tolerance_atr: float = 0.3   # zone half-width as ATR fraction
    min_touches:        int   = 2     # minimum price touches for zone validity
    lookback_period:    int   = 200   # bars to look back
    volume_weight:      bool  = True  # weight zones by volume at touch


@dataclass
class CandlestickConfig:
    """Candlestick-pattern detection thresholds."""
    pin_bar_wick_ratio:   float = 2.0   # wick must be ≥ N × body for pin bar
    doji_body_pct:        float = 0.05  # body ≤ 5 % of range → doji
    engulf_overlap_pct:   float = 0.0   # how much the current body covers prev body


@dataclass
class ChartPatternConfig:
    """Chart-pattern detection settings."""
    triangle_tolerance:   float = 0.02  # 2 % tolerance for trendline fit
    double_top_pct:       float = 0.015 # two highs within 1.5 % = double top
    lookback_swings:      int   = 10    # swings to consider for pattern search


@dataclass
class FibonacciConfig:
    """Fibonacci retracement settings."""
    levels: List[float] = field(
        default_factory=lambda: [0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    )
    key_levels: List[float] = field(
        default_factory=lambda: [0.382, 0.5, 0.618, 0.786]
    )
    confluence_tolerance_pct: float = 0.003   # 0.3 % from fib = confluence


@dataclass
class LiquiditySMCConfig:
    """Smart Money Concepts and liquidity-pool settings."""
    equal_hl_tolerance_pct: float = 0.002   # 0.2 % = "equal" highs/lows
    fvg_min_size_atr:       float = 0.3     # minimum FVG size in ATR
    ob_lookback:            int   = 50      # bars to search for order blocks


@dataclass
class MultiTimeframeConfig:
    """Multi-timeframe analysis settings.

    All seven timeframes are always fetched regardless of which primary TF
    the bot is launched with.  The engine skips any TF whose data was not
    supplied, so missing TFs degrade gracefully to the next available one.
    Weights are normalised to sum=1.0 by the engine.
    """
    timeframes: List[str] = field(
        default_factory=lambda: ["1d", "4h", "1h", "30m", "15m", "5m", "1m"]
    )
    tf_weights: List[float] = field(
        default_factory=lambda: [0.30, 0.25, 0.18, 0.12, 0.08, 0.05, 0.02]
    )  # Daily dominant; 1M lowest weight (engine normalises to sum=1.0)


@dataclass
class MLConfig:
    """Machine-learning engine settings."""
    model_type:              str   = "xgboost"    # "xgboost" | "lightgbm" | "neural"
    min_samples_train:       int   = 50
    model_update_frequency:  int   = 100          # retrain every N new setups
    test_size:               float = 0.2
    n_estimators:            int   = 200
    max_depth:               int   = 6
    learning_rate:           float = 0.05
    feature_importance_plot: bool  = True


@dataclass
class BacktestConfig:
    """Backtesting engine settings."""
    initial_capital:  float = 10_000.0
    risk_per_trade:   float = 0.01           # 1 % per trade
    commission_pct:   float = 0.0003         # 0.03 % per side
    slippage_pct:     float = 0.0001
    default_rr:       float = 2.0   # fallback; live bot uses direction-specific values
    rr_min_long:      float = 4.0   # 1:4 R:R minimum for bullish (long) setups
    rr_min_short:     float = 2.0   # 1:2 R:R minimum for bearish (short) setups
    max_open_trades:  int   = 3


@dataclass
class APIConfig:
    """FastAPI settings."""
    host:          str  = "0.0.0.0"
    port:          int  = 8000
    reload:        bool = False
    max_upload_mb: int  = 20


@dataclass
class UIConfig:
    """Streamlit UI settings."""
    port:   int = 8501
    theme:  str = "dark"     # "dark" | "light"


@dataclass
class PrometheusConfig:
    """Master configuration object."""

    # ── Identity ──────────────────────────────
    system_name:    str = "Prometheus"
    version:        str = "1.0.0"
    default_asset:  str = "XAUUSD"

    # ── Sub-configs ───────────────────────────
    market_structure:  MarketStructureConfig  = field(default_factory=MarketStructureConfig)
    support_resistance: SupportResistanceConfig = field(default_factory=SupportResistanceConfig)
    candlestick:       CandlestickConfig      = field(default_factory=CandlestickConfig)
    chart_patterns:    ChartPatternConfig     = field(default_factory=ChartPatternConfig)
    fibonacci:         FibonacciConfig        = field(default_factory=FibonacciConfig)
    liquidity_smc:     LiquiditySMCConfig     = field(default_factory=LiquiditySMCConfig)
    multi_timeframe:   MultiTimeframeConfig   = field(default_factory=MultiTimeframeConfig)
    ml:                MLConfig               = field(default_factory=MLConfig)
    backtest:          BacktestConfig         = field(default_factory=BacktestConfig)
    api:               APIConfig              = field(default_factory=APIConfig)
    ui:                UIConfig               = field(default_factory=UIConfig)

    # ── Storage ───────────────────────────────
    database_url: str = f"sqlite:///{STORAGE_DIR}/prometheus.db"

    # ── Computation ───────────────────────────
    gpu_enabled:    bool = False
    num_workers:    int  = 4
    async_analysis: bool = True

    # ── Logging ───────────────────────────────
    log_level:    str  = "INFO"
    log_to_file:  bool = True
    log_file:     str  = str(BASE_DIR / "prometheus.log")


# Singleton instance used across the project
CONFIG = PrometheusConfig()


# ─────────────────────────────────────────────
# Logging setup (called once at import)
# ─────────────────────────────────────────────
def setup_logging(config: PrometheusConfig = CONFIG) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if config.log_to_file:
        handlers.append(logging.FileHandler(config.log_file))

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


setup_logging()
