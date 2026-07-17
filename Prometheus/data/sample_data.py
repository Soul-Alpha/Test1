"""
Synthetic OHLCV data generators for testing and demos.
All prices are calibrated to XAUUSD (Gold/USD) ranges.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd


# ── XAUUSD synthetic generator ───────────────────────────────────────────────

def generate_xauusd_ohlcv(
    n_bars:    int   = 500,
    timeframe: str   = "4H",
    start_price: float = 2000.0,
    seed:      int   = 42,
) -> pd.DataFrame:
    """
    Generate synthetic XAUUSD OHLCV data with realistic price action.

    Features:
    - Random-walk price with momentum and mean-reversion components
    - Realistic spread / intra-bar volatility
    - Volume correlated with price movement
    - Occasional spikes (stop hunts) and strong directional moves

    Args:
        n_bars:      Number of bars to generate.
        timeframe:   Label only (used for timestamp step calculation).
        start_price: Starting price.
        seed:        Random seed for reproducibility.

    Returns:
        DataFrame with columns: open, high, low, close, volume
    """
    rng = np.random.default_rng(seed)

    # Map timeframe to minutes
    tf_minutes = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "4h": 240, "4H": 240,
        "1d": 1440, "1w": 10080,
    }
    step_minutes = tf_minutes.get(timeframe, 240)

    # --- Price series ---
    returns    = rng.normal(0, 0.0008, n_bars)          # base returns
    momentum   = pd.Series(returns).ewm(span=8).mean().values  # momentum component
    price_log  = np.cumsum(returns * 0.6 + momentum * 0.4)
    closes     = start_price * np.exp(price_log)

    # Inject a few trending regimes
    n_regimes = n_bars // 80
    for _ in range(n_regimes):
        start_idx = rng.integers(0, n_bars - 40)
        length    = rng.integers(20, 50)
        direction = rng.choice([-1, 1])
        drift     = rng.uniform(0.0005, 0.002) * direction
        end_idx   = min(start_idx + length, n_bars)
        closes[start_idx:end_idx] *= np.exp(
            np.cumsum(np.full(end_idx - start_idx, drift))
        )

    # Recalculate log-returns after regime injection
    log_returns = np.diff(np.log(closes), prepend=np.log(closes[0]))

    # --- OHLCV construction ---
    records = []
    timestamp = datetime(2024, 1, 1, 0, 0)

    for i in range(n_bars):
        close = closes[i]
        # Intra-bar volatility ~ ATR proxy
        vol   = close * rng.uniform(0.001, 0.004)

        o = closes[i - 1] if i > 0 else start_price
        c = close
        direction = 1 if c >= o else -1

        # High and low
        high = max(o, c) + abs(rng.normal(0, vol))
        low  = min(o, c) - abs(rng.normal(0, vol))

        # Spike probability for stop hunts
        if rng.random() < 0.03:
            spike = rng.uniform(1.0, 2.0) * vol
            if rng.random() < 0.5:
                high += spike    # wick up
            else:
                low  -= spike    # wick down

        volume = rng.uniform(800, 2500) * (1 + 3 * abs(c - o) / close)

        records.append({
            "timestamp": timestamp,
            "open":   round(o,    4),
            "high":   round(high, 4),
            "low":    round(low,  4),
            "close":  round(c,    4),
            "volume": round(volume, 1),
        })
        timestamp += timedelta(minutes=step_minutes)

    df = pd.DataFrame(records).set_index("timestamp")

    # Sanitize: high >= max(o,c), low <= min(o,c)
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"]  = df[["open", "close", "low"]].min(axis=1)

    return df


def generate_trending_data(
    n_bars:    int   = 300,
    direction: str   = "bullish",
    start_price: float = 2000.0,
    seed:      int   = 7,
) -> pd.DataFrame:
    """
    Generate a clearly trending OHLCV dataset (for testing structure engines).

    Args:
        direction: "bullish" or "bearish"
    """
    rng  = np.random.default_rng(seed)
    sign = 1 if direction == "bullish" else -1
    drift = sign * 0.001

    closes = [start_price]
    for _ in range(n_bars - 1):
        ret = rng.normal(drift, 0.0006)
        closes.append(closes[-1] * np.exp(ret))
    closes = np.array(closes)

    records = []
    timestamp = datetime(2024, 1, 1, 0, 0)
    for i in range(n_bars):
        c   = closes[i]
        o   = closes[i - 1] if i > 0 else start_price
        vol = c * rng.uniform(0.001, 0.003)
        records.append({
            "timestamp": timestamp,
            "open":   round(o, 4),
            "high":   round(max(o, c) + abs(rng.normal(0, vol)), 4),
            "low":    round(min(o, c) - abs(rng.normal(0, vol)), 4),
            "close":  round(c, 4),
            "volume": round(rng.uniform(1000, 3000), 1),
        })
        timestamp += timedelta(hours=4)

    df = pd.DataFrame(records).set_index("timestamp")
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"]  = df[["open", "close", "low"]].min(axis=1)
    return df


def generate_ranging_data(
    n_bars:   int   = 200,
    centre:   float = 2000.0,
    range_pct: float = 0.015,
    seed:     int   = 99,
) -> pd.DataFrame:
    """Generate a ranging / consolidating market dataset."""
    rng  = np.random.default_rng(seed)
    half = centre * range_pct

    closes = []
    price  = centre
    for _ in range(n_bars):
        ret = rng.normal(0, 0.0004)
        price = np.clip(price * np.exp(ret), centre - half, centre + half)
        closes.append(price)

    records = []
    timestamp = datetime(2024, 1, 1, 0, 0)
    for i in range(n_bars):
        c   = closes[i]
        o   = closes[i - 1] if i > 0 else centre
        vol = c * rng.uniform(0.0005, 0.002)
        records.append({
            "timestamp": timestamp,
            "open":   round(o, 4),
            "high":   round(max(o, c) + abs(rng.normal(0, vol)), 4),
            "low":    round(min(o, c) - abs(rng.normal(0, vol)), 4),
            "close":  round(c, 4),
            "volume": round(rng.uniform(500, 1500), 1),
        })
        timestamp += timedelta(hours=4)

    df = pd.DataFrame(records).set_index("timestamp")
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"]  = df[["open", "close", "low"]].min(axis=1)
    return df


if __name__ == "__main__":
    # Quick smoke-test: generate and print
    df = generate_xauusd_ohlcv(n_bars=50)
    print(df.tail())
    print(f"\n{len(df)} bars | price range: {df['low'].min():.2f} – {df['high'].max():.2f}")
