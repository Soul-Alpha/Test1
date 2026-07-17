"""
Backtesting Engine
==================
Event-driven backtест over historical OHLCV data.

Features:
  - Position sizing based on fixed % risk per trade
  - Stop-loss / take-profit management
  - Slippage and commission modelling
  - Full trade log
  - Performance metrics:
      - Win rate
      - Profit factor
      - Expectancy
      - Max drawdown
      - Sharpe ratio
      - Calmar ratio
      - Average RR
      - Setup quality breakdown

The engine integrates with the full analysis pipeline: for each bar it re-runs
the configured analysis function and enters/exits positions accordingly.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class Trade:
    trade_id:    str
    direction:   str       # "long" | "short"
    entry_price: float
    sl_price:    float
    tp_price:    float
    size:        float     # units / contracts
    entry_bar:   int
    exit_bar:    Optional[int]  = None
    exit_price:  Optional[float] = None
    pnl:         float          = 0.0
    rr:          float          = 0.0
    status:      str            = "open"   # "open" | "won" | "lost" | "breakeven"
    ml_score:    float          = 0.0      # ML win_probability at entry (0 = not available)

    @property
    def risk_amount(self) -> float:
        return abs(self.entry_price - self.sl_price) * self.size

    @property
    def rr_achieved(self) -> float:   # alias used by API/UI
        return self.rr

    @property
    def is_win(self) -> bool:
        return self.status == "won"


@dataclass
class BacktestResult:
    trades:          List[Trade] = field(default_factory=list)
    equity_curve:    List[float] = field(default_factory=list)
    total_return_pct: float = 0.0
    win_rate:        float = 0.0
    profit_factor:   float = 0.0
    expectancy:      float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio:    float = 0.0
    calmar_ratio:    float = 0.0
    avg_rr:          float = 0.0
    total_trades:    int   = 0
    winning_trades:  int   = 0
    losing_trades:   int   = 0
    avg_win_pct:     float = 0.0
    avg_loss_pct:    float = 0.0
    final_equity:    float = 0.0
    narrative:       str   = ""

    @property
    def net_return_pct(self) -> float:   # alias used by API/UI
        return self.total_return_pct


# ─────────────────────────────────────────────
# Signal type
# ─────────────────────────────────────────────

@dataclass
class Signal:
    direction:   str    # "long" | "short" | "bullish" | "bearish" | "none"
    entry_price: float
    sl_price:    float
    tp_price:    float
    confidence:  float  = 0.5
    ml_score:    float  = 0.0   # ML win_probability (0 = not provided)

    def __post_init__(self):
        """Normalise bullish/bearish aliases to long/short."""
        _map = {"bullish": "long", "bearish": "short"}
        self.direction = _map.get(self.direction, self.direction)


# ─────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────

class Backtester:
    """
    Walk-forward backtester.

    Usage::

        def my_strategy(df_up_to_bar: pd.DataFrame) -> Signal:
            # Your analysis pipeline here
            return Signal("long", entry, sl, tp)

        bt      = Backtester(initial_capital=10_000, risk_per_trade=0.01)
        result  = bt.run(df, my_strategy)
        print(result.narrative)
    """

    def __init__(
        self,
        initial_capital:  float = 10_000.0,
        risk_per_trade:   float = 0.01,       # 1 % of equity per trade
        commission_pct:   float = 0.0003,     # 0.03 % per side
        slippage_pct:     float = 0.0001,
        default_rr:       float = 2.0,
        max_open_trades:  int   = 3,
        warmup_bars:      int   = 50,
        signal_stride:    int   = 5,    # re-evaluate strategy every N bars
        cooldown_bars:    int   = 3,    # bars to skip after a stop-loss
        min_rr:           float = 1.5,  # reject trades below this R:R
        signal_expiry_atr:float = 0.5,  # invalidate cached signal if price drifts > N×ATR
        dual_tp:          bool  = True, # split each entry into TP1(1:1) + TP2(orig) half-lots
    ) -> None:
        self.initial_capital   = initial_capital
        self.risk_pct          = risk_per_trade
        self.commission        = commission_pct
        self.slippage          = slippage_pct
        self.default_rr        = default_rr
        self.max_open          = max_open_trades
        self.warmup            = warmup_bars
        self.signal_stride     = max(1, signal_stride)
        self.cooldown_bars     = cooldown_bars
        self.min_rr            = min_rr
        self.signal_expiry_atr = signal_expiry_atr
        self.dual_tp           = dual_tp

    def run(
        self,
        df:            pd.DataFrame,
        strategy_fn:   Callable[[pd.DataFrame], Signal],
        min_confidence: float = 0.50,
        progress_cb:   Optional[Callable[[int, int], None]] = None,
    ) -> BacktestResult:
        """
        Walk forward through df, calling strategy_fn every ``signal_stride`` bars.

        Args:
            df:              Full OHLCV DataFrame.
            strategy_fn:     Callable(df_slice) → Signal.  Called every N bars.
            min_confidence:  Signals below this confidence are ignored.
            progress_cb:     Optional callable(current_bar, total_bars) for UI updates.

        Returns:
            BacktestResult with full trade log and performance metrics.
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        equity       = self.initial_capital
        open_trades: List[Trade] = []
        closed:      List[Trade] = []
        equity_curve = [equity]
        cached_signal: Optional[Signal] = None   # reused between stride bars
        cached_signal_price: float = 0.0          # price when signal was generated
        cached_signal_atr: float = 0.0            # ATR at signal generation
        pending_trade: Optional[Signal] = None    # signal waiting for next-bar open entry
        last_loss_bar: int = -999                 # for cooldown tracking
        total_bars   = len(df) - self.warmup

        # Pre-compute ATR-14 for signal expiry checks
        atr_series = (df["high"] - df["low"]).rolling(14).mean().fillna(
            (df["high"] - df["low"]).mean()
        )

        for i in range(self.warmup, len(df)):
            bar_open  = float(df["open"].iloc[i])
            bar_high  = float(df["high"].iloc[i])
            bar_low   = float(df["low"].iloc[i])
            bar_close = float(df["close"].iloc[i])
            bar_atr   = float(atr_series.iloc[i])

            # ── Execute pending signal at this bar's open (next-bar entry) ────
            if pending_trade is not None and len(open_trades) < self.max_open:
                sig = pending_trade
                pending_trade = None
                entry = bar_open * (1 + self.slippage if sig.direction == "long" else 1 - self.slippage)
                risk  = abs(entry - sig.sl_price)
                rr    = abs(sig.tp_price - entry) / risk if risk > 0 else 0.0
                if risk > 0 and rr >= self.min_rr:
                    size = (equity * self.risk_pct) / risk
                    equity -= entry * size * self.commission
                    if self.dual_tp:
                        # ── Dual-leg entry: TP1 at 1:1 RR (half), TP2 at original target (half)
                        _is_long_leg = sig.direction == "long"
                        _tp1 = (entry + risk) if _is_long_leg else (entry - risk)
                        _tp2 = sig.tp_price
                        _half = size / 2.0
                        open_trades.append(Trade(
                            trade_id=str(uuid.uuid4()) + "-L1",
                            direction=sig.direction,
                            entry_price=entry,
                            sl_price=sig.sl_price,
                            tp_price=_tp1,
                            size=_half,
                            entry_bar=i,
                            ml_score=sig.ml_score,
                        ))
                        open_trades.append(Trade(
                            trade_id=str(uuid.uuid4()) + "-L2",
                            direction=sig.direction,
                            entry_price=entry,
                            sl_price=sig.sl_price,
                            tp_price=_tp2,
                            size=_half,
                            entry_bar=i,
                            ml_score=sig.ml_score,
                        ))
                    else:
                        open_trades.append(Trade(
                            trade_id=str(uuid.uuid4()),
                            direction=sig.direction,
                            entry_price=entry,
                            sl_price=sig.sl_price,
                            tp_price=sig.tp_price,
                            size=size,
                            entry_bar=i,
                            ml_score=sig.ml_score,
                        ))

            # ── Manage open positions ──────────────────────────────────────────
            still_open: List[Trade] = []
            for trade in open_trades:
                closed_trade = self._check_exit(trade, bar_high, bar_low, bar_close, i)
                if closed_trade:
                    equity += closed_trade.pnl - self._commission_cost(closed_trade)
                    closed.append(closed_trade)
                    if closed_trade.status == "lost":
                        last_loss_bar = i    # start cooldown
                    logger.debug(
                        "Trade %s %s @ %.4f → pnl %.2f",
                        closed_trade.trade_id[:8],
                        closed_trade.status,
                        closed_trade.exit_price,
                        closed_trade.pnl,
                    )
                else:
                    still_open.append(trade)
            open_trades = still_open

            # ── Evaluate new signal (only every signal_stride bars) ──────────
            in_cooldown = (i - last_loss_bar) < self.cooldown_bars
            if len(open_trades) < self.max_open and not in_cooldown:
                if (i - self.warmup) % self.signal_stride == 0:
                    try:
                        df_slice = df.iloc[: i + 1]
                        new_sig = strategy_fn(df_slice)
                        if new_sig is not None:
                            cached_signal       = new_sig
                            cached_signal_price = bar_close
                            cached_signal_atr   = bar_atr
                        else:
                            cached_signal = None
                    except Exception as exc:
                        logger.debug("Strategy error at bar %d: %s", i, exc)
                        cached_signal = None

                # Expire cached signal if price drifted too far
                if cached_signal is not None and cached_signal_atr > 0:
                    drift = abs(bar_close - cached_signal_price)
                    if drift > self.signal_expiry_atr * cached_signal_atr:
                        cached_signal = None

                signal = cached_signal if cached_signal is not None else Signal("none", 0.0, 0.0, 0.0)

                if signal.direction != "none" and signal.confidence >= min_confidence:
                    # Queue for next-bar open entry (no look-ahead bias)
                    pending_trade = signal
                    cached_signal = None   # consume signal

            equity_curve.append(equity)
            # progress callback
            if progress_cb and (i - self.warmup) % max(1, total_bars // 50) == 0:
                progress_cb(i - self.warmup, total_bars)
        last_close = float(df["close"].iloc[-1])
        for trade in open_trades:
            trade.exit_price = last_close
            trade.exit_bar   = len(df) - 1
            trade.pnl        = self._pnl(trade, last_close)
            trade.status     = "closed_eod"
            equity          += trade.pnl
            closed.append(trade)

        result = self._compute_metrics(closed, equity_curve)
        logger.info(
            "Backtest complete: %d trades | WR %.1f%% | PF %.2f | DD %.1f%%",
            result.total_trades,
            result.win_rate * 100,
            result.profit_factor,
            result.max_drawdown_pct * 100,
        )
        return result

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _check_exit(
        self,
        trade:     Trade,
        bar_high:  float,
        bar_low:   float,
        bar_close: float,
        bar_idx:   int,
    ) -> Optional[Trade]:
        """Check if SL or TP was hit during the bar."""
        if trade.direction == "long":
            if bar_low <= trade.sl_price:
                trade.exit_price = trade.sl_price
                trade.pnl        = self._pnl(trade, trade.sl_price)
                trade.status     = "lost"
            elif bar_high >= trade.tp_price:
                trade.exit_price = trade.tp_price
                trade.pnl        = self._pnl(trade, trade.tp_price)
                trade.status     = "won"
            else:
                return None
        else:  # short
            if bar_high >= trade.sl_price:
                trade.exit_price = trade.sl_price
                trade.pnl        = self._pnl(trade, trade.sl_price)
                trade.status     = "lost"
            elif bar_low <= trade.tp_price:
                trade.exit_price = trade.tp_price
                trade.pnl        = self._pnl(trade, trade.tp_price)
                trade.status     = "won"
            else:
                return None

        trade.exit_bar = bar_idx
        risk = abs(trade.entry_price - trade.sl_price)
        reward = abs((trade.exit_price or 0) - trade.entry_price)
        trade.rr = reward / risk if risk > 0 else 0.0
        return trade

    def _pnl(self, trade: Trade, exit_price: float) -> float:
        if trade.direction == "long":
            return (exit_price - trade.entry_price) * trade.size
        return (trade.entry_price - exit_price) * trade.size

    def _commission_cost(self, trade: Trade) -> float:
        return trade.entry_price * trade.size * self.commission

    def _compute_metrics(
        self, trades: List[Trade], equity_curve: List[float]
    ) -> BacktestResult:
        if not trades:
            return BacktestResult(narrative="No trades executed during backtest period.")

        wins   = [t for t in trades if t.status == "won"]
        losses = [t for t in trades if t.status == "lost"]

        total_pnl    = sum(t.pnl for t in trades)
        gross_profit = sum(t.pnl for t in wins) if wins else 0.0
        gross_loss   = abs(sum(t.pnl for t in losses)) if losses else 1e-8

        win_rate      = len(wins) / len(trades) if trades else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_rr        = float(np.mean([t.rr for t in wins])) if wins else 0.0

        # Expectancy per trade
        avg_win  = gross_profit / len(wins)  if wins   else 0.0
        avg_loss = gross_loss   / len(losses) if losses else 0.0
        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

        # Drawdown
        eq    = np.array(equity_curve)
        peaks = np.maximum.accumulate(eq)
        dds   = (peaks - eq) / (peaks + 1e-8)
        max_dd = float(dds.max())

        # Sharpe (daily returns)
        returns  = np.diff(eq) / (eq[:-1] + 1e-8)
        sharpe   = float(np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252) if len(returns) > 1 else 0.0
        calmar   = (total_pnl / self.initial_capital) / (max_dd + 1e-8)
        total_ret = (eq[-1] - eq[0]) / eq[0] if len(eq) > 0 else 0.0

        avg_win_pct  = avg_win  / self.initial_capital
        avg_loss_pct = avg_loss / self.initial_capital

        result = BacktestResult(
            trades=trades,
            equity_curve=list(eq),
            total_return_pct=round(total_ret * 100, 2),
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 2),
            expectancy=round(expectancy, 2),
            max_drawdown_pct=round(max_dd, 4),
            sharpe_ratio=round(sharpe, 3),
            calmar_ratio=round(calmar, 3),
            avg_rr=round(avg_rr, 2),
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            avg_win_pct=round(avg_win_pct * 100, 2),
            avg_loss_pct=round(avg_loss_pct * 100, 2),
            final_equity=round(float(eq[-1]) if len(eq) > 0 else self.initial_capital, 2),
        )
        result.narrative = self._build_narrative(result)
        return result

    def _build_narrative(self, r: BacktestResult) -> str:
        return (
            f"Backtest Results:\n"
            f"  Total trades:     {r.total_trades}  "
            f"({r.winning_trades}W / {r.losing_trades}L)\n"
            f"  Win rate:         {r.win_rate:.1%}\n"
            f"  Profit factor:    {r.profit_factor:.2f}\n"
            f"  Expectancy:       ${r.expectancy:.2f} per trade\n"
            f"  Total return:     {r.total_return_pct:.2f}%\n"
            f"  Max drawdown:     {r.max_drawdown_pct:.2%}\n"
            f"  Sharpe ratio:     {r.sharpe_ratio:.3f}\n"
            f"  Calmar ratio:     {r.calmar_ratio:.3f}\n"
            f"  Average RR:       {r.avg_rr:.2f}\n"
            f"  Avg win:          {r.avg_win_pct:.2f}%  |  Avg loss: {r.avg_loss_pct:.2f}%"
        )
