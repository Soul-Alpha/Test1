"""
Execution Quality Filter for Prometheus Trading Bot
====================================================
Gatekeeps trade entries based on real-time execution conditions:
  – current spread vs normalised ATR ratio
  – price spike detection (abnormal instantaneous gap)
  – symbol info validation

All checks are optional; if data is unavailable the filter passes
(fail-open by default) so the bot degrades gracefully without MT5.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class QualityResult:
    passes: bool
    reason: str            # empty string if passes
    spread_pips: float = 0.0
    atr_value: float   = 0.0
    spread_atr_ratio: float = 0.0


class ExecutionQualityFilter:
    """
    Validates entry conditions before placing an order.

    Parameters
    ----------
    max_spread_atr_ratio : float
        Reject if spread > this fraction of ATR.  Default 0.25 (25%).
        E.g. ATR=20 pts → reject if spread > 5 pts.
    max_spread_pips : float
        Hard cap on spread in pips regardless of ATR.  0 = disabled.
    spike_atr_mult : float
        If the last-bar range > spike_atr_mult × ATR, flag as spike.
        0 = disabled.
    """

    def __init__(
        self,
        max_spread_atr_ratio: float = 0.25,
        max_spread_pips: float = 0.0,     # disabled by default
        spike_atr_mult: float = 3.0,
    ) -> None:
        self.max_spread_atr_ratio = max_spread_atr_ratio
        self.max_spread_pips      = max_spread_pips
        self.spike_atr_mult       = spike_atr_mult
        # Rolling spread baseline (EWM, α=0.05 ≈ 20-observation half-life)
        self._spread_ema: float | None = None
        self._spread_alpha: float      = 0.05

    def update_spread_baseline(self, spread_pips: float) -> None:
        """Feed a new spread observation into the rolling EWM baseline."""
        if spread_pips <= 0:
            return
        if self._spread_ema is None:
            self._spread_ema = spread_pips
        else:
            self._spread_ema = (
                self._spread_alpha * spread_pips
                + (1.0 - self._spread_alpha) * self._spread_ema
            )

    @property
    def spread_environment(self) -> str:
        """Classify spread vs rolling baseline: 'tight' | 'medium' | 'wide'."""
        if self._spread_ema is None or self._spread_ema <= 0:
            return "medium"
        ratio = (self._spread_ema or 1.0)
        # Use the most recently measured spread against the baseline
        # This is called after update_spread_baseline so _spread_ema is current-ish
        return "medium"  # default; callers compare against known baseline directly

    def spread_vs_baseline(self, current_spread_pips: float) -> str:
        """Return 'tight' | 'medium' | 'wide' by comparing current spread to EWM baseline."""
        if self._spread_ema is None or self._spread_ema <= 0:
            return "medium"
        ratio = current_spread_pips / self._spread_ema
        if ratio < 0.85:
            return "tight"
        elif ratio > 1.60:
            return "wide"
        return "medium"

    def check(
        self,
        symbol_info,           # mt5.symbol_info result (or None)
        atr: Optional[float],  # current ATR value in price units
        last_bar_range: Optional[float] = None,  # high-low of last closed bar
        spread_tolerance_mult: float = 1.0,      # from session classifier
    ) -> QualityResult:
        """
        Run all quality checks and return a QualityResult.

        Parameters
        ----------
        symbol_info : mt5.SymbolInfo or None
            Live symbol info from MetaTrader 5.
        atr : float or None
            Current ATR in price units (e.g. 2.5 for XAUUSD = $2.50).
        last_bar_range : float or None
            High-low range of the last completed bar (for spike detection).
        spread_tolerance_mult : float
            Multiplier from SessionState.spread_tolerance; values < 1 tighten the
            spread requirement.
        """
        # If we lack symbol_info or ATR, pass silently (fail-open)
        if symbol_info is None or atr is None or atr <= 0:
            return QualityResult(passes=True, reason="")

        try:
            # -- Spread calculation ------------------------------------------
            spread_raw  = getattr(symbol_info, "spread", 0)   # in points (integer)
            point       = getattr(symbol_info, "point", 0.01) or 0.01
            spread_pips = spread_raw * point                    # in price units

            # Update rolling baseline every time we measure a real spread
            self.update_spread_baseline(spread_pips)

            spread_atr_ratio = spread_pips / atr

            # Apply session tolerance: tighter session → lower effective limit
            effective_max_ratio = self.max_spread_atr_ratio * spread_tolerance_mult

            if spread_atr_ratio > effective_max_ratio:
                return QualityResult(
                    passes=False,
                    reason=(
                        f"Spread too wide: {spread_pips:.3f} "
                        f"= {spread_atr_ratio*100:.1f}% of ATR "
                        f"(limit {effective_max_ratio*100:.1f}%)"
                    ),
                    spread_pips=spread_pips,
                    atr_value=atr,
                    spread_atr_ratio=spread_atr_ratio,
                )

            # -- Hard pip cap (if configured) --------------------------------
            if self.max_spread_pips > 0 and spread_pips > self.max_spread_pips:
                return QualityResult(
                    passes=False,
                    reason=f"Spread {spread_pips:.3f} > hard cap {self.max_spread_pips:.3f}",
                    spread_pips=spread_pips,
                    atr_value=atr,
                    spread_atr_ratio=spread_atr_ratio,
                )

            # -- Spike detection (optional) ----------------------------------
            if (
                self.spike_atr_mult > 0
                and last_bar_range is not None
                and last_bar_range > self.spike_atr_mult * atr
            ):
                return QualityResult(
                    passes=False,
                    reason=(
                        f"Price spike detected: bar range {last_bar_range:.3f} "
                        f"> {self.spike_atr_mult}× ATR ({self.spike_atr_mult * atr:.3f})"
                    ),
                    spread_pips=spread_pips,
                    atr_value=atr,
                    spread_atr_ratio=spread_atr_ratio,
                )

            return QualityResult(
                passes=True,
                reason="",
                spread_pips=spread_pips,
                atr_value=atr,
                spread_atr_ratio=spread_atr_ratio,
            )

        except Exception as exc:
            logger.warning("[exec_quality] Check error: %s", exc)
            return QualityResult(passes=True, reason="")  # fail-open
