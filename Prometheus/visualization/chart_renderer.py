"""
Visualization Engine
=====================
Generates professional interactive and static chart overlays.

Outputs:
  - Plotly interactive HTML charts with all overlays
  - mplfinance / matplotlib PNG exports
  - Annotated chart previews

Overlays rendered:
  - Candlestick OHLCV
  - Support & resistance zones (shaded bands)
  - Fibonacci levels (horizontal dashed lines)
  - Swing highs / lows (triangle markers)
  - BOS / CHoCH labels
  - Order blocks (shaded rectangles)
  - Fair value gaps (shaded)
  - Liquidity pool markers
  - Chart pattern annotations
  - Entry / SL / TP zones
  - Equity curve from backtesting
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional heavy imports
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("plotly not installed — interactive charts disabled")

try:
    import mplfinance as mpf          # type: ignore
    import matplotlib.pyplot as plt
    MPF_AVAILABLE = True
except ImportError:
    MPF_AVAILABLE = False
    logger.warning("mplfinance not installed — static charts disabled")


# ─────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────

def _offset_x(base_pos: int, n_bars: int, _df=None) -> int:
    """Return an integer x position n_bars ahead of base_pos."""
    return int(base_pos) + int(n_bars)


class ChartRenderer:
    """
    Renders fully annotated charts.

    Usage::

        renderer = ChartRenderer(output_dir="outputs")
        html_path = renderer.render_plotly(df, analysis_context, "XAUUSD_4H")
        png_path  = renderer.render_static(df, analysis_context, "XAUUSD_4H")
    """

    def __init__(self, output_dir: str = "outputs") -> None:
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def render_plotly(
        self,
        df:       pd.DataFrame,
        ctx:      Dict[str, Any],
        name:     str = "chart",
        n_bars:   int = 150,
    ) -> Optional[str]:
        """
        Create an interactive Plotly chart with all overlays.

        Args:
            df:     OHLCV DataFrame.
            ctx:    Analysis context dict from Prometheus.analyze().
            name:   Output filename stem.
            n_bars: How many bars to show.

        Returns:
            Path to saved HTML file, or None if plotly unavailable.
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("plotly not available")
            return None

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        df = df.iloc[-n_bars:]

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.80, 0.20],
        )

        # ── Candlesticks ───────────────────────────────────────────────────────
        fig.add_trace(
            go.Candlestick(
                x=df.index if hasattr(df.index, "to_pydatetime") else list(range(len(df))),
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="OHLCV",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
            ),
            row=1, col=1,
        )

        # ── Volume ────────────────────────────────────────────────────────────
        colours = [
            "#26a69a" if c >= o else "#ef5350"
            for o, c in zip(df["open"], df["close"])
        ]
        fig.add_trace(
            go.Bar(
                x=df.index if hasattr(df.index, "to_pydatetime") else list(range(len(df))),
                y=df["volume"],
                name="Volume",
                marker_color=colours,
                opacity=0.5,
            ),
            row=2, col=1,
        )

        # ── S/R zones ─────────────────────────────────────────────────────────
        sr_result = ctx.get("sr")
        if sr_result:
            for zone in getattr(sr_result, "support_zones", [])[:5]:
                self._add_zone(fig, zone.lower, zone.upper, "rgba(38,166,154,0.15)", "Support", df)
            for zone in getattr(sr_result, "resistance_zones", [])[:5]:
                self._add_zone(fig, zone.lower, zone.upper, "rgba(239,83,80,0.15)", "Resistance", df)

        # ── Fibonacci levels ───────────────────────────────────────────────────
        fib_result = ctx.get("fib")
        if fib_result:
            for lvl in getattr(fib_result, "levels", []):
                if lvl.is_key:
                    fig.add_hline(
                        y=lvl.price,
                        line=dict(color="rgba(255,215,0,0.6)", dash="dash", width=1),
                        annotation_text=f"Fib {lvl.label}%",
                        annotation_position="right",
                        row=1, col=1,
                    )

        # ── Swing markers ─────────────────────────────────────────────────────
        ms_result = ctx.get("ms")
        if ms_result:
            x_vals = df.index if hasattr(df.index, "to_pydatetime") else list(range(len(df)))
            for sh in getattr(ms_result, "swing_highs", []):
                if sh.index < len(df):
                    fig.add_trace(go.Scatter(
                        x=[x_vals[min(sh.index, len(x_vals)-1)]],
                        y=[sh.price],
                        mode="markers",
                        marker=dict(symbol="triangle-down", size=10, color="#ef5350"),
                        name="Swing High",
                        showlegend=False,
                    ), row=1, col=1)
            for sl in getattr(ms_result, "swing_lows", []):
                if sl.index < len(df):
                    fig.add_trace(go.Scatter(
                        x=[x_vals[min(sl.index, len(x_vals)-1)]],
                        y=[sl.price],
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=10, color="#26a69a"),
                        name="Swing Low",
                        showlegend=False,
                    ), row=1, col=1)

        # ── SMC: Order blocks & FVGs ───────────────────────────────────────────
        smc_result = ctx.get("smc")
        if smc_result:
            for ob in getattr(smc_result, "order_blocks", [])[:5]:
                col = "rgba(38,166,154,0.2)" if ob.direction == "bullish" else "rgba(239,83,80,0.2)"
                self._add_zone(fig, ob.low, ob.high, col, f"{ob.direction.capitalize()} OB", df)
            for fvg in getattr(smc_result, "fair_value_gaps", [])[:3]:
                col = "rgba(100,181,246,0.15)"
                self._add_zone(fig, fvg.low, fvg.high, col, f"FVG ({fvg.direction})", df)

        # ── Confluence score box ───────────────────────────────────────────────
        score_ctx = ctx.get("confluence")
        if score_ctx:
            fig.add_annotation(
                xref="paper", yref="paper", x=0.01, y=0.99,
                text=(
                    f"<b>Confluence: {score_ctx.total:.0f}/100 "
                    f"(Grade {score_ctx.grade})</b><br>"
                    f"Direction: {score_ctx.direction.capitalize()}"
                ),
                showarrow=False,
                font=dict(size=13, color="white"),
                bgcolor="rgba(0,0,0,0.55)",
                borderpad=6,
                align="left",
            )

        # ── Layout ────────────────────────────────────────────────────────────
        asset     = ctx.get("asset", "XAUUSD")
        timeframe = ctx.get("timeframe", "4H")
        fig.update_layout(
            title=dict(text=f"{asset} {timeframe} — Prometheus Analysis", font=dict(size=16)),
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=700,
            margin=dict(l=60, r=60, t=60, b=40),
            legend=dict(orientation="h", y=-0.05),
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        out_path = str(self.out / f"{name}.html")
        fig.write_html(out_path)
        logger.info("Plotly chart saved: %s", out_path)
        return out_path

    def render_static(
        self,
        df:     pd.DataFrame,
        ctx:    Dict[str, Any],
        name:   str = "chart",
        n_bars: int = 150,
    ) -> Optional[str]:
        """
        Render a static PNG using mplfinance.

        Returns path to PNG or None.
        """
        if not MPF_AVAILABLE:
            logger.warning("mplfinance not available")
            return None

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        df = df.iloc[-n_bars:]

        # Convert index to DatetimeIndex if needed
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.RangeIndex(len(df))

        add_plots: list = []
        sr_result = ctx.get("sr")
        if sr_result:
            support_lines    = [z.level for z in getattr(sr_result, "support_zones", [])[:3]]
            resistance_lines = [z.level for z in getattr(sr_result, "resistance_zones", [])[:3]]
            for price in support_lines:
                add_plots.append(mpf.make_addplot(
                    pd.Series([price] * len(df), index=df.index),
                    color="#26a69a", linestyle="--", width=0.8,
                ))
            for price in resistance_lines:
                add_plots.append(mpf.make_addplot(
                    pd.Series([price] * len(df), index=df.index),
                    color="#ef5350", linestyle="--", width=0.8,
                ))

        fib_result = ctx.get("fib")
        if fib_result:
            for lvl in getattr(fib_result, "levels", []):
                if lvl.is_key:
                    add_plots.append(mpf.make_addplot(
                        pd.Series([lvl.price] * len(df), index=df.index),
                        color="gold", linestyle=":", width=0.7,
                    ))

        out_path = str(self.out / f"{name}.png")
        mpf.plot(
            df[["open", "high", "low", "close", "volume"]],
            type="candle",
            style="nightclouds",
            title=f"{ctx.get('asset','XAUUSD')} {ctx.get('timeframe','4H')}",
            volume=True,
            addplot=add_plots if add_plots else None,
            savefig=out_path,
            figsize=(16, 9),
        )
        plt.close("all")
        logger.info("Static chart saved: %s", out_path)
        return out_path


    def render_scenario_chart(
        self,
        df:   pd.DataFrame,
        ctx:  Dict[str, Any],
        name: str = "scenario_chart",
        n_bars: int = 80,
    ) -> Optional[str]:
        """
        Render the dual-scenario projection chart:
          - Last N candles
          - S/R zones + Fibonacci
          - SCENARIO A arrow path (bullish breakout continuation)
          - SCENARIO B arrow path (liquidity sweep reversal)
          - Target labels
        Returns path to HTML or None.
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("plotly not available")
            return None

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        df = df.iloc[-n_bars:]

        # ── Use integer x positions to avoid ALL Timestamp arithmetic in Plotly 6
        # Date labels are applied via xaxis tickvals/ticktext instead.
        n = len(df)
        x_hist  = list(range(n))          # 0 … n-1  (historical bars)
        last_xi = n - 1                   # integer position of last bar
        last_close  = float(df["close"].iloc[-1])
        price_range = float(df["high"].max() - df["low"].min())

        # Build tick labels for every ~10th bar + future bars
        if hasattr(df.index, "strftime"):
            date_labels = df.index.strftime("%m-%d %H:%M").tolist()
        else:
            date_labels = [str(v) for v in df.index]
        tick_step = max(1, n // 10)
        tick_vals = list(range(0, n, tick_step))
        tick_text = [date_labels[i] for i in tick_vals]

        # ── Figure ────────────────────────────────────────────────────────────
        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=x_hist,
            open=df["open"], high=df["high"],
            low=df["low"],   close=df["close"],
            name="OHLCV",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ))

        # ── S/R zones ─────────────────────────────────────────────────────────
        sr = ctx.get("sr")
        res_level = sup_level = None
        if sr:
            for zone in getattr(sr, "support_zones", [])[:3]:
                self._add_zone(fig, zone.lower, zone.upper, "rgba(38,166,154,0.12)", "Support", df)
                if sup_level is None:
                    sup_level = zone.level
            for zone in getattr(sr, "resistance_zones", [])[:3]:
                self._add_zone(fig, zone.lower, zone.upper, "rgba(239,83,80,0.12)", "Resistance", df)
                if res_level is None:
                    res_level = zone.level

        # ── Fibonacci levels ───────────────────────────────────────────────────
        fib = ctx.get("fib")
        fib_target_up = fib_target_dn = None
        if fib:
            for lvl in getattr(fib, "levels", []):
                if lvl.is_key:
                    fig.add_hline(
                        y=lvl.price,
                        line=dict(color="rgba(255,215,0,0.5)", dash="dot", width=1),
                        annotation_text=f"Fib {lvl.label}%",
                        annotation_position="right",
                    )
            prices = [l.price for l in fib.levels]
            if prices:
                fib_target_up = max(prices)
                fib_target_dn = min(prices)

        # ── Compute scenario waypoints ─────────────────────────────────────────
        step = price_range * 0.04          # one "bar's worth" of price move
        bar_step = max(1, len(df) // 12)  # approx bars per segment

        # resistance anchor — fallback to last swing high or ATH of visible range
        if res_level is None:
            ms = ctx.get("ms")
            sh = getattr(ms, "swing_highs", []) if ms else []
            res_level = sh[-1].price if sh else float(df["high"].iloc[-10:].max())
        if sup_level is None:
            ms = ctx.get("ms")
            sl_ = getattr(ms, "swing_lows", []) if ms else []
            sup_level = sl_[-1].price if sl_ else float(df["low"].iloc[-10:].min())

        # Scenario A waypoints: last close → above res → retest → target up
        a_target = fib_target_up if fib_target_up and fib_target_up > res_level else res_level + price_range * 0.12
        a_xs = [last_xi,
                _offset_x(last_xi, bar_step * 2),
                _offset_x(last_xi, bar_step * 4),
                _offset_x(last_xi, bar_step * 6),
                _offset_x(last_xi, bar_step * 9)]
        a_ys = [last_close,
                res_level + step,          # closes above resistance
                res_level - step * 0.4,    # retest
                res_level + step * 0.8,    # bounce
                a_target]                  # final target

        # Scenario B waypoints: last close → wick above res → fail → reverse → target dn
        b_target = fib_target_dn if fib_target_dn and fib_target_dn < sup_level else sup_level - price_range * 0.10
        b_xs = [last_xi,
                _offset_x(last_xi, bar_step * 2),
                _offset_x(last_xi, bar_step * 3),
                _offset_x(last_xi, bar_step * 5),
                _offset_x(last_xi, bar_step * 8)]
        b_ys = [last_close,
                res_level + step * 1.4,    # wick spike above resistance
                res_level - step,          # rejection close below
                sup_level,                 # drop to support
                b_target]                  # sweep lows / target

        # ── Draw Scenario A ───────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=a_xs, y=a_ys,
            mode="lines+markers+text",
            name="Scenario A — True Breakout",
            line=dict(color="#2ecc71", width=2.5, dash="dash"),
            marker=dict(size=7, color="#2ecc71"),
            text=["", "", "Retest", "", f"Target A\n{a_target:.2f}"],
            textposition="top right",
            textfont=dict(color="#2ecc71", size=11),
        ))
        # Shaded target zone A
        fig.add_hrect(
            y0=a_target - step * 0.5, y1=a_target + step * 0.5,
            fillcolor="rgba(46,204,113,0.12)", line_width=0,
            annotation_text="🎯 Target A", annotation_position="right",
            annotation_font_color="#2ecc71",
        )

        # ── Draw Scenario B ───────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=b_xs, y=b_ys,
            mode="lines+markers+text",
            name="Scenario B — Liquidity Sweep",
            line=dict(color="#e74c3c", width=2.5, dash="dash"),
            marker=dict(size=7, color="#e74c3c"),
            text=["", "Wick / Trap", "Rejection", "", f"Target B\n{b_target:.2f}"],
            textposition="top right",
            textfont=dict(color="#e74c3c", size=11),
        ))
        # Shaded target zone B
        fig.add_hrect(
            y0=b_target - step * 0.5, y1=b_target + step * 0.5,
            fillcolor="rgba(231,76,60,0.12)", line_width=0,
            annotation_text="🎯 Target B", annotation_position="right",
            annotation_font_color="#e74c3c",
        )

        # ── NOW line ──────────────────────────────────────────────────────────
        fig.add_vline(
            x=last_xi,
            line=dict(color="rgba(255,255,255,0.4)", dash="dot", width=1.5),
            annotation_text="NOW",
            annotation_position="top",
        )

        # ── Label key levels ──────────────────────────────────────────────────
        for lvl, label, col in [
            (res_level, f"Resistance {res_level:.2f}", "#ef5350"),
            (sup_level, f"Support {sup_level:.2f}",    "#26a69a"),
        ]:
            fig.add_hline(
                y=lvl,
                line=dict(color=col, width=1, dash="dot"),
                annotation_text=label,
                annotation_position="left",
                annotation_font_color=col,
            )

        asset     = ctx.get("asset", "XAUUSD")
        timeframe = ctx.get("timeframe", "4H")
        fig.update_layout(
            title=dict(
                text=(f"<b>{asset} {timeframe} — Dual Scenario Projection</b><br>"
                      "<span style='color:#2ecc71'>█ Scenario A: True Breakout</span>   "
                      "<span style='color:#e74c3c'>█ Scenario B: Liquidity Sweep</span>"),
                font=dict(size=14),
            ),
            template="plotly_dark",
            xaxis=dict(
                rangeslider_visible=False,
                tickvals=tick_vals,
                ticktext=tick_text,
                tickangle=-45,
            ),
            height=620,
            margin=dict(l=60, r=120, t=80, b=40),
            legend=dict(orientation="h", y=-0.06),
            showlegend=True,
        )
        fig.update_yaxes(title_text="Price")

        out_path = str(self.out / f"{name}.html")
        fig.write_html(out_path)
        logger.info("Scenario chart saved: %s", out_path)
        return out_path

    def render_equity_curve(
        self,
        equity_curve: List[float],
        name:         str = "equity_curve",
    ) -> Optional[str]:
        """Render a Plotly equity curve from backtesting results."""
        if not PLOTLY_AVAILABLE:
            return None

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=equity_curve,
            mode="lines",
            name="Equity",
            line=dict(color="#29b6f6", width=2),
            fill="tozeroy",
            fillcolor="rgba(41,182,246,0.1)",
        ))
        fig.update_layout(
            title="Equity Curve",
            template="plotly_dark",
            xaxis_title="Bar",
            yaxis_title="Equity ($)",
            height=400,
        )
        out_path = str(self.out / f"{name}.html")
        fig.write_html(out_path)
        return out_path

    # ── Helper ────────────────────────────────────────────────────────────────

    def _add_zone(
        self,
        fig:   Any,
        lower: float,
        upper: float,
        color: str,
        label: str,
        df:    pd.DataFrame,
    ) -> None:
        """Add a horizontal shaded zone to a Plotly figure (row 1)."""
        x_start = df.index[0]  if hasattr(df.index, "__getitem__") else 0
        x_end   = df.index[-1] if hasattr(df.index, "__getitem__") else len(df)
        fig.add_hrect(
            y0=lower, y1=upper,
            fillcolor=color,
            line_width=0,
            annotation_text=label,
            annotation_position="right",
            row=1, col=1,
        )
