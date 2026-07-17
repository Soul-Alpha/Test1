"""
Prometheus Gradio Dashboard — Full Feature Edition (port 7860)
==============================================================
All 8 tabs matching the Streamlit dashboard, sharing the same backend.

Tabs:
  📥 Input & Analyze  — CSV/JSON upload, MTF support, score card + AI report
  📈 Chart            — Candlestick + OB/FVG/S&R/entry/SL/TP overlays
  🧠 AI Report        — Dual scenarios, component breakdown, final signal card
  🔮 Patterns & SMC   — Trade-setup chart, chart patterns, candlestick signals,
                         OBs, FVGs, AMD cycle, Fibonacci, MTF alignment
  🤖 ML Prediction    — Quality score, win probability, feature importances, outcome labelling
  📉 Backtesting      — Walk-forward backtest, equity curve, trade log
  🗂 History          — Filtered DB history with full report viewer + CSV export
  🟢 Live Bot         — Full status, open positions, adaptive learning, recent trades
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402

TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]

DARK_CSS = """
body, .gradio-container { background:#0e1117!important; color:#dde1ea!important;
    font-family:'Inter',sans-serif!important; }
.prometheus-header { background:#1c2232; padding:18px 28px; border-radius:10px;
    border-bottom:2px solid #00c4ff; margin-bottom:12px; }
.prometheus-header h1 { margin:0; color:#00c4ff; font-size:1.8rem; }
.prometheus-header p  { margin:4px 0 0; color:#7f8c8d; font-size:.85rem; }
"""

_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
)


# ─────────────────────────────────────────────────────────────────────────────
# CSV / JSON parsers
# ─────────────────────────────────────────────────────────────────────────────

def _read_csv_smart(path: str) -> pd.DataFrame:
    import io as _io
    with open(path, "rb") as fh:
        raw = fh.read()

    encodings  = ["utf-16", "utf-16-le", "utf-16-be", "utf-8-sig", "utf-8", "latin-1", "cp1252"]
    delimiters = [",", "\t", ";"]
    last_err: Optional[Exception] = None

    known = {"open","high","low","close","date","time","datetime","vol","volume","tickvol","spread"}
    EXNESS_7 = ["datetime","open","high","low","close","volume","spread"]
    EXNESS_6 = ["datetime","open","high","low","close","volume"]
    MT4_8    = ["date","time","open","high","low","close","volume","spread"]
    MT4_7    = ["date","time","open","high","low","close","volume"]

    def _try(text: str, delim: str) -> pd.DataFrame:
        text = text.replace("\r\n","\n").replace("\r","\n").strip()
        first_cells = [c.strip().lower().replace("<","").replace(">","")
                       for c in (text.splitlines()[0] if text else "").split(delim)]
        has_header = any(w in known for w in first_cells)
        if has_header:
            df = pd.read_csv(_io.StringIO(text), sep=delim, engine="python")
            df.columns = [c.strip().lower().replace("<","").replace(">","") for c in df.columns]
        else:
            ncols = len(first_cells)
            names = ({9:["date","time","open","high","low","close","tickvol","volume","spread"],
                      8:MT4_8, 7:EXNESS_7, 6:EXNESS_6}.get(ncols, MT4_7))
            df = pd.read_csv(_io.StringIO(text), sep=delim, engine="python", header=None, names=names)
        if df.shape[1] < 4:
            raise ValueError("Too few columns")
        if "date" in df.columns and "time" in df.columns:
            df["datetime"] = pd.to_datetime(
                df["date"].astype(str).str.replace(".","-",regex=False)
                + " " + df["time"].astype(str), errors="coerce")
            df = df.drop(columns=["date","time"]).set_index("datetime")
        elif "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(
                df["datetime"].astype(str).str.replace(".","-",regex=False), errors="coerce")
            df = df.set_index("datetime")
        else:
            df = df.set_index(df.columns[0])
            df.index = pd.to_datetime(
                df.index.astype(str).str.replace(".","-",regex=False), errors="coerce")
        df.index.name = "datetime"
        df = df[~df.index.isna()]
        for old, new in [("vol","volume"),("tickvol","volume"),("tick_volume","volume")]:
            if old in df.columns and "volume" not in df.columns:
                df = df.rename(columns={old: new})
            elif old in df.columns:
                df = df.drop(columns=[old], errors="ignore")
        df = df.drop(columns=["spread"], errors="ignore")
        for col in ("open","high","low","close"):
            if col not in df.columns:
                raise ValueError(f"Missing: {col}")
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if "volume" not in df.columns:
            df["volume"] = 0.0
        df = df.dropna(subset=["open","high","low","close"])
        if len(df) < 2:
            raise ValueError("Fewer than 2 valid rows")
        return df.sort_index()

    for enc in encodings:
        for delim in delimiters:
            try:
                return _try(raw.decode(enc), delim)
            except Exception as exc:
                last_err = exc

    raise ValueError(f"Could not parse CSV. Last error: {last_err}")


def _load_json_ohlcv(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame(data)
    df.columns = [c.lower() for c in df.columns]
    for col in ("open","high","low","close"):
        if col not in df.columns:
            raise ValueError(f"Missing: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    dt_col = next((c for c in df.columns if "date" in c or "time" in c), None)
    if dt_col:
        df.index = pd.to_datetime(df[dt_col], errors="coerce")
        df = df.drop(columns=[dt_col])
    df.index.name = "datetime"
    return df.dropna(subset=["open","high","low","close"]).sort_index()


# ── Engine singleton ──────────────────────────────────────────────────────────
_engine = None

def _get_engine():
    global _engine
    if _engine is None:
        from prometheus_core import Prometheus
        _engine = Prometheus()
    return _engine


# ─────────────────────────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_base_chart(df: pd.DataFrame, result=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.8, 0.2], vertical_spacing=0.02)

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="OHLCV",
        increasing_line_color="#2ecc71", decreasing_line_color="#e74c3c",
    ), row=1, col=1)

    bar_colors = ["#2ecc71" if float(c) >= float(o) else "#e74c3c"
                  for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"],
        name="Volume", marker_color=bar_colors, opacity=0.55,
    ), row=2, col=1)

    if result is not None:
        smc = getattr(result, "smc", None)
        if smc:
            for fvg in (getattr(smc, "fair_value_gaps", None) or []):
                try:
                    if getattr(fvg, "filled", False):
                        continue
                    fig.add_hrect(y0=float(fvg.low), y1=float(fvg.high),
                                  fillcolor="rgba(0,196,255,0.07)", line_width=0,
                                  annotation_text="FVG", annotation_position="top left",
                                  annotation_font_color="#00c4ff", annotation_font_size=10,
                                  row=1, col=1)
                except Exception:
                    pass
            for ob in (getattr(smc, "order_blocks", None) or []):
                try:
                    is_bull = getattr(ob, "direction", "bullish") == "bullish"
                    fill   = "rgba(46,204,113,0.10)" if is_bull else "rgba(231,76,60,0.10)"
                    border = "#2ecc71" if is_bull else "#e74c3c"
                    fig.add_hrect(y0=float(ob.low), y1=float(ob.high),
                                  fillcolor=fill, line_width=1, line_color=border,
                                  annotation_text="OB BULL" if is_bull else "OB BEAR",
                                  annotation_position="top right",
                                  annotation_font_color=border, annotation_font_size=10,
                                  row=1, col=1)
                except Exception:
                    pass
        sr = getattr(result, "sr", None)
        if sr:
            for zone in (getattr(sr, "support_zones", None) or [])[:6]:
                try:
                    lvl = float(getattr(zone, "price", getattr(zone, "level", None)))
                    fig.add_hline(y=lvl, line_dash="dot", line_color="#2ecc71", line_width=1,
                                  annotation_text="S", annotation_font_color="#2ecc71",
                                  annotation_font_size=10, row=1, col=1)
                except Exception:
                    pass
            for zone in (getattr(sr, "resistance_zones", None) or [])[:6]:
                try:
                    lvl = float(getattr(zone, "price", getattr(zone, "level", None)))
                    fig.add_hline(y=lvl, line_dash="dot", line_color="#e74c3c", line_width=1,
                                  annotation_text="R", annotation_font_color="#e74c3c",
                                  annotation_font_size=10, row=1, col=1)
                except Exception:
                    pass

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#dde1ea", family="Inter"),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=40, r=20, t=30, b=20), height=580,
    )
    fig.update_xaxes(gridcolor="#1c2232", showgrid=True)
    fig.update_yaxes(gridcolor="#1c2232", showgrid=True)
    return fig


def _build_setup_chart(df: pd.DataFrame, result=None,
                       entry=None, sl=None, tp1=None, tp2=None):
    import plotly.graph_objects as go

    if df is None or len(df) < 5:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1117",
                          plot_bgcolor="#0e1117", height=480,
                          title="No data — run analysis first")
        return fig

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    n = min(60, len(df))
    df_w = df.iloc[-n:]
    xi = list(range(n))
    last_xi = n - 1
    full_len = len(df)
    bar_offset = full_len - n

    if hasattr(df_w.index, "strftime"):
        labels = df_w.index.strftime("%m-%d %H:%M").tolist()
    else:
        labels = [str(v) for v in df_w.index]
    tstep = max(1, n // 8)
    tvals = list(range(0, n, tstep))
    ttxt  = [labels[i] for i in tvals]

    price_lo = float(df_w["low"].min())
    price_hi = float(df_w["high"].max())
    price_range = max(price_hi - price_lo, 1e-6)

    def visible(y):
        return price_lo * 0.997 <= y <= price_hi * 1.003

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=xi, open=df_w["open"], high=df_w["high"],
        low=df_w["low"], close=df_w["close"], name="OHLCV",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350", showlegend=False,
    ))

    if result is not None:
        smc = getattr(result, "smc", None)
        if smc and getattr(smc, "order_blocks", None):
            for ob in smc.order_blocks:
                if not (visible(ob.high) or visible(ob.low)):
                    continue
                ob_color  = ("rgba(38,166,154,0.18)" if ob.direction == "bullish"
                             else "rgba(239,83,80,0.18)")
                ob_border = "#26a69a" if ob.direction == "bullish" else "#ef5350"
                ob_xi = ob.bar_idx - bar_offset
                ob_x0 = max(0, ob_xi - 2)
                tag = ("🟢 OB" if ob.direction == "bullish" else "🔴 OB") + (" ✓" if ob.mitigated else "")
                fig.add_shape(type="rect", x0=ob_x0, x1=last_xi, y0=ob.low, y1=ob.high,
                              fillcolor=ob_color, line=dict(color=ob_border, width=1, dash="dot"),
                              xref="x", yref="y", layer="below")
                fig.add_annotation(x=ob_x0, y=(ob.high + ob.low) / 2, text=f"<b>{tag}</b>",
                                   showarrow=False, xanchor="left",
                                   font=dict(color="#111111", size=10),
                                   bgcolor=ob_border, borderpad=2, opacity=0.85)

        if smc and getattr(smc, "fair_value_gaps", None):
            fvg_label_ys: list = []
            for fvg in smc.fair_value_gaps:
                if getattr(fvg, "filled", False) or not (visible(fvg.high) or visible(fvg.low)):
                    continue
                fvg_size  = abs(fvg.high - fvg.low)
                fvg_alpha = max(0.08, 0.22 - 0.18 * (fvg_size / price_range))
                is_bull   = fvg.direction == "bullish"
                fill      = (f"rgba(100,181,246,{fvg_alpha:.2f})" if is_bull
                             else f"rgba(255,167,38,{fvg_alpha:.2f})")
                border    = "#64b5f6" if is_bull else "#ffa726"
                fvg_x0    = max(0, fvg.start_idx - bar_offset)
                tag       = "⬆ FVG" if is_bull else "⬇ FVG"
                fig.add_shape(type="rect", x0=fvg_x0, x1=last_xi, y0=fvg.low, y1=fvg.high,
                              fillcolor=fill, line=dict(color=border, width=1, dash="dot"),
                              xref="x", yref="y", layer="below")
                label_y = fvg.mid
                min_gap = price_range * 0.018
                for used_y in fvg_label_ys:
                    if abs(label_y - used_y) < min_gap:
                        label_y = used_y + min_gap * (1 if is_bull else -1)
                fvg_label_ys.append(label_y)
                fig.add_annotation(x=fvg_x0, y=label_y, text=f"<b>{tag}</b>",
                                   showarrow=False, xanchor="left",
                                   font=dict(color="#111111", size=10),
                                   bgcolor=border, borderpad=2, opacity=0.85)

        pat = getattr(result, "pat", None)
        if pat and getattr(pat, "patterns", None):
            label_step = price_range * 0.06
            pat_label_ys: list = []
            for p in pat.patterns[:5]:
                p_x0 = max(0, p.start_idx - bar_offset)
                p_x1 = min(last_xi, p.end_idx - bar_offset)
                if p_x1 < 0 or p_x0 > last_xi:
                    continue
                p_color = ("#26a69a" if p.direction == "bullish"
                           else "#ef5350" if p.direction == "bearish" else "#90a4ae")
                r_c, g_c, b_c = int(p_color[1:3],16), int(p_color[3:5],16), int(p_color[5:7],16)
                for vx in [p_x0, max(p_x0+1, p_x1)]:
                    fig.add_shape(type="line", x0=vx, x1=vx, y0=price_lo, y1=price_hi,
                                  line=dict(color=p_color, width=1, dash="longdash"),
                                  xref="x", yref="y", layer="below")
                fig.add_shape(type="rect", x0=p_x0, x1=max(p_x0+1, p_x1),
                              y0=price_lo, y1=price_hi,
                              fillcolor=f"rgba({r_c},{g_c},{b_c},0.04)", line_width=0,
                              xref="x", yref="y", layer="below")
                candidate_y = price_hi - label_step * 0.5
                for used_y in pat_label_ys:
                    if abs(candidate_y - used_y) < label_step:
                        candidate_y = used_y - label_step
                candidate_y = max(price_lo + label_step, candidate_y)
                pat_label_ys.append(candidate_y)
                emoji = "📈" if p.direction == "bullish" else "📉" if p.direction == "bearish" else "◀▶"
                fig.add_annotation(x=p_x0, y=candidate_y,
                                   text=f"<b>{emoji} {p.name} ({p.confidence:.0%})</b>",
                                   showarrow=False, xanchor="left",
                                   font=dict(color=p_color, size=10),
                                   bgcolor="rgba(20,20,20,0.75)", bordercolor=p_color, borderpad=3)

    # Entry / SL / TP lines
    if None not in (entry, sl, tp1, tp2):
        try:
            e_v, s_v, t1_v, t2_v = float(entry), float(sl), float(tp1), float(tp2)
            conf = getattr(result, "confluence", None) if result else None
            direction = (conf.direction if conf else "sideways") if conf else "sideways"
            is_long  = direction in ("bullish", "long")
            is_short = direction in ("bearish", "short")

            def _hline(y, color, dash, label):
                fig.add_shape(type="line", x0=0, x1=last_xi, y0=y, y1=y,
                              line=dict(color=color, width=1.5, dash=dash), xref="x", yref="y")
                fig.add_annotation(x=last_xi + 0.5, y=y,
                                   text=f"<b>{label}: {y:.2f}</b>",
                                   showarrow=False, xanchor="left",
                                   font=dict(color="#111111", size=11),
                                   bgcolor=color, bordercolor=color, borderwidth=1, borderpad=4)

            _hline(t2_v, "#00bcd4", "solid", "TP2")
            _hline(t1_v, "#00bcd4", "solid", "TP1")
            _hline(e_v,  "#fdd835", "dash",  "Entry")
            _hline(s_v,  "#ef5350", "solid", "SL")

            fig.add_hrect(y0=min(s_v, e_v), y1=max(s_v, e_v),
                          fillcolor="rgba(239,83,80,0.13)", line_width=0)
            fig.add_hrect(y0=min(t1_v, t2_v), y1=max(t1_v, t2_v),
                          fillcolor="rgba(38,166,154,0.13)", line_width=0)

            rr  = abs(t1_v - e_v) / max(abs(e_v - s_v), 1e-6)
            rr2 = abs(t2_v - e_v) / max(abs(e_v - s_v), 1e-6)
            grade = (conf.grade if conf else "?") if conf else "?"
            score = (conf.total if conf else 0)   if conf else 0
            side  = "🟢 BUY" if is_long else "🔴 SELL" if is_short else "⚪ NEUTRAL"
            fig.update_layout(title=dict(
                text=(f"<b>{side}</b> | Grade <b>{grade}</b> | "
                      f"Score <b>{score:.0f}</b> | "
                      f"R:R TP1 <b>1:{rr:.1f}</b>  TP2 <b>1:{rr2:.1f}</b>"),
                font=dict(size=13),
            ))
        except Exception:
            pass

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#dde1ea", family="Inter"),
        xaxis=dict(rangeslider_visible=False, tickvals=tvals, ticktext=ttxt, tickangle=-30),
        yaxis=dict(side="right"), height=520,
        margin=dict(l=10, r=160, t=60, b=40),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Data extractors
# ─────────────────────────────────────────────────────────────────────────────

def _score_card_html(score: float, grade: str, direction: str, price: float) -> str:
    GC = {"A":"#2ecc71","B":"#f1c40f","C":"#e67e22","D":"#e74c3c","F":"#7f8c8d"}
    gc = GC.get(grade, "#dde1ea")
    dc = ("#2ecc71" if "bull" in direction.lower()
          else "#e74c3c" if "bear" in direction.lower() else "#f1c40f")

    def card(lbl, val, col):
        return (f'<div style="background:#1c2232;border-radius:8px;padding:14px 22px;'
                f'border-left:4px solid {col};min-width:110px">'
                f'<div style="font-size:.72rem;color:#7f8c8d;text-transform:uppercase">{lbl}</div>'
                f'<div style="font-size:1.9rem;font-weight:700;color:{col};line-height:1.1">{val}</div></div>')

    return ('<div style="display:flex;flex-wrap:wrap;gap:14px;padding:6px 0">'
            + card("Grade", grade, gc)
            + card("Score", f"{score:.1f}", "#00c4ff")
            + card("Direction", direction.upper(), dc)
            + card("Price", f"{price:.2f}", "#dde1ea")
            + "</div>")


def _rr_metrics_html(entry, sl, tp1, tp2, direction="sideways") -> str:
    try:
        e, s, t1, t2 = float(entry), float(sl), float(tp1), float(tp2)
        rr1 = abs(t1 - e) / max(abs(e - s), 1e-6)
        rr2 = abs(t2 - e) / max(abs(e - s), 1e-6)
        dc = ("#2ecc71" if "bull" in direction.lower()
              else "#e74c3c" if "bear" in direction.lower() else "#aaa")

        def mc(lbl, val, col):
            return (f'<div style="background:#1c2232;border-radius:6px;padding:8px 16px;'
                    f'border-left:3px solid {col}">'
                    f'<div style="font-size:.7rem;color:#7f8c8d">{lbl}</div>'
                    f'<div style="font-size:1rem;font-weight:700;color:{col}">{val}</div></div>')

        return (f'<div style="display:flex;flex-wrap:wrap;gap:10px;padding:4px 0">'
                + mc("Entry",   f"{e:.2f}",  "#fdd835")
                + mc("SL",      f"{s:.2f}",  "#ef5350")
                + mc(f"TP1 (1:{rr1:.1f})", f"{t1:.2f}", "#00bcd4")
                + mc(f"TP2 (1:{rr2:.1f})", f"{t2:.2f}", "#26a69a")
                + mc("Direction", direction.upper(), dc)
                + "</div>")
    except Exception:
        return ""


def _derive_levels(df: pd.DataFrame, result) -> Tuple[float, float, float, float]:
    price = float(df["close"].iloc[-1])
    atr   = float((df["high"] - df["low"]).rolling(14).mean().iloc[-1])
    conf  = getattr(result, "confluence", None)
    direction = (conf.direction if conf else "sideways").lower()
    is_long  = direction in ("bullish", "long")
    is_short = direction in ("bearish", "short")

    sup_lvl = res_lvl = None
    sr = getattr(result, "sr", None)
    if sr:
        ns = getattr(sr, "nearest_support", None)
        nr = getattr(sr, "nearest_resistance", None)
        if ns:
            _s = float(getattr(ns, "lower", getattr(ns, "level", 0)))
            if _s < price:
                sup_lvl = _s
        if nr:
            _r = float(getattr(nr, "upper", getattr(nr, "level", 0)))
            if _r > price:
                res_lvl = _r

    if is_long:
        entry = price
        sl    = (sup_lvl - atr * 0.2) if sup_lvl else price - 1.5 * atr
        tp1   = res_lvl if res_lvl else price + 2.0 * atr
        tp2   = tp1 + (tp1 - entry)
        if sl >= entry or (entry - sl) < atr:
            sl = entry - atr
        if tp1 <= entry:
            tp1 = entry + 2.0 * atr; tp2 = tp1 + (tp1 - entry)
    elif is_short:
        entry = price
        sl    = (res_lvl + atr * 0.2) if res_lvl else price + 1.5 * atr
        tp1   = sup_lvl if sup_lvl else price - 2.0 * atr
        tp2   = tp1 - (entry - tp1)
        if sl <= entry or (sl - entry) < atr:
            sl = entry + atr
        if tp1 >= entry:
            tp1 = entry - 2.0 * atr; tp2 = tp1 - (entry - tp1)
    else:
        entry = price; sl = price - 1.5*atr; tp1 = price + 2.0*atr; tp2 = price + 4.0*atr

    return round(entry, 2), round(sl, 2), round(tp1, 2), round(tp2, 2)


def _extract_ai_report(result) -> Tuple:
    # Component scores
    try:
        items = list(result.confluence.component_scores.items())
        comp_df = pd.DataFrame(items, columns=["Component", "Score"])
    except Exception:
        comp_df = pd.DataFrame(columns=["Component", "Score"])

    # Reasons
    try:
        reasons = result.confluence.reasons or []
        reasons_html = "<br>".join(f"• {r}" for r in reasons) or "—"
    except Exception:
        reasons_html = "—"

    # Dual scenario
    try:
        rep  = result.report
        bull = getattr(rep, "bullish_scenario", "") or ""
        bear = getattr(rep, "bearish_scenario", "") or ""
        inv  = getattr(rep, "invalidation", "")    or ""
        if bull or bear:
            scenarios_html = (
                '<div style="display:flex;gap:12px;flex-wrap:wrap">'
                + (f'<div style="flex:1;min-width:280px;background:#0d3b1e;border:2px solid #2ecc71;'
                   f'border-radius:10px;padding:16px"><h4 style="color:#2ecc71;margin-top:0">'
                   f'📈 SCENARIO A — TRUE BREAKOUT</h4>'
                   f'<pre style="color:#cfffdf;white-space:pre-wrap;font-size:.82rem">{bull}</pre></div>'
                   if bull else "")
                + (f'<div style="flex:1;min-width:280px;background:#3b0d0d;border:2px solid #e74c3c;'
                   f'border-radius:10px;padding:16px"><h4 style="color:#e74c3c;margin-top:0">'
                   f'📉 SCENARIO B — LIQUIDITY SWEEP</h4>'
                   f'<pre style="color:#ffd5d5;white-space:pre-wrap;font-size:.82rem">{bear}</pre></div>'
                   if bear else "")
                + "</div>"
                + (f'<div style="background:#1a1a2e;border:1px solid #f39c12;border-radius:8px;'
                   f'padding:12px;margin-top:12px"><span style="color:#f39c12;font-weight:700">⚠️ Invalidation</span><br>'
                   f'<pre style="color:#fdf3d0;white-space:pre-wrap;font-size:.82rem;margin:6px 0 0">{inv}</pre></div>'
                   if inv else "")
            )
        else:
            scenarios_html = "—"
    except Exception:
        scenarios_html = "—"

    # Final signal
    try:
        sig = getattr(result.report, "final_signal", "") or ""
        if sig and "Insufficient" not in sig and "ranging" not in sig.lower():
            is_buy    = "BUY" in sig.upper()
            sig_bg    = "#0a2e1a" if is_buy else "#2e0a0a"
            sig_border= "#2ecc71" if is_buy else "#e74c3c"
            sig_hdr   = "🟢 BUY SIGNAL" if is_buy else "🔴 SELL SIGNAL"
            sig_hcol  = "#2ecc71" if is_buy else "#e74c3c"
            sig_tcol  = "#c8ffd4" if is_buy else "#ffd4d4"
            signal_html = (
                f'<div style="background:{sig_bg};border:2px solid {sig_border};'
                f'border-radius:12px;padding:20px"><h3 style="color:{sig_hcol};margin-top:0;'
                f'letter-spacing:2px">{sig_hdr}</h3>'
                f'<pre style="color:{sig_tcol};white-space:pre-wrap;font-size:.88rem;line-height:1.7;margin:0">{sig}</pre></div>'
            )
        else:
            signal_html = f'<div style="color:#aaa;padding:8px">{sig or "No signal generated."}</div>'
    except Exception:
        signal_html = "—"

    return comp_df, reasons_html, scenarios_html, signal_html


def _extract_patterns_smc(result) -> Tuple:
    # Chart patterns
    try:
        pat = result.pat
        if pat and pat.patterns:
            patterns_df = pd.DataFrame([{
                "Pattern": p.name, "Direction": p.direction,
                "Confidence": f"{p.confidence:.0%}",
                "Target": f"{p.target_price:.4f}" if p.target_price else "—",
                "Invalidation": f"{p.invalidation:.4f}" if getattr(p,"invalidation",None) else "—",
            } for p in pat.patterns[:15]])
        else:
            patterns_df = pd.DataFrame(columns=["Pattern","Direction","Confidence","Target","Invalidation"])
    except Exception:
        patterns_df = pd.DataFrame()

    # Candlestick signals
    try:
        cs = result.cs
        if cs and cs.top_signals:
            cs_df = pd.DataFrame([{
                "Pattern": s.pattern, "Direction": s.direction,
                "Raw Score": f"{s.raw_score:.1f}", "Final": f"{s.final_score:.1f}",
            } for s in cs.top_signals])
        else:
            cs_df = pd.DataFrame(columns=["Pattern","Direction","Raw Score","Final"])
    except Exception:
        cs_df = pd.DataFrame()

    # SMC metrics HTML
    try:
        smc = result.smc
        ob_n  = len(smc.order_blocks)    if smc and smc.order_blocks    else 0
        fvg_n = len(smc.fair_value_gaps) if smc and smc.fair_value_gaps else 0
        sh_n  = len(smc.stop_hunts)      if smc and smc.stop_hunts      else 0
        lp_n  = len(smc.liquidity_pools) if smc and smc.liquidity_pools else 0
        narr  = getattr(smc, "narrative", "") or "" if smc else ""

        def mc(lbl, val):
            return (f'<div style="background:#1c2232;border-radius:6px;padding:10px 18px;'
                    f'border-left:3px solid #00c4ff;min-width:90px">'
                    f'<div style="font-size:.7rem;color:#7f8c8d">{lbl}</div>'
                    f'<div style="font-size:1.4rem;font-weight:700;color:#00c4ff">{val}</div></div>')

        smc_html = (
            f'<div style="display:flex;flex-wrap:wrap;gap:10px;padding:4px 0">'
            + mc("Order Blocks", ob_n) + mc("FVGs", fvg_n)
            + mc("Stop Hunts", sh_n) + mc("Liq Pools", lp_n)
            + "</div>"
            + (f'<div style="margin-top:8px;color:#aaa;font-size:.85rem">{narr}</div>' if narr else "")
        )
    except Exception:
        smc_html = "SMC data unavailable."

    # OB dataframe
    try:
        smc = result.smc
        if smc and smc.order_blocks:
            ob_df = pd.DataFrame([{
                "Type": ob.direction, "High": f"{ob.high:.4f}", "Low": f"{ob.low:.4f}",
                "Mitigated": ob.mitigated, "Strength": f"{ob.strength:.2f}",
            } for ob in smc.order_blocks[:12]])
        else:
            ob_df = pd.DataFrame(columns=["Type","High","Low","Mitigated","Strength"])
    except Exception:
        ob_df = pd.DataFrame()

    # FVG dataframe
    try:
        smc = result.smc
        if smc and smc.fair_value_gaps:
            fvg_df = pd.DataFrame([{
                "Type": f.direction, "High": f"{f.high:.4f}", "Low": f"{f.low:.4f}",
                "Filled": getattr(f, "filled", False),
            } for f in smc.fair_value_gaps[:12]])
        else:
            fvg_df = pd.DataFrame(columns=["Type","High","Low","Filled"])
    except Exception:
        fvg_df = pd.DataFrame()

    # AMD HTML
    try:
        amd = getattr(result, "amd", None)
        if amd is not None:
            phase_colors = {"accumulation":"#1565c0","manipulation":"#e65100",
                            "distribution":"#2e7d32","unknown":"#555555"}
            phase_emojis = {"accumulation":"🔵","manipulation":"🟠","distribution":"🟢","unknown":"⚪"}
            pc = phase_colors.get(amd.phase, "#555")
            pe = phase_emojis.get(amd.phase, "⚪")
            asian_str = f"{amd.asian_low:.4f} – {amd.asian_high:.4f}" if amd.asian_high else "—"
            sweep_str = (f"✅ {amd.sweep_side.upper()} @ {amd.sweep_price:.4f}"
                         if amd.manipulation_swept else "⏳ Awaiting")

            def amc(lbl, val):
                return (f'<div style="background:#1c2232;border-radius:6px;padding:8px 14px;'
                        f'border-left:3px solid #00c4ff">'
                        f'<div style="font-size:.7rem;color:#7f8c8d">{lbl}</div>'
                        f'<div style="font-weight:700">{val}</div></div>')

            amd_html = (
                f'<div style="margin-bottom:8px">'
                f'<span style="background:{pc};color:#fff;padding:3px 10px;border-radius:4px;font-weight:700">'
                f'{pe} {amd.phase.upper()}</span>'
                + (f' &nbsp; Direction: <b>{amd.direction.upper()}</b>' if amd.direction != "neutral" else "")
                + f' &nbsp; Confidence: <b>{amd.confidence:.0%}</b></div>'
                + f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px">'
                + amc("Asian Range", asian_str)
                + amc("Manipulation", sweep_str)
                + amc("Dist FVGs", len(amd.entry_fvgs) if amd.entry_fvgs else 0)
                + "</div>"
                + (f'<div style="color:#aaa;font-size:.85rem">{amd.note}</div>' if amd.note else "")
                + (f'<div style="background:#0d3b1e;border:1px solid #2ecc71;border-radius:6px;'
                   f'padding:8px;margin-top:8px"><span style="color:#2ecc71;font-weight:700">'
                   f'Best Entry FVG ({amd.best_entry_fvg.direction.upper()})</span>: '
                   f'{amd.best_entry_fvg.low:.4f} – {amd.best_entry_fvg.high:.4f}'
                   f' (mid: {amd.best_entry_fvg.mid:.4f})</div>'
                   if amd.best_entry_fvg else "")
            )
        else:
            amd_html = "AMD cycle data unavailable (datetime index required)."
    except Exception as e:
        amd_html = f"AMD error: {e}"

    # Fibonacci dataframe
    try:
        fib = result.fib
        if fib and fib.levels:
            fib_df = pd.DataFrame([{
                "Level": lvl.label, "Price": f"{lvl.price:.4f}",
                "Ratio": f"{lvl.ratio:.3f}", "Key": "✅" if lvl.is_key else "",
                "Reaction": f"{lvl.reaction_score:.2f}",
            } for lvl in fib.levels])
        else:
            fib_df = pd.DataFrame(columns=["Level","Price","Ratio","Key","Reaction"])
    except Exception:
        fib_df = pd.DataFrame()

    # MTF dataframe
    try:
        mtf = result.mtf
        if mtf and mtf.biases:
            mtf_df = pd.DataFrame([{
                "TF": b.timeframe.upper(),
                "Trend": ("🟢 Bullish" if b.bias == "bullish"
                          else "🔴 Bearish" if b.bias == "bearish" else "🟡 Sideways"),
                "Score":  f"{b.score:+.2f}" if hasattr(b, "score")  else "—",
                "Weight": f"{int(b.weight*100)}%" if hasattr(b, "weight") else "—",
            } for b in mtf.biases])
        else:
            mtf_df = pd.DataFrame(columns=["TF","Trend","Score","Weight"])
    except Exception:
        mtf_df = pd.DataFrame()

    return patterns_df, cs_df, smc_html, ob_df, fvg_df, amd_html, fib_df, mtf_df


def _extract_ml(result) -> Tuple[str, str]:
    try:
        mp = result.ml_prediction
        if mp:
            def mc(lbl, val):
                return (f'<div style="background:#1c2232;border-radius:6px;padding:10px 18px;'
                        f'border-left:3px solid #00c4ff">'
                        f'<div style="font-size:.7rem;color:#7f8c8d">{lbl}</div>'
                        f'<div style="font-size:1.3rem;font-weight:700;color:#00c4ff">{val}</div></div>')
            ml_html = (
                f'<div style="display:flex;flex-wrap:wrap;gap:10px;padding:4px 0">'
                + mc("Quality Score",   f"{mp.quality_score:.1%}")
                + mc("Win Probability", f"{mp.win_probability:.1%}")
                + mc("Model", getattr(mp, "model_used", "—"))
                + "</div>"
            )
        else:
            ml_html = "No ML prediction — model needs labeled training data."
    except Exception:
        ml_html = "ML prediction unavailable."

    try:
        engine = _get_engine()
        learner = engine.learner
        stats = {
            "total_setups": len(learner.records),
            "labeled": sum(1 for r in learner.records if r.outcome is not None),
            "model_trained": learner.model is not None,
        }
        model_stats_str = json.dumps(stats, indent=2)
    except Exception as e:
        model_stats_str = f"Could not load model stats: {e}"

    return ml_html, model_stats_str


# ─────────────────────────────────────────────────────────────────────────────
# DB / file helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_history(asset_filter: str = "", tf_filter: str = "",
                  limit: int = 50) -> pd.DataFrame:
    try:
        from storage.database import list_analyses
        rows = list_analyses(asset=asset_filter or None,
                             timeframe=tf_filter or None, limit=int(limit))
        if not rows:
            return pd.DataFrame(columns=["ID","Time","Asset","TF","Price","Score","Grade","Direction"])
        return pd.DataFrame([{
            "ID":        r.get("id", "—"),
            "Time":      str(r.get("created_at", ""))[:16],
            "Asset":     r.get("asset", "—"),
            "TF":        r.get("timeframe", "—"),
            "Price":     f"{r['current_price']:.2f}" if r.get("current_price") else "—",
            "Score":     f"{r['confluence_score']:.1f}" if r.get("confluence_score") else "—",
            "Grade":     r.get("grade", "—"),
            "Direction": r.get("direction", "—"),
        } for r in rows])
    except Exception as exc:
        return pd.DataFrame({"Error": [str(exc)]})


def _history_summary_html(df: pd.DataFrame) -> str:
    if df.empty or "Error" in df.columns:
        return ""
    total = len(df)
    try:
        avg_score = df["Score"].apply(lambda v: float(v) if v != "—" else None).dropna().mean()
        score_str = f"{avg_score:.1f}"
    except Exception:
        score_str = "—"
    try:
        most_asset = df["Asset"].value_counts().idxmax()
    except Exception:
        most_asset = "—"
    grade_a = (df.get("Grade", pd.Series()) == "A").sum()
    grade_b = (df.get("Grade", pd.Series()) == "B").sum()

    def mc(lbl, val):
        return (f'<div style="background:#1c2232;border-radius:6px;padding:8px 14px;'
                f'border-left:3px solid #00c4ff">'
                f'<div style="font-size:.7rem;color:#7f8c8d">{lbl}</div>'
                f'<div style="font-size:1.1rem;font-weight:700;color:#dde1ea">{val}</div></div>')

    return (f'<div style="display:flex;flex-wrap:wrap;gap:10px;padding:4px 0">'
            + mc("Total Records", total) + mc("Avg Score", score_str)
            + mc("Most Analyzed", most_asset)
            + mc("A-Grade Setups", grade_a) + mc("B-Grade Setups", grade_b)
            + "</div>")


def _load_bot_status() -> Tuple[str, pd.DataFrame, str, pd.DataFrame]:
    """Returns (status_html, positions_df, learn_html, trades_df)."""
    status_path = ROOT / "live_bot" / "bot_status.json"
    learn_path  = ROOT / "live_bot" / "learning_state.json"

    # ── Bot status ────────────────────────────────────────────────────────────
    try:
        with open(status_path, "r", encoding="utf-8") as fh:
            s = json.load(fh)

        from datetime import datetime, timezone
        poll_ts = s.get("last_poll", "")
        age_sec = None
        if poll_ts:
            try:
                age_sec = (datetime.now(timezone.utc).replace(tzinfo=None) - datetime.fromisoformat(poll_ts)).total_seconds()
            except Exception:
                pass
        online = age_sec is not None and age_sec < s.get("poll_interval", 60) * 2.5
        badge_col = "#22c55e" if online else "#ef4444"
        badge_txt = "🟢 ONLINE" if online else "🔴 OFFLINE"
        halted = s.get("trading_halted", False)

        def mc(lbl, val, col="#dde1ea"):
            return (f'<div style="background:#1c2232;border-radius:6px;padding:8px 14px;'
                    f'border-left:3px solid {col};min-width:90px">'
                    f'<div style="font-size:.7rem;color:#7f8c8d">{lbl}</div>'
                    f'<div style="font-size:1rem;font-weight:700;color:{col}">{val}</div></div>')

        status_html = (
            (f'<div style="background:#2e0a0a;border:1px solid #ef4444;border-radius:6px;'
             f'padding:8px;margin-bottom:8px;color:#ef4444;font-weight:700">'
             f'🚨 TRADING HALTED — {s.get("halt_reason","Circuit breaker")}</div>' if halted else "")
            + f'<div style="margin-bottom:6px">'
            + f'<span style="color:{badge_col};font-weight:700;font-size:1.1rem">{badge_txt}</span>'
            + (f' &nbsp;·&nbsp; Last poll: {poll_ts[:19].replace("T"," ")} UTC' if poll_ts else "")
            + (f' &nbsp;·&nbsp; {int(age_sec)}s ago' if age_sec else "")
            + "</div>"
            + f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px">'
            + mc("Mode", "DRY RUN" if s.get("dry_run") else "🔴 LIVE",
                 "#f59e0b" if s.get("dry_run") else "#ef4444")
            + mc("Asset", f"{s.get('asset','?')} {s.get('timeframe','?')}", "#00c4ff")
            + mc("Trades Sent", str(s.get("total_trades", 0)))
            + mc("Open Now", str(s.get("open_count", 0)))
            + (mc("Balance", f"${s['balance']:,.2f}", "#22c55e") if s.get("balance") else "")
            + (mc("Equity",  f"${s.get('equity',0):,.2f}") if s.get("equity") else "")
            + (mc("Unrealised", f"${s.get('total_unrealised',0):+,.2f}",
                  "#22c55e" if (s.get("total_unrealised",0) or 0) >= 0 else "#ef4444")
               if s.get("total_unrealised") is not None else "")
            + "</div>"
        )

        # Last signal card
        la_action = s.get("last_action", "—")
        la_grade  = s.get("last_signal_grade", "—")
        la_score  = s.get("last_signal_score", 0)
        la_dir    = s.get("last_signal_direction", "—")
        status_html += (
            f'<div style="background:#1c2232;border-radius:6px;padding:8px 14px;margin-top:6px">'
            f'<div style="font-size:.75rem;color:#7f8c8d;margin-bottom:4px">Last Signal</div>'
            f'Grade <b>{la_grade}</b> &nbsp;|&nbsp; Score <b>{la_score:.0f}</b>'
            f' &nbsp;|&nbsp; Direction <b>{la_dir}</b>'
            f'<div style="color:#aaa;font-size:.85rem;margin-top:4px">{la_action}</div></div>'
        )

        # Regime/session engine status
        _regime_d = s.get("regime", {})
        _sess_d   = s.get("session_detail", {})
        if _regime_d or _sess_d:
            rname = _regime_d.get("name","?").replace("_"," ").title() if _regime_d else "—"
            rlot  = _regime_d.get("lot_scalar", 1.0) if _regime_d else 1.0
            rtp   = _regime_d.get("tp_scalar",  1.0) if _regime_d else 1.0
            rconf = _regime_d.get("confidence",  0)   if _regime_d else 0
            sname = _sess_d.get("name","—").replace("_"," ").title() if _sess_d else "—"
            sdead = _sess_d.get("skip_new_entries", False) if _sess_d else False
            r_col = ("#22c55e" if "Expansion" in rname else
                     "#f59e0b" if "Compression" in rname else "#60a5fa")
            s_col = "#ef4444" if sdead else "#22c55e"
            status_html += (
                f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:8px">'
                + mc("Regime", rname, r_col)
                + mc("Conf", f"{rconf:.0%}", r_col)
                + mc("Lot×", f"{rlot:.2f}")
                + mc("TP×", f"{rtp:.2f}")
                + mc("Session", sname, s_col)
                + mc("Entry", "🚫 DEAD" if sdead else "✅ ACTIVE", s_col)
                + "</div>"
            )

        # Open positions
        open_pos = s.get("open_positions", [])
        if open_pos:
            pos_df = pd.DataFrame(open_pos)
            _pref_cols = [
                "ticket", "direction", "lots", "entry", "current", "unrealised",
                "open_minutes", "time_to_smart_min", "time_to_hard_min",
                "time_exit_profit_gap_usd", "entry_regime",
                "time_exit_smart_eligible", "time_exit_hard_due",
                "sl", "tp", "comment",
            ]
            _show = [c for c in _pref_cols if c in pos_df.columns]
            if _show:
                pos_df = pos_df[_show]
        else:
            pos_df = pd.DataFrame(
                columns=["ticket", "direction", "entry", "sl", "tp", "current", "unrealised", "lots"]
            )

    except FileNotFoundError:
        status_html = "bot_status.json not found — bot may not be running."
        pos_df = pd.DataFrame()
    except Exception as e:
        status_html = f"Error reading bot status: {e}"
        pos_df = pd.DataFrame()

    # ── Learning state ────────────────────────────────────────────────────────
    try:
        with open(learn_path, "r", encoding="utf-8") as fh:
            ls = json.load(fh)

        total_w = ls.get("wins", 0)
        total_l = ls.get("losses", 0)
        total_t = total_w + total_l
        wr_all  = total_w / total_t * 100 if total_t else 0
        adj     = ls.get("score_adjust", 0.0)
        streak  = ls.get("streak", 0)
        best_s  = ls.get("best_streak", 0)
        worst_s = ls.get("worst_streak", 0)
        tot_pnl = ls.get("total_pnl", 0.0)
        last20  = ls.get("last_20_results", [])
        r20_wr  = sum(last20) / len(last20) * 100 if last20 else 0
        saved_at = ls.get("saved_at", "")[:19].replace("T", " ")

        def lmc(lbl, val, col):
            return (f'<div style="background:#1c2232;border-radius:6px;padding:8px 14px;'
                    f'border-left:3px solid {col}">'
                    f'<div style="font-size:.7rem;color:#7f8c8d">{lbl}</div>'
                    f'<div style="font-size:1rem;font-weight:700;color:{col}">{val}</div></div>')

        learn_html = (
            f'<div style="font-size:.75rem;color:#7f8c8d;margin-bottom:8px">Saved: {saved_at} UTC</div>'
            f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px">'
            + lmc("All-time WR",  f"{wr_all:.0f}%  ({total_t} trades)",
                  "#22c55e" if wr_all >= 55 else "#f59e0b")
            + lmc("Last-20 WR",   f"{r20_wr:.0f}%",
                  "#22c55e" if r20_wr >= 55 else "#f59e0b")
            + lmc("Score Adj",    f"{adj:+.1f}",
                  "#22c55e" if adj >= 0 else "#ef4444")
            + lmc("Streak",       f"{'+' if streak > 0 else ''}{streak}  (best:{best_s} worst:{worst_s})",
                  "#22c55e" if streak > 0 else "#ef4444" if streak < 0 else "#aaa")
            + lmc("Total P&L",    f"${tot_pnl:+.2f}",
                  "#22c55e" if tot_pnl >= 0 else "#ef4444")
            + "</div>"
        )

        # Grade stats table
        gs = ls.get("grade_stats", {})
        if gs:
            learn_html += ("<div style='font-weight:700;color:#aaa;font-size:.85rem;margin-top:4px'>"
                           "Grade Performance</div>"
                           "<table style='border-collapse:collapse;font-size:.83rem;width:auto'><tr>"
                           + "".join(f"<th style='padding:3px 8px;color:#7f8c8d'>{h}</th>"
                                     for h in ["Grade","Seen","Taken","W","L","WR"])
                           + "</tr>")
            for gk in ["A","B","C","D"]:
                gv = gs.get(gk, {})
                gw = gv.get("wins",0); gl = gv.get("losses",0); gt = gw + gl
                wr_str = f"{gw/gt*100:.0f}%" if gt else "—"
                learn_html += (f"<tr><td style='padding:3px 8px;font-weight:700;color:#00c4ff'>{gk}</td>"
                               f"<td style='padding:3px 8px'>{gv.get('seen',0)}</td>"
                               f"<td style='padding:3px 8px'>{gv.get('acted',0)}</td>"
                               f"<td style='padding:3px 8px;color:#22c55e'>{gw}</td>"
                               f"<td style='padding:3px 8px;color:#ef4444'>{gl}</td>"
                               f"<td style='padding:3px 8px'>{wr_str}</td></tr>")
            learn_html += "</table>"

        # LTF alignment
        ltf = ls.get("ltf_stats", {})
        if ltf:
            ltf_labels = {"both_confirmed":"Both 1M+5M ✅","one_counter":"One counter ⚠️",
                          "trap":"Trap blocked 🚫","unknown":"Unknown ❓"}
            learn_html += ("<div style='font-weight:700;color:#aaa;font-size:.85rem;margin-top:8px'>"
                           "1M+5M LTF Entry Timing</div>"
                           "<table style='border-collapse:collapse;font-size:.83rem;width:auto'><tr>"
                           + "".join(f"<th style='padding:3px 8px;color:#7f8c8d'>{h}</th>"
                                     for h in ["State","W","L","WR"])
                           + "</tr>")
            for k, v in ltf.items():
                lw = v.get("wins",0); ll = v.get("losses",0); lt2 = lw + ll
                wr_s = f"{lw/lt2*100:.0f}%" if lt2 else "—"
                learn_html += (f"<tr><td style='padding:3px 8px'>{ltf_labels.get(k,k)}</td>"
                               f"<td style='padding:3px 8px;color:#22c55e'>{lw}</td>"
                               f"<td style='padding:3px 8px;color:#ef4444'>{ll}</td>"
                               f"<td style='padding:3px 8px'>{wr_s}</td></tr>")
            learn_html += "</table>"

        # OB stats
        obs = ls.get("ob_stats", {})
        if obs:
            learn_html += ("<div style='font-weight:700;color:#aaa;font-size:.85rem;margin-top:8px'>"
                           "Order Block Performance</div>"
                           "<table style='border-collapse:collapse;font-size:.83rem;width:auto'><tr>"
                           + "".join(f"<th style='padding:3px 8px;color:#7f8c8d'>{h}</th>"
                                     for h in ["OB Dir","Entries","Wins","WR"])
                           + "</tr>")
            for odir, ov in obs.items():
                oh = ov.get("hits",0); ow = ov.get("wins",0)
                learn_html += (f"<tr><td style='padding:3px 8px'>{odir.title()}</td>"
                               f"<td style='padding:3px 8px'>{oh}</td>"
                               f"<td style='padding:3px 8px;color:#22c55e'>{ow}</td>"
                               f"<td style='padding:3px 8px'>{ow/oh*100:.0f}% if oh else —</td></tr>")
            learn_html += "</table>"

    except FileNotFoundError:
        learn_html = "learning_state.json not found — no closed trades yet."
    except Exception as e:
        learn_html = f"Error reading learning state: {e}"

    # ── Recent trades ─────────────────────────────────────────────────────────
    try:
        from storage.database import list_trades
        trades = list_trades(asset=None, source="live", limit=50)
        if trades:
            trades_df = pd.DataFrame(trades)
            display_cols = ["created_at","direction","entry_price","sl_price","tp_price",
                            "exit_price","size","pnl","status","session","regime",
                            "score_at_entry","exit_reason","mae","mfe"]
            trades_df = trades_df[[c for c in display_cols if c in trades_df.columns]].copy()
            trades_df.rename(columns={
                "created_at":"Time","direction":"Dir","entry_price":"Entry",
                "sl_price":"SL","tp_price":"TP","exit_price":"Exit",
                "size":"Lots","pnl":"P&L","status":"Status","session":"Session",
                "regime":"Regime","score_at_entry":"Score",
                "exit_reason":"Exit Reason","mae":"MAE","mfe":"MFE",
            }, inplace=True)
            if "Time" in trades_df.columns:
                trades_df["Time"] = trades_df["Time"].astype(str).str[:16]
            for pc in ["Entry","SL","TP","Exit"]:
                if pc in trades_df.columns:
                    trades_df[pc] = trades_df[pc].apply(
                        lambda v: f"{v:.2f}" if pd.notna(v) else "—")
            if "P&L" in trades_df.columns:
                trades_df["P&L"] = trades_df["P&L"].apply(
                    lambda v: f"+{v:.2f}" if (pd.notna(v) and v >= 0) else (f"{v:.2f}" if pd.notna(v) else "—"))
            if "Score" in trades_df.columns:
                trades_df["Score"] = trades_df["Score"].apply(
                    lambda v: f"{v:.0f}" if pd.notna(v) else "—")
        else:
            trades_df = pd.DataFrame(columns=["Time","Dir","Entry","SL","TP","Exit","Lots","P&L","Status"])
    except Exception as e:
        trades_df = pd.DataFrame({"Error": [str(e)]})

    return status_html, pos_df, learn_html, trades_df


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis handler
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(
    file_obj,
    tf1_file, tf2_file, tf3_file,
    tf1_label, tf2_label, tf3_label,
    asset, timeframe, json_text,
):
    """Full pipeline. Returns 29 values populating all tabs simultaneously."""
    _E = pd.DataFrame()
    _EMPTY = (
        "—", "—", "—",
        "Upload a CSV/JSON file or paste JSON to begin.", "",
        None,
        0.0, 0.0, 0.0, 0.0, "",
        _E, "—", "—", "—",
        None,
        _E, _E, "—", _E, _E, "—", _E, _E,
        "—", "—",
        None, None,
    )

    asset     = (asset or "XAUUSD").strip().upper()
    timeframe = (timeframe or "4H").strip()

    try:
        df = None
        tf_data: dict = {}

        # Primary file
        if file_obj is not None:
            path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
            df = _load_json_ohlcv(path) if path.lower().endswith(".json") else _read_csv_smart(path)
            tf_data[timeframe.lower()] = df

        # JSON paste fallback
        if df is None and json_text and json_text.strip() not in ("", "[]"):
            try:
                rows = json.loads(json_text)
                df_j = pd.DataFrame(rows)
                df_j.columns = [c.lower() for c in df_j.columns]
                dt_col = next((c for c in df_j.columns if "time" in c or "date" in c), None)
                if dt_col:
                    df_j.index = pd.to_datetime(df_j[dt_col], errors="coerce")
                    df_j = df_j.drop(columns=[dt_col])
                df_j.index.name = "datetime"
                for col in ("open","high","low","close"):
                    df_j[col] = pd.to_numeric(df_j[col], errors="coerce")
                if "volume" not in df_j.columns:
                    df_j["volume"] = 0.0
                df = df_j.dropna(subset=["open","high","low","close"]).sort_index()
                tf_data[timeframe.lower()] = df
            except Exception as je:
                empty = list(_EMPTY)
                empty[3] = f"JSON parse error:\n{je}"
                return tuple(empty)

        if df is None or len(df) < 20:
            return _EMPTY

        # Extra TF files
        for ef, lbl in [(tf1_file, tf1_label), (tf2_file, tf2_label), (tf3_file, tf3_label)]:
            if ef is not None and lbl and lbl.strip():
                try:
                    ep = ef.name if hasattr(ef, "name") else str(ef)
                    tf_data[lbl.strip().lower()] = _read_csv_smart(ep)
                except Exception:
                    pass

        mtf_payload = tf_data if len(tf_data) > 1 else None

        # Run engine
        engine = _get_engine()
        result = engine.analyze_data(
            df=df, asset=asset, timeframe=timeframe,
            tf_data=mtf_payload, render_chart=False, save_to_db=True,
        )

        # Confluence basics
        conf      = result.confluence
        score     = conf.total     if conf else 0.0
        grade     = conf.grade     if conf else "F"
        direction = conf.direction if conf else "—"
        price     = result.current_price or 0.0
        report_txt= result.report.full_text if result.report else "No report generated."
        summary   = _score_card_html(score, grade, direction, price)
        score_str = f"{score:.1f} / 100"

        # Charts
        base_chart  = _build_base_chart(df, result)
        entry, sl, tp1, tp2 = _derive_levels(df, result)
        setup_chart = _build_setup_chart(df, result, entry, sl, tp1, tp2)
        rr_h        = _rr_metrics_html(entry, sl, tp1, tp2, direction)

        # AI Report tab
        comp_df, reasons_html, scenarios_html, signal_html = _extract_ai_report(result)

        # Patterns & SMC tab
        (patterns_df, cs_df, smc_html,
         ob_df, fvg_df, amd_html,
         fib_df, mtf_df) = _extract_patterns_smc(result)

        # ML tab
        ml_html, model_stats = _extract_ml(result)

        return (
            score_str, grade, direction,
            report_txt, summary,
            base_chart,
            entry, sl, tp1, tp2, rr_h,
            comp_df, reasons_html, scenarios_html, signal_html,
            setup_chart,
            patterns_df, cs_df, smc_html,
            ob_df, fvg_df, amd_html,
            fib_df, mtf_df,
            ml_html, model_stats,
            result, df,
        )

    except Exception:
        err = traceback.format_exc()
        empty = list(_EMPTY)
        empty[3] = f"Analysis failed:\n\n{err}"
        return tuple(empty)


def refresh_setup_chart_fn(result_state, df_state, entry, sl, tp1, tp2):
    if df_state is None or result_state is None:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1117",
                          title="Run analysis first", height=480)
        return fig, ""
    try:
        conf = getattr(result_state, "confluence", None)
        direction = (conf.direction if conf else "sideways") if conf else "sideways"
        fig = _build_setup_chart(df_state, result_state,
                                 float(entry), float(sl), float(tp1), float(tp2))
        rr_h = _rr_metrics_html(entry, sl, tp1, tp2, direction)
        return fig, rr_h
    except Exception as e:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.update_layout(title=f"Error: {e}", template="plotly_dark",
                          paper_bgcolor="#0e1117", height=480)
        return fig, ""


def run_backtest_handler(df_state, asset, timeframe,
                         capital, risk_pct, min_conf, max_bars, stride, min_rr):
    if df_state is None:
        return "Load OHLCV data first (run analysis in the Input tab).", None, pd.DataFrame()
    try:
        from backtesting.backtester import Backtester, Signal as BTSignal

        df_bt = df_state.iloc[-int(max_bars):].copy()
        engine = _get_engine()
        risk_frac = float(risk_pct) / 100.0
        min_conf_raw = float(min_conf)

        def _strategy(df_slice):
            if len(df_slice) < 30:
                return None
            # ── Session filter: London only (07:00–11:59 UTC) ───────────────────────
            # Mirrors live bot session gate: only London open + London sessions trade.
            try:
                _ts = df_slice.index[-1]
                _hr = _ts.hour if hasattr(_ts, "hour") else _ts.to_pydatetime().hour
                if not (7 <= _hr <= 11):
                    return None
            except Exception:
                pass  # no timestamp index — skip filter gracefully
            try:
                res = engine.analyze_data(df_slice, asset=asset, timeframe=timeframe,
                                          render_chart=False, save_to_db=False)
                if not res.confluence or res.confluence.total < min_conf_raw:
                    return None
                direction = res.confluence.direction
                if direction not in ("bullish","bearish","long","short"):
                    return None
                price = float(df_slice["close"].iloc[-1])
                atr   = float((df_slice["high"] - df_slice["low"]).rolling(14).mean().iloc[-1])
                # ── Regime filter: block compression + mean reversion ────────────────
                # Matches live bot regime gates (0W/8L compression, 0W/28L mean-reversion).
                try:
                    _cl   = df_slice["close"]
                    _rm   = _cl.rolling(20).mean()
                    _rs   = _cl.rolling(20).std()
                    _bbw  = ((2 * _rs) / (_rm + 1e-8)).dropna()
                    if len(_bbw) >= 30:
                        _bw_now = float(_bbw.iloc[-1])
                        _bw_p25 = float(_bbw.quantile(0.25))
                        _bw_p50 = float(_bbw.quantile(0.50))
                        if _bw_now <= _bw_p25:          # compression: BB width bottom quartile
                            return None
                        _above  = int((_cl.tail(20) > _rm.tail(20)).sum())
                        if _bw_now <= _bw_p50 and 7 <= _above <= 13:  # mean reversion
                            return None
                except Exception:
                    pass
                is_long = direction in ("bullish","long")
                sl_price = price - 1.5*atr if is_long else price + 1.5*atr
                tp_price = price + 3.0*atr if is_long else price - 3.0*atr
                if res.sr:
                    ns = res.sr.nearest_support
                    nr = res.sr.nearest_resistance
                    if is_long and ns:
                        _s = float(getattr(ns,"lower",getattr(ns,"level",price)))
                        if _s < price:
                            sl_price = _s - atr * 0.2
                    if is_long and nr:
                        _r = float(getattr(nr,"level",price))
                        if _r > price:
                            tp_price = _r
                    if not is_long and nr:
                        _r = float(getattr(nr,"upper",getattr(nr,"level",price)))
                        if _r > price:
                            sl_price = _r + atr * 0.2
                    if not is_long and ns:
                        _s = float(getattr(ns,"level",price))
                        if _s < price:
                            tp_price = _s
                return BTSignal(direction="long" if is_long else "short",
                                entry_price=price, sl_price=sl_price, tp_price=tp_price,
                                confidence=res.confluence.total / 100.0)
            except Exception:
                return None

        backtester = Backtester(initial_capital=float(capital), risk_per_trade=risk_frac,
                                signal_stride=int(stride), min_rr=float(min_rr),
                                dual_tp=True)
        bt = backtester.run(df_bt, _strategy, min_confidence=float(min_conf) / 100.0)

        def mc(lbl, val, col):
            return (f'<div style="background:#1c2232;border-radius:6px;padding:8px 14px;'
                    f'border-left:3px solid {col}">'
                    f'<div style="font-size:.7rem;color:#7f8c8d">{lbl}</div>'
                    f'<div style="font-size:1.2rem;font-weight:700;color:{col}">{val}</div></div>')

        res_html = (
            f'<div style="display:flex;flex-wrap:wrap;gap:10px;padding:4px 0">'
            + mc("Trades",        str(bt.total_trades),        "#00c4ff")
            + mc("Win Rate",      f"{bt.win_rate:.1%}",        "#22c55e" if bt.win_rate >= 0.5 else "#f59e0b")
            + mc("Profit Factor", f"{bt.profit_factor:.2f}",   "#22c55e" if bt.profit_factor >= 1.5 else "#f59e0b")
            + mc("Net Return",    f"{bt.net_return_pct:.1%}",  "#22c55e" if bt.net_return_pct >= 0 else "#ef4444")
            + mc("Max Drawdown",  f"{bt.max_drawdown_pct:.1%}","#ef4444")
            + mc("Sharpe",        f"{bt.sharpe_ratio:.2f}",    "#22c55e" if bt.sharpe_ratio >= 1 else "#aaa")
            + mc("Avg R:R",       f"{bt.avg_rr:.2f}",          "#00c4ff")
            + mc("Final Equity",  f"${bt.final_equity:,.2f}",  "#22c55e" if bt.final_equity >= float(capital) else "#ef4444")
            + "</div>"
        )

        import plotly.graph_objects as go
        eq_fig = None
        if bt.equity_curve:
            color = "#22c55e" if bt.equity_curve[-1] >= float(capital) else "#ef4444"
            eq_fig = go.Figure(go.Scatter(
                y=bt.equity_curve, mode="lines",
                line=dict(color=color, width=2), fill="tozeroy", name="Equity",
            ))
            eq_fig.add_hline(y=float(capital), line_color="#555", line_width=1, line_dash="dash")
            eq_fig.update_layout(
                template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                font=dict(color="#dde1ea"),
                title=f"Equity Curve — {bt.total_trades} trades  |  ${bt.final_equity:,.2f} final",
                height=300, margin=dict(l=40, r=20, t=40, b=20),
                yaxis_title="Equity $", xaxis_title="Trade #",
            )

        trades_df = pd.DataFrame()
        if bt.trades:
            trades_df = pd.DataFrame([{
                "Direction": t.direction,
                "Entry": f"{t.entry_price:.4f}",
                "Exit":  f"{t.exit_price:.4f}" if t.exit_price else "open",
                "P&L":   f"{t.pnl:+.4f}",
                "R:R":   f"{t.rr_achieved:.2f}",
                "Win":   "✅" if t.is_win else "❌",
            } for t in bt.trades[-50:]])

        return res_html, eq_fig, trades_df

    except Exception:
        return f"Backtest failed:\n{traceback.format_exc()}", None, pd.DataFrame()


def label_outcome_handler(run_id: str, outcome: str, rr: float) -> str:
    try:
        engine = _get_engine()
        engine.label_outcome(run_id=run_id.strip(),
                             outcome=1 if outcome == "Win" else 0,
                             rr=float(rr) if float(rr) > 0 else None)
        return "✅ Outcome recorded. Model will retrain automatically."
    except Exception as e:
        return f"❌ Error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Build UI
# ─────────────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Prometheus · Market Analysis") as demo:

        gr.HTML("""
        <div class="prometheus-header">
            <h1>📊 Prometheus</h1>
            <p>Institutional AI Market Analysis · XAUUSDm &nbsp;|&nbsp; Gradio UI · port 7860</p>
        </div>
        """)

        result_state = gr.State(None)
        df_state     = gr.State(None)

        with gr.Tabs():

            # ══════════════════════════════════════════════════════════════════
            # TAB 1  Input & Analyze
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("📥 Input & Analyze"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=280):
                        gr.Markdown("### ⚙️ Settings")
                        file_input  = gr.File(label="Primary OHLCV file (CSV / TXT / JSON)",
                                              file_types=[".csv",".txt",".json"])
                        asset_input = gr.Textbox(label="Asset Symbol", value="XAUUSD")
                        tf_input    = gr.Dropdown(label="Primary Timeframe",
                                                  choices=TIMEFRAMES, value="4h")
                        with gr.Accordion("📋 Paste JSON (alternative input)", open=False):
                            gr.Markdown("JSON array: `[{datetime, open, high, low, close, volume}, …]`")
                            json_input = gr.Textbox(label="JSON", lines=5,
                                                    placeholder='[{"datetime":"2024-01-01","open":2000,…}]')
                        with gr.Accordion("➕ Extra Timeframes for MTF", open=False):
                            gr.Markdown("Upload additional TF CSVs to boost MTF confluence scoring.")
                            with gr.Row():
                                tf1_label = gr.Textbox(label="TF2 label", value="1d", scale=1)
                                tf1_file  = gr.File(label="TF2 CSV", file_types=[".csv",".txt"], scale=2)
                            with gr.Row():
                                tf2_label = gr.Textbox(label="TF3 label", value="1h", scale=1)
                                tf2_file  = gr.File(label="TF3 CSV", file_types=[".csv",".txt"], scale=2)
                            with gr.Row():
                                tf3_label = gr.Textbox(label="TF4 label", value="15m", scale=1)
                                tf3_file  = gr.File(label="TF4 CSV", file_types=[".csv",".txt"], scale=2)
                        run_btn = gr.Button("▶  Run Full Analysis", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 Results")
                        summary_html_out = gr.HTML()
                        with gr.Row():
                            score_out = gr.Textbox(label="Confluence Score", interactive=False, scale=2)
                            grade_out = gr.Textbox(label="Grade",            interactive=False, scale=1)
                            dir_out   = gr.Textbox(label="Direction",        interactive=False, scale=2)
                        report_out = gr.Textbox(label="AI Institutional Report",
                                                lines=22, max_lines=45, interactive=False)

            # ══════════════════════════════════════════════════════════════════
            # TAB 2  Chart
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("📈 Chart"):
                chart_plot = gr.Plot(label="OHLCV + SMC Overlays", show_label=False)
                gr.Markdown("_Run analysis in **📥 Input & Analyze** to populate._")

                with gr.Accordion("🎯 Trade Setup Chart & Level Editor", open=True):
                    rr_chart_out = gr.HTML()
                    with gr.Row():
                        entry_inp = gr.Number(label="Entry",     precision=2, value=0.0)
                        sl_inp    = gr.Number(label="Stop Loss", precision=2, value=0.0)
                        tp1_inp   = gr.Number(label="TP 1",      precision=2, value=0.0)
                        tp2_inp   = gr.Number(label="TP 2",      precision=2, value=0.0)
                    refresh_setup_btn = gr.Button("↻  Rebuild Setup Chart", size="sm")
                    setup_chart_main  = gr.Plot(show_label=False)

            # ══════════════════════════════════════════════════════════════════
            # TAB 3  AI Report
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("🧠 AI Report"):
                with gr.Row():
                    gr.Markdown("### Confluence Breakdown & Institutional Report")
                    ai_refresh_btn = gr.Button("↻  Refresh", size="sm", scale=0)
                comp_df_out   = gr.Dataframe(label="Component Scores", interactive=False, wrap=True)
                reasons_out   = gr.HTML(label="Confluence Reasons")
                scenarios_out = gr.HTML(label="Dual Scenario Analysis")
                signal_out    = gr.HTML(label="Final Signal")

            # ══════════════════════════════════════════════════════════════════
            # TAB 4  Patterns & SMC
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("🔮 Patterns & SMC"):
                with gr.Row():
                    gr.Markdown("### Patterns, SMC & Trade Setup")
                    pat_refresh_btn = gr.Button("↻  Refresh", size="sm", scale=0)
                rr_pat_out    = gr.HTML()
                setup_chart_out = gr.Plot(label="Trade Setup Prediction", show_label=False)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Chart Patterns")
                        patterns_df_out = gr.Dataframe(interactive=False, wrap=True)
                        gr.Markdown("#### Candlestick Signals")
                        cs_df_out = gr.Dataframe(interactive=False, wrap=True)
                    with gr.Column():
                        gr.Markdown("#### SMC Overview")
                        smc_html_out = gr.HTML()
                        with gr.Accordion("Order Blocks", open=True):
                            ob_df_out = gr.Dataframe(interactive=False, wrap=True)
                        with gr.Accordion("Fair Value Gaps", open=True):
                            fvg_df_out = gr.Dataframe(interactive=False, wrap=True)

                with gr.Accordion("🔄 AMD Cycle (ICT)", open=True):
                    amd_html_out = gr.HTML()

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Fibonacci Levels")
                        fib_df_out = gr.Dataframe(interactive=False, wrap=True)
                    with gr.Column():
                        gr.Markdown("#### Multi-Timeframe Alignment")
                        mtf_df_out = gr.Dataframe(interactive=False, wrap=True)

            # ══════════════════════════════════════════════════════════════════
            # TAB 5  ML Prediction
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("🤖 ML Prediction"):
                with gr.Row():
                    gr.Markdown("### ML Setup Quality Prediction")
                    ml_refresh_btn = gr.Button("↻  Refresh", size="sm", scale=0)
                ml_html_out     = gr.HTML()
                model_stats_out = gr.Textbox(label="Model Stats (JSON)",
                                              lines=5, interactive=False)
                gr.Markdown("---")
                gr.Markdown("### Label Trade Outcome")
                gr.Markdown("Feed real outcomes back to improve future predictions.")
                with gr.Row():
                    label_run_id      = gr.Textbox(label="Run ID", scale=3)
                    label_outcome_inp = gr.Radio(["Win","Loss"], label="Outcome", value="Win", scale=1)
                    label_rr          = gr.Number(label="R:R Achieved (0=skip)",
                                                  value=0.0, precision=2, scale=1)
                label_btn    = gr.Button("Record Outcome", variant="primary")
                label_status = gr.Textbox(label="", interactive=False, lines=1)

            # ══════════════════════════════════════════════════════════════════
            # TAB 6  Backtesting
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("📉 Backtesting"):
                gr.Markdown("### Walk-Forward Backtesting")
                gr.Markdown("_Load OHLCV data first via the **📥 Input & Analyze** tab, then configure and run._")
                with gr.Row():
                    bt_capital  = gr.Number(label="Initial Capital ($)", value=10000.0, precision=2)
                    bt_risk     = gr.Slider(label="Risk / Trade (%)", minimum=0.5, maximum=10.0,
                                            value=1.0, step=0.5)
                    bt_min_conf = gr.Slider(label="Min Confluence Score", minimum=40, maximum=95,
                                            value=65, step=1)
                with gr.Row():
                    bt_max_bars = gr.Slider(label="Max bars", minimum=100, maximum=1500,
                                             value=800, step=100)
                    bt_stride   = gr.Slider(label="Re-eval every N bars", minimum=1, maximum=20,
                                             value=5, step=1)
                    bt_min_rr   = gr.Slider(label="Min R:R", minimum=1.0, maximum=4.0,
                                             value=1.5, step=0.5)
                bt_run_btn   = gr.Button("▶ Run Backtest", variant="primary")
                bt_results   = gr.HTML()
                bt_equity    = gr.Plot(show_label=False)
                bt_trades_df = gr.Dataframe(label="Trade Log (last 50)", interactive=False, wrap=True)

            # ══════════════════════════════════════════════════════════════════
            # TAB 7  History
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("🗂 History"):
                with gr.Row():
                    gr.Markdown("### Analysis History")
                    hist_refresh_btn = gr.Button("↻  Refresh", size="sm", scale=0)
                with gr.Row():
                    hist_asset  = gr.Textbox(label="Filter Asset", placeholder="XAUUSD", scale=2)
                    hist_tf     = gr.Textbox(label="Filter TF",    placeholder="4H",     scale=1)
                    hist_limit  = gr.Number( label="Limit",         value=50, precision=0, scale=1)
                hist_summary   = gr.HTML()
                history_table  = gr.Dataframe(interactive=False, wrap=True)
                with gr.Row():
                    hist_row_id   = gr.Number(label="Row ID to load report", value=0,
                                              precision=0, scale=1)
                    hist_view_btn = gr.Button("📄 Load Report", scale=1)
                hist_report_out = gr.Textbox(label="Full AI Report",
                                              lines=20, interactive=False)

            # ══════════════════════════════════════════════════════════════════
            # TAB 8  Live Bot
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("🟢 Live Bot"):
                with gr.Row():
                    gr.Markdown("### Live Bot — Prometheus Auto-Trader")
                    bot_refresh_btn = gr.Button("↻  Refresh", size="sm", scale=0)

                bot_status_out = gr.HTML("Click ↻ Refresh to load bot status.")

                with gr.Accordion("📂 Open Positions", open=True):
                    positions_df_out = gr.Dataframe(interactive=False, wrap=True)

                with gr.Accordion("🧠 Adaptive Learning State", open=True):
                    learning_html_out = gr.HTML("Click ↻ Refresh to load learning state.")

                with gr.Accordion("🗒 Recent Trades (last 50)", open=False):
                    trades_df_out = gr.Dataframe(interactive=False, wrap=True)

        # ──────────────────────────────────────────────────────────────────────
        # Event bindings
        # ──────────────────────────────────────────────────────────────────────

        _inputs = [file_input, tf1_file, tf2_file, tf3_file,
                   tf1_label, tf2_label, tf3_label,
                   asset_input, tf_input, json_input]

        _outputs = [
            # Analyze tab
            score_out, grade_out, dir_out, report_out, summary_html_out,
            # Chart tab
            chart_plot, entry_inp, sl_inp, tp1_inp, tp2_inp, rr_chart_out,
            # AI Report tab
            comp_df_out, reasons_out, scenarios_out, signal_out,
            # Patterns tab
            setup_chart_out, patterns_df_out, cs_df_out, smc_html_out,
            ob_df_out, fvg_df_out, amd_html_out, fib_df_out, mtf_df_out,
            # ML tab
            ml_html_out, model_stats_out,
            # State
            result_state, df_state,
        ]

        run_btn.click(fn=run_analysis, inputs=_inputs, outputs=_outputs)

        # After analysis, also wire up Chart tab's setup chart
        run_btn.click(
            fn=lambda rs, ds, e, s, t1, t2: (
                _build_setup_chart(ds, rs, float(e), float(s), float(t1), float(t2))
                if ds is not None else None,
                _rr_metrics_html(e, s, t1, t2, getattr(getattr(rs, "confluence", None), "direction", "sideways"))
                if rs is not None else "",
            ),
            inputs=[result_state, df_state, entry_inp, sl_inp, tp1_inp, tp2_inp],
            outputs=[setup_chart_main, rr_chart_out],
        )

        # Chart tab — manual level refresh
        refresh_setup_btn.click(
            fn=refresh_setup_chart_fn,
            inputs=[result_state, df_state, entry_inp, sl_inp, tp1_inp, tp2_inp],
            outputs=[setup_chart_main, rr_chart_out],
        )

        # AI Report tab — manual refresh
        ai_refresh_btn.click(
            fn=lambda r: _extract_ai_report(r) if r is not None else (
                pd.DataFrame(), "—", "—", "—"),
            inputs=[result_state],
            outputs=[comp_df_out, reasons_out, scenarios_out, signal_out],
        )

        # Patterns tab — manual refresh
        def _pat_refresh(rs, ds, e, s, t1, t2):
            if rs is None:
                e_df = pd.DataFrame()
                return (None, e_df, e_df, "—", e_df, e_df, "—", e_df, e_df, "", "")
            pats, cs, smc_h, obs, fvgs, amd_h, fibs, mtf = _extract_patterns_smc(rs)
            chart = _build_setup_chart(ds, rs, float(e), float(s), float(t1), float(t2)) if ds is not None else None
            rr_h  = _rr_metrics_html(e, s, t1, t2,
                                      getattr(getattr(rs, "confluence", None), "direction", "sideways"))
            return chart, pats, cs, smc_h, obs, fvgs, amd_h, fibs, mtf, rr_h, rr_h

        pat_refresh_btn.click(
            fn=_pat_refresh,
            inputs=[result_state, df_state, entry_inp, sl_inp, tp1_inp, tp2_inp],
            outputs=[setup_chart_out, patterns_df_out, cs_df_out, smc_html_out,
                     ob_df_out, fvg_df_out, amd_html_out, fib_df_out, mtf_df_out,
                     rr_pat_out, rr_chart_out],
        )

        # ML tab — manual refresh
        ml_refresh_btn.click(
            fn=lambda r: _extract_ml(r) if r is not None else ("—", "—"),
            inputs=[result_state],
            outputs=[ml_html_out, model_stats_out],
        )

        # Label outcome
        label_btn.click(
            fn=label_outcome_handler,
            inputs=[label_run_id, label_outcome_inp, label_rr],
            outputs=[label_status],
        )

        # Backtesting
        bt_run_btn.click(
            fn=run_backtest_handler,
            inputs=[df_state, asset_input, tf_input,
                    bt_capital, bt_risk, bt_min_conf, bt_max_bars, bt_stride, bt_min_rr],
            outputs=[bt_results, bt_equity, bt_trades_df],
        )

        # History
        def _load_hist(asset, tf, limit):
            df = _load_history(asset, tf, int(limit or 50))
            return _history_summary_html(df), df

        hist_refresh_btn.click(
            fn=_load_hist,
            inputs=[hist_asset, hist_tf, hist_limit],
            outputs=[hist_summary, history_table],
        )

        def _load_report(row_id):
            try:
                return _load_history_report(int(row_id)) if row_id else ""
            except Exception as e:
                return f"Error: {e}"

        def _load_history_report(row_id) -> str:
            try:
                from storage.database import get_analysis
                full = get_analysis(int(row_id))
                if full and full.get("report_json"):
                    rdata = json.loads(full["report_json"])
                    return rdata.get("full_text", "No report text stored.")
                return "No detailed report stored for this record."
            except Exception as e:
                return f"Error: {e}"

        hist_view_btn.click(
            fn=_load_report,
            inputs=[hist_row_id],
            outputs=[hist_report_out],
        )

        # Live Bot
        bot_refresh_btn.click(
            fn=_load_bot_status,
            inputs=[],
            outputs=[bot_status_out, positions_df_out, learning_html_out, trades_df_out],
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, threading, time

    parser = argparse.ArgumentParser(description="Prometheus Gradio Dashboard")
    parser.add_argument("--port",  type=int, default=7860)
    parser.add_argument("--host",  type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print(f"\n  Prometheus Gradio Dashboard")
    print(f"  http://localhost:{args.port}\n")

    demo = build_ui()

    def _prewarm():
        time.sleep(3)
        try:
            _get_engine()
            print("  Engine pre-warm complete.")
        except Exception as exc:
            print(f"  Engine pre-warm failed (non-fatal): {exc}")

    threading.Thread(target=_prewarm, daemon=True).start()

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        max_threads=4,
        max_file_size="200mb",
        theme=_THEME,
        css=DARK_CSS,
    )
