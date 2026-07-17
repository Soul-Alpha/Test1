"""
Prometheus Streamlit Dashboard
================================
Institutional-grade market analysis UI.

Run:
    streamlit run ui/dashboard.py

Features:
- Upload chart images or paste OHLCV CSV/JSON
- Full engine pipeline with live progress
- Interactive Plotly chart with all overlays
- AI narrative report viewer
- ML prediction sidebar
- Backtesting tab with equity curve
- History browser
- Dark UI theme with professional styling
"""

from __future__ import annotations

import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# Add project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import CONFIG, OUTPUTS_DIR
from ui.dashboard_registry_support import render_registry_metrics, render_registry_tables, render_registry_texts

_PROMETHEUS_EVOLUTION_F = ROOT / "storage" / "olympus" / "prometheus_evolution_intelligence.json"
_ZEUS_VALIDATION_F = ROOT / "storage" / "olympus" / "zeus_validation_status.json"
_INSTITUTIONAL_RUNTIME_F = ROOT / "storage" / "olympus" / "institutional_risk_performance_runtime.json"


def _load_json_silent(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


_EVO_PAYLOAD = _load_json_silent(_PROMETHEUS_EVOLUTION_F)
_ZEUS_VALIDATION = _load_json_silent(_ZEUS_VALIDATION_F)
_INSTITUTIONAL_RUNTIME = _load_json_silent(_INSTITUTIONAL_RUNTIME_F)
_EVO_META = _EVO_PAYLOAD.get("meta", {}) if isinstance(_EVO_PAYLOAD, dict) else {}
_PROMETHEUS_VERSION = _EVO_META.get("evolution_layer_version", "v2.2")

# ── Page config (MUST be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Prometheus · Market Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Dark professional palette */
    :root {
        --bg: #0e1117;
        --surface: #1c2232;
        --accent: #00c4ff;
        --gold: #f1c40f;
        --success: #2ecc71;
        --danger: #e74c3c;
        --text: #dde1ea;
        --muted: #7f8c8d;
    }
    body { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; }
    .metric-card {
        background: var(--surface); border-radius: 10px;
        padding: 16px 20px; margin-bottom: 12px;
        border-left: 4px solid var(--accent);
    }
    .bull-card  { border-left-color: var(--success); }
    .bear-card  { border-left-color: var(--danger);  }
    .grade-A    { color: #2ecc71; font-weight: 700; font-size: 2rem; }
    .grade-B    { color: #f39c12; font-weight: 700; font-size: 2rem; }
    .grade-C    { color: #e67e22; font-weight: 700; font-size: 2rem; }
    .grade-D    { color: #e74c3c; font-weight: 700; font-size: 2rem; }
    .grade-F    { color: #7f8c8d; font-weight: 700; font-size: 2rem; }
    .report-box {
        background: var(--surface); border-radius: 8px;
        padding: 20px; line-height: 1.8;
        white-space: pre-wrap; font-family: monospace; font-size: 0.85rem;
        max-height: 60vh; overflow-y: auto;
    }
    div[data-testid="stSidebar"] { background: #13192b; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session state defaults ──────────────────────────────────────────────────
# ── Chart image annotation helper ─────────────────────────────────────────
def _annotate_chart_image(image_bytes: bytes, direction: str, asset: str = "", timeframe: str = "") -> bytes:
    """Overlay trend-direction arrows on a chart screenshot using PIL."""
    try:
        from PIL import Image, ImageDraw
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size

        # Colour scheme per direction
        if direction in ("bullish", "long"):
            clr = (39, 174, 96)       # green
            label = "▲  BULLISH TREND"
            arrow_dir = "up"
        elif direction in ("bearish", "short"):
            clr = (192, 57, 43)       # red
            label = "▼  BEARISH TREND"
            arrow_dir = "down"
        else:
            clr = (211, 84, 0)        # orange
            label = "◀▶  RANGING / NEUTRAL"
            arrow_dir = "range"

        # ── Semi-transparent banner (top-right) ───────────────────────────
        banner_w, banner_h, pad = min(w, 320), 56, 12
        x0, y0 = w - banner_w - pad, pad
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ov = ImageDraw.Draw(overlay)
        ov.rounded_rectangle([x0, y0, x0 + banner_w, y0 + banner_h],
                             radius=10, fill=(0, 0, 0, 170))
        ov.rounded_rectangle([x0, y0, x0 + banner_w, y0 + banner_h],
                             radius=10, outline=clr + (220,), width=3)
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Banner label text
        try:
            from PIL import ImageFont
            fnt = ImageFont.truetype("arial.ttf", 20)
        except Exception:
            fnt = None
        if fnt:
            bbox = draw.textbbox((0, 0), label, font=fnt)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((x0 + (banner_w - tw) // 2, y0 + (banner_h - th) // 2),
                      label, fill=clr, font=fnt)
        else:
            draw.text((x0 + 10, y0 + 16), label, fill=clr)

        # ── Large directional arrow (right side, vertically centred) ──────
        ax = w - w // 8          # x-centre of arrow column
        mid_y = h // 2
        aw, ah = max(30, w // 18), max(60, h // 5)   # arrow width / height
        shaft_w = max(8, aw // 4)

        if arrow_dir == "up":
            tip_y  = mid_y - ah
            base_y = mid_y + ah // 4
            # shaft
            draw.rectangle([ax - shaft_w, base_y, ax + shaft_w, mid_y], fill=clr)
            # head
            draw.polygon([(ax, tip_y), (ax - aw, mid_y), (ax + aw, mid_y)], fill=clr)
        elif arrow_dir == "down":
            tip_y  = mid_y + ah
            base_y = mid_y - ah // 4
            draw.rectangle([ax - shaft_w, mid_y, ax + shaft_w, base_y], fill=clr)
            draw.polygon([(ax, tip_y), (ax - aw, mid_y), (ax + aw, mid_y)], fill=clr)
        else:  # ranging — bidirectional horizontal
            left_x  = ax - aw
            right_x = ax + aw
            draw.rectangle([left_x + aw // 2, mid_y - shaft_w,
                            right_x - aw // 2, mid_y + shaft_w], fill=clr)
            draw.polygon([(left_x, mid_y), (left_x + aw // 2, mid_y - aw // 2),
                          (left_x + aw // 2, mid_y + aw // 2)], fill=clr)
            draw.polygon([(right_x, mid_y), (right_x - aw // 2, mid_y - aw // 2),
                          (right_x - aw // 2, mid_y + aw // 2)], fill=clr)

        out = io.BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()
    except Exception:
        return image_bytes  # fall back to original if PIL unavailable


def _init_state():
    defaults = {
        "result":            None,
        "bt_result":         None,
        "history":           [],
        "engine_ready":      False,
        "engine":            None,
        "last_df":           None,
        "last_image_bytes":  None,
        "tf_data":           {},       # Dict[str, pd.DataFrame] — one entry per loaded TF
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

_init_state()


# ── Robust CSV reader (handles UTF-16/UTF-8/Latin-1 & tab/semi/comma) ─────
def _read_csv_smart(file_obj) -> pd.DataFrame:
    """
    Handles Exness / MT4 / MT5 CSV exports in any encoding and layout:
      - UTF-16 LE BOM  (Exness standard)
      - With or without a header row
      - <DATE> <TIME> split columns OR combined datetime column
      - Date format YYYY.MM.DD HH:MM  or  YYYY-MM-DD HH:MM
      - Comma / tab / semicolon delimited
    """
    import io as _io

    raw = file_obj.read()
    encodings = ["utf-16", "utf-16-le", "utf-16-be", "utf-8-sig", "utf-8", "latin-1", "cp1252"]
    delimiters = [",", "\t", ";"]
    last_err = None

    # Column name sets that indicate a real header vs data row
    known_header_words = {"open", "high", "low", "close", "date", "time",
                          "datetime", "vol", "volume", "tickvol", "spread"}
    # Fallback column names for headerless 7-col Exness files
    EXNESS_7  = ["datetime", "open", "high", "low", "close", "volume", "spread"]
    EXNESS_6  = ["datetime", "open", "high", "low", "close", "volume"]
    MT4_8     = ["date", "time", "open", "high", "low", "close", "volume", "spread"]
    MT4_7     = ["date", "time", "open", "high", "low", "close", "volume"]

    def _try_parse(text: str, delim: str):
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

        # ── Detect whether first row is a header ──────────────────────────
        first_line = text.splitlines()[0] if text else ""
        first_cells = [c.strip().lower().replace("<", "").replace(">", "")
                       for c in first_line.split(delim)]
        has_header = any(w in known_header_words for w in first_cells)

        if has_header:
            df = pd.read_csv(_io.StringIO(text), sep=delim, engine="python")
            df.columns = [c.strip().lower().replace("<", "").replace(">", "")
                          for c in df.columns]
        else:
            # Headerless — infer columns from count
            ncols = len(first_cells)
            if ncols == 7:
                names = EXNESS_7
            elif ncols == 6:
                names = EXNESS_6
            elif ncols == 8:
                names = MT4_8
            elif ncols == 9:   # date, time, open, high, low, close, tickvol, vol, spread
                names = ["date", "time", "open", "high", "low", "close",
                         "tickvol", "volume", "spread"]
            else:
                names = None   # let pandas number them
            df = pd.read_csv(_io.StringIO(text), sep=delim, engine="python",
                             header=None, names=names)

        if df.shape[1] < 4:
            raise ValueError("Too few columns")

        # ── Merge date + time if split ────────────────────────────────────
        if "date" in df.columns and "time" in df.columns:
            df["datetime"] = pd.to_datetime(
                df["date"].astype(str).str.replace(".", "-", regex=False)
                + " " + df["time"].astype(str),
                errors="coerce",
            )
            df = df.drop(columns=["date", "time"]).set_index("datetime")
        elif "datetime" in df.columns:
            # Handle YYYY.MM.DD HH:MM format from Exness
            dt_str = df["datetime"].astype(str).str.replace(".", "-", regex=False)
            df["datetime"] = pd.to_datetime(dt_str, errors="coerce")
            df = df.set_index("datetime")
        else:
            # Use first column as index
            df = df.set_index(df.columns[0])
            idx_str = df.index.astype(str).str.replace(".", "-", regex=False)
            df.index = pd.to_datetime(idx_str, errors="coerce")

        df.index.name = "datetime"
        df = df[~df.index.isna()]

        # ── Alias / drop extra columns ────────────────────────────────────
        for old, new in [("vol", "volume"), ("tickvol", "volume"),
                         ("tick_volume", "volume")]:
            if old in df.columns and "volume" not in df.columns:
                df = df.rename(columns={old: new})
            elif old in df.columns:
                df = df.drop(columns=[old], errors="ignore")
        df = df.drop(columns=["spread"], errors="ignore")

        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if "volume" not in df.columns:
            df["volume"] = 0.0

        df = df.dropna(subset=["open", "high", "low", "close"])
        if len(df) < 2:
            raise ValueError("Fewer than 2 valid rows after parsing")
        return df.sort_index()

    for enc in encodings:
        for delim in delimiters:
            try:
                text = raw.decode(enc)
                return _try_parse(text, delim)
            except Exception as e:
                last_err = e
                continue

    raise ValueError(
        f"Could not parse CSV.\nLast error: {last_err}\n\n"
        f"Expected format: YYYY.MM.DD HH:MM,open,high,low,close,vol,spread\n"
        f"(Exness MT5 export — no header row, UTF-16, comma-delimited)"
    )


# ── Engine loader (cached across reruns) ───────────────────────────────────
@st.cache_resource(show_spinner="Loading Prometheus engines…")
def _load_engine():
    from prometheus_core import Prometheus
    return Prometheus()


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        f"https://img.shields.io/badge/Prometheus-{_PROMETHEUS_VERSION}-00c4ff?style=for-the-badge",
        use_column_width=True,
    )
    st.markdown("### ⚙️ Analysis Settings")

    asset     = st.text_input("Asset Symbol", value="XAUUSD", max_chars=20).upper()
    timeframe = st.selectbox("Primary Timeframe", ["1m","5m","15m","30m","1h","4h","1d","1w"], index=5)

    st.markdown("---")
    st.markdown("### 📥 Data Input")
    input_mode = st.radio("Input Mode", ["🟢 Live from MT5", "Upload CSV", "Upload Chart Image", "Paste JSON"])

    render_chart  = st.checkbox("Generate Interactive Chart", value=True)
    run_backtest  = st.checkbox("Run Backtest on Data",       value=False)

    st.markdown("---")
    st.markdown("### 🎛️ Engine Thresholds")
    min_confluence = st.slider("Min Confluence Score", 0, 100, 50)
    pivot_sens     = st.slider("Pivot Sensitivity (bars)", 2, 10, 5)

    st.markdown("---")
    st.markdown("### 🔄 Live Bot Refresh")
    _refresh_label = st.select_slider(
        "Auto-refresh interval",
        options=["Off", "10s", "30s", "60s", "2m"],
        value="30s",
    )
    _refresh_map   = {"Off": 0, "10s": 10, "30s": 30, "60s": 60, "2m": 120}
    _live_refresh_s = _refresh_map[_refresh_label]

    st.markdown("---")
    st.caption("Prometheus © 2025 · Institutional AI Analysis")


# ── Header ─────────────────────────────────────────────────────────────────
col_title, col_status = st.columns([4, 1])
with col_title:
    st.title("📊 Prometheus Market Analysis Bot")
    st.caption(f"Version **{_PROMETHEUS_VERSION}** · Analyzing **{asset}** on **{timeframe}** timeframe")
with col_status:
    if st.session_state.result:
        r = st.session_state.result
        score = r.confluence.total if r.confluence else 0
        st.metric("Confluence", f"{score:.0f}/100")


# ── Main tabs ──────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📥 Input & Analyze",
    "📈 Chart",
    "🧠 AI Report",
    "🔮 Patterns & SMC",
    "🤖 ML Prediction",
    "📉 Backtesting",
    "🗂️ History",
    "🟢 Live Bot",
])


# ==========================================================================
# TAB 0 – Input & Analyze
# ==========================================================================
with tabs[0]:
    st.subheader("Load Market Data")

    df_input: Optional[pd.DataFrame]         = None
    image_bytes: Optional[bytes]             = None
    _tf_data_raw: Dict[str, pd.DataFrame]   = {}   # keyed by lowercase tf label

    # ── Sync from Live Bot ────────────────────────────────────────────────────
    # One-click shortcut: reads asset/tf/symbol from bot_status.json and
    # fetches + analyses automatically — no manual config needed.
    _sync_status_f = Path(__file__).parent.parent / "live_bot" / "bot_status.json"
    _sync_asset = "XAUUSDm"
    _sync_tf    = "4h"
    _sync_bars  = 500
    _bot_online = False
    _sync_age_str = "unknown"

    if _sync_status_f.exists():
        try:
            _sync_s   = json.loads(_sync_status_f.read_text(encoding="utf-8"))
            _sync_asset = _sync_s.get("asset", "XAUUSDm")
            _sync_tf    = _sync_s.get("timeframe", "4H").lower()
            _sync_bars  = _sync_s.get("n_candles", 500)
            _sync_lp    = _sync_s.get("last_poll", "")
            if _sync_lp:
                _sync_age_sec = (datetime.utcnow() - datetime.fromisoformat(_sync_lp)).total_seconds()
                _sync_age_str = f"{int(_sync_age_sec)}s ago"
                _bot_online   = _sync_age_sec < _sync_s.get("poll_interval", 60) * 3
        except Exception:
            pass

    _sb_col1, _sb_col2 = st.columns([3, 1])
    with _sb_col1:
        _sync_btn = st.button(
            f"🤖 Sync from Bot  —  **{_sync_asset}** {_sync_tf.upper()} "
            f"| {_sync_bars} bars | last poll {_sync_age_str}",
            type="primary",
            use_container_width=True,
            help="Fetches the exact asset & timeframe the bot uses and runs the full analysis pipeline automatically.",
        )
    with _sb_col2:
        st.markdown(
            f"<div style='padding:8px 0;color:#{'22c55e' if _bot_online else 'f59e0b'};font-size:.85rem'>"
            f"{'🟢 Bot online' if _bot_online else '🟡 Bot offline — last settings used'}</div>",
            unsafe_allow_html=True,
        )

    if _sync_btn:
        _MT5_TF_MAP_SYNC = {
            "1m": "TIMEFRAME_M1", "5m": "TIMEFRAME_M5",
            "15m": "TIMEFRAME_M15", "30m": "TIMEFRAME_M30",
            "1h": "TIMEFRAME_H1", "4h": "TIMEFRAME_H4",
            "1d": "TIMEFRAME_D1", "1w": "TIMEFRAME_W1",
        }
        _ALL_TFS_SYNC = [
            ("1m","TIMEFRAME_M1",300), ("5m","TIMEFRAME_M5",300),
            ("15m","TIMEFRAME_M15",300), ("30m","TIMEFRAME_M30",300),
            ("1h","TIMEFRAME_H1",300), ("4h","TIMEFRAME_H4",300),
            ("1d","TIMEFRAME_D1",200),
        ]
        _sync_ok = False
        _sync_err_msg = ""
        _sync_tf_failed: list = []
        with st.spinner(f"Connecting to MT5 and fetching {_sync_asset} {_sync_tf.upper()}…"):
            try:
                import MetaTrader5 as _mt5_sync
                if not _mt5_sync.initialize():
                    _sync_err_msg = f"MT5 init failed: {_mt5_sync.last_error()}"
                else:
                    _tf_const_sync = getattr(_mt5_sync, _MT5_TF_MAP_SYNC.get(_sync_tf, "TIMEFRAME_H4"))
                    _rates_sync    = _mt5_sync.copy_rates_from_pos(_sync_asset, _tf_const_sync, 0, int(_sync_bars))
                    if _rates_sync is None or len(_rates_sync) == 0:
                        _sync_err_msg = (f"No data returned for {_sync_asset}. "
                                         f"Make sure the MT5 terminal is open and logged in.")
                    else:
                        _df_sync = pd.DataFrame(_rates_sync)
                        _df_sync["time"] = pd.to_datetime(_df_sync["time"], unit="s")
                        _df_sync = _df_sync.set_index("time")
                        _df_sync.index.name = "datetime"
                        _df_sync.rename(columns={"tick_volume": "volume"}, inplace=True)
                        _df_sync = _df_sync[["open","high","low","close","volume"]].dropna()

                        # Fetch all MTF timeframes
                        _sync_extra: dict = {}
                        _sync_tf_failed: list = []
                        for _etf, _eattr, _ebars in _ALL_TFS_SYNC:
                            if _etf == _sync_tf:
                                continue
                            try:
                                _er = _mt5_sync.copy_rates_from_pos(
                                    _sync_asset, getattr(_mt5_sync, _eattr), 0, _ebars)
                                if _er is not None and len(_er) >= 20:
                                    _edf2 = pd.DataFrame(_er)
                                    _edf2["time"] = pd.to_datetime(_edf2["time"], unit="s")
                                    _edf2 = _edf2.set_index("time")
                                    _edf2.index.name = "datetime"
                                    _edf2.rename(columns={"tick_volume": "volume"}, inplace=True)
                                    _sync_extra[_etf] = _edf2[["open","high","low","close","volume"]].dropna()
                                else:
                                    _sync_tf_failed.append(_etf)
                            except Exception as _etf_err:
                                _sync_tf_failed.append(f"{_etf}({_etf_err})")

                        # Build MTF payload — always pass a dict so MTF engine activates
                        # (engine needs len > 1; we have primary + extras)
                        _sync_tf_payload = {_sync_tf: _df_sync}
                        _sync_tf_payload.update(_sync_extra)
                        if len(_sync_tf_payload) <= 1:
                            _sync_tf_payload = None  # single-TF fallback

                        # Run full analysis
                        _sync_engine = _load_engine()
                        _sync_engine.ms_engine.sensitivity = pivot_sens
                        _sync_result = _sync_engine.analyze_data(
                            df=_df_sync,
                            asset=_sync_asset,
                            timeframe=_sync_tf.upper(),
                            tf_data=_sync_tf_payload,
                            render_chart=render_chart,
                            save_to_db=False,
                        )
                        st.session_state.result           = _sync_result
                        st.session_state.last_df          = _df_sync
                        st.session_state.last_image_bytes = getattr(_sync_result, "chart_bytes", None)
                        st.session_state.tf_data          = _sync_tf_payload or {_sync_tf: _df_sync}
                        _sync_ok = True
            except Exception as _exc:
                _sync_err_msg = str(_exc)

        if _sync_ok:
            _sc = st.session_state.result.confluence if st.session_state.result else None
            _loaded_tfs = sorted(st.session_state.tf_data.keys()) if st.session_state.tf_data else []
            st.success(
                f"✅ **{_sync_asset} {_sync_tf.upper()}** — {len(st.session_state.last_df):,} bars | "
                + (f"Grade **{_sc.grade}** · Score **{_sc.total:.0f}** · **{(_sc.direction or '?').upper()}**"
                   if _sc else "Analysis complete")
                + f" | MTF: {', '.join(t.upper() for t in _loaded_tfs)}"
            )
            if _sync_tf_failed:
                st.caption(f"TFs not fetched: {', '.join(_sync_tf_failed)}")
            st.info("Switch to the **Chart**, **AI Report** or **Patterns & SMC** tabs to explore the results.")
        elif _sync_err_msg:
            st.error(f"Sync failed: {_sync_err_msg}")

    st.markdown("---")

    # ------- Live from MT5 ---------------------------------------------------
    if input_mode == "🟢 Live from MT5":
        _MT5_TF_MAP = {
            "1m": "TIMEFRAME_M1",  "5m": "TIMEFRAME_M5",
            "15m": "TIMEFRAME_M15", "30m": "TIMEFRAME_M30",
            "1h": "TIMEFRAME_H1",  "4h": "TIMEFRAME_H4",
            "1d": "TIMEFRAME_D1",  "1w": "TIMEFRAME_W1",
        }
        _mt5_n_bars = st.number_input("Bars to fetch", min_value=50, max_value=2000, value=500, step=50)
        _mt5_fetch  = st.button("📡 Fetch from MT5", type="primary")

        if _mt5_fetch or st.session_state.get("_mt5_df_cached") is not None:
            if _mt5_fetch:
                try:
                    import MetaTrader5 as _mt5
                    if not _mt5.initialize():
                        st.error(f"MT5 init failed: {_mt5.last_error()}. Make sure the MT5 terminal is open and logged in.")
                    else:
                        _tf_attr = _MT5_TF_MAP.get(timeframe.lower(), "TIMEFRAME_H4")
                        _tf_const = getattr(_mt5, _tf_attr)
                        # Auto-resolve broker suffix (e.g. XAUUSDm)
                        _sym = asset
                        _all_syms = [s.name for s in (_mt5.symbols_get(f"*{asset}*") or [])]
                        if _all_syms and asset not in _all_syms:
                            _sym = _all_syms[0]
                        _rates = _mt5.copy_rates_from_pos(_sym, _tf_const, 0, int(_mt5_n_bars))
                        # Fetch all other TFs in the same session so MTF table shows real data
                        _ALL_DASHBOARD_TFS = [
                            ("1m",  "TIMEFRAME_M1",  300),
                            ("5m",  "TIMEFRAME_M5",  300),
                            ("15m", "TIMEFRAME_M15", 300),
                            ("30m", "TIMEFRAME_M30", 300),
                            ("1h",  "TIMEFRAME_H1",  300),
                            ("4h",  "TIMEFRAME_H4",  300),
                            ("1d",  "TIMEFRAME_D1",  200),
                        ]
                        _extra_tf_cache: dict = {}
                        for _etf_str, _etf_attr, _etf_bars in _ALL_DASHBOARD_TFS:
                            if _etf_str == timeframe.lower():
                                continue
                            try:
                                _etf_const = getattr(_mt5, _etf_attr)
                                _erates = _mt5.copy_rates_from_pos(_sym, _etf_const, 0, _etf_bars)
                                if _erates is not None and len(_erates) >= 20:
                                    _edf = pd.DataFrame(_erates)
                                    _edf["time"] = pd.to_datetime(_edf["time"], unit="s")
                                    _edf = _edf.set_index("time")
                                    _edf.rename(columns={"tick_volume": "volume"}, inplace=True)
                                    _edf.columns = [c.lower() for c in _edf.columns]
                                    for _c in ("open", "high", "low", "close", "volume"):
                                        if _c not in _edf.columns:
                                            _edf[_c] = 0.0
                                    _extra_tf_cache[_etf_str] = _edf
                            except Exception:
                                pass
                        _mt5.shutdown()
                        if _rates is None or len(_rates) == 0:
                            st.error(f"No data returned for {_sym} {timeframe}. Check the symbol name and timeframe.")
                        else:
                            _mdf = pd.DataFrame(_rates)
                            _mdf["time"] = pd.to_datetime(_mdf["time"], unit="s")
                            _mdf = _mdf.set_index("time")
                            _mdf.rename(columns={"tick_volume": "volume"}, inplace=True)
                            _mdf.columns = [c.lower() for c in _mdf.columns]
                            for _c in ("open", "high", "low", "close", "volume"):
                                if _c not in _mdf.columns:
                                    _mdf[_c] = 0.0
                            st.session_state["_mt5_df_cached"] = _mdf
                            st.session_state["_mt5_extra_tf_cached"] = _extra_tf_cache
                            _n_extra = len(_extra_tf_cache)
                            st.success(f"✅ Fetched {len(_mdf):,} bars of {_sym} {timeframe.upper()} from MT5 + {_n_extra} other TFs ({_mdf.index[0]} → {_mdf.index[-1]})")
                except ImportError:
                    st.error("MetaTrader5 package not installed in this environment.")
                except Exception as _me:
                    st.error(f"MT5 fetch error: {_me}")

            _cached_mdf = st.session_state.get("_mt5_df_cached")
            if _cached_mdf is not None:
                df_input = _cached_mdf
                _tf_data_raw[timeframe.lower()] = df_input
                # Merge any other TFs fetched in the same session
                _cached_extra = st.session_state.get("_mt5_extra_tf_cached", {})
                _tf_data_raw.update(_cached_extra)
                st.dataframe(df_input.tail(3), use_container_width=True)
                if _mt5_fetch:
                    pass  # message already shown above
                else:
                    st.info(f"Using cached MT5 data: {len(df_input):,} bars. Click **Fetch from MT5** to refresh.")

    # ------- Upload CSV -------------------------------------------------------
    elif input_mode == "Upload CSV":
        with st.expander("ℹ️ How to get OHLCV data from MetaTrader 5", expanded=False):
            st.markdown("""
**Option A — Python (recommended)**
```python
import MetaTrader5 as mt5, pandas as pd
mt5.initialize()
rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_H4, 0, 500)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df = df.set_index('time')
df.rename(columns={'tick_volume':'volume'}, inplace=True)
df.to_csv("XAUUSD_4H.csv")
```

**Option B — Inside MT5 terminal**
1. Open the XAUUSD chart on **4H** timeframe
2. Press **F2** → History Center → select XAUUSD / H4 → click **Export**
3. Save the `.csv` file and upload it here

**Required columns:** `open, high, low, close, volume`  
The index column must be a date/time.
""")

        # ── Primary timeframe CSV ──────────────────────────────────────────────
        st.markdown(f"##### Primary Timeframe — `{timeframe}`")
        uploaded = st.file_uploader(
            f"Upload OHLCV CSV for {timeframe} (columns: open, high, low, close, volume)",
            type=["csv"],
        )
        if uploaded:
            try:
                df_input = _read_csv_smart(uploaded)
                df_input.columns = [c.lower() for c in df_input.columns]
                for col in ("open", "high", "low", "close", "volume"):
                    if col not in df_input.columns:
                        df_input[col] = 0.0
                note = " (last 1,500 bars used by engines)" if len(df_input) > 1500 else ""
                st.success(f"✅ **[{timeframe}]** {len(df_input):,} bars  ({df_input.index[0]} → {df_input.index[-1]}){note}")
                st.dataframe(df_input.tail(3), use_container_width=True)
                _tf_data_raw[timeframe.lower()] = df_input
            except Exception as e:
                st.error(f"CSV parse error: {e}")

        # ── Additional timeframes for MTF confirmation ─────────────────────────
        _ALL_TF_OPTIONS = ["1D", "4H", "1H", "30M", "15M", "5M", "1M"]
        _extra_tf_defaults = [tf for tf in _ALL_TF_OPTIONS if tf.lower() != timeframe.lower()]

        with st.expander("➕ Add more timeframes for MTF confirmation (optional)", expanded=False):
            st.caption(
                "Upload CSVs for up to 3 additional timeframes. "
                "Each extra dataset improves the multi-timeframe confluence score and the MTF trend table."
            )
            _extra_cols = st.columns(3)
            for _slot in range(3):
                with _extra_cols[_slot]:
                    _def_tf = _extra_tf_defaults[_slot] if _slot < len(_extra_tf_defaults) else _extra_tf_defaults[0]
                    _extra_tf_label = st.selectbox(
                        f"TF {_slot + 2} label",
                        _ALL_TF_OPTIONS,
                        index=_ALL_TF_OPTIONS.index(_def_tf) if _def_tf in _ALL_TF_OPTIONS else 0,
                        key=f"extra_tf_label_{_slot}",
                    )
                    _extra_uploaded = st.file_uploader(
                        f"CSV for {_extra_tf_label}",
                        type=["csv"],
                        key=f"extra_tf_csv_{_slot}",
                    )
                    if _extra_uploaded:
                        try:
                            _edf = _read_csv_smart(_extra_uploaded)
                            _edf.columns = [c.lower() for c in _edf.columns]
                            for _col in ("open", "high", "low", "close", "volume"):
                                if _col not in _edf.columns:
                                    _edf[_col] = 0.0
                            _tf_data_raw[_extra_tf_label.lower()] = _edf
                            st.success(f"✅ [{_extra_tf_label}] {len(_edf):,} bars")
                        except Exception as _ex:
                            st.error(f"[{_extra_tf_label}] parse error: {_ex}")

        # Status line when multiple TFs are loaded
        if len(_tf_data_raw) > 1:
            _loaded_labels = "  ·  ".join(f"`{k.upper()}`" for k in sorted(_tf_data_raw.keys()))
            st.info(f"🔀 **Multi-timeframe mode active** — {_loaded_labels}  ({len(_tf_data_raw)} timeframes loaded)")

    # ------- Upload Image -------------------------------------------------------
    elif input_mode == "Upload Chart Image":
        uploaded_img = st.file_uploader("Upload chart screenshot (PNG/JPG)", type=["png","jpg","jpeg"])
        if uploaded_img:
            image_bytes = uploaded_img.read()
            st.image(image_bytes, caption="Uploaded chart", use_column_width=True)

        # Optional: also provide CSV
        st.info(
            "📊 **For full analysis** (Fibonacci, Chart Patterns, SMC, Entry/SL/TP levels) "
            "also upload the matching OHLCV CSV.  "
            "Export it from MT5: *F2 → History Center → XAUUSD → H4 → Export*"
        )
        uploaded_csv = st.file_uploader(
            "Upload OHLCV CSV alongside image (enables all 7 engines)",
            type=["csv"], key="csv_alongside"
        )
        if uploaded_csv:
            try:
                df_input = _read_csv_smart(uploaded_csv)
                df_input.columns = [c.lower() for c in df_input.columns]
                note = " (last 1 500 bars used)" if len(df_input) > 1500 else ""
                st.success(f"CSV loaded: {len(df_input):,} bars{note}.")
            except Exception as e:
                st.warning(f"Could not parse CSV: {e}")

    # ------- Paste JSON -------------------------------------------------------
    else:  # Paste JSON
        sample_json = json.dumps([
            {"timestamp": "2024-01-01T00:00:00", "open": 2000, "high": 2010, "low": 1995, "close": 2005, "volume": 1500},
            {"timestamp": "2024-01-01T04:00:00", "open": 2005, "high": 2020, "low": 2001, "close": 2015, "volume": 2100},
        ], indent=2)
        json_text = st.text_area("Paste OHLCV JSON array", value=sample_json, height=200)
        if json_text.strip():
            try:
                rows = json.loads(json_text)
                df_input = pd.DataFrame(rows)
                df_input.columns = [c.lower() for c in df_input.columns]
                if "timestamp" in df_input.columns:
                    df_input["timestamp"] = pd.to_datetime(df_input["timestamp"])
                    df_input = df_input.set_index("timestamp")
                st.success(f"Parsed {len(df_input)} bars.")
            except Exception as e:
                st.error(f"JSON parse error: {e}")

    st.markdown("---")

    can_analyze = (df_input is not None and len(df_input) >= 20) or (image_bytes is not None)
    analyze_btn = st.button("🚀 Run Full Analysis", type="primary", disabled=not can_analyze)

    if analyze_btn and can_analyze:
        engine = _load_engine()
        # Override pivot sensitivity from sidebar
        engine.ms_engine.sensitivity = pivot_sens

        progress = st.progress(0, text="Initializing…")

        with st.spinner("Running analysis pipeline…"):
            try:
                if image_bytes and df_input is None:
                    # Vision-only
                    import tempfile, os
                    from vision.chart_analyzer import CV2_AVAILABLE as _CV2
                    if not _CV2:
                        progress.empty()
                        st.warning(
                            "**OpenCV is not installed** — image vision analysis is unavailable.\n\n"
                            "Install it and restart the dashboard:\n"
                            "```\npip install opencv-python-headless\n```"
                        )
                        st.stop()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        tmp.write(image_bytes)
                        tmp_path = tmp.name
                    progress.progress(30, text="Analyzing chart image…")
                    result = engine.analyze_image(tmp_path, asset=asset, timeframe=timeframe)
                    os.unlink(tmp_path)
                elif image_bytes and df_input is not None:
                    import tempfile, os
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        tmp.write(image_bytes)
                        tmp_path = tmp.name
                    progress.progress(20, text="Running quantitative analysis…")
                    result = engine.analyze_image(tmp_path, asset=asset, timeframe=timeframe, df=df_input)
                    os.unlink(tmp_path)
                else:
                    progress.progress(10, text="Market structure…")
                    # Pass all loaded TFs to the MTF engine (None = single-TF mode)
                    _mtf_payload = _tf_data_raw if len(_tf_data_raw) > 1 else None
                    result = engine.analyze_data(
                        df=df_input,
                        asset=asset,
                        timeframe=timeframe,
                        tf_data=_mtf_payload,
                        render_chart=render_chart,
                    )

                st.session_state.result            = result
                st.session_state.last_df           = df_input
                st.session_state.last_image_bytes  = image_bytes  # may be None
                st.session_state.tf_data           = _tf_data_raw  # persist loaded TFs
                progress.empty()
                st.toast("Analysis complete — check the Chart and AI Report tabs.", icon="✅")
                st.rerun()

            except Exception as exc:
                progress.empty()
                st.error(f"Analysis failed: {exc}")
                st.exception(exc)

    # Quick metrics below button
    if st.session_state.result:
        r = st.session_state.result
        st.markdown("---")
        st.subheader("Quick Summary")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if r.ms:
                struct = r.ms.structure_type.name
                delta  = f"{r.ms.trend_strength:.0%}"
            else:
                struct = r.confluence.direction.upper() if r.confluence else "N/A"
                delta  = "Vision"
            st.metric("Structure", struct, delta)
        with c2:
            score  = f"{r.confluence.total:.0f}" if r.confluence else "0"
            grade  = r.confluence.grade if r.confluence else "F"
            st.metric("Confluence", score, grade)
        with c3:
            direct = r.confluence.direction if r.confluence else "?"
            st.metric("Bias", direct.upper())
        with c4:
            st.metric("Current Price", f"{r.current_price:.4f}" if r.current_price else "N/A")


# ==========================================================================
# TAB 1 – Chart
# ==========================================================================
with tabs[1]:
    r = st.session_state.result
    if r is None:
        st.info("Run an analysis first (Input & Analyze tab).")
    else:
        chart_tab, scenario_tab = st.tabs(["📊 Analysis Chart", "🎯 Scenario Projection"])

        # ── Analysis Chart ────────────────────────────────────────────────────
        with chart_tab:
            # ── If an image was uploaded: show it with trend-direction arrows ──
            _img_bytes = st.session_state.get("last_image_bytes")
            if _img_bytes is not None:
                _direction = (
                    (r.report.trend_bias if r.report and r.report.trend_bias else None)
                    or (r.confluence.direction if r.confluence else "unknown")
                )
                _annotated = _annotate_chart_image(_img_bytes, _direction or "unknown", r.asset, r.timeframe)
                st.image(
                    _annotated,
                    caption=f"{r.asset} {r.timeframe} — Trend: {(_direction or 'unknown').upper()}",
                    use_column_width=True,
                )

            # ── Existing OHLCV / HTML chart (shown below image if CSV also provided) ─
            _ic = r.interactive_chart
            _sc = r.static_chart
            if _ic and Path(_ic).exists() and Path(_ic).suffix.lower() == ".html":
                html_content = Path(_ic).read_text(encoding="utf-8")
                st.components.v1.html(html_content, height=700, scrolling=False)
            elif _ic and Path(_ic).exists() and Path(_ic).suffix.lower() in (".png", ".jpg", ".jpeg"):
                st.image(str(_ic), use_column_width=True)
            elif _sc and Path(_sc).exists():
                st.image(str(_sc), use_column_width=True)
            elif _img_bytes is None:  # only show fallback if no image was uploaded
                df_plot = st.session_state.last_df
                if df_plot is not None:
                    try:
                        import plotly.graph_objects as go
                        fig = go.Figure(data=[go.Candlestick(
                            x=df_plot.index,
                            open=df_plot["open"], high=df_plot["high"],
                            low=df_plot["low"],   close=df_plot["close"],
                            name=asset,
                        )])
                        fig.update_layout(
                            template="plotly_dark",
                            title=f"{asset} {timeframe}",
                            xaxis_rangeslider_visible=False,
                            height=600,
                        )
                        if r.sr:
                            for z in r.sr.support_zones[:4]:
                                fig.add_hline(y=z.level, line_dash="dash", line_color="rgba(46,204,113,0.6)", annotation_text="S")
                            for z in r.sr.resistance_zones[:4]:
                                fig.add_hline(y=z.level, line_dash="dash", line_color="rgba(231,76,60,0.6)", annotation_text="R")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Chart render failed: {e}")
                else:
                    if st.session_state.get("last_image_bytes") is None:
                        st.info("No chart data available.")

            if r.sr:
                with st.expander("S/R Zones", expanded=False):
                    rows_display = []
                    for z in (r.sr.support_zones + r.sr.resistance_zones)[:12]:
                        rows_display.append({
                            "Level": f"{z.level:.4f}",
                            "Type":  z.zone_type,
                            "Confidence": f"{z.confidence:.0%}",
                            "Touches": z.touches,
                            "Fresh": "Yes" if z.is_fresh else "No",
                        })
                    st.dataframe(pd.DataFrame(rows_display), use_container_width=True, hide_index=True)

        # ── Scenario Projection Chart ─────────────────────────────────────────
        with scenario_tab:
            df_plot = st.session_state.last_df
            if df_plot is None:
                st.info("Upload OHLCV data to generate the scenario projection chart.")
            else:
                try:
                    from visualization.chart_renderer import ChartRenderer as _CR
                    _renderer = _CR(output_dir=str(Path("outputs")))
                    _ctx = {
                        "asset": r.asset, "timeframe": r.timeframe,
                        "sr": r.sr, "fib": r.fib, "smc": r.smc, "ms": r.ms,
                        "confluence": r.confluence,
                    }
                    _sc_path = _renderer.render_scenario_chart(df_plot, _ctx, f"{r.asset}_{r.timeframe}_scenario")
                    if _sc_path and Path(_sc_path).exists():
                        st.components.v1.html(Path(_sc_path).read_text(encoding="utf-8"), height=660, scrolling=False)
                    else:
                        st.warning("Could not render scenario chart — install plotly: `pip install plotly`")
                except Exception as _e:
                    st.warning(f"Scenario chart error: {_e}")


# ==========================================================================
# TAB 2 – AI Report
# ==========================================================================
with tabs[2]:
    st.subheader("AI-Generated Analysis Report")
    r = st.session_state.result
    if r is None:
        st.info("Run an analysis first.")
    else:
        # Confluence grade badge
        if r.confluence:
            grade     = r.confluence.grade
            col_score, col_grade, col_dir = st.columns([2, 1, 2])
            with col_score:
                st.metric("Confluence Score", f"{r.confluence.total:.1f} / 100")
            with col_grade:
                st.markdown(
                    f'<span class="grade-{grade}">{grade}</span>',
                    unsafe_allow_html=True,
                )
            with col_dir:
                color = "green" if r.confluence.direction == "bullish" else "red" if r.confluence.direction == "bearish" else "gray"
                st.markdown(
                    f'<span style="color:{color}; font-size:1.2rem; font-weight:700;">'
                    f'{r.confluence.direction.upper()} BIAS'
                    f'</span>',
                    unsafe_allow_html=True,
                )
            st.markdown("---")

            # Component scores
            with st.expander("Component Score Breakdown"):
                items = list(r.confluence.component_scores.items())
                bar_data = pd.DataFrame(items, columns=["Component", "Score"])
                st.dataframe(bar_data.set_index("Component"), use_container_width=True)

            # Reasons
            if r.confluence.reasons:
                with st.expander("Confluence Reasons"):
                    for reason in r.confluence.reasons:
                        st.write(f"• {reason}")

            # Entry/invalidation
            cols = st.columns(2)
            with cols[0]:
                if r.confluence.entry_zone:
                    st.info(f"Entry Zone: {r.confluence.entry_zone[0]:.4f} – {r.confluence.entry_zone[1]:.4f}")
            with cols[1]:
                if r.confluence.invalidation_levels:
                    lvls = ", ".join(f"{l:.4f}" for l in r.confluence.invalidation_levels[:3])
                    st.warning(f"Invalidation: {lvls}")

        # Full text report
        if r.report and r.report.full_text:
            st.markdown("---")

            # ── Dual Scenario Cards ──────────────────────────────────────────
            if r.report.bullish_scenario or r.report.bearish_scenario:
                st.markdown("### 🔀 Dual Scenario Analysis")
                scol_a, scol_b = st.columns(2)
                with scol_a:
                    st.markdown(
                        '<div style="background:#0d3b1e;border:2px solid #2ecc71;border-radius:10px;padding:16px;">'
                        '<h4 style="color:#2ecc71;margin-top:0;">📈 SCENARIO A — TRUE BREAKOUT</h4>'
                        f'<pre style="color:#cfffdf;white-space:pre-wrap;font-size:0.82rem;">'
                        f'{r.report.bullish_scenario}</pre></div>',
                        unsafe_allow_html=True,
                    )
                with scol_b:
                    st.markdown(
                        '<div style="background:#3b0d0d;border:2px solid #e74c3c;border-radius:10px;padding:16px;">'
                        '<h4 style="color:#e74c3c;margin-top:0;">📉 SCENARIO B — LIQUIDITY SWEEP</h4>'
                        f'<pre style="color:#ffd5d5;white-space:pre-wrap;font-size:0.82rem;">'
                        f'{r.report.bearish_scenario}</pre></div>',
                        unsafe_allow_html=True,
                    )
                if r.report.invalidation:
                    st.markdown(
                        f'<div style="background:#1a1a2e;border:1px solid #f39c12;border-radius:8px;'
                        f'padding:12px;margin-top:12px;">'
                        f'<span style="color:#f39c12;font-weight:700;">⚠️ Invalidation</span><br>'
                        f'<pre style="color:#fdf3d0;white-space:pre-wrap;font-size:0.82rem;margin:6px 0 0 0;">'
                        f'{r.report.invalidation}</pre></div>',
                        unsafe_allow_html=True,
                    )
                st.markdown("---")

            with st.expander("📄 Full Institutional Report", expanded=True):
                st.markdown(
                    f'<div class="report-box">{r.report.full_text}</div>',
                    unsafe_allow_html=True,
                )
            # Download button
            st.download_button(
                "⬇️ Download Report (TXT)",
                data=r.report.full_text,
                file_name=f"prometheus_{r.asset}_{r.timeframe}_{r.run_id}.txt",
                mime="text/plain",
            )

            # ── 🎯 Final Signal Card ─────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 🎯 Final Trading Signal")
            _sig = getattr(r.report, "final_signal", "") or ""
            if _sig and "Insufficient" not in _sig and "ranging" not in _sig.lower():
                _is_buy = "BUY" in _sig.upper()
                _sig_bg     = "#0a2e1a" if _is_buy else "#2e0a0a"
                _sig_border = "#2ecc71" if _is_buy else "#e74c3c"
                _sig_header = "🟢 BUY SIGNAL" if _is_buy else "🔴 SELL SIGNAL"
                _sig_hcol   = "#2ecc71" if _is_buy else "#e74c3c"
                _sig_tcol   = "#c8ffd4" if _is_buy else "#ffd4d4"

                # Parse lines to extract key fields for metric display
                _entry = _sl = _tp1 = _tp2 = None
                for _ln in _sig.splitlines():
                    if "Entry Area" in _ln:
                        try: _entry = float(_ln.split(":")[1].strip().split()[0])
                        except: pass
                    elif "Stop Loss" in _ln:
                        try: _sl = float(_ln.split(":")[1].strip().split()[0])
                        except: pass
                    elif "Take Profit 1" in _ln:
                        try: _tp1 = float(_ln.split(":")[1].strip().split()[0])
                        except: pass
                    elif "Take Profit 2" in _ln:
                        try: _tp2 = float(_ln.split(":")[1].strip().split()[0])
                        except: pass

                # Metrics row
                _m1, _m2, _m3, _m4 = st.columns(4)
                with _m1:
                    st.metric("Entry Area", f"{_entry:.2f}" if _entry else "—")
                with _m2:
                    st.metric("Stop Loss", f"{_sl:.2f}" if _sl else "—",
                              delta=f"Risk {abs(_entry-_sl):.2f}" if (_entry and _sl) else None,
                              delta_color="inverse")
                with _m3:
                    st.metric("Take Profit 1", f"{_tp1:.2f}" if _tp1 else "—",
                              delta=f"R:R 1:{abs(_tp1-_entry)/max(abs(_entry-_sl),0.001):.1f}" if (_tp1 and _entry and _sl) else None)
                with _m4:
                    st.metric("Take Profit 2", f"{_tp2:.2f}" if _tp2 else "—",
                              delta=f"R:R 1:{abs(_tp2-_entry)/max(abs(_entry-_sl),0.001):.1f}" if (_tp2 and _entry and _sl) else None)

                # Full signal card
                st.markdown(
                    f'<div style="background:{_sig_bg};border:2px solid {_sig_border};'
                    f'border-radius:12px;padding:20px;margin-top:8px;">'
                    f'<h3 style="color:{_sig_hcol};margin-top:0;letter-spacing:2px;">{_sig_header}</h3>'
                    f'<pre style="color:{_sig_tcol};white-space:pre-wrap;font-size:0.88rem;'
                    f'line-height:1.7;margin:0;">{_sig}</pre>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                _reason = _sig if _sig else "No quantitative price data — upload OHLCV CSV for precise levels."
                st.info(f"⏳ {_reason}")


# ==========================================================================
# TAB 3 – Patterns & SMC
# ==========================================================================
with tabs[3]:
    r = st.session_state.result
    if r is None:
        st.info("Run an analysis first.")
    else:
        # ══════════════════════════════════════════════════════════════════
        # TRADE SETUP PREDICTION CHART  (entry / SL / TP1 / TP2 + MTF table)
        # ══════════════════════════════════════════════════════════════════
        df_setup = st.session_state.get("last_df")
        if df_setup is not None and len(df_setup) >= 20:
            st.subheader("📊 Trade Setup Prediction")

            # ── Derive entry / SL / TP from analysis results ──────────────
            # ALWAYS anchor entry to the last close in the displayed data.
            # r.current_price is the price at analysis run-time and may be stale.
            price_now  = float(df_setup["close"].iloc[-1])
            price_orig = r.current_price or price_now
            atr        = float((df_setup["high"] - df_setup["low"]).rolling(14).mean().iloc[-1])
            direction  = (r.confluence.direction if r.confluence else "sideways").lower()
            is_long    = direction in ("bullish", "long")
            is_short   = direction in ("bearish", "short")

            # Warn if the analysis is stale (price has moved >1 ATR from when it was run)
            _price_drift = abs(price_now - price_orig)
            _is_stale = _price_drift > atr * 0.5

            # SL / TP from S&R — only use a level if it sits on the correct side of NOW
            sup_lvl = res_lvl = None
            if r.sr:
                ns = r.sr.nearest_support
                nr = r.sr.nearest_resistance
                if ns:
                    _s = float(ns.lower)
                    if _s < price_now:          # support must be below current price
                        sup_lvl = _s
                if nr:
                    _r2 = float(nr.upper)
                    if _r2 > price_now:         # resistance must be above current price
                        res_lvl = _r2

            # Build levels anchored to CURRENT price (price_now)
            _MIN_SL_ATR = 1.0   # SL must be at least 1.0 × ATR away from entry
            if is_long:
                entry = price_now
                sl    = (sup_lvl - atr * 0.2) if sup_lvl else price_now - 1.5 * atr
                tp1   = res_lvl if res_lvl else price_now + 2.0 * atr
                tp2   = tp1 + (tp1 - entry)
                # sanity: sl must be < entry and ≥ 1× ATR away
                if sl >= entry or (entry - sl) < atr * _MIN_SL_ATR:
                    sl = entry - atr * _MIN_SL_ATR
                if tp1 <= entry:
                    tp1 = entry + 2.0 * atr
                    tp2 = tp1 + (tp1 - entry)
            elif is_short:
                entry = price_now
                sl    = (res_lvl + atr * 0.2) if res_lvl else price_now + 1.5 * atr
                tp1   = sup_lvl if sup_lvl else price_now - 2.0 * atr
                tp2   = tp1 - (entry - tp1)
                # sanity: sl must be > entry and ≥ 1× ATR away
                if sl <= entry or (sl - entry) < atr * _MIN_SL_ATR:
                    sl = entry + atr * _MIN_SL_ATR
                if tp1 >= entry:
                    tp1 = entry - 2.0 * atr
                    tp2 = tp1 - (entry - tp1)
            else:
                entry = sl = tp1 = tp2 = None

            # Show stale-signal notice BEFORE the chart
            if _is_stale:
                st.warning(
                    f"⚠️ **Stale signal** — analysis was generated when price was "
                    f"**{price_orig:.4f}**. Current price is **{price_now:.4f}** "
                    f"(drift: {_price_drift:.2f}, ~{_price_drift/atr:.1f}× ATR). "
                    "Levels below are recalculated from current price. "
                    "Re-run analysis for a fresh signal.",
                    icon=None,
                )

            # Allow manual override via UI
            with st.expander("⚙️ Adjust levels", expanded=False):
                ov1, ov2, ov3, ov4 = st.columns(4)
                entry = ov1.number_input("Entry",     value=round(entry or price_now, 4), step=0.01, format="%.4f", key="sc_entry")
                sl    = ov2.number_input("Stop Loss", value=round(sl or price_now - atr, 4), step=0.01, format="%.4f", key="sc_sl")
                tp1   = ov3.number_input("TP 1",     value=round(tp1 or price_now + atr, 4), step=0.01, format="%.4f", key="sc_tp1")
                tp2   = ov4.number_input("TP 2",     value=round(tp2 or price_now + 2*atr, 4), step=0.01, format="%.4f", key="sc_tp2")

            # ── Build Plotly chart ────────────────────────────────────────
            import plotly.graph_objects as _go

            _df = df_setup.copy()
            _df.columns = [c.lower() for c in _df.columns]
            _n  = min(60, len(_df))
            _df = _df.iloc[-_n:]
            _xi = list(range(_n))
            _last_xi = _n - 1

            if hasattr(_df.index, "strftime"):
                _labels = _df.index.strftime("%m-%d %H:%M").tolist()
            else:
                _labels = [str(v) for v in _df.index]
            _tstep = max(1, _n // 8)
            _tvals = list(range(0, _n, _tstep))
            _ttxt  = [_labels[i] for i in _tvals]

            _fig = _go.Figure()

            # Candlesticks
            _fig.add_trace(_go.Candlestick(
                x=_xi,
                open=_df["open"], high=_df["high"],
                low=_df["low"],   close=_df["close"],
                name="OHLCV",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
                showlegend=False,
            ))

            # ── Order Blocks ──────────────────────────────────────────────
            # bar_idx is absolute index in the full df; we mapped last _n bars
            # to xi 0.._n-1, so: xi = bar_idx - (len(full_df) - _n)
            _full_len = len(df_setup)
            _bar_offset = _full_len - _n
            _price_lo = _df["low"].min()
            _price_hi = _df["high"].max()
            _visible  = lambda y: _price_lo * 0.997 <= y <= _price_hi * 1.003

            if r.smc and r.smc.order_blocks:
                for _ob in r.smc.order_blocks:
                    if not (_visible(_ob.high) or _visible(_ob.low)):
                        continue
                    _ob_color = ("rgba(38,166,154,0.18)" if _ob.direction == "bullish"
                                 else "rgba(239,83,80,0.18)")
                    _ob_border = ("#26a69a" if _ob.direction == "bullish" else "#ef5350")
                    _ob_xi  = _ob.bar_idx - _bar_offset
                    _ob_x0  = max(0, _ob_xi - 2)
                    _ob_x1  = _last_xi
                    _ob_tag = "🟢 OB" if _ob.direction == "bullish" else "🔴 OB"
                    if _ob.mitigated:
                        _ob_tag += " ✓"
                    _fig.add_shape(
                        type="rect",
                        x0=_ob_x0, x1=_ob_x1, y0=_ob.low, y1=_ob.high,
                        fillcolor=_ob_color,
                        line=dict(color=_ob_border, width=1, dash="dot"),
                        xref="x", yref="y", layer="below",
                    )
                    _fig.add_annotation(
                        x=_ob_x0, y=(_ob.high + _ob.low) / 2,
                        text=f"<b>{_ob_tag}</b>",
                        showarrow=False, xanchor="left",
                        font=dict(color="#111111", size=10),
                        bgcolor=_ob_border, borderpad=2, opacity=0.85,
                    )

            # ── Fair Value Gaps ────────────────────────────────────────────
            if r.smc and r.smc.fair_value_gaps:
                _fvg_label_ys: list[float] = []   # track used label y-positions
                for _fvg in r.smc.fair_value_gaps:
                    if _fvg.filled:
                        continue
                    if not (_visible(_fvg.high) or _visible(_fvg.low)):
                        continue
                    # Scale opacity down for large zones to avoid obscuring candles
                    _fvg_size   = abs(_fvg.high - _fvg.low)
                    _fvg_range  = _price_hi - _price_lo if (_price_hi - _price_lo) > 0 else 1
                    _fvg_alpha  = max(0.08, 0.22 - 0.18 * (_fvg_size / _fvg_range))
                    _fvg_is_bull = _fvg.direction == "bullish"
                    _fvg_fill   = (f"rgba(100,181,246,{_fvg_alpha:.2f})" if _fvg_is_bull
                                   else f"rgba(255,167,38,{_fvg_alpha:.2f})")
                    _fvg_border = "#64b5f6" if _fvg_is_bull else "#ffa726"
                    _fvg_xi     = _fvg.start_idx - _bar_offset
                    _fvg_x0     = max(0, _fvg_xi)
                    _fvg_tag    = "⬆ FVG" if _fvg_is_bull else "⬇ FVG"
                    _fig.add_shape(
                        type="rect",
                        x0=_fvg_x0, x1=_last_xi, y0=_fvg.low, y1=_fvg.high,
                        fillcolor=_fvg_fill,
                        line=dict(color=_fvg_border, width=1, dash="dot"),
                        xref="x", yref="y", layer="below",
                    )
                    # De-duplicate labels that land too close on the y-axis
                    _min_gap    = _fvg_range * 0.018
                    _label_y    = _fvg.mid
                    for _used_y in _fvg_label_ys:
                        if abs(_label_y - _used_y) < _min_gap:
                            _label_y = _used_y + _min_gap * (1 if _fvg_is_bull else -1)
                    _fvg_label_ys.append(_label_y)
                    _fig.add_annotation(
                        x=_fvg_x0, y=_label_y,
                        text=f"<b>{_fvg_tag}</b>",
                        showarrow=False, xanchor="left",
                        font=dict(color="#111111", size=10),
                        bgcolor=_fvg_border, borderpad=2, opacity=0.85,
                    )

            # ── Chart Patterns ─────────────────────────────────────────────
            if r.pat and r.pat.patterns:
                _price_range  = _price_hi - _price_lo if (_price_hi - _price_lo) > 0 else 1
                _label_step   = _price_range * 0.06   # vertical gap between stacked labels
                _pat_label_ys: list[float] = []        # track used label y-positions
                for _pi, _pat in enumerate(r.pat.patterns[:5]):
                    _p_x0 = max(0, _pat.start_idx - _bar_offset)
                    _p_x1 = min(_last_xi, _pat.end_idx - _bar_offset)
                    if _p_x1 < 0 or _p_x0 > _last_xi:
                        continue
                    _p_color = ("#26a69a" if _pat.direction == "bullish"
                                else "#ef5350" if _pat.direction == "bearish"
                                else "#90a4ae")
                    _r, _g, _b = int(_p_color[1:3],16), int(_p_color[3:5],16), int(_p_color[5:7],16)
                    # Thin dashed vertical lines at pattern start & end (cleaner than full rect)
                    for _vx in [_p_x0, max(_p_x0 + 1, _p_x1)]:
                        _fig.add_shape(
                            type="line",
                            x0=_vx, x1=_vx,
                            y0=_price_lo, y1=_price_hi,
                            line=dict(color=_p_color, width=1, dash="longdash"),
                            xref="x", yref="y", layer="below",
                        )
                    # Subtle background tint between start and end
                    _fig.add_shape(
                        type="rect",
                        x0=_p_x0, x1=max(_p_x0 + 1, _p_x1),
                        y0=_price_lo, y1=_price_hi,
                        fillcolor=f"rgba({_r},{_g},{_b},0.04)",
                        line_width=0,
                        xref="x", yref="y", layer="below",
                    )
                    # Place label INSIDE chart near top, staggered vertically
                    _candidate_y = _price_hi - _label_step * 0.5
                    for _used_y in _pat_label_ys:
                        if abs(_candidate_y - _used_y) < _label_step:
                            _candidate_y = _used_y - _label_step
                    _candidate_y = max(_price_lo + _label_step, _candidate_y)
                    _pat_label_ys.append(_candidate_y)
                    _emoji = "📈" if _pat.direction == "bullish" else "📉" if _pat.direction == "bearish" else "◀▶"
                    _pat_label = f"{_emoji} {_pat.name} ({_pat.confidence:.0%})"
                    _fig.add_annotation(
                        x=_p_x0, y=_candidate_y,
                        text=f"<b>{_pat_label}</b>",
                        showarrow=False, xanchor="left",
                        font=dict(color=_p_color, size=10),
                        bgcolor="rgba(20,20,20,0.75)", bordercolor=_p_color,
                        borderpad=3,
                    )
                    if _pat.target_price and _visible(_pat.target_price):
                        _fig.add_shape(
                            type="line",
                            x0=_p_x1, x1=_last_xi,
                            y0=_pat.target_price, y1=_pat.target_price,
                            line=dict(color=_p_color, width=1, dash="dot"),
                            xref="x", yref="y",
                        )

            # ── Shaded SL / entry / TP zones ─────────────────────────────
            # SL zone (red fill)
            _sl_ref = min(sl, entry) if is_long else max(sl, entry)
            _entry_ref = entry
            _fig.add_hrect(
                y0=min(_sl_ref, _entry_ref), y1=max(_sl_ref, _entry_ref),
                fillcolor="rgba(239,83,80,0.13)", line_width=0,
            )
            # TP1→TP2 zone (green fill)
            _fig.add_hrect(
                y0=min(tp1, tp2), y1=max(tp1, tp2),
                fillcolor="rgba(38,166,154,0.13)", line_width=0,
            )

            # ── Horizontal lines with badge annotations ───────────────────
            def _hline(fig, y, color, dash, label, pos="right"):
                fig.add_shape(
                    type="line",
                    x0=0, x1=_last_xi, y0=y, y1=y,
                    line=dict(color=color, width=1.5, dash=dash),
                    xref="x", yref="y",
                )
                # Badge annotation on right edge
                fig.add_annotation(
                    x=_last_xi + 0.5, y=y,
                    text=f"<b>{label} : {y:.2f}</b>",
                    showarrow=False, xanchor="left",
                    font=dict(color="#111111", size=11),
                    bgcolor=color, bordercolor=color,
                    borderwidth=1, borderpad=4,
                )

            _green  = "#00bcd4"
            _amber  = "#fdd835"
            _red_sl = "#ef5350"

            _hline(_fig, tp2,   _green,  "solid", "TP 2")
            _hline(_fig, tp1,   _green,  "solid", "TP 1")
            _hline(_fig, entry, _amber,  "dash",  "Entry")
            _hline(_fig, sl,    _red_sl, "solid", "Stop loss")

            # ── Current price ticker (last close = entry for this chart) ────
            _cur = price_now
            _fig.add_annotation(
                x=_last_xi + 0.5, y=_cur,
                text=f"<b>{_cur:.2f}</b>",
                showarrow=False, xanchor="left",
                font=dict(color="#111111", size=11),
                bgcolor="#26a69a" if is_long else "#ef5350" if is_short else "#607d8b",
                borderpad=3,
            )

            # ── BUY/SELL label badge at midpoint ──────────────────────────
            if is_long or is_short:
                _mid_x = _n // 3
                _mid_y = (entry + sl) / 2
                _side_label = "Buy" if is_long else "Sell"
                _side_color = "#00bcd4" if is_long else "#ef5350"
                _fig.add_annotation(
                    x=_mid_x, y=_mid_y,
                    text=f"<b>{_side_label}</b>",
                    showarrow=True, arrowhead=2,
                    arrowcolor=_side_color,
                    font=dict(color="#111111", size=13),
                    bgcolor=_side_color, borderpad=5,
                    ax=0, ay=30 if is_long else -30,
                )

            # R:R display
            _rr = abs(tp1 - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0
            _rr2 = abs(tp2 - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0

            _fig.update_layout(
                title=dict(
                    text=(f"<b>{asset} {timeframe}</b>  |  "
                          f"{'🟢 BUY' if is_long else '🔴 SELL' if is_short else '⚪ SIDEWAYS'}  |  "
                          f"Grade: <b>{r.confluence.grade if r.confluence else '?'}</b>  |  "
                          f"Score: <b>{r.confluence.total if r.confluence else 0:.0f}</b>  |  "
                          f"R:R TP1 <b>1:{_rr:.1f}</b>  TP2 <b>1:{_rr2:.1f}</b>"),
                    font=dict(size=13),
                ),
                template="plotly_dark",
                xaxis=dict(
                    rangeslider_visible=False,
                    tickvals=_tvals, ticktext=_ttxt, tickangle=-30,
                ),
                yaxis=dict(side="right"),
                height=560,
                margin=dict(l=10, r=160, t=70, b=40),
                showlegend=True,
                legend=dict(
                    orientation="h", y=-0.12, x=0,
                    font=dict(size=10),
                    bgcolor="rgba(0,0,0,0)",
                ),
            )
            # Invisible dummy traces for the legend
            _ob_n  = len(r.smc.order_blocks)  if (r.smc and r.smc.order_blocks)  else 0
            _fvg_n = len([f for f in r.smc.fair_value_gaps if not f.filled]) if (r.smc and r.smc.fair_value_gaps) else 0
            _pat_n = len(r.pat.patterns) if (r.pat and r.pat.patterns) else 0
            if _ob_n:
                _fig.add_trace(_go.Scatter(x=[None], y=[None], mode="markers",
                    marker=dict(symbol="square", size=10, color="#26a69a"),
                    name=f"Order Blocks ({_ob_n})", showlegend=True))
            if _fvg_n:
                _fig.add_trace(_go.Scatter(x=[None], y=[None], mode="markers",
                    marker=dict(symbol="square", size=10, color="#64b5f6"),
                    name=f"FVGs ({_fvg_n} unfilled)", showlegend=True))
            if _pat_n:
                _fig.add_trace(_go.Scatter(x=[None], y=[None], mode="markers",
                    marker=dict(symbol="triangle-up", size=10, color="#90a4ae"),
                    name=f"Patterns ({_pat_n})", showlegend=True))

            st.plotly_chart(_fig, use_container_width=True)

            # ── R:R summary row + ATR-normalised SL health ────────────────
            _sl_dist   = abs(entry - sl)
            _sl_atr    = _sl_dist / atr if atr > 0 else 0
            _sl_health = ("✅ Good" if _sl_atr >= 1.0
                         else "⚠️ Tight" if _sl_atr >= 0.5
                         else "🔴 Too tight")

            sm1, sm2, sm3, sm4, sm5 = st.columns(5)
            sm1.metric("Entry",      f"{entry:.4f}")
            sm2.metric("Stop Loss",  f"{sl:.4f}",
                       delta=f"{sl-entry:+.4f}  [{_sl_atr:.1f}×ATR] {_sl_health}",
                       delta_color="inverse")
            sm3.metric("TP 1",       f"{tp1:.4f}", delta=f"{tp1-entry:+.4f}")
            sm4.metric("TP 2",       f"{tp2:.4f}", delta=f"{tp2-entry:+.4f}")
            sm5.metric("R:R (TP1)",  f"1 : {_rr:.2f}",
                       delta="Good" if _rr >= 1.5 else "Low",
                       delta_color="normal" if _rr >= 1.5 else "inverse")

            # VWAP badge (from analysis result)
            if r.vwap:
                _v   = r.vwap
                _vbg = "#26a69a" if _v.signal == "above" else "#ef5350" if _v.signal == "below" else "#607d8b"
                _arr = "▲" if _v.signal == "above" else "▼" if _v.signal == "below" else "↔"
                st.markdown(
                    f"<div style='display:inline-flex;gap:14px;align-items:center;"
                    f"background:{_vbg}22;border:1px solid {_vbg};"
                    f"border-radius:6px;padding:6px 14px;margin-top:6px'>"
                    f"<span style='color:{_vbg};font-weight:700'>VWAP</span>"
                    f"<span><b>{_v.vwap:.4f}</b></span>"
                    f"<span style='color:{_vbg}'>{_arr} price {_v.signal}</span>"
                    f"<span style='color:#aaa'>{_v.atr_distance:.2f}× ATR away</span>"
                    f"<span style='color:#aaa'>±1σ band: {_v.band1_lower:.2f} – {_v.band1_upper:.2f}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # ── Lot size calculator ────────────────────────────────────────
            with st.expander("💰 Lot Size Calculator", expanded=False):
                lc1, lc2, lc3 = st.columns(3)
                _balance  = lc1.number_input("Account Balance ($)",   value=10000.0, step=500.0,  format="%.2f", key="ls_balance")
                _risk_pct = lc2.number_input("Risk per trade (%)",    value=1.0,     step=0.1,    format="%.2f", min_value=0.1, max_value=10.0, key="ls_risk")
                _pip_val  = lc3.number_input("Pip value per lot ($)", value=10.0,    step=1.0,    format="%.2f", key="ls_pipval",
                                             help="XAUUSD ≈ $10/lot/pip. Adjust for your broker.")
                _risk_amt  = _balance * _risk_pct / 100.0
                _sl_pips   = _sl_dist * 10.0    # 1 price point = 10 pips on XAUUSD
                _lot_size  = _risk_amt / (_sl_pips * _pip_val) if _sl_pips > 0 else 0.0
                _lot_size  = round(max(0.01, _lot_size), 2)
                la, lb, lc_col, ld = st.columns(4)
                la.metric("Risk Amount",  f"${_risk_amt:.2f}")
                lb.metric("SL Distance",  f"{_sl_dist:.4f} pts  ({_sl_pips:.1f} pips)")
                lc_col.metric("Lot Size", f"{_lot_size:.2f} lots",
                              delta="standard" if 0.01 <= _lot_size <= 2.0 else "check size",
                              delta_color="normal" if 0.01 <= _lot_size <= 2.0 else "inverse")
                ld.metric("Max Loss",     f"${_risk_amt:.2f}  ({_risk_pct:.1f}%)")
                st.caption(
                    f"lot = risk$ ÷ (SL pips × pip value) "
                    f"= ${_risk_amt:.2f} ÷ ({_sl_pips:.1f} × ${_pip_val:.2f}) = **{_lot_size:.2f} lots**"
                )

            # ── Prepare Trade (send manual override to bot) ───────────────
            with st.expander("📤 Prepare Trade — Send to Bot", expanded=False):
                st.caption(
                    "Fill in the trade details below and click **Send to Bot**. "
                    "The bot will execute it on its next poll cycle. "
                    "Leave Lots = 0 to let the bot auto-size based on risk settings."
                )
                _price_now   = float(r.current_price) if r.current_price else 0.0
                _atr_now     = float(r.ms.current_atr) if (r.ms and r.ms.current_atr) else 0.0
                _sup_now     = float(r.sr.nearest_support.level)    if (r.sr and r.sr.nearest_support)    else None
                _res_now     = float(r.sr.nearest_resistance.level)  if (r.sr and r.sr.nearest_resistance) else None

                ptrow1a, ptrow1b, ptrow1c = st.columns(3)
                _pt_dir      = ptrow1a.selectbox("Direction", ["BUY", "SELL"], key="pt_dir")
                _pt_order    = ptrow1b.radio("Order type", ["Market", "Limit"], horizontal=True, key="pt_order")
                _pt_lots     = ptrow1c.number_input("Lots (0 = auto)", value=0.0, step=0.01, format="%.2f", min_value=0.0, key="pt_lots")

                _is_limit    = (_pt_order == "Limit")
                # Suggest the nearest zone edge as the default limit entry price
                if _is_limit:
                    if _pt_dir == "BUY":
                        _default_entry = _sup_now if _sup_now else round(_price_now - _atr_now, 4)
                    else:
                        _default_entry = _res_now if _res_now else round(_price_now + _atr_now, 4)
                    _default_entry = round(_default_entry, 4)

                ptrow2a, ptrow2b, ptrow2c = st.columns(3)
                if _is_limit:
                    _pt_entry = ptrow2a.number_input(
                        "Entry (limit price)", value=_default_entry,
                        step=0.1, format="%.4f", key="pt_entry",
                        help="Bot will place a Buy/Sell Limit order at this price.",
                    )
                else:
                    _pt_entry = None
                    ptrow2a.info("Market order — executes at current price on next poll.")

                _pt_sl = ptrow2b.number_input("Stop Loss",   value=round(sl  or _price_now, 4), step=0.1, format="%.4f", key="pt_sl")
                _pt_tp = ptrow2c.number_input("Take Profit", value=round(tp1 or _price_now, 4), step=0.1, format="%.4f", key="pt_tp")
                _pt_comment = st.text_input("Comment (optional)", value="Prom-manual", max_chars=31, key="pt_comment")

                _bot_status_file = Path(__file__).parent.parent / "live_bot" / "bot_status.json"
                _manual_file     = Path(__file__).parent.parent / "live_bot" / "manual_trade.json"

                # Live bot-status feedback
                _pt_bs = {}
                try:
                    if _bot_status_file.exists():
                        _pt_bs = json.loads(_bot_status_file.read_text(encoding="utf-8"))
                except Exception:
                    pass
                _pt_connected = _pt_bs.get("mt5_connected", False)
                _pt_halted    = _pt_bs.get("trading_halted", False)
                _pt_last_act  = _pt_bs.get("last_action", "")
                _pt_last_poll = _pt_bs.get("last_poll", "")
                _bsc1, _bsc2, _bsc3 = st.columns(3)
                _bsc1.metric("Bot MT5",   "Connected" if _pt_connected else "Disconnected")
                _bsc2.metric("Trading",   "HALTED" if _pt_halted else "Active")
                _bsc3.metric("Last poll", _pt_last_poll[-8:] if _pt_last_poll else "-")
                if _pt_last_act:
                    if "FAILED" in _pt_last_act or "HALTED" in _pt_last_act or "Error" in _pt_last_act:
                        st.error(f"Bot result: {_pt_last_act}")
                    elif "MANUAL" in _pt_last_act or "executed" in _pt_last_act.lower() or "LTF-scalp" in _pt_last_act:
                        st.success(f"Bot result: {_pt_last_act}")
                    else:
                        st.caption(f"Bot result: {_pt_last_act}")
                if _pt_halted:
                    st.info("Trading is halted but manual trades bypass the halt and will still execute.")
                st.divider()
                pt_btn_col, pt_status_col = st.columns([1, 3])
                if pt_btn_col.button("📤 Send to Bot", type="primary", key="pt_send"):
                    from datetime import datetime as _dt_pt  # noqa: PLC0415
                    _payload = {
                        "direction":  _pt_dir,
                        "order_type": "limit" if _is_limit else "market",
                        "entry":      _pt_entry,   # None for market orders
                        "sl":         _pt_sl,
                        "tp":         _pt_tp,
                        "lots":       _pt_lots,
                        "comment":    _pt_comment or "Prom-manual",
                        "created_at": _dt_pt.utcnow().isoformat(),
                    }
                    try:
                        _manual_file.write_text(json.dumps(_payload, indent=2), encoding="utf-8")
                        _order_desc = (f"Limit @ {_pt_entry:.4f}" if _is_limit else "Market")
                        pt_status_col.success(
                            f"✅ Queued! Bot will execute on next poll (≤60 s). "
                            f"{_pt_dir} {_order_desc} | SL={_pt_sl:.4f} | TP={_pt_tp:.4f} | "
                            f"Refresh dashboard to see Bot result above."
                        )
                    except Exception as _e:
                        pt_status_col.error(f"❌ Could not write trade file: {_e}")

                # Show if a pending override is waiting
                if _manual_file.exists():
                    try:
                        _pending = json.loads(_manual_file.read_text(encoding="utf-8"))
                        _pend_otype = _pending.get("order_type", "market").capitalize()
                        _pend_entry = _pending.get("entry")
                        _pend_entry_str = f" @ {_pend_entry:.4f}" if _pend_entry else ""
                        st.warning(
                            f"⏳ **Pending:** {_pending.get('direction')} {_pend_otype}{_pend_entry_str} "
                            f"| SL={_pending.get('sl')} | TP={_pending.get('tp')} "
                            f"— waiting for bot poll."
                        )
                    except Exception:
                        pass

            # ── Multi-Timeframe trend table (like image bottom-right) ─────
            if r.mtf and r.mtf.biases:
                _saved_tf_data = st.session_state.get("tf_data", {})
                _mtf_c1, _mtf_c2 = st.columns([2, 3])
                with _mtf_c1:
                    st.markdown("**Multi-Timeframe Trend**")
                    _tf_rows = []
                    for _b in r.mtf.biases:
                        _arrow  = "↑ Bullish" if _b.bias == "bullish" else "↓ Bearish" if _b.bias == "bearish" else "→ Sideways"
                        _col    = "🟢" if _b.bias == "bullish" else "🔴" if _b.bias == "bearish" else "🟡"
                        _source = "📂 real" if _b.timeframe.lower() in _saved_tf_data else "⚙️ inferred"
                        _bars   = f"{len(_saved_tf_data[_b.timeframe.lower()]):,}" if _b.timeframe.lower() in _saved_tf_data else "—"
                        _tf_rows.append({
                            "TF":     _b.timeframe.upper(),
                            "Trend":  f"{_col} {_arrow}",
                            "Source": _source,
                            "Bars":   _bars,
                        })
                    _tf_df = pd.DataFrame(_tf_rows)
                    _align   = r.mtf.confluence_level.upper()
                    _overall = "🟢 BULLISH" if r.mtf.primary_bias == "bullish" else "🔴 BEARISH" if r.mtf.primary_bias == "bearish" else "🟡 SIDEWAYS"
                    st.dataframe(_tf_df, hide_index=True, use_container_width=True)
                    st.caption(f"Overall: **{_overall}** · Confluence: **{_align}** · Score: {r.mtf.alignment_score:+.2f}")
                with _mtf_c2:
                    # Prompt to load more TFs if only 1 was uploaded (CSV mode only)
                    if len(_saved_tf_data) <= 1 and input_mode != "🟢 Live from MT5":
                        st.info(
                            "💡 **Upload additional timeframe CSVs** in the "
                            "**Input & Analyze → ➕ Add more timeframes** expander "
                            "to get real MTF data instead of inferred biases. "
                            "More TFs → sharper confluence signal."
                        )
                    else:
                        # Mini bar-count chart
                        _bc_data = {k.upper(): len(v) for k, v in _saved_tf_data.items()}
                        _bc_df = pd.DataFrame(list(_bc_data.items()), columns=["Timeframe", "Bars"])
                        st.markdown("**Bars loaded per timeframe**")
                        st.bar_chart(_bc_df.set_index("Timeframe"))

            st.divider()

        col_pat, col_smc = st.columns(2)

        with col_pat:
            st.subheader("Chart Patterns")
            if r.pat and r.pat.patterns:
                for p in r.pat.patterns[:8]:
                    suffix = "📈" if p.direction == "bullish" else "📉"
                    with st.expander(f"{suffix} {p.name} ({p.confidence:.0%})"):
                        st.write(p.description)
                        if p.target_price:
                            st.metric("Target", f"{p.target_price:.4f}")
                        if p.invalidation:
                            st.metric("Invalidation", f"{p.invalidation:.4f}")
            else:
                if st.session_state.get("last_df") is None:
                    st.warning(
                        "📂 **No OHLCV data loaded.**  "
                        "Chart Patterns are computed from real candle swing points.  "
                        "Upload a CSV (or export from MT5 via *F2 → History Center*) "
                        "to unlock this section."
                    )
                else:
                    st.info("No chart patterns detected in the uploaded data.")

            st.markdown("---")
            st.subheader("Candlestick Signals")
            if r.cs and r.cs.top_signals:
                cs_data = []
                for sig in r.cs.top_signals:
                    cs_data.append({
                        "Pattern":    sig.pattern,
                        "Direction":  sig.direction,
                        "Raw Score":  f"{sig.raw_score:.1f}",
                        "Final":      f"{sig.final_score:.1f}",
                    })
                st.dataframe(pd.DataFrame(cs_data), use_container_width=True, hide_index=True)
            else:
                if st.session_state.get("last_df") is None:
                    st.warning(
                        "📂 **No OHLCV data loaded.**  "
                        "Candlestick signals (Engulfing, Doji, Hammer, etc.) are detected "
                        "from actual candle open/high/low/close values — not from the image.  "
                        "Upload a CSV to enable this."
                    )
                else:
                    st.info("No candlestick signals detected.")

        with col_smc:
            st.subheader("Smart Money Concepts (SMC)")
            if r.smc:
                if r.smc.narrative:
                    st.markdown(f"**Narrative:** {r.smc.narrative}")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Order Blocks",    len(r.smc.order_blocks))
                    st.metric("Stop Hunts",      len(r.smc.stop_hunts))
                with col2:
                    st.metric("Fair Value Gaps", len(r.smc.fair_value_gaps))
                    st.metric("Liquidity Pools", len(r.smc.liquidity_pools))

                if r.smc.order_blocks:
                    with st.expander("Order Blocks"):
                        ob_data = []
                        for ob in r.smc.order_blocks[:8]:
                            ob_data.append({
                                "Type":      ob.direction,
                                "High":      f"{ob.high:.4f}",
                                "Low":       f"{ob.low:.4f}",
                                "Mitigated": ob.mitigated,
                                "Strength":  f"{ob.strength:.2f}",
                            })
                        st.dataframe(pd.DataFrame(ob_data), use_container_width=True, hide_index=True)

                if r.smc.fair_value_gaps:
                    with st.expander("Fair Value Gaps"):
                        fvg_data = []
                        for fvg in r.smc.fair_value_gaps[:8]:
                            fvg_data.append({
                                "Type":   fvg.direction,
                                "High":   f"{fvg.high:.4f}",
                                "Low":    f"{fvg.low:.4f}",
                                "Filled": fvg.filled,
                            })
                        st.dataframe(pd.DataFrame(fvg_data), use_container_width=True, hide_index=True)
            else:
                if st.session_state.get("last_df") is None:
                    st.warning(
                        "📂 **No OHLCV data loaded.**  "
                        "SMC analysis (Order Blocks, Fair Value Gaps, Stop Hunts, Liquidity Pools) "
                        "requires actual candle price data.  "
                        "Upload a CSV alongside the chart image to enable this section."
                    )
                else:
                    st.info("No SMC data available.")

            st.markdown("---")

            # ── AMD Cycle (ICT) ───────────────────────────────────────────
            st.subheader("🔄 AMD Cycle (ICT)")
            _amd = getattr(r, "amd", None)
            if _amd is not None:
                # Phase badge
                _phase_colors = {
                    "accumulation": ("#1565c0", "🔵"),
                    "manipulation": ("#e65100", "🟠"),
                    "distribution": ("#2e7d32", "🟢"),
                    "unknown":      ("#555555", "⚪"),
                }
                _pc, _pe = _phase_colors.get(_amd.phase, ("#555555", "⚪"))
                st.markdown(
                    f"<span style='background:{_pc};color:#fff;padding:3px 10px;"
                    f"border-radius:4px;font-weight:700'>{_pe} {_amd.phase.upper()}</span>"
                    + (f"&nbsp;&nbsp;Direction: <b>{_amd.direction.upper()}</b>" if _amd.direction != "neutral" else "")
                    + f"&nbsp;&nbsp;Confidence: <b>{_amd.confidence:.0%}</b>",
                    unsafe_allow_html=True,
                )
                st.caption("")

                _amd_col1, _amd_col2, _amd_col3 = st.columns(3)
                with _amd_col1:
                    _asian_str = (f"{_amd.asian_low:.4f} – {_amd.asian_high:.4f}"
                                  if _amd.asian_high else "—")
                    st.metric("Asian Range", _asian_str,
                              delta=f"{_amd.asian_range:.1f} pts" if _amd.asian_range else None)
                with _amd_col2:
                    st.metric("Manipulation Sweep",
                              f"{'✅ ' + _amd.sweep_side.upper() if _amd.manipulation_swept else '⏳ Awaiting'}",
                              delta=f"@ {_amd.sweep_price:.4f}" if _amd.sweep_price else None)
                with _amd_col3:
                    st.metric("Distribution FVGs", len(_amd.entry_fvgs) if _amd.entry_fvgs else 0)

                if _amd.note:
                    st.info(_amd.note)

                if _amd.best_entry_fvg:
                    _efvg = _amd.best_entry_fvg
                    st.success(
                        f"**Best Entry FVG** ({_efvg.direction.upper()}): "
                        f"{_efvg.low:.4f} – {_efvg.high:.4f}  |  Mid: {_efvg.mid:.4f}  "
                        f"— retrace into this gap for distribution entry"
                    )
                if _amd.entry_fvgs and len(_amd.entry_fvgs) > 1:
                    with st.expander(f"All {len(_amd.entry_fvgs)} distribution FVGs"):
                        _fvg_rows = []
                        for _ef in _amd.entry_fvgs:
                            _fvg_rows.append({
                                "Direction": _ef.direction,
                                "Low":  f"{_ef.low:.4f}",
                                "High": f"{_ef.high:.4f}",
                                "Mid":  f"{_ef.mid:.4f}",
                                "Filled": _ef.filled,
                            })
                        st.dataframe(pd.DataFrame(_fvg_rows), use_container_width=True, hide_index=True)
            else:
                if st.session_state.get("last_df") is None:
                    st.warning(
                        "📂 **No OHLCV data loaded.**  "
                        "AMD cycle detection requires intraday candle data with a datetime index. "
                        "Upload a CSV with timestamps to enable this section."
                    )
                else:
                    st.info("AMD cycle data not available (datetime index required).")

            st.markdown("---")
            st.subheader("Fibonacci")
            if r.fib and r.fib.levels:
                if r.fib.current_level:
                    st.info(f"Current: **{r.fib.current_level.label}** @ {r.fib.current_level.price:.4f}")
                fib_data = []
                for lvl in r.fib.levels:
                    fib_data.append({
                        "Level":         lvl.label,
                        "Price":         f"{lvl.price:.4f}",
                        "Ratio":         f"{lvl.ratio:.3f}",
                        "Key":           "✅" if lvl.is_key else "",
                        "Reaction Score":f"{lvl.reaction_score:.2f}",
                    })
                st.dataframe(pd.DataFrame(fib_data), use_container_width=True, hide_index=True)
            elif st.session_state.get("last_df") is None:
                st.warning(
                    "📂 **No OHLCV data loaded.**  "
                    "Fibonacci retracement and extension levels are calculated from the "
                    "swing high and swing low of the candlestick data.  "
                    "Export from MT5: *F2 → History Center → XAUUSD → 4H → Export*, "
                    "then upload the CSV alongside your chart image."
                )
            else:
                st.info("Fibonacci levels could not be calculated from the available data.")


# ==========================================================================
# TAB 4 – ML Prediction
# ==========================================================================
with tabs[4]:
    r = st.session_state.result
    if r is None:
        st.info("Run an analysis first.")
    else:
        st.subheader("ML Setup Quality Prediction")
        if r.ml_prediction:
            mp = r.ml_prediction
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Quality Score",   f"{mp.quality_score:.1%}")
            with c2:
                st.metric("Win Probability", f"{mp.win_probability:.1%}")
            with c3:
                st.metric("Model Used", mp.model_used)

            if mp.feature_importances:
                with st.expander("Feature Importances"):
                    fi_data = sorted(mp.feature_importances.items(), key=lambda x: x[1], reverse=True)
                    fi_df = pd.DataFrame(fi_data, columns=["Feature", "Importance"])
                    st.bar_chart(fi_df.set_index("Feature"))
        else:
            st.info("No ML prediction available. The model needs labeled data to train.")

        st.markdown("---")
        st.subheader("Label Trade Outcome")
        st.caption("Feed real outcomes back to improve the ML model.")

        with st.form("label_form"):
            run_id_input  = st.text_input("Run ID", value=r.run_id if r else "")
            outcome_input = st.radio("Outcome", ["Win", "Loss"])
            rr_input      = st.number_input("R:R Achieved (optional)", min_value=0.0, max_value=50.0, value=0.0)
            label_btn     = st.form_submit_button("Record Outcome")

        if label_btn:
            engine = _load_engine()
            engine.label_outcome(
                run_id=run_id_input,
                outcome=1 if outcome_input == "Win" else 0,
                rr=rr_input if rr_input > 0 else None,
            )
            st.success("Outcome recorded. The model will retrain automatically when enough data is collected.")

        st.markdown("---")
        st.subheader("Model Stats")
        engine = _load_engine()
        learner = engine.learner
        st.json({
            "total_setups":  len(learner.records),
            "labeled":       sum(1 for r_ in learner.records if r_.outcome is not None),
            "model_trained": learner.model is not None,
        })


# ==========================================================================
# TAB 5 – Backtesting
# ==========================================================================
with tabs[5]:
    st.subheader("Walk-Forward Backtesting")
    df_bt = st.session_state.last_df
    if df_bt is None:
        st.info("Load OHLCV data first (Input & Analyze tab).")
    else:
        st.caption(f"Using {len(df_bt):,} bars of {asset} {timeframe} data.")
        col_bt1, col_bt2, col_bt3 = st.columns(3)
        with col_bt1:
            bt_capital   = st.number_input("Initial Capital ($)", value=10_000.0, step=1000.0)
        with col_bt2:
            bt_risk      = st.slider("Risk per Trade (%)", 0.5, 10.0, 1.0, step=0.5) / 100
        with col_bt3:
            bt_min_conf  = st.slider("Min Confluence Score", 40, 90, 65,
                                     help="65+ = high-quality setups only")
        col_bt4, col_bt5, col_bt6 = st.columns(3)
        with col_bt4:
            bt_max_bars = st.slider("Max bars to backtest", 200, min(len(df_bt), 1500), min(len(df_bt), 800), step=100,
                                    help="Fewer bars = much faster. 800 bars ≈ 17 days on M30.")
        with col_bt5:
            bt_stride = st.slider("Signal re-evaluate every N bars", 1, 20, 5,
                                  help="Higher = faster but less granular.")
        with col_bt6:
            bt_min_rr = st.slider("Min R:R ratio", 1.0, 4.0, 1.5, step=0.5,
                                   help="Reject trades below this reward-to-risk ratio.")

        col_bt7, _col_bt8 = st.columns(2)
        with col_bt7:
            bt_ml_floor = st.slider(
                "ML Quality Floor (0 = off)", 0.0, 0.85, 0.0, step=0.05,
                help="Skip entries where ML win probability < this. 0 = ML filter disabled. "
                     "Try 0.50–0.60 to see the impact of the improved features.",
            )

        if st.button("▶️ Run Backtest", type="primary"):
            from backtesting.backtester import Backtester, Signal as BTSignal

            df_bt_slice = df_bt.iloc[-bt_max_bars:].copy()
            engine = _load_engine()
            progress_bt = st.progress(0, text="Backtesting…")
            bt_strategy_errors: list[str] = []
            ml_skipped = [0]
            ml_signals = [0]
            sess_filtered = [0]
            conf_filtered = [0]
            trend_filtered = [0]
            regime_filtered = [0]

            def _bt_strategy(df_slice: pd.DataFrame):
                if len(df_slice) < 30:
                    return None
                try:
                    # Session filter: empirically-grounded session rules (all TFs).
                    # Open:    asian 00-08, london_open 08-10, london 10-12, ny_lunch 12-13
                    #          london_ny_overlap 13-17 (unblocked — highest directional volume)
                    # Blocked: ny_afternoon 17-20 (0W/28L),
                    #          dead_zone 20-24, weekends
                    try:
                        bar_ts = df_slice.index[-1]
                        if bar_ts.weekday() >= 5:   # Saturday / Sunday
                            sess_filtered[0] += 1
                            return None
                        bar_hour = int(bar_ts.hour)
                        # Blocked: ny_afternoon 17-20 only
                        if 17 <= bar_hour < 20:
                            sess_filtered[0] += 1
                            return None
                    except Exception:
                        pass

                    res = engine.analyze_data(df_slice, asset=asset, timeframe=timeframe,
                                              render_chart=False, save_to_db=False)
                    if not res.confluence or res.confluence.total < bt_min_conf:
                        conf_filtered[0] += 1
                        return None

                    direction = res.confluence.direction

                    # ── Trend filter: only trade in direction of market structure ──
                    if res.ms:
                        ms_dir = res.ms.structure_type.name.lower()
                        if ms_dir in ("bullish", "bearish") and ms_dir != direction:
                            trend_filtered[0] += 1
                            return None  # counter-trend — skip
                    if direction not in ("bullish", "bearish", "long", "short"):
                        trend_filtered[0] += 1
                        return None  # sideways — skip

                    price = float(df_slice["close"].iloc[-1])
                    atr   = float((df_slice["high"] - df_slice["low"]).rolling(14).mean().iloc[-1])
                    if not pd.notna(atr) or atr <= 0:
                        return None

                    # Regime filter: block compression and mean-reversion conditions.
                    try:
                        close_series = df_slice["close"]
                        rolling_mean = close_series.rolling(20).mean()
                        rolling_std = close_series.rolling(20).std()
                        bb_width = ((2 * rolling_std) / (rolling_mean + 1e-8)).dropna()
                        if len(bb_width) >= 30:
                            bb_now = float(bb_width.iloc[-1])
                            if bb_now <= float(bb_width.quantile(0.25)):
                                return None
                            closes_above_mean = int((close_series.tail(20) > rolling_mean.tail(20)).sum())
                            if bb_now <= float(bb_width.quantile(0.50)) and 7 <= closes_above_mean <= 13:
                                return None
                    except Exception:
                        pass

                    # ML quality gate
                    if bt_ml_floor > 0 and res.ml_prediction:
                        if res.ml_prediction.win_probability < bt_ml_floor:
                            ml_skipped[0] += 1
                            return None
                    ml_signals[0] += 1

                    is_long = direction in ("bullish", "long")

                    # ── S/R-based SL/TP ─────────────────────────────────────
                    sl_price = tp_price = None
                    if res.sr:
                        if is_long:
                            ns = res.sr.nearest_support
                            nr = res.sr.nearest_resistance
                            ns_lower = float(getattr(ns, "lower", getattr(ns, "level", price))) if ns else None
                            nr_level = float(getattr(nr, "level", price)) if nr else None
                            if ns_lower is not None and ns_lower < price:
                                sl_price = ns_lower - atr * 0.2   # buffer below zone
                            if nr_level is not None and nr_level > price:
                                tp_price = nr_level
                        else:
                            nr = res.sr.nearest_resistance
                            ns = res.sr.nearest_support
                            nr_upper = float(getattr(nr, "upper", getattr(nr, "level", price))) if nr else None
                            ns_level = float(getattr(ns, "level", price)) if ns else None
                            if nr_upper is not None and nr_upper > price:
                                sl_price = nr_upper + atr * 0.2
                            if ns_level is not None and ns_level < price:
                                tp_price = ns_level

                    # Fall back to ATR if S/R not available
                    if sl_price is None:
                        sl_price = price - 1.5 * atr if is_long else price + 1.5 * atr
                    if tp_price is None:
                        tp_price = price + 3.0 * atr if is_long else price - 3.0 * atr

                    return BTSignal(
                        direction="long" if is_long else "short",
                        entry_price=price,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        confidence=res.confluence.total / 100.0,
                        ml_score=res.ml_prediction.win_probability if res.ml_prediction else 0.0,
                    )
                except Exception as exc:
                    if len(bt_strategy_errors) < 5:
                        bt_strategy_errors.append(f"{type(exc).__name__}: {exc}")
                return None

            backtester = Backtester(initial_capital=bt_capital, risk_per_trade=bt_risk,
                                    signal_stride=bt_stride, min_rr=bt_min_rr,
                                    dual_tp=True)
            try:
                def _update_progress(done, total):
                    pct = int(done / total * 100) if total else 0
                    strat_calls = (total // bt_stride) or 1
                    progress_bt.progress(pct, text=f"Backtesting… {pct}%  (~{strat_calls} strategy calls)")

                bt_result = backtester.run(df_bt_slice, _bt_strategy,
                                           min_confidence=bt_min_conf / 100.0,
                                           progress_cb=_update_progress)
                st.session_state.bt_result = bt_result
                st.session_state.bt_strategy_errors = bt_strategy_errors
                st.session_state.bt_ml_signals = ml_signals[0]
                st.session_state.bt_ml_skipped = ml_skipped[0]
                st.session_state.bt_ml_floor_used = bt_ml_floor
                st.session_state.bt_funnel = {
                    "evaluated": ml_signals[0] + ml_skipped[0] + sess_filtered[0] + conf_filtered[0] + trend_filtered[0] + regime_filtered[0],
                    "session":   sess_filtered[0],
                    "confluence": conf_filtered[0],
                    "trend":     trend_filtered[0],
                    "regime":    regime_filtered[0],
                    "ml":        ml_skipped[0],
                    "passed":    ml_signals[0],
                    "traded":    bt_result.total_trades,
                }
                progress_bt.empty()
                st.toast("Backtest complete!", icon="✅")
                st.rerun()
            except Exception as e:
                progress_bt.empty()
                st.error(f"Backtest failed: {e}")

        bt = st.session_state.bt_result
        if bt:
            st.markdown("---")
            st.subheader("Results")
            _bt_errs = st.session_state.get("bt_strategy_errors", [])
            if _bt_errs:
                st.warning("Strategy warnings during backtest — first: " + _bt_errs[0])

            # Signal funnel summary
            _funnel = st.session_state.get("bt_funnel", {})
            if _funnel:
                _traded = _funnel.get("traded", 0)
                if _traded < 30:
                    st.warning(
                        f"⚠️ Only **{_traded} trades** executed — statistically insufficient "
                        f"(minimum 30 recommended). Lower the ML floor, confluence threshold, or increase bars."
                    )
                with st.expander("Signal Funnel — how many setups reached each filter", expanded=_traded < 20):
                    _fe = _funnel.get("evaluated", 0) or 1
                    _rows = [
                        ("Strategy evaluations",   _funnel.get("evaluated", 0), ""),
                        ("Session filtered out",   _funnel.get("session", 0),   f"{_funnel.get('session',0)/_fe:.0%}"),
                        ("Below confluence floor", _funnel.get("confluence", 0),f"{_funnel.get('confluence',0)/_fe:.0%}"),
                        ("Counter-trend / sideways",_funnel.get("trend", 0),   f"{_funnel.get('trend',0)/_fe:.0%}"),
                        ("Regime filtered out",    _funnel.get("regime", 0),    f"{_funnel.get('regime',0)/_fe:.0%}"),
                        ("ML rejected",            _funnel.get("ml", 0),        f"{_funnel.get('ml',0)/_fe:.0%}"),
                        ("Passed all filters",     _funnel.get("passed", 0),    f"{_funnel.get('passed',0)/_fe:.0%}"),
                        ("Actually traded",        _traded,                     f"{_traded/_fe:.0%}"),
                    ]
                    import pandas as _pd2
                    st.dataframe(
                        _pd2.DataFrame(_rows, columns=["Stage", "Count", "% of evaluated"]),
                        hide_index=True, use_container_width=True,
                    )

            m1, m2, m3, m4 = st.columns(4)
            with m1: st.metric("Total Trades",    bt.total_trades)
            with m2: st.metric("Win Rate",        f"{bt.win_rate:.1%}")
            with m3: st.metric("Profit Factor",   f"{bt.profit_factor:.2f}")
            with m4: st.metric("Net Return",      f"{bt.net_return_pct:.1f}%")

            m5, m6, m7, m8 = st.columns(4)
            with m5: st.metric("Max Drawdown",    f"{bt.max_drawdown_pct:.1%}")
            with m6: st.metric("Sharpe Ratio",    f"{bt.sharpe_ratio:.2f}")
            with m7: st.metric("Avg R:R",         f"{bt.avg_rr:.2f}")
            with m8: st.metric("Final Equity",    f"${bt.final_equity:,.2f}")

            if bt.narrative:
                st.caption(bt.narrative)

            # ML filter impact summary
            if bt_ml_floor > 0 or st.session_state.get("bt_ml_floor_used", 0) > 0:
                _ml_signals = st.session_state.get("bt_ml_signals", 0)
                _ml_skipped = st.session_state.get("bt_ml_skipped", 0)
                _ml_floor_used = st.session_state.get("bt_ml_floor_used", bt_ml_floor)
                _ml_total = _ml_signals + _ml_skipped
                if _ml_total > 0 and _ml_floor_used > 0:
                    _ml_pct = _ml_skipped / _ml_total * 100
                    _ml_a, _ml_b, _ml_c = st.columns(3)
                    _ml_a.metric("ML Evaluated", _ml_total)
                    _ml_b.metric("ML Rejected", _ml_skipped,
                                 help=f"Signals below {_ml_floor_used:.0%} ML quality floor")
                    _ml_c.metric("ML Accept Rate", f"{100 - _ml_pct:.0f}%")

            if bt.total_trades == 0:
                st.info("No trades were executed for the selected bars and filters. Try more bars, a lower confluence floor, or a London-heavy dataset.")

            # Equity curve
            if bt.equity_curve:
                eq_df = pd.DataFrame({"Equity": bt.equity_curve})
                st.line_chart(eq_df)

            # Trades table
            if bt.trades:
                with st.expander("Trade Log (last 50)"):
                    tlog = []
                    for t in bt.trades[-50:]:
                        tlog.append({
                            "Direction": t.direction,
                            "Entry":     f"{t.entry_price:.4f}",
                            "Exit":      f"{t.exit_price:.4f}" if t.exit_price else "open",
                            "PnL":       f"{t.pnl:+.4f}",
                            "R:R":       f"{t.rr_achieved:.2f}",
                            "ML Score":  f"{t.ml_score:.2f}" if t.ml_score > 0 else "—",
                            "Win":       "✅" if t.is_win else "❌",
                        })
                    st.dataframe(pd.DataFrame(tlog), use_container_width=True, hide_index=True)


# ==========================================================================
# TAB 6 – History
# ==========================================================================
with tabs[6]:
    st.subheader("Analysis History")

    # ── Filters + controls ────────────────────────────────────────────────
    col_flt1, col_flt2, col_flt3, col_flt4 = st.columns([2, 2, 1, 1])
    with col_flt1:
        hist_asset = st.text_input("Filter by asset", value="", key="hist_asset",
                                   placeholder="e.g. XAUUSD")
    with col_flt2:
        hist_tf = st.text_input("Filter by timeframe", value="", key="hist_tf",
                                placeholder="e.g. 4H")
    with col_flt3:
        hist_limit = st.number_input("Limit", min_value=5, max_value=500, value=50)
    with col_flt4:
        st.write("")
        st.write("")
        load_hist = st.button("🔄 Refresh", use_container_width=True)

    # Auto-load on first visit or on button click
    if load_hist or (not st.session_state.history):
        try:
            from storage.database import list_analyses, get_analysis, delete_analysis
            rows = list_analyses(
                asset=hist_asset or None,
                timeframe=hist_tf or None,
                limit=int(hist_limit),
            )
            st.session_state.history = rows
        except Exception as e:
            st.error(f"DB error: {e}")

    history = st.session_state.history

    if not history:
        st.info("No analysis history found. Run a full analysis from the **Analysis** tab — results are saved automatically.")
    else:
        # ── Summary stats ─────────────────────────────────────────────────
        hist_df_raw = pd.DataFrame(history)
        total = len(hist_df_raw)
        avg_conf = hist_df_raw["confluence_score"].dropna().mean()
        grade_counts = hist_df_raw["grade"].value_counts().to_dict()
        most_asset = hist_df_raw["asset"].value_counts().idxmax() if total else "—"

        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Total Analyses", total)
        s2.metric("Avg Confluence", f"{avg_conf:.1f}" if not pd.isna(avg_conf) else "—")
        s3.metric("Most Analyzed", most_asset)
        s4.metric("A-Grade Setups", grade_counts.get("A", 0))
        s5.metric("B-Grade Setups", grade_counts.get("B", 0))

        st.divider()

        # ── Grade badge + direction arrow helpers ─────────────────────────
        grade_color = {"A": "🟢", "B": "🔵", "C": "🟡", "D": "🔴", "F": "⛔"}
        dir_arrow   = {"bullish": "↑ Bull", "bearish": "↓ Bear",
                       "long": "↑ Long", "short": "↓ Short",
                       "sideways": "→ Side", "neutral": "→ Neutral"}

        # ── Export button ─────────────────────────────────────────────────
        csv_export = hist_df_raw.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export to CSV", csv_export,
                           file_name="prometheus_history.csv", mime="text/csv")

        st.write(f"**Showing {total} records** — click an ID to view full report")

        # ── Styled table ──────────────────────────────────────────────────
        for row in history:
            grade    = row.get("grade", "") or ""
            badge    = grade_color.get(grade, "⚪") + f" {grade}"
            direction = row.get("direction", "") or ""
            arrow    = dir_arrow.get(direction.lower(), f"→ {direction}")
            score    = row.get("confluence_score") or 0
            price    = row.get("current_price")
            sup      = row.get("nearest_support")
            res_     = row.get("nearest_resistance")
            struct   = row.get("structure", "") or ""
            ts       = str(row.get("created_at", ""))[:16]

            header = (f"#{row['id']} | {row['asset']} {row['timeframe']} | "
                      f"{ts} | {badge} | Score: {score:.0f} | {arrow}")

            with st.expander(header, expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Price",     f"{price:.4f}" if price else "—")
                c2.metric("Support",   f"{sup:.4f}"  if sup  else "—")
                c3.metric("Resistance",f"{res_:.4f}" if res_ else "—")
                c4.metric("Structure", struct.capitalize() if struct else "—")

                # Load full report on demand
                col_view, col_del = st.columns([3, 1])
                with col_view:
                    if st.button("📄 Load Full Report", key=f"hist_view_{row['id']}"):
                        try:
                            from storage.database import get_analysis
                            full = get_analysis(row["id"])
                            if full and full.get("report_json"):
                                import json as _json
                                rdata = _json.loads(full["report_json"])
                                full_text = rdata.get("full_text", "No report text stored.")
                                st.markdown("**AI Report:**")
                                st.text(full_text)
                                key_levels = full.get("key_levels_json")
                                if key_levels:
                                    try:
                                        kl = _json.loads(key_levels)
                                        if kl:
                                            st.markdown("**Key Levels:**")
                                            st.json(kl)
                                    except Exception:
                                        pass
                            else:
                                st.info("No detailed report stored for this record.")
                        except Exception as _ex:
                            st.error(f"Could not load report: {_ex}")
                with col_del:
                    if st.button("🗑️ Delete", key=f"hist_del_{row['id']}",
                                 type="secondary"):
                        try:
                            from storage.database import delete_analysis
                            if delete_analysis(row["id"]):
                                st.success(f"Record #{row['id']} deleted.")
                                # Refresh list
                                from storage.database import list_analyses
                                st.session_state.history = list_analyses(
                                    asset=hist_asset or None,
                                    timeframe=hist_tf or None,
                                    limit=int(hist_limit),
                                )
                                st.rerun()
                        except Exception as _ex:
                            st.error(f"Delete failed: {_ex}")


# TAB 7 – Live Bot
# ==========================================================================
with tabs[7]:
    import subprocess as _sp
    import json as _json
    from pathlib import Path as _Path
    from datetime import datetime as _dt, timezone as _tz

    # ── Auto-refresh ─────────────────────────────────────────────────────────
    from streamlit_autorefresh import st_autorefresh as _st_autorefresh
    if _live_refresh_s > 0:
        _st_autorefresh(interval=_live_refresh_s * 1000, key="live_bot_autorefresh")

    _BOT_DIR       = _Path(__file__).parent.parent / "live_bot"
    _STATUS_F      = _BOT_DIR / "bot_status.json"
    _STOP_F        = _BOT_DIR / "stop_flag"          # must match STOP_FLAG in trader.py
    _WATCHDOG_STOP = _BOT_DIR / "watchdog_stop"      # tells watchdog not to restart
    _VENV_PY       = _Path(r"c:\Users\Chaba\Documents\tradingBots\.venv\Scripts\python.exe")
    _TRADER_PY     = _BOT_DIR / "trader.py"
    _WATCHDOG_PY   = _BOT_DIR / "watchdog.py"

    st.subheader("🟢 Live Bot — Prometheus Auto-Trader")
    st.caption(
        "The bot runs as a **separate process**. It polls the analysis DB every N seconds "
        "and executes trades on MetaTrader 5 when a fresh signal meets your thresholds."
    )

    # ── Bot configuration form ───────────────────────────────────────────────
    with st.expander("⚙️ Bot Configuration", expanded=True):
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            bot_asset    = st.text_input("Asset", value=asset, key="bot_asset")
            bot_grade    = st.selectbox("Min Grade", ["A", "B", "C", "D"], index=1, key="bot_grade")
        with bc2:
            bot_tf       = st.text_input("Timeframe", value=timeframe, key="bot_tf")
            bot_score    = st.slider("Min Confluence Score", 40, 95, 70, key="bot_score")
        with bc3:
            bot_risk     = st.slider("Risk % per Trade", 0.5, 10.0, 1.0, step=0.5, key="bot_risk")
            bot_poll     = st.number_input("Poll Interval (sec)", min_value=10, max_value=3600,
                                           value=60, key="bot_poll")
        # Reflect the running bot's actual live/dry-run state in the checkbox
        _running_dry = True
        _running_entry_mode = "zone_only"
        if _STATUS_F.exists():
            try:
                _status_cfg = _json.loads(_STATUS_F.read_text(encoding="utf-8"))
                _running_dry        = _status_cfg.get("dry_run", True)
                _running_entry_mode = _status_cfg.get("entry_mode", "zone_only")
            except Exception:
                pass
        bot_live = st.checkbox(
            "⚡ Enable LIVE MT5 orders (uncheck = dry-run only)",
            value=not _running_dry, key="bot_live",
        )
        if bot_live:
            st.warning(
                "LIVE mode will send real orders to MetaTrader 5. "
                "Ensure MT5 is open, logged in, and algorithmic trading is enabled. "
                "Only enable after testing in dry-run mode."
            )
        bot_entry_mode = st.radio(
            "Entry Mode",
            options=["market_any", "zone_only"],
            index=1 if _running_entry_mode == "zone_only" else 0,
            horizontal=True,
            key="bot_entry_mode",
            help=(
                "🟠 **market_any** — enter at market whenever score qualifies (more trades, uses S/R for SL). "
                "🔵 **zone_only** — only enter when price is at/near a fresh OB or S/R zone (precise, fewer trades)."
            ),
        )
        # Write both entry_mode and live_mode to control file every render
        # so the running bot picks up changes on its next poll
        _CTRL_F = _BOT_DIR / "bot_control.json"
        _CTRL_F.write_text(
            _json.dumps({"entry_mode": bot_entry_mode, "live_mode": bot_live}, indent=2),
            encoding="utf-8",
        )
    _live_flag = "--live" if bot_live else ""
    _cmd_parts = [
        f'"{_VENV_PY}"',
        f'"{_WATCHDOG_PY}"',
        f"--asset {bot_asset}",
        f"--tf {bot_tf}",
        f"--min-grade {bot_grade}",
        f"--min-score {bot_score}",
        f"--risk {bot_risk}",
        f"--poll {bot_poll}",
        "--candles 500",
    ]
    if bot_live:
        _cmd_parts.append("--live")
    _cmd_parts.append(f"--entry-mode {bot_entry_mode}")
    _full_cmd = " ".join(_cmd_parts)

    st.markdown("**Run this command in a terminal to start the bot (with auto-restart watchdog):**")
    st.code(_full_cmd, language="bash")

    # ── Start / Stop buttons ─────────────────────────────────────────────────
    sb1, sb2, sb3 = st.columns([2, 2, 4])
    with sb1:
        if st.button("▶️ Start Bot", type="primary", use_container_width=True):
            # Guard: check whether lock port 47820 is already bound (real running bot)
            import socket as _sock
            _already_running = False
            try:
                _chk = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
                _chk.setsockopt(_sock.SOL_SOCKET, _sock.SO_REUSEADDR, 0)
                _chk.bind(("127.0.0.1", 47820))
                _chk.close()
                # bind succeeded → port free → no bot running
            except OSError:
                _already_running = True   # port in use → bot already running
            except Exception:
                pass

            # Also block if watchdog process is alive (restarting after a crash)
            if not _already_running:
                _wdog_pid_f = _BOT_DIR / "watchdog.pid"
                if _wdog_pid_f.exists():
                    try:
                        import os as _os
                        _wdog_pid = int(_wdog_pid_f.read_text().strip())
                        _os.kill(_wdog_pid, 0)   # raises if dead
                        _already_running = True
                    except (OSError, ValueError):
                        pass  # stale watchdog.pid — safe to start

            if _already_running:
                st.warning("⚠️ Bot (or watchdog) is already running. Stop it first before starting a new instance.")
            else:
                try:
                    _STOP_F.unlink(missing_ok=True)
                    _WATCHDOG_STOP.unlink(missing_ok=True)
                    _sp.Popen(
                        [str(_VENV_PY), str(_WATCHDOG_PY),
                         "--asset", bot_asset, "--tf", bot_tf,
                         "--min-grade", bot_grade,
                         "--min-score", str(bot_score),
                         "--risk", str(bot_risk),
                         "--poll", str(bot_poll),
                         "--candles", "500",
                         "--entry-mode", bot_entry_mode]
                        + (["--live"] if bot_live else []),
                        creationflags=_sp.CREATE_NEW_CONSOLE,
                        cwd=str(_BOT_DIR.parent),
                    )
                    st.success("Bot launched via watchdog — it will auto-restart on any crash.")
                    import time as _time; _time.sleep(2)
                    st.rerun()
                except Exception as _ex:
                    st.error(f"Could not start bot: {_ex}")
    with sb2:
        if st.button("⏹️ Stop Bot", use_container_width=True):
            # Write both flags: stop_flag tells trader.py to exit cleanly;
            # watchdog_stop tells the watchdog not to restart it.
            _STOP_F.write_text("stop", encoding="utf-8")
            _WATCHDOG_STOP.write_text("stop", encoding="utf-8")
            st.info("Stop signal sent — the bot and watchdog will exit after the current poll.")

    st.divider()

    # ── Status card ──────────────────────────────────────────────────────────
    st.subheader("Bot Status")
    if _STATUS_F.exists():
        try:
            _s       = _json.loads(_STATUS_F.read_text(encoding="utf-8"))
            _poll_ts = _s.get("last_poll", "")
            _age_sec = None
            if _poll_ts:
                try:
                    _lp    = _dt.fromisoformat(_poll_ts)
                    _age_sec = (_dt.utcnow() - _lp).total_seconds()
                except Exception:
                    pass

            _online = _age_sec is not None and _age_sec < _s.get("poll_interval", 60) * 2.5
            _badge  = "🟢 **ONLINE**" if _online else "🔴 **OFFLINE / STALE**"

            # ── Circuit breaker banner ─────────────────────────────────────────
            if _s.get("trading_halted"):
                st.error(
                    f"🚨 **TRADING HALTED** — {_s.get('halt_reason', 'Circuit breaker triggered.')}"
                    f"  Existing positions are still being managed.",
                    icon="🚨",
                )

            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            sc1.metric("Status",      "Online" if _online else "Offline")
            sc2.metric("Mode",        "DRY RUN" if _s.get("dry_run") else "🔴 LIVE")
            sc3.metric("Asset",       f"{_s.get('asset','?')} {_s.get('timeframe','?')}")
            sc4.metric("Trades Sent", str(_s.get("total_trades", 0)))
            sc5.metric("Open Now",    str(_s.get("open_count", 0)), delta=f"Version {_PROMETHEUS_VERSION}", delta_color="off")

            _em = _s.get("entry_mode", "zone_only")
            _em_colour = "#f59e0b" if _em == "market_any" else "#60a5fa"
            _em_label  = "🟠 market_any (enters at market)" if _em == "market_any" else "🔵 zone_only (waits for zone)"
            st.markdown(
                f"<span style='background:{_em_colour}22;border:1px solid {_em_colour};"
                f"border-radius:4px;padding:3px 10px;font-size:.85rem;color:{_em_colour}'>"
                f"⚪ Entry Mode: <b>{_em_label}</b></span>",
                unsafe_allow_html=True,
            )

            if _s.get("balance"):
                sa1, sa2, sa3, sa4 = st.columns(4)
                sa1.metric("Balance",      f"${_s['balance']:,.2f}")
                sa2.metric("Equity",       f"${_s.get('equity',0):,.2f}")
                sa3.metric("Free Margin",  f"${_s.get('margin_free',0):,.2f}")
                _unreal = _s.get("total_unrealised", 0.0) or 0.0
                sa4.metric("Unrealised P&L", f"${_unreal:+,.2f}",
                           delta=f"${_unreal:+,.2f}",
                           delta_color="normal" if _unreal >= 0 else "inverse")

            st.markdown("### Institutional Intelligence Status")
            _registry_sources = {
                "prometheus_evolution": _EVO_PAYLOAD,
                "zeus_validation": _ZEUS_VALIDATION,
                "institutional_risk_performance_runtime": _INSTITUTIONAL_RUNTIME,
            }
            render_registry_metrics(dashboard="prometheus_market_analysis", section="execution", sources=_registry_sources, columns_count=6)
            render_registry_metrics(dashboard="prometheus_market_analysis", section="risk", sources=_registry_sources, columns_count=5)
            render_registry_metrics(dashboard="prometheus_market_analysis", section="research", sources=_registry_sources, columns_count=5)
            render_registry_metrics(dashboard="prometheus_market_analysis", section="edge", sources=_registry_sources, columns_count=6)
            render_registry_texts(dashboard="prometheus_market_analysis", section="research", sources=_registry_sources)

            st.markdown(f"**{_badge}** · Last poll: {_poll_ts[:19].replace('T',' ')} UTC")
            if _age_sec is not None:
                st.caption(f"Poll age: {int(_age_sec)}s (expected ≤{_s.get('poll_interval',60)}s)")

            _open = _s.get("open_positions", [])
            _profit_floor_cfg = 15.0
            if _open and _open[0].get("time_exit_profit_min_usd") is not None:
                try:
                    _profit_floor_cfg = float(_open[0].get("time_exit_profit_min_usd"))
                except Exception:
                    _profit_floor_cfg = 15.0

            _open_enriched = []
            for _p in _open:
                _row = dict(_p)
                _open_m = _row.get("open_minutes")
                if _open_m is None and _row.get("open_since"):
                    try:
                        _opened = _dt.fromisoformat(str(_row.get("open_since")))
                        _open_m = (_dt.utcnow() - _opened).total_seconds() / 60.0
                    except Exception:
                        _open_m = None
                if _open_m is not None:
                    _open_m = float(_open_m)
                    _row["open_minutes"] = round(_open_m, 2)

                _smart_min = float(_row.get("time_exit_smart_minutes", 15.0) or 15.0)
                _hard_min = float(_row.get("time_exit_hard_minutes", 30.0) or 30.0)
                _unr = float(_row.get("unrealised", 0.0) or 0.0)

                if _open_m is not None:
                    _row.setdefault("time_to_smart_min", round(max(0.0, _smart_min - _open_m), 2))
                    _row.setdefault("time_to_hard_min", round(max(0.0, _hard_min - _open_m), 2))
                    _row.setdefault(
                        "time_exit_smart_eligible",
                        bool(_open_m >= _smart_min and _unr >= _profit_floor_cfg),
                    )
                    _row.setdefault(
                        "time_exit_hard_due",
                        bool(_open_m >= _hard_min and _unr >= _profit_floor_cfg),
                    )
                else:
                    _row.setdefault("time_to_smart_min", None)
                    _row.setdefault("time_to_hard_min", None)
                    _row.setdefault("time_exit_smart_eligible", False)
                    _row.setdefault("time_exit_hard_due", False)

                _row.setdefault("time_exit_profit_min_usd", _profit_floor_cfg)
                _row.setdefault("time_exit_profit_gap_usd", round(max(0.0, _profit_floor_cfg - _unr), 2))
                _row.setdefault("time_smart_partial_done", False)
                _open_enriched.append(_row)

            _open = _open_enriched
            _smart_ready = sum(1 for _p in _open if bool(_p.get("time_exit_smart_eligible", False)))
            _hard_due = sum(1 for _p in _open if bool(_p.get("time_exit_hard_due", False)))
            _total_open = len(_open)
            _profit_floor = (_open[0].get("time_exit_profit_min_usd") if _open else None)
            _profit_floor_lbl = f"${float(_profit_floor):.0f}" if _profit_floor is not None else "$15"
            st.markdown(
                f"<div style='display:flex;flex-wrap:wrap;gap:8px;margin-top:2px'>"
                f"<span style='background:#0f172a;border:1px solid #38bdf8;color:#38bdf8;border-radius:999px;padding:2px 10px;font-size:.78rem'>"
                f"⏱ Smart Ready: <b>{_smart_ready}/{_total_open}</b></span>"
                f"<span style='background:#1f2937;border:1px solid #f59e0b;color:#f59e0b;border-radius:999px;padding:2px 10px;font-size:.78rem'>"
                f"⌛ Hard Due: <b>{_hard_due}</b></span>"
                f"<span style='background:#111827;border:1px solid #10b981;color:#10b981;border-radius:999px;padding:2px 10px;font-size:.78rem'>"
                f"💵 Timeout Floor: <b>{_profit_floor_lbl}</b></span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # ── Open Positions table ─────────────────────────────────────────
            st.markdown(f"#### 📂 Open Positions ({len(_open)})")
            if _open:
                _op_df = pd.DataFrame(_open)
                _ordered_cols = [
                    "ticket", "direction", "lots", "entry", "current", "unrealised",
                    "open_minutes", "time_to_smart_min", "time_to_hard_min",
                    "time_exit_profit_gap_usd", "entry_regime", "time_exit_smart_eligible",
                    "time_smart_partial_done", "time_exit_hard_due", "sl", "tp", "comment",
                ]
                _show_cols = [c for c in _ordered_cols if c in _op_df.columns]
                if _show_cols:
                    _op_df = _op_df[_show_cols]

                # colour unrealised column
                def _color_pnl(val):
                    color = "#22c55e" if val >= 0 else "#ef4444"
                    return f"color: {color}; font-weight: bold"
                _styled = (
                    _op_df.style
                    .map(_color_pnl, subset=["unrealised"])
                    .format({
                        "entry": "{:.4f}", "sl": "{:.4f}", "tp": "{:.4f}",
                        "current": "{:.4f}", "unrealised": "${:+.2f}", "lots": "{:.2f}",
                        "open_minutes": "{:.1f}m",
                        "time_to_smart_min": "{:.1f}m",
                        "time_to_hard_min": "{:.1f}m",
                        "time_exit_profit_gap_usd": "${:.2f}",
                    })
                )
                st.dataframe(_styled, use_container_width=True, hide_index=True)

                st.caption(
                    "Timeout harvest: smart exits can trigger once time_to_smart hits 0 and profit gap reaches $0. "
                    "Hard timeout triggers at time_to_hard=0 if profit floor is met. "
                    "This allows banking exceptional P&L before TP1 while still taking continuation re-entries on new OBs."
                )

                # ── 5M reversal exit events ──────────────────────────────
                _m5_exits = _s.get("m5_exit_events", [])
                if _m5_exits:
                    st.markdown(
                        "<div style='background:#1e1e1e;border-left:3px solid #f59e0b;"
                        "padding:8px 12px;border-radius:4px;margin-top:6px'>"
                        "<span style='color:#f59e0b;font-weight:600'>⚡ 5M Reversal Exit</span>&nbsp;&nbsp;"
                        + "&nbsp;·&nbsp;".join(
                            f"<span style='color:#d4d4d4;font-size:.85rem'>{e}</span>"
                            for e in _m5_exits
                        )
                        + "</div>",
                        unsafe_allow_html=True,
                    )

                # ── Trailing stop events ─────────────────────────────────
                _trail_evt = _s.get("trail_events", [])
                _last_trail = _s.get("last_trail_action")
                if _last_trail or _trail_evt:
                    _trail_lines = _trail_evt or ([_last_trail] if _last_trail else [])
                    st.markdown(
                        "<div style='background:#1e1e1e;border-left:3px solid #a78bfa;"
                        "padding:8px 12px;border-radius:4px;margin-top:6px'>"
                        "<span style='color:#a78bfa;font-weight:600'>🎯 Trailing SL</span>&nbsp;&nbsp;"
                        + "&nbsp;·&nbsp;".join(
                            f"<span style='color:#d4d4d4;font-size:.85rem'>{e}</span>"
                            for e in _trail_lines
                        )
                        + "</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No open positions right now." if _online else
                        "Bot offline — open positions not available.")

            # ── Adaptive Learning panel ──────────────────────────────────────
            _lrn = _s.get("learning", {})
            if _lrn:
                with st.expander("🧠 Adaptive Learning Stats", expanded=False):
                    lc1, lc2, lc3, lc4 = st.columns(4)
                    _wins    = _lrn.get("wins", 0)
                    _losses  = _lrn.get("losses", 0)
                    _seen    = _lrn.get("total_seen", 0)
                    _adj     = _lrn.get("score_adjust", 0.0)
                    _total   = _wins + _losses
                    _wr      = (_wins / _total * 100) if _total else 0.0
                    lc1.metric("Wins",   _wins)
                    lc2.metric("Losses", _losses)
                    lc3.metric("Win Rate", f"{_wr:.0f}%" if _total else "—")
                    lc4.metric("Score Threshold Adj", f"{_adj:+.1f}",
                               help="Negative = bot relaxed from good performance; "
                                    "Positive = bot raised bar after losses")
                    _eff = _s.get("effective_min_score") or (
                        _s.get("min_score", 70) + _adj)
                    st.caption(
                        f"Effective min score: **{_eff:.0f}** "
                        f"(base {_s.get('min_score',70):.0f} {_adj:+.1f} learning adj) · "
                        f"Signals seen: {_seen}"
                    )

                    # Grade breakdown
                    _gs = _lrn.get("grade_stats", {})
                    if _gs:
                        st.markdown("**Grade breakdown** (signals seen vs acted on):")
                        _gs_rows = [{"Grade": g, "Seen": v.get("seen",0),
                                     "Acted": v.get("acted",0)}
                                    for g, v in sorted(_gs.items())]
                        st.dataframe(pd.DataFrame(_gs_rows), hide_index=True,
                                     use_container_width=False)

                    # Direction breakdown
                    _ds = _lrn.get("direction_stats", {})
                    if _ds:
                        st.markdown("**Direction W/L** (from closed MT5 deals):")
                        _ds_rows = [{"Direction": d, "Wins": v.get("wins",0),
                                     "Losses": v.get("losses",0)}
                                    for d, v in _ds.items()]
                        st.dataframe(pd.DataFrame(_ds_rows), hide_index=True,
                                     use_container_width=False)

                    # OB (Order Block) hit/win rate by direction
                    _obs = _lrn.get("ob_stats", {})
                    if _obs:
                        st.markdown("**Order Block entry stats** (hits & win rate per OB direction):")
                        _ob_rows = []
                        for ob_dir, ob_v in _obs.items():
                            hits = ob_v.get("hits", 0)
                            wins = ob_v.get("wins", 0)
                            wr   = f"{wins/hits*100:.0f}%" if hits else "—"
                            _ob_rows.append({"OB Direction": ob_dir.title(), "Entries": hits,
                                             "Wins": wins, "Win Rate": wr})
                        st.dataframe(pd.DataFrame(_ob_rows), hide_index=True,
                                     use_container_width=False)

                    # LTF (1M+5M) alignment win rate
                    _ltf_s = _lrn.get("ltf_stats", {})
                    if _ltf_s:
                        st.markdown("**1M+5M LTF alignment win rate** (tracks which entry timing wins):")
                        _ltf_label = {
                            "both_confirmed": "✅ Both 1M+5M confirmed",
                            "one_counter":    "⚠️ One LTF counter-trend",
                            "unknown":        "❓ LTF data unavailable",
                            "trap":           "🚫 Momentum trap (blocked)",
                        }
                        _ltf_rows = []
                        for k, v in _ltf_s.items():
                            w = v.get("wins", 0); l = v.get("losses", 0)
                            _ltf_rows.append({
                                "LTF State":  _ltf_label.get(k, k),
                                "Wins":       w,
                                "Losses":     l,
                                "Win Rate":   f"{w/(w+l)*100:.0f}%" if (w+l) else "—",
                            })
                        st.dataframe(pd.DataFrame(_ltf_rows), hide_index=True,
                                     use_container_width=False)

                    # Timeout-exit effectiveness
                    _tes = _lrn.get("time_exit_stats", {})
                    if _tes:
                        st.markdown("**⏱ Timeout harvest stats** (smart vs hard exits):")
                        _te_rows = []
                        for _k, _lbl in (("time_smart", "Smart timeout"), ("time_hard", "Hard timeout")):
                            _v = _tes.get(_k, {}) or {}
                            _cnt = int(_v.get("count", 0) or 0)
                            _w = int(_v.get("wins", 0) or 0)
                            _p = float(_v.get("pnl", 0.0) or 0.0)
                            _te_rows.append({
                                "Mode": _lbl,
                                "Count": _cnt,
                                "Wins": _w,
                                "Win Rate": f"{(_w / _cnt * 100):.0f}%" if _cnt else "—",
                                "P&L $": f"{_p:+.2f}",
                            })
                        st.dataframe(pd.DataFrame(_te_rows), hide_index=True,
                                     use_container_width=False)

                    # PnL sparkline (unrealised history)
                    _pnl_hist = _lrn.get("pnl_history", [])
                    if len(_pnl_hist) > 1:
                        import plotly.graph_objects as _go
                        _fig_pnl = _go.Figure(
                            _go.Scatter(y=_pnl_hist, mode="lines+markers",
                                        line=dict(color="#22c55e" if _pnl_hist[-1] >= 0
                                                  else "#ef4444", width=2),
                                        fill="tozeroy")
                        )
                        _fig_pnl.update_layout(
                            title="Unrealised P&L history (last 20 polls)",
                            height=200, margin=dict(l=0, r=0, t=30, b=0),
                            yaxis_title="$", xaxis_title="Poll #",
                        )
                        st.plotly_chart(_fig_pnl, use_container_width=True)

                    # Closed trade analytics from DB
                    try:
                        from storage.database import list_trades as _lt_lrn
                        _all_trades = _lt_lrn(asset=None, source="live", limit=200)
                        _closed_trades = [t for t in _all_trades
                                          if t.get("status") in ("win", "loss", "closed")
                                          and t.get("pnl") is not None]
                        if _closed_trades:
                            import plotly.graph_objects as _pgo
                            _cpnl = [t["pnl"] for t in _closed_trades]
                            _cum  = []
                            _s_acc = 0.0
                            for _v in _cpnl:
                                _s_acc += _v
                                _cum.append(round(_s_acc, 2))
                            _cum_color = "#22c55e" if _cum[-1] >= 0 else "#ef4444"
                            _fig_cum = _pgo.Figure(
                                _pgo.Scatter(
                                    y=_cum, mode="lines+markers",
                                    line=dict(color=_cum_color, width=2),
                                    fill="tozeroy",
                                    hovertemplate="Trade %{x}: $%{y:+.2f}<extra></extra>",
                                )
                            )
                            _fig_cum.add_hline(y=0, line_color="#555", line_width=1)
                            _fig_cum.update_layout(
                                title=f"Cumulative closed P&L — {len(_closed_trades)} trades (${_cum[-1]:+.2f} total)",
                                height=220, margin=dict(l=0, r=0, t=35, b=0),
                                yaxis_title="$ cumulative", xaxis_title="Trade #",
                                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                                font=dict(color="#ccc"),
                            )
                            st.plotly_chart(_fig_cum, use_container_width=True)

                            # Score vs PnL scatter
                            _scores = [t.get("score_at_entry") for t in _closed_trades]
                            _pnls   = [t.get("pnl") for t in _closed_trades]
                            _dirs   = [t.get("direction", "?") for t in _closed_trades]
                            _sc_pnl_pairs = [(s, p, d) for s, p, d in zip(_scores, _pnls, _dirs) if s is not None]
                            if _sc_pnl_pairs:
                                _xs, _ys, _ds = zip(*_sc_pnl_pairs)
                                _dot_colors = ["#22c55e" if p >= 0 else "#ef4444" for p in _ys]
                                _fig_sc = _pgo.Figure(
                                    _pgo.Scatter(
                                        x=_xs, y=_ys, mode="markers",
                                        marker=dict(color=_dot_colors, size=9, opacity=0.8,
                                                    line=dict(width=1, color="#555")),
                                        text=[f"Score:{s:.0f} PnL:${p:+.2f} {d}" for s, p, d in _sc_pnl_pairs],
                                        hovertemplate="%{text}<extra></extra>",
                                    )
                                )
                                _fig_sc.add_hline(y=0, line_color="#555", line_width=1)
                                _fig_sc.update_layout(
                                    title="Signal score vs closed P&L",
                                    height=220, margin=dict(l=0, r=0, t=35, b=0),
                                    xaxis_title="Score at entry", yaxis_title="P&L $",
                                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                                    font=dict(color="#ccc"),
                                )
                                st.plotly_chart(_fig_sc, use_container_width=True)

                            # Regime & session win rate breakdown
                            # Start from accumulated learning state, then layer in DB trades
                            _lrn_live = _s.get("learning") or {}
                            _reg_stats: dict = {k: dict(v) for k, v in _lrn_live.get("regime_stats", {}).items()}
                            _ses_stats: dict = {k: dict(v) for k, v in _lrn_live.get("session_stats", {}).items()}
                            for _t in _closed_trades:
                                _reg = _t.get("regime") or "unknown"
                                _ses = _t.get("session") or "unknown"
                                _won = _t.get("pnl", 0) >= 0
                                for _key, _d in ((_reg, _reg_stats), (_ses, _ses_stats)):
                                    if _key not in _d:
                                        _d[_key] = {"wins": 0, "losses": 0, "pnl": 0.0}
                                    _d[_key]["wins" if _won else "losses"] += 1
                                    _d[_key]["pnl"] = round(_d[_key].get("pnl", 0.0) + _t.get("pnl", 0), 2)
                            if any(k != "unknown" for k in _reg_stats):
                                st.markdown("**Win rate by market regime:**")
                                _rg_rows = []
                                for _rk, _rv in sorted(_reg_stats.items()):
                                    _rw = _rv["wins"]; _rl = _rv["losses"]
                                    _rg_rows.append({
                                        "Regime": _rk.replace("_", " ").title(),
                                        "Wins": _rw, "Losses": _rl,
                                        "Win Rate": f"{_rw/(_rw+_rl)*100:.0f}%" if (_rw+_rl) else "—",
                                        "P&L $": f"{_rv['pnl']:+.2f}",
                                    })
                                st.dataframe(pd.DataFrame(_rg_rows), hide_index=True, use_container_width=False)
                            if any(k != "unknown" for k in _ses_stats):
                                st.markdown("**Win rate by trading session:**")
                                _ss_rows = []
                                for _sk, _sv in sorted(_ses_stats.items()):
                                    _sw = _sv["wins"]; _sl = _sv["losses"]
                                    _ss_rows.append({
                                        "Session": _sk.replace("_", " ").title(),
                                        "Wins": _sw, "Losses": _sl,
                                        "Win Rate": f"{_sw/(_sw+_sl)*100:.0f}%" if (_sw+_sl) else "—",
                                        "P&L $": f"{_sv['pnl']:+.2f}",
                                    })
                                st.dataframe(pd.DataFrame(_ss_rows), hide_index=True, use_container_width=False)
                    except Exception:
                        pass

            # ── Pending limit orders ─────────────────────────────────────────
            _plims = _s.get("pending_limits", {})
            if _plims:
                with st.expander(f"⏳ Pending Limit Orders ({len(_plims)})", expanded=True):
                    _plim_rows = [{"Ticket": t, "Polls Remaining": p}
                                  for t, p in _plims.items()]
                    st.dataframe(pd.DataFrame(_plim_rows), hide_index=True,
                                 use_container_width=True)
                    st.caption("Limit orders are auto-cancelled after "
                               f"{_plims and max(_plims.values()) or '?'} polls if not filled.")

            with st.expander("Last signal / action"):
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                col_s1.metric("Grade",     _s.get("last_signal_grade", "—"))
                col_s2.metric("Direction", _s.get("last_signal_direction", "—"))
                col_s3.metric("Score",     f"{_s.get('last_signal_score', 0):.0f}")
                col_s4.metric("Eff. Min",  f"{_s.get('effective_min_score', _s.get('min_score',70)):.0f}")
                st.info(_s.get("last_action", "—"))

            # ── Architecture Engines Status ──────────────────────────────────
            _regime_d  = _s.get("regime", {})
            _sess_d    = _s.get("session_detail", {})
            _eq        = _s.get("last_exec_quality", {})
            _htf_mult  = _s.get("last_htf_lot_mult")
            if _regime_d or _sess_d or _eq or _htf_mult is not None:
                with st.expander("⚙️ Architecture Engine Status", expanded=True):
                    _ae1, _ae2, _ae3, _ae4, _ae5 = st.columns(5)

                    # Regime
                    if _regime_d:
                        _rname = _regime_d.get("name", "UNKNOWN").replace("_", " ").title()
                        _rconf = _regime_d.get("confidence", 0)
                        _rlot  = _regime_d.get("lot_scalar", 1.0)
                        _rbe   = _regime_d.get("be_atr_mult", 0.5)
                        _rtp   = _regime_d.get("tp_scalar", 1.0)
                        _rct   = _regime_d.get("allow_countertrend", False)
                        _regime_color = (
                            "#22c55e" if "Trend Expansion" in _rname
                            else "#f59e0b" if "Compression" in _rname
                            else "#60a5fa" if "Mean Reversion" in _rname
                            else "#ef4444" if "Volatility" in _rname
                            else "#a78bfa"
                        )
                        with _ae1:
                            st.markdown(
                                f"<div style='background:#1e1e1e;border-left:3px solid {_regime_color};"
                                f"border-radius:4px;padding:8px 12px;margin-bottom:8px'>"
                                f"<div style='color:#aaa;font-size:.7rem;text-transform:uppercase'>Market Regime</div>"
                                f"<div style='font-size:1rem;font-weight:700;color:{_regime_color}'>{_rname}</div>"
                                f"<div style='color:#aaa;font-size:.75rem'>Conf: {_rconf:.0%} &nbsp;·&nbsp; "
                                f"Lot×{_rlot:.2f} &nbsp;·&nbsp; TP×{_rtp:.2f}</div>"
                                f"<div style='color:#aaa;font-size:.75rem'>BE mult: {_rbe:.2f} &nbsp;·&nbsp; "
                                f"Counter: {'✅' if _rct else '❌'}</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        _ae1.caption("Regime engine N/A")

                    # Session
                    if _sess_d:
                        _sname   = _sess_d.get("name", "unknown").replace("_", " ").title()
                        _stol    = _sess_d.get("spread_tolerance", 1.0)
                        _sdead   = _sess_d.get("skip_new_entries", False)
                        _sess_color = "#ef4444" if _sdead else "#22c55e"
                        _sess_label = "🚫 DEAD ZONE" if _sdead else "✅ ACTIVE"
                        with _ae2:
                            st.markdown(
                                f"<div style='background:#1e1e1e;border-left:3px solid {_sess_color};"
                                f"border-radius:4px;padding:8px 12px;margin-bottom:8px'>"
                                f"<div style='color:#aaa;font-size:.7rem;text-transform:uppercase'>Session</div>"
                                f"<div style='font-size:1rem;font-weight:700;color:{_sess_color}'>{_sname}</div>"
                                f"<div style='color:#aaa;font-size:.75rem'>{_sess_label} &nbsp;·&nbsp; "
                                f"Spread tol: ×{_stol:.1f}</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        _ae2.caption("Session engine N/A")

                    # HTF Gate
                    with _ae3:
                        _htf_val  = _htf_mult if _htf_mult is not None else 1.0
                        _htf_color = "#22c55e" if _htf_val >= 1.0 else "#f59e0b" if _htf_val > 0 else "#ef4444"
                        _htf_label = "Full alignment" if _htf_val >= 1.0 else ("Partial — reduced lot" if _htf_val > 0 else "Blocked")
                        st.markdown(
                            f"<div style='background:#1e1e1e;border-left:3px solid {_htf_color};"
                            f"border-radius:4px;padding:8px 12px;margin-bottom:8px'>"
                            f"<div style='color:#aaa;font-size:.7rem;text-transform:uppercase'>HTF Gate</div>"
                            f"<div style='font-size:1rem;font-weight:700;color:{_htf_color}'>Lot ×{_htf_val:.2f}</div>"
                            f"<div style='color:#aaa;font-size:.75rem'>{_htf_label}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    # Exec Quality
                    with _ae4:
                        if _eq:
                            _eq_pass  = _eq.get("passed", True)
                            _eq_reason = _eq.get("reason") or "OK"
                            _eq_spread = _eq.get("spread_pips")
                            _eq_color  = "#22c55e" if _eq_pass else "#ef4444"
                            _eq_label  = "✅ PASSED" if _eq_pass else "❌ BLOCKED"
                            _spread_html = (
                                f'<div style="color:#aaa;font-size:.75rem">Spread: {_eq_spread} pips</div>'
                                if _eq_spread else ''
                            )
                            st.markdown(
                                f"<div style='background:#1e1e1e;border-left:3px solid {_eq_color};"
                                f"border-radius:4px;padding:8px 12px;margin-bottom:8px'>"
                                f"<div style='color:#aaa;font-size:.7rem;text-transform:uppercase'>Exec Quality</div>"
                                f"<div style='font-size:1rem;font-weight:700;color:{_eq_color}'>{_eq_label}</div>"
                                f"<div style='color:#aaa;font-size:.75rem'>{_eq_reason}</div>"
                                f"{_spread_html}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.caption("Exec quality — awaiting first entry attempt")

                    # Chart Pattern card (5th)
                    _la_ae = _s.get("live_analysis") or {}
                    _pat_d = _la_ae.get("top_pattern")
                    with _ae5:
                        if _pat_d and _pat_d.get("name"):
                            _pname  = _pat_d.get("name", "—")
                            _plabel = _pat_d.get("type_label", "Unknown")
                            _pconf  = _pat_d.get("confidence", 0.0)
                            _pdir   = _pat_d.get("direction", "neutral")
                            _pcolor = (
                                "#22c55e" if _pdir == "bullish"
                                else "#ef4444" if _pdir == "bearish"
                                else "#a78bfa"
                            )
                            st.markdown(
                                f"<div style='background:#1e1e1e;border-left:3px solid {_pcolor};"
                                f"border-radius:4px;padding:8px 12px;margin-bottom:8px'>"
                                f"<div style='color:#aaa;font-size:.7rem;text-transform:uppercase'>Chart Pattern</div>"
                                f"<div style='font-size:.9rem;font-weight:700;color:{_pcolor}'>{_pname}</div>"
                                f"<div style='color:#aaa;font-size:.75rem'>{_plabel} &nbsp;·&nbsp; "
                                f"Conf: {_pconf:.0%}</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.caption("No pattern detected")

            # ── Time Zones card ──────────────────────────────────────────────
            with st.expander("🕐 Time Zones — Session Status (UTC)", expanded=False):
                _tz_sessions = [
                    ("Asian",             "00:00 – 07:59", "open",         "Low volatility — structurally valid setups allowed"),
                    ("London Open",       "08:00 – 09:59", "open",         "High momentum · AMD manipulation window"),
                    ("London",            "10:00 – 11:59", "open",         "Continuation setups"),
                    ("NY Lunch",          "12:00 – 12:59", "open_reduced", "Choppy — 50% lot sizing"),
                    ("London/NY Overlap", "13:00 – 16:59", "open",         "Strongest directional volume — now open"),
                    ("NY Afternoon",      "17:00 – 19:59", "blocked",      "0W / 28L &nbsp;·&nbsp; −128 pnl — empirically toxic"),
                    ("Dead Zone",         "20:00 – 23:59", "open",    "Now open — Asian accumulation begins"),
                    ("Weekends",          "Sat – Sun (all day)", "blocked", "No institutional liquidity"),
                ]
                _tz_html = (
                    "<table style='width:100%;border-collapse:collapse;font-size:.8rem'>"
                    "<thead><tr style='border-bottom:1px solid #333'>"
                    "<th style='text-align:left;padding:5px 10px;color:#888'>Session</th>"
                    "<th style='text-align:left;padding:5px 10px;color:#888'>Hours (UTC)</th>"
                    "<th style='text-align:center;padding:5px 10px;color:#888'>Status</th>"
                    "<th style='text-align:left;padding:5px 10px;color:#888'>Reason</th>"
                    "</tr></thead><tbody>"
                )
                for _tz_name, _tz_hours, _tz_status, _tz_reason in _tz_sessions:
                    if _tz_status == "open":
                        _tz_nc = "#22c55e"
                        _tz_badge = "<span style='background:#14532d;color:#22c55e;padding:2px 7px;border-radius:3px;font-size:.72rem;font-weight:700'>OPEN</span>"
                    elif _tz_status == "open_reduced":
                        _tz_nc = "#f59e0b"
                        _tz_badge = "<span style='background:#78350f;color:#f59e0b;padding:2px 7px;border-radius:3px;font-size:.72rem;font-weight:700'>OPEN &frac12; SIZE</span>"
                    else:
                        _tz_nc = "#ef4444"
                        _tz_badge = "<span style='background:#7f1d1d;color:#ef4444;padding:2px 7px;border-radius:3px;font-size:.72rem;font-weight:700'>BLOCKED</span>"
                    _tz_html += (
                        f"<tr style='border-bottom:1px solid #222'>"
                        f"<td style='padding:6px 10px;color:{_tz_nc};font-weight:600'>{_tz_name}</td>"
                        f"<td style='padding:6px 10px;color:#ccc;font-family:monospace;font-size:.78rem'>{_tz_hours}</td>"
                        f"<td style='padding:6px 10px;text-align:center'>{_tz_badge}</td>"
                        f"<td style='padding:6px 10px;color:#777;font-size:.75rem'>{_tz_reason}</td>"
                        f"</tr>"
                    )
                _tz_html += "</tbody></table>"
                st.markdown(_tz_html, unsafe_allow_html=True)

            # ── Live Analysis panel (from autonomous engine) ─────────────────
            _la = _s.get("live_analysis")
            if _la:
                st.markdown("---")
                st.markdown("### 🔬 Live Market Analysis")
                _la_updated = _la.get("updated_at", "")[:19].replace("T", " ")
                st.caption(f"Last analysed: {_la_updated} UTC · {_la.get('price', '—')} current price")

                # ── Grade + Score + Direction hero row ───────────────────────
                _grade   = _la.get("grade", "?")
                _score   = _la.get("score", 0)
                _dir     = _la.get("direction", "sideways")
                _la_eff  = _s.get("effective_min_score") or _s.get("min_score", 65)
                _qualifies = (
                    {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}.get(_grade, 0)
                    >= {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}.get(_s.get("min_grade", "B"), 3)
                    and _score >= _la_eff
                )
                _grade_color = {"A": "#22c55e", "B": "#86efac", "C": "#fbbf24",
                                "D": "#f97316", "F": "#ef4444"}.get(_grade, "#aaa")
                _dir_arrow = "📈" if _dir == "bullish" else "📉" if _dir == "bearish" else "↔️"
                _rr_min_req = _la.get("rr_min", 4.0 if _dir == "bullish" else 2.0)
                _signal_badge = (
                    "🟢 QUALIFIES — would trade" if _qualifies and not _s.get("dry_run") and len(_s.get("open_positions", [])) < 3
                    else "🟡 QUALIFIES — max positions open" if _qualifies
                    else "🔴 BELOW THRESHOLD"
                )

                st.markdown(
                    f"""<div style='display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px'>
                    <div style='background:#1e1e1e;border-radius:8px;padding:12px 20px;text-align:center;min-width:90px'>
                        <div style='font-size:2rem;font-weight:800;color:{_grade_color}'>{_grade}</div>
                        <div style='color:#aaa;font-size:.75rem'>GRADE</div>
                    </div>
                    <div style='background:#1e1e1e;border-radius:8px;padding:12px 20px;text-align:center;min-width:90px'>
                        <div style='font-size:2rem;font-weight:800;color:#60a5fa'>{_score:.0f}</div>
                        <div style='color:#aaa;font-size:.75rem'>SCORE / 100</div>
                    </div>
                    <div style='background:#1e1e1e;border-radius:8px;padding:12px 20px;text-align:center;min-width:110px'>
                        <div style='font-size:1.5rem;font-weight:700'>{_dir_arrow} {_dir.upper()}</div>
                        <div style='color:#aaa;font-size:.75rem'>DIRECTION</div>
                    </div>
                    <div style='background:#1e1e1e;border-radius:8px;padding:12px 20px;text-align:center;min-width:90px'>
                        <div style='font-size:1.5rem;font-weight:700;color:#a78bfa'>1:{_rr_min_req:.0f}</div>
                        <div style='color:#aaa;font-size:.75rem'>MIN R:R</div>
                    </div>
                    <div style='background:#1e1e1e;border-radius:8px;padding:12px 20px;display:flex;align-items:center'>
                        <span style='font-size:1rem'>{_signal_badge}</span>
                    </div>
                    </div>""",
                    unsafe_allow_html=True,
                )

                # ── Key levels + structure ───────────────────────────────────
                la_c1, la_c2, la_c3, la_c4 = st.columns(4)
                _price_val = _la.get("price")
                _atr_val   = _la.get("atr")
                _sup_val   = _la.get("nearest_support")
                _res_val   = _la.get("nearest_resistance")
                _struct    = _la.get("structure", "—")
                _strength  = _la.get("strength")
                _vwap_sig  = _la.get("vwap_signal", "—")
                _vwap_val  = _la.get("vwap_value")

                la_c1.metric("Current Price", f"{_price_val:.2f}" if _price_val else "—")
                la_c1.metric("ATR",           f"{_atr_val:.2f}" if _atr_val else "—")
                la_c2.metric("Nearest Support",    f"{_sup_val:.2f}" if _sup_val else "—",
                             delta=f"{_price_val - _sup_val:.2f} pts below" if (_price_val and _sup_val) else None,
                             delta_color="off")
                la_c2.metric("Nearest Resistance", f"{_res_val:.2f}" if _res_val else "—",
                             delta=f"{_res_val - _price_val:.2f} pts above" if (_price_val and _res_val) else None,
                             delta_color="off")
                la_c3.metric("Structure",  f"{_struct.upper()}" if _struct else "—")
                la_c3.metric("Strength",   f"{_strength}%" if _strength else "—")
                la_c4.metric("VWAP Signal", _vwap_sig.upper() if _vwap_sig else "—")
                la_c4.metric("VWAP Value",  f"{_vwap_val:.2f}" if _vwap_val else "—")

                # ── SMC + pattern counts ─────────────────────────────────────
                la_d1, la_d2, la_d3, la_d4 = st.columns(4)
                la_d1.metric("Order Blocks",     _la.get("ob_count", 0))
                la_d2.metric("Fair Value Gaps",  _la.get("fvg_count", 0))
                la_d3.metric("BOS Events",       _la.get("bos_count", 0))
                la_d4.metric("CHoCH Events",     _la.get("choch_count", 0))

                # ── Confluence reasons ───────────────────────────────────────
                _reasons = _la.get("reasons", [])
                if _reasons:
                    with st.expander("📋 Confluence Reasons", expanded=True):
                        for _r in _reasons:
                            _bullet = "🟢" if any(w in _r.lower() for w in ("bullish","support","above","long")) \
                                      else "🔴" if any(w in _r.lower() for w in ("bearish","resistance","below","short")) \
                                      else "⚪"
                            st.markdown(f"{_bullet} {_r}")

                # ── Fibonacci key levels ─────────────────────────────────────
                _fibs = _la.get("fib_levels", [])
                if _fibs:
                    with st.expander("📐 Nearest Fibonacci Levels"):
                        for _fl in _fibs:
                            _dist = abs(_fl["price"] - _price_val) if _price_val else None
                            _tag = f"**{_fl['pct']}%** — {_fl['price']:.2f}"
                            _tag += f" ({_dist:.2f} pts away)" if _dist else ""
                            st.markdown(f"- {_tag}")

                # ── Multi-Timeframe alignment table ──────────────────────────
                _mtf_biases = _la.get("mtf_biases", [])
                if _mtf_biases:
                    _mtf_score  = _la.get("mtf_score", 0)
                    _mtf_bias   = _la.get("mtf_bias", "sideways")
                    _mtf_conf   = _la.get("mtf_confluence", "low").upper()
                    _mtf_color  = "🟢" if _mtf_bias == "bullish" else "🔴" if _mtf_bias == "bearish" else "🟡"
                    _mtf_conf_color = {"LOW": "🔴", "MEDIUM": "🟡", "HIGH": "🟢"}.get(_mtf_conf, "⚪")
                    with st.expander(
                        f"📊 Multi-Timeframe Alignment  —  {_mtf_color} {_mtf_bias.upper()}  "
                        f"| Score {_mtf_score:+.2f}  |  {_mtf_conf_color} {_mtf_conf} confluence",
                        expanded=True,
                    ):
                        _mtf_rows = []
                        for _b in _mtf_biases:
                            _bias_icon = "🟢 Bullish" if _b["bias"] == "bullish" \
                                else "🔴 Bearish" if _b["bias"] == "bearish" \
                                else "⚪ Sideways" if _b["bias"] == "sideways" \
                                else "❓ Unknown"
                            _bar_filled = int(abs(_b["score"]) * 10)
                            _bar = ("▓" * _bar_filled) + ("░" * (10 - _bar_filled))
                            _mtf_rows.append({
                                "Timeframe": _b["tf"].upper(),
                                "Bias": _bias_icon,
                                "Score": f"{_b['score']:+.2f}",
                                "Weight": f"{int(_b['weight'] * 100)}%",
                                "Strength": _bar,
                            })
                        import pandas as _pd_mtf
                        st.dataframe(
                            _pd_mtf.DataFrame(_mtf_rows),
                            use_container_width=True,
                            hide_index=True,
                        )

            with st.expander("Full status JSON"):
                st.json(_s)

        except Exception as _ex:
            st.error(f"Could not read status file: {_ex}")
    else:
        st.info("No status file found. Start the bot to see live status here.")

    st.divider()

    # ── Recent bot trades from DB ─────────────────────────────────────────────
    st.subheader("Recent Trades (DB)")
    if st.button("🔄 Refresh Trades", key="refresh_live_trades"):
        st.rerun()
    try:
        from storage.database import list_trades as _list_trades
        _trades = _list_trades(asset=bot_asset or None, source="live", limit=50)
        if _trades:
            _tdf = pd.DataFrame(_trades)

            # ── Summary stats ─────────────────────────────────────────────
            _closed = _tdf[_tdf["status"].isin(["win", "loss", "closed"])]
            _wins   = (_tdf["status"] == "win").sum()
            _losses = (_tdf["status"] == "loss").sum()
            _total_pnl = _tdf["pnl"].fillna(0).sum()
            _open_count = (_tdf["status"] == "open").sum()
            _avg_score  = _tdf["score_at_entry"].dropna().mean() if "score_at_entry" in _tdf.columns else None
            _avg_hold   = _tdf["hold_seconds"].dropna().mean()   if "hold_seconds"   in _tdf.columns else None
            _cols = st.columns(5)
            _cols[0].metric("Total Trades", len(_tdf))
            _cols[1].metric("Open", int(_open_count))
            _cols[2].metric(
                "Win / Loss",
                f"{int(_wins)} / {int(_losses)}",
                delta=f"{int(_wins)/(int(_wins)+int(_losses))*100:.0f}% WR" if (_wins + _losses) > 0 else None,
            )
            _cols[3].metric(
                "Total P&L",
                f"${_total_pnl:+.2f}",
                delta_color="normal" if _total_pnl >= 0 else "inverse",
            )
            if _avg_score:
                _cols[4].metric("Avg Score at Entry", f"{_avg_score:.0f}")
            elif _avg_hold:
                _avg_h_min = int(_avg_hold // 60)
                _cols[4].metric("Avg Hold Time", f"{_avg_h_min}m")

            # ── Formatted display columns (include new analytics) ─────────
            _display_cols = [
                "created_at", "direction", "entry_price", "sl_price",
                "tp_price", "exit_price", "size", "pnl", "status",
                "session", "regime", "score_at_entry", "exit_reason",
                "mae", "mfe", "hold_seconds",
            ]
            _show = _tdf[[c for c in _display_cols if c in _tdf.columns]].copy()
            _show.rename(columns={
                "created_at":    "Time",
                "direction":     "Dir",
                "entry_price":   "Entry",
                "sl_price":      "SL",
                "tp_price":      "TP",
                "exit_price":    "Exit",
                "size":          "Lots",
                "pnl":           "P&L $",
                "status":        "Status",
                "session":       "Session",
                "regime":        "Regime",
                "score_at_entry":"Score",
                "exit_reason":   "Exit Reason",
                "mae":           "MAE",
                "mfe":           "MFE",
                "hold_seconds":  "Hold (s)",
            }, inplace=True)
            # Format prices
            for _pc in ["Entry", "SL", "TP", "Exit"]:
                if _pc in _show.columns:
                    _show[_pc] = _show[_pc].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
            if "P&L $" in _show.columns:
                _show["P&L $"] = _show["P&L $"].apply(lambda v: f"+{v:.2f}" if (pd.notna(v) and v >= 0) else (f"{v:.2f}" if pd.notna(v) else "—"))
            if "Lots" in _show.columns:
                _show["Lots"] = _show["Lots"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
            if "Time" in _show.columns:
                _show["Time"] = _show["Time"].str[:19]  # trim microseconds
            if "Score" in _show.columns:
                _show["Score"] = _show["Score"].apply(lambda v: f"{v:.0f}" if pd.notna(v) else "—")
            if "Regime" in _show.columns:
                _show["Regime"] = _show["Regime"].apply(lambda v: v.replace("_", " ").title() if pd.notna(v) else "—")
            if "Session" in _show.columns:
                _show["Session"] = _show["Session"].apply(lambda v: v.replace("_", " ").title() if pd.notna(v) else "—")
            if "MAE" in _show.columns:
                _show["MAE"] = _show["MAE"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
            if "MFE" in _show.columns:
                _show["MFE"] = _show["MFE"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")

            # Colour P&L column
            def _color_pnl_cell(val):
                try:
                    num = float(str(val).replace("+", ""))
                    return "color: #22c55e" if num >= 0 else "color: #ef4444"
                except Exception:
                    return ""

            _styled_show = _show.style
            if "P&L $" in _show.columns:
                _styled_show = _styled_show.map(_color_pnl_cell, subset=["P&L $"])

            st.dataframe(_styled_show, use_container_width=True, hide_index=True)

            # ── Per-regime & per-session P&L breakdown ───────────────────
            _closed_df = _tdf[_tdf["status"].isin(["win", "loss", "closed"])].copy()
            if not _closed_df.empty:
                _bk1, _bk2 = st.columns(2)
                if "regime" in _closed_df.columns and _closed_df["regime"].notna().any():
                    _rg = _closed_df.groupby("regime")["pnl"].agg(["count", "sum", "mean"]).reset_index()
                    _rg.columns = ["Regime", "Trades", "Total P&L $", "Avg P&L $"]
                    _rg["Regime"] = _rg["Regime"].str.replace("_", " ").str.title()
                    _rg["Total P&L $"] = _rg["Total P&L $"].apply(lambda v: f"${v:+.2f}")
                    _rg["Avg P&L $"]   = _rg["Avg P&L $"].apply(lambda v: f"${v:+.2f}")
                    _bk1.markdown("**P&L by Regime**")
                    _bk1.dataframe(_rg, hide_index=True, use_container_width=True)
                if "session" in _closed_df.columns and _closed_df["session"].notna().any():
                    _sg = _closed_df.groupby("session")["pnl"].agg(["count", "sum", "mean"]).reset_index()
                    _sg.columns = ["Session", "Trades", "Total P&L $", "Avg P&L $"]
                    _sg["Session"] = _sg["Session"].str.replace("_", " ").str.title()
                    _sg["Total P&L $"] = _sg["Total P&L $"].apply(lambda v: f"${v:+.2f}")
                    _sg["Avg P&L $"]   = _sg["Avg P&L $"].apply(lambda v: f"${v:+.2f}")
                    _bk2.markdown("**P&L by Session**")
                    _bk2.dataframe(_sg, hide_index=True, use_container_width=True)
        else:
            st.info("No live trades recorded yet. Trades will appear here once the bot starts executing.")
    except Exception as _ex:
        st.warning(f"Could not load trades: {_ex}")

    # ══════════════════════════════════════════════════════════════════════════
    # LEARNED INTELLIGENCE PANEL — what the bot has figured out
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("🧬 What the Bot Has Learned")

    try:
        import json as _json_lrn
        _lpath = Path(__file__).parent.parent / "live_bot" / "learning_state.json"
        with open(_lpath, encoding="utf-8") as _lf:
            _lrn2 = _json_lrn.load(_lf)

        _saved_at = _lrn2.get("saved_at", "")
        if _saved_at:
            st.caption(f"Learning state last updated: {_saved_at[:19].replace('T', ' ')} UTC")

        # ── Top-level insight cards ───────────────────────────────────────────
        _total_w  = _lrn2.get("wins", 0)
        _total_l  = _lrn2.get("losses", 0)
        _total_t  = _total_w + _total_l
        _wr_all   = _total_w / _total_t * 100 if _total_t else 0
        _adj      = _lrn2.get("score_adjust", 0.0)
        _streak   = _lrn2.get("streak", 0)
        _best_str = _lrn2.get("best_streak", 0)
        _worst_str= _lrn2.get("worst_streak", 0)
        _tot_pnl  = _lrn2.get("total_pnl", 0.0)
        _last20   = _lrn2.get("last_20_results", [])
        _r20_wr   = sum(_last20) / len(_last20) * 100 if _last20 else 0

        _adj_color = "#22c55e" if _adj >= 0 else "#ef4444"
        _adj_label = "relaxed (more signals passing)" if _adj > 0 else \
                     "tightened (fewer signals passing)" if _adj < 0 else "neutral"
        _streak_color = "#22c55e" if _streak > 0 else "#ef4444" if _streak < 0 else "#aaa"

        _ins_c1, _ins_c2, _ins_c3, _ins_c4, _ins_c5 = st.columns(5)
        _ins_c1.metric("All-time Win Rate",  f"{_wr_all:.0f}%",
                       f"{_total_t} closed trades")
        _ins_c2.metric("Last 20 Win Rate",  f"{_r20_wr:.0f}%",
                       f"{sum(_last20)}/{len(_last20)} wins" if _last20 else "—")
        _ins_c3.metric("Score Gate Adjust", f"{_adj:+.1f} pts",
                       _adj_label, delta_color="normal" if _adj >= 0 else "inverse")
        _ins_c4.metric("Current Streak",    f"{'+' if _streak > 0 else ''}{_streak}",
                       f"Best: {_best_str}  Worst: {_worst_str}")
        _ins_c5.metric("Total Realised P&L", f"${_tot_pnl:+.2f}")

        st.markdown("---")

        # ── Human-readable narrative ──────────────────────────────────────────
        _narrative_lines = []

        # Score adjustment narrative
        if abs(_adj) >= 5:
            _narrative_lines.append(
                f"⚙️ **Score gate is {'lowered' if _adj < 0 else 'raised'} by {abs(_adj):.1f} pts** — "
                f"the bot {'is being more selective about entries' if _adj < 0 else 'is accepting more signals than usual'} "
                f"based on recent performance."
            )

        # Grade A vs B performance
        _gs = _lrn2.get("grade_stats", {})
        _a  = _gs.get("A", {}); _b = _gs.get("B", {})
        _a_wr = _a.get("wins",0)/(_a.get("wins",0)+_a.get("losses",1)-1) * 100 \
                if (_a.get("wins",0)+_a.get("losses",0)) > 0 else None
        _b_wr = _b.get("wins",0)/(_b.get("wins",0)+_b.get("losses",1)-1) * 100 \
                if (_b.get("wins",0)+_b.get("losses",0)) > 0 else None
        if _a_wr is not None:
            _narrative_lines.append(
                f"🅰️ **Grade A signals**: {_a.get('wins',0)}W / {_a.get('losses',0)}L "
                f"({_a.get('acted',0)} taken from {_a.get('seen',0)} seen)."
            )
        if _b_wr is not None:
            _narrative_lines.append(
                f"🅱️ **Grade B signals**: {_b.get('wins',0)}W / {_b.get('losses',0)}L "
                f"({_b.get('acted',0)} taken from {_b.get('seen',0)} seen)."
            )

        # LTF alignment insight
        _ltf2 = _lrn2.get("ltf_stats", {})
        _both = _ltf2.get("both_confirmed", {}); _one = _ltf2.get("one_counter", {})
        _both_t = _both.get("wins",0)+_both.get("losses",0)
        _one_t  = _one.get("wins",0)+_one.get("losses",0)
        if _both_t > 0:
            _narrative_lines.append(
                f"📡 **When both 1M+5M confirmed** the 4H direction: "
                f"{_both.get('wins',0)}W / {_both.get('losses',0)}L "
                f"({_both.get('wins',0)/_both_t*100:.0f}% win rate). "
                + (f"When only one LTF confirmed: {_one.get('wins',0)}W / {_one.get('losses',0)}L "
                   f"({_one.get('wins',0)/_one_t*100:.0f}%)." if _one_t > 0 else "")
            )

        # Best regime
        _rs2 = _lrn2.get("regime_stats", {})
        _rs2_real = {k: v for k, v in _rs2.items() if k != "unknown" and (v.get("wins",0)+v.get("losses",0)) > 0}
        if _rs2_real:
            _best_reg = max(_rs2_real, key=lambda k: _rs2_real[k].get("pnl", 0))
            _worst_reg= min(_rs2_real, key=lambda k: _rs2_real[k].get("pnl", 0))
            _br = _rs2_real[_best_reg]
            _wr2 = _rs2_real[_worst_reg]
            _narrative_lines.append(
                f"🏆 **Best regime** so far: **{_best_reg.replace('_',' ').title()}** "
                f"({_br.get('wins',0)}W / {_br.get('losses',0)}L, "
                f"${_br.get('pnl',0):+.2f} P&L)."
            )
            if _worst_reg != _best_reg and _wr2.get("pnl", 0) < 0:
                _narrative_lines.append(
                    f"⚠️ **Weakest regime**: **{_worst_reg.replace('_',' ').title()}** "
                    f"({_wr2.get('wins',0)}W / {_wr2.get('losses',0)}L, "
                    f"${_wr2.get('pnl',0):+.2f} P&L) — bot noted."
                )

        # Best session
        _ss2 = _lrn2.get("session_stats", {})
        _ss2_real = {k: v for k, v in _ss2.items() if k != "unknown" and (v.get("wins",0)+v.get("losses",0)) > 0}
        if _ss2_real:
            _best_ses = max(_ss2_real, key=lambda k: _ss2_real[k].get("pnl", 0))
            _bsv = _ss2_real[_best_ses]
            _narrative_lines.append(
                f"🕐 **Best session**: **{_best_ses.replace('_',' ').title()}** "
                f"({_bsv.get('wins',0)}W / {_bsv.get('losses',0)}L, "
                f"${_bsv.get('pnl',0):+.2f} P&L)."
            )

        # Exit reason insight
        _er2 = _lrn2.get("exit_reason_stats", {})
        if _er2:
            for _ek, _ev in sorted(_er2.items(), key=lambda x: -x[1].get("pnl", 0))[:2]:
                _cnt = _ev.get("count",0)
                _epnl= _ev.get("pnl",0)
                _elab = {"5m_exit": "5M momentum exits", "5m_partial": "5M partial exits",
                         "tp1": "TP1 hits", "tp2": "TP2 hits", "sl": "Stop-loss hits",
                         "trail": "Trailing stop exits"}.get(_ek, _ek)
                _narrative_lines.append(
                    f"🚪 **{_elab}**: {_cnt} exits totalling ${_epnl:+.2f}."
                )

        if _narrative_lines:
            for _nl in _narrative_lines:
                st.markdown(_nl)
        else:
            st.info("Not enough closed trades yet to generate learning insights.")

        st.markdown("---")

        # ── Detailed breakdown columns ────────────────────────────────────────
        _lrn_col1, _lrn_col2, _lrn_col3 = st.columns(3)

        # Grade performance table
        with _lrn_col1:
            st.markdown("**Grade Performance**")
            _gp_rows = []
            for _gk in ["A", "B", "C", "D"]:
                _gv = _gs.get(_gk, {})
                _gw = _gv.get("wins", 0); _gl = _gv.get("losses", 0)
                _gt = _gw + _gl
                _gp_rows.append({
                    "Grade":    _gk,
                    "Seen":     _gv.get("seen", 0),
                    "Taken":    _gv.get("acted", 0),
                    "W": _gw, "L": _gl,
                    "WR":       f"{_gw/_gt*100:.0f}%" if _gt else "—",
                })
            st.dataframe(pd.DataFrame(_gp_rows), hide_index=True,
                         use_container_width=True)

        # LTF alignment table
        with _lrn_col2:
            st.markdown("**1M+5M Entry Timing**")
            _ltf_label2 = {
                "both_confirmed": "Both confirmed ✅",
                "one_counter":    "One counter ⚠️",
                "unknown":        "LTF unknown ❓",
                "trap":           "Trap blocked 🚫",
            }
            _ltf_rows2 = []
            for _lk, _lv in _ltf2.items():
                _lw = _lv.get("wins", 0); _ll = _lv.get("losses", 0)
                _lt = _lw + _ll
                _ltf_rows2.append({
                    "LTF State": _ltf_label2.get(_lk, _lk),
                    "W": _lw, "L": _ll,
                    "WR": f"{_lw/_lt*100:.0f}%" if _lt else "—",
                })
            if _ltf_rows2:
                st.dataframe(pd.DataFrame(_ltf_rows2), hide_index=True,
                             use_container_width=True)
            else:
                st.caption("No LTF data yet.")

        # Exit reasons table
        with _lrn_col3:
            st.markdown("**Exit Reason Breakdown**")
            _er_rows = []
            _er_labels = {"5m_exit": "5M Exit", "5m_partial": "5M Partial",
                          "tp1": "TP1", "tp2": "TP2", "sl": "Stop Loss",
                          "trail": "Trail Stop"}
            for _ek2, _ev2 in _er2.items():
                _er_rows.append({
                    "Exit Type": _er_labels.get(_ek2, _ek2),
                    "Count": _ev2.get("count", 0),
                    "Wins":  _ev2.get("wins", 0),
                    "P&L $": f"${_ev2.get('pnl', 0):+.2f}",
                })
            if _er_rows:
                st.dataframe(pd.DataFrame(_er_rows), hide_index=True,
                             use_container_width=True)
            else:
                st.caption("No exit data yet.")

        # ── Regime & session win-rate bars ────────────────────────────────────
        if _rs2_real or _ss2_real:
            _bar_c1, _bar_c2 = st.columns(2)

            if _rs2_real:
                import plotly.graph_objects as _plgo2
                _rnames = [k.replace("_"," ").title() for k in _rs2_real]
                _rpnls  = [_rs2_real[k].get("pnl", 0) for k in _rs2_real]
                _rcolors= ["#22c55e" if p >= 0 else "#ef4444" for p in _rpnls]
                _rfig = _plgo2.Figure(_plgo2.Bar(
                    x=_rnames, y=_rpnls, marker_color=_rcolors,
                    hovertemplate="%{x}: $%{y:+.2f}<extra></extra>",
                ))
                _rfig.update_layout(
                    title="P&L by Regime", height=220,
                    margin=dict(l=0, r=0, t=35, b=0),
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="#ccc"), yaxis_title="P&L $",
                )
                _bar_c1.plotly_chart(_rfig, use_container_width=True)

            if _ss2_real:
                _snames = [k.replace("_"," ").title() for k in _ss2_real]
                _spnls  = [_ss2_real[k].get("pnl", 0) for k in _ss2_real]
                _scolors= ["#22c55e" if p >= 0 else "#ef4444" for p in _spnls]
                _sfig = _plgo2.Figure(_plgo2.Bar(
                    x=_snames, y=_spnls, marker_color=_scolors,
                    hovertemplate="%{x}: $%{y:+.2f}<extra></extra>",
                ))
                _sfig.update_layout(
                    title="P&L by Session", height=220,
                    margin=dict(l=0, r=0, t=35, b=0),
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="#ccc"), yaxis_title="P&L $",
                )
                _bar_c2.plotly_chart(_sfig, use_container_width=True)

        # ── OB stats ─────────────────────────────────────────────────────────
        _ob2 = _lrn2.get("ob_stats", {})
        if _ob2:
            st.markdown("**Order Block Performance** (by OB direction at entry)")
            _ob_rows2 = []
            for _odir, _ov in _ob2.items():
                _oh = _ov.get("hits", 0); _ow = _ov.get("wins", 0)
                _ob_rows2.append({
                    "OB Direction": _odir.title(),
                    "Entries Used": _oh,
                    "Wins Attributed": _ow,
                    "Hit→Win":  f"{_ow/_oh*100:.0f}%" if _oh else "—",
                })
            st.dataframe(pd.DataFrame(_ob_rows2), hide_index=True,
                         use_container_width=False)

    except FileNotFoundError:
        st.info("Learning state file not found — bot hasn't closed any trades yet.")
    except Exception as _lex:
        st.warning(f"Could not load learning state: {_lex}")
