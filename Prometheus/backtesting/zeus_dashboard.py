"""
Zeus — Institutional Validation Engine Dashboard
================================================
Interactive Streamlit UI for configuring, running, and analysing
the Zeus institutional validation workspace and retained walk-forward scalp backtest.

Run:
    streamlit run backtesting/zeus_dashboard.py --server.port=8502
    (different port from the live-bot dashboard on 8501)

Tabs:
    🏛 Validation Overview — institutional validation KPIs and lifecycle
    📬 Validation Queue — incoming research candidates and queue states
    📑 Validation Reports — completed validation detail and pipeline stages
    🧾 Evidence Library — additive institutional evidence index
    📊 Overview — retained core backtest metrics + equity curve + drawdown
    🎯 Segments — retained win-rate breakdowns by entry / zone / LTF / grade / session
    📋 Trade Log — retained filterable / sortable trade table
    🤖 ML & Patterns — retained XGBoost feature importance + pattern heat-map
    💡 What Works — retained insight bullets + top win-rate combinations
"""

from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Project root on path ───────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

_ZEUS_VALIDATION_STATUS_F = _ROOT / "storage" / "olympus" / "zeus_validation_status.json"
_ZEUS_VALIDATION_REPORTS_F = _ROOT / "storage" / "olympus" / "zeus_validation_reports.jsonl"

# ── Page config (MUST be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Zeus · Institutional Validation Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --bg: #0a0d14;
    --surface: #141924;
    --surface2: #1c2436;
    --accent: #7b61ff;
    --gold: #f1c40f;
    --success: #27ae60;
    --danger: #e74c3c;
    --warn: #f39c12;
    --text: #dde1ea;
    --muted: #7f8c9a;
}
body { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; }
.zeus-header {
    background: linear-gradient(135deg, #141924 0%, #1a1040 100%);
    border-radius: 12px; padding: 24px 32px; margin-bottom: 24px;
    border: 1px solid #2a1f6e;
}
.zeus-header h1 { margin:0; font-size: 2.2rem; color: #a78bfa; }
.zeus-header p  { margin:4px 0 0; color: var(--muted); font-size: 0.95rem; }
.kpi-card {
    background: var(--surface2); border-radius: 10px;
    padding: 16px 20px; text-align: center;
    border: 1px solid #263050;
}
.kpi-card .label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
.kpi-card .value { font-size: 1.6rem; font-weight: 700; margin-top: 4px; }
.kpi-green  { color: #27ae60; }
.kpi-red    { color: #e74c3c; }
.kpi-gold   { color: #f1c40f; }
.kpi-purple { color: #a78bfa; }
.kpi-blue   { color: #3498db; }
.insight-box {
    background: var(--surface2); border-radius: 8px;
    padding: 14px 18px; margin: 8px 0;
    border-left: 4px solid #7b61ff;
    font-size: 0.9rem; line-height: 1.6;
}
.combo-row {
    background: var(--surface2); border-radius: 8px;
    padding: 12px 18px; margin: 6px 0;
    display: flex; justify-content: space-between; align-items: center;
}
.tag {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 0.75rem; font-weight: 600; margin-right: 6px;
}
.tag-limit   { background: #1a3a5c; color: #5dade2; }
.tag-market  { background: #2d1b00; color: #f39c12; }
.tag-ob      { background: #1a3319; color: #27ae60; }
.tag-sr      { background: #2d2b00; color: #f1c40f; }
.tag-A       { background: #0d3318; color: #27ae60; }
.tag-B       { background: #1a2e00; color: #82e0aa; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────────────────────
if "zeus_result"    not in st.session_state: st.session_state.zeus_result    = None
if "zeus_running"   not in st.session_state: st.session_state.zeus_running   = False
if "zeus_progress"  not in st.session_state: st.session_state.zeus_progress  = 0.0
if "zeus_log"       not in st.session_state: st.session_state.zeus_log       = ""
if "zeus_error"     not in st.session_state: st.session_state.zeus_error     = None
if "zeus_primary_df" not in st.session_state: st.session_state.zeus_primary_df = None
if "zeus_ctx_dfs"   not in st.session_state: st.session_state.zeus_ctx_dfs   = None


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="zeus-header">
    <h1>⚡ ZEUS</h1>
    <p><strong>Institutional Validation Engine</strong></p>
    <p>Research · Validation · Evidence · Approval</p>
    <p>Zeus v2.0 · retained backtesting workflow with additive institutional validation authority</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — configuration
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Backtest Configuration")

    st.markdown("**Market**")
    asset  = st.text_input("Asset symbol", value="XAUUSDm")
    tf_opt = st.selectbox("Primary timeframe", ["30m", "15m", "5m"], index=0)

    st.markdown("**Date range** (leave blank to use bar count)")
    col1, col2 = st.columns(2)
    with col1:
        date_from_str = st.text_input("From (YYYY-MM-DD)", value="", placeholder="2025-01-01")
    with col2:
        date_to_str   = st.text_input("To (YYYY-MM-DD)",   value="", placeholder="2026-06-01")

    st.markdown("**Simulation**")
    stride    = st.number_input("Signal stride (bars)", min_value=1, max_value=20, value=5,
                                  help="Evaluate a new signal every Nth bar. Lower = more trade opportunities, slower run.")
    slippage  = st.number_input("Slippage (pts)", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
    warmup_bars = st.number_input("Warmup bars (skip)", min_value=10, max_value=200, value=50, step=10,
                                   help="Skip first N bars to allow indicators to stabilise. Fewer = more bars evaluated.")

    n_bars = st.number_input("Bar count (if no dates)", min_value=100, max_value=10000,
                              value=300, step=50)
    _est_mins = max(1, int(n_bars / stride * 2 / 60))
    st.caption(f"⏱ Est. ~{_est_mins} min at stride {stride} (each bar ~2 s)")

    st.markdown("**Account**")
    balance  = st.number_input("Initial balance ($)", min_value=10.0, max_value=100000.0,
                                value=120.0, step=10.0)
    risk_pct = st.slider("Risk % per trade", min_value=0.5, max_value=5.0,
                          value=2.0, step=0.5)

    st.markdown("**Concurrent position limits**")
    _col1, _col2, _col3 = st.columns(3)
    with _col1:
        small_acct_thresh = st.number_input("Small acct ($)", min_value=10.0, max_value=5000.0,
                                             value=120.0, step=10.0,
                                             help="Balance below this = small-account regime")
        small_max_open   = st.number_input("Small max open", min_value=1, max_value=10, value=2,
                                            help="Max concurrent trades when balance < Small acct ($)")
    with _col2:
        medium_acct_thresh = st.number_input("Medium acct ($)", min_value=100.0, max_value=10000.0,
                                              value=500.0, step=50.0,
                                              help="Balance below this = medium-account regime")
        medium_max_open  = st.number_input("Medium max open", min_value=1, max_value=10, value=3,
                                            help="Max concurrent trades when balance < Medium acct ($)")
    with _col3:
        st.markdown("&nbsp;", unsafe_allow_html=True)  # spacer
        normal_max_open  = st.number_input("Normal max open", min_value=1, max_value=10, value=5,
                                            help="Max concurrent trades for full-size accounts")

    st.markdown("**Signal quality**")
    min_grade = st.selectbox("Minimum grade", ["A", "B", "C"], index=0,
                              help="Phase 1 default: A only")
    min_score = st.slider("Minimum score", min_value=50, max_value=99, value=80, step=1,
                           help="Phase 1 default: 80 (all wins scored ≥87)")
    entry_mode = st.selectbox("Entry mode", ["zone_only", "market_any"], index=0)
    strategy_name = st.selectbox(
        "Strategy profile",
        ["zeus_ltf", "ltf_pullback", "trend_follow", "custom"],
        index=0,
        help="Select the LTF strategy profile that should be used for this backtest session.",
    )

    st.markdown("**Enabled sessions**")
    _ALL_SESSIONS = [
        "asian",
        "london_open",
        "london",
        "ny_lunch",
        "london_ny_overlap",
        "ny_afternoon",
        "dead_zone",
    ]
    # ny_afternoon and dead_zone are always hard-blocked by the backtester
    # regardless of this list; default only shows the tradeable ones.
    _DEFAULT_SESSIONS = [
        "asian",
        "london_open",
        "london",
        "ny_lunch",
        "london_ny_overlap",
    ]
    enabled_sessions = st.multiselect(
        "Sessions to include",
        options=_ALL_SESSIONS,
        default=_DEFAULT_SESSIONS,
        help=(
            "Controls which session windows are eligible for new entries. "
            "ny_afternoon and dead_zone are always hard-blocked by the backtester engine."
        ),
    )

    st.markdown("**Pending limit gates**")
    _lc1, _lc2 = st.columns(2)
    with _lc1:
        limit_expiry = st.number_input(
            "Limit expiry (bars)", min_value=1, max_value=1000, value=240, step=10,
            help="Cancel unfilled limit orders after this many bars. Lower = fewer stale entries."
        )
    with _lc2:
        max_limit_dist = st.slider(
            "Max limit dist (ATR)", min_value=0.5, max_value=6.0, value=3.0, step=0.25,
            help="Skip limit order if zone is >N×ATR from current price. Lower = only near-price zones."
        )

    st.markdown("**Phase 1 — Filter Optimisation**")
    sr_premium = st.slider(
        "SR zone score premium", min_value=0, max_value=30, value=15, step=1,
        help="Extra score pts required when no fresh OB exists (one_counter state). "
             "Higher = fewer SR entries, higher quality."
    )
    block_atr  = st.slider(
        "Block high-ATR rank (one_counter)", min_value=0.0, max_value=1.0, value=0.85, step=0.05,
        help="Block one_counter entries when ATR rank ≥ this. 0 = disabled. "
             "Losses clustered at 0.83-0.94."
    )
    bc_min_score = st.slider(
        "both_confirmed min score", min_value=70, max_value=95, value=80, step=1,
        help="both_confirmed requires Grade A + score ≥ this. Was 85, now 80 (100% WR at both levels)."
    )

    st.markdown("**Phase 2 — R:R Optimisation**")
    tp1_rr = st.slider("TP1 R:R", min_value=0.5, max_value=3.0, value=1.5, step=0.25,
                        help="Phase 2 default: 1.5 (was 1.0). Locks 50% profit at better ratio.")
    tp2_rr = st.slider("TP2 R:R", min_value=1.0, max_value=8.0, value=5.0, step=0.5,
                        help="Phase 2 default: 5.0 (was 3.0). Lets OB runners go further.")
    trail_mult = st.slider("Trail ATR mult", min_value=0.5, max_value=3.0, value=1.2, step=0.1,
                            help="Phase 2 default: 1.2 (was 1.5). Tighter trail on confirmed moves.")
    be_trigger = st.slider("BE ATR trigger", min_value=0.2, max_value=1.0, value=0.35, step=0.05,
                            help="Phase 2 default: 0.35 (was 0.5). Lock BE earlier on winners.")

    st.markdown("**ML**")
    train_ml = st.checkbox("Train XGBoost after run", value=True)

    # ── Strategy Lab ──────────────────────────────────────────────────────
    with st.expander("🧪 Strategy Lab", expanded=False):
        st.caption(
            "All flags default to OFF — enabling them adds live-bot execution "
            "discipline on top of the Prometheus signal. Custom strategy_fn is "
            "not yet wirable via the UI (use run_scalp_backtest.py for that)."
        )

        st.markdown("**Phase 1 — Entry Discipline**")
        _lab_c1, _lab_c2 = st.columns(2)
        with _lab_c1:
            lab_cooldown = st.number_input(
                "Cooldown bars", min_value=0, max_value=50, value=0, step=1,
                help="Minimum bars between any two entries. 0 = disabled."
            )
            lab_flip_bars = st.number_input(
                "Direction flip min bars", min_value=0, max_value=50, value=0, step=1,
                help="Bars of history required before a direction flip entry is allowed. 0 = disabled."
            )
        with _lab_c2:
            lab_sess_halt = st.number_input(
                "Session-loss halt (N losses)", min_value=0, max_value=10, value=0, step=1,
                help="Halt all entries in a session after N consecutive losses. 0 = disabled."
            )

        st.markdown("**Phase 2 — Exit Parity**")
        lab_m5_sev = st.checkbox(
            "M5 severity exit",
            value=False,
            help="Classify each bar by body ratio: weak → tighten SL · moderate → 30% partial · strong → full close.",
        )
        lab_timeout = st.checkbox(
            "Timeout exits",
            value=False,
            help="Smart 50% partial at smart_bars if profit ≥ floor, then full close at hard_bars.",
        )
        if lab_timeout:
            _to1, _to2, _to3 = st.columns(3)
            with _to1:
                lab_smart_bars = st.number_input("Smart partial bars", min_value=1, max_value=200, value=15, step=1)
            with _to2:
                lab_hard_bars  = st.number_input("Hard close bars",   min_value=1, max_value=500, value=30,  step=1)
            with _to3:
                lab_profit_min = st.number_input("Profit floor ($)",  min_value=0.0, max_value=500.0, value=15.0, step=1.0)
        else:
            lab_smart_bars = 15
            lab_hard_bars  = 30
            lab_profit_min = 15.0

        st.markdown("**Phase 3 — Daily Circuit Breakers**")
        _cb1, _cb2, _cb3 = st.columns(3)
        with _cb1:
            lab_max_loss_pct = st.number_input(
                "Max daily loss %", min_value=0.0, max_value=20.0, value=0.0, step=0.5,
                help="Halt all new entries today once daily loss reaches this % of day-start equity. 0 = disabled."
            )
        with _cb2:
            lab_protect_pct = st.number_input(
                "Daily profit protect %", min_value=0.0, max_value=20.0, value=0.0, step=0.5,
                help="Scale lot down once daily gain hits this %. 0 = disabled."
            )
        with _cb3:
            lab_lot_scalar = st.number_input(
                "Lot scalar (protect)", min_value=0.1, max_value=1.0, value=0.5, step=0.05,
                help="Lot multiplier applied when daily profit protect is active.",
                disabled=(lab_protect_pct == 0.0),
            )

    st.markdown("**Output**")
    save_report = st.checkbox("Save JSON report", value=False)
    report_path = st.text_input("Report path", value="outputs/zeus_report.json") if save_report else None

    st.markdown("---")
    with st.expander("🏛 Validation Workspace Filters", expanded=False):
        validation_modes_selected = st.multiselect(
            "Validation Modes",
            [
                "Historical Backtest",
                "Walk Forward",
                "Out-of-Sample",
                "Monte Carlo",
                "Feature Validation",
                "Execution Validation",
                "Pattern Validation",
                "Capital Validation",
                "Recommendation Validation",
            ],
            default=[
                "Historical Backtest",
                "Walk Forward",
                "Out-of-Sample",
                "Monte Carlo",
            ],
            help="Filters validation workspace views only. Backtest workflow remains unchanged.",
        )
        queue_states_selected = st.multiselect(
            "Queue States",
            ["Queued", "Running", "Paused", "Completed", "Rejected", "Approved"],
            default=["Queued", "Running", "Paused", "Completed", "Rejected", "Approved"],
            help="Filters validation queue and evidence views only.",
        )
        validation_search = st.text_input("Search Evidence / Queue", value="")

    st.markdown("---")

    # CSV upload alternative
    csv_file = st.file_uploader("Upload CSV (alternative to MT5)", type=["csv"])

    # Run controls
    run_btn  = st.button("▶ Run Zeus Backtest", type="primary", use_container_width=True)
    clear_btn= st.button("🗑 Clear results", use_container_width=True)

    if clear_btn:
        st.session_state.zeus_result   = None
        st.session_state.zeus_progress = 0.0
        st.session_state.zeus_log      = ""
        st.session_state.zeus_error    = None
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Date parsing helper
# ─────────────────────────────────────────────────────────────────────────────
def _parse_date(s: str) -> Optional[datetime]:
    if not s or not s.strip():
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Run the backtest
# ─────────────────────────────────────────────────────────────────────────────
if run_btn and not st.session_state.zeus_running:
    st.session_state.zeus_running  = True
    st.session_state.zeus_result   = None
    st.session_state.zeus_progress = 0.0
    st.session_state.zeus_error    = None
    st.session_state.zeus_log      = ""

    # Parse config
    _date_from = _parse_date(date_from_str)
    _date_to   = _parse_date(date_to_str)

    try:
        from backtesting.scalp_backtester import ScalpBacktester, ScalpBacktestConfig

        cfg = ScalpBacktestConfig(
            asset                  = asset,
            primary_tf             = tf_opt,
            date_from              = _date_from,
            date_to                = _date_to,
            n_bars                 = int(n_bars),
            initial_balance        = float(balance),
            risk_pct               = float(risk_pct),
            min_grade              = min_grade,
            min_score              = float(min_score),
            entry_mode             = entry_mode,
            strategy_name          = strategy_name,
            enabled_sessions       = enabled_sessions,
            slippage_pts           = float(slippage),
            signal_stride          = int(stride),
            warmup_bars            = int(warmup_bars),
            # Concurrent position limits
            small_acct_threshold   = float(small_acct_thresh),
            small_acct_max_open    = int(small_max_open),
            medium_acct_threshold  = float(medium_acct_thresh),
            medium_acct_max_open   = int(medium_max_open),
            normal_acct_max_open   = int(normal_max_open),
            # Pending limit gates
            limit_order_expiry     = int(limit_expiry),
            max_limit_dist_atr     = float(max_limit_dist),
            # Phase 1 filters
            sr_min_score_premium   = float(sr_premium),
            block_high_atr_rank    = float(block_atr),
            both_confirmed_min_score = float(bc_min_score),
            # Phase 2 R:R
            tp1_rr                 = float(tp1_rr),
            tp2_rr                 = float(tp2_rr),
            trail_atr_mult         = float(trail_mult),
            be_atr_trigger         = float(be_trigger),
            train_ml               = train_ml,
            report_path            = report_path,
            verbose                = False,
            # Strategy Lab — Phase 1 entry discipline
            entry_cooldown_bars    = int(lab_cooldown),
            direction_flip_min_bars= int(lab_flip_bars),
            session_dir_loss_halt  = int(lab_sess_halt),
            # Strategy Lab — Phase 2 exit parity
            m5_severity_enable     = bool(lab_m5_sev),
            time_exit_enable       = bool(lab_timeout),
            time_exit_smart_bars   = int(lab_smart_bars),
            time_exit_hard_bars    = int(lab_hard_bars),
            time_exit_profit_min   = float(lab_profit_min),
            # Strategy Lab — Phase 3 circuit breakers
            max_daily_loss_pct     = float(lab_max_loss_pct),
            daily_profit_protect_pct = float(lab_protect_pct),
            daily_profit_lot_scalar  = float(lab_lot_scalar),
        )

        bt = ScalpBacktester(cfg)

        # Progress bar placeholder
        _prog_bar  = st.progress(0.0, text="Initialising Zeus engine...")
        _prog_text = st.empty()

        def _cb(done: int, total: int) -> None:
            pct = done / max(1, total)
            st.session_state.zeus_progress = pct
            _prog_bar.progress(pct, text=f"Running… {done}/{total} bars ({pct:.0%})")

        # CSV path
        primary_df = None
        if csv_file is not None:
            primary_df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            primary_df.columns = [c.lower() for c in primary_df.columns]
            _prog_text.info(f"Loaded {len(primary_df)} bars from CSV")

        # Run (blocking — shows spinner)
        with st.spinner("Zeus is analysing history…"):
            result = bt.run(primary_df=primary_df, progress_cb=_cb)

        _prog_bar.progress(1.0, text="Complete ✓")
        _prog_text.empty()

        st.session_state.zeus_result  = result
        st.session_state.zeus_running = False
        st.rerun()

    except Exception as exc:
        import traceback
        st.session_state.zeus_error   = traceback.format_exc()
        st.session_state.zeus_running = False
        st.rerun()

# Show error if any
if st.session_state.zeus_error:
    with st.expander("❌ Backtest error", expanded=True):
        st.code(st.session_state.zeus_error)


# ─────────────────────────────────────────────────────────────────────────────
# Helper chart builders
# ─────────────────────────────────────────────────────────────────────────────
_DARK = dict(
    paper_bgcolor="#0a0d14",
    plot_bgcolor="#0a0d14",
    font=dict(color="#dde1ea"),
)
_DARK_GRID = dict(gridcolor="#1c2436", zerolinecolor="#1c2436")


def _apply_dark(fig: go.Figure) -> go.Figure:
    """Apply dark theme + grid colours to a figure."""
    fig.update_layout(**_DARK)
    fig.update_xaxes(**_DARK_GRID)
    fig.update_yaxes(**_DARK_GRID)
    return fig


def _load_json_silent(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _load_jsonl_silent(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
            except Exception:
                continue
    except Exception:
        return []
    return rows


def _validation_search_match(row: Dict[str, Any], term: str) -> bool:
    if not term:
        return True
    return term.lower() in json.dumps(row, default=str).lower()


def _passes_mode_filter(row: Dict[str, Any], selected_modes: List[str]) -> bool:
    if not selected_modes:
        return True
    row_modes = row.get("validation_modes", []) or []
    if not isinstance(row_modes, list):
        return True
    return any(mode in row_modes for mode in selected_modes)


def _build_queue_df(items: List[Dict[str, Any]], selected_modes: List[str], selected_states: List[str], search_term: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in items:
        queue_state = str(item.get("queue_state") or "Queued")
        if selected_states and queue_state not in selected_states:
            continue
        if not _passes_mode_filter(item, selected_modes):
            continue
        if not _validation_search_match(item, search_term):
            continue
        origin = item.get("research_origin", {}) if isinstance(item.get("research_origin"), dict) else {}
        rows.append(
            {
                "Report ID": item.get("report_id", "unknown"),
                "Source System": origin.get("source", item.get("candidate_source_system", "unknown")),
                "Research Category": origin.get("research_category", item.get("domain", "unknown")),
                "Version": origin.get("version", item.get("validation_version", "zeus-v2.0")),
                "Submission Time": item.get("submission_time", item.get("timestamp", "")),
                "Priority": item.get("priority", "Normal"),
                "Current Stage": item.get("lifecycle", "zeus_validation"),
                "Queue State": queue_state,
                "Validation Status": item.get("status", "pending"),
                "Overall Confidence": item.get("confidence", 0.0),
            }
        )
    return pd.DataFrame(rows)


def _build_evidence_df(reports: List[Dict[str, Any]], selected_modes: List[str], search_term: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in reports:
        if not _passes_mode_filter(item, selected_modes):
            continue
        if not _validation_search_match(item, search_term):
            continue
        origin = item.get("research_origin", {}) if isinstance(item.get("research_origin"), dict) else {}
        rows.append(
            {
                "Report ID": item.get("report_id", "unknown"),
                "Source": origin.get("source", item.get("candidate_source_system", "unknown")),
                "Version": origin.get("version", item.get("validation_version", "zeus-v2.0")),
                "Mission": origin.get("mission", item.get("mission", "Institutional Validation Engine")),
                "Submission Date": origin.get("submission_date", item.get("timestamp", "")),
                "Validation Status": item.get("status", "pending"),
                "Evidence Score": item.get("evidence_score", 0.0),
                "Sample Size": item.get("sample_size", 0),
                "Operator Approval": item.get("operator_approval_status", "Pending Operator Review"),
            }
        )
    return pd.DataFrame(rows)


def _build_timeline_df(timeline: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Stage": row.get("stage", "unknown"),
                "Status": row.get("status", "Pending"),
                "Timestamp": row.get("timestamp", ""),
            }
            for row in timeline or []
        ]
    )


def _equity_chart(eq: List[float], initial: float) -> go.Figure:
    x = list(range(len(eq)))
    y = eq
    peaks  = np.maximum.accumulate(y)
    dd     = [(peaks[i] - y[i]) / peaks[i] * 100 for i in range(len(y))]

    fig = go.Figure()
    # Equity line
    fig.add_trace(go.Scatter(
        x=x, y=y, name="Equity",
        line=dict(color="#a78bfa", width=2),
        fill="tozeroy", fillcolor="rgba(123,97,255,0.08)",
        hovertemplate="Bar %{x}<br>$%{y:.2f}<extra></extra>",
    ))
    # Initial balance reference
    fig.add_hline(y=initial, line=dict(color="#7f8c9a", dash="dash", width=1))
    fig.update_layout(title="Equity Curve", height=320, margin=dict(l=0, r=0, t=40, b=0))
    _apply_dark(fig)
    return fig


def _drawdown_chart(eq: List[float]) -> go.Figure:
    peaks = np.maximum.accumulate(eq)
    dd    = [(peaks[i] - eq[i]) / (peaks[i] + 1e-8) * 100 for i in range(len(eq))]
    fig   = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(dd))), y=dd, name="Drawdown %",
        fill="tozeroy", fillcolor="rgba(231,76,60,0.25)",
        line=dict(color="#e74c3c", width=1.5),
        hovertemplate="Bar %{x}<br>DD: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(title="Drawdown (%)", height=200, margin=dict(l=0, r=0, t=40, b=0))
    _apply_dark(fig)
    fig.update_yaxes(autorange="reversed")
    return fig


def _segment_chart(data: Dict[str, Dict], title: str, min_n: int = 3) -> Optional[go.Figure]:
    rows = [(k, v) for k, v in data.items() if v.get("n", 0) >= min_n]
    if not rows:
        return None
    rows.sort(key=lambda kv: kv[1].get("wr", 0), reverse=True)
    labels = [r[0] for r in rows]
    wr     = [r[1].get("wr", 0) * 100 for r in rows]
    ns     = [r[1].get("n", 0) for r in rows]
    pnl    = [r[1].get("pnl", 0) for r in rows]
    colors = ["#27ae60" if w >= 60 else "#f39c12" if w >= 50 else "#e74c3c" for w in wr]
    fig = go.Figure(go.Bar(
        x=wr, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{w:.1f}%  (n={n}, ${p:+.0f})" for w, n, p in zip(wr, ns, pnl)],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}% WR<extra></extra>",
    ))
    fig.add_vline(x=50, line=dict(color="#7f8c9a", dash="dash", width=1))
    fig.update_layout(
        title=title, height=max(200, 60 + 50 * len(rows)),
        margin=dict(l=0, r=120, t=40, b=0),
    )
    fig.update_xaxes(range=[0, max(wr) * 1.25 + 5], title_text="Win Rate %")
    _apply_dark(fig)
    return fig


def _feature_importance_chart(fi: List[Dict]) -> go.Figure:
    labels = [f["feature"] for f in fi]
    values = [f["importance"] for f in fi]
    colors = px.colors.sequential.Purples_r[:len(labels)]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color="#a78bfa"),
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="XGBoost Feature Importance",
        height=max(250, 50 + 35 * len(labels)),
        margin=dict(l=0, r=80, t=40, b=0),
    )
    fig.update_xaxes(title_text="Importance")
    _apply_dark(fig)
    return fig


def _scatter_chart(trades) -> go.Figure:
    if not trades:
        return go.Figure()
    scores  = [t.score for t in trades]
    rrs     = [t.rr for t in trades]
    statuses= [t.status for t in trades]
    grades  = [t.grade for t in trades]
    colors  = ["#27ae60" if s == "won" else "#e74c3c" for s in statuses]
    hover   = [
        f"Grade {g} | Score {sc:.0f}<br>RR: {rr:.2f} | {st}<br>${getattr(t,'pnl',0):+.2f}"
        for g, sc, rr, st, t in zip(grades, scores, rrs, statuses, trades)
    ]
    fig = go.Figure(go.Scatter(
        x=scores, y=rrs,
        mode="markers",
        marker=dict(color=colors, size=7, opacity=0.7, line=dict(width=0)),
        text=hover, hovertemplate="%{text}<extra></extra>",
    ))
    fig.add_vline(x=float(min_score), line=dict(color="#7b61ff", dash="dash", width=1))
    fig.add_hline(y=1.0,             line=dict(color="#7f8c9a",  dash="dot",  width=1))
    fig.update_layout(
        title="Score vs R:R (green=win, red=loss)",
        xaxis_title="Confluence Score",
        yaxis_title="R:R Achieved",
        height=350, margin=dict(l=0, r=0, t=40, b=0),
    )
    _apply_dark(fig)
    return fig


def _pnl_histogram(trades) -> go.Figure:
    if not trades:
        return go.Figure()
    pnls = [t.pnl for t in trades]
    fig  = go.Figure(go.Histogram(
        x=pnls, nbinsx=30,
        marker=dict(color="#a78bfa", line=dict(color="#141924", width=0.5)),
    ))
    fig.add_vline(x=0, line=dict(color="#7f8c9a", width=1, dash="dash"))
    fig.update_layout(
        title="P&L Distribution", xaxis_title="P&L ($)", yaxis_title="Trades",
        height=250, margin=dict(l=0, r=0, t=40, b=0),
    )
    _apply_dark(fig)
    return fig


def _hour_chart(by_hour: Dict[str, Dict]) -> go.Figure:
    hours = sorted(by_hour.items(), key=lambda kv: int(kv[0]))
    hs    = [kv[0] for kv in hours]
    wrs   = [kv[1].get("wr", 0) * 100 for kv in hours]
    ns    = [kv[1].get("n", 0) for kv in hours]
    colors = ["#27ae60" if w >= 60 else "#f39c12" if w >= 50 else "#e74c3c" for w in wrs]
    fig   = go.Figure(go.Bar(
        x=[f"{h}:00" for h in hs], y=wrs,
        marker=dict(color=colors),
        text=[f"{w:.0f}% (n={n})" for w, n in zip(wrs, ns)],
        textposition="outside",
    ))
    fig.add_hline(y=50, line=dict(color="#7f8c9a", dash="dash", width=1))
    fig.update_layout(
        title="Win Rate by UTC Hour",
        xaxis_title="Hour (UTC)", yaxis_title="Win Rate %",
        height=280, margin=dict(l=0, r=0, t=40, b=0),
    )
    _apply_dark(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# KPI card HTML helper
# ─────────────────────────────────────────────────────────────────────────────
def _kpi(label: str, value: str, cls: str = "") -> str:
    return f"""
    <div class="kpi-card">
      <div class="label">{label}</div>
      <div class="value {cls}">{value}</div>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Tag helpers
# ─────────────────────────────────────────────────────────────────────────────
def _tag(label: str, kind: str) -> str:
    return f'<span class="tag tag-{kind}">{label}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# No-result placeholder
# ─────────────────────────────────────────────────────────────────────────────
validation_status = _load_json_silent(_ZEUS_VALIDATION_STATUS_F, {})
validation_reports = _load_jsonl_silent(_ZEUS_VALIDATION_REPORTS_F)
if not validation_reports and isinstance(validation_status, dict):
    validation_reports = list(validation_status.get("reports", []) or [])

validation_summary = validation_status.get("summary", {}) if isinstance(validation_status, dict) else {}
validation_queue = (validation_status.get("queue", {}) or {}).get("items", validation_reports) if isinstance(validation_status, dict) else validation_reports
validation_queue_df = _build_queue_df(
    validation_queue if isinstance(validation_queue, list) else [],
    validation_modes_selected,
    queue_states_selected,
    validation_search,
)
evidence_df = _build_evidence_df(validation_reports, validation_modes_selected, validation_search)

st.markdown("### Institutional Validation Overview")
vo1, vo2, vo3, vo4 = st.columns(4)
with vo1:
    st.markdown(_kpi("Pending Validations", str(validation_summary.get("pending_validations", 0)), "kpi-gold"), unsafe_allow_html=True)
with vo2:
    st.markdown(_kpi("Running Validations", str(validation_summary.get("running_validations", 0)), "kpi-blue"), unsafe_allow_html=True)
with vo3:
    st.markdown(_kpi("Validated Research", str(validation_summary.get("validated_research", 0)), "kpi-green"), unsafe_allow_html=True)
with vo4:
    st.markdown(_kpi("Rejected Research", str(validation_summary.get("rejected_research", 0)), "kpi-red"), unsafe_allow_html=True)

vo5, vo6, vo7, vo8 = st.columns(4)
with vo5:
    st.markdown(_kpi("Validation Success Rate", f"{float(validation_summary.get('validation_success_rate', 0.0) or 0.0):.1%}", "kpi-green"), unsafe_allow_html=True)
with vo6:
    st.markdown(_kpi("Evidence Library Size", str(validation_summary.get("evidence_library_size", len(validation_reports))), "kpi-purple"), unsafe_allow_html=True)
with vo7:
    st.markdown(_kpi("Avg Validation Duration", str(validation_summary.get("average_validation_duration", 0.0)), "kpi-blue"), unsafe_allow_html=True)
with vo8:
    st.markdown(_kpi("Research Throughput", str(validation_summary.get("research_throughput", 0.0)), "kpi-gold"), unsafe_allow_html=True)

val_tab_overview, val_tab_queue, val_tab_reports, val_tab_evidence = st.tabs([
    "🏛 Validation Overview", "📬 Validation Queue", "📑 Validation Reports", "🧾 Evidence Library"
])

with val_tab_overview:
    # ── Pattern Validation Board ───────────────────────────────────────────────
    st.markdown("#### 🔬 Pattern Validation Board")
    st.caption(
        "Validates Hermes outputs: pattern validity, statistical significance, counterfactual analysis, "
        "replay validation, regime robustness, pattern stability, and knowledge approval. "
        "Each row is an independent Hermes research candidate."
    )
    _pvb = [r for r in validation_reports
            if str(r.get("domain", "") or r.get("candidate_source_system", "")).lower() in ("pattern", "hermes")]
    _pvb_passed   = sum(1 for r in _pvb if str(r.get("status", "")).lower() in ("passed", "approved"))
    _pvb_rejected = sum(1 for r in _pvb if str(r.get("status", "")).lower() in ("failed", "rejected"))
    _pvb_running  = sum(1 for r in _pvb if str(r.get("status", "")).lower() == "running")
    _pvb_pending  = sum(1 for r in _pvb if str(r.get("status", "")).lower() in ("pending", "queued"))
    _pvb_approved = sum(1 for r in _pvb if bool(r.get("approved_for_adoption")))
    _pvb_confs    = [float(r.get("confidence") or 0) for r in _pvb if r.get("confidence") is not None]
    _pvb_evids    = [float(r.get("evidence_score") or 0) for r in _pvb if r.get("evidence_score") is not None]
    _pvb_tp       = f"{_pvb_passed}/{max(1, len(_pvb))}"

    pb1, pb2, pb3, pb4, pb5, pb6, pb7, pb8, pb9 = st.columns(9)
    pb1.metric("Pending",       _pvb_pending)
    pb2.metric("Running",       _pvb_running)
    pb3.metric("Validated",     _pvb_passed)
    pb4.metric("Approved",      _pvb_approved)
    pb5.metric("Rejected",      _pvb_rejected)
    pb6.metric("Queue Health",  f"{len(_pvb)} total")
    pb7.metric("Avg Confidence",f"{sum(_pvb_confs)/max(1,len(_pvb_confs)):.2f}" if _pvb_confs else "—")
    pb8.metric("Evidence Quality", f"{sum(_pvb_evids)/max(1,len(_pvb_evids)):.2f}" if _pvb_evids else "—")
    pb9.metric("Throughput",    _pvb_tp)

    st.markdown("---")

    # ── Strategy Validation Board ──────────────────────────────────────────────
    st.markdown("#### ⚙️ Strategy Validation Board")
    st.caption(
        "Validates Prometheus outputs: strategy validation, position sizing, drawdown analysis, risk, "
        "expectancy improvement, portfolio impact, Monte Carlo, and execution approval. "
        "Each row is an independent Prometheus research candidate. Separate ledger from Pattern Board."
    )
    _svb = [r for r in validation_reports
            if str(r.get("domain", "") or r.get("candidate_source_system", "")).lower() not in ("pattern", "hermes")]
    _svb_passed   = sum(1 for r in _svb if str(r.get("status", "")).lower() in ("passed", "approved"))
    _svb_rejected = sum(1 for r in _svb if str(r.get("status", "")).lower() in ("failed", "rejected"))
    _svb_running  = sum(1 for r in _svb if str(r.get("status", "")).lower() == "running")
    _svb_pending  = sum(1 for r in _svb if str(r.get("status", "")).lower() in ("pending", "queued"))
    _svb_approved = sum(1 for r in _svb if bool(r.get("approved_for_adoption")))
    _svb_confs    = [float(r.get("confidence") or 0) for r in _svb if r.get("confidence") is not None]
    _svb_evids    = [float(r.get("evidence_score") or 0) for r in _svb if r.get("evidence_score") is not None]
    _svb_impr     = next((str(r.get("improvement_estimate")) for r in _svb if r.get("improvement_estimate")), "—")
    _svb_tp       = f"{_svb_passed}/{max(1, len(_svb))}"

    sb1, sb2, sb3, sb4, sb5, sb6, sb7, sb8, sb9 = st.columns(9)
    sb1.metric("Pending",       _svb_pending)
    sb2.metric("Running",       _svb_running)
    sb3.metric("Validated",     _svb_passed)
    sb4.metric("Approved",      _svb_approved)
    sb5.metric("Rejected",      _svb_rejected)
    sb6.metric("Exp. Improvement", _svb_impr)
    sb7.metric("Avg Confidence",f"{sum(_svb_confs)/max(1,len(_svb_confs)):.2f}" if _svb_confs else "—")
    sb8.metric("Evidence Quality", f"{sum(_svb_evids)/max(1,len(_svb_evids)):.2f}" if _svb_evids else "—")
    sb9.metric("Throughput",    _svb_tp)

    st.markdown("---")

    st.markdown("### Validation Health")
    vh1, vh2, vh3, vh4, vh5 = st.columns(5)
    vh1.metric("Research Waiting", validation_summary.get("pending_validations", 0))
    vh2.metric("Research Running", validation_summary.get("running_validations", 0))
    vh3.metric("Research Approved", (validation_status.get("queue", {}) or {}).get("counts", {}).get("Approved", 0) if isinstance(validation_status, dict) else 0)
    vh4.metric("Research Rejected", validation_summary.get("rejected_research", 0))
    vh5.metric("Institutional Confidence", validation_summary.get("institutional_confidence", 0.0))

    vh6, vh7, vh8 = st.columns(3)
    vh6.metric("Average Evidence Score", validation_summary.get("average_evidence_score", 0.0))
    vh7.metric("Validation Version", validation_status.get("version", "zeus-v2.0") if isinstance(validation_status, dict) else "zeus-v2.0")
    vh8.metric("Validation Engine", validation_status.get("validation_engine", "zeus") if isinstance(validation_status, dict) else "zeus")

    lifecycle = validation_status.get("lifecycle", {}) if isinstance(validation_status, dict) else {}
    if lifecycle:
        st.markdown("### Validation Lifecycle")
        lifecycle_rows = [
            {"Stage": stage, "Count": (lifecycle.get("counts", {}) or {}).get(stage, 0)}
            for stage in (lifecycle.get("order", []) or [])
        ]
        st.dataframe(pd.DataFrame(lifecycle_rows), use_container_width=True, hide_index=True)

    st.markdown("### Validation Pipeline")
    if validation_reports:
        pipeline_df = pd.DataFrame(
            [
                {"Stage": key, "Status": value}
                for key, value in (validation_reports[0].get("validation_pipeline", {}) or {}).items()
            ]
        )
        if not pipeline_df.empty:
            st.dataframe(pipeline_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No validation reports available yet.")

    st.markdown("### Operator Governance")
    st.info("Zeus validates evidence only. It does not trade, deploy strategy changes, mutate ML models, or automatically adopt improvements. Operator approval remains mandatory.")

with val_tab_queue:
    st.markdown("### Validation Queue")
    if not validation_queue_df.empty:
        st.dataframe(validation_queue_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No validation queue items match the current filters.")

with val_tab_reports:
    st.markdown("### Validation Report Workspace")
    report_options = {
        f"{row.get('report_id', 'unknown')} · {row.get('research_origin', {}).get('research_category', row.get('domain', 'unknown'))}": row
        for row in validation_reports
        if _passes_mode_filter(row, validation_modes_selected) and _validation_search_match(row, validation_search)
    }
    if report_options:
        selected_label = st.selectbox("Select Validation Report", list(report_options.keys()))
        selected_report = report_options[selected_label]

        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Validation Status", selected_report.get("status", "pending"))
        rc2.metric("Overall Confidence", selected_report.get("confidence", 0.0))
        rc3.metric("Evidence Score", selected_report.get("evidence_score", 0.0))
        rc4.metric("Sample Size", selected_report.get("sample_size", 0))

        rc5, rc6, rc7, rc8 = st.columns(4)
        rc5.metric("Improvement Estimate", selected_report.get("improvement_estimate", "Pending validation"))
        rc6.metric("Leakage Status", selected_report.get("leakage_status", "Pending"))
        rc7.metric("Operator Approval", selected_report.get("operator_approval_status", "Pending Operator Review"))
        rc8.metric("Validation Version", selected_report.get("validation_version", "zeus-v2.0"))

        st.markdown("### Research Origin")
        st.json(selected_report.get("research_origin", {}))

        st.markdown("### Validation Results")
        result_rows = [
            {"Validation": "Historical Result", "Status": selected_report.get("historical_result", "Pending")},
            {"Validation": "Walk Forward Result", "Status": selected_report.get("walk_forward_result", "Pending")},
            {"Validation": "Out-of-Sample Result", "Status": selected_report.get("out_of_sample_result", "Pending")},
            {"Validation": "Monte Carlo Result", "Status": selected_report.get("monte_carlo_result", "Pending")},
            {"Validation": "Leakage Status", "Status": selected_report.get("leakage_status", "Pending")},
            {"Validation": "Execution Validation", "Status": selected_report.get("execution_validation_result", "Pending")},
            {"Validation": "Capital Validation", "Status": selected_report.get("capital_validation_result", "Pending")},
        ]
        st.dataframe(pd.DataFrame(result_rows), use_container_width=True, hide_index=True)

        st.markdown("### ML Validation")
        ml_rows = [
            {"Check": "Leakage Check", "Status": selected_report.get("leakage_status", "Pending")},
            {"Check": "Future Data Detection", "Status": "Pending" if selected_report.get("domain") == "feature" else "N/A"},
            {"Check": "Feature Drift", "Status": "Pending" if selected_report.get("domain") == "feature" else "N/A"},
            {"Check": "Feature Importance", "Status": "Pending" if selected_report.get("domain") == "feature" else "N/A"},
            {"Check": "Generalisation", "Status": "Passed" if selected_report.get("generalisation") else "Pending"},
            {"Check": "Training Integrity", "Status": "Pending" if selected_report.get("domain") == "feature" else "N/A"},
            {"Check": "Validation Status", "Status": selected_report.get("status", "pending")},
        ]
        st.dataframe(pd.DataFrame(ml_rows), use_container_width=True, hide_index=True)

        st.markdown("### Capital Validation")
        capital_rows = [
            {"Metric": "Growth Simulation", "Value": selected_report.get("capital_validation_result", "Pending")},
            {"Metric": "Compounding Analysis", "Value": selected_report.get("improvement_estimate", "Pending validation")},
            {"Metric": "Risk of Ruin", "Value": (selected_report.get("metrics", {}) or {}).get("risk_of_ruin", "Pending")},
            {"Metric": "Recovery Efficiency", "Value": (selected_report.get("metrics", {}) or {}).get("recovery_efficiency", "Pending")},
            {"Metric": "Drawdown Stability", "Value": (selected_report.get("metrics", {}) or {}).get("drawdown_stability", "Pending")},
            {"Metric": "Capital Efficiency", "Value": (selected_report.get("metrics", {}) or {}).get("capital_efficiency", "Pending")},
            {"Metric": "Survival Probability", "Value": (selected_report.get("metrics", {}) or {}).get("survival_probability", "Pending")},
        ]
        st.dataframe(pd.DataFrame(capital_rows), use_container_width=True, hide_index=True)

        if str(selected_report.get("domain", "")) == "pattern":
            st.markdown("### Pattern Validation")
            pattern_rows = [
                {"Field": "Pattern Name", "Value": selected_report.get("candidate_id", "unknown")},
                {"Field": "Evidence", "Value": selected_report.get("evidence", {})},
                {"Field": "Historical Win Rate", "Value": (selected_report.get("evidence", {}) or {}).get("win_rate", "Pending")},
                {"Field": "Expectancy", "Value": (selected_report.get("evidence", {}) or {}).get("expectancy", "Pending")},
                {"Field": "Context Stability", "Value": (selected_report.get("metrics", {}) or {}).get("context_stability", "Pending")},
                {"Field": "Edge Strength", "Value": (selected_report.get("metrics", {}) or {}).get("edge_strength", "Pending")},
                {"Field": "Validation Result", "Value": selected_report.get("status", "pending")},
                {"Field": "Recommendation", "Value": selected_report.get("recommendation", "Pending")},
            ]
            st.dataframe(pd.DataFrame(pattern_rows), use_container_width=True, hide_index=True)

        if str(selected_report.get("research_origin", {}).get("source", "")) == "prometheus":
            st.markdown("### Execution Validation")
            execution_rows = [
                {"Field": "Execution Intelligence", "Value": (selected_report.get("research_origin", {}) or {}).get("research_category", "Execution")},
                {"Field": "Execution Policy", "Value": selected_report.get("execution_validation_result", "Pending")},
                {"Field": "Execution Efficiency", "Value": (selected_report.get("metrics", {}) or {}).get("execution_efficiency", "Pending")},
                {"Field": "Liquidity Exposure", "Value": (selected_report.get("metrics", {}) or {}).get("liquidity_exposure", "Pending")},
                {"Field": "Structural Entry Quality", "Value": (selected_report.get("metrics", {}) or {}).get("structural_entry_quality", "Pending")},
                {"Field": "Risk Validation", "Value": (selected_report.get("metrics", {}) or {}).get("risk_validation", "Pending")},
                {"Field": "Capital Impact", "Value": selected_report.get("improvement_estimate", "Pending validation")},
                {"Field": "Expected Improvement", "Value": selected_report.get("improvement_estimate", "Pending validation")},
            ]
            st.dataframe(pd.DataFrame(execution_rows), use_container_width=True, hide_index=True)

        st.markdown("### Timeline")
        timeline_df = _build_timeline_df(selected_report.get("timeline", []) or [])
        if not timeline_df.empty:
            st.dataframe(timeline_df, use_container_width=True, hide_index=True)

        st.markdown("### Validation Recommendation")
        st.info(selected_report.get("recommendation", "Pending Zeus validation."))
    else:
        st.caption("No validation reports match the current filters.")

with val_tab_evidence:
    st.markdown("### Evidence Library Summary")
    evidence_summary = validation_status.get("evidence_library", {}) if isinstance(validation_status, dict) else {}
    if evidence_summary:
        es1, es2, es3, es4 = st.columns(4)
        es1.metric("Validated Strategies", evidence_summary.get("validated_strategies", 0))
        es2.metric("Validated Patterns", evidence_summary.get("validated_patterns", 0))
        es3.metric("Validated Features", evidence_summary.get("validated_features", 0))
        es4.metric("Validated Recommendations", evidence_summary.get("validated_recommendations", 0))
        es5, es6, es7, es8 = st.columns(4)
        es5.metric("Capital Studies", evidence_summary.get("capital_studies", 0))
        es6.metric("Execution Policies", evidence_summary.get("execution_policies", 0))
        es7.metric("Institutional Reports", evidence_summary.get("institutional_reports", 0))
        es8.metric("Historical Validation Records", evidence_summary.get("historical_validation_records", 0))
    if not evidence_df.empty:
        st.markdown("### Searchable Evidence Library")
        st.dataframe(evidence_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No evidence rows match the current filters.")

res = st.session_state.zeus_result

if res is None:
    st.markdown("""
    <div style="text-align:center; padding:80px 0; color:#7f8c9a;">
      <div style="font-size:4rem;">⚡</div>
            <h2 style="color:#a78bfa; margin:16px 0 8px;">Zeus Institutional Validation Workspace is ready</h2>
      <p>Configure the backtest in the sidebar and click <strong>▶ Run Zeus Backtest</strong>.</p>
      <p style="font-size:0.85rem; margin-top:20px;">
                Validation overview, queue, reports, and evidence library are active above.<br>
                Existing backtesting remains unchanged: Zeus can still fetch historical data, run the full
                Prometheus analysis pipeline on every <em>signal_stride</em> bars, simulate limit and market
                entries, and produce the retained backtest report with ML-driven insights.
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Results UI
# ─────────────────────────────────────────────────────────────────────────────
tab_ov, tab_seg, tab_trades, tab_ml, tab_insights = st.tabs([
    "📊 Overview", "🎯 Segments", "📋 Trade Log", "🤖 ML & Patterns", "💡 What Works"
])


# ═══════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════
with tab_ov:
    # ── KPI cards ─────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    ret_cls  = "kpi-green" if res.total_return_pct >= 0 else "kpi-red"
    dd_cls   = "kpi-red"   if res.max_drawdown_pct > 0.25 else "kpi-warn" if res.max_drawdown_pct > 0.12 else "kpi-green"
    pf_cls   = "kpi-green" if res.profit_factor >= 1.5  else "kpi-warn"  if res.profit_factor >= 1.0  else "kpi-red"
    wr_cls   = "kpi-green" if res.win_rate >= 0.60      else "kpi-warn"  if res.win_rate >= 0.50      else "kpi-red"

    with k1: st.markdown(_kpi("Return",          f"{res.total_return_pct:+.1f}%",   ret_cls), unsafe_allow_html=True)
    with k2: st.markdown(_kpi("Win Rate",         f"{res.win_rate:.1%}",             wr_cls),  unsafe_allow_html=True)
    with k3: st.markdown(_kpi("Profit Factor",    f"{res.profit_factor:.2f}",        pf_cls),  unsafe_allow_html=True)
    with k4: st.markdown(_kpi("Max Drawdown",     f"{res.max_drawdown_pct:.1%}",     dd_cls),  unsafe_allow_html=True)
    with k5: st.markdown(_kpi("Sharpe",           f"{res.sharpe_ratio:.2f}",         "kpi-purple"), unsafe_allow_html=True)
    with k6: st.markdown(_kpi("Avg R:R",          f"{res.avg_rr:.2f}",               "kpi-blue"),   unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Run summary ────────────────────────────────
    st.caption(
        f"Backtest complete · {res.total_trades} signals evaluated · "
        f"equity ${res.initial_balance:.0f} → ${res.final_equity:.2f}"
    )

    # ── Second row: trade stats ────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(_kpi("Total Trades",  str(res.total_trades),      "kpi-purple"), unsafe_allow_html=True)
    with c2: st.markdown(_kpi("Wins",          str(res.winning_trades),    "kpi-green"),  unsafe_allow_html=True)
    with c3: st.markdown(_kpi("Losses",        str(res.losing_trades),     "kpi-red"),    unsafe_allow_html=True)
    with c4: st.markdown(_kpi("Expectancy",    f"${res.expectancy:.2f}",
                               "kpi-green" if res.expectancy >= 0 else "kpi-red"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Balance ────────────────────────────────────
    st.markdown(
        f"**Balance:** ${res.initial_balance:.2f} → "
        f"<span style='color:{'#27ae60' if res.final_equity > res.initial_balance else '#e74c3c'}; "
        f"font-weight:700; font-size:1.2rem'>${res.final_equity:.2f}</span>",
        unsafe_allow_html=True,
    )

    # ── Equity curve ──────────────────────────────
    st.plotly_chart(_equity_chart(res.equity_curve, res.initial_balance),
                    use_container_width=True)

    # ── Drawdown ──────────────────────────────────
    st.plotly_chart(_drawdown_chart(res.equity_curve), use_container_width=True)

    # ── P&L histogram ─────────────────────────────
    st.plotly_chart(_pnl_histogram(res.trades), use_container_width=True)

    # ── Score vs RR scatter ────────────────────────
    if res.trades:
        st.plotly_chart(_scatter_chart(res.trades), use_container_width=True)

    # ── Hour win rate ──────────────────────────────
    if res.by_hour:
        st.plotly_chart(_hour_chart(res.by_hour), use_container_width=True)


# ═══════════════════════════════════════════════════
# TAB 2 — SEGMENTS
# ═══════════════════════════════════════════════════
with tab_seg:
    _MIN_N = 3   # minimum trades to show a segment

    seg_pairs = [
        (res.by_entry_type,   "By Entry Type (Market vs Limit)"),
        (res.by_zone_type,    "By Zone Type (OB / S&R / No zone)"),
        (res.by_ltf_state,    "By LTF State (one_counter / both_confirmed / unknown)"),
        (res.by_grade,        "By Grade (A / B / C)"),
        (res.by_session,      "By Session (London / NY / Asian)"),
        (res.by_pattern_type, "By Chart Pattern Type"),
        (res.by_regime,       "By Market Regime"),
    ]

    for data, title in seg_pairs:
        if data:
            fig = _segment_chart(data, title, min_n=_MIN_N)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption(f"{title}: not enough data (min {_MIN_N} trades per segment)")


# ═══════════════════════════════════════════════════
# TAB 3 — TRADE LOG
# ═══════════════════════════════════════════════════
with tab_trades:
    if not res.trades:
        st.info("No trades recorded.")
    else:
        # Build DataFrame
        rows = []
        for t in res.trades:
            rows.append({
                "ID":        t.trade_id,
                "Dir":       t.direction,
                "Type":      t.entry_type,
                "Entry":     round(t.entry_price, 4),
                "SL":        round(t.sl_price, 4),
                "TP1":       round(t.tp1_price, 4),
                "Exit":      round(t.exit_price, 4) if t.exit_price else None,
                "Lot":       t.size,
                "P&L $":     t.pnl,
                "R:R":       t.rr,
                "Status":    t.status,
                "Reason":    t.exit_reason,
                "Grade":     t.grade,
                "Score":     round(t.score, 1),
                "LTF":       t.ltf_state,
                "Zone":      t.zone_type,
                "Session":   t.session,
                "Regime":    t.regime,
                "Hour (UTC)":t.hour_utc,
            })
        df = pd.DataFrame(rows)

        # Filters
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            f_status = st.multiselect("Status", ["won", "lost", "open"], default=["won", "lost"])
        with col_f2:
            f_entry  = st.multiselect("Entry type", ["market", "limit"], default=["market", "limit"])
        with col_f3:
            f_grade  = st.multiselect("Grade", ["A", "B", "C", "D", "F"], default=["A", "B", "C"])
        with col_f4:
            f_zone   = st.multiselect("Zone type", ["ob", "sr", "no_zone"], default=["ob", "sr", "no_zone"])

        mask = (
            df["Status"].isin(f_status) &
            df["Type"].isin(f_entry) &
            df["Grade"].isin(f_grade) &
            df["Zone"].isin(f_zone)
        )
        df_show = df[mask].reset_index(drop=True)

        # Colour P&L
        def _colour_pnl(val):
            if isinstance(val, (int, float)):
                return "color: #27ae60" if val >= 0 else "color: #e74c3c"
            return ""

        st.caption(f"Showing {len(df_show)} of {len(df)} trades")
        try:
            styled = df_show.style.map(_colour_pnl, subset=["P&L $"])
        except AttributeError:
            # pandas < 2.1 uses applymap
            styled = df_show.style.applymap(_colour_pnl, subset=["P&L $"])  # type: ignore
        st.dataframe(styled, use_container_width=True, height=500)

        # Download CSV
        csv_bytes = df_show.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download trade log CSV", csv_bytes,
            file_name="zeus_trades.csv", mime="text/csv",
        )


# ═══════════════════════════════════════════════════
# TAB 4 — ML & PATTERNS
# ═══════════════════════════════════════════════════
with tab_ml:
    if not res.feature_importance:
        # Diagnose why ML is empty
        try:
            import xgboost  # noqa: F401
            xgb_ok = True
        except ImportError:
            xgb_ok = False
        if not xgb_ok:
            st.error(
                "**XGBoost is not installed.** Run `pip install xgboost` in the Prometheus "
                "venv then restart the Zeus dashboard."
            )
        elif res.total_trades < 5:
            st.warning(
                f"**Too few closed trades ({res.total_trades}) for ML training** — need ≥ 5. "
                "Increase bar count or lower the minimum grade/score."
            )
        else:
            st.info("ML training did not produce results. Check the terminal for details.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(_kpi("ML Accuracy", f"{res.ml_accuracy:.1%}" if res.ml_accuracy else "N/A",
                              "kpi-purple"), unsafe_allow_html=True)
        with c2:
            st.markdown(_kpi("AUC-ROC", f"{res.ml_roc_auc:.3f}" if res.ml_roc_auc else "N/A",
                              "kpi-blue"), unsafe_allow_html=True)
        with c3:
            n_feats = len(res.feature_importance)
            st.markdown(_kpi("Features", str(n_feats), "kpi-gold"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(_feature_importance_chart(res.feature_importance), use_container_width=True)

        # Feature importance table
        fi_df = pd.DataFrame(res.feature_importance)
        st.dataframe(fi_df, use_container_width=True, height=350)

        # Download feature CSV
        _feat_csv = Path(_ROOT / "outputs" / "scalp_bt_features.csv")
        if _feat_csv.exists():
            st.markdown("---")
            st.markdown("**Training data CSV** (`outputs/scalp_bt_features.csv`)")
            feat_data = _feat_csv.read_bytes()
            st.download_button(
                "⬇ Download feature CSV", feat_data,
                file_name="zeus_features.csv", mime="text/csv",
            )

        # Pattern breakdown
        if res.by_pattern_type:
            st.markdown("---")
            fig = _segment_chart(res.by_pattern_type, "Win Rate by Chart Pattern Type", min_n=2)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        # ML interpretation note
        st.markdown("---")
        st.markdown("""
**Interpreting Feature Importance**

| Feature | What it measures |
|---|---|
| `score` | Confluence score — strongest single predictor |
| `zone_ob` | Entry was at an Order Block — premium accuracy |
| `ltf_state` | LTF momentum at entry (one_counter = best) |
| `grade_A` / `grade_B` | Signal grade — proxy for multi-component alignment |
| `entry_limit` | Limit vs market fill — limit = higher accuracy |
| `sl_atr` | SL width relative to ATR — tight SL = better entries |
| `hour_utc` | Time of day — session behaviour |
| `atr_rank` | Volatility rank — high ATR = bigger moves possible |
        """)


# ═══════════════════════════════════════════════════
# TAB 5 — WHAT WORKS
# ═══════════════════════════════════════════════════
with tab_insights:
    # Insights bullets
    if res.insights:
        st.markdown("### 💡 Data-Driven Insights")
        for ins in res.insights:
            st.markdown(f"""
            <div class="insight-box">• {ins}</div>
            """, unsafe_allow_html=True)
    else:
        st.info("Not enough data for automated insights (need ≥ 5 trades per segment).")

    # Top combinations
    if res.winning_combos:
        st.markdown("---")
        st.markdown("### 🏆 Top Win-Rate Combinations *(n ≥ 5 trades)*")
        for i, c in enumerate(res.winning_combos[:8], 1):
            wr_col = "#27ae60" if c["wr"] >= 0.70 else "#f39c12" if c["wr"] >= 0.55 else "#e74c3c"
            entry_tag = _tag(c["entry_type"], c["entry_type"])
            zone_tag  = _tag(c["zone_type"],  c["zone_type"])
            grade_tag = _tag(f"Grade {c['grade']}", c["grade"])
            ltf_label = c["ltf_state"].replace("_", " ")
            st.markdown(f"""
            <div class="kpi-card" style="margin-bottom:8px;">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                  <span style="font-weight:700; color:#dde1ea;">#{i}</span>&nbsp;&nbsp;
                  {entry_tag}{zone_tag}{grade_tag}
                  <span style="color:#7f8c9a; font-size:0.85rem; margin-left:6px;">LTF: {ltf_label}</span>
                </div>
                <div style="text-align:right;">
                  <span style="font-size:1.4rem; font-weight:700; color:{wr_col}">{c['wr']:.0%}</span>
                  <span style="color:#7f8c9a; font-size:0.8rem; margin-left:6px;">WR &nbsp;|&nbsp; {c['n']} trades</span>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # General methodology notes
    st.markdown("---")
    st.markdown("""
### 📖 Zeus Methodology Notes

**Why limit orders outperform market orders:**
Zone limit orders fill only when price revisits an OB/S&R edge — this is the exact zone the
institutional order flow is defending. Market entries capture the signal direction but often
enter mid-move rather than at the structural level.

**Why `one_counter` LTF state is optimal:**
When one LTF is aligned and one is counter-trend, price is in a pullback within the larger move.
This is the precise ICT entry model: higher TF bias + lower TF retracement = highest probability fill.
`both_confirmed` (all LTFs aligned) often signals an extended entry — the move is already underway.

**Why Grade A setups justify exceptions:**
Grade A (score ≥ 85) represents stacked confluence across 12 components. Even in adverse sessions
or regimes, Grade A setups have enough structural edge to overcome unfavourable timing.

**Small account behaviour ($50–$120):**
Zeus simulates single-leg TP1 entries with full lot sizing (2% risk rule). This maximises
slot efficiency — two quality positions rather than one large dual-leg position.
    """)

    # Export full JSON report
    if res.feature_importance or res.winning_combos or res.insights:
        st.markdown("---")
        try:
            report_data = {
                "generated_at":     datetime.utcnow().isoformat(),
                "asset":            asset,
                "primary_tf":       tf_opt,
                "initial_balance":  res.initial_balance,
                "final_equity":     res.final_equity,
                "total_return_pct": res.total_return_pct,
                "win_rate":         res.win_rate,
                "profit_factor":    res.profit_factor,
                "max_drawdown_pct": res.max_drawdown_pct,
                "sharpe_ratio":     res.sharpe_ratio,
                "total_trades":     res.total_trades,
                "winning_trades":   res.winning_trades,
                "losing_trades":    res.losing_trades,
                "by_entry_type":    res.by_entry_type,
                "by_zone_type":     res.by_zone_type,
                "by_ltf_state":     res.by_ltf_state,
                "by_grade":         res.by_grade,
                "by_session":       res.by_session,
                "winning_combos":   res.winning_combos,
                "feature_importance": res.feature_importance,
                "ml_accuracy":      res.ml_accuracy,
                "ml_roc_auc":       res.ml_roc_auc,
                "insights":         res.insights,
            }
            report_bytes = json.dumps(report_data, indent=2, default=str).encode("utf-8")
            st.download_button(
                "⬇ Download full JSON report", report_bytes,
                file_name="zeus_report.json", mime="application/json",
            )
        except Exception:
            pass
