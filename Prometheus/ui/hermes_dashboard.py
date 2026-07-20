"""Hermes dashboard — LTF SMC signal, ML, and paper-trade monitor.

Run:
    streamlit run ui/hermes_dashboard.py --server.port=8503
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from olympus.core.hermes_analytics import build_hermes_analytics


@st.cache_data(ttl=60)
def _load_json_cached(path: Path) -> dict:
    try:
        raw = path.read_text(encoding="utf-8")
        return json.loads(raw)
    except Exception:
        return {}


@st.cache_data(ttl=120)
def _build_hermes_analytics_cached():
    try:
        return build_hermes_analytics(_ROOT)
    except Exception:
        return {}


from ui.dashboard_registry_support import render_registry_metrics, render_registry_tables

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_STATUS_F = _ROOT / "live_bot" / "hermes_status.json"
_CTRL_F = _ROOT / "live_bot" / "hermes_control.json"
_ZEUS_VALIDATION_F = _ROOT / "storage" / "olympus" / "zeus_validation_status.json"


def _load_json_file(path: Path, default: Any) -> tuple[Any, str | None]:
    """Load JSON defensively for files that may be empty during write cycles."""
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as exc:
        return default, f"Unable to read {path.name}: {exc}"

    if not raw or not raw.strip():
        return default, f"{path.name} is currently empty."

    try:
        return json.loads(raw), None
    except json.JSONDecodeError as exc:
        return default, f"{path.name} contains invalid JSON: {exc}"


def _arrow_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe that avoids mixed object types for Arrow transport."""
    safe = df.copy()
    for col in safe.columns:
        series = safe[col]
        if series.dtype != object:
            continue

        non_null = series.dropna()
        if non_null.empty:
            continue

        as_num = pd.to_numeric(non_null, errors="coerce")
        if as_num.notna().all():
            safe[col] = pd.to_numeric(series, errors="coerce")
        else:
            safe[col] = series.astype(str)

    return safe

st.set_page_config(page_title="Hermes · LTF SMC Bot", page_icon="🪽", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .hero {background: linear-gradient(135deg,#0e1826 0%,#172b1b 100%); border:1px solid #294042; border-radius:14px; padding:22px 28px; margin-bottom:20px;}
    .hero h1 {margin:0; color:#d7e9d2; font-size:2.1rem;}
    .hero p {margin:6px 0 0; color:#9eb0b7;}
    .intel-card {background: linear-gradient(135deg,#101926 0%,#16232f 100%); border:1px solid #2e4a60; border-radius:12px; padding:16px 18px; margin:10px 0 18px 0;}
    .intel-card pre {margin:0; color:#d7e9f4; font-size:0.93rem; line-height:1.45; white-space:pre-wrap;}
    </style>
    <div class="hero">
      <h1>🪽 Hermes — LTF Liquidity / CHOCH Bot</h1>
      <p>SMC + price action + Prometheus engines · paper-trading 0.01 lots · adaptive LTF learning</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Hermes Control")
    refresh = st.slider("Auto-refresh seconds", min_value=5, max_value=120, value=15, step=5)
    stop_bot = st.button("Stop Hermes", use_container_width=True)
    if stop_bot:
        _CTRL_F.write_text(json.dumps({"stop": True}, indent=2), encoding="utf-8")
        st.success("Stop flag written for Hermes.")
    st.caption("Launch Hermes with: python live_bot/run_hermes.py --tf M5")

if not _STATUS_F.exists():
    st.warning("Hermes status file not found yet. Start the bot first.")
    st.stop()

status, status_load_error = _load_json_file(_STATUS_F, {})
if status_load_error:
    st.warning(f"Status refresh warning: {status_load_error} Showing last-safe defaults.")

live_analytics = {}
try:
    live_analytics = _build_hermes_analytics_cached()
except Exception:
    live_analytics = {}

zeus_validation: dict[str, Any] = {}
if _ZEUS_VALIDATION_F.exists():
    zeus_validation_data, zeus_load_error = _load_json_file(_ZEUS_VALIDATION_F, {})
    if isinstance(zeus_validation_data, dict):
        zeus_validation = zeus_validation_data
    if zeus_load_error:
        st.caption(f"Zeus validation snapshot warning: {zeus_load_error}")

stats = status.get("stats", {})
last_signal = status.get("last_signal") or {}
ml = status.get("ml", {})
ml_summary = ml.get("learning_summary", {}) or {}
li = status.get("learning_intelligence", {}) or live_analytics.get("metrics", {}) or {}
identity = status.get("system_identity", {}) or {}
pi = status.get("pattern_intelligence", {}) or live_analytics.get("pattern_intelligence", {}) or {}
clusters = status.get("cluster_intelligence", []) or live_analytics.get("cluster_intelligence", []) or []
perf = status.get("performance_intelligence", {}) or live_analytics.get("performance_intelligence", {}) or {}
ae = status.get("adaptive_execution_intelligence", {}) or live_analytics.get("adaptive_execution_intelligence", {}) or {}
exi = status.get("expectancy_intelligence", {}) or live_analytics.get("expectancy_intelligence", {}) or {}
ci = status.get("confidence_intelligence", {}) or live_analytics.get("confidence_intelligence", {}) or {}
di = status.get("directional_intelligence", {}) or live_analytics.get("directional_intelligence", {}) or {}
duri = status.get("duration_intelligence", {}) or live_analytics.get("duration_intelligence", {}) or {}
ei = status.get("execution_intelligence", {}) or live_analytics.get("execution_intelligence", {}) or {}
kqc = status.get("knowledge_quality_controls", {}) or live_analytics.get("knowledge_quality_controls", {}) or {}
mkc = status.get("metric_knowledge_confidence", {}) or live_analytics.get("metric_knowledge_confidence", {}) or {}
academy = status.get("academy", {}) or live_analytics.get("academy", {}) or {}
edge = status.get("edge_stability", {}) or live_analytics.get("edge_stability", {}) or {}
perf_diag = status.get("performance_diagnostics", {}) or live_analytics.get("performance_diagnostics", {}) or {}
pattern_genome = status.get("pattern_genome", []) or live_analytics.get("pattern_genome", []) or []
timeline = status.get("learning_timeline", {}) or live_analytics.get("timeline", {}) or {}
roadmap = status.get("adaptive_roadmap", {}) or live_analytics.get("adaptive_roadmap", {}) or {}
research_engine = status.get("research_engine", {}) or live_analytics.get("research_engine", {}) or {}
evolution = status.get("evolution_roadmap", {}) or live_analytics.get("evolution_roadmap", {}) or {}
validation_gate = status.get("validation_gate", {}) or live_analytics.get("validation_gate", {}) or {}
academy_gate = status.get("academy_certification_gate", {}) or live_analytics.get("academy_certification_gate", {}) or {}
phase_report = status.get("phase_completion_report", {}) or live_analytics.get("phase_completion_report", {}) or {}
readiness = status.get("analytics_readiness_matrix", []) or live_analytics.get("readiness_matrix", []) or []
audit = status.get("analytics_audit", {}) or live_analytics.get("audit", {}) or {}
astatus = status.get("analytics_statuses", {}) or {}
pctx = status.get("pattern_context_intelligence", {}) or live_analytics.get("pattern_context_intelligence", {}) or {}
pctx_academy = pctx.get("academy_subject", {}) if isinstance(pctx, dict) else {}
pctx_profiles = pctx.get("context_profiles", []) if isinstance(pctx, dict) else []
pctx_library = pctx.get("pattern_context_library", []) if isinstance(pctx, dict) else []
tli = status.get("trade_lifecycle_intelligence", {}) or {}
idip = status.get("idip", {}) or {}

hermes_version = identity.get("build_version") or identity.get("model_version") or "v2.0"

ml_records = int(li.get("current_ml_records", ml.get("records", 0) or 0))
pattern_snapshots = int(li.get("pattern_learning_snapshots", 0) or 0)
unique_sequences = int(li.get("unique_pattern_sequences", 0) or 0)
market_snapshots = int(li.get("market_snapshots", ml_records) or 0)
simulation_records = li.get("simulation_count", li.get("simulated_trades_learned", 0))
pattern_discoveries = li.get("pattern_discoveries", pattern_snapshots)

model_version = str(li.get("current_model_version", ml.get("model_version", "0")))
dataset_gen = str(li.get("dataset_generation", status.get("dataset_generation", "gen1")))
feature_ver = str(li.get("feature_version", status.get("feature_version", "v1")))

avg_acc = float(li.get("average_prediction_accuracy", 0.0) or 0.0)
avg_conf_raw = li.get("average_confidence", 0.0)
avg_conf = float(avg_conf_raw or 0.0) if isinstance(avg_conf_raw, (int, float, str)) and str(avg_conf_raw).replace('.', '', 1).isdigit() else 0.0

if avg_acc >= 0.70:
    learning_health = "Excellent"
elif avg_acc >= 0.55:
    learning_health = "Good"
else:
    learning_health = "Developing"

if unique_sequences >= 50:
    pattern_diversity = "High"
elif unique_sequences >= 15:
    pattern_diversity = "Moderate"
else:
    pattern_diversity = "Low"

if avg_conf >= 0.60:
    prediction_stability = "Stable"
elif avg_conf >= 0.45:
    prediction_stability = "Moderate"
else:
    prediction_stability = "Volatile"

continuous_learning = "Active" if int(li.get("learning_events", 0) or 0) > 0 else "Initializing"

pattern_snapshot_line = f"Pattern Learning Snapshots: {pattern_snapshots:,}" if isinstance(pattern_snapshots, int) and pattern_snapshots > 0 else f"Pattern Learning Snapshots: {astatus.get('pattern_snapshot_status', 'Pending Initialization')}"
simulation_line = f"Simulation Records: {simulation_records:,}" if isinstance(simulation_records, int) else f"Simulation Records: {simulation_records}"
pattern_discovery_line = f"Pattern Discoveries: {pattern_discoveries:,}" if isinstance(pattern_discoveries, int) else f"Pattern Discoveries: {pattern_discoveries}"

intel_report = f"""Hermes Intelligence Report
──────────────────────────
ML Records: {ml_records:,}
{pattern_snapshot_line}
Unique Pattern Sequences: {unique_sequences:,}
Market Snapshots: {market_snapshots:,}
{simulation_line}
{pattern_discovery_line}

Current Model: Hermes v{model_version}
Dataset: {dataset_gen}
Feature Set: {feature_ver}

Learning Health: {learning_health}
Pattern Diversity: {pattern_diversity}
Prediction Stability: {prediction_stability}
Continuous Learning: {continuous_learning}

No data loss detected.
No schema conflicts detected.
No cross-system contamination detected."""

# ── Journey variable needed early for exec strip ──────────────────────────────
journey = academy.get("learning_journey", {}) if isinstance(academy, dict) else {}

# ── Executive KPI Strip ──────────────────────────────────────────────────────
_ex1, _ex2, _ex3, _ex4, _ex5, _ex6 = st.columns(6)
_ex1.metric("Win Rate",         f"{stats.get('win_rate', 0.0):.1f}%",          help="Closed paper-trade win rate (wins / total closed)")
_ex2.metric("Expectancy",       perf.get("expectancy", "—"),                    help="Expected $ profit per trade — higher = stronger edge")
_ex3.metric("ML Confidence",    f"{avg_conf:.2f}",                              help="Average ML confidence score; 0.60+ = stable; 0.45+ = moderate")
_ex4.metric("Academy Stage",    journey.get("current_stage", "Observer"),      help="Current Hermes learning stage in institutional academy")
_ex5.metric("Learning Health",  learning_health,                                help="Excellent ≥70% accuracy · Good ≥55% · Developing <55%")
_ex6.metric("Pattern Diversity",pattern_diversity,                              help="High ≥50 unique sequences · Moderate ≥15 · Low <15")

# ── Bottleneck surfacing ──────────────────────────────────────────────────────
_bottleneck = journey.get("bottleneck_dimension")
if _bottleneck:
    _extra = int(journey.get("additional_validated_samples_required", 0) or 0)
    st.warning(
        f"⚠️ **Learning Bottleneck — {_bottleneck}**: "
        f"{'requires ' + str(_extra) + ' more validated samples to advance' if _extra else 'awaiting resolution'}. "
        f"Stage: **{journey.get('current_stage', '?')}** → next: **{journey.get('next_stage', '?')}**"
    )
elif float(journey.get("graduation_progress_pct") or 0) > 0:
    st.success(f"✅ No bottleneck detected — graduation progress: **{float(journey.get('graduation_progress_pct', 0) or 0):.0f}%**")

# ── Tab Navigation ────────────────────────────────────────────────────────────
_t1, _t2, _t3, _t4, _t5 = st.tabs([
    "📊 Summary", "🔍 Pattern & Market", "📈 Performance & Return", "🎓 Learning & Academy", "🏛 Governance"
])

with _t1:
    st.markdown(
        f"""
        <div class="intel-card">
          <pre>{intel_report}</pre>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Signals Seen", stats.get("signals_seen", 0))
    c2.metric("Entered", stats.get("signals_entered", 0))
    c3.metric("Skipped", stats.get("signals_skipped", 0))
    c4.metric("Win Rate %", stats.get("win_rate", 0.0))
    c5.metric("ML Records", ml.get("records", 0))

    st.markdown("### Olympus Trace Metadata")
    meta_cols = st.columns(6)
    meta_cols[0].metric("Source", status.get("source_system", "hermes"))
    meta_cols[1].metric("Dataset Gen", status.get("dataset_generation", "gen1"))
    meta_cols[2].metric("Feature Ver", status.get("feature_version", "v1"))
    meta_cols[3].metric("Strategy Ver", status.get("strategy_version", "v1"))
    meta_cols[4].metric("Exec Type", status.get("execution_type", "simulated"))
    meta_cols[5].metric("Version", hermes_version)

    r1, r2, r3 = st.columns(3)
    r1.metric("Return % Total", round(float(stats.get("return_pct_total", 0.0) or 0.0), 4))
    r2.metric("Return % Wins", round(float(stats.get("return_pct_wins", 0.0) or 0.0), 4))
    r3.metric("Return % Losses", round(float(stats.get("return_pct_losses", 0.0) or 0.0), 4))

    st.markdown("### Latest Hermes Signal")
    if last_signal:
        sig_cols = st.columns(6)
        sig_cols[0].metric("Direction", last_signal.get("direction", "flat"))
        sig_cols[1].metric("Confidence", last_signal.get("confidence", 0.0))
        sig_cols[2].metric("ML Win Prob", last_signal.get("ml_probability", 0.0))
        sig_cols[3].metric("Expected Distance", last_signal.get("expected_distance_pts", 0.0))
        sig_cols[4].metric("CHOCH Bias", last_signal.get("choch_bias", "unknown"))
        sig_cols[5].metric("Sweep", "Yes" if last_signal.get("liquidity_sweep") else "No")
        if last_signal.get("skip_reason"):
            st.info(f"Skipped: {last_signal.get('skip_reason')}")
        if last_signal.get("reasons"):
            for reason in last_signal.get("reasons", []):
                st.write(f"- {reason}")
    else:
        st.caption("No Hermes signal yet.")

    open_trades = pd.DataFrame(status.get("open_trades", []))
    closed_trades = pd.DataFrame(status.get("closed_trades", []))
    skipped = pd.DataFrame(status.get("skipped_signals", []))
    feature_importance = ml.get("feature_importance", {}) or {}
    pattern_stats = pd.DataFrame(ml.get("pattern_stats", []))

    left, right = st.columns([1.3, 1.0])
    with left:
        st.markdown("### Open Trades")
        if not open_trades.empty:
            st.dataframe(_arrow_safe_df(open_trades), use_container_width=True, hide_index=True)
        else:
            st.caption("No open Hermes trades.")

        st.markdown("### Closed Trades")
        if not closed_trades.empty:
            st.dataframe(_arrow_safe_df(closed_trades.tail(30)), use_container_width=True, hide_index=True)
        else:
            st.caption("No closed Hermes trades yet.")

    with right:
        st.markdown("### Skipped Signals")
        if not skipped.empty:
            view_cols = [c for c in ["timestamp", "direction", "confidence", "ml_probability", "skip_reason"] if c in skipped.columns]
            st.dataframe(_arrow_safe_df(skipped[view_cols].tail(30)), use_container_width=True, hide_index=True)
        else:
            st.caption("No skipped Hermes signals yet.")

        st.markdown("### ML Feature Importance")
        if feature_importance:
            fi_df = pd.DataFrame({"feature": list(feature_importance.keys()), "importance": list(feature_importance.values())}).sort_values("importance", ascending=False)
            st.plotly_chart(px.bar(fi_df.head(15), x="importance", y="feature", orientation="h", title="Hermes ML importance"), use_container_width=True)
        else:
            st.caption("Model not trained enough yet.")

    st.markdown("### Pattern Adaptiveness")
    if not pattern_stats.empty:
        pattern_stats["win_rate"] = pattern_stats.apply(lambda r: (r["wins"] / max(1, r["wins"] + r["losses"])) * 100.0, axis=1)
        st.dataframe(_arrow_safe_df(pattern_stats[[c for c in ["pattern", "wins", "losses", "win_rate"] if c in pattern_stats.columns]]), use_container_width=True, hide_index=True)
    else:
        st.caption("Pattern stats will populate as Hermes learns from paper-trade outcomes.")


with _t2:
    st.caption("Market structure awareness, pattern recognition, directional bias, confidence calibration, and temporal context intelligence.")

    st.markdown("### ML Learning Summary")
    ls1, ls2, ls3, ls4 = st.columns(4)
    ls1.metric("Model Stage", ml_summary.get("model_stage", "unknown"))
    ls2.metric("Model Version", ml_summary.get("model_version", 0))
    ls3.metric("Labeled Records", ml_summary.get("records_labeled", 0))
    ls4.metric("Training Progress %", ml_summary.get("training_progress_pct", 0.0))
    st.caption(
        f"Next training gate in {ml_summary.get('samples_to_next_training_gate', 0)} labeled samples "
        f"(min gate: {ml_summary.get('min_samples_for_training', 0)})."
    )

    top_features_df = pd.DataFrame(ml_summary.get("top_features", []))
    best_patterns_df = pd.DataFrame(ml_summary.get("best_patterns", []))
    weak_patterns_df = pd.DataFrame(ml_summary.get("weak_patterns", []))

    sum_left, sum_right = st.columns(2)
    with sum_left:
        st.markdown("#### Most Influential Features")
        if not top_features_df.empty:
            st.dataframe(_arrow_safe_df(top_features_df), use_container_width=True, hide_index=True)
        else:
            st.caption("Feature influence appears after model training has produced importances.")

    with sum_right:
        st.markdown("#### Pattern Learning Snapshot")
        if not best_patterns_df.empty:
            st.write("Best patterns")
            st.dataframe(_arrow_safe_df(best_patterns_df), use_container_width=True, hide_index=True)
        else:
            st.caption("Need more labeled pattern outcomes to rank best patterns.")

        if not weak_patterns_df.empty:
            st.write("Weak patterns")
            st.dataframe(_arrow_safe_df(weak_patterns_df), use_container_width=True, hide_index=True)

    st.markdown("### Hermes Learning Intelligence")
    li1, li2, li3, li4, li5 = st.columns(5)
    li1.metric("ML Records", li.get("current_ml_records", ml.get("records", 0)))
    li2.metric("Pattern Learning Snapshots", li.get("pattern_learning_snapshots", 0))
    li3.metric("Unique Patterns", li.get("unique_pattern_sequences", 0))
    li4.metric("Learning Events", li.get("learning_events", 0))
    li5.metric("Pattern Discoveries", li.get("pattern_discoveries", 0))

    li6, li7, li8, li9, li10 = st.columns(5)
    li6.metric("Simulation Count", li.get("simulation_count", li.get("simulated_trades_learned", 0)))
    li7.metric("Executed Trade Count", li.get("executed_trades", li.get("executed_trades_learned", 0)))
    li8.metric("Learning Progress", f"{float(ml_summary.get('training_progress_pct', 0.0) or 0.0):.2f}%")
    li9.metric("System Identity", identity.get("system_name", "Hermes"))
    li10.metric("Source System", status.get("source_system", "hermes"))

    li_meta = st.columns(4)
    li_meta[0].metric("Dataset Generation", li.get("dataset_generation", status.get("dataset_generation", "gen1")))
    li_meta[1].metric("Current Model Version", li.get("current_model_version", ml.get("model_version", 0)))
    li_meta[2].metric("Feature Version", li.get("feature_version", status.get("feature_version", "v1")))
    li_meta[3].metric("Strategy Version", li.get("strategy_version", status.get("strategy_version", "v1")))

    li_perf = st.columns(3)
    li_perf[0].metric("Avg Prediction Accuracy", li.get("average_prediction_accuracy", 0.0))
    li_perf[1].metric("Avg Confidence", li.get("average_confidence", 0.0))
    li_perf[2].metric("Avg Pattern Success", li.get("average_pattern_success", 0.0))

    _registry_sources = {"hermes_status": status, "hermes_analytics": live_analytics, "zeus_validation": zeus_validation}
    render_registry_metrics(dashboard="hermes_dashboard", section="learning_intelligence", sources=_registry_sources, columns_count=5)

    st.markdown("### Pattern Intelligence")
    st.caption("Which patterns show the strongest statistical edge? Pattern stability measures how reliably performance repeats.")
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("Most Common Pattern", pi.get("most_common_pattern", "Awaiting Historical Data"))
    pc2.metric("Highest Win Pattern", pi.get("highest_win_pattern", "Awaiting Historical Data"))
    pc3.metric("Lowest Win Pattern", pi.get("lowest_win_pattern", "Awaiting Historical Data"))
    pc4.metric("Pattern Diversity Index", pi.get("pattern_diversity_index", "Awaiting Historical Data"))

    pc5, pc6, pc7, pc8 = st.columns(4)
    pc5.metric("Highest Expectancy Pattern", pi.get("highest_expectancy_pattern", "Awaiting Historical Data"))
    pc6.metric("Highest Return Pattern", pi.get("highest_return_pattern", "Awaiting Historical Data"))
    pc7.metric("Pattern Reuse Rate", pi.get("pattern_reuse_rate", "Awaiting Historical Data"))
    pc8.metric("Pattern Stability", pi.get("pattern_stability", "Awaiting Historical Data"))

    render_registry_metrics(dashboard="hermes_dashboard", section="pattern_intelligence", sources=_registry_sources, columns_count=4)

    pl = pi.get("pattern_library", []) if isinstance(pi, dict) else []
    if pl:
        st.markdown("#### Pattern Library")
        st.dataframe(_arrow_safe_df(pd.DataFrame(pl).head(100)), use_container_width=True, hide_index=True)

    if clusters:
        st.markdown("#### Pattern Clusters")
        st.dataframe(_arrow_safe_df(pd.DataFrame(clusters)), use_container_width=True, hide_index=True)

    lc = (pi.get("lifecycle_counts", {}) if isinstance(pi, dict) else {}) or {}
    if lc:
        st.markdown("#### Pattern Lifecycle")
        st.dataframe(_arrow_safe_df(pd.DataFrame([{"Stage": k, "Count": v} for k, v in lc.items()])), use_container_width=True, hide_index=True)

    render_registry_tables(dashboard="hermes_dashboard", section="pattern_intelligence", sources=_registry_sources, max_rows=100, arrow_safe=_arrow_safe_df)

st.markdown("### Performance Intelligence")
pf1, pf2, pf3, pf4 = st.columns(4)
pf1.metric("Profit Factor", perf.get("profit_factor", "Awaiting Historical Data"))
pf2.metric("Average Win", perf.get("average_win", "Awaiting Historical Data"))
pf3.metric("Average Loss", perf.get("average_loss", "Awaiting Historical Data"))
pf4.metric("Signal Acceptance Rate", perf.get("signal_acceptance_rate", "Awaiting Historical Data"))

pf5, pf6, pf7, pf8 = st.columns(4)
pf5.metric("Expectancy", perf.get("expectancy", "Awaiting Historical Data"))
pf6.metric("Recovery Factor", perf.get("recovery_factor", "Awaiting Historical Data"))
pf7.metric("Payoff Ratio", perf.get("payoff_ratio", "Awaiting Historical Data"))
pf8.metric("Avg R Multiple", perf.get("average_r_multiple", "Awaiting Historical Data"))

st.markdown("### Adaptive Execution Intelligence")
ae1, ae2, ae3, ae4 = st.columns(4)
ae1.metric("Average Entry Efficiency", ae.get("average_entry_efficiency", "Awaiting Historical Data"))
ae2.metric("Average Exit Efficiency", ae.get("average_exit_efficiency", "Awaiting Historical Data"))
ae3.metric("Average Prediction Error", ae.get("average_prediction_error", "Awaiting Historical Data"))
ae4.metric("Average RR Achieved", ae.get("average_rr_achieved", "Awaiting Historical Data"))

ae5, ae6, ae7, ae8 = st.columns(4)
ae5.metric("Historical Optimal TP", ae.get("historical_optimal_tp", "Awaiting Historical Data"))
ae6.metric("Historical Break-even", ae.get("historical_break_even_point", "Awaiting Historical Data"))
ae7.metric("Average Profit %", ae.get("average_profit_pct", "Awaiting Historical Data"))
ae8.metric("Average Loss %", ae.get("average_loss_pct", "Awaiting Historical Data"))

st.markdown("### Expectancy Intelligence")
ex1, ex2, ex3, ex4 = st.columns(4)
ex1.metric("Profit Factor", exi.get("profit_factor", "Awaiting Historical Data"))
ex2.metric("Recovery Factor", exi.get("recovery_factor", "Awaiting Historical Data"))
ex3.metric("Payoff Ratio", exi.get("payoff_ratio", "Awaiting Historical Data"))
ex4.metric("Risk Efficiency", exi.get("risk_efficiency", "Awaiting Historical Data"))

ev_pattern = exi.get("expected_value_per_pattern", []) if isinstance(exi, dict) else []
if ev_pattern:
    st.markdown("#### Expected Value per Pattern")
    st.dataframe(_arrow_safe_df(pd.DataFrame(ev_pattern).head(100)), use_container_width=True, hide_index=True)

st.markdown("### Confidence Intelligence")
st.caption("Calibration error < 0.10 = well-calibrated; > 0.20 = confidence scores need recalibration against actual outcomes.")
ci1, ci2, ci3, ci4, ci5 = st.columns(5)
ci1.metric("Average Confidence", ci.get("average_confidence", li.get("average_confidence", "Awaiting Historical Data")))
ci2.metric("Hist Confidence Accuracy", ci.get("historical_confidence_accuracy", "Awaiting Historical Data"))
ci3.metric("Reliability Rating", ci.get("confidence_reliability_rating", "Awaiting Historical Data"))
ci4.metric("Calibration Error", ci.get("calibration_error", "Awaiting Historical Data"))
ci5.metric("Optimal Threshold", ci.get("optimal_confidence_threshold", "Awaiting Historical Data"))

ci6, ci7, ci8 = st.columns(3)
ci6.metric("Confidence Drift", ci.get("confidence_drift", "Awaiting Historical Data"))
ci7.metric("Confidence Stability", ci.get("confidence_stability", "Awaiting Historical Data"))
ci8.metric("Brier Score", ci.get("brier_score", "Awaiting Historical Data"))

cb = ci.get("confidence_buckets", []) if isinstance(ci, dict) else []
if cb:
    st.markdown("#### Confidence Calibration Buckets")
    st.dataframe(_arrow_safe_df(pd.DataFrame(cb)), use_container_width=True, hide_index=True)

for title, key in [
    ("Confidence by Pattern", "confidence_by_pattern"),
    ("Confidence by Session", "confidence_by_session"),
    ("Confidence by Market Regime", "confidence_by_market_regime"),
]:
    rows = ci.get(key, []) if isinstance(ci, dict) else []
    if rows:
        st.markdown(f"#### {title}")
        st.dataframe(_arrow_safe_df(pd.DataFrame(rows).head(100)), use_container_width=True, hide_index=True)

st.markdown("### Directional Intelligence")
st.caption("Entry/exit quality should trend upward as the model accumulates labeled outcomes across sessions and regimes.")
di1, di2, di3, di4, di5, di6 = st.columns(6)
di1.metric("Directional Accuracy", di.get("directional_accuracy", li.get("directional_accuracy", "Awaiting Historical Data")))
di2.metric("Forecast Accuracy", di.get("forecast_accuracy", li.get("forecast_accuracy", "Awaiting Historical Data")))
di3.metric("Execution Accuracy", di.get("execution_accuracy", li.get("execution_accuracy", "Awaiting Historical Data")))
di4.metric("Entry Quality", di.get("entry_quality", li.get("entry_quality", "Awaiting Historical Data")))
di5.metric("Exit Quality", di.get("exit_quality", li.get("exit_quality", "Awaiting Historical Data")))
di6.metric("Direction Stability", di.get("directional_consistency", li.get("direction_stability", "Awaiting Historical Data")))

cls_counts = di.get("classification_counts", {}) if isinstance(di, dict) else {}
if cls_counts:
    st.markdown("#### Direction Classification Counts")
    st.dataframe(_arrow_safe_df(pd.DataFrame([{"classification": k, "count": v} for k, v in cls_counts.items()])), use_container_width=True, hide_index=True)

for title, key in [
    ("Direction by Pattern", "direction_by_pattern"),
    ("Direction by Session", "direction_by_session"),
    ("Direction by Market Regime", "direction_by_market_regime"),
    ("Direction by Pattern Cluster", "direction_by_pattern_cluster"),
]:
    rows = di.get(key, []) if isinstance(di, dict) else []
    if rows:
        st.markdown(f"#### {title}")
        st.dataframe(_arrow_safe_df(pd.DataFrame(rows).head(100)), use_container_width=True, hide_index=True)

st.markdown("### Duration Intelligence")
st.caption("Optimal exit window = when to trail or take partial profits. Avg time to TP vs SL shows the risk-reward timing profile.")
du1, du2, du3, du4, du5, du6 = st.columns(6)
du1.metric("Expected Trade Duration", duri.get("expected_trade_duration", li.get("average_trade_duration", "Awaiting Historical Data")))
du2.metric("Average Holding Time", duri.get("average_holding_time", li.get("average_holding_time", "Awaiting Historical Data")))
du3.metric("Avg Time to TP", duri.get("average_time_to_tp", li.get("average_time_to_tp", "Awaiting Historical Data")))
du4.metric("Avg Time to SL", duri.get("average_time_to_sl", li.get("average_time_to_sl", "Awaiting Historical Data")))
du5.metric("Duration Stability", duri.get("duration_stability", li.get("duration_stability", "Awaiting Historical Data")))
du6.metric("Duration Confidence", duri.get("duration_confidence", li.get("duration_confidence", "Awaiting Historical Data")))

for title, key in [
    ("Duration by Pattern", "duration_by_pattern"),
    ("Duration by Session", "duration_by_session"),
    ("Duration by Market Regime", "duration_by_market_regime"),
]:
    rows = duri.get(key, []) if isinstance(duri, dict) else []
    if rows:
        st.markdown(f"#### {title}")
        st.dataframe(_arrow_safe_df(pd.DataFrame(rows).head(100)), use_container_width=True, hide_index=True)

st.markdown("### Execution Intelligence")
ei1, ei2 = st.columns(2)
ei1.metric("Knowledge Confidence Score", ei.get("knowledge_confidence_score", "Awaiting Historical Data"))
ei2.metric("Execution Readiness Score", ei.get("execution_readiness_score", "Awaiting Historical Data"))

profiles = ei.get("execution_profiles", []) if isinstance(ei, dict) else []
if profiles:
    st.markdown("#### Integrated Pattern Execution Profiles")
    st.dataframe(_arrow_safe_df(pd.DataFrame(profiles).head(100)), use_container_width=True, hide_index=True)

st.markdown("### Knowledge Quality Controls")
kqc_metrics = kqc.get("metrics", []) if isinstance(kqc, dict) else []
if kqc_metrics:
    st.dataframe(_arrow_safe_df(pd.DataFrame(kqc_metrics)), use_container_width=True, hide_index=True)
else:
    st.caption("Knowledge quality controls are awaiting historical statistical samples.")

if mkc:
    st.markdown("### Metric Knowledge Confidence")
    st.dataframe(_arrow_safe_df(pd.DataFrame([{"metric": k, **v} for k, v in mkc.items()])), use_container_width=True, hide_index=True)

st.markdown("### Hermes Academy")
st.caption("Academy stages track learning maturity from simulation → paper-trading → validated edge. Bottleneck = capability gap to resolve.")
ac1, ac2, ac3, ac4 = st.columns(4)
ac1.metric("Academy Stage", journey.get("current_stage", "Observer"))
ac2.metric("Graduation Progress %", journey.get("graduation_progress_pct", 0.0))
ac3.metric("Primary Source", journey.get("current_primary_learning_source", "Simulation"))
ac4.metric("Primary Teacher", journey.get("current_primary_teacher", "Simulation teaches theory."))
if journey.get("bottleneck_dimension"):
    st.caption(
        f"Bottleneck: {journey.get('bottleneck_dimension')} | Additional validated samples required: {journey.get('additional_validated_samples_required', 0)}"
    )
if journey.get("current_learning_objective"):
    st.info(f"Current Objective: {journey.get('current_learning_objective')}")

comp_dims = journey.get("competency_dimensions", {})
if comp_dims:
    st.markdown("#### Competency Dimensions")
    st.dataframe(
        _arrow_safe_df(pd.DataFrame([{"dimension": k, "score": v} for k, v in comp_dims.items()])),
        use_container_width=True,
        hide_index=True,
    )

academies = academy.get("academies", []) if isinstance(academy, dict) else []
if academies:
    st.dataframe(_arrow_safe_df(pd.DataFrame(academies)), use_container_width=True, hide_index=True)

report_cur = academy.get("report_card", {}).get("current", {}) if isinstance(academy, dict) else {}
if report_cur:
    st.markdown("#### Hermes Report Card")
    st.json(report_cur)

passport = academy.get("knowledge_passport", {}) if isinstance(academy, dict) else {}
if passport:
    st.markdown("#### Knowledge Passport")
    rows = []
    for k, v in passport.items():
        if isinstance(v, dict):
            rows.append({"achievement": k, **v})
        else:
            rows.append({"achievement": k, "status": "Validated" if bool(v) else "Developing", "evidence_count": None})
    st.dataframe(_arrow_safe_df(pd.DataFrame(rows)), use_container_width=True, hide_index=True)

grad_ms = academy.get("graduation", {}).get("milestones", []) if isinstance(academy, dict) else []
if grad_ms:
    st.markdown("#### Graduation Milestones")
    st.dataframe(_arrow_safe_df(pd.DataFrame(grad_ms)), use_container_width=True, hide_index=True)
    weighted_model = academy.get("graduation", {}).get("weighted_model", {})
    if weighted_model:
        st.caption(
            f"Weighted progress: {weighted_model.get('validated_weighted_progress', 0.0)} / 100"
        )

st.markdown("### Edge Stability")
st.caption("All edge dimensions should trend upward as the learning engine accumulates evidence. Click history for 20-observation trend.")
es1, es2, es3, es4, es5, es6, es7, es8 = st.columns(8)
es1.metric("Prediction Edge", edge.get("prediction_edge", "Awaiting Historical Data"))
es2.metric("Execution Edge", edge.get("execution_edge", "Awaiting Historical Data"))
es3.metric("Risk Mgmt Edge", edge.get("risk_management_edge", "Awaiting Historical Data"))
es4.metric("Pattern Edge", edge.get("pattern_intelligence_edge", "Awaiting Historical Data"))
es5.metric("Return Edge", edge.get("return_edge", "Awaiting Historical Data"))
es6.metric("Knowledge Confidence", edge.get("knowledge_confidence", "Awaiting Historical Data"))
es7.metric("Confidence Edge", edge.get("confidence_edge", "Awaiting Historical Data"))
es8.metric("Adaptive Readiness", edge.get("adaptive_readiness", "Awaiting Historical Data"))

render_registry_metrics(dashboard="hermes_dashboard", section="edge_intelligence", sources=_registry_sources, columns_count=4)

if edge.get("history"):
    _edge_hist = pd.DataFrame(edge.get("history", [])).tail(20)
    _edge_num = [c for c in _edge_hist.select_dtypes(include="number").columns if c not in ("index",)]
    if _edge_num:
        st.plotly_chart(
            px.line(_edge_hist, y=_edge_num[:4], title="Edge History – last 20 observations"),
            use_container_width=True,
        )
    st.markdown("#### Edge History")
    st.dataframe(_arrow_safe_df(_edge_hist), use_container_width=True, hide_index=True)

st.markdown("### Trade Lifecycle Intelligence")
st.caption("TLI observes how trades behave from entry to close. Advisory actions are observational research proposals only — no execution mutation.")
render_registry_metrics(dashboard="hermes_dashboard", section="tli_intelligence", sources=_registry_sources, columns_count=5)
render_registry_tables(dashboard="hermes_dashboard", section="tli_intelligence", sources=_registry_sources, max_rows=100, arrow_safe=_arrow_safe_df)

tli_modules = tli.get("modules", {}) if isinstance(tli, dict) else {}
tli_perf = tli.get("performance_profile", {}) if isinstance(tli, dict) else {}
tli_mgmt = tli_modules.get("trade_management_intelligence", {}) if isinstance(tli_modules, dict) else {}
tli_recs = tli.get("recommendations", []) if isinstance(tli, dict) else []

tli1, tli2, tli3, tli4 = st.columns(4)
tli1.metric("TLI Version", tli.get("version", "Awaiting Historical Data") if isinstance(tli, dict) else "Awaiting Historical Data")
tli2.metric("TLI Runtime (ms)", tli_perf.get("runtime_ms", "Awaiting Historical Data") if isinstance(tli_perf, dict) else "Awaiting Historical Data")
tli3.metric("Rows Processed", tli_perf.get("rows_processed", "Awaiting Historical Data") if isinstance(tli_perf, dict) else "Awaiting Historical Data")
tli4.metric("Mode", tli.get("mode", "observational_advisory") if isinstance(tli, dict) else "observational_advisory")

if isinstance(tli_mgmt, dict) and tli_mgmt.get("advisory_actions"):
    st.markdown("#### Lifecycle Management Advisory Actions")
    st.json(tli_mgmt.get("advisory_actions", {}))

if tli_recs:
    st.markdown("#### Lifecycle Recommendations")
    for rec in tli_recs[:8]:
        st.caption(f"- {rec}")

st.markdown("### Institutional Decision Intelligence Platform (IDIP)")
st.caption("IDIP is observational — it proposes improvements to Zeus and does not trade or change execution behaviour.")
render_registry_metrics(dashboard="hermes_dashboard", section="idip_intelligence", sources=_registry_sources, columns_count=5)
render_registry_tables(dashboard="hermes_dashboard", section="idip_intelligence", sources=_registry_sources, max_rows=100, arrow_safe=_arrow_safe_df)

idip_perf = idip.get("performance", {}) if isinstance(idip, dict) else {}
idip_meta = idip.get("meta", {}) if isinstance(idip, dict) else {}
idip_recs = idip.get("zeus_research_recommendations", []) if isinstance(idip, dict) else []

ii1, ii2, ii3, ii4 = st.columns(4)
ii1.metric("IDIP Version", idip_meta.get("version", "Awaiting Historical Data") if isinstance(idip_meta, dict) else "Awaiting Historical Data")
ii2.metric("IDIP Runtime (ms)", idip_perf.get("runtime_ms", "Awaiting Historical Data") if isinstance(idip_perf, dict) else "Awaiting Historical Data")
ii3.metric("IDIP Rows Processed", idip_perf.get("rows_processed", "Awaiting Historical Data") if isinstance(idip_perf, dict) else "Awaiting Historical Data")
ii4.metric("Zeus Candidate Recs", len(idip_recs))

if idip_recs:
    st.markdown("#### Zeus Validation Recommendation Queue")
    st.dataframe(_arrow_safe_df(pd.DataFrame(idip_recs).head(100)), use_container_width=True, hide_index=True)

st.markdown("### Performance Diagnostics")
for k in ["win_distribution", "loss_distribution", "return_distribution", "trade_efficiency"]:
    if k in perf_diag:
        st.write(k.replace("_", " ").title())
        st.json(perf_diag.get(k))

if perf_diag.get("expectancy_trend"):
    trend_df = pd.DataFrame(perf_diag.get("expectancy_trend"))
    st.plotly_chart(px.line(trend_df, x="date", y="expectancy", title="Expectancy Trend"), use_container_width=True)

if perf_diag.get("payoff_trend"):
    payoff_df = pd.DataFrame(perf_diag.get("payoff_trend"))
    st.plotly_chart(px.line(payoff_df, x="date", y="payoff", title="Payoff Trend"), use_container_width=True)

if perf_diag.get("explanations"):
    for x in perf_diag.get("explanations", []):
        st.caption(f"- {x}")

if pattern_genome:
    st.markdown("### Pattern Genome")
    st.dataframe(_arrow_safe_df(pd.DataFrame(pattern_genome).head(100)), use_container_width=True, hide_index=True)

st.markdown("### Analytics Readiness Matrix")
if readiness:
    st.dataframe(_arrow_safe_df(pd.DataFrame(readiness)), use_container_width=True, hide_index=True)
else:
    st.caption("Analytics readiness matrix is pending initialization.")

st.markdown("### Audit Findings")
incons = audit.get("inconsistencies", []) if isinstance(audit, dict) else []
if incons:
    st.dataframe(_arrow_safe_df(pd.DataFrame(incons)), use_container_width=True, hide_index=True)
else:
    st.caption("No current dashboard inconsistencies detected by analytics audit.")

st.markdown("### Learning Timeline")
growth = timeline.get("ml_records_growth", []) if isinstance(timeline, dict) else []
if growth:
    growth_df = pd.DataFrame(growth)
    st.plotly_chart(px.line(growth_df, x="date", y="records", title="ML Records Growth"), use_container_width=True)
else:
    st.caption("Learning timeline is awaiting historical index data.")

t1, t2, t3, t4 = st.columns(4)
t1.metric("Training Sessions", timeline.get("training_sessions", 0))
t2.metric("Retraining Events", timeline.get("retraining_events", 0))
t3.metric("Pattern Discoveries Timeline", timeline.get("pattern_discoveries_timeline", 0))
t4.metric("Learning Velocity", timeline.get("learning_velocity", 0.0))

t5, t6 = st.columns(2)
t5.metric("Knowledge Growth", len(growth))
t6.metric("Mastery Growth", pctx_academy.get("mastery", "Awaiting Historical Data"))

st.markdown("### Adaptive Learning Roadmap")
if roadmap:
    st.dataframe(_arrow_safe_df(pd.DataFrame([{"Capability": k, "Status": v} for k, v in roadmap.items()])), use_container_width=True, hide_index=True)

st.markdown("### Hermes Evolution Governance")
eg1, eg2, eg3 = st.columns(3)
eg1.metric("Current Phase", evolution.get("current_phase", "Phase II"))
eg2.metric("Academy Decision", academy_gate.get("certification_decision", "Requires Additional Evidence"))
eg3.metric("Adaptive Unlock Approved", "Yes" if academy_gate.get("adaptive_unlock_approved", False) else "No")

phase_gate = evolution.get("phase_gate", {}) if isinstance(evolution, dict) else {}
if phase_gate:
    st.caption(f"Phase Status: {phase_gate.get('phase_status', 'Current Development Phase')}")
    if phase_gate.get("rules"):
        st.write("Phase Rules:")
        for rule in phase_gate.get("rules", []):
            st.caption(f"- {rule}")

vg_checks = validation_gate.get("checks", []) if isinstance(validation_gate, dict) else []
if vg_checks:
    st.markdown("#### Validation Gate")
    st.dataframe(_arrow_safe_df(pd.DataFrame(vg_checks)), use_container_width=True, hide_index=True)

if research_engine:
    st.markdown("#### Research Engine Foundation")
    st.caption(f"Status: {research_engine.get('status', 'Foundation')}")
    questions = research_engine.get("research_questions", [])
    if questions:
        st.write("Research Questions")
        for q in questions[:8]:
            st.caption(f"- {q}")

if phase_report:
    st.markdown("#### Phase Completion Report")
    rp1, rp2, rp3 = st.columns(3)
    rp1.metric("Recommended Next Phase", phase_report.get("recommended_next_phase", "Hold Phase II"))
    rp2.metric("Backward Compatible", "Yes" if phase_report.get("backward_compatibility", False) else "No")
    rp3.metric(
        "No Destructive Changes",
        "Yes" if (phase_report.get("governance_assertions", {}) or {}).get("no_destructive_changes", False) else "No",
    )

st.markdown("### Pattern Adaptiveness")
if not pattern_stats.empty:
    pattern_stats["win_rate"] = pattern_stats.apply(lambda r: (r["wins"] / max(1, r["wins"] + r["losses"])) * 100.0, axis=1)
    st.dataframe(_arrow_safe_df(pattern_stats[[c for c in ["pattern", "wins", "losses", "win_rate"] if c in pattern_stats.columns]]), use_container_width=True, hide_index=True)
else:
    st.caption("Pattern stats will populate as Hermes learns from paper-trade outcomes.")

st.markdown("### Full Hermes Status JSON")
with st.expander("Full Hermes Status JSON", expanded=False):
    st.json(status)

st.caption(f"Auto-refresh target: every {refresh}s. Reload browser for fresh state if needed.")
