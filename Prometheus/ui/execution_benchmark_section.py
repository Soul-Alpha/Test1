"""Read-only benchmark section within the existing Zeus workspace."""
from __future__ import annotations

import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime, time, timedelta
from pathlib import Path

from olympus.core.execution_benchmark import ExecutionBenchmarkLedger, build_benchmark_report


def render_execution_benchmark(root: Path) -> None:
    import streamlit as st
    from config import CONFIG

    st.markdown("### Backtest → paper → live execution evidence")
    st.caption("Advisory only · explicit operator approval required · no automatic adoption")
    ledger = ExecutionBenchmarkLedger(
        root / "storage" / "olympus" / CONFIG.execution_benchmark.database_name
    )
    try:
        candidates = ledger.candidates()
        if not candidates:
            st.info("No benchmark candidates registered. Existing trade history is unchanged. "
                    "Register a frozen candidate and import finalized observations using "
                    "python -m olympus.core.execution_benchmark_cli. See EXECUTION_BENCHMARK.md.")
            return
        source = st.selectbox("Benchmark bot", sorted({c.source_system for c in candidates}),
                              key="benchmark_source")
        candidate_id = st.selectbox("Frozen candidate",
                                   [c.candidate_id for c in candidates if c.source_system == source],
                                   key="benchmark_candidate")
        today = datetime.now(UTC).date()
        start = st.date_input("Decision window start (UTC)", today - timedelta(days=30),
                              key="benchmark_start")
        end = st.date_input("Decision window end, exclusive (UTC)", today + timedelta(days=1),
                            key="benchmark_end")
        result = build_benchmark_report(
            ledger, source, candidate_id,
            datetime.combine(start, time.min, UTC).isoformat(),
            datetime.combine(end, time.min, UTC).isoformat(),
        )
        candidate = ledger.candidate(source, candidate_id)
        st.caption(f"Frozen candidate fingerprint: {result.candidate_fingerprint}")
        st.caption(f"Account currency: {candidate.account_currency} · venue: {candidate.venue}")
        st.dataframe([asdict(stage) for stage in result.stages], use_container_width=True)
        st.dataframe([asdict(pair) for pair in result.comparisons], use_container_width=True)
        st.warning("Drawdown uses closed trades only. Open-position risk and statistical uncertainty "
                   "require separate validation. Metrics cover complete observations only; missing "
                   "fields block evidence checks. Empty blockers do not authorize live trading.")
        with st.expander("Frozen parameters and limits"):
            st.json(asdict(candidate))
        st.download_button("Download benchmark report", data=__import__("json").dumps(
            result.as_dict(), indent=2, allow_nan=False),
            file_name="zeus-execution-benchmark.json", mime="application/json",
            key="benchmark_download")
    except (ValueError, TypeError, KeyError, sqlite3.Error, OSError) as exc:
        st.error(f"Benchmark evidence unavailable: {exc}")
