"""Hermes — LTF SMC / Price Action paper-trading bot for XAUUSDm.

Hermes reuses Prometheus engines to build a richer lower-timeframe feature
store centered on liquidity sweeps, order blocks, fair value gaps, and CHoCH.
It paper-trades fixed 0.01 lots to bootstrap labeled LTF data for ML.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from prometheus_core import Prometheus, PrometheusResult
from ml.pattern_learner import PatternLearner, SetupRecord, MLPrediction
from engines.market_structure import StructureType
from olympus.contracts import ExecutionType, SourceSystem
from olympus.core.config import OlympusCoreConfig
from olympus.core.hermes_analytics import build_hermes_analytics
from olympus.core.identity import SystemIdentity
from olympus.core.isolation import IsolationGuard, IsolationPolicy
from olympus.core.lineage import EventType, append_lineage_event, build_event
from olympus.core.model_registry import ModelRegistry
from olympus.core.research_repository import ResearchRepository
from olympus.core.pattern_snapshot import append_pattern_snapshot
from olympus.core.return_intelligence import build_return_intelligence
from live_bot.idip_status import update_idip_status
from olympus.core.institutional_decision_intelligence_platform import (
    build_institutional_decision_intelligence_platform,
    write_idip_artifacts,
)
from olympus.core.trade_lifecycle_intelligence import (
    build_trade_lifecycle_intelligence,
    write_trade_lifecycle_intelligence_artifacts,
)
from olympus.core.institutional_dataset_architecture import (
    build_institutional_dataset_architecture,
    write_institutional_dataset_architecture_artifacts,
)
from olympus.core.version_registry import VersionRegistry
from olympus.versions import (
    FEATURE_VERSION,
    HERMES_DATASET_GENERATION,
    HERMES_MODEL_FAMILY,
    HERMES_STRATEGY_VERSION,
)

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None  # type: ignore[assignment]
    MT5_AVAILABLE = False

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
HERMES_STATUS_FILE = _HERE / "hermes_status.json"
HERMES_CONTROL_FILE = _HERE / "hermes_control.json"
HERMES_MODEL_DIR = _ROOT / "models" / "hermes"
HERMES_SOURCE_SYSTEM = SourceSystem.HERMES.value


def _json_safe(value: Any) -> Any:
    """Convert runtime objects (numpy/enum/datetime) into JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(_json_safe(k)): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


@dataclass
class HermesSignal:
    signal_id: str
    timestamp: str
    asset: str
    timeframe: str
    direction: str
    expected_distance_pts: float
    confidence: float
    ml_probability: float
    choch_bias: str
    liquidity_sweep: bool
    fresh_ob_count: int
    fresh_fvg_count: int
    reasons: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class HermesTrade:
    trade_id: str
    signal_id: str
    direction: str
    entry_price: float
    lots: float
    sl_price: float
    tp_price: float
    opened_at: str
    status: str = "open"
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl: float = 0.0
    predicted_distance_pts: float = 0.0
    realized_distance_pts: float = 0.0
    session: str = "unknown"
    regime: str = "unknown"
    trend_state: str = "unknown"
    volatility_state: str = "unknown"
    pattern_name: str = "unknown"
    pattern_family: str = "unknown"
    pattern_cluster: str = "unknown"
    model_version: str = "0"
    feature_version: str = FEATURE_VERSION
    signal_confidence: float = 0.0
    mfe_price_high: float = 0.0
    mfe_price_low: float = 0.0
    mfe_pct: float = 0.0
    mae_pct: float = 0.0
    potential_return_pct: float = 0.0
    potential_loss_pct: float = 0.0
    captured_return_pct: float = 0.0
    lost_opportunity_pct: float = 0.0
    risk_utilization_pct: float = 0.0
    return_efficiency_pct: float = 0.0
    loss_efficiency_pct: float = 0.0
    opportunity_efficiency_pct: float = 0.0
    execution_efficiency_pct: float = 0.0
    exit_quality: str = "Unknown"


class HermesBot:
    """Lower-timeframe paper-trading bot specialized for SMC liquidity logic."""

    def __init__(
        self,
        asset: str = "XAUUSDm",
        timeframe: str = "M5",
        candles: int = 500,
        poll_interval: int = 60,
        fixed_lot: float = 0.01,
        simulate_bootstrap: bool = True,
    ) -> None:
        self.asset = asset
        self.timeframe = timeframe.upper()
        self.candles = candles
        self.poll_interval = poll_interval
        self.fixed_lot = fixed_lot
        self.simulate_bootstrap = simulate_bootstrap
        self._execution_type = ExecutionType.SIMULATED.value
        self._enable_trade_lifecycle_intelligence = (
            str(os.getenv("ENABLE_TRADE_LIFECYCLE_INTELLIGENCE", "1") or "1").strip().lower()
            not in ("0", "false", "no", "off")
        )
        self._enable_idip = (
            str(os.getenv("ENABLE_IDIP", "1") or "1").strip().lower()
            not in ("0", "false", "no", "off")
        )
        self._enable_institutional_dataset_architecture = (
            str(os.getenv("ENABLE_INSTITUTIONAL_DATASET_ARCHITECTURE", "1") or "1").strip().lower()
            not in ("0", "false", "no", "off")
        )
        self._enable_institutional_learning_scientist = (
            str(os.getenv("ENABLE_INSTITUTIONAL_LEARNING_SCIENTIST", "1") or "1").strip().lower()
            not in ("0", "false", "no", "off")
        )
        self._enable_capital_intelligence_engine = (
            str(os.getenv("ENABLE_CAPITAL_INTELLIGENCE_ENGINE", "1") or "1").strip().lower()
            not in ("0", "false", "no", "off")
        )
        self._enable_knowledge_graph_engine = (
            str(os.getenv("ENABLE_KNOWLEDGE_GRAPH_ENGINE", "1") or "1").strip().lower()
            not in ("0", "false", "no", "off")
        )
        self._enable_decision_replay_counterfactual_intelligence = (
            str(os.getenv("ENABLE_DECISION_REPLAY_COUNTERFACTUAL_INTELLIGENCE", "1") or "1").strip().lower()
            not in ("0", "false", "no", "off")
        )
        self._enable_meta_learning_engine = (
            str(os.getenv("ENABLE_META_LEARNING_ENGINE", "1") or "1").strip().lower()
            not in ("0", "false", "no", "off")
        )
        self._enable_autonomous_research_orchestrator = (
            str(os.getenv("ENABLE_AUTONOMOUS_RESEARCH_ORCHESTRATOR", "1") or "1").strip().lower()
            not in ("0", "false", "no", "off")
        )
        self._enable_research_prioritization_engine = (
            str(os.getenv("ENABLE_RESEARCH_PRIORITIZATION_ENGINE", "1") or "1").strip().lower()
            not in ("0", "false", "no", "off")
        )
        self._enable_knowledge_evolution_engine = (
            str(os.getenv("ENABLE_KNOWLEDGE_EVOLUTION_ENGINE", "1") or "1").strip().lower()
            not in ("0", "false", "no", "off")
        )
        self._enable_explainability_engine = (
            str(os.getenv("ENABLE_EXPLAINABILITY_ENGINE", "1") or "1").strip().lower()
            not in ("0", "false", "no", "off")
        )
        self._enable_knowledge_coverage_intelligence = (
            str(os.getenv("ENABLE_KNOWLEDGE_COVERAGE_INTELLIGENCE", "1") or "1").strip().lower()
            not in ("0", "false", "no", "off")
        )
        self._enable_institutional_research_director = (
            str(os.getenv("ENABLE_INSTITUTIONAL_RESEARCH_DIRECTOR", "1") or "1").strip().lower()
            not in ("0", "false", "no", "off")
        )

        self._identity = SystemIdentity(
            system_id="hermes-core",
            system_name="Hermes",
            system_role="adaptive_ai_researcher_execution_specialist",
            development_stage="production",
            owner="olympus",
            model_version="0",
            dataset_generation=HERMES_DATASET_GENERATION,
            build_version="olympus-core-v1",
            feature_version=FEATURE_VERSION,
            strategy_version=HERMES_STRATEGY_VERSION,
        )

        core_cfg = OlympusCoreConfig.from_env(_ROOT)

        # Runtime isolation is strict by default; can be relaxed via explicit flags.
        self._isolation = IsolationGuard(
            system=HERMES_SOURCE_SYSTEM,
            policy=IsolationPolicy(
                allow_cross_system_models=core_cfg.allow_cross_system_models,
                allow_cross_system_datasets=core_cfg.allow_cross_system_datasets,
                allow_cross_system_configs=core_cfg.allow_cross_system_configs,
                allow_cross_system_strategies=core_cfg.allow_cross_system_strategies,
            ),
        )
        if not self._isolation.guard_model_path(HERMES_MODEL_DIR):
            raise RuntimeError(f"Blocked foreign model path for Hermes: {HERMES_MODEL_DIR}")
        if not self._isolation.guard_dataset_path(HERMES_MODEL_DIR / "setups.json"):
            raise RuntimeError("Blocked foreign dataset path for Hermes")
        if not self._isolation.guard_config_path(HERMES_CONTROL_FILE):
            raise RuntimeError("Blocked foreign config path for Hermes")
        if not self._isolation.guard_strategy_path(Path(__file__)):
            raise RuntimeError("Blocked foreign strategy path for Hermes")

        self._version_registry = VersionRegistry(_ROOT)
        self._model_registry = ModelRegistry(_ROOT)

        self.engine = Prometheus()
        self.learner = PatternLearner(model_dir=str(HERMES_MODEL_DIR), min_samples_train=50)

        self._status: dict[str, Any] = {
            "bot": "Hermes",
            "asset": self.asset,
            "timeframe": self.timeframe,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "source_system": HERMES_SOURCE_SYSTEM,
            "dataset_generation": HERMES_DATASET_GENERATION,
            "feature_version": FEATURE_VERSION,
            "strategy_version": HERMES_STRATEGY_VERSION,
            "execution_type": ExecutionType.SIMULATED.value,
            "system_identity": self._identity.as_dict(),
            "last_poll": "",
            "signals": [],
            "open_trades": [],
            "closed_trades": [],
            "skipped_signals": [],
            "ml": {},
            "stats": {
                "signals_seen": 0,
                "signals_entered": 0,
                "signals_skipped": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
            },
        }
        self._open_trades: list[HermesTrade] = []
        self._closed_trades: list[HermesTrade] = []
        self._last_signal: Optional[HermesSignal] = None
        self._symbol_selected = False
        self._bootstrap_done = False
        self._last_df = None
        self._last_feature_meta: dict = {}
        self._return_report_interval = max(5, int(os.getenv("HERMES_RETURN_REPORT_INTERVAL", "10") or 10))
        self._last_return_report_closed_count = 0
        self._research_repository = ResearchRepository(_ROOT)

        # Register the currently loaded model/version as metadata only.
        self._register_runtime_versions()

        # Write an initial status file immediately so the dashboard doesn't
        # show "status file not found" while the first poll is in progress.
        self._write_status()

    def _emit_lineage_event(self, event_type: EventType, payload: Optional[dict[str, Any]] = None) -> None:
        try:
            ev = build_event(
                event_type=event_type,
                source_system=HERMES_SOURCE_SYSTEM,
                instrument=self.asset,
                timeframe=self.timeframe,
                model_version=str(getattr(self.learner, "model_version", 0)),
                feature_version=FEATURE_VERSION,
                strategy_version=HERMES_STRATEGY_VERSION,
                dataset_generation=HERMES_DATASET_GENERATION,
                execution_type=self._execution_type,
                payload=payload or {},
            )
            append_lineage_event(_ROOT, ev)
        except Exception:
            # Never let lineage telemetry break live bot behavior.
            pass

    def _register_runtime_versions(self) -> None:
        try:
            model_ver = str(getattr(self.learner, "model_version", 0))
            rec_count = len(getattr(self.learner, "records", []))
            self._version_registry.register_simple(
                system=HERMES_SOURCE_SYSTEM,
                model_version=model_ver,
                feature_version=FEATURE_VERSION,
                strategy_version=HERMES_STRATEGY_VERSION,
                dataset_generation=HERMES_DATASET_GENERATION,
                record_count=rec_count,
                active=True,
                metadata={"model_family": HERMES_MODEL_FAMILY},
            )
            self._model_registry.register_simple(
                model_name=HERMES_MODEL_FAMILY,
                system=HERMES_SOURCE_SYSTEM,
                version=model_ver,
                training_record_count=rec_count,
                feature_version=FEATURE_VERSION,
                dataset_generation=HERMES_DATASET_GENERATION,
                description="Hermes adaptive pattern learner",
                status="active",
                metadata={"timeframe": self.timeframe, "asset": self.asset},
            )
            self._emit_lineage_event(EventType.VERSION_USED, {"model_family": HERMES_MODEL_FAMILY})
        except Exception:
            pass

    def _append_pattern_snapshot(self, result: PrometheusResult, signal: HermesSignal) -> None:
        """Append a non-destructive market observation snapshot for analytics."""
        try:
            ms = result.ms
            smc = result.smc
            snapshot = {
                "source_system": HERMES_SOURCE_SYSTEM,
                "asset": self.asset,
                "timeframe": self.timeframe,
                "market_structure": {
                    "structure_type": str(getattr(ms, "structure_type", "unknown")),
                    "trend_strength": float(getattr(ms, "trend_strength", 0.0) or 0.0),
                },
                "trend": {
                    "direction": signal.direction,
                },
                "liquidity": {
                    "sweep_present": bool(self._last_feature_meta.get("sweep_present", False)),
                    "liquidity_pool_count": int(self._last_feature_meta.get("liquidity_pool_count", 0) or 0),
                },
                "choch": {
                    "count": int(self._last_feature_meta.get("choch_count", 0) or 0),
                    "bias": str(self._last_feature_meta.get("choch_bias", "unknown")),
                },
                "bos": {
                    "count": len(getattr(ms, "bos_events", []) or []),
                },
                "order_blocks": {
                    "fresh_count": int(self._last_feature_meta.get("fresh_ob_count", 0) or 0),
                    "total_count": len(getattr(smc, "order_blocks", []) or []),
                },
                "fair_value_gaps": {
                    "fresh_count": int(self._last_feature_meta.get("fresh_fvg_count", 0) or 0),
                    "total_count": len(getattr(smc, "fair_value_gaps", []) or []),
                },
                "volatility": {
                    "atr": float(getattr(ms, "current_atr", 0.0) or 0.0),
                },
                "session": {
                    "label": "unknown",
                },
                "prediction": {
                    "signal_id": signal.signal_id,
                    "direction": signal.direction,
                    "ml_probability": float(signal.ml_probability),
                    "expected_distance_pts": float(signal.expected_distance_pts),
                    "skipped": bool(signal.skipped),
                    "skip_reason": signal.skip_reason,
                },
                "confidence": float(signal.confidence),
                "trade_outcome": "pending",
                "dataset_generation": HERMES_DATASET_GENERATION,
                "feature_version": FEATURE_VERSION,
                "strategy_version": HERMES_STRATEGY_VERSION,
                "execution_type": self._execution_type,
            }
            append_pattern_snapshot(_ROOT, snapshot)
        except Exception:
            pass

    def _tf_constant(self):
        if not MT5_AVAILABLE:
            return None
        return {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
        }.get(self.timeframe, mt5.TIMEFRAME_M5)

    def _connect_mt5(self) -> bool:
        if not MT5_AVAILABLE:
            return False
        if mt5.initialize():
            if not self._symbol_selected:
                try:
                    mt5.symbol_select(self.asset, True)
                    self._symbol_selected = True
                except Exception:
                    pass
            return True
        return False

    def _fetch_candles(self, n: Optional[int] = None):
        if pd is None or not MT5_AVAILABLE:
            return None
        if not self._connect_mt5():
            return None
        rates = mt5.copy_rates_from_pos(self.asset, self._tf_constant(), 0, int(n or self.candles))
        if rates is None or len(rates) < 50:
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "volume"}).set_index("time")
        return df[["open", "high", "low", "close", "volume"]]

    def _build_setup_record(self, result: PrometheusResult) -> SetupRecord:
        ms = result.ms
        smc = result.smc
        pat = result.pat
        latest_pattern = pat.patterns[0] if (pat and getattr(pat, "patterns", None)) else None
        choch_count = len(ms.choch_events) if ms else 0
        choch_bias = "unknown"
        if ms and ms.choch_events:
            latest_choch = max(ms.choch_events, key=lambda e: e.index)
            choch_bias = latest_choch.direction

        fresh_obs = [ob for ob in (smc.order_blocks if smc else []) if not ob.mitigated]
        fresh_fvgs = [g for g in (smc.fair_value_gaps if smc else []) if not g.filled]
        stop_hunts = smc.stop_hunts if smc else []
        pools = smc.liquidity_pools if smc else []
        latest_stop_hunt = stop_hunts[-1] if stop_hunts else None

        structure_type = 0
        if ms:
            if ms.structure_type == StructureType.BULLISH:
                structure_type = 1
            elif ms.structure_type == StructureType.BEARISH:
                structure_type = 2
            elif ms.structure_type == StructureType.SIDEWAYS:
                structure_type = 3

        pattern_conf = 0.0
        pattern_name = ""
        if latest_pattern is not None:
            pattern_conf = float(getattr(latest_pattern, "confidence", 0.0) or 0.0)
            pattern_name = str(getattr(latest_pattern, "pattern", "") or "")

        volume_ratio = 1.0
        if pd is not None:
            df = self._last_df
            if df is not None and len(df) >= 20:
                avg_vol = float(df["volume"].tail(20).mean() or 1.0)
                cur_vol = float(df["volume"].iloc[-1] or avg_vol)
                volume_ratio = cur_vol / max(avg_vol, 1e-9)

        entry_price = float(result.current_price or 0.0)
        record = SetupRecord(
            setup_id=str(uuid.uuid4())[:12],
            asset=self.asset,
            timeframe=self.timeframe,
            timestamp=datetime.now(timezone.utc).isoformat(),
            structure_type=structure_type,
            trend_strength=float(ms.trend_strength if ms else 0.0),
            mtf_score=float(result.mtf.alignment_score if result.mtf else 0.0),
            sr_confidence=float(result.sr.nearest_support.confidence if (result.sr and result.sr.nearest_support) else 0.0),
            candlestick_score=float(sum(getattr(c, "confidence", 0.0) or getattr(c, "final_score", 0.0) for c in (result.cs.signals[:2] if result.cs and result.cs.signals else []))),
            pattern_confidence=pattern_conf,
            fib_proximity=1 if (result.fib and getattr(result.fib, "nearest_level", None) is not None) else 0,
            ob_present=1 if fresh_obs else 0,
            stop_hunt=1 if latest_stop_hunt else 0,
            confluence_score=float(result.confluence.total if result.confluence else 0.0),
            volume_ratio=float(volume_ratio),
            pattern_type_id=0,
            prior_trend_aligned=1 if choch_count > 0 or (ms and ms.trend_strength >= 0.55) else 0,
            entry_price=entry_price,
            source_system=HERMES_SOURCE_SYSTEM,
            model_version_used=str(getattr(self.learner, "model_version", 0)),
            feature_version=FEATURE_VERSION,
            strategy_version=HERMES_STRATEGY_VERSION,
            execution_type=ExecutionType.SIMULATED.value,
            dataset_generation=HERMES_DATASET_GENERATION,
        )
        record.tp_price = entry_price
        record.sl_price = entry_price
        record.exit_price = None
        # attach richer Hermes-only metadata for dashboard/logging outside learner schema
        self._last_feature_meta = {
            "choch_bias": choch_bias,
            "choch_count": choch_count,
            "fresh_ob_count": len(fresh_obs),
            "fresh_fvg_count": len(fresh_fvgs),
            "liquidity_pool_count": len(pools),
            "sweep_present": bool(latest_stop_hunt),
            "pattern_name": pattern_name,
            "pattern_family": self._infer_pattern_family(
                result,
                pattern_name,
                sweep_present=bool(latest_stop_hunt),
                fresh_ob_count=len(fresh_obs),
                fresh_fvg_count=len(fresh_fvgs),
            ),
            "pattern_cluster": self._infer_pattern_cluster(
                result,
                pattern_name,
                sweep_present=bool(latest_stop_hunt),
                fresh_ob_count=len(fresh_obs),
                fresh_fvg_count=len(fresh_fvgs),
            ),
        }
        return record

    def _infer_pattern_family(
        self,
        result: PrometheusResult,
        pattern_name: str = "",
        *,
        sweep_present: bool = False,
        fresh_ob_count: int = 0,
        fresh_fvg_count: int = 0,
    ) -> str:
        ms = result.ms
        if sweep_present:
            return "Liquidity Sweep"
        if fresh_ob_count and fresh_fvg_count:
            return "Mitigation Stack"
        if ms and getattr(ms, "structure_type", None) == StructureType.BULLISH:
            return "Bullish Expansion"
        if ms and getattr(ms, "structure_type", None) == StructureType.BEARISH:
            return "Bearish Expansion"
        if pattern_name:
            return "Pattern Driven"
        return "Unknown"

    def _infer_pattern_cluster(
        self,
        result: PrometheusResult,
        pattern_name: str = "",
        *,
        sweep_present: bool = False,
        fresh_ob_count: int = 0,
        fresh_fvg_count: int = 0,
    ) -> str:
        family = self._infer_pattern_family(
            result,
            pattern_name,
            sweep_present=sweep_present,
            fresh_ob_count=fresh_ob_count,
            fresh_fvg_count=fresh_fvg_count,
        )
        direction = str(getattr(result.confluence, "direction", "unknown") or "unknown").lower()
        session = self._current_session_label()
        regime = self._regime_label(result)
        return f"{family} | {direction} | {session} | {regime}"

    def _current_session_label(self) -> str:
        hour = datetime.now(timezone.utc).hour
        if 0 <= hour < 7:
            return "Asian"
        if 7 <= hour < 10:
            return "London Open"
        if 10 <= hour < 13:
            return "London"
        if 13 <= hour < 16:
            return "NY"
        if 16 <= hour < 19:
            return "NY Lunch"
        return "Rollover"

    def _regime_label(self, result: PrometheusResult) -> str:
        ms = result.ms
        if ms is None:
            return "Dead Liquidity"
        if self._last_feature_meta.get("sweep_present"):
            return "Liquidity Sweep"
        if getattr(ms, "trend_strength", 0.0) >= 0.70:
            return "Trend Expansion"
        if getattr(ms, "trend_strength", 0.0) >= 0.50:
            return "Trend Exhaustion"
        if getattr(ms, "structure_type", None) == StructureType.SIDEWAYS:
            return "Mean Reversion"
        return "Compression"

    def _trend_label(self, result: PrometheusResult) -> str:
        direction = str(getattr(result.confluence, "direction", "unknown") or "unknown").lower()
        if direction == "bullish":
            return "bullish"
        if direction == "bearish":
            return "bearish"
        return "sideways"

    def _volatility_label(self, result: PrometheusResult, df) -> str:
        ms = result.ms
        atr = float(getattr(ms, "current_atr", 0.0) or 0.0)
        if df is not None and len(df) >= 20 and "high" in df and "low" in df:
            recent = df.tail(20)
            spread_proxy = float((recent["high"] - recent["low"]).mean() or 0.0)
            if spread_proxy > 0:
                ratio = atr / max(spread_proxy, 1e-9)
                if ratio >= 1.25:
                    return "High"
                if ratio >= 0.80:
                    return "Moderate"
                return "Low"
        if atr >= 2.5:
            return "High"
        if atr >= 1.0:
            return "Moderate"
        return "Low"

    def _refresh_trade_excursions(self, trade: HermesTrade, df) -> None:
        if df is None or len(df) == 0:
            return
        opened = datetime.fromisoformat(trade.opened_at.replace("Z", "+00:00"))
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=timezone.utc)
        recent = df[df.index >= opened] if hasattr(df, "index") else df
        if recent is None or len(recent) == 0:
            recent = df.tail(1)
        high = float(recent["high"].max())
        low = float(recent["low"].min())
        trade.mfe_price_high = max(trade.mfe_price_high, high)
        trade.mfe_price_low = low if trade.mfe_price_low == 0.0 else min(trade.mfe_price_low, low)
        entry = trade.entry_price
        if entry <= 0:
            return
        if trade.direction == "long":
            trade.mfe_pct = max(trade.mfe_pct, max(0.0, ((trade.mfe_price_high - entry) / entry) * 100.0))
            trade.mae_pct = max(trade.mae_pct, max(0.0, ((entry - trade.mfe_price_low) / entry) * 100.0))
        else:
            trade.mfe_pct = max(trade.mfe_pct, max(0.0, ((entry - trade.mfe_price_low) / entry) * 100.0))
            trade.mae_pct = max(trade.mae_pct, max(0.0, ((trade.mfe_price_high - entry) / entry) * 100.0))

    def _finalize_return_fields(self, trade: HermesTrade) -> None:
        trade.potential_return_pct = max(0.0, trade.mfe_pct)
        trade.potential_loss_pct = max(0.0, trade.mae_pct)
        realized_return_pct = self._return_pct(trade)
        trade.captured_return_pct = round(max(0.0, min(100.0, (max(0.0, realized_return_pct) / max(1e-9, trade.potential_return_pct)) * 100.0)) if trade.potential_return_pct > 0 else 0.0, 4)
        trade.lost_opportunity_pct = round(max(0.0, 100.0 - trade.captured_return_pct), 4)
        trade.risk_utilization_pct = round(max(0.0, min(100.0, (trade.potential_loss_pct / max(1e-9, trade.potential_return_pct + trade.potential_loss_pct)) * 100.0)) if (trade.potential_return_pct + trade.potential_loss_pct) > 0 else 0.0, 4)
        trade.return_efficiency_pct = round(max(-100.0, min(100.0, (realized_return_pct / max(1e-9, trade.potential_return_pct + trade.potential_loss_pct)) * 100.0)) if (trade.potential_return_pct + trade.potential_loss_pct) > 0 else 0.0, 4)
        trade.loss_efficiency_pct = round(max(0.0, min(100.0, (abs(realized_return_pct) / max(1e-9, trade.potential_loss_pct)) * 100.0)) if trade.potential_loss_pct > 0 and realized_return_pct < 0 else 0.0, 4)
        trade.opportunity_efficiency_pct = round(trade.captured_return_pct, 4)
        trade.execution_efficiency_pct = round(max(0.0, min(100.0, (trade.captured_return_pct * 0.55) + ((100.0 - trade.risk_utilization_pct) * 0.45))), 4)
        if trade.exit_reason in ("tp", "take_profit") and trade.captured_return_pct >= 85.0:
            trade.exit_quality = "Excellent Exit"
        elif trade.captured_return_pct >= 70.0 and realized_return_pct >= 0.0:
            trade.exit_quality = "Good Exit"
        elif trade.captured_return_pct >= 45.0 and realized_return_pct >= 0.0:
            trade.exit_quality = "Acceptable Exit"
        elif trade.exit_reason in ("micro_time_exit", "time_exit") and realized_return_pct > 0.0 and trade.captured_return_pct < 45.0:
            trade.exit_quality = "Premature Exit"
        elif trade.exit_reason in ("micro_time_exit", "time_exit") and realized_return_pct >= 0.0:
            trade.exit_quality = "Late Exit"
        elif trade.exit_reason in ("sl", "stop_loss") and trade.mfe_pct > trade.mae_pct and trade.mfe_pct > 0.25:
            trade.exit_quality = "Reversal Exit"
        elif trade.exit_reason in ("structure_exit", "structure"):
            trade.exit_quality = "Structure Exit"
        elif trade.exit_reason in ("volatility_exit", "volatility"):
            trade.exit_quality = "Volatility Exit"
        elif trade.exit_reason in ("liquidity_exit", "liquidity"):
            trade.exit_quality = "Liquidity Exit"
        elif trade.mfe_pct > 0.5 and realized_return_pct < trade.mfe_pct * 0.35:
            trade.exit_quality = "Missed Expansion"
        else:
            trade.exit_quality = "Unknown"

    def _build_return_intelligence(self) -> dict[str, Any]:
        return build_return_intelligence(
            self._closed_trades,
            status=self._status,
            feature_version=FEATURE_VERSION,
            model_version=str(getattr(self.learner, "model_version", 0)),
            report_interval=self._return_report_interval,
        )

    def _publish_return_research(self, return_intelligence: dict[str, Any]) -> None:
        closed_count = len(self._closed_trades)
        if closed_count < self._return_report_interval:
            return
        if closed_count == self._last_return_report_closed_count:
            return
        if closed_count % self._return_report_interval != 0:
            return

        report = return_intelligence.get("research_report", {}) if isinstance(return_intelligence, dict) else {}
        proposals = report.get("zeus_research_proposals", []) if isinstance(report, dict) else []
        summary = return_intelligence.get("summary", {}) if isinstance(return_intelligence, dict) else {}
        title = f"Hermes Return Intelligence Research Report ({closed_count} closed trades)"
        summary_text = (
            f"Average return {summary.get('average_return_pct', 'n/a')}%, "
            f"capture ratio {summary.get('average_captured_return_pct', 'n/a')}%, "
            f"risk utilization {summary.get('average_risk_utilization_pct', 'n/a')}%"
        )
        try:
            self._research_repository.add_simple(
                artifact_type="return_intelligence_research",
                source_system="hermes",
                title=title,
                summary=summary_text,
                payload={
                    "return_intelligence": return_intelligence,
                    "research_report": report,
                    "zeus_research_proposals": proposals,
                },
            )
        except Exception:
            pass
        self._last_return_report_closed_count = closed_count

    def _predict_signal(self, result: PrometheusResult, record: SetupRecord) -> HermesSignal:
        ml_pred: MLPrediction = self.learner.predict(record)
        c = result.confluence
        base_direction = str(c.direction if c else "sideways").lower()
        direction = "flat"
        if base_direction == "bullish":
            direction = "long"
        elif base_direction == "bearish":
            direction = "short"

        choch_bias = str(self._last_feature_meta.get("choch_bias", "unknown"))
        choch_conflict = (choch_bias == "bullish" and direction == "short") or (choch_bias == "bearish" and direction == "long")
        has_smc_support = bool(
            self._last_feature_meta.get("sweep_present")
            or self._last_feature_meta.get("fresh_ob_count", 0) > 0
            or self._last_feature_meta.get("fresh_fvg_count", 0) > 0
        )
        if choch_conflict and not has_smc_support:
            direction = "flat"

        atr = float(result.ms.current_atr if result.ms else 0.0)
        expected_distance_pts = max(30.0, atr * 100.0 * (0.8 + ml_pred.win_probability)) if atr > 0 else 50.0
        confidence = min(0.99, max(0.05, 0.55 * ml_pred.win_probability + 0.45 * ((c.total if c else 0.0) / 100.0)))
        reasons = []
        if self._last_feature_meta.get("sweep_present"):
            reasons.append("Recent liquidity sweep detected")
        if self._last_feature_meta.get("fresh_ob_count", 0) > 0:
            reasons.append("Fresh order block available")
        if self._last_feature_meta.get("fresh_fvg_count", 0) > 0:
            reasons.append("Unfilled fair value gap present")
        if self._last_feature_meta.get("choch_count", 0) > 0:
            reasons.append(f"CHOCH bias: {choch_bias}")
        if choch_conflict and has_smc_support:
            reasons.append(f"CHOCH conflict tolerated by fresh SMC support: {choch_bias}")
        if not reasons:
            reasons.append("Signal lacks Hermes LTF confirmation")

        sig = HermesSignal(
            signal_id=record.setup_id,
            timestamp=record.timestamp,
            asset=self.asset,
            timeframe=self.timeframe,
            direction=direction,
            expected_distance_pts=float(round(expected_distance_pts, 2)),
            confidence=float(round(confidence, 4)),
            ml_probability=float(round(ml_pred.win_probability, 4)),
            choch_bias=choch_bias,
            liquidity_sweep=bool(self._last_feature_meta.get("sweep_present")),
            fresh_ob_count=int(self._last_feature_meta.get("fresh_ob_count", 0)),
            fresh_fvg_count=int(self._last_feature_meta.get("fresh_fvg_count", 0)),
            reasons=reasons,
        )
        if sig.direction == "flat":
            sig.skipped = True
            sig.skip_reason = "CHOCH/SMC alignment insufficient"
        elif sig.confidence < 0.60 or sig.ml_probability < 0.58:
            sig.skipped = True
            sig.skip_reason = "Hermes confidence below LTF threshold"
        return sig

    def _open_paper_trade(self, sig: HermesSignal, price: float, result: PrometheusResult) -> HermesTrade:
        direction_mult = 1.0 if sig.direction == "long" else -1.0
        dist_price = sig.expected_distance_pts / 100.0
        sl = price - direction_mult * max(dist_price * 0.45, 0.25)
        tp = price + direction_mult * max(dist_price * 0.90, 0.50)
        trade = HermesTrade(
            trade_id=str(uuid.uuid4())[:12],
            signal_id=sig.signal_id,
            direction=sig.direction,
            entry_price=float(price),
            lots=float(self.fixed_lot),
            sl_price=float(sl),
            tp_price=float(tp),
            opened_at=datetime.now(timezone.utc).isoformat(),
            predicted_distance_pts=sig.expected_distance_pts,
            session=self._current_session_label(),
            regime=self._regime_label(result),
            trend_state=self._trend_label(result),
            volatility_state=self._volatility_label(result, self._last_df),
            pattern_name=str(self._last_feature_meta.get("pattern_name", "unknown") or "unknown"),
            pattern_family=str(self._last_feature_meta.get("pattern_family", "Unknown") or "Unknown"),
            pattern_cluster=str(self._last_feature_meta.get("pattern_cluster", "Unknown") or "Unknown"),
            model_version=str(getattr(self.learner, "model_version", 0)),
            feature_version=FEATURE_VERSION,
            signal_confidence=float(sig.confidence),
        )
        self._open_trades.append(trade)
        return trade

    def _manage_paper_trades(self, df) -> None:
        if not self._open_trades or df is None or len(df) == 0:
            return
        last_close = float(df["close"].iloc[-1])
        for trade in list(self._open_trades):
            self._refresh_trade_excursions(trade, df)
            is_long = trade.direction == "long"
            pnl_pts = (last_close - trade.entry_price) * 100.0 if is_long else (trade.entry_price - last_close) * 100.0
            # micromanage to secure positive closes once trade reaches >35% of expected move
            if pnl_pts >= trade.predicted_distance_pts * 0.35:
                lock_price = trade.entry_price + (0.08 if is_long else -0.08)
                if is_long:
                    trade.sl_price = max(trade.sl_price, lock_price)
                else:
                    trade.sl_price = min(trade.sl_price, lock_price)
            hit_tp = last_close >= trade.tp_price if is_long else last_close <= trade.tp_price
            hit_sl = last_close <= trade.sl_price if is_long else last_close >= trade.sl_price
            time_exit = len(df) >= 6 and pnl_pts > 0 and pnl_pts < trade.predicted_distance_pts * 0.25
            if hit_tp or hit_sl or time_exit:
                trade.exit_price = last_close
                trade.realized_distance_pts = float(round(abs(last_close - trade.entry_price) * 100.0, 2))
                trade.pnl = float(round((pnl_pts * trade.lots), 2))
                trade.status = "won" if trade.pnl > 0 else "lost"
                trade.exit_reason = "tp" if hit_tp else "sl" if hit_sl else "micro_time_exit"
                self._refresh_trade_excursions(trade, df)
                self._finalize_return_fields(trade)
                self._open_trades.remove(trade)
                self._closed_trades.append(trade)
                self.learner.update_outcome(trade.signal_id, 1 if trade.pnl > 0 else 0, rr=(trade.realized_distance_pts / max(1.0, trade.predicted_distance_pts)), exit_price=last_close)
                self._emit_lineage_event(EventType.TRADE_CLOSED, {"trade_id": trade.trade_id, "status": trade.status, "pnl": trade.pnl})
                self._emit_lineage_event(EventType.PATTERN_LEARNED, {"signal_id": trade.signal_id})
                self._emit_lineage_event(EventType.ML_UPDATED, {"model_version": str(getattr(self.learner, "model_version", 0))})

    def _return_pct(self, trade: HermesTrade) -> float:
        """Signed percent return for a closed trade, direction-aware."""
        if trade.entry_price <= 0 or trade.exit_price is None:
            return 0.0
        raw = ((float(trade.exit_price) - float(trade.entry_price)) / float(trade.entry_price)) * 100.0
        return float(raw if trade.direction == "long" else -raw)

    def _build_ml_learning_summary(self, feature_importance: dict[str, Any]) -> dict[str, Any]:
        """Human-readable snapshot of what Hermes ML is currently learning."""
        records_total = len(self.learner.records)
        labeled = sum(1 for r in self.learner.records if r.outcome is not None)
        unlabeled = max(0, records_total - labeled)
        min_samples = int(getattr(self.learner, "min_samples", 50) or 50)
        model_version = int(getattr(self.learner, "model_version", 0) or 0)

        if labeled < min_samples:
            next_training_in = min_samples - labeled
        else:
            next_training_in = (20 - (labeled % 20)) % 20

        fi_items = sorted(feature_importance.items(), key=lambda kv: float(kv[1]), reverse=True)
        top_features = [
            {"feature": str(name), "importance": float(round(float(val), 4))}
            for name, val in fi_items[:5]
        ]

        pattern_stats = [s for s in self.learner.get_all_stats() if (s.wins + s.losses) > 0]
        pattern_rows = [
            {
                "pattern": s.pattern,
                "samples": int(s.wins + s.losses),
                "wins": int(s.wins),
                "losses": int(s.losses),
                "win_rate_pct": float(round((s.wins / max(1, s.wins + s.losses)) * 100.0, 2)),
            }
            for s in pattern_stats
        ]
        strong_patterns = [r for r in pattern_rows if r["samples"] >= 5]
        best_patterns = sorted(strong_patterns, key=lambda r: (r["win_rate_pct"], r["samples"]), reverse=True)[:3]
        weak_patterns = sorted(strong_patterns, key=lambda r: (r["win_rate_pct"], -r["samples"]))[:3]

        return {
            "model_stage": "trained" if model_version > 0 else "statistical_fallback",
            "model_version": model_version,
            "records_total": records_total,
            "records_labeled": labeled,
            "records_unlabeled": unlabeled,
            "min_samples_for_training": min_samples,
            "training_progress_pct": float(round(min(100.0, (labeled / max(1, min_samples)) * 100.0), 2)),
            "samples_to_next_training_gate": int(next_training_in),
            "top_features": top_features,
            "best_patterns": best_patterns,
            "weak_patterns": weak_patterns,
        }

    def _write_status(self) -> None:
        wins = sum(1 for t in self._closed_trades if t.status == "won")
        losses = sum(1 for t in self._closed_trades if t.status == "lost")
        total_closed = wins + losses
        return_pcts = [self._return_pct(t) for t in self._closed_trades]
        win_return_pct = sum(r for r in return_pcts if r > 0)
        loss_return_pct = sum(r for r in return_pcts if r < 0)
        total_return_pct = win_return_pct + loss_return_pct
        fi = self.learner.feature_importance() or {}
        self._status["last_poll"] = datetime.now(timezone.utc).isoformat()
        self._status["open_trades"] = [asdict(t) for t in self._open_trades][-20:]
        self._status["closed_trades"] = [asdict(t) for t in self._closed_trades][-50:]
        self._status["last_signal"] = asdict(self._last_signal) if self._last_signal else None
        self._status["ml"] = {
            "records": len(self.learner.records),
            "feature_importance": fi,
            "model_version": getattr(self.learner, "model_version", 0),
            "feature_version": FEATURE_VERSION,
            "strategy_version": HERMES_STRATEGY_VERSION,
            "source_system": HERMES_SOURCE_SYSTEM,
            "dataset_generation": HERMES_DATASET_GENERATION,
            "pattern_stats": [asdict(s) for s in self.learner.get_all_stats()[:20]],
            "learning_summary": self._build_ml_learning_summary(fi),
        }

        try:
            lrn_stats = build_hermes_analytics(_ROOT)
            self._status["learning_intelligence"] = lrn_stats.get("metrics", {})
            self._status["pattern_intelligence"] = lrn_stats.get("pattern_intelligence", {})
            self._status["cluster_intelligence"] = lrn_stats.get("cluster_intelligence", [])
            self._status["confidence_intelligence"] = lrn_stats.get("confidence_intelligence", {})
            self._status["directional_intelligence"] = lrn_stats.get("directional_intelligence", {})
            self._status["duration_intelligence"] = lrn_stats.get("duration_intelligence", {})
            self._status["execution_intelligence"] = lrn_stats.get("execution_intelligence", {})
            self._status["knowledge_quality_controls"] = lrn_stats.get("knowledge_quality_controls", {})
            self._status["metric_knowledge_confidence"] = lrn_stats.get("metric_knowledge_confidence", {})
            self._status["pattern_genome"] = lrn_stats.get("pattern_genome", [])
            self._status["academy"] = lrn_stats.get("academy", {})
            self._status["edge_stability"] = lrn_stats.get("edge_stability", {})
            self._status["performance_diagnostics"] = lrn_stats.get("performance_diagnostics", {})
            self._status["performance_intelligence"] = lrn_stats.get("performance_intelligence", {})
            self._status["adaptive_execution_intelligence"] = lrn_stats.get("adaptive_execution_intelligence", {})
            self._status["expectancy_intelligence"] = lrn_stats.get("expectancy_intelligence", {})
            self._status["adaptive_roadmap"] = lrn_stats.get("adaptive_roadmap", {})
            self._status["research_engine"] = lrn_stats.get("research_engine", {})
            self._status["evolution_roadmap"] = lrn_stats.get("evolution_roadmap", {})
            self._status["validation_gate"] = lrn_stats.get("validation_gate", {})
            self._status["academy_certification_gate"] = lrn_stats.get("academy_certification_gate", {})
            self._status["phase_completion_report"] = lrn_stats.get("phase_completion_report", {})
            self._status["learning_timeline"] = lrn_stats.get("timeline", {})
            self._status["analytics_readiness_matrix"] = lrn_stats.get("readiness_matrix", [])
            self._status["analytics_audit"] = lrn_stats.get("audit", {})
            self._status["analytics_statuses"] = lrn_stats.get("statuses", {})
            self._status["return_intelligence"] = lrn_stats.get("return_intelligence", {})
            self._status["return_research_report"] = lrn_stats.get("return_research_report", {})
            self._status["zeus_research_proposals"] = lrn_stats.get("zeus_research_proposals", [])
            self._status["pattern_context_intelligence"] = lrn_stats.get("pattern_context_intelligence", {})
            self._status["pattern_context_research_library"] = lrn_stats.get("pattern_context_research_library", [])
            self._status["shared_knowledge_contracts"] = lrn_stats.get("shared_knowledge_contracts", [])
            self._status["olympus_intelligence_auditor"] = lrn_stats.get("olympus_intelligence_auditor", {})
            self._status["olympus_observability"] = lrn_stats.get("olympus_observability", {})
            self._status["olympus_audit_report"] = lrn_stats.get("olympus_audit_report", {})
        except Exception as exc:
            self._status["analytics_error"] = str(exc)

        return_intelligence = self._build_return_intelligence()
        self._status["return_intelligence"] = return_intelligence
        self._status["return_research_report"] = return_intelligence.get("research_report", {})
        self._status["zeus_research_proposals"] = return_intelligence.get("zeus_research_proposals", [])
        self._status["edge_stability"] = return_intelligence.get("edge_stability", self._status.get("edge_stability", {}))
        self._status["academy_return_intelligence"] = return_intelligence.get("academy_subject", {})
        self._publish_return_research(return_intelligence)

        if self._enable_trade_lifecycle_intelligence:
            try:
                tli = build_trade_lifecycle_intelligence(
                    _ROOT,
                    status=self._status,
                    open_trades=[asdict(t) for t in self._open_trades],
                    closed_trades=[asdict(t) for t in self._closed_trades],
                )
                tli_artifacts = write_trade_lifecycle_intelligence_artifacts(_ROOT, tli)
                self._status["trade_lifecycle_intelligence"] = tli
                self._status["trade_lifecycle_artifacts"] = tli_artifacts
                self._status["trade_duration_intelligence"] = tli.get("modules", {}).get("trade_duration_intelligence", {})
                self._status["trade_state_machine"] = tli.get("modules", {}).get("trade_state_machine", {})
                self._status["trade_management_intelligence"] = tli.get("modules", {}).get("trade_management_intelligence", {})
                self._status["trade_exit_intelligence"] = tli.get("modules", {}).get("exit_intelligence", {})
                self._status["trade_reward_capture_intelligence"] = tli.get("modules", {}).get("reward_capture_intelligence", {})
                self._status["trade_lifecycle_analytics"] = tli.get("modules", {}).get("trade_lifecycle_analytics", {})
            except Exception as exc:
                self._status["trade_lifecycle_error"] = str(exc)

        if self._enable_idip:
            try:
                idip = build_institutional_decision_intelligence_platform(
                    _ROOT,
                    status=self._status,
                    open_trades=[asdict(t) for t in self._open_trades],
                    closed_trades=[asdict(t) for t in self._closed_trades],
                    feature_flags={
                        "enable_institutional_learning_scientist": self._enable_institutional_learning_scientist,
                        "enable_capital_intelligence_engine": self._enable_capital_intelligence_engine,
                        "enable_knowledge_graph_engine": self._enable_knowledge_graph_engine,
                        "enable_decision_replay_counterfactual_intelligence": self._enable_decision_replay_counterfactual_intelligence,
                        "enable_meta_learning_engine": self._enable_meta_learning_engine,
                        "enable_autonomous_research_orchestrator": self._enable_autonomous_research_orchestrator,
                        "enable_research_prioritization_engine": self._enable_research_prioritization_engine,
                        "enable_knowledge_evolution_engine": self._enable_knowledge_evolution_engine,
                        "enable_explainability_engine": self._enable_explainability_engine,
                        "enable_knowledge_coverage_intelligence": self._enable_knowledge_coverage_intelligence,
                        "enable_institutional_research_director": self._enable_institutional_research_director,
                    },
                )
                idip_artifacts = write_idip_artifacts(_ROOT, idip)
                update_idip_status(self._status, idip=idip, idip_artifacts=idip_artifacts)
            except Exception as exc:
                self._status["idip_error"] = str(exc)

        if self._enable_institutional_dataset_architecture:
            try:
                ida = build_institutional_dataset_architecture(
                    _ROOT,
                    status=self._status,
                    open_trades=[asdict(t) for t in self._open_trades],
                    closed_trades=[asdict(t) for t in self._closed_trades],
                    feature_flags={
                        "enable_institutional_dataset_architecture": self._enable_institutional_dataset_architecture,
                        "enable_trade_lifecycle_intelligence": self._enable_trade_lifecycle_intelligence,
                        "enable_idip": self._enable_idip,
                    },
                )
                ida_artifacts = write_institutional_dataset_architecture_artifacts(_ROOT, ida)
                self._status["institutional_dataset_architecture"] = ida
                self._status["institutional_dataset_artifacts"] = ida_artifacts
                self._status["dataset_quality_score"] = ida.get("dataset_quality", {})
                self._status["zeus_validation_boards"] = ida.get("zeus_validation_boards", {})
                self._status["institutional_knowledge_base"] = ida.get("institutional_knowledge_base", {})
            except Exception as exc:
                self._status["institutional_dataset_error"] = str(exc)

        self._status["stats"] = {
            "signals_seen": self._status["stats"].get("signals_seen", 0),
            "signals_entered": self._status["stats"].get("signals_entered", 0),
            "signals_skipped": self._status["stats"].get("signals_skipped", 0),
            "wins": wins,
            "losses": losses,
            "win_rate": round((wins / total_closed) * 100.0, 2) if total_closed else 0.0,
            "return_pct_total": float(round(total_return_pct, 4)),
            "return_pct_wins": float(round(win_return_pct, 4)),
            "return_pct_losses": float(round(loss_return_pct, 4)),
        }
        try:
            HERMES_STATUS_FILE.write_text(json.dumps(_json_safe(self._status), indent=2), encoding="utf-8")
            self._emit_lineage_event(EventType.DASHBOARD_UPDATED, {"status_file": str(HERMES_STATUS_FILE)})
        except OSError as exc:
            # Keep the trading/learning loop alive even if status persistence temporarily fails.
            logger.error("Hermes status persistence failed: %s", exc)

    def _bootstrap_with_simulation(self, df) -> None:
        if self._bootstrap_done or not self.simulate_bootstrap or pd is None or df is None or len(df) < 200:
            return
        # Lightweight historical bootstrap: create labeled setups from recent slices.
        for i in range(80, min(len(df) - 12, 220), 4):
            window = df.iloc[: i + 1]
            future = df.iloc[i + 1 : i + 13]
            try:
                result = self.engine.analyze_data(window, asset=self.asset, timeframe=self.timeframe, tf_data={self.timeframe.lower(): window}, render_chart=False, save_to_db=False)
            except Exception:
                continue
            if not result or not result.confluence:
                continue
            rec = self._build_setup_record(result)
            direction = str(result.confluence.direction or "sideways").lower()
            if direction not in ("bullish", "bearish"):
                continue
            entry = float(result.current_price or window["close"].iloc[-1])
            future_high = float(future["high"].max())
            future_low = float(future["low"].min())
            move_up = (future_high - entry) * 100.0
            move_down = (entry - future_low) * 100.0
            expected = max(move_up, move_down)
            won = int((direction == "bullish" and move_up > move_down and move_up >= 20.0) or (direction == "bearish" and move_down > move_up and move_down >= 20.0))
            rec.outcome = won
            rec.rr_achieved = expected / 20.0 if expected > 0 else 0.0
            self.learner.add_setup(rec)
            self._emit_lineage_event(EventType.SIMULATION_CREATED, {"setup_id": rec.setup_id, "outcome": won})
        try:
            self.learner.train()
            self._emit_lineage_event(EventType.ML_UPDATED, {"train_source": "bootstrap"})
        except Exception as exc:
            logger.debug("Hermes bootstrap training skipped: %s", exc)
        self._bootstrap_done = True

    def poll_once(self) -> dict[str, Any]:
        self._last_df = self._fetch_candles(self.candles)
        df = self._last_df
        if df is None or len(df) < 80:
            self._status["error"] = "No MT5 candles available for Hermes"
            self._write_status()
            return self._status

        self._bootstrap_with_simulation(df)
        self._manage_paper_trades(df)

        tf_data = {self.timeframe.lower(): df}
        result = self.engine.analyze_data(df, asset=self.asset, timeframe=self.timeframe, tf_data=tf_data, render_chart=False, save_to_db=False)
        if not result or not result.confluence:
            self._status["error"] = "Prometheus analysis unavailable"
            self._write_status()
            return self._status

        record = self._build_setup_record(result)
        signal = self._predict_signal(result, record)
        self._last_signal = signal
        self._append_pattern_snapshot(result, signal)
        self._emit_lineage_event(EventType.PREDICTION_CREATED, {"signal_id": signal.signal_id, "direction": signal.direction, "confidence": signal.confidence})
        self._status["stats"]["signals_seen"] += 1
        self.learner.add_setup(record)
        self._emit_lineage_event(EventType.SIGNAL_GENERATED, {"signal_id": signal.signal_id, "skipped": signal.skipped})

        if signal.skipped:
            self._status["stats"]["signals_skipped"] += 1
            self._status.setdefault("skipped_signals", []).append(asdict(signal))
            self._status["skipped_signals"] = self._status["skipped_signals"][-80:]
        else:
            entry = float(result.current_price or df["close"].iloc[-1])
            trade = self._open_paper_trade(signal, entry, result)
            self._emit_lineage_event(EventType.TRADE_ENTERED, {"trade_id": trade.trade_id, "signal_id": signal.signal_id})
            self._status["stats"]["signals_entered"] += 1
            self._status.setdefault("signals", []).append(asdict(signal))
            self._status["signals"] = self._status["signals"][-80:]

        self._emit_lineage_event(EventType.TRADE_MANAGED, {"open_trades": len(self._open_trades)})

        self._write_status()
        return self._status

    def run(self) -> None:
        logger.info("Hermes started | %s %s | fixed lot %.2f", self.asset, self.timeframe, self.fixed_lot)
        # Clear any stale stop flag left from a previous session
        if HERMES_CONTROL_FILE.exists():
            try:
                HERMES_CONTROL_FILE.write_text(json.dumps({"stop": False}), encoding="utf-8")
            except Exception:
                pass
        while True:
            try:
                if HERMES_CONTROL_FILE.exists():
                    ctrl = json.loads(HERMES_CONTROL_FILE.read_text(encoding="utf-8"))
                    if ctrl.get("stop"):
                        logger.info("Hermes stop requested via control file")
                        break
                self.poll_once()
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                logger.exception("Hermes poll error: %s", exc)
                self._status["error"] = str(exc)
                try:
                    self._write_status()
                except Exception as status_exc:
                    logger.error("Hermes status fallback write failed: %s", status_exc)
            time.sleep(self.poll_interval)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    HermesBot().run()


if __name__ == "__main__":
    main()
