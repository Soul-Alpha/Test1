"""
Prometheus FastAPI REST API
============================
Exposes all analysis capabilities over HTTP so the Streamlit UI
and any external clients can consume them without importing Python packages.

Run:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /analyze              – full pipeline on JSON OHLCV payload
    POST /upload-chart         – multipart image upload → full analysis
    POST /analyze-image        – image URL / base64 → vision analysis
    GET  /market-structure     – structure-only analysis
    GET  /patterns             – chart pattern detection
    POST /backtest             – walk-forward backtest
    POST /label-outcome        – record trade outcome for ML training
    GET  /ml/stats             – ML model performance stats
    GET  /history              – past analyses from DB
    GET  /health               – readiness check
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from fastapi import (
    BackgroundTasks, FastAPI, File, Form, HTTPException,
    Query, UploadFile, status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, validator

# --- Local imports -----------------------------------------------------------
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from config import CONFIG
from prometheus_core import Prometheus, PrometheusResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Prometheus Market Analysis API",
    description=(
        "Institutional-grade AI-powered market analysis for XAUUSD and all assets. "
        "Detects market structure, S/R zones, candlestick patterns, chart patterns, "
        "Fibonacci levels, liquidity, SMC concepts, and generates full AI narratives."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Singleton Prometheus instance (loaded once at startup)
# ---------------------------------------------------------------------------
_prometheus: Optional[Prometheus] = None


def get_engine() -> Prometheus:
    global _prometheus
    if _prometheus is None:
        _prometheus = Prometheus()
    return _prometheus


@app.on_event("startup")
async def _startup():
    get_engine()
    logger.info("Prometheus engine ready.")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class OHLCVRow(BaseModel):
    timestamp: Optional[str] = None
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    float = 0.0


class AnalyzeRequest(BaseModel):
    asset:       str             = Field("XAUUSD", description="Asset symbol")
    timeframe:   str             = Field("4H",     description="Primary timeframe")
    data:        List[OHLCVRow]  = Field(...,       description="OHLCV bars (oldest first)")
    tf_data:     Optional[Dict[str, List[OHLCVRow]]] = Field(
        None, description="Additional timeframes for MTF analysis"
    )
    render:      bool            = Field(True,  description="Generate chart files")
    save_to_db:  bool            = Field(True,  description="Save result to database")

    @validator("data")
    def _min_bars(cls, v):
        if len(v) < 20:
            raise ValueError("Provide at least 20 OHLCV bars.")
        return v


class LabelRequest(BaseModel):
    run_id:     str
    outcome:    int   = Field(..., ge=0, le=1, description="1=win, 0=loss")
    rr:         Optional[float] = Field(None, description="Risk-reward achieved")
    exit_price: Optional[float] = None


class BacktestRequest(BaseModel):
    asset:          str            = "XAUUSD"
    timeframe:      str            = "4H"
    data:           List[OHLCVRow] = Field(..., description="Historical OHLCV bars")
    initial_capital: float         = 10_000.0
    risk_per_trade:  float         = 0.01
    min_confluence:  float         = 55.0


class SwingData(BaseModel):
    index: int
    price: float
    swing_type: str     # "HIGH" | "LOW"
    strength: float


class SRZoneData(BaseModel):
    level:       float
    zone_type:   str    # "support" | "resistance"
    confidence:  float
    label:       str


class PatternData(BaseModel):
    pattern_type:  str
    direction:     str
    confidence:    float
    target_price:  Optional[float]
    invalidation:  Optional[float]
    description:   str


class SMCSummary(BaseModel):
    order_blocks:    int
    fvg_count:       int
    liquidity_pools: int
    stop_hunts:      int
    bias:            str


class ConfluenceData(BaseModel):
    total:         float
    grade:         str
    direction:     str
    reasons:       List[str]
    component_scores: Dict[str, float]
    entry_zone:    Optional[List[float]]
    invalidation_levels: List[float]


class ReportData(BaseModel):
    trend_bias:         str
    market_structure:   str
    candlestick_signal: str
    patterns_summary:   str
    fibonacci_summary:  str
    smc_summary:        str
    mtf_summary:        str
    bull_scenario:      str
    bear_scenario:      str
    invalidation:       str
    risk_profile:       str
    full_text:          str


class AnalyzeResponse(BaseModel):
    run_id:            str
    timestamp:         str
    asset:             str
    timeframe:         str
    current_price:     Optional[float]
    structure_type:    Optional[str]
    trend_strength:    Optional[float]
    sr_zones:          List[SRZoneData]
    patterns:          List[PatternData]
    fib_current_level: Optional[str]
    smc:               Optional[SMCSummary]
    mtf_bias:          Optional[str]
    confluence:        Optional[ConfluenceData]
    report:            Optional[ReportData]
    interactive_chart: Optional[str]
    static_chart:      Optional[str]
    ml_quality_score:  Optional[float]
    ml_win_probability: Optional[float]


# ---------------------------------------------------------------------------
# Helper: DataFrame builder
# ---------------------------------------------------------------------------

def _rows_to_df(rows: List[OHLCVRow]) -> pd.DataFrame:
    records = [r.dict() for r in rows]
    df = pd.DataFrame(records)
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.set_index("timestamp")
    else:
        df = df.drop(columns=["timestamp"], errors="ignore")
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df


def _result_to_response(r: PrometheusResult) -> AnalyzeResponse:
    """Serialize PrometheusResult to AnalyzeResponse Pydantic model."""

    sr_zones: List[SRZoneData] = []
    if r.sr:
        for z in (r.sr.support_zones + r.sr.resistance_zones)[:12]:
            sr_zones.append(SRZoneData(
                level=round(z.level, 4),
                zone_type=z.zone_type,
                confidence=round(z.confidence, 3),
                label=z.label,
            ))

    patterns: List[PatternData] = []
    if r.pat:
        for p in r.pat.patterns[:8]:
            patterns.append(PatternData(
                pattern_type=p.pattern_type,
                direction=p.direction,
                confidence=round(p.confidence, 3),
                target_price=round(p.target_price, 4) if p.target_price else None,
                invalidation=round(p.invalidation, 4) if p.invalidation else None,
                description=p.description,
            ))

    smc_data: Optional[SMCSummary] = None
    if r.smc:
        smc_data = SMCSummary(
            order_blocks=len(r.smc.order_blocks),
            fvg_count=len(r.smc.fair_value_gaps),
            liquidity_pools=len(r.smc.liquidity_pools),
            stop_hunts=len(r.smc.stop_hunts),
            bias=r.smc.bias,
        )

    confluence_data: Optional[ConfluenceData] = None
    if r.confluence:
        confluence_data = ConfluenceData(
            total=round(r.confluence.total, 1),
            grade=r.confluence.grade,
            direction=r.confluence.direction,
            reasons=r.confluence.reasons,
            component_scores={k: round(v, 1) for k, v in r.confluence.component_scores.items()},
            entry_zone=r.confluence.entry_zone,
            invalidation_levels=r.confluence.invalidation_levels,
        )

    report_data: Optional[ReportData] = None
    if r.report:
        rp = r.report
        report_data = ReportData(
            trend_bias=getattr(rp, "trend_bias", ""),
            market_structure=getattr(rp, "market_structure", ""),
            candlestick_signal=getattr(rp, "candlestick_signal", ""),
            patterns_summary=getattr(rp, "patterns_summary", ""),
            fibonacci_summary=getattr(rp, "fibonacci_summary", ""),
            smc_summary=getattr(rp, "smc_summary", ""),
            mtf_summary=getattr(rp, "mtf_summary", ""),
            bull_scenario=getattr(rp, "bull_scenario", ""),
            bear_scenario=getattr(rp, "bear_scenario", ""),
            invalidation=getattr(rp, "invalidation", ""),
            risk_profile=getattr(rp, "risk_profile", ""),
            full_text=getattr(rp, "full_text", ""),
        )

    return AnalyzeResponse(
        run_id=r.run_id,
        timestamp=r.timestamp,
        asset=r.asset,
        timeframe=r.timeframe,
        current_price=r.current_price,
        structure_type=(r.ms.structure_type.name if r.ms else None),
        trend_strength=round(r.ms.trend_strength, 3) if r.ms else None,
        sr_zones=sr_zones,
        patterns=patterns,
        fib_current_level=(
            f"{r.fib.current_level.label} ({r.fib.current_level.price:.4f})"
            if r.fib and r.fib.current_level else None
        ),
        smc=smc_data,
        mtf_bias=(r.mtf.primary_bias if r.mtf else None),
        confluence=confluence_data,
        report=report_data,
        interactive_chart=r.interactive_chart,
        static_chart=r.static_chart,
        ml_quality_score=(
            round(r.ml_prediction.quality_score, 3) if r.ml_prediction else None
        ),
        ml_win_probability=(
            round(r.ml_prediction.win_probability, 3) if r.ml_prediction else None
        ),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Meta"])
def health():
    """Readiness probe."""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
def analyze(payload: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Full Prometheus analysis pipeline on OHLCV data.

    Accepts a JSON list of OHLCV rows and returns the complete analysis
    including market structure, S/R zones, candlestick patterns, chart
    patterns, Fibonacci, SMC, MTF alignment, confluence score, and
    full AI-generated narrative.
    """
    engine = get_engine()

    df = _rows_to_df(payload.data)

    tf_dfs: Optional[Dict[str, pd.DataFrame]] = None
    if payload.tf_data:
        tf_dfs = {tf: _rows_to_df(rows) for tf, rows in payload.tf_data.items()}

    try:
        result = engine.analyze_data(
            df=df,
            asset=payload.asset,
            timeframe=payload.timeframe,
            tf_data=tf_dfs,
            render_chart=payload.render,
            save_to_db=payload.save_to_db,
        )
    except Exception as exc:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return _result_to_response(result)


@app.post("/upload-chart", response_model=AnalyzeResponse, tags=["Analysis"])
async def upload_chart(
    file:       UploadFile = File(..., description="Chart screenshot (PNG/JPG)"),
    asset:      str        = Form("XAUUSD"),
    timeframe:  str        = Form("4H"),
    ohlcv_json: Optional[str] = Form(None, description="JSON-encoded OHLCV list (optional)"),
):
    """
    Upload a chart screenshot for vision-based analysis.
    Optionally supply JSON OHLCV data to layer quantitative analysis on top.
    """
    if file.content_type not in ("image/png", "image/jpeg", "image/jpg", "image/webp"):
        raise HTTPException(status_code=400, detail="Only PNG/JPG images accepted.")

    engine = get_engine()
    raw = await file.read()

    # Save to temp file
    suffix = pathlib.Path(file.filename).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    df = None
    if ohlcv_json:
        try:
            rows_raw = json.loads(ohlcv_json)
            rows = [OHLCVRow(**r) for r in rows_raw]
            df = _rows_to_df(rows)
        except Exception as e:
            logger.warning("Could not parse ohlcv_json: %s", e)

    try:
        result = engine.analyze_image(tmp_path, asset=asset, timeframe=timeframe, df=df)
    except Exception as exc:
        logger.exception("Image analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return _result_to_response(result)


@app.get(
    "/interactive-chart/{filename}",
    response_class=HTMLResponse,
    tags=["Charts"],
)
def get_interactive_chart(filename: str):
    """Return an interactive Plotly HTML chart by filename."""
    from config import OUTPUTS_DIR
    path = OUTPUTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Chart not found.")
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/static-chart/{filename}", tags=["Charts"])
def get_static_chart(filename: str):
    """Return a static PNG chart by filename."""
    from config import OUTPUTS_DIR
    path = OUTPUTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Chart not found.")
    return FileResponse(str(path), media_type="image/png")


@app.post("/label-outcome", tags=["ML"])
def label_outcome(payload: LabelRequest):
    """
    Record the real outcome of a previous analysis to improve the ML model.
    Call this after a trade completes.
    """
    get_engine().label_outcome(
        run_id=payload.run_id,
        outcome=payload.outcome,
        rr=payload.rr,
        exit_price=payload.exit_price,
    )
    return {"status": "recorded", "run_id": payload.run_id}


@app.get("/ml/stats", tags=["ML"])
def ml_stats():
    """Return ML model statistics and per-pattern win rates."""
    learner = get_engine().learner
    try:
        stats = {
            "total_setups":  len(learner.records),
            "labeled":       sum(1 for r in learner.records if r.outcome is not None),
            "model_trained": learner.model is not None,
            "pattern_stats": {k: v.__dict__ for k, v in learner.pattern_stats.items()},
        }
        if learner.model is not None:
            stats["feature_importance"] = learner.feature_importance()
    except Exception as e:
        stats = {"error": str(e)}
    return stats


@app.post("/backtest", tags=["Backtesting"])
def backtest(payload: BacktestRequest):
    """
    Run a walk-forward backtest over provided historical data.

    The Prometheus analysis engine is used as the signal generator:
    each bar slice is analyzed and a signal is produced when
    confluence ≥ min_confluence.
    """
    from backtesting.backtester import Backtester, Signal

    df = _rows_to_df(payload.data)
    engine = get_engine()

    def _strategy(df_slice: pd.DataFrame) -> Optional[Signal]:
        if len(df_slice) < 30:
            return None
        try:
            r = engine.analyze_data(
                df_slice, asset=payload.asset, timeframe=payload.timeframe,
                render_chart=False, save_to_db=False,
            )
            if r.confluence and r.confluence.total >= payload.min_confluence:
                direction = r.confluence.direction
                price = float(df_slice["close"].iloc[-1])
                atr = float((df_slice["high"] - df_slice["low"]).rolling(14).mean().iloc[-1])
                return Signal(
                    direction=direction,
                    entry_price=price,
                    stop_loss=price - 1.5 * atr if direction == "bullish" else price + 1.5 * atr,
                    take_profit=price + 3.0 * atr if direction == "bullish" else price - 3.0 * atr,
                    confidence=r.confluence.total,
                )
        except Exception:
            pass
        return None

    backtester = Backtester(
        initial_capital=payload.initial_capital,
        risk_per_trade=payload.risk_per_trade,
    )
    bt_result = backtester.run(df, _strategy, min_confidence=payload.min_confluence)

    return {
        "asset":           payload.asset,
        "timeframe":       payload.timeframe,
        "total_trades":    bt_result.total_trades,
        "win_rate":        round(bt_result.win_rate, 3),
        "profit_factor":   round(bt_result.profit_factor, 3),
        "expectancy":      round(bt_result.expectancy, 4),
        "max_drawdown_pct": round(bt_result.max_drawdown_pct, 3),
        "sharpe_ratio":    round(bt_result.sharpe_ratio, 3),
        "calmar_ratio":    round(bt_result.calmar_ratio, 3),
        "avg_rr":          round(bt_result.avg_rr, 3),
        "net_return_pct":  round(bt_result.net_return_pct, 3),
        "final_equity":    round(bt_result.final_equity, 2),
        "trade_log": [
            {
                "entry": round(t.entry_price, 4),
                "exit":  round(t.exit_price, 4) if t.exit_price else None,
                "pnl":   round(t.pnl, 4),
                "rr":    round(t.rr_achieved, 3),
                "direction": t.direction,
                "is_win": t.is_win,
            }
            for t in bt_result.trades[-50:]   # last 50 trades
        ],
    }


@app.get("/history", tags=["Storage"])
def history(
    asset:     Optional[str] = Query(None),
    timeframe: Optional[str] = Query(None),
    limit:     int           = Query(20, ge=1, le=200),
):
    """Return past analysis records from the database."""
    from storage.database import list_analyses
    try:
        rows = list_analyses(asset=asset, timeframe=timeframe, limit=limit)
        return {"count": len(rows), "records": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/market-structure", tags=["Analysis"])
def market_structure_only(
    data_json: str = Query(..., description="JSON array of OHLCV rows"),
):
    """Fast endpoint: market structure analysis only (no full pipeline)."""
    try:
        rows_raw = json.loads(data_json)
        rows = [OHLCVRow(**r) for r in rows_raw]
        df = _rows_to_df(rows)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid data: {e}")

    engine = get_engine()
    ms = engine.ms_engine.analyze(df)
    return {
        "structure_type": ms.structure_type.name,
        "trend_strength": round(ms.trend_strength, 3),
        "swing_highs": [{"index": s.index, "price": s.price, "strength": round(s.strength, 2)} for s in ms.swing_highs[-10:]],
        "swing_lows":  [{"index": s.index, "price": s.price, "strength": round(s.strength, 2)} for s in ms.swing_lows[-10:]],
        "bos_count":   len(ms.bos_events),
        "choch_count": len(ms.choch_events),
        "narrative":   ms.narrative,
    }


# ---------------------------------------------------------------------------
# Debug / admin routes (can be protected by API key middleware in production)
# ---------------------------------------------------------------------------

@app.post("/admin/retrain-ml", tags=["Admin"])
def retrain_ml():
    """Force-retrain the ML model on all labeled setups."""
    learner = get_engine().learner
    labeled = [r for r in learner.records if r.outcome is not None]
    if len(labeled) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 10 labeled setups (have {len(labeled)})."
        )
    try:
        learner.train()
        return {"status": "trained", "samples": len(labeled)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
