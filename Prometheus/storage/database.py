"""
Database Storage Layer
=======================
SQLAlchemy ORM models for persisting all Prometheus data.

Tables:
  - analyses       : full analysis results per asset/timeframe
  - chart_images   : uploaded chart screenshots metadata
  - setups         : ML training records
  - trades         : backtest and live trade records
  - sr_zones       : detected S/R zones (for cross-session reuse)
  - pattern_cache  : cached pattern results

Supports SQLite (default) and PostgreSQL (set DATABASE_URL env var).

SQLite durability hardening
----------------------------
- WAL (Write-Ahead Logging) journal mode is enabled at every connection to
  prevent corruption on crash and to allow concurrent readers.
- ``PRAGMA synchronous = NORMAL`` balances durability against throughput.
- ``PRAGMA foreign_keys = ON`` enforces referential integrity.
- ``PRAGMA integrity_check`` is run at startup; any failure raises so operators
  cannot silently consume a corrupted database.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from sqlalchemy import (
        JSON, Boolean, Column, DateTime, Float, Integer, String, Text,
        create_engine, event, text,
    )
    from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logger.warning("sqlalchemy not installed — database persistence disabled")

# ── Engine setup ──────────────────────────────────────────────────────────────

DATABASE_URL = os.getenv("PROMETHEUS_DATABASE_URL", None)
if DATABASE_URL is None:
    _db_dir = Path(__file__).parent
    DATABASE_URL = f"sqlite:///{_db_dir}/prometheus.db"

if SQLALCHEMY_AVAILABLE:
    _is_sqlite = "sqlite" in DATABASE_URL

    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False} if _is_sqlite else {},
        echo=False,
    )

    if _is_sqlite:
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, _connection_record):
            """Apply durability PRAGMAs on every new SQLite connection.

            WAL mode survives application crashes without requiring an explicit
            checkpoint; NORMAL synchronous is safe for append-only workloads.
            """
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # ── ORM Base ──────────────────────────────────────────────────────────────
    class Base(DeclarativeBase):
        pass

    # ── Models ────────────────────────────────────────────────────────────────

    class AnalysisRecord(Base):
        __tablename__ = "analyses"
        id               = Column(Integer, primary_key=True, autoincrement=True)
        created_at       = Column(DateTime, default=datetime.utcnow, index=True)
        asset            = Column(String(16), index=True, nullable=False)
        timeframe        = Column(String(8),  index=True, nullable=False)
        current_price    = Column(Float)
        trend_bias       = Column(String(32))
        structure_type   = Column(String(32))
        confluence_score = Column(Float)
        confidence_grade = Column(String(4))
        primary_direction = Column(String(16))
        report_json      = Column(Text)
        nearest_support  = Column(Float)
        nearest_resistance = Column(Float)
        key_levels_json  = Column(Text)
        chart_image_id   = Column(Integer, nullable=True)
        annotated_chart_path = Column(String(512), nullable=True)

    class ChartImageRecord(Base):
        __tablename__ = "chart_images"
        id               = Column(Integer, primary_key=True, autoincrement=True)
        uploaded_at      = Column(DateTime, default=datetime.utcnow)
        filename         = Column(String(256))
        file_path        = Column(String(512))
        asset            = Column(String(16))
        timeframe        = Column(String(8))
        theme            = Column(String(8))
        candles_detected = Column(Integer)
        direction_hint   = Column(String(16))
        vision_json      = Column(Text)

    class SetupRecord(Base):
        __tablename__ = "setups"
        id                 = Column(Integer, primary_key=True, autoincrement=True)
        setup_id           = Column(String(64), unique=True, index=True)
        created_at         = Column(DateTime, default=datetime.utcnow)
        asset              = Column(String(16), index=True)
        timeframe          = Column(String(8))
        structure_type     = Column(Integer)
        trend_strength     = Column(Float)
        mtf_score          = Column(Float)
        sr_confidence      = Column(Float)
        candlestick_score  = Column(Float)
        pattern_confidence = Column(Float)
        fib_proximity      = Column(Integer)
        ob_present         = Column(Integer)
        stop_hunt          = Column(Integer)
        confluence_score   = Column(Float)
        outcome            = Column(Integer, nullable=True)
        rr_achieved        = Column(Float, nullable=True)
        entry_price        = Column(Float, nullable=True)
        sl_price           = Column(Float, nullable=True)
        tp_price           = Column(Float, nullable=True)
        exit_price         = Column(Float, nullable=True)

    class TradeRecord(Base):
        __tablename__ = "trades"
        id           = Column(Integer, primary_key=True, autoincrement=True)
        trade_id     = Column(String(64), unique=True, index=True)
        created_at   = Column(DateTime, default=datetime.utcnow)
        source       = Column(String(16), default="backtest")
        asset        = Column(String(16), index=True)
        timeframe    = Column(String(8))
        direction    = Column(String(8))
        entry_price  = Column(Float)
        sl_price     = Column(Float)
        tp_price     = Column(Float)
        exit_price   = Column(Float, nullable=True)
        size         = Column(Float)
        pnl          = Column(Float, nullable=True)
        rr           = Column(Float, nullable=True)
        status       = Column(String(16), default="open")
        entry_bar    = Column(Integer)
        exit_bar     = Column(Integer, nullable=True)
        # ── Analytics extension (added v2) ─────────────────────────────────
        session         = Column(String(32), nullable=True)   # trading session at entry
        regime          = Column(String(32), nullable=True)    # market regime at entry
        spread_at_entry = Column(Float, nullable=True)         # spread in price units
        score_at_entry  = Column(Float, nullable=True)         # confluence score
        exit_reason     = Column(String(64), nullable=True)    # "tp1", "sl", "trail", "5m_exit" …
        mae             = Column(Float, nullable=True)         # max adverse excursion ($)
        mfe             = Column(Float, nullable=True)         # max favourable excursion ($)
        hold_seconds    = Column(Integer, nullable=True)       # seconds position was held

    class SRZoneRecord(Base):
        __tablename__ = "sr_zones"
        id          = Column(Integer, primary_key=True, autoincrement=True)
        detected_at = Column(DateTime, default=datetime.utcnow)
        asset       = Column(String(16), index=True)
        timeframe   = Column(String(8))
        zone_type   = Column(String(16))
        level       = Column(Float)
        upper       = Column(Float)
        lower       = Column(Float)
        touches     = Column(Integer)
        confidence  = Column(Float)
        is_fresh    = Column(Boolean)

# ── Database access helpers ───────────────────────────────────────────────────

def init_db() -> None:
    if not SQLALCHEMY_AVAILABLE:
        logger.warning("SQLAlchemy unavailable — skipping DB init")
        return
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialised at %s", DATABASE_URL)
    # ── SQLite durability: verify integrity at startup ─────────────────────────
    if "sqlite" in DATABASE_URL:
        _verify_sqlite_integrity()
    # ── Migrate existing SQLite DBs: add any missing analytics columns ────────
    _migrate_trades_table()


def _verify_sqlite_integrity() -> None:
    """Run SQLite integrity_check at startup.

    A non-OK result indicates on-disk corruption and raises RuntimeError so
    the platform refuses to start rather than silently operating on bad data.
    Corruption at startup always has higher severity than a delayed failure.
    """
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("PRAGMA integrity_check(10)")).fetchall()
        messages = [r[0] for r in rows]
        if messages == ["ok"]:
            logger.info("SQLite integrity check: OK")
            return
        # Any non-ok message means the file is corrupt.
        logger.critical(
            "SQLite integrity_check FAILED — database may be corrupt: %s",
            messages,
        )
        raise RuntimeError(
            f"SQLite integrity check failed: {messages}. "
            "Restore from backup before starting Prometheus."
        )
    except RuntimeError:
        raise
    except Exception as exc:
        # Unexpected error running the check itself — warn but do not block start.
        logger.warning("SQLite integrity check could not run: %s", exc)


def _migrate_trades_table() -> None:
    """Add analytics columns to existing trades table if they don't exist yet.

    SQLite doesn't support IF NOT EXISTS in ALTER TABLE, so we probe each
    column individually and silently skip those that already exist.
    """
    if "sqlite" not in DATABASE_URL:
        return  # PostgreSQL handles schema via create_all
    new_cols = [
        ("session",         "VARCHAR(32)"),
        ("regime",          "VARCHAR(32)"),
        ("spread_at_entry", "REAL"),
        ("score_at_entry",  "REAL"),
        ("exit_reason",     "VARCHAR(64)"),
        ("mae",             "REAL"),
        ("mfe",             "REAL"),
        ("hold_seconds",    "INTEGER"),
    ]
    try:
        with engine.connect() as conn:
            for col_name, col_type in new_cols:
                try:
                    conn.execute(
                        __import__("sqlalchemy").text(
                            f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}"
                        )
                    )
                    conn.commit()
                    logger.debug("DB migration: added trades.%s", col_name)
                except Exception:
                    pass  # column already exists
    except Exception as exc:
        logger.debug("DB migration skipped: %s", exc)


def get_session():
    if not SQLALCHEMY_AVAILABLE:
        raise RuntimeError("SQLAlchemy not installed")
    return SessionLocal()


def save_analysis(asset: str, timeframe: str, report_dict: Dict[str, Any]) -> Optional[int]:
    if not SQLALCHEMY_AVAILABLE:
        return None
    with get_session() as db:
        rec = AnalysisRecord(
            asset              = asset,
            timeframe          = timeframe,
            current_price      = report_dict.get("current_price"),
            trend_bias         = report_dict.get("trend_bias", "")[:32],
            structure_type     = report_dict.get("structure_type", "")[:32],
            confluence_score   = report_dict.get("confluence_score"),
            confidence_grade   = report_dict.get("confidence_grade", "")[:4],
            primary_direction  = report_dict.get("primary_direction", "")[:16],
            nearest_support    = report_dict.get("nearest_support"),
            nearest_resistance = report_dict.get("nearest_resistance"),
            report_json        = json.dumps(report_dict),
            key_levels_json    = json.dumps(report_dict.get("key_levels", [])),
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)
        logger.debug("Analysis saved: id=%d %s %s", rec.id, asset, timeframe)
        return rec.id


def save_chart_image(
    filename: str, file_path: str, asset: str,
    timeframe: str, vision_dict: Dict[str, Any],
) -> Optional[int]:
    if not SQLALCHEMY_AVAILABLE:
        return None
    with get_session() as db:
        rec = ChartImageRecord(
            filename=filename, file_path=file_path, asset=asset,
            timeframe=timeframe,
            theme=vision_dict.get("theme", "")[:8],
            candles_detected=vision_dict.get("candles_detected", 0),
            direction_hint=vision_dict.get("dominant_direction", "")[:16],
            vision_json=json.dumps(vision_dict),
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)
        return rec.id


def save_setup(setup_dict: Dict[str, Any]) -> None:
    if not SQLALCHEMY_AVAILABLE:
        return
    with get_session() as db:
        rec = SetupRecord(**{
            k: v for k, v in setup_dict.items()
            if hasattr(SetupRecord, k) and k != "id"
        })
        db.merge(rec)
        db.commit()


def list_analyses(
    asset: Optional[str] = None,
    timeframe: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    if not SQLALCHEMY_AVAILABLE:
        return []
    with get_session() as db:
        q = db.query(AnalysisRecord).order_by(AnalysisRecord.created_at.desc())
        if asset:
            q = q.filter(AnalysisRecord.asset == asset.upper())
        if timeframe:
            q = q.filter(AnalysisRecord.timeframe == timeframe.upper())
        rows = q.limit(limit).all()
        return [
            {
                "id":                r.id,
                "created_at":        str(r.created_at),
                "asset":             r.asset,
                "timeframe":         r.timeframe,
                "current_price":     r.current_price,
                "direction":         r.primary_direction or "",
                "structure":         r.structure_type or "",
                "confluence_score":  r.confluence_score,
                "grade":             r.confidence_grade or "",
                "nearest_support":   r.nearest_support,
                "nearest_resistance":r.nearest_resistance,
            }
            for r in rows
        ]


def get_analysis(record_id: int) -> Optional[Dict[str, Any]]:
    """Return the full record dict (including report_json) for a given ID."""
    if not SQLALCHEMY_AVAILABLE:
        return None
    with get_session() as db:
        r = db.query(AnalysisRecord).filter(AnalysisRecord.id == record_id).first()
        if r is None:
            return None
        return {
            "id":                r.id,
            "created_at":        str(r.created_at),
            "asset":             r.asset,
            "timeframe":         r.timeframe,
            "current_price":     r.current_price,
            "direction":         r.primary_direction or "",
            "structure":         r.structure_type or "",
            "trend_bias":        r.trend_bias or "",
            "confluence_score":  r.confluence_score,
            "grade":             r.confidence_grade or "",
            "nearest_support":   r.nearest_support,
            "nearest_resistance":r.nearest_resistance,
            "report_json":       r.report_json,
            "key_levels_json":   r.key_levels_json,
        }


def delete_analysis(record_id: int) -> bool:
    """Delete an analysis record by ID. Returns True if deleted."""
    if not SQLALCHEMY_AVAILABLE:
        return False
    with get_session() as db:
        r = db.query(AnalysisRecord).filter(AnalysisRecord.id == record_id).first()
        if r is None:
            return False
        db.delete(r)
        db.commit()
        return True


def list_trades(
    asset: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Return recent trade records from the trades table."""
    if not SQLALCHEMY_AVAILABLE:
        return []
    with get_session() as db:
        q = db.query(TradeRecord).order_by(TradeRecord.created_at.desc())
        if asset:
            q = q.filter(TradeRecord.asset == asset.upper())
        if source:
            q = q.filter(TradeRecord.source == source)
        rows = q.limit(limit).all()
        return [
            {
                "id":             r.id,
                "trade_id":       r.trade_id,
                "created_at":     str(r.created_at),
                "source":         r.source,
                "asset":          r.asset,
                "timeframe":      r.timeframe,
                "direction":      r.direction,
                "entry_price":    r.entry_price,
                "sl_price":       r.sl_price,
                "tp_price":       r.tp_price,
                "exit_price":     r.exit_price,
                "size":           r.size,
                "pnl":            r.pnl,
                "rr":             r.rr,
                "status":         r.status,
                # analytics extension
                "session":        getattr(r, "session", None),
                "regime":         getattr(r, "regime", None),
                "spread_at_entry":getattr(r, "spread_at_entry", None),
                "score_at_entry": getattr(r, "score_at_entry", None),
                "exit_reason":    getattr(r, "exit_reason", None),
                "mae":            getattr(r, "mae", None),
                "mfe":            getattr(r, "mfe", None),
                "hold_seconds":   getattr(r, "hold_seconds", None),
            }
            for r in rows
        ]


def save_trade(trade_dict: Dict[str, Any]) -> Optional[int]:
    """Insert or update a live trade record. Uses trade_id as the upsert key.

    Call with status="open" when a position is opened, then call again with
    exit_price / pnl / status="win"|"loss" when the position closes.
    """
    if not SQLALCHEMY_AVAILABLE:
        return None
    try:
        allowed_cols = {c.key for c in TradeRecord.__table__.columns}
        with get_session() as db:
            existing = db.query(TradeRecord).filter(
                TradeRecord.trade_id == str(trade_dict.get("trade_id", ""))
            ).first()
            if existing:
                for k, v in trade_dict.items():
                    if k in allowed_cols and k not in ("id", "trade_id", "created_at"):
                        setattr(existing, k, v)
                db.commit()
                return existing.id
            else:
                rec = TradeRecord(**{k: v for k, v in trade_dict.items() if k in allowed_cols})
                db.add(rec)
                db.commit()
                db.refresh(rec)
                return rec.id
    except Exception as exc:
        logger.warning("save_trade error: %s", exc)
        return None

