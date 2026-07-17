# AGENTS.md

## Repository Overview

**Prometheus** is an institutional-grade AI market analysis system for financial trading. It provides multi-timeframe technical analysis, machine-learning-based confluence scoring, Smart Money Concepts (SMC) detection, backtesting, a REST API, and a Streamlit dashboard.

All application source code lives under the `Prometheus/` subdirectory.

---

## Directory Structure

```
Prometheus/
├── main.py                  # CLI entry point (analyze / serve / ui / backtest / demo / image)
├── prometheus_core.py       # Core Prometheus orchestrator class
├── config.py                # Centralised configuration (PrometheusConfig singleton: CONFIG)
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container image (python:3.11-slim)
├── docker-compose.yml       # API (port 8000) + UI (port 8501) services
├── .env.example             # Environment variable template → copy to .env
│
├── engines/                 # Pure-analysis engines (no side effects)
│   ├── market_structure.py  # Swing-high/low, BOS, CHoCH detection
│   ├── support_resistance.py
│   ├── candlestick_engine.py
│   ├── chart_patterns.py
│   ├── fibonacci_engine.py
│   ├── liquidity_smc.py     # Smart Money Concepts / FVG / order blocks
│   ├── multi_timeframe.py
│   ├── vwap_engine.py
│   └── amd_engine.py        # Accumulation / Manipulation / Distribution
│
├── olympus/                 # Institutional intelligence layer
│   ├── contracts.py         # Shared data contracts / Pydantic models
│   ├── versions.py
│   └── core/                # ~40 specialist engines
│       ├── institutional_decision_intelligence_platform.py  # IDIP orchestrator
│       ├── trade_lifecycle_intelligence.py
│       ├── zeus_validation_operations.py
│       ├── prometheus_evolution_intelligence.py
│       └── ...              # see olympus/core/ for the full list
│
├── hera/                    # Service infrastructure
│   ├── api_gateway.py
│   ├── lifecycle.py
│   ├── scheduler.py
│   └── ...
│
├── live_bot/                # Live trading via MetaTrader 5
│   ├── hermes.py            # Primary live-trading bot
│   ├── trader.py
│   ├── regime_classifier.py
│   └── watchdog.py
│
├── api/                     # FastAPI REST server
│   └── server.py
│
├── ui/                      # Streamlit dashboard
│   └── dashboard.py
│
├── backtesting/             # Walk-forward backtester
│   └── backtester.py
│
├── ml/                      # Machine-learning pattern learner
│   └── pattern_learner.py
│
├── data/                    # Sample / synthetic data generators
├── models/                  # Persisted ML model artefacts (gitignored)
├── storage/                 # SQLite DB + IDIP JSONL artefacts (gitignored)
├── outputs/                 # Analysis reports (gitignored)
└── tests/                   # pytest test suite
```

---

## Environment Setup

```bash
cd Prometheus

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Copy the environment template
cp .env.example .env
# Edit .env as needed (database URL, log level, ports, etc.)
```

---

## Running the Application

All commands are run from the `Prometheus/` directory.

```bash
# Start the FastAPI REST API (http://localhost:8000)
python main.py serve

# Launch the Streamlit dashboard (http://localhost:8501)
python main.py ui

# Analyse a CSV file of OHLCV data
python main.py analyze --csv data/xauusd.csv --asset XAUUSD --tf 4H

# Run a walk-forward backtest
python main.py backtest --csv data/xauusd.csv --asset XAUUSD --tf 4H

# Quick demo on synthetic data
python main.py demo

# Analyse a chart screenshot (no CSV required)
python main.py image path/to/chart.png --asset XAUUSD --tf 4H
```

### Docker

```bash
# Build and start both services (API + UI)
docker-compose up --build

# API:  http://localhost:8000
# UI:   http://localhost:8501
```

---

## Running Tests

Tests use **pytest** and live in `Prometheus/tests/`.

```bash
cd Prometheus
pytest tests/ -v
```

Run a single test file:

```bash
pytest tests/test_market_structure.py -v
```

---

## Key Architecture Conventions

### Configuration
- All tunable parameters live in `config.py` via the `PrometheusConfig` dataclass.
- Import the global singleton: `from config import CONFIG`.
- Secrets and runtime overrides belong in `.env` (never committed); see `.env.example`.

### Engines
- Engines in `engines/` are **stateless** analysis functions. They take a pandas DataFrame of OHLCV data and return typed result objects.
- All engines accept the global config or relevant sub-config objects at construction time.

### Olympus / IDIP Layer
- The Olympus layer (`olympus/`) is **additive only** — it must not mutate historical data or automatically modify execution behaviour.
- All IDIP outputs (`storage/olympus/idip_*.json[l]`) are advisory. Adoption requires Zeus validation and explicit operator approval.
- Artefact files are append-only JSONL. Never truncate or overwrite existing records.
- All public recommendation payloads must include `governance_flags`, an evidence score, and a `no_auto_adopt` marker.

### Pydantic Models & Contracts
- Shared data contracts are defined in `olympus/contracts.py`.
- All inter-module data exchange should use these typed contracts rather than raw dicts.

### Live Bot
- The live bot (`live_bot/hermes.py`) connects to MetaTrader 5. It must never be started in test or CI environments.
- Bot control flags are stored in `live_bot/bot_control.json` and `live_bot/hermes_control.json`.

### Storage
- Default database: SQLite at `storage/prometheus.db` (configurable via `PROMETHEUS_DATABASE_URL`).
- Do not commit the `storage/`, `models/`, `outputs/`, or `charts/` directories.

---

## Code Style Guidelines

- Python 3.11+; use `from __future__ import annotations` at the top of every module.
- Type-annotate all function signatures.
- Use dataclasses or Pydantic `BaseModel` for structured data; avoid plain `dict` for public interfaces.
- Log via the standard `logging` module using a per-module logger: `logger = logging.getLogger(__name__)`.
- Do not `print()` in library code; use logging. CLI commands in `main.py` may `print()` for user-facing output.
- Keep engines in `engines/` free of I/O (no file reads, no DB writes). Side effects belong in the orchestrator (`prometheus_core.py`) or the Olympus layer.

---

## Governance & Safety Rules

1. **No auto-adopt**: AI-generated recommendations are always advisory. No code path may automatically apply a recommendation to live execution without explicit operator approval and Zeus validation.
2. **Immutable history**: Historical trade data and JSONL artefacts must never be overwritten or deleted programmatically.
3. **Backwards compatibility**: All changes to the Olympus layer must be additive and versioned. Breaking schema changes are not permitted.
4. **No execution mutation**: Engines and analytics modules must not modify open orders, positions, or account state.
