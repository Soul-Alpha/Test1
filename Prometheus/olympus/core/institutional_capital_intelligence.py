from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CAPITAL_INTELLIGENCE_VERSION = "capital-intel-v1.0"
STATUS_AWAITING = "Awaiting Historical Data"


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)

SUPPORTED_EVENT_TYPES = {
    "trading_profit": "Trading Profit",
    "trading_loss": "Trading Loss",
    "deposit": "Deposit",
    "withdrawal": "Withdrawal",
    "commission": "Commission",
    "swap": "Swap",
    "broker_adjustment": "Broker Adjustment",
    "internal_transfer": "Internal Transfer",
    "external_transfer": "External Transfer",
    "operational_adjustment": "Operational Adjustment",
    "unknown": "Unknown",
}


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return float(value)
    except Exception:
        return None


def _classify_event(raw_type: str, amount: float) -> str:
    t = (raw_type or "").strip().lower().replace(" ", "_")
    if t in SUPPORTED_EVENT_TYPES:
        return t
    if t in {"profit", "tp", "trade_profit"} or amount > 0 and "trade" in t:
        return "trading_profit"
    if t in {"loss", "sl", "trade_loss"} or amount < 0 and "trade" in t:
        return "trading_loss"
    if "deposit" in t:
        return "deposit"
    if "withdraw" in t:
        return "withdrawal"
    if "commission" in t:
        return "commission"
    if "swap" in t:
        return "swap"
    if "broker" in t and "adjust" in t:
        return "broker_adjustment"
    if "internal" in t and "transfer" in t:
        return "internal_transfer"
    if "external" in t and "transfer" in t:
        return "external_transfer"
    if "operational" in t or "ops" in t:
        return "operational_adjustment"
    return "unknown"


def _event_ledger(event_type: str) -> str:
    if event_type in {"trading_profit", "trading_loss", "commission", "swap"}:
        return "trading_ledger"
    if event_type in {"deposit", "withdrawal", "internal_transfer", "external_transfer", "broker_adjustment"}:
        return "capital_ledger"
    if event_type in {"operational_adjustment", "unknown"}:
        return "operational_ledger"
    return "learning_ledger"


def _is_trading_eligible(event_type: str) -> bool:
    return event_type in {"trading_profit", "trading_loss", "commission", "swap"}


def _sum_amount(rows: list[dict[str, Any]]) -> float:
    total = 0.0
    for row in rows:
        amt = _safe_float(row.get("amount"))
        if amt is not None:
            total += float(amt)
    return round(total, 4)


def build_capital_intelligence(
    *,
    status: dict[str, Any],
    closed_trades: list[dict[str, Any]],
    account_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    events: list[dict[str, Any]] = []

    # Trade-derived capital events are always trading-eligible.
    for trade in closed_trades:
        pnl = _safe_float(trade.get("pnl"))
        if pnl is None:
            ret = _safe_float(trade.get("realized_return_pct"))
            pnl = 0.0 if ret is None else float(ret)
        events.append(
            {
                "event_id": f"trade-{trade.get('trade_id', 'unknown')}",
                "timestamp": trade.get("closed_at") or generated_at,
                "event_type": "trading_profit" if pnl >= 0 else "trading_loss",
                "amount": round(float(pnl), 4),
                "source": "closed_trade",
            }
        )

        commission = _safe_float(trade.get("commission"))
        if commission is not None and abs(commission) > 0:
            events.append(
                {
                    "event_id": f"commission-{trade.get('trade_id', 'unknown')}",
                    "timestamp": trade.get("closed_at") or generated_at,
                    "event_type": "commission",
                    "amount": round(float(commission), 4),
                    "source": "closed_trade",
                }
            )

        swap = _safe_float(trade.get("swap"))
        if swap is not None and abs(swap) > 0:
            events.append(
                {
                    "event_id": f"swap-{trade.get('trade_id', 'unknown')}",
                    "timestamp": trade.get("closed_at") or generated_at,
                    "event_type": "swap",
                    "amount": round(float(swap), 4),
                    "source": "closed_trade",
                }
            )

    for evt in account_events or []:
        amt = _safe_float(evt.get("amount"))
        if amt is None:
            continue
        raw_t = str(evt.get("event_type") or evt.get("type") or "unknown")
        event_type = _classify_event(raw_t, float(amt))
        events.append(
            {
                "event_id": str(evt.get("event_id") or evt.get("id") or f"evt-{len(events)+1:06d}"),
                "timestamp": evt.get("timestamp") or generated_at,
                "event_type": event_type,
                "amount": round(float(amt), 4),
                "source": str(evt.get("source") or "account_event"),
            }
        )

    events.sort(key=lambda x: str(x.get("timestamp") or ""))

    trading_ledger: list[dict[str, Any]] = []
    capital_ledger: list[dict[str, Any]] = []
    operational_ledger: list[dict[str, Any]] = []
    learning_ledger: list[dict[str, Any]] = []

    for evt in events:
        event_type = _classify_event(str(evt.get("event_type") or "unknown"), _safe_float(evt.get("amount")) or 0.0)
        ledger_name = _event_ledger(event_type)
        row = {
            **evt,
            "event_type": event_type,
            "event_label": SUPPORTED_EVENT_TYPES.get(event_type, "Unknown"),
            "ledger": ledger_name,
            "ml_eligible": _is_trading_eligible(event_type),
            "strategy_stat_eligible": _is_trading_eligible(event_type),
            "drawdown_eligible": _is_trading_eligible(event_type),
            "recovery_eligible": _is_trading_eligible(event_type),
            "circuit_breaker_eligible": _is_trading_eligible(event_type),
        }
        if ledger_name == "trading_ledger":
            trading_ledger.append(row)
        elif ledger_name == "capital_ledger":
            capital_ledger.append(row)
        elif ledger_name == "operational_ledger":
            operational_ledger.append(row)
        else:
            learning_ledger.append(row)

    start_balance = _safe_float(status.get("start_balance"))
    if start_balance is None:
        start_balance = _safe_float(status.get("balance"))
    if start_balance is None:
        start_balance = 0.0

    raw_curve: list[dict[str, Any]] = []
    strategy_curve: list[dict[str, Any]] = []
    raw_equity = float(start_balance)
    strategy_equity = float(start_balance)

    for evt in events:
        amount = float(_safe_float(evt.get("amount")) or 0.0)
        event_type = str(evt.get("event_type") or "unknown")
        raw_equity += amount
        if _is_trading_eligible(event_type):
            strategy_equity += amount

        raw_curve.append(
            {
                "timestamp": evt.get("timestamp"),
                "event_id": evt.get("event_id"),
                "event_type": event_type,
                "equity": round(raw_equity, 4),
            }
        )
        strategy_curve.append(
            {
                "timestamp": evt.get("timestamp"),
                "event_id": evt.get("event_id"),
                "event_type": event_type,
                "equity": round(strategy_equity, 4),
            }
        )

    trading_total = _sum_amount(trading_ledger)
    deposits = _sum_amount([r for r in capital_ledger if r.get("event_type") == "deposit"])
    withdrawals = _sum_amount([r for r in capital_ledger if r.get("event_type") == "withdrawal"])

    payload = {
        "version": CAPITAL_INTELLIGENCE_VERSION,
        "generated_at": generated_at,
        "additive_only": True,
        "execution_modification_allowed": False,
        "supported_event_types": list(SUPPORTED_EVENT_TYPES.values()),
        "ledgers": {
            "trading_ledger": trading_ledger,
            "capital_ledger": capital_ledger,
            "operational_ledger": operational_ledger,
            "learning_ledger": learning_ledger,
        },
        "eligibility_policy": {
            "trading_only_for_ml_labels": True,
            "trading_only_for_expectancy": True,
            "trading_only_for_win_rate": True,
            "trading_only_for_drawdown": True,
            "trading_only_for_position_sizing": True,
            "deposits_withdrawals_excluded_from_strategy_stats": True,
        },
        "equity_curves": {
            "raw_equity_curve": raw_curve,
            "strategy_equity_curve": strategy_curve,
        },
        "summary": {
            "events_total": len(events),
            "trading_events": len(trading_ledger),
            "capital_events": len(capital_ledger),
            "operational_events": len(operational_ledger),
            "learning_events": len(learning_ledger),
            "trading_growth": trading_total,
            "capital_injections": deposits,
            "capital_withdrawals": withdrawals,
            "organic_growth": trading_total,
            "raw_equity": round(raw_equity, 4),
            "strategy_equity": round(strategy_equity, 4),
            "capital_efficiency": round((strategy_equity / raw_equity), 4) if abs(raw_equity) > 1e-9 else STATUS_AWAITING,
            "drawdown_attribution": {
                "trading_only": round(_sum_amount([r for r in trading_ledger if float(r.get("amount", 0.0)) < 0]), 4),
                "non_trading_excluded": True,
            },
            "recovery_attribution": {
                "trading_only": round(_sum_amount([r for r in trading_ledger if float(r.get("amount", 0.0)) > 0]), 4),
                "non_trading_excluded": True,
            },
        },
    }
    return payload


def write_capital_intelligence_artifacts(root_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    storage = root_dir / "storage" / "olympus"
    storage.mkdir(parents=True, exist_ok=True)

    runtime_path = storage / "capital_intelligence_runtime.json"
    _write_json_atomic(runtime_path, payload)

    return {"capital_intelligence": str(runtime_path)}
