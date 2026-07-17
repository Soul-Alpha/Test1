from types import SimpleNamespace

from backtesting.scalp_backtester import ScalpBacktestConfig, ScalpBacktester, ScalpTrade


def test_profitable_sl_exit_is_counted_as_win_with_trail_stop_type():
    bt = ScalpBacktester(ScalpBacktestConfig(train_ml=False))
    bt._equity = 1000.0

    trade = ScalpTrade(
        trade_id="t1",
        direction="long",
        entry_type="market",
        entry_price=100.0,
        sl_price=95.0,
        tp1_price=110.0,
        tp2_price=120.0,
        size=1.0,
        entry_bar=0,
    )
    bt._open.append(trade)

    bt._close_position(trade, bar_i=1, exit_price=102.0, reason="sl")

    closed = bt._closed[-1]
    assert closed.status == "won"
    assert closed.exit_reason == "sl"
    assert closed.stop_type == "trail_sl"
    assert closed.pnl > 0


def test_invalid_short_geometry_is_rejected_for_market_entry():
    bt = ScalpBacktester(ScalpBacktestConfig(train_ml=False))
    bt._equity = 1000.0

    # Invalid short geometry: SL below entry and TP above entry.
    out = bt._open_position(
        bar_i=0,
        bar_ts=None,
        direction="short",
        price=100.0,
        sl=95.0,
        tp1=105.0,
        tp2=110.0,
        lot=0.01,
        entry_type="market",
        meta={},
        atr=1.0,
    )

    assert out is None
    assert bt._invalid_trade_setup_count == 1


def test_unknown_regime_uses_fallback_score_floor_instead_of_automatic_skip():
    cfg = ScalpBacktestConfig(train_ml=False, min_score=80.0, unknown_regime_score_premium=25.0)
    bt = ScalpBacktester(cfg)

    result = SimpleNamespace(
        confluence=SimpleNamespace(grade="A", total=90.0, direction="bullish"),
        mtf=None,
        pat=None,
        smc=None,
    )

    qualifies, meta = bt._qualifies(
        result=result,
        equity=1000.0,
        regime=None,
        session_name="london",
        current_ltf_biases=[],
    )

    assert qualifies is False
    assert meta["regime"] == "unknown"
    assert bt._regime_unavailable_count == 1
    assert bt._skipped_by_regime_count == 1
