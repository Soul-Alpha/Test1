from backtesting.scalp_backtester import ScalpBacktestConfig


def test_zeus_config_supports_session_and_strategy_controls():
    cfg = ScalpBacktestConfig(
        enabled_sessions=["asian", "london_open", "london_ny_overlap"],
        strategy_name="zeus_ltf",
    )

    assert cfg.enabled_sessions == [
        "asian",
        "london_open",
        "london_ny_overlap",
    ]
    assert cfg.strategy_name == "zeus_ltf"
