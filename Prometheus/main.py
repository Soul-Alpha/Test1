"""
Prometheus - Institutional AI Market Analysis System
=====================================================
CLI entry point.

Commands:
    python main.py analyze  --asset XAUUSD --tf 4h --csv data/xauusd.csv
    python main.py serve    [--host 0.0.0.0] [--port 8000]
    python main.py ui       [--port 8501]
    python main.py backtest --asset XAUUSD --tf 4h --csv data/xauusd.csv
    python main.py demo                          # runs demo on synthetic data

Usage examples::

    # Start the REST API
    python main.py serve

    # Open the Streamlit dashboard
    python main.py ui

    # Run analysis on a CSV file and print the report
    python main.py analyze --csv path/to/ohlcv.csv --asset XAUUSD --tf 4H

    # Run a backtest
    python main.py backtest --csv path/to/ohlcv.csv --min-confluence 55
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Add project root
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import CONFIG, setup_logging

setup_logging()
logger = logging.getLogger("prometheus")


# ---------------------------------------------------------------------------
# Sub-command implementations
# ---------------------------------------------------------------------------

def cmd_analyze(args: argparse.Namespace) -> None:
    """Run full analysis on CSV OHLCV data (and optionally a chart image)."""
    import pandas as pd
    from prometheus_core import Prometheus

    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]

    bot    = Prometheus()
    result = bot.analyze_data(df, asset=args.asset.upper(), timeframe=args.tf.upper())

    # Optional image analysis
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            logger.error("Image not found: %s", img_path)
        else:
            vision = bot.analyze_image(
                str(img_path),
                asset=args.asset.upper(),
                timeframe=args.tf.upper(),
            )
            print("\n[Chart Vision Analysis]")
            if vision:
                print(f"  Theme:            {vision.theme}")
                print(f"  Candles detected: {vision.candles_detected}")
                print(f"  Dominant trend:   {vision.dominant_direction}")
                if vision.detected_patterns:
                    print(f"  Patterns:         {', '.join(vision.detected_patterns)}")
                if vision.price_levels:
                    levels_str = ", ".join(f"{p:.2f}" for p in vision.price_levels[:6])
                    print(f"  Price levels:     {levels_str}")
            else:
                print("  (opencv not installed — vision analysis skipped)")

    print("\n" + "=" * 70)
    print(f"  Prometheus Analysis Report — {result.asset} {result.timeframe}")
    print("=" * 70)

    if result.ms:
        print(f"\n[Market Structure]\n  {result.ms.narrative}")

    if result.confluence:
        print(
            f"\n[Confluence]\n"
            f"  Score: {result.confluence.total:.1f}/100  Grade: {result.confluence.grade}\n"
            f"  Bias:  {result.confluence.direction.upper()}\n"
            f"  Reasons:\n" +
            "\n".join(f"    • {r}" for r in result.confluence.reasons)
        )

    if result.report and result.report.full_text:
        print(f"\n[AI Report]\n{result.report.full_text}")

    if result.interactive_chart:
        print(f"\nInteractive chart: {result.interactive_chart}")
    if result.static_chart:
        print(f"Static chart:      {result.static_chart}")
    print("\n" + "=" * 70)


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the FastAPI server with uvicorn."""
    host = args.host or CONFIG.api.host
    port = args.port or CONFIG.api.port

    logger.info("Starting API server at http://%s:%s", host, port)
    try:
        import uvicorn
        uvicorn.run("api.server:app", host=host, port=port, reload=args.reload)
    except ImportError:
        logger.error("uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)


def cmd_ui(args: argparse.Namespace) -> None:
    """Launch the consolidated Prometheus Trading Command Center."""
    port = args.port or CONFIG.ui.port
    dashboard = ROOT / "ui" / "prometheus_command_center.py"

    logger.info("Launching Streamlit UI at http://localhost:%s", port)
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard),
         "--server.port", str(port),
         "--server.headless", "false"],
        check=False,
    )


def cmd_backtest(args: argparse.Namespace) -> None:
    """Run a walk-forward backtest on a CSV file."""
    import pandas as pd
    from backtesting.backtester import Backtester, Signal
    from prometheus_core import Prometheus

    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]

    bot = Prometheus()
    asset     = args.asset.upper()
    timeframe = args.tf.upper()
    min_conf  = args.min_confluence

    def strategy(df_slice: pd.DataFrame):
        if len(df_slice) < 30:
            return None
        try:
            r = bot.analyze_data(df_slice, asset=asset, timeframe=timeframe,
                                 render_chart=False, save_to_db=False)
            if r.confluence and r.confluence.total >= min_conf:
                direction = r.confluence.direction
                price = float(df_slice["close"].iloc[-1])
                atr   = float((df_slice["high"] - df_slice["low"]).rolling(14).mean().iloc[-1])
                return Signal(
                    direction=direction, entry_price=price,
                    stop_loss=price - 1.5 * atr if direction == "bullish" else price + 1.5 * atr,
                    take_profit=price + 3.0 * atr if direction == "bullish" else price - 3.0 * atr,
                    confidence=r.confluence.total,
                )
        except Exception:
            pass
        return None

    backtester = Backtester(
        initial_capital=args.capital,
        risk_per_trade=args.risk / 100,
    )

    logger.info("Running backtest on %d bars…", len(df))
    bt = backtester.run(df, strategy, min_confidence=min_conf)

    print("\n" + "=" * 60)
    print(f"  Backtest Results — {asset} {timeframe}")
    print("=" * 60)
    print(f"  Total Trades   : {bt.total_trades}")
    print(f"  Win Rate       : {bt.win_rate:.1%}")
    print(f"  Profit Factor  : {bt.profit_factor:.2f}")
    print(f"  Net Return     : {bt.net_return_pct:.1%}")
    print(f"  Max Drawdown   : {bt.max_drawdown_pct:.1%}")
    print(f"  Sharpe Ratio   : {bt.sharpe_ratio:.2f}")
    print(f"  Avg R:R        : {bt.avg_rr:.2f}")
    print(f"  Final Equity   : ${bt.final_equity:,.2f}")
    print("=" * 60)

    # Save equity curve chart
    if bt.equity_curve:
        try:
            renderer = __import__("visualization.chart_renderer", fromlist=["ChartRenderer"]).ChartRenderer()
            chart_path = renderer.render_equity_curve(bt.equity_curve, f"{asset}_{timeframe}_backtest")
            print(f"\nEquity curve: {chart_path}")
        except Exception as e:
            logger.warning("Could not render equity curve: %s", e)


def cmd_demo(args: argparse.Namespace) -> None:
    """Run a complete demo on synthetic XAUUSD data."""
    from data.sample_data import generate_xauusd_ohlcv
    from prometheus_core import Prometheus

    logger.info("Generating synthetic XAUUSD data…")
    df = generate_xauusd_ohlcv(n_bars=500, timeframe="4H")

    bot    = Prometheus()
    result = bot.analyze_data(df, asset="XAUUSD", timeframe="4H")

    print("\n" + "=" * 70)
    print("  DEMO — Prometheus Synthetic XAUUSD 4H Analysis")
    print("=" * 70)
    if result.confluence:
        print(f"\n  Confluence: {result.confluence.total:.1f}/100  Grade: {result.confluence.grade}")
        print(f"  Bias:       {result.confluence.direction.upper()}")
    if result.report:
        print(f"\n{result.report.full_text[:1200]}…")
    if result.interactive_chart:
        print(f"\n  Chart saved to: {result.interactive_chart}")
    print("\n  Demo complete. Launch 'python main.py ui' for the full dashboard.\n")


def cmd_image(args: argparse.Namespace) -> None:
    """Analyze a chart screenshot image — no CSV required."""
    from prometheus_core import Prometheus

    img_path = Path(args.image)
    if not img_path.exists():
        logger.error("Image not found: %s", img_path)
        sys.exit(1)

    bot    = Prometheus()
    result = bot.analyze_image(
        str(img_path),
        asset=args.asset.upper(),
        timeframe=args.tf.upper(),
    )

    print("\n" + "=" * 70)
    print(f"  Chart Vision Analysis — {result.asset} {result.timeframe}")
    print("=" * 70)

    if result.report and result.report.vision_summary:
        print(f"\n[Vision Summary]\n{result.report.vision_summary}")
    elif result.report and result.report.full_text:
        print(f"\n[Analysis]\n{result.report.full_text}")
    else:
        print("\n  opencv not installed — install it with:")
        print("  pip install opencv-python-headless")

    print("\n" + "=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="prometheus",
        description="Prometheus Institutional AI Market Analysis System",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # analyze
    pa = sub.add_parser("analyze", help="Analyze a CSV OHLCV file")
    pa.add_argument("--csv",    required=True, help="Path to OHLCV CSV file")
    pa.add_argument("--asset",  default="XAUUSD")
    pa.add_argument("--tf",     default="4H",  help="Timeframe (e.g. 4H, 1D)")
    pa.add_argument("--image",  default=None,  help="Optional path to a chart screenshot (PNG/JPG) for vision analysis")

    # serve
    ps = sub.add_parser("serve", help="Start FastAPI REST API")
    ps.add_argument("--host",   default=None)
    ps.add_argument("--port",   default=None, type=int)
    ps.add_argument("--reload", action="store_true", help="Enable hot-reload")

    # ui
    pu = sub.add_parser("ui", help="Launch Streamlit dashboard")
    pu.add_argument("--port", default=None, type=int)

    # backtest
    pb = sub.add_parser("backtest", help="Run a walk-forward backtest")
    pb.add_argument("--csv",            required=True)
    pb.add_argument("--asset",          default="XAUUSD")
    pb.add_argument("--tf",             default="4H")
    pb.add_argument("--capital",        default=10_000.0, type=float)
    pb.add_argument("--risk",           default=1.0, type=float, help="Risk per trade in percent")
    pb.add_argument("--min-confluence", dest="min_confluence", default=55.0, type=float)

    # image
    pi = sub.add_parser("image", help="Analyze a chart screenshot (no CSV needed)")
    pi.add_argument("image", help="Path to PNG/JPG chart screenshot")
    pi.add_argument("--asset", default="XAUUSD")
    pi.add_argument("--tf",    default="4H", help="Timeframe (e.g. 4H, 1D)")

    # demo
    sub.add_parser("demo", help="Run demo on synthetic XAUUSD data")

    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser  = _build_parser()
    args    = parser.parse_args()
    cmd_map = {
        "analyze":  cmd_analyze,
        "image":    cmd_image,
        "serve":    cmd_serve,
        "ui":       cmd_ui,
        "backtest": cmd_backtest,
        "demo":     cmd_demo,
    }
    cmd_map[args.command](args)
