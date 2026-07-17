"""
Prometheus Scalp Backtester — CLI Runner
=========================================
Fetches historical data from MT5 and runs the LTF scalp backtest,
then prints a detailed report and optionally saves JSON.

Usage
-----
  cd C:\\Users\\Chaba\\Documents\\tradingBots\\Prometheus

  # Basic run (last 1000 bars on 30M, $120 account):
  python backtesting/run_scalp_backtest.py --asset XAUUSDm --tf 30m --balance 120

  # Full date-range run with ML training:
  python backtesting/run_scalp_backtest.py ^
    --asset XAUUSDm --tf 30m ^
    --from 2025-01-01 --to 2026-06-01 ^
    --balance 120 --risk 2.0 ^
    --min-grade B --min-score 65 ^
    --entry-mode zone_only ^
    --train-ml ^
    --report outputs/scalp_bt_report.json

  # 15M timeframe, larger account, no ML:
  python backtesting/run_scalp_backtest.py ^
    --asset XAUUSDm --tf 15m ^
    --balance 500 --risk 1.5 ^
    --no-train-ml

  # Load from CSV instead of MT5 (pass primary_df manually — see example below):
  python backtesting/run_scalp_backtest.py --csv my_data.csv --tf 30m --balance 120
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# ── Project root on path ───────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from backtesting.scalp_backtester import ScalpBacktester, ScalpBacktestConfig


def _parse_date(s: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(f"Cannot parse date '{s}'. Use YYYY-MM-DD.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prometheus LTF Scalp Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Market
    parser.add_argument("--asset",   default="XAUUSDm", help="MT5 symbol (default: XAUUSDm)")
    parser.add_argument("--tf",      default="30m",
                        help="Primary timeframe: 5m | 15m | 30m (default: 30m)")
    parser.add_argument("--from",    dest="date_from", type=_parse_date, default=None,
                        metavar="YYYY-MM-DD", help="Start date (default: last n_bars)")
    parser.add_argument("--to",      dest="date_to",   type=_parse_date, default=None,
                        metavar="YYYY-MM-DD", help="End date (default: now)")
    parser.add_argument("--n-bars",  type=int, default=2000,
                        help="Number of bars when date range not set (default: 2000)")

    # Account
    parser.add_argument("--balance", type=float, default=120.0,
                        help="Starting balance in USD (default: 120)")
    parser.add_argument("--risk",    type=float, default=2.0,
                        help="Risk %% per trade (default: 2.0)")

    # Signal quality
    parser.add_argument("--min-grade", default="B",
                        help="Minimum grade: A | B | C (default: B)")
    parser.add_argument("--min-score", type=float, default=65.0,
                        help="Minimum confluence score (default: 65)")
    parser.add_argument("--entry-mode", default="zone_only",
                        choices=["zone_only", "market_any"],
                        help="Entry mode (default: zone_only)")

    # Simulation
    parser.add_argument("--slippage", type=float, default=2.0,
                        help="Slippage in price points (default: 2.0)")
    parser.add_argument("--commission", type=float, default=0.0003,
                        help="Commission %% per side (default: 0.0003)")
    parser.add_argument("--stride",   type=int, default=3,
                        help="Signal evaluation every N bars (default: 3)")
    parser.add_argument("--warmup",   type=int, default=50,
                        help="Warmup bars before trading starts (default: 50)")

    # ML
    parser.add_argument("--train-ml",    dest="train_ml",    action="store_true",  default=True,
                        help="Train XGBoost after backtest (default: on)")
    parser.add_argument("--no-train-ml", dest="train_ml",    action="store_false",
                        help="Skip ML training")

    # CSV fallback
    parser.add_argument("--csv", default=None,
                        help="Path to CSV with OHLCV columns (skips MT5 fetch)")

    # Output
    parser.add_argument("--report", default=None,
                        help="Path to save JSON report (e.g. outputs/report.json)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print progress every 5%% of bars")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity (default: INFO)")

    args = parser.parse_args()

    # ── Logging ───────────────────────────────────────────────────────────
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy sub-module loggers unless DEBUG
    if args.log_level != "DEBUG":
        for noisy in ["matplotlib", "PIL", "urllib3", "engineio", "socketio"]:
            logging.getLogger(noisy).setLevel(logging.ERROR)

    # ── Build config ───────────────────────────────────────────────────────
    cfg = ScalpBacktestConfig(
        asset           = args.asset,
        primary_tf      = args.tf.lower(),
        date_from       = args.date_from,
        date_to         = args.date_to,
        n_bars          = args.n_bars,
        initial_balance = args.balance,
        risk_pct        = args.risk,
        min_grade       = args.min_grade.upper(),
        min_score       = args.min_score,
        entry_mode      = args.entry_mode,
        slippage_pts    = args.slippage,
        commission_pct  = args.commission,
        signal_stride   = args.stride,
        warmup_bars     = args.warmup,
        train_ml        = args.train_ml,
        report_path     = args.report,
        verbose         = args.verbose,
    )

    print(f"\nPrometheus Scalp Backtester")
    print(f"  Asset:      {cfg.asset}")
    print(f"  Timeframe:  {cfg.primary_tf.upper()}")
    if cfg.date_from:
        print(f"  Period:     {cfg.date_from.date()} → {(cfg.date_to or datetime.utcnow()).date()}")
    else:
        print(f"  Bars:       {cfg.n_bars}")
    print(f"  Balance:    ${cfg.initial_balance:.2f}  Risk: {cfg.risk_pct}%")
    print(f"  Min grade:  {cfg.min_grade}  Min score: {cfg.min_score}")
    print(f"  Entry mode: {cfg.entry_mode}")
    print(f"  Train ML:   {'yes' if cfg.train_ml else 'no'}")
    print()

    # ── Run ───────────────────────────────────────────────────────────────
    bt = ScalpBacktester(cfg)

    # CSV input fallback
    if args.csv:
        import pandas as pd
        try:
            primary_df = pd.read_csv(args.csv, index_col=0, parse_dates=True)
            primary_df.columns = [c.lower() for c in primary_df.columns]
            print(f"Loaded {len(primary_df)} bars from {args.csv}")
            result = bt.run(primary_df=primary_df)
        except Exception as exc:
            print(f"Error loading CSV: {exc}")
            sys.exit(1)
    else:
        result = bt.run()   # fetches from MT5

    # ── Print report ───────────────────────────────────────────────────────
    bt.print_report(result)

    # Exit code: 0 if profitable, 1 if not
    sys.exit(0 if result.total_return_pct >= 0 else 1)


if __name__ == "__main__":
    main()
