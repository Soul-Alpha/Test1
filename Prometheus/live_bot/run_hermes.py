"""CLI launcher for Hermes."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure Prometheus root is on sys.path regardless of how this script is invoked
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from live_bot.hermes import HermesBot


def main() -> None:
    ap = argparse.ArgumentParser(description="Hermes LTF SMC paper-trading bot")
    ap.add_argument("--asset", default="XAUUSDm")
    ap.add_argument("--tf", default="M5")
    ap.add_argument("--candles", type=int, default=500)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--lot", type=float, default=0.01)
    ap.add_argument("--no-bootstrap", action="store_true")
    args = ap.parse_args()

    bot = HermesBot(
        asset=args.asset,
        timeframe=args.tf,
        candles=args.candles,
        poll_interval=args.poll,
        fixed_lot=args.lot,
        simulate_bootstrap=not args.no_bootstrap,
    )
    bot.run()


if __name__ == "__main__":
    main()
