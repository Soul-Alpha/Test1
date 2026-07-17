"""
Reconcile DB trades marked "open" against actual MT5 deal history.
Run once after a bot restart to backfill win/loss/pnl for already-closed positions.

Usage:
    & "c:\\Users\\Chaba\\Documents\\tradingBots\\.venv\\Scripts\\python.exe" fix_db_trades.py
"""
import sys, datetime
sys.path.insert(0, 'c:/Users/Chaba/Documents/tradingBots/Prometheus')

import MetaTrader5 as mt5
from storage.database import list_trades, save_trade

def main():
    if not mt5.initialize():
        print("MT5 init failed:", mt5.last_error()); return

    # Pull all historical deals going back 30 days
    now   = datetime.datetime.now()
    start = now - datetime.timedelta(days=30)
    all_deals = mt5.history_deals_get(start, now) or []

    # Index closing deals by position_id
    close_deals: dict[int, list] = {}
    for d in all_deals:
        if d.entry == mt5.DEAL_ENTRY_OUT:
            close_deals.setdefault(d.position_id, []).append(d)

    print(f"MT5 closing deals fetched: {len(close_deals)} positions")

    # Get all "open" DB trades
    open_trades = [t for t in list_trades(source='live', limit=500) if t.get('status') == 'open']
    print(f"DB trades with status='open': {len(open_trades)}")

    updated = skipped = 0
    for trade in open_trades:
        tid = str(trade.get('trade_id', ''))
        if not tid.startswith('live_'):
            skipped += 1; continue
        try:
            pos_id = int(tid.replace('live_', ''))
        except ValueError:
            skipped += 1; continue

        closes = close_deals.get(pos_id)
        if not closes:
            skipped += 1; continue   # still open or too old

        net_pnl = sum(d.profit for d in closes)
        exit_price = closes[-1].price  # last closing deal price
        status = 'win' if net_pnl > 0 else 'loss'

        save_trade({
            'trade_id':   tid,
            'status':     status,
            'pnl':        round(net_pnl, 2),
            'exit_price': exit_price,
            'exit_bar':   0,
        })
        print(f"  Updated {tid}: {status.upper()} ${net_pnl:+.2f}  exit={exit_price:.4f}")
        updated += 1

    print()
    print(f"Done — updated: {updated}  skipped/still-open: {skipped}")
    mt5.shutdown()

if __name__ == '__main__':
    main()
