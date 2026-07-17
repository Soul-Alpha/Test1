"""Diagnose why specific DB orphans weren't reconciled."""
import sys, os, sqlite3
sys.path.insert(0, ".")
os.environ["PYTHONIOENCODING"] = "utf-8"
from pathlib import Path
from datetime import datetime, timedelta
import MetaTrader5 as mt5

DB_PATH = Path("storage/prometheus.db")

# 1. Get all DB open records
conn = sqlite3.connect(str(DB_PATH))
conn.row_factory = sqlite3.Row
cur = conn.cursor()
cur.execute("""
    SELECT trade_id, created_at, direction, entry_price, size, status
    FROM trades WHERE status='open'
    ORDER BY rowid DESC
""")
db_open = [dict(r) for r in cur.fetchall()]
conn.close()

print(f"DB 'open' records: {len(db_open)}")

# 2. Get MT5 history
mt5.initialize()

# Check current open positions
curr_positions = mt5.positions_get()
curr_tickets = {p.ticket for p in curr_positions} if curr_positions else set()
print(f"MT5 open positions: {len(curr_tickets)} — {sorted(curr_tickets)}")

# Get all close deals from last 60 days
now   = datetime.now()
start = now - timedelta(days=60)
all_deals = mt5.history_deals_get(start, now) or []
print(f"MT5 history deals (60d): {len(all_deals)}")

# Build close_deals dict (position_id → list of close deals)
close_deals = {}
for d in all_deals:
    if d.entry == mt5.DEAL_ENTRY_OUT:
        close_deals.setdefault(d.position_id, []).append(d)
print(f"Unique positions with close deals: {len(close_deals)}")

# 3. Walk through DB 'open' records and diagnose each
mt5_open_and_correct = 0
orphan_found_in_history = 0
orphan_not_in_history = 0
orphan_details = []

for trade in db_open:
    tid = str(trade["trade_id"])
    if not tid.startswith("live_"):
        orphan_not_in_history += 1
        continue
    try:
        pos_id = int(tid[5:])
    except ValueError:
        orphan_not_in_history += 1
        continue

    if pos_id in curr_tickets:
        mt5_open_and_correct += 1
    elif pos_id in close_deals:
        orphan_found_in_history += 1
        closes = close_deals[pos_id]
        net_pnl = sum(d.profit for d in closes)
        orphan_details.append({
            "tid": tid,
            "pos_id": pos_id,
            "found": True,
            "net_pnl": net_pnl,
            "close_time": datetime.fromtimestamp(closes[-1].time).strftime("%m-%d %H:%M"),
            "comment": closes[-1].comment[:25],
        })
    else:
        orphan_not_in_history += 1
        orphan_details.append({
            "tid": tid,
            "pos_id": pos_id,
            "found": False,
            "net_pnl": None,
            "close_time": None,
            "comment": None,
        })

print()
print(f"=== DIAGNOSIS ===")
print(f"DB open = genuinely still open in MT5  : {mt5_open_and_correct}")
print(f"DB open = found in MT5 history (closed): {orphan_found_in_history}")
print(f"DB open = NOT in MT5 history or open   : {orphan_not_in_history}")
print()
print("These CAN be reconciled on next startup:")
print(f"  (orphan_found_in_history = {orphan_found_in_history} records)")
for d in [x for x in orphan_details if x["found"]][:10]:
    print(f"  {d['tid']:35} → closed {d['close_time']}  pnl=${d['net_pnl']:.2f}  {d['comment']}")

print()
print("These CANNOT be reconciled (missing from MT5 history):")
print(f"  (orphan_not_in_history = {orphan_not_in_history} records)")
for d in [x for x in orphan_details if not x["found"]][:10]:
    print(f"  {d['tid']}")
print()
print("=== ROOT CAUSE ===")
if orphan_found_in_history > 0:
    print(f"  {orphan_found_in_history} records ARE in MT5 history but weren't reconciled.")
    print(f"  => The startup reconciliation will fix them when bot next starts.")
if orphan_not_in_history > 0:
    print(f"  {orphan_not_in_history} records are NOT in MT5 history AND not currently open.")
    print(f"  => These are for Leg 2 positions (r2.order) that have no MT5 history entry")
    print(f"     because the DB trade_id was r1.order (different from r2.order).")
print("DONE")
