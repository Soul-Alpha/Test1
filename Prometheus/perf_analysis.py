"""One-shot 7-day performance analysis using MT5 deal history."""
import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
from collections import defaultdict

mt5.initialize()
since = datetime.now(timezone.utc) - timedelta(days=7)
deals = mt5.history_deals_get(since, datetime.now(timezone.utc))
acc = mt5.account_info()

balance = acc.balance
equity  = acc.equity
profit  = acc.profit

# Filter to XAUUSDm closes (entry=1) with non-zero PnL
closes = [d for d in deals if d.symbol == "XAUUSDm" and d.entry == 1 and d.profit != 0.0]
opens  = [d for d in deals if d.symbol == "XAUUSDm" and d.entry == 0]

wins    = [d for d in closes if d.profit > 0]
losses  = [d for d in closes if d.profit < 0]
net_pnl = sum(d.profit for d in closes)
win_pnl = sum(d.profit for d in wins)
loss_pnl= sum(d.profit for d in losses)
wr      = len(wins) / len(closes) * 100 if closes else 0

# Deposit/withdrawal
deposits = [d for d in deals if d.symbol == "" and d.profit > 0]
dep_total = sum(d.profit for d in deposits)

approx_start = balance - net_pnl - dep_total

print("=" * 54)
print("  PROMETHEUS — 7-DAY PERFORMANCE REPORT")
print("=" * 54)
print(f"  Current balance : ${balance:.2f}")
print(f"  Current equity  : ${equity:.2f}")
print(f"  Open PnL        : ${profit:.2f}")
print(f"  Deposits in 7d  : ${dep_total:.2f}")
print(f"  Approx start bal: ${approx_start:.2f}")
print()
print(f"  XAUUSDm opens   : {len(opens)}")
print(f"  XAUUSDm closes  : {len(closes)}")
print(f"  Wins            : {len(wins)}")
print(f"  Losses          : {len(losses)}")
print(f"  Win rate        : {wr:.1f}%")
print(f"  Net realised PnL: ${net_pnl:.2f}")
if wins:
    print(f"  Avg win         : ${win_pnl/len(wins):.2f}")
if losses:
    print(f"  Avg loss        : ${loss_pnl/len(losses):.2f}")
if wins and losses:
    pf = abs(win_pnl / loss_pnl) if loss_pnl != 0 else float("inf")
    rr = (win_pnl / len(wins)) / abs(loss_pnl / len(losses)) if losses else 0
    print(f"  Profit factor   : {pf:.2f}")
    print(f"  Avg R:R         : {rr:.2f}")

# By day
print()
print("  BY DAY")
print("  " + "-" * 50)
by_day = defaultdict(list)
for d in closes:
    day = datetime.fromtimestamp(d.time).strftime("%Y-%m-%d")
    by_day[day].append(d.profit)
for day in sorted(by_day):
    ps = by_day[day]
    w = sum(1 for p in ps if p > 0)
    l = sum(1 for p in ps if p < 0)
    bar = "+" * w + "-" * l
    print(f"  {day}: {len(ps):3d} trades  W{w}/L{l}  PnL ${sum(ps):+7.2f}  {bar}")

# By comment tag
print()
print("  BY STRATEGY TAG")
print("  " + "-" * 50)
by_tag = defaultdict(list)
for d in closes:
    by_tag[d.comment or "(unlabelled)"].append(d.profit)
for tag, ps in sorted(by_tag.items(), key=lambda x: sum(x[1]), reverse=True):
    w = sum(1 for p in ps if p > 0)
    l = sum(1 for p in ps if p < 0)
    print(f"  {tag:28s}  n={len(ps):3d}  W{w}/L{l}  ${sum(ps):+7.2f}")

# Best / worst
print()
print("  TOP 5 WINS")
for d in sorted(closes, key=lambda x: x.profit, reverse=True)[:5]:
    t = datetime.fromtimestamp(d.time).strftime("%m-%d %H:%M")
    print(f"    ${d.profit:+7.2f}  {t}  {d.comment}")
print("  TOP 5 LOSSES")
for d in sorted(closes, key=lambda x: x.profit)[:5]:
    t = datetime.fromtimestamp(d.time).strftime("%m-%d %H:%M")
    print(f"    ${d.profit:+7.2f}  {t}  {d.comment}")

# Open positions
open_pos = mt5.positions_get(symbol="XAUUSDm")
if open_pos:
    print()
    print(f"  OPEN POSITIONS ({len(open_pos)})")
    print("  " + "-" * 50)
    for p in open_pos:
        side = "BUY " if p.type == 0 else "SELL"
        print(f"  {p.ticket}  {side}  {p.volume}lot  entry={p.price_open:.3f}  now={p.price_current:.3f}  PnL=${p.profit:.2f}")
    print(f"  Unrealised total: ${sum(p.profit for p in open_pos):.2f}")

print()
print("=" * 54)
mt5.shutdown()
