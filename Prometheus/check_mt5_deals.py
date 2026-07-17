import sys, datetime
sys.path.insert(0, 'c:/Users/Chaba/Documents/tradingBots/Prometheus')
import MetaTrader5 as mt5

mt5.initialize()
now   = datetime.datetime.now()
start = now.replace(hour=0, minute=0, second=0, microsecond=0) - datetime.timedelta(days=2)
deals = mt5.history_deals_get(start, now) or []

prom   = [d for d in deals if d.magic in (777000, 777001, 777002)]
wins   = [d for d in prom if d.profit > 0]
losses = [d for d in prom if d.profit < 0]
total  = sum(d.profit for d in prom)

print("=" * 60)
print("MT5 CLOSED DEALS (last 2 days — Prometheus only)")
print("=" * 60)
print("Total deals : %d" % len(prom))
print("Wins        : %d  ($%.2f)" % (len(wins), sum(d.profit for d in wins)))
print("Losses      : %d  ($%.2f)" % (len(losses), sum(d.profit for d in losses)))
print("Net P&L     : $%.2f" % total)
if losses:
    print("Win rate    : %.1f%%" % (len(wins)/len(prom)*100))
    avg_w = sum(d.profit for d in wins)/len(wins) if wins else 0
    avg_l = sum(d.profit for d in losses)/len(losses) if losses else 0
    print("Avg win     : $%.2f" % avg_w)
    print("Avg loss    : $%.2f" % avg_l)
print()
print("Last 25 closed deals:")
print("  %-6s  %8s  %9s  %s" % ("Result","P&L","Entry","Comment"))
print("  " + "-" * 50)
for d in sorted(prom, key=lambda x: x.time, reverse=True)[:25]:
    tag = "WIN " if d.profit > 0 else ("LOSS" if d.profit < 0 else "EVEN")
    print("  %-6s  %+8.2f  %9.3f  %s" % (tag, d.profit, d.price, d.comment or ""))

mt5.shutdown()
