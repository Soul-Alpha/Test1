import sys
sys.path.insert(0, 'c:/Users/Chaba/Documents/tradingBots/Prometheus')
from storage.database import list_trades

rows = list_trades(source='live', limit=100)
wins   = [r for r in rows if r['status'] == 'win']
losses = [r for r in rows if r['status'] == 'loss']
opens  = [r for r in rows if r['status'] == 'open']
closed = wins + losses

win_pnl  = sum(r['pnl'] or 0 for r in wins)
loss_pnl = sum(r['pnl'] or 0 for r in losses)
total_pnl = sum(r['pnl'] or 0 for r in rows)
wr = len(wins) / len(closed) * 100 if closed else 0
avg_win  = win_pnl  / len(wins)   if wins   else 0
avg_loss = loss_pnl / len(losses) if losses else 0

print("=" * 55)
print(f"  Total trades : {len(rows)}")
print(f"  Wins         : {len(wins)}   (${win_pnl:+.2f})")
print(f"  Losses       : {len(losses)}  (${loss_pnl:+.2f})")
print(f"  Open now     : {len(opens)}")
print(f"  Win rate     : {wr:.1f}%")
print(f"  Avg win      : ${avg_win:+.2f}")
print(f"  Avg loss     : ${avg_loss:+.2f}")
print(f"  Total P&L    : ${total_pnl:+.2f}")
if avg_loss != 0:
    print(f"  Profit factor: {abs(win_pnl/loss_pnl):.2f}" if loss_pnl else "  Profit factor: inf")
print("=" * 55)
print()
print("Recent 20 trades:")
print(f"  {'Status':<7} {'Dir':<5} {'TF':<5} {'Entry':>9} {'PnL':>8}")
print("  " + "-" * 40)
for r in rows[:20]:
    p = r['pnl'] or 0
    print(f"  {r['status']:<7} {(r['direction'] or '?'):<5} {(r['timeframe'] or '?'):<5} "
          f"{str(r['entry_price'] or '?'):>9}  {p:>+7.2f}")
