import json
d = json.loads(open('live_bot/learning_state.json').read())
total = d['wins'] + d['losses']
wr = d['wins']/total*100 if total else 0
print(f"Wins: {d['wins']}  Losses: {d['losses']}  WR: {wr:.1f}%")
print(f"Total seen: {d['total_seen']}  Entry rate: {total/max(1,d['total_seen'])*100:.1f}%")
print(f"Total PnL: ${d['total_pnl']:.2f}")
print(f"Streak: {d['streak']:+d}  Best: {d['best_streak']}  Worst: {d['worst_streak']}")
print(f"Score adjust: {d['score_adjust']:+.1f}")
print(f"Last 20: {d['last_20_results']}")
print()
print("Grade stats:")
for g, s in (d.get('grade_stats') or {}).items():
    t2 = s['wins']+s['losses']
    print(f"  {g}: W={s['wins']} L={s['losses']} seen={s.get('seen','?')} WR={s['wins']/t2*100 if t2 else 0:.0f}%")
print()
print("Direction stats:")
for g, s in (d.get('direction_stats') or {}).items():
    t2 = s['wins']+s['losses']
    print(f"  {g}: W={s['wins']} L={s['losses']} WR={s['wins']/t2*100 if t2 else 0:.0f}%")
print()
print("OB stats:")
for g, s in (d.get('ob_stats') or {}).items():
    print(f"  {g}: hits={s['hits']} wins={s['wins']} WR={s['wins']/max(1,s['hits'])*100:.0f}%")
print()
print("LTF stats:")
for g, s in (d.get('ltf_stats') or {}).items():
    if isinstance(s, dict):
        t2 = s['wins']+s['losses']
        print(f"  {g}: W={s['wins']} L={s['losses']} WR={s['wins']/t2*100 if t2 else 0:.0f}%")
    else:
        print(f"  {g}: {s}")
