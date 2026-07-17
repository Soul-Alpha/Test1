"""Patch _bootstrap_from_db to also seed from DB when the file has zero wins+losses."""
import json
from pathlib import Path

new_fn = (
    'def _bootstrap_from_db() -> None:\n'
    '    """Reconstruct wins/losses from the trade DB.\n'
    '\n'
    '    Runs if no learning file exists, OR if the file has zero win/loss data\n'
    '    (e.g. it was saved before any trades ever closed).  Ensures historical\n'
    '    performance is never silently lost across restarts.\n'
    '    """\n'
    '    file_has_data = False\n'
    '    if LEARNING_FILE.exists():\n'
    '        try:\n'
    '            saved = json.loads(LEARNING_FILE.read_text(encoding="utf-8"))\n'
    '            file_has_data = (saved.get("wins", 0) + saved.get("losses", 0)) > 0\n'
    '        except Exception:\n'
    '            pass\n'
    '    if file_has_data:\n'
    '        return   # file already has real data -- _load_learning will handle it\n'
    '    try:\n'
    '        from storage.database import list_trades as _lt\n'
    '        all_trades = _lt(source="live", limit=1000)\n'
    '        for t in all_trades:\n'
    '            status = (t.get("status") or "").lower()\n'
    '            pnl    = t.get("pnl") or 0.0\n'
    '            if status == "win":\n'
    '                _LEARNING["wins"]      += 1\n'
    '                _LEARNING["total_pnl"] += pnl\n'
    '                _LEARNING["last_20_results"].append(1)\n'
    '            elif status == "loss":\n'
    '                _LEARNING["losses"]    += 1\n'
    '                _LEARNING["total_pnl"] += pnl\n'
    '                _LEARNING["last_20_results"].append(0)\n'
    '        _LEARNING["last_20_results"] = _LEARNING["last_20_results"][-20:]\n'
    '        if _LEARNING["wins"] + _LEARNING["losses"] > 0:\n'
    '            logger.info(\n'
    '                "[LML] Bootstrapped from DB: W=%d L=%d total_pnl=$%.2f",\n'
    '                _LEARNING["wins"], _LEARNING["losses"], _LEARNING["total_pnl"],\n'
    '            )\n'
    '    except Exception as _exc:\n'
    '        logger.debug("[LML] DB bootstrap error: %s", _exc)\n'
)

trader_path = Path('live_bot/trader.py')
content = trader_path.read_text(encoding='utf-8')

# Find start and end of the old function
old_start = 'def _bootstrap_from_db() -> None:\n'
old_end_marker = '        logger.debug("[LML] DB bootstrap error: %s", _exc)\n'

i = content.find(old_start)
j = content.find(old_end_marker, i)
if i == -1 or j == -1:
    print("MARKERS NOT FOUND — checking content around line 206:")
    lines = content.splitlines()
    for idx, ln in enumerate(lines[203:215], start=204):
        print(idx, repr(ln))
else:
    j_end = j + len(old_end_marker)
    content = content[:i] + new_fn + content[j_end:]
    trader_path.write_text(content, encoding='utf-8')
    print(f'OK: replaced _bootstrap_from_db (chars {i}-{j_end})')
