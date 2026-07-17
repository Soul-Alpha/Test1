"""One-shot patch: inject small-account single-leg early return before Leg 2."""
import pathlib, re

f = pathlib.Path(__file__).parent / "trader.py"
content = f.read_text(encoding="utf-8")

# --- Find the exact separator line (bytes as they appear in file) ---
# The line looks like: "        # ─── Leg 2: TP2 – trail to 1:5 ───..."
# We search by the unique surrounding context instead.
MARKER = 'logger.debug("DB save_trade (open) error: %s", _dbe)\n'

if MARKER not in content:
    print("ERROR: marker not found")
    raise SystemExit(1)

INSERT = (
    '\n        # Small accounts: single-leg mode -- skip Leg 2, keep MAX_OPEN slots free.\n'
    '        if _is_small:\n'
    '            logger.info(\n'
    '                "[small_acct] Single-leg market entry: TP1=%.5f lot=%.2f bal=$%.2f",\n'
    '                tp1, _single_leg_lot, _cur_bal,\n'
    '            )\n'
    '            self._total_trades += 1\n'
    '            side_str = "BUY" if is_long else "SELL"\n'
    '            return f"[{label}] Small-acct single-leg {side_str} @ {price:.5f}"\n'
)

# Insert right after the MARKER (only the first occurrence — the save_trade block)
idx = content.index(MARKER) + len(MARKER)
content = content[:idx] + INSERT + content[idx:]

f.write_text(content, encoding="utf-8")
print("Patched OK")
