"""
Patch: Add LTF awareness to limit order placement and monitoring.
- Placement gate: skip BuyLimit/SellLimit when both LTFs already oppose direction
- Pending monitor: cancel live pending limits when both LTFs flip to oppose direction
- Track is_long per pending limit ticket for direction-aware cancellation
"""
import re, sys

fpath = "live_bot/trader.py"
content = open(fpath, encoding="utf-8").read()
original = content

# ── Change 1: __init__ — add _current_ltf_biases and _pending_limit_dirs ──────
OLD1 = "        self._last_ltf_state: str   = \"unknown\"   # LTF alignment at last qualifying signal\n\n        self._ltf_entry_state: dict = {}           # {ticket: ltf_state} \u2014 set at open, read at close"
NEW1 = (
    "        self._last_ltf_state: str       = \"unknown\"   # LTF alignment at last qualifying signal\n"
    "        self._current_ltf_biases: list  = []           # raw LTF biases last poll \u2014 used by pending limit monitor\n"
    "        self._pending_limit_dirs: dict  = {}           # {ticket: is_long} \u2014 direction tracker for limit orders\n\n"
    "        self._ltf_entry_state: dict     = {}           # {ticket: ltf_state} \u2014 set at open, read at close"
)

if OLD1 in content:
    content = content.replace(OLD1, NEW1, 1)
    print("Change 1 OK: __init__ new fields")
else:
    print("MISS 1 -- trying probe")
    idx = content.find("_last_ltf_state: str")
    print(repr(content[idx:idx+200]))
    sys.exit(1)

# ── Change 2: store _current_ltf_biases from raw biases each poll ──────────────
OLD2 = (
    "            _ltf_aligned = [b for b in _ltf_biases if b.bias == _signal_bias]\n\n"
    "            _ltf_counter = [b for b in _ltf_biases if b.bias != _signal_bias]\n\n"
    "            if len(_ltf_biases) >= 2:"
)
NEW2 = (
    "            _ltf_aligned = [b for b in _ltf_biases if b.bias == _signal_bias]\n\n"
    "            _ltf_counter = [b for b in _ltf_biases if b.bias != _signal_bias]\n"
    "            # Store raw biases for pending-limit LTF monitor (direction-agnostic)\n"
    "            self._current_ltf_biases = [b.bias for b in _ltf_biases]\n\n"
    "            if len(_ltf_biases) >= 2:"
)
if OLD2 in content:
    content = content.replace(OLD2, NEW2, 1)
    print("Change 2 OK: _current_ltf_biases capture")
else:
    print("MISS 2 -- probe:")
    idx = content.find("_ltf_aligned = [b for b in")
    print(repr(content[idx:idx+250]))
    sys.exit(1)

# ── Change 3: LTF trap gate in _place_limit_order (after zone-too-far check) ──
# Find the MIN_SL_ATR line inside _place_limit_order and inject the gate before it.
# We anchor on the unique surrounding context.
ANCHOR3 = "        MIN_SL_ATR = 1.0\n\n        fallback_sl_dist = max(atr * MIN_SL_ATR, price * 0.003)\n\n        buf = atr * 0.15 if atr else price * 0.002"
INJECT3 = (
    "        # LTF trap gate \u2014 don't place a limit when both lower TFs already oppose direction.\n"
    "        # A BuyLimit placed while 30M+1H are both bearish will fill into downside momentum.\n"
    "        _lim_opp = \"bearish\" if is_long else \"bullish\"\n"
    "        if (len(self._current_ltf_biases) >= 2\n"
    "                and all(b == _lim_opp for b in self._current_ltf_biases)):\n"
    "            _side = \"BuyLimit\" if is_long else \"SellLimit\"\n"
    "            return (\n"
    "                f\"[limit] {_side} @ {zone_price:.4f} skipped \u2014 LTF trap \"\n"
    "                f\"(both LTFs {_lim_opp}, opposing entry direction). \"\n"
    "                f\"Waiting for LTF alignment before placing.\"\n"
    "            )\n\n"
    "        MIN_SL_ATR = 1.0\n\n"
    "        fallback_sl_dist = max(atr * MIN_SL_ATR, price * 0.003)\n\n"
    "        buf = atr * 0.15 if atr else price * 0.002"
)
if ANCHOR3 in content:
    # Only replace the FIRST occurrence (inside _place_limit_order)
    content = content.replace(ANCHOR3, INJECT3, 1)
    print("Change 3 OK: LTF placement gate")
else:
    print("MISS 3 -- probe MIN_SL_ATR:")
    idx = content.find("MIN_SL_ATR = 1.0")
    print(repr(content[max(0,idx-50):idx+150]))
    sys.exit(1)

# ── Change 4: track is_long direction in _place_limit_order tracking block ────
OLD4 = (
    "        # Track both so they expire together\n\n"
    "        self._pending_limits[order1] = LIMIT_ORDER_EXPIRY\n\n"
    "        self._pending_limits[order2] = LIMIT_ORDER_EXPIRY\n\n"
    "        # Tag both limit legs for LTF/grade learning (position_id == order on fill)\n\n"
    "        self._ltf_entry_state[order1] = self._last_ltf_state\n\n"
    "        self._ltf_entry_state[order2] = self._last_ltf_state"
)
NEW4 = (
    "        # Track both so they expire together\n\n"
    "        self._pending_limits[order1] = LIMIT_ORDER_EXPIRY\n\n"
    "        self._pending_limits[order2] = LIMIT_ORDER_EXPIRY\n"
    "        # Track direction for each leg \u2014 used by the LTF opposition monitor\n"
    "        self._pending_limit_dirs[order1] = is_long\n"
    "        self._pending_limit_dirs[order2] = is_long\n\n"
    "        # Tag both limit legs for LTF/grade learning (position_id == order on fill)\n\n"
    "        self._ltf_entry_state[order1] = self._last_ltf_state\n\n"
    "        self._ltf_entry_state[order2] = self._last_ltf_state"
)
if OLD4 in content:
    content = content.replace(OLD4, NEW4, 1)
    print("Change 4 OK: direction tracking")
else:
    print("MISS 4 -- probe pending_limits[order1]:")
    idx = content.find("self._pending_limits[order1] = LIMIT_ORDER_EXPIRY")
    print(repr(content[max(0,idx-50):idx+300]))
    sys.exit(1)

# ── Change 5: LTF opposition cancel + cleanup in _manage_pending_limits ────────
OLD5 = (
    "            else:\n\n"
    "                self._pending_limits[ticket] = polls_left\n\n"
    "        for t in expired:\n\n"
    "            self._pending_limits.pop(t, None)\n\n"
    "        return msgs"
)
NEW5 = (
    "            else:\n\n"
    "                # LTF opposition check \u2014 cancel early if both LTFs flipped against this limit's direction.\n"
    "                # Example: BuyLimit placed while 30M+1H were aligned; they have since both turned bearish\n"
    "                # \u2014 letting it fill now means filling into momentum opposing the trade.\n"
    "                _lim_dir = self._pending_limit_dirs.get(ticket)\n"
    "                if _lim_dir is not None and len(self._current_ltf_biases) >= 2:\n"
    "                    _opp = \"bearish\" if _lim_dir else \"bullish\"\n"
    "                    if all(b == _opp for b in self._current_ltf_biases):\n"
    "                        r = mt5.order_send({\n"
    "                            \"action\":  mt5.TRADE_ACTION_REMOVE,\n"
    "                            \"order\":   ticket,\n"
    "                            \"comment\": \"LTF trap\",\n"
    "                        })\n"
    "                        if r and r.retcode == mt5.TRADE_RETCODE_DONE:\n"
    "                            _lim_label = \"BuyLimit\" if _lim_dir else \"SellLimit\"\n"
    "                            msgs.append(\n"
    "                                f\"[limit] Cancelled #{ticket} \u2014 LTF trap \"\n"
    "                                f\"(both LTFs {self._current_ltf_biases} opposing {_lim_label})\"\n"
    "                            )\n"
    "                            logger.info(\n"
    "                                \"[limit] Cancelled #%s \u2014 LTF trap: both LTFs %s opposing %s\",\n"
    "                                ticket, self._current_ltf_biases, _lim_label,\n"
    "                            )\n"
    "                        expired.append(ticket)\n"
    "                        continue\n\n"
    "                self._pending_limits[ticket] = polls_left\n\n"
    "        for t in expired:\n\n"
    "            self._pending_limits.pop(t, None)\n"
    "            self._pending_limit_dirs.pop(t, None)  # clean direction tracker on expiry/cancel\n\n"
    "        return msgs"
)
if OLD5 in content:
    content = content.replace(OLD5, NEW5, 1)
    print("Change 5 OK: LTF cancel + cleanup")
else:
    print("MISS 5 -- probe else block:")
    idx = content.find("self._pending_limits[ticket] = polls_left")
    print(repr(content[max(0,idx-80):idx+150]))
    sys.exit(1)

# ── Write ──────────────────────────────────────────────────────────────────────
if content == original:
    print("ERROR: no changes made")
    sys.exit(1)

open(fpath, "w", encoding="utf-8").write(content)
print("Patched OK")
