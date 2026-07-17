"""
patch_pending_limit_sync.py  v2
================================
Fixes: "failed cancel order #XXXXX buy 0 at market [Invalid request]"

Root cause: _pending_limits dict goes out of sync when MT5 fills a limit
order. The bot keeps trying to TRADE_ACTION_REMOVE an order that is now
an open position — MT5 rejects it as "Invalid request".

Fixes applied:
  1. In _manage_pending_limits(): check mt5.orders_get(ticket=t) before
     each TRADE_ACTION_REMOVE. If order no longer exists as pending, just
     pop it from _pending_limits silently.
  2. In _manage_pending_limits(): at the top of every call, reconcile
     _pending_limits against mt5.orders_get() — remove any tickets that
     are no longer pending (filled, cancelled externally, etc.).
"""

import pathlib, sys

TRADER = pathlib.Path(__file__).parent / "live_bot" / "trader.py"
src = TRADER.read_bytes()

for enc in ("utf-8-sig", "utf-8", "latin-1"):
    try:
        text = src.decode(enc)
        ENCODING = enc
        break
    except UnicodeDecodeError:
        continue

print(f"Encoding detected: {ENCODING}")

# =============================================================================
# PATCH 1 — reconcile _pending_limits at start of _manage_pending_limits
# =============================================================================
# Anchor: the first few lines of _manage_pending_limits()

OLD_MANAGE_TOP = """\
    def _manage_pending_limits(self) -> list[str]:
        \"\"\"Expire stale pending limit orders that haven't been filled.\"\"\"
        if self.dry_run or not MT5_AVAILABLE or not self._pending_limits:
            return []
        msgs: list[str] = []
        expired = []
        for ticket, polls_left in list(self._pending_limits.items()):\
"""

NEW_MANAGE_TOP = """\
    def _manage_pending_limits(self) -> list[str]:
        \"\"\"Expire stale pending limit orders that haven't been filled.\"\"\"
        if self.dry_run or not MT5_AVAILABLE or not self._pending_limits:
            return []

        # ── Reconcile against live MT5 pending orders ─────────────────────────
        # Orders may have been filled or cancelled externally since the last poll.
        # Remove any ticket from _pending_limits that is no longer pending in MT5
        # so we never try to TRADE_ACTION_REMOVE an already-filled order.
        try:
            _live_orders = mt5.orders_get(symbol=self.asset) or []
            _live_tickets = {o.ticket for o in _live_orders}
            _stale_filled = [t for t in list(self._pending_limits) if t not in _live_tickets]
            for _t in _stale_filled:
                logger.info(
                    "[limit] Order #%s no longer pending in MT5 (filled/cancelled) — "
                    "removing from tracker without cancel attempt.", _t
                )
                self._pending_limits.pop(_t, None)
                self._pending_limit_dirs.pop(_t, None)
        except Exception as _rec_exc:
            logger.warning("[limit] Reconcile error: %s", _rec_exc)

        if not self._pending_limits:
            return []

        msgs: list[str] = []
        expired = []
        for ticket, polls_left in list(self._pending_limits.items()):\
"""

if "Reconcile against live MT5 pending orders" in text:
    print("PATCH 1: Reconciliation already present — skipping.")
elif OLD_MANAGE_TOP not in text:
    print("ERROR: Could not find _manage_pending_limits() top anchor.")
    sys.exit(1)
else:
    text = text.replace(OLD_MANAGE_TOP, NEW_MANAGE_TOP, 1)
    print("PATCH 1: Reconciliation block inserted. OK")

# =============================================================================
# PATCH 2 — guard each TRADE_ACTION_REMOVE with an existence check
# =============================================================================
# The expiry block sends TRADE_ACTION_REMOVE without checking first.
# We add a guard: skip the send if the order is no longer pending.

OLD_EXPIRY_SEND = """\
            if polls_left <= 0:
                req = {
                    "action":   mt5.TRADE_ACTION_REMOVE,
                    "order":    ticket,
                    "comment":  "Expired",
                }
                r = mt5.order_send(req)
                if r and r.retcode == mt5.TRADE_RETCODE_DONE:
                    msgs.append(f"[limit] Cancelled expired order #{ticket}")
                    logger.info("Cancelled expired limit order #%s", ticket)
                expired.append(ticket)\
"""

NEW_EXPIRY_SEND = """\
            if polls_left <= 0:
                # Guard: only remove if the order is still pending in MT5
                _still_pending = any(
                    o.ticket == ticket
                    for o in (mt5.orders_get(symbol=self.asset) or [])
                )
                if _still_pending:
                    req = {
                        "action":   mt5.TRADE_ACTION_REMOVE,
                        "order":    ticket,
                        "comment":  "Expired",
                    }
                    r = mt5.order_send(req)
                    if r and r.retcode == mt5.TRADE_RETCODE_DONE:
                        msgs.append(f"[limit] Cancelled expired order #{ticket}")
                        logger.info("Cancelled expired limit order #%s", ticket)
                    elif r:
                        logger.warning(
                            "[limit] Expiry cancel #%s failed: retcode=%s (%s)",
                            ticket, r.retcode, r.comment,
                        )
                else:
                    logger.info(
                        "[limit] Expired order #%s already filled/cancelled — skipping remove.",
                        ticket,
                    )
                expired.append(ticket)\
"""

if "Guard: only remove if the order is still pending in MT5" in text:
    print("PATCH 2: Expiry guard already present — skipping.")
elif OLD_EXPIRY_SEND not in text:
    print("ERROR: Could not find expiry TRADE_ACTION_REMOVE block.")
    sys.exit(1)
else:
    text = text.replace(OLD_EXPIRY_SEND, NEW_EXPIRY_SEND, 1)
    print("PATCH 2: Expiry guard inserted. OK")

# =============================================================================
# PATCH 3 — guard the LTF trap cancel
# =============================================================================

OLD_LTF_CANCEL = """\
                        r = mt5.order_send({
                            "action":  mt5.TRADE_ACTION_REMOVE,
                            "order":   ticket,
                            "comment": "LTF trap",
                        })\
"""

NEW_LTF_CANCEL = """\
                        # Guard: only cancel if order is still pending
                        _still_pend_ltf = any(
                            o.ticket == ticket
                            for o in (mt5.orders_get(symbol=self.asset) or [])
                        )
                        if not _still_pend_ltf:
                            logger.info(
                                "[limit] LTF-trap cancel #%s — order already filled, skipping.",
                                ticket,
                            )
                            expired.append(ticket)
                            continue
                        r = mt5.order_send({
                            "action":  mt5.TRADE_ACTION_REMOVE,
                            "order":   ticket,
                            "comment": "LTF trap",
                        })\
"""

if "Guard: only cancel if order is still pending" in text:
    print("PATCH 3: LTF trap guard already present — skipping.")
elif OLD_LTF_CANCEL not in text:
    print("ERROR: Could not find LTF trap TRADE_ACTION_REMOVE block.")
    sys.exit(1)
else:
    text = text.replace(OLD_LTF_CANCEL, NEW_LTF_CANCEL, 1)
    print("PATCH 3: LTF trap guard inserted. OK")

# =============================================================================
# Write back
# =============================================================================
TRADER.write_bytes(text.encode(ENCODING))
print(f"\ntrader.py written ({TRADER.stat().st_size:,} bytes). Done.")
