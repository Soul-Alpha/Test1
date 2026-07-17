import pathlib, sys
NL = "\r\n"
t = pathlib.Path("live_bot/trader.py").read_bytes().decode("utf-8-sig")

# PATCH 1: reconcile after "return []" inside _manage_pending_limits
ANCHOR1 = "def _manage_pending_limits(self) -> list[str]:"
p1_marker = "Reconcile-filled-limits"
if p1_marker in t:
    print("P1: skip")
else:
    pos = t.find(ANCHOR1)
    if pos == -1:
        print("P1 ERROR: method not found"); sys.exit(1)
    ret_pos = t.find("return []", pos)
    after_ret = ret_pos + len("return []")
    block = (NL + NL +
        "        # " + p1_marker + " ---- purge filled/cancelled orders ----" + NL +
        "        try:" + NL +
        "            _lv = {o.ticket for o in (mt5.orders_get(symbol=self.asset) or [])}" + NL +
        "            for _tk in [x for x in list(self._pending_limits) if x not in _lv]:" + NL +
        "                logger.info(\"[pending] #%s no longer pending, removing\", _tk)" + NL +
        "                self._pending_limits.pop(_tk, None)" + NL +
        "                self._pending_limit_dirs.pop(_tk, None)" + NL +
        "        except Exception as _ex:" + NL +
        "            logger.warning(\"[pending] reconcile err: %s\", _ex)" + NL +
        "        if not self._pending_limits:" + NL +
        "            return []"
    )
    t = t[:after_ret] + block + t[after_ret:]
    print("P1: OK")

# PATCH 2: guard expiry remove - find req = { with "Expired" comment and prepend check
p2_marker = "Guard-expiry-2026"
if p2_marker in t:
    print("P2: skip")
else:
    expired_idx = t.find('"comment":  "Expired",')
    if expired_idx == -1:
        print("P2 ERROR"); sys.exit(1)
    req_idx = t.rfind("                req = {", 0, expired_idx)
    guard = ("                # " + p2_marker + NL +
             "                if not any(o.ticket == ticket for o in (mt5.orders_get(symbol=self.asset) or [])):" + NL +
             "                    logger.info(\"[pending] #%s already filled/gone, skip expiry remove\", ticket)" + NL +
             "                    expired.append(ticket)" + NL +
             "                    continue" + NL + NL
    )
    t = t[:req_idx] + guard + t[req_idx:]
    print("P2: OK")

# PATCH 3: guard LTF trap remove
p3_marker = "Guard-ltftrap-2026"
if p3_marker in t:
    print("P3: skip")
else:
    ltf_idx = t.find('"comment": "LTF trap"')
    if ltf_idx == -1:
        print("P3 ERROR"); sys.exit(1)
    send_idx = t.rfind("r = mt5.order_send({", 0, ltf_idx)
    guard_ltf = ("                        # " + p3_marker + NL +
                 "                        if not any(o.ticket == ticket for o in (mt5.orders_get(symbol=self.asset) or [])):" + NL +
                 "                            logger.info(\"[pending] #%s already filled, skip LTF trap remove\", ticket)" + NL +
                 "                            expired.append(ticket)" + NL +
                 "                            continue" + NL
    )
    t = t[:send_idx] + guard_ltf + t[send_idx:]
    print("P3: OK")

pathlib.Path("live_bot/trader.py").write_bytes(t.encode("utf-8-sig"))
print("Written OK", pathlib.Path("live_bot/trader.py").stat().st_size, "bytes")
