"""
patch_ltf_sell_limit.py
=======================
Makes Prometheus place SellLimits (and BuyLimits) based on LTF OBs
instead of HTF OBs, giving more precise limit-order prices.

Changes:
  1. Inserts _find_ltf_ob() helper before _place_limit_order().
  2. After HTF OB detection, overrides ob_zone with the nearest fresh
     LTF OB for short entries (and optionally long entries).
"""

import pathlib, re, sys

TRADER = pathlib.Path(__file__).parent / "live_bot" / "trader.py"

src = TRADER.read_bytes()

# ── Detect file encoding ──────────────────────────────────────────────────────
for enc in ("utf-8-sig", "utf-8", "latin-1"):
    try:
        text = src.decode(enc)
        ENCODING = enc
        break
    except UnicodeDecodeError:
        continue

print(f"Encoding detected: {ENCODING}")

# =============================================================================
# PATCH 1 — insert _find_ltf_ob() before _place_limit_order()
# =============================================================================

NEW_METHOD = '''
    def _find_ltf_ob(
        self,
        direction: str,
        approx_price: float,
        atr_approx: float,
    ):
        """Return the nearest fresh LTF Order Block as (low, high) or None.

        Uses 5m bars first, falls back to 15m.  The LTF OBs give a more
        precise limit-order price than the HTF OBs from the main analysis.
        """
        if not self._last_tf_data or self.engine is None:
            return None
        try:
            ltf_df = (
                self._last_tf_data.get("5m")
                or self._last_tf_data.get("15m")
                or self._last_tf_data.get("30m")
            )
            if ltf_df is None or len(ltf_df) < 20:
                return None
            smc_eng = getattr(self.engine, "smc_engine", None)
            if smc_eng is None:
                return None
            import pandas as pd   # noqa: PLC0415
            df = ltf_df.copy()
            df.columns = [c.lower() for c in df.columns]
            atr_s = (df["high"] - df["low"]).rolling(14).mean()
            ltf_atr = float(atr_s.iloc[-1]) if not atr_s.empty else None
            obs = smc_eng.detect_order_blocks(df, ltf_atr)
            fresh = [ob for ob in obs if ob.direction == direction and not ob.mitigated]
            if not fresh:
                return None
            ref_fn = (lambda b: b.high) if direction == "bullish" else (lambda b: b.low)
            fresh.sort(key=lambda b: (
                abs(ref_fn(b) - approx_price),
                -b.strength,
            ))
            max_dist = MAX_LIMIT_DISTANCE_ATR * (atr_approx if atr_approx > 0 else 1.0)
            for ob in fresh:
                dist = abs(ref_fn(ob) - approx_price)
                if atr_approx == 0 or dist <= max_dist:
                    logger.info(
                        "[ltf_ob] LTF %s OB %.4f-%.4f selected (dist=%.4f, strength=%.2f)",
                        direction, ob.low, ob.high, dist, ob.strength,
                    )
                    return (ob.low, ob.high)
        except Exception as exc:
            logger.warning("[ltf_ob] Detection failed: %s", exc)
        return None

'''

TARGET_DEF = "    def _place_limit_order("

if "_find_ltf_ob" in text:
    print("PATCH 1: _find_ltf_ob() already present — skipping.")
elif TARGET_DEF not in text:
    print("ERROR: Could not find _place_limit_order() insertion point.")
    sys.exit(1)
else:
    text = text.replace(TARGET_DEF, NEW_METHOD + TARGET_DEF, 1)
    print("PATCH 1: Inserted _find_ltf_ob() before _place_limit_order(). OK")

# =============================================================================
# PATCH 2 — override ob_zone with LTF OB for short entries after HTF block
# =============================================================================
# We anchor on the end of the HTF OB selection loop, then inject the LTF
# override immediately before the dry-run check.

ANCHOR_OLD = """        if self.dry_run or not MT5_AVAILABLE or not self._mt5_connected:

            side = "BUY" if is_long else "SELL\""""

LTF_OVERRIDE = """\
        # \u2500\u2500 LTF Order Block override for limit-order placement \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        # Sell limits and buy limits use LTF OBs for precise entry price.
        # HTF OBs identify the zone; LTF OBs pinpoint where price retraces to.
        if is_short:
            _ltf_ob = self._find_ltf_ob("bearish", _approx_price, _atr_approx)
            if _ltf_ob:
                logger.info(
                    "[ob] SellLimit: overriding HTF OB %s with LTF bearish OB %.4f-%.4f",
                    ob_zone, _ltf_ob[0], _ltf_ob[1],
                )
                ob_zone = _ltf_ob
        elif is_long:
            _ltf_ob = self._find_ltf_ob("bullish", _approx_price, _atr_approx)
            if _ltf_ob:
                logger.info(
                    "[ob] BuyLimit: overriding HTF OB %s with LTF bullish OB %.4f-%.4f",
                    ob_zone, _ltf_ob[0], _ltf_ob[1],
                )
                ob_zone = _ltf_ob

"""

ANCHOR_NEW = LTF_OVERRIDE + ANCHOR_OLD

if "LTF Order Block override for limit-order placement" in text:
    print("PATCH 2: LTF OB override already present — skipping.")
elif ANCHOR_OLD not in text:
    # Try a byte-by-byte search (mojibake variant)
    print("WARNING: Anchor not found as plain text. Trying bytes search.")
    # Fallback: find by unique sub-string
    anchor_short = '        if self.dry_run or not MT5_AVAILABLE or not self._mt5_connected:'
    if anchor_short not in text:
        print("ERROR: Could not find dry-run anchor for patch 2.")
        sys.exit(1)
    # inject before the first occurrence inside _execute_from_result
    # (find it after _execute_from_result definition)
    exec_pos = text.find("def _execute_from_result(")
    dry_pos  = text.find(anchor_short, exec_pos)
    if dry_pos == -1:
        print("ERROR: dry-run anchor not found after _execute_from_result().")
        sys.exit(1)
    text = text[:dry_pos] + LTF_OVERRIDE + text[dry_pos:]
    print("PATCH 2: LTF OB override injected via fallback anchor. OK")
else:
    text = text.replace(ANCHOR_OLD, ANCHOR_NEW, 1)
    print("PATCH 2: LTF OB override injected before dry-run check. OK")

# =============================================================================
# Write back
# =============================================================================
TRADER.write_bytes(text.encode(ENCODING))
print(f"\ntrader.py written ({TRADER.stat().st_size:,} bytes). Done.")
