import sys, re

path = "live_bot/trader.py"
with open(path, "r", encoding="utf-8", errors="replace") as f:
    t = f.read()

idx = t.find("    def _check_manual_override")
end = t.find("\n    def _poll", idx)

NEW = '''    def _check_manual_override(self) -> Optional[str]:
        """Check for a manual trade JSON written by the dashboard."""
        if not MANUAL_TRADE_FILE.exists():
            return None
        try:
            data = json.loads(MANUAL_TRADE_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Cannot read manual_trade.json: %s", exc)
            MANUAL_TRADE_FILE.unlink(missing_ok=True)
            return None
        required = {"direction", "sl", "tp"}
        if not required.issubset(data.keys()):
            logger.warning("manual_trade.json missing fields: %s", required - data.keys())
            MANUAL_TRADE_FILE.unlink(missing_ok=True)
            return None
        MANUAL_TRADE_FILE.unlink(missing_ok=True)
        raw_dir     = str(data["direction"]).upper()
        is_long     = raw_dir in ("BUY", "LONG", "BULLISH")
        is_short    = raw_dir in ("SELL", "SHORT", "BEARISH")
        is_limit    = str(data.get("order_type", "market")).lower() == "limit"
        entry_price = data.get("entry")
        if entry_price is not None:
            entry_price = float(entry_price)
        if not is_long and not is_short:
            logger.warning("manual_trade.json -- unrecognised direction: %s", raw_dir)
            return None
        sl      = float(data["sl"])
        tp      = float(data["tp"])
        lots    = float(data.get("lots") or 0)
        comment = str(data.get("comment") or "Prom-manual")[:31]
        _otype  = ("BuyLimit" if is_long else "SellLimit") if is_limit else ("BUY" if is_long else "SELL")
        logger.info("Manual override: %s SL=%.4f TP=%.4f entry=%s lots=%s",
                    _otype, sl, tp, entry_price or "market", lots or "auto")
        if self.dry_run or not MT5_AVAILABLE or not self._mt5_connected:
            msg = (f"[DRY RUN] [manual] Would {_otype} {self.asset} "
                   f"entry={entry_price or 'market'} SL={sl:.4f} TP={tp:.4f}")
            logger.info(msg)
            self._total_trades += 1
            return msg
        tick = mt5.symbol_info_tick(self.asset)
        if not tick:
            return f"Manual override failed: no tick data for {self.asset}"
        sym  = mt5.symbol_info(self.asset)
        fill = mt5.ORDER_FILLING_IOC
        if sym and (sym.filling_mode & mt5.ORDER_FILLING_FOK):
            fill = mt5.ORDER_FILLING_FOK
        if is_limit and entry_price is not None:
            if lots <= 0:
                sl_dist = abs(entry_price - sl)
                lots    = self._calc_lot(sl_dist)
            lim_type = mt5.ORDER_TYPE_BUY_LIMIT if is_long else mt5.ORDER_TYPE_SELL_LIMIT
            req = {
                "action":       mt5.TRADE_ACTION_PENDING,
                "symbol":       self.asset,
                "volume":       lots,
                "type":         lim_type,
                "price":        round(entry_price, 5),
                "sl":           round(sl, 5),
                "tp":           round(tp, 5),
                "deviation":    20,
                "magic":        777_002,
                "comment":      comment,
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": fill,
            }
            r2 = mt5.order_send(req)
            if r2 is None or r2.retcode != mt5.TRADE_RETCODE_DONE:
                code = r2.retcode if r2 else "None"
                err  = r2.comment if r2 else "no response"
                return f"Manual limit FAILED retcode={code} ({err})"
            self._total_trades += 1
            msg = (f"[MANUAL-LIMIT] {_otype} ticket={r2.order} "
                   f"{lots} lots @ {entry_price:.4f} SL={sl:.4f} TP={tp:.4f}")
            logger.info(msg)
            return msg
        else:
            price = tick.ask if is_long else tick.bid
            if lots <= 0:
                sl_dist = abs(price - sl)
                lots    = self._calc_lot(sl_dist)
            req = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "symbol":       self.asset,
                "volume":       lots,
                "type":         mt5.ORDER_TYPE_BUY if is_long else mt5.ORDER_TYPE_SELL,
                "price":        price,
                "sl":           round(sl, 5),
                "tp":           round(tp, 5),
                "deviation":    20,
                "magic":        777_000,
                "comment":      comment,
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": fill,
            }
            r2 = mt5.order_send(req)
            if r2 is None or r2.retcode != mt5.TRADE_RETCODE_DONE:
                code = r2.retcode if r2 else "None"
                err  = r2.comment if r2 else "no response"
                return f"Manual order FAILED retcode={code} ({err})"
            self._total_trades += 1
            msg = (f"[MANUAL] {'BUY' if is_long else 'SELL'} ticket={r2.order} "
                   f"{lots} lots @ {price:.4f} SL={sl:.4f} TP={tp:.4f}")
            logger.info(msg)
            return msg

'''

t = t[:idx] + NEW + t[end:]
with open(path, "w", encoding="utf-8") as f:
    f.write(t)
print("patched OK")
