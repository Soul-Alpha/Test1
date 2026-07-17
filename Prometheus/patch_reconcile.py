"""Insert _reconcile_db_on_startup into trader.py"""

new_method = """
    # -- Startup DB reconciliation -------------------------------------------------

    def _reconcile_db_on_startup(self) -> None:
        \"\"\"On the first poll, backfill any DB trades still marked 'open' that
        MT5 has already closed.  Handles bot restarts where positions closed
        during downtime.\"\"\"
        if self.dry_run or not MT5_AVAILABLE:
            return
        try:
            open_db = [t for t in list_trades(source="live", limit=1000)
                       if t.get("status") == "open" and t.get("trade_id")]
            if not open_db:
                return
            now   = datetime.utcnow()
            start = now - timedelta(days=30)
            all_deals = mt5.history_deals_get(
                datetime.utcfromtimestamp(start.timestamp()), now
            ) or []
            close_deals = {}
            for d in all_deals:
                if d.entry == mt5.DEAL_ENTRY_OUT:
                    close_deals.setdefault(d.position_id, []).append(d)

            updated = 0
            for trade in open_db:
                tid = str(trade["trade_id"])
                if not tid.startswith("live_"):
                    continue
                try:
                    pos_id = int(tid[5:])
                except ValueError:
                    continue
                closes = close_deals.get(pos_id)
                if not closes:
                    continue
                net_pnl    = sum(d.profit for d in closes)
                exit_price = closes[-1].price
                status     = "win" if net_pnl > 0 else "loss"
                try:
                    save_trade({
                        "trade_id":   tid,
                        "status":     status,
                        "pnl":        round(net_pnl, 2),
                        "exit_price": exit_price,
                        "exit_bar":   0,
                    })
                    updated += 1
                except Exception as _dbe:
                    logger.debug("startup reconcile save error for %s: %s", tid, _dbe)

            if updated:
                logger.info(
                    "Startup DB reconciliation: updated %d previously-open trades.", updated
                )
        except Exception as exc:
            logger.warning("Startup DB reconciliation failed: %s", exc)

"""

with open('live_bot/trader.py', 'r', encoding='utf-8') as f:
    content = f.read()

marker = '    def _learn_from_closes(self, now_open:'
idx = content.find(marker)
if idx == -1:
    print('ERROR: marker not found')
else:
    content = content[:idx] + new_method + content[idx:]
    with open('live_bot/trader.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'OK: inserted _reconcile_db_on_startup ({len(new_method)} chars) before _learn_from_closes at idx={idx}')
