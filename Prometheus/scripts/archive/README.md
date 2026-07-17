# Patch Script Archive — Provenance Inventory

This directory contains legacy one-off patch and hotfix scripts that were
previously located in the Prometheus root directory.  They have been **moved
here as part of Phase 1.4 (Patch and Operational Script Governance)** and must
not be executed without explicit operator review and approval.

> **Do not re-run any script in this directory without first reading its
> provenance record below and confirming that its intended change has not
> already been applied to the current codebase.**

---

## Script inventory

| Script | Classification | Purpose | Applied? | Notes |
|--------|----------------|---------|----------|-------|
| `patch3.py` | Hotfix – live-bot execution | Patches `_check_manual_override` in `live_bot/trader.py` to validate required fields before processing manual trade JSON. | Unknown — requires inspection of current `trader.py` to verify. | Do not re-run if the manual-override validation logic is already present. |
| `patch_bootstrap.py` | Hotfix – learning state | Patches `_bootstrap_from_db` in `live_bot/trader.py` so that it re-seeds from the trade database even when the learning file exists but has zero win/loss data. | Unknown — requires inspection of current `trader.py` to verify. | Prevents silent loss of historical learning state across restarts. |
| `patch_ltf_sell_limit.py` | Feature hotfix – execution | Replaces HTF OB-based limit prices with LTF OB prices for more precise limit-order entry in `live_bot/trader.py`. | Unknown — requires inspection of current `trader.py` to verify. | Injects `_find_ltf_ob()` helper method into trader. |
| `patch_pending_limit_sync.py` | Bug fix – execution | Fixes "Invalid request" errors when cancelling limit orders that MT5 has already filled. Adds reconciliation logic in `_manage_pending_limits`. | Unknown — requires inspection of current `trader.py` to verify. | Critical for preventing repeated spurious MT5 error logs. |
| `patch_reconcile.py` | Bug fix – execution | Inserts `_reconcile_db_on_startup` into `live_bot/trader.py` to back-fill DB trades still marked "open" that MT5 closed while the bot was offline. | Unknown — requires inspection of current `trader.py` to verify. | Prevents stale state after bot restarts. |
| `patch_sync2.py` | Hotfix – execution | Extends `_manage_pending_limits` with a filled-limits reconciliation pass to purge tickets no longer present in MT5. | Unknown — requires inspection of current `trader.py` to verify. | Superseded by or complements `patch_pending_limit_sync.py`. |

---

## Governance actions required

1. **Verify application status** — For each script, inspect `live_bot/trader.py`
   to confirm whether the described change is already present.  Document the
   finding in the table above under the *Applied?* column.

2. **Promote or close** — If a fix is confirmed present in the codebase, close
   the script as *applied* and note the commit or PR that incorporated it.
   If it is not present, open a governed migration (proper PR with review and
   tests) rather than re-running this patch script.

3. **Archive freeze** — These files are preserved read-only for lineage.  Once
   all entries are resolved, this inventory becomes the permanent provenance
   record.  Do not delete the scripts until their application status is
   confirmed and recorded in version control history.

---

## How to add a new operational script

Operational scripts should **never** be created in the repository root.  Instead:

1. Create the script under `scripts/` (not `scripts/archive/`).
2. Add an entry to `scripts/README.md` documenting purpose, safe invocation,
   and whether it is idempotent.
3. Open a pull request for review before first execution in any production or
   paper-trading environment.
