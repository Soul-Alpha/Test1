# Zeus execution benchmark v1

This is an additive observational foundation for the backtest → paper → live
journey. It neither submits orders nor changes the existing Zeus scheduler,
approval gates, bot controls, strategies, datasets, or historical trade records.
The existing Zeus Validation workspace contains the report section; no dashboard
process is added.

## Register a frozen candidate

From Prometheus/, create a JSON file matching BenchmarkCandidate in
olympus/contracts.py. Required fields: candidate_id, source_system (prometheus
or hermes), model_version, feature_version, strategy_version, dataset_generation,
instrument, timeframe, account_currency, venue, parameters_json (an encoded JSON
object), frozen_at (timezone-aware ISO timestamp), initial_balance, and limits.

Every limits object explicitly supplies minimum_closed_trades (>=2),
minimum_expectancy (strict lower bound in account currency per complete trade),
minimum_profit_factor (>=1), maximum_realized_drawdown_pct (0–100, excluding 0),
maximum_adverse_slippage_bps and maximum_latency_seconds. Choose these before
evaluation using the separate Zeus risk policy; there are no default profitability
or live-adoption thresholds. Limits apply independently to all three stages.
Changing parameters, versions, venue, capital or limits requires a new candidate_id.

Run:

```bash
python -m olympus.core.execution_benchmark_cli register candidate.json
```

The returned SHA-256 fingerprint identifies the exact frozen specification.
The tool does not verify the authenticity of supplied timestamps or broker data;
keep original source records for audit.

## Import observations

Supply newline-delimited BenchmarkObservation JSON objects with the fingerprint.
All records carry source_system, candidate_id, stage (backtest/paper/live),
signal_id, observation_id and decision_at. Use stable signal identifiers derived
from the same decision event across execution modes. Do not match trades merely
because their timestamps or directions are similar.

```bash
python -m olympus.core.execution_benchmark_cli observe finalized-observations.jsonl
```

The v1 importer accepts one finalized closed or rejected outcome per candidate,
stage and signal. Multiple fills must be consolidated upstream using the
volume-weighted entry/exit and total costs. Open positions, amendments, partial
closes, automatic broker collection and bot instrumentation are not implemented.
A new observation identifier is not a license to replace an earlier outcome.
Exact replays are ignored; conflicting replays fail without changing history.
Each line commits independently; rerunning the file resumes safely after a failure.

Missing fields remain null; explicit known zero costs must be supplied as zero.
Commission and other_cost are nonnegative account-currency charges; swap_cost is
signed (negative for credits). gross_pnl excludes these explicit charges.
Spread/slippage already reflected in fill prices must not be deducted twice.
Cash deposits and withdrawals are not trades and must never be imported as P&L.

Complete closed-trade evidence requires intended and actual entry/size/stop/target,
fill/close times, actual exit, gross P&L, all explicit costs, spread_bps and exit_reason.
An absent stop/target is unknown in v1 and blocks completeness. This deliberately
does not infer intent from eventual fills or backfill missing historical fields.
Rejections require a reason and cannot carry fills or P&L. Their complete execution
metrics are unknown, not zero.

## Review the same evaluation window

```bash
python -m olympus.core.execution_benchmark_cli report --source hermes --candidate candidate-1 --start 2026-10-01T00:00:00Z --end 2026-10-20T00:00:00Z
```

The same UTC [start,end) decision window applies to all stages. Decisions must
follow freezing for prospective paper/live records. Backtests may predate it.
Outcomes closing at/after the end remain visible as incomplete and block checks;
their future P&L does not enter metrics. Comparisons cover finalized outcomes
inside the window and report unmatched and incompatible signals separately.
Matching also requires identical decision time, side and intended entry/size/stop/target.
No cross-candidate or cross-bot matching is performed.

Metrics are subtotals over complete evidence, not estimated full-account results:
net P&L, mean net trade expectancy, profit factor and realized closed-trade drawdown
starting at the declared initial balance. Drawdown excludes open exposure,
intratrade lows and external cash flows. Profit factor is null when no losses
exist, with a separate no_losing_trades flag; infinity never enters JSON.
All-zero returns fail expectancy and profit-factor checks.

Entry differences are signed in the adverse direction (positive = worse);
size and P&L differences are right stage minus left stage. Every difference
metric includes its own pair count. Per-trade latency and adverse slippage maxima
are checked against frozen limits; averages are also displayed.
Evidence score is a completeness fraction, not a confidence probability.

## Governance and next integration

The report always sets no_auto_adopt=true and approved_for_adoption=false.
An empty stage blocker list only means the supplied observations meet these
descriptive checks. Statistical confidence, leakage tests, independent validation,
mark-to-market risk, adequate regime coverage and explicit operator approval
remain mandatory separate work. No legacy validation status is promoted.

The ledger is stored separately under storage/olympus/execution_benchmark.sqlite3.
SQLite transactions and unique keys protect concurrent imports. Update/delete
triggers reject destructive application writes. There are no migrations over
existing trade databases. Dashboard reads do not create a missing ledger.

Next integration should instrument finalized, attributable bot/backtester outcomes
using a shared signal identifier and add mark-to-market equity evidence. Do not
infer paper/live provenance from legacy simulated rows.

## Validation

Run from Prometheus/:

```bash
python -m pytest tests/test_execution_benchmark.py -v
```

The existing repository CI also compiles modules, runs critical static checks and
executes the full test suite. It must not start live_bot/hermes.py or connect MT5.
