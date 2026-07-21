# Historical Evidence Architecture

Hermes runtime status files are current-state snapshots. They are deliberately
small and may be replaced at process startup, so they are not used as the
canonical historical store.

## Canonical Hermes ledger

`storage/olympus/hermes_evidence.jsonl` is the local append-only evidence
ledger. It records idempotent events for:

- prediction creation;
- simulated trade entry; and
- simulated trade closure with the full return and execution-quality payload.

The ledger is runtime data and is ignored by Git. Back it up with the other
local `storage/olympus` artifacts. Do not commit account evidence or use the
ledger to alter live execution rules.

At startup Hermes imports the last compatible runtime snapshot into the ledger,
then reconstructs open and closed paper-trade state. Snapshot-derived closure
times are marked as observations rather than exact timestamps and are excluded
from duration metrics.

## Evidence identity and provenance

Historical bootstrap setup IDs are deterministic over instrument, timeframe,
candle timestamp, strategy version, and feature version. `PatternLearner`
rejects an already-known identity, preventing restarts from duplicating the same
bootstrap observation.

Every ledger event carries source, execution type, model, feature, strategy,
and dataset-generation metadata. Backtest, simulated, and live evidence must
remain distinguishable.

## Readiness statuses

The Hermes analytics payload exposes `evidence_readiness`. Each metric family
reports its source, sample count, required count, maturity stage, missing fields,
and one of these states:

- `ready`
- `insufficient_samples`
- `missing_history`
- `missing_fields`
- `missing_runtime_snapshot`

The dashboard renders this registry so missing artifacts and insufficient
evidence are not both presented as the generic “Awaiting Historical Data”.

Evidence maturity thresholds are 10 (insufficient floor), 30 (emerging), 75
(developing), 150 (validated), and 300 (elite). These labels are observational
and cannot authorize strategy adoption or change trading behavior.
