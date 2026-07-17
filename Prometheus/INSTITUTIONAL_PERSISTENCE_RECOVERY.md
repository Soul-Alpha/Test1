# Olympus Institutional Persistence and Recovery

## Purpose
Olympus institutional knowledge must survive process restarts without resetting cumulative metrics.

## Root Cause Summary
- Runtime dashboards read point-in-time JSON snapshots directly.
- Several writers updated snapshots with non-atomic `write_text(...)`, which can expose empty/truncated files during write windows.
- On startup, dashboards defaulted missing/invalid payloads to `{}` and rendered zero-like values before new runtime cycles completed.
- Result: temporary institutional amnesia (display-level resets), despite append-only history existing.

## Corrective Architecture
1. Added an Institutional State Recovery Engine:
   - Module: `olympus/core/institutional_state_recovery.py`
   - Reconstructs state from runtime snapshots first, then append-only histories, then IDIP engine snapshots.
   - Produces per-dataset recovery audit sources (`runtime`, `history`, `idip_runtime`, `default`).
2. Wired KGD to recovery engine:
   - File: `ui/knowledge_growth_dashboard.py`
   - Dashboard now restores institutional metrics from persisted history if runtime snapshots are missing/invalid.
3. Startup recovery preflight:
   - File: `start_all.ps1`
   - Executes recovery engine before service startup and dashboard publication.
4. Atomic writes for critical institutional runtime snapshots:
   - `institutional_learning_scientist.py`
   - `knowledge_evolution_engine.py`
   - `institutional_decision_intelligence_platform.py`
   - `institutional_capital_intelligence.py`
   - `zeus_validation_operations.py`

## Runtime vs Institutional State
- Runtime state may reset: workers, loop counters, current task pointer.
- Institutional state must persist: learning history, validation history, knowledge graph, decision history, replay history, coverage/evolution metrics.

## Recovery Lifecycle
1. Load persisted artifacts.
2. Validate readability and non-empty payload.
3. Recover from append-only history when runtime snapshots are unavailable.
4. Recover from IDIP engine snapshots for subsystem payloads without dedicated history files.
5. Reconstruct Zeus summary from validation reports when status snapshot is missing.
6. Publish recovered state to dashboards before live event processing updates metrics.

## Validation Procedure
1. Generate institutional data (trades, reports, IDIP cycles).
2. Record baseline KGD metrics.
3. Stop services.
4. Start services via `start_all.ps1`.
5. Verify KGD metrics restore immediately from recovered state.
6. Verify Zeus metrics restore from status/history.
7. Verify Hermes/Prometheus continue appending new data after recovery.

## Operational Notes
- Disk pressure can prevent writing optional recovery snapshots; recovery still runs and reports `snapshot_write_error` in audit output.
- This implementation is additive and does not mutate historical append-only logs.
