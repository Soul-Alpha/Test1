# Olympus Engineering Directive - Phase 1 Review and Roadmap

## Governance Basis
- Authority: Olympus Engineering Constitution v1.0
- Operating mode: Additive only, backward compatible, feature flagged, versioned, observable, immutable where required
- Execution guardrail: No automatic Prometheus/Hermes execution-path mutation; Zeus validation remains mandatory for adoption

## Architectural Review Summary

### 1) Existing IDIP Implementation Review
- Findings:
  - `olympus/core/institutional_decision_intelligence_platform.py` (v1.0) is additive and observational.
  - IDIP already includes lifecycle, attribution, counterfactual, risk, portfolio, and knowledge outputs.
  - IDIP artifact persistence exists (`idip_runtime.json`, `idip_history.jsonl`, recommendation queue, knowledge base).
- Risks:
  - Institutional learning is embedded but not yet a standalone permanent subsystem with dedicated mandatory outputs.
  - Capital event classification and independent ledgers are not explicitly enforced.
  - Knowledge graph of full decision path is not yet explicit/versioned.
- Recommendation:
  - Keep IDIP as orchestrator and compose dedicated permanent engines.
- Approval: Approved with additive extension required.

### 2) Hermes Intelligence Review
- Findings:
  - Hermes status pipeline is active and already publishes TLI/IDIP payloads.
  - Feature flags exist for TLI and IDIP.
- Risks:
  - Missing fine-grained feature flags for newly required institutional subsystems.
- Recommendation:
  - Add granular `ENABLE_*` flags for each new subsystem and surface status observability.
- Approval: Approved with additive controls.

### 3) Prometheus Evolution Review
- Findings:
  - Persistent intelligence engines and institutional validation architecture already exist.
  - Prometheus evolution is observational and versioned.
- Risks:
  - Cross-subsystem knowledge growth loop is not fully unified as an explicit self-improvement chain artifact.
- Recommendation:
  - Link IDIP outputs to learning queue + Zeus-oriented evidence path using immutable records.
- Approval: Approved with integration extension.

### 4) Zeus Validation Review
- Findings:
  - Zeus validation contracts and lifecycle are contract-first and adoption-gated.
- Risks:
  - New subsystems need deterministic recommendation routing and evidence metadata.
- Recommendation:
  - Emit structured research candidates from replay/learning engines into Zeus queue artifacts.
- Approval: Approved.

### 5) Runtime Artifact Review
- Findings:
  - Existing artifacts are append-friendly and dashboard-observable.
- Risks:
  - Missing mandatory institutional learning artifacts and explicit capital intelligence artifacts.
- Recommendation:
  - Add mandatory files for institutional learning scientist and new engine artifacts.
- Approval: Approved.

## Remaining Architectural Gaps
1. Permanent standalone Institutional Learning Scientist subsystem with required outputs.
2. Capital Intelligence Engine with account-event classification and independent ledgers.
3. Versioned institutional Knowledge Graph for full decision path lineage.
4. Replay and counterfactual subsystem that replays every closed trade and emits Zeus evidence only.
5. Dedicated Knowledge Growth Dashboard (KGD) as central intelligence growth interface.
6. Explicit self-improvement loop contract artifact across all stages.

## Mandatory Specialist Team - Independent Recommendations

### Olympus Architect
- Findings: Current architecture supports additive composition through Olympus core modules.
- Risks: Scope creep can create coupled logic if embedded in one file.
- Recommendations: Create dedicated modules and compose from IDIP orchestrator.
- Required implementation: 4 new core modules + orchestrator wiring.
- Approval: Approve.

### Hermes Specialist
- Findings: Hermes status writer is stable and extensible.
- Risks: Runtime bloat if unbounded processing per poll.
- Recommendations: Single-pass bounded computation and history windows.
- Required implementation: Feature flags + bounded replay windows.
- Approval: Approve.

### Prometheus Specialist
- Findings: Prometheus evolution contracts already support institutional status integration.
- Risks: Mixed capital/trading events could pollute strategy analytics.
- Recommendations: Enforce strategy-vs-raw equity split and trading-only learning stats.
- Required implementation: Capital intelligence ledgers and default strategy-equity policy.
- Approval: Approve.

### Zeus Validation Specialist
- Findings: Validation lifecycle is established and operator-gated.
- Risks: Unstructured recommendations reduce auditability.
- Recommendations: Emit recommendation candidates with IDs, evidence score, lifecycle state.
- Required implementation: Zeus-ready recommendation records from learning/replay engines.
- Approval: Approve.

### Institutional Learning Scientist
- Findings: Existing learning intelligence is broad but not isolated as mandatory subsystem.
- Risks: Drift and decay can go undetected without dedicated metrics.
- Recommendations: Add concept/behaviour/model drift, knowledge growth, learning velocity, hypothesis queue.
- Required implementation: `institutional_learning_scientist.py` + 6 required artifacts.
- Approval: Approve.

### Capital Intelligence Engineer
- Findings: Capital analytics exist but not full account-event ledgering.
- Risks: Deposits/withdrawals contaminating drawdown and expectancy.
- Recommendations: Event typing and independent ledgers; strategy equity default.
- Required implementation: `institutional_capital_intelligence.py` + ledger artifacts.
- Approval: Approve.

### Knowledge Graph Engineer
- Findings: Pattern/trade data exists but full decision path graph missing.
- Risks: Entry-level memory only; weak institutional memory depth.
- Recommendations: Build path-level graph and versioned graph evolution.
- Required implementation: `institutional_knowledge_graph.py` + graph artifacts + path query.
- Approval: Approve.

### Decision Intelligence Engineer
- Findings: Attribution exists but no unified decision-path memory retrieval.
- Risks: Repeating weak decision paths due to shallow memory.
- Recommendations: Hash full path and map to outcomes/lessons.
- Required implementation: Decision path index in knowledge graph.
- Approval: Approve.

### Trade Lifecycle Intelligence Engineer
- Findings: TLI and IDIP cover lifecycle metrics well.
- Risks: Replay coverage not guaranteed for every closed trade.
- Recommendations: Enforce full closed-trade replay coverage each cycle.
- Required implementation: Replay module with per-trade record completeness.
- Approval: Approve.

### Pattern Intelligence Engineer
- Findings: Pattern context exists with maturity states.
- Risks: Pattern lifecycle not tied strongly to full path outcomes.
- Recommendations: Link pattern nodes to decision path graph edges and lesson nodes.
- Required implementation: Pattern-node graph linkage.
- Approval: Approve.

### Risk Engineer
- Findings: Risk metrics present in IDIP.
- Risks: Capital shocks from non-trading events triggering false protection modes.
- Recommendations: Trading-only risk control inputs + attribution fields.
- Required implementation: Drawdown/recovery attribution split by ledger.
- Approval: Approve.

### Institutional Portfolio Engineer
- Findings: Portfolio concentration metrics exist.
- Risks: Capital event noise can distort portfolio intelligence.
- Recommendations: Compute portfolio intelligence from trading ledger/strategy equity by default.
- Required implementation: Strategy-equity-first policy in capital engine outputs.
- Approval: Approve.

### Machine Learning Engineer
- Findings: Dataset/feature lineage exists.
- Risks: ML label contamination by non-trading events.
- Recommendations: Explicit ML eligibility flags on event classes.
- Required implementation: Capital event classifier includes `ml_eligible` and `strategy_stat_eligible`.
- Approval: Approve.

### Quantitative Research Engineer
- Findings: Counterfactual research exists but can be expanded and structured.
- Risks: Scenario uplift assumptions without confidence tagging.
- Recommendations: Deterministic replay scenarios with confidence metadata.
- Required implementation: replay/counterfactual engine with evidence confidence.
- Approval: Approve.

### Behavioural Intelligence Engineer
- Findings: Behaviour signals implicit in exits/timing.
- Risks: Drift in execution behaviour not explicitly tracked.
- Recommendations: Add behavioural drift index and deterioration flags.
- Required implementation: institutional learning drift artifacts.
- Approval: Approve.

### Performance Engineer
- Findings: Runtime performance telemetry exists.
- Risks: New modules increase poll-time latency.
- Recommendations: bounded windows, O(n) passes, lightweight persistence.
- Required implementation: strict bounded replay rows and history windows.
- Approval: Approve.

### Database Engineer
- Findings: JSON/JSONL artifact pattern is stable and immutable-friendly.
- Risks: Duplicate queue/knowledge records without key checks.
- Recommendations: Deduplicate by stable IDs for append-only artifacts.
- Required implementation: dedupe in writers for queue/knowledge/graph edges.
- Approval: Approve.

### QA Engineer
- Findings: Existing TLI and IDIP tests pass.
- Risks: New engines without tests could regress silently.
- Recommendations: Add unit tests for contracts, ledger separation, replay coverage, file outputs.
- Required implementation: new tests for subsystem outputs and IDIP orchestration.
- Approval: Approve.

### Documentation Engineer
- Findings: Prior architecture docs are strong.
- Risks: Operational ambiguity without directive-specific implementation notes.
- Recommendations: add completion doc and dashboard usage references.
- Required implementation: this roadmap + implementation summary updates.
- Approval: Approve.

## Conflict Resolution (Evidence-Based)
- Conflict A: Embed learning inside IDIP vs standalone subsystem.
  - Decision: Standalone subsystem composed by IDIP.
  - Evidence: Better modular testing, clear ownership, required artifact contract compliance.
- Conflict B: Use raw equity for all intelligence vs strategy equity default.
  - Decision: Strategy equity default; raw equity retained for accounting observability only.
  - Evidence: Prevents contamination from deposits/withdrawals and preserves true strategy signal.
- Conflict C: Rich replay breadth vs runtime overhead.
  - Decision: Replay all completed trades each cycle with bounded scenario set per trade and bounded output windows.
  - Evidence: Meets completeness + performance constraints.

## Final Implementation Roadmap (Approved Before Coding)

### Wave 1 - Core Subsystems
1. Add `institutional_learning_scientist.py` and required artifact outputs:
   - `institutional_learning.json`
   - `hypotheses.json`
   - `knowledge_growth.json`
   - `learning_velocity.json`
   - `research_queue.json`
   - `concept_drift.json`
2. Add `institutional_capital_intelligence.py`:
   - Event classifier supporting all mandatory event types
   - Trading/Capital/Operational/Learning ledgers
   - Raw Equity Curve + Strategy Equity Curve
   - Trading-only eligibility controls for learning/stats
3. Add `institutional_knowledge_graph.py`:
   - Full decision-path graph nodes/edges and path hash lookup
   - Versioned graph evolution artifacts
4. Add `decision_replay_counterfactual_intelligence.py`:
   - Replay each closed trade
   - Alternative decision scenario outcomes (evidence-only)
   - Zeus candidate recommendations (non-executing)

### Wave 2 - Orchestration and Governance
1. Extend IDIP orchestrator to compose the four subsystems.
2. Add granular feature flags and version metadata propagation.
3. Extend IDIP artifact writer to persist subsystem artifacts immutably.
4. Preserve strict Zeus governance gates for all recommendations.

### Wave 3 - Dashboard and Ops
1. Add Knowledge Growth Dashboard:
   - `ui/knowledge_growth_dashboard.py`
   - Dedicated metrics for Knowledge, Learning, Decision, Lifecycle, Capital, Data Quality, Institutional Progress.
2. Add startup/tunnel lifecycle support for the new dashboard port.
3. Keep all existing dashboards unchanged (additive only).

### Wave 4 - QA and Validation
1. Add focused tests for subsystem contracts and artifact persistence.
2. Validate compile + tests + runtime smoke.
3. Verify no Prometheus execution behavior changes.

## Director Approval
- Director verdict: Proceed to implementation exactly per roadmap above.
- Preconditions satisfied: review complete, specialist recommendations complete, conflicts resolved, roadmap approved.
