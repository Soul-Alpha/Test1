# Olympus Institutional Decision Intelligence Platform (IDIP) v1.0

## Governing Authority
- Olympus Engineering Constitution v1.0
- Mode: additive only, backwards compatible, versioned, observable, modular, explainable
- Enforcement: no automatic execution behavior modifications

## Engineering Director Workflow Completion
1. Current Olympus architecture reviewed.
2. Hermes intelligence stack reviewed.
3. Prometheus evolution intelligence reviewed.
4. Zeus validation architecture reviewed.
5. Architectural self-improvement gaps identified.
6. Mandatory specialists activated.
7. Independent specialist analysis collected.
8. Conflicts resolved through evidence and governance constraints.
9. IDIP target architecture defined.
10. Phased implementation roadmap defined.
11. Implementation initiated after consensus.

## Architecture Gap Analysis
- Gap 1: Learning is still heavily trade-outcome centered rather than decision-sequence centered.
- Gap 2: Exit reasons remain partially ambiguous, weakening attribution quality.
- Gap 3: Lifecycle replay and counterfactual analysis are fragmented across modules.
- Gap 4: Institutional recommendation queue is not yet unified across lifecycle, risk, and portfolio evidence.
- Gap 5: Hermes-generated recommendations are not persisted in a dedicated immutable decision-knowledge pipeline.
- Gap 6: Portfolio-level concentration and allocation intelligence are underrepresented in Hermes outputs.

## Mandatory Specialist Review
### Olympus Architect
- Findings: Existing additive analytics pattern is robust and supports an IDIP orchestrator.
- Risks: Tight coupling to Hermes runtime could reduce reuse by Prometheus.
- Recommendations: Build independent IDIP core module with source-neutral contracts.
- Required Changes: New core module + artifact pipeline + optional runtime integration.
- Decision: Approval.

### Hermes Specialist
- Findings: Hermes status payload already includes sufficient lifecycle and trade context.
- Risks: Poll-loop overhead if expensive replay/counterfactual work is uncached.
- Recommendations: Keep computations bounded and incremental.
- Required Changes: Status integration boundary only; no trade execution mutation.
- Decision: Approval.

### Prometheus Specialist
- Findings: Prometheus evolution artifacts can be consumed as external evidence.
- Risks: Execution coupling could violate mission boundaries.
- Recommendations: Mark all IDIP outputs as advisory/research unless Zeus+governance approved.
- Required Changes: Governance flags and explicit no-auto-adopt policy in payload.
- Decision: Approval.

### Zeus Validation Specialist
- Findings: Zeus contract supports recommendation validation and lifecycle states.
- Risks: Recommendations without evidence scoring reduce validation quality.
- Recommendations: Emit evidence score and sample metadata for each recommendation.
- Required Changes: IDIP recommendation queue with evidence-rich payload.
- Decision: Approval.

### Machine Learning Engineer
- Findings: Existing pattern and confidence signals enable decision-attribution features.
- Risks: Overfitting if recommendations are generated from tiny samples.
- Recommendations: Attach reliability stage and sample confidence to proposals.
- Required Changes: Sample-aware confidence and maturity fields.
- Decision: Approval.

### Pattern Intelligence Engineer
- Findings: Pattern lifecycle sequence clustering is feasible from current fields.
- Risks: Missing transition granularity for open trades.
- Recommendations: Use inferred sequence now; evolve to event-native states later.
- Required Changes: Lifecycle sequence map + pattern lifecycle profiling.
- Decision: Approval.

### Decision Intelligence Engineer
- Findings: Decision attribution requires per-trade scoring + reason taxonomy.
- Risks: Attribution drift if unknown exits dominate.
- Recommendations: Add deterministic exit classifier and decision-impact scoring.
- Required Changes: Exit intelligence + attribution intelligence engine.
- Decision: Approval.

### Risk Engineer
- Findings: Adaptive risk intelligence should remain advisory under governance.
- Risks: Implicit policy automation can breach constitution.
- Recommendations: Keep risk optimization in recommendation mode only.
- Required Changes: risk budgeting/allocation outputs marked advisory.
- Decision: Approval.

### Institutional Portfolio Engineer
- Findings: Portfolio concentration and exposure metrics are missing in Hermes layer.
- Risks: Local trade optimization can degrade portfolio efficiency.
- Recommendations: Add session/regime/pattern concentration and capital contribution.
- Required Changes: Portfolio intelligence engine.
- Decision: Approval.

### Quantitative Research Engineer
- Findings: Counterfactual replay can produce hypothesis evidence without mutating history.
- Risks: Counterfactual assumptions may become opaque.
- Recommendations: Publish explicit assumptions and confidence per replay.
- Required Changes: Counterfactual intelligence engine + immutable research output.
- Decision: Approval.

### Performance Engineer
- Findings: Current architecture supports low-overhead additive writers.
- Risks: Unbounded JSON payload growth.
- Recommendations: bounded windows, append-only JSONL, baseline runtime profiling.
- Required Changes: runtime profiling and bounded output lists.
- Decision: Approval.

### Database Engineer
- Findings: JSON/JSONL append-only store matches existing institutional storage pattern.
- Risks: Large files over long horizon.
- Recommendations: Keep denormalized runtime JSON + append-only history/knowledge.
- Required Changes: versioned artifact naming and deduplicated recommendation queue.
- Decision: Approval.

### QA Engineer
- Findings: Contract and persistence tests are required before rollout.
- Risks: Silent schema drift.
- Recommendations: Add deterministic schema and artifact tests.
- Required Changes: targeted IDIP unit tests.
- Decision: Approval.

### Documentation Engineer
- Findings: Need explicit governance/approval pipeline documentation.
- Risks: Ambiguous ownership between Hermes, Zeus, Prometheus.
- Recommendations: Include system roles and no-auto-adopt rule in architecture docs.
- Required Changes: architecture proposal and roadmap docs.
- Decision: Approval.

## Conflict Resolution
- Conflict A: automatic adaptive execution vs governance-only adoption.
  - Resolution: IDIP emits recommendations only; adoption requires Zeus validation + operator approval.
- Conflict B: full-scale replay depth vs runtime overhead.
  - Resolution: bounded replay windows and summarized counterfactuals in runtime payload.
- Conflict C: immediate DB schema expansion vs low-risk additive integration.
  - Resolution: append-only JSON/JSONL artifacts first; DB projection deferred.

## IDIP Target Architecture
- Core orchestrator: `olympus/core/institutional_decision_intelligence_platform.py`
- Runtime source integration: Hermes status publishing boundary
- Immutable artifacts:
  - `storage/olympus/idip_runtime.json`
  - `storage/olympus/idip_history.jsonl`
  - `storage/olympus/idip_recommendation_queue.jsonl`
  - `storage/olympus/idip_knowledge_base.jsonl`

### Engines
- Trade Lifecycle Intelligence
- Exit Intelligence
- Duration Intelligence
- Reward Capture Intelligence
- Position Management Intelligence
- Institutional Risk Intelligence
- Portfolio Intelligence
- Decision Attribution Intelligence
- Counterfactual Intelligence
- Pattern Lifecycle Intelligence
- Institutional Knowledge Intelligence

## Phased Roadmap
### Phase 1: Foundation (now)
- Build IDIP orchestrator and versioned payload.
- Build immutable artifact writers.
- Integrate Hermes runtime publishing with feature flag.

### Phase 2: Governance and Validation Integration
- Emit Zeus-ready recommendation candidates with evidence score.
- Add recommendation queue quality controls and dedupe.

### Phase 3: Portfolio + Counterfactual Expansion
- Enhance counterfactual scenarios and confidence scoring.
- Add richer portfolio concentration and risk-budget analytics.

### Phase 4: Prometheus Consumption Readiness
- Add optional Prometheus read adapter for validated knowledge only.
- Keep adoption behind explicit governance approval flag.

### Phase 5: Institutional Operations
- Add archival, compaction, and quality monitoring jobs.
- Expand QA regression and performance benchmark suite.

## Success Criteria Mapping
- Unknown/Pending metrics are reduced through evidence growth, not fabrication.
- Historical data remains immutable.
- Governance boundary remains explicit and enforceable.
- Execution integrity is preserved.
