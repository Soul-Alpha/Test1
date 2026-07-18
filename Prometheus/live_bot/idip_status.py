from __future__ import annotations

from typing import Any


def update_idip_status(status: dict[str, Any], *, idip: dict[str, Any], idip_artifacts: dict[str, Any]) -> None:
    status["idip"] = idip
    status["idip_artifacts"] = idip_artifacts
    status["idip_summary"] = idip.get("summary", {})
    status["idip_engines"] = {
        "exit_intelligence": idip.get("engines", {}).get("exit_intelligence", {}),
        "duration_intelligence": idip.get("engines", {}).get("duration_intelligence", {}),
        "reward_capture_intelligence": idip.get("engines", {}).get("reward_capture_intelligence", {}),
        "institutional_risk_intelligence": idip.get("engines", {}).get("institutional_risk_intelligence", {}),
        "portfolio_intelligence": idip.get("engines", {}).get("portfolio_intelligence", {}),
        "decision_attribution_intelligence": idip.get("engines", {}).get("decision_attribution_intelligence", {}),
    }
    status["idip_recommendations"] = idip.get("zeus_research_recommendations", [])
    status["idip_self_improvement_loop"] = idip.get("self_improvement_loop", {})
    status["idip_meta_learning"] = idip.get("engines", {}).get("meta_learning_engine", {})
    status["idip_aro"] = idip.get("engines", {}).get("autonomous_research_orchestrator", {})
    status["idip_research_prioritization"] = idip.get("engines", {}).get("research_prioritization_engine", {})
    status["idip_research_director"] = idip.get("engines", {}).get("institutional_research_director", {})
    status["idip_explainability"] = idip.get("engines", {}).get("explainability_engine", {})
