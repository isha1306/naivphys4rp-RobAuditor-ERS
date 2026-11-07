"""Rank candidate recovery plans using confidence, risk, and cost."""

from dataclasses import dataclass
from typing import List, Sequence

from .models import RecoveryPlan, PlanStatus


@dataclass
class RankingWeights:
    """Weights used to calculate recovery plan scores."""

    confidence: float = 0.5
    risk: float = 0.3
    cost: float = 0.2

    def __post_init__(self):
        if min(self.confidence, self.risk, self.cost) < 0:
            raise ValueError("Ranking weights cannot be negative.")

        total = self.confidence + self.risk + self.cost

        if total <= 0:
            raise ValueError("At least one ranking weight must be positive.")

        # Normalize weights to sum to 1.
        self.confidence /= total
        self.risk /= total
        self.cost /= total


def calculate_plan_score(
    plan: RecoveryPlan,
    weights: RankingWeights = None,
) -> float:
    """Calculate a score for a candidate recovery plan."""

    if weights is None:
        weights = RankingWeights()

    metadata = plan.metadata or {}

    confidence = float(metadata.get("confidence_score", 0.0))
    risk = float(metadata.get("risk_score", 1.0))

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Confidence score must be between 0 and 1.")

    if not 0.0 <= risk <= 1.0:
        raise ValueError("Risk score must be between 0 and 1.")

    # More actions generally mean greater execution cost.
    action_count = len(plan.actions or [])
    cost = min(action_count / 5.0, 1.0)

    score = (
        weights.confidence * confidence
        - weights.risk * risk
        - weights.cost * cost
    )

    return score


def rank_plans(
    plans: Sequence[RecoveryPlan],
    weights: RankingWeights = None,
) -> List[RecoveryPlan]:
    """Rank executable candidate plans from best to worst.

    Plans rejected by safety checks or requiring human intervention
    are excluded from executable ranking.
    """

    if weights is None:
        weights = RankingWeights()

    eligible_plans = []

    for plan in plans:
        if plan.status not in {
            PlanStatus.PROPOSED,
            PlanStatus.VALIDATED,
        }:
            continue

        if not plan.actions:
            continue

        plan.score = calculate_plan_score(plan, weights)
        eligible_plans.append(plan)

    return sorted(
        eligible_plans,
        key=lambda plan: plan.score,
        reverse=True,
    )