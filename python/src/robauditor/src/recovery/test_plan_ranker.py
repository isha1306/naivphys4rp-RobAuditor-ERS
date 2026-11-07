from .models import (
    RecoveryAction,
    RecoveryPlan,
    RecoveryStrategy,
    PlanStatus,
)

from .plan_ranker import (
    RankingWeights,
    calculate_plan_score,
    rank_plans,
)


def make_plan(
    recovery_id,
    confidence,
    risk,
    action_count,
    status=PlanStatus.PROPOSED,
):
    return RecoveryPlan(
        recovery_id=recovery_id,
        failed_action_id="grasp_001",
        actions=[
            RecoveryAction(name=f"action_{i}")
            for i in range(action_count)
        ],
        strategies=[RecoveryStrategy.GOB],
        status=status,
        explanation="Test recovery plan",
        accepted_suggestions=["test_verifier"],
        rejected_suggestions={},
        safety_constraints=[],
        max_attempts=3,
        metadata={
            "confidence_score": confidence,
            "risk_score": risk,
        },
    )


def test_high_confidence_low_risk_plan_ranks_first():
    plan_a = make_plan("plan_a", 0.9, 0.1, 1)
    plan_b = make_plan("plan_b", 0.7, 0.4, 3)

    ranked = rank_plans([plan_b, plan_a])

    assert ranked[0].recovery_id == "plan_a"
    assert ranked[1].recovery_id == "plan_b"


def test_unsafe_plan_is_excluded():
    safe_plan = make_plan("safe", 0.8, 0.2, 2)

    unsafe_plan = make_plan(
        "unsafe",
        1.0,
        0.0,
        1,
        status=PlanStatus.HUMAN_INTERVENTION,
    )

    ranked = rank_plans([safe_plan, unsafe_plan])

    assert len(ranked) == 1
    assert ranked[0].recovery_id == "safe"


def test_empty_plan_is_excluded():
    empty_plan = make_plan("empty", 0.9, 0.1, 0)

    ranked = rank_plans([empty_plan])

    assert len(ranked) == 0


def test_custom_weights():
    plan_a = make_plan("plan_a", 0.9, 0.5, 1)
    plan_b = make_plan("plan_b", 0.7, 0.1, 1)

    weights = RankingWeights(
        confidence=0.1,
        risk=0.8,
        cost=0.1,
    )

    ranked = rank_plans(
        [plan_a, plan_b],
        weights,
    )

    # With risk weighted heavily, the lower-risk plan wins.
    assert ranked[0].recovery_id == "plan_b"


def test_score_is_stored_on_plan():
    plan = make_plan("plan_a", 0.8, 0.2, 2)

    score = calculate_plan_score(plan)

    assert isinstance(score, float)

    ranked = rank_plans([plan])

    assert plan.score == score


if __name__ == "__main__":
    test_high_confidence_low_risk_plan_ranks_first()
    test_unsafe_plan_is_excluded()
    test_empty_plan_is_excluded()
    test_custom_weights()
    test_score_is_stored_on_plan()

    print("All plan ranker tests passed!")