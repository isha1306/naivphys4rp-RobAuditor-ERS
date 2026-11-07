#!/usr/bin/env python3
"""Controlled comparison of deterministic and probabilistic recovery synthesis.

Runs the same synthetic three-verifier conflict through:
  1) single-plan mode (candidate_count=1)
  2) multi-candidate mode (candidate_count=5) over repeated fixed seeds.

This is a planning experiment only. It does not execute robot actions or use
live sensors. Candidate generation and probability scoring are those of the
current recovery framework.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from recovery.models import (
    RecoveryAction,
    RecoveryContext,
    RecoveryStrategy,
    RecoverySuggestion,
    VerifierResult,
)
from recovery.recovery_manager import RecoveryManager


OUTPUT_DIR = Path("results/recovery_experiments/probabilistic_synthesis")
SEEDS = list(range(30))
CANDIDATE_COUNT = 5
TEMPERATURE = 0.15


def build_inputs():
    """Return one fixed conflict case shared by every experimental run."""
    proposals = [
        {
            "verifier_id": "kinematics_verifier",
            "direction": "left",
            "confidence": 0.95,
            "risk": 0.20,
            "explanation": "Leftward motion is preferred by the kinematic check.",
        },
        {
            "verifier_id": "task_logic_verifier",
            "direction": "right",
            "confidence": 0.90,
            "risk": 0.20,
            "explanation": "Rightward motion preserves the task sequence.",
        },
        {
            "verifier_id": "object_state_verifier",
            "direction": "right",
            "confidence": 0.85,
            "risk": 0.10,
            "explanation": "Rightward motion avoids the estimated object-state hazard.",
        },
    ]

    verifier_results = []
    for item in proposals:
        action = RecoveryAction(
            name="move_robot",
            parameters={"direction": item["direction"]},
            description=f"Move robot {item['direction']}.",
            estimated_risk=item["risk"],
        )
        suggestion = RecoverySuggestion(
            verifier_id=item["verifier_id"],
            strategy=RecoveryStrategy.INA,
            actions=[action],
            confidence=item["confidence"],
            explanation=item["explanation"],
            failure_type="conflicting_motion_recommendations",
        )
        verifier_results.append(
            VerifierResult(
                verifier_id=item["verifier_id"],
                decision=False,
                confidence=item["confidence"],
                explanation=item["explanation"],
                suggestion=suggestion,
                evidence={"synthetic_case": True, "estimated_risk": item["risk"]},
            )
        )

    context = RecoveryContext(
        failed_action_id="probabilistic_comparison_001",
        failed_action="approach_canister",
        participants={"robot": "UR10", "object": "canister"},
        object_state="stationary",
        robot_state={"status": "ready", "safe": True},
        world_state={"scenario": "synthetic_left_right_conflict"},
        previous_actions=["perceive_canister"],
        remaining_task=["grasp_canister", "place_canister"],
        safety_state={"safe": True},
        attempt_count=0,
        max_attempts=2,
    )
    return verifier_results, context


def action_signature(plan) -> str:
    if plan is None:
        return "NO_PLAN"
    parts = []
    for action in getattr(plan, "actions", []) or []:
        params = json.dumps(
            getattr(action, "parameters", {}) or {},
            sort_keys=True,
            default=str,
        )
        parts.append(f"{action.name}:{params}")
    return " -> ".join(parts) if parts else "EMPTY_PLAN"


def enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def run_once(candidate_count: int, seed: int) -> Dict[str, Any]:
    verifier_results, context = build_inputs()
    manager = RecoveryManager(
        selection_policy="consensus",
        candidate_count=candidate_count,
        sampling_temperature=TEMPERATURE,
        random_seed=seed,
        max_allowed_risk=0.7,
    )
    result = manager.process(verifier_results, context)

    candidates = list(getattr(result, "candidate_plans", []) or [])
    if not candidates and getattr(result, "plan", None) is not None:
        candidates = [result.plan]

    probabilities = dict(getattr(result, "selection_probabilities", {}) or {})
    selected_plan = getattr(result, "plan", None)

    candidate_rows = []
    for candidate in candidates:
        candidate_rows.append(
            {
                "candidate_id": getattr(candidate, "recovery_id", ""),
                "signature": action_signature(candidate),
                "status": enum_value(getattr(candidate, "status", "")),
                "score": getattr(candidate, "score", None),
                "probability": probabilities.get(
                    getattr(candidate, "recovery_id", ""), None
                ),
            }
        )

    return {
        "mode": "single_plan" if candidate_count == 1 else "multi_candidate",
        "seed": seed,
        "status": enum_value(getattr(result, "status", "")),
        "selected_candidate_id": (
            getattr(selected_plan, "recovery_id", "") if selected_plan else ""
        ),
        "selected_signature": action_signature(selected_plan),
        "candidate_count": len(candidates),
        "validated_candidate_count": sum(
            row["status"] == "VALIDATED" for row in candidate_rows
        ),
        "candidate_rows": candidate_rows,
        "selection_probabilities": probabilities,
        "message": getattr(result, "message", ""),
        "rejected_reasons": list(getattr(result, "rejected_reasons", []) or []),
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # One deterministic baseline and 30 independently seeded multi-plan runs.
    runs: List[Dict[str, Any]] = [run_once(candidate_count=1, seed=0)]
    runs.extend(
        run_once(candidate_count=CANDIDATE_COUNT, seed=seed)
        for seed in SEEDS
    )

    single = runs[0]
    multi = runs[1:]
    selection_counts = Counter(row["selected_signature"] for row in multi)
    candidate_signatures = sorted(
        {
            candidate["signature"]
            for row in multi
            for candidate in row["candidate_rows"]
            if candidate["status"] == "VALIDATED"
        }
    )

    summary = {
        "experiment": "deterministic_vs_probabilistic_recovery_synthesis",
        "description": (
            "Same synthetic verifier conflict; compare one-plan mode with "
            "five-candidate probabilistic mode over fixed random seeds."
        ),
        "configuration": {
            "selection_policy": "consensus",
            "single_candidate_count": 1,
            "multi_candidate_limit": CANDIDATE_COUNT,
            "sampling_temperature": TEMPERATURE,
            "seeds": SEEDS,
            "verifiers": [
                "kinematics_verifier",
                "task_logic_verifier",
                "object_state_verifier",
            ],
        },
        "single_plan_run": single,
        "multi_candidate_runs": len(multi),
        "multi_candidate_all_validated": all(
            row["status"] == "VALIDATED" for row in multi
        ),
        "multi_candidate_min_validated": min(
            (row["validated_candidate_count"] for row in multi), default=0
        ),
        "multi_candidate_max_validated": max(
            (row["validated_candidate_count"] for row in multi), default=0
        ),
        "distinct_validated_plan_signatures": candidate_signatures,
        "selection_counts": dict(selection_counts),
        "selection_frequencies": {
            signature: count / len(multi)
            for signature, count in selection_counts.items()
        },
        "interpretation_note": (
            "Frequencies are empirical outcomes of this finite seeded sample, "
            "not calibrated real-world success probabilities. The inputs are "
            "synthetic and no robot actions are executed."
        ),
        "runs": runs,
    }

    json_path = OUTPUT_DIR / "probabilistic_synthesis_results.json"
    csv_path = OUTPUT_DIR / "probabilistic_synthesis_runs.csv"
    json_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mode",
                "seed",
                "status",
                "candidate_count",
                "validated_candidate_count",
                "selected_candidate_id",
                "selected_signature",
            ],
        )
        writer.writeheader()
        for row in runs:
            writer.writerow(
                {key: row.get(key) for key in writer.fieldnames}
            )

    print("[probabilistic synthesis] SUMMARY")
    print(f"  single-plan status: {single['status']}")
    print(f"  single-plan selection: {single['selected_signature']}")
    print(f"  multi-candidate runs: {len(multi)}")
    print(
        "  validated candidate range: "
        f"{summary['multi_candidate_min_validated']}–"
        f"{summary['multi_candidate_max_validated']}"
    )
    print(f"  distinct validated plans: {len(candidate_signatures)}")
    print("  empirical selection counts:")
    for signature, count in selection_counts.most_common():
        print(f"    {signature}: {count}/{len(multi)}")
    print(f"  JSON: {json_path.resolve()}")
    print(f"  CSV:  {csv_path.resolve()}")

    if single["status"] != "VALIDATED":
        print("[FAIL] deterministic baseline did not validate.")
        return 1
    if not multi or not all(row["status"] == "VALIDATED" for row in multi):
        print("[FAIL] at least one multi-candidate run did not validate.")
        return 1
    if not candidate_signatures:
        print("[FAIL] no validated candidate plans were recorded.")
        return 1
    if len(candidate_signatures) < 2:
        print(
            "[WARNING] Only one distinct validated plan was generated. "
            "The probabilistic path ran, but this input did not create "
            "meaningful alternative plans."
        )
        return 2

    print("[PASS] deterministic and multi-candidate synthesis completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
