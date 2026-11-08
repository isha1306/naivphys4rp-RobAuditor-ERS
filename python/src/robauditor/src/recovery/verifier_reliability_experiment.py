#!/usr/bin/env python3
"""Verifier reliability and weighted-consensus experiment.

Controlled model-level experiment. It compares five decision policies over
synthetic verifier outputs. It does not execute robot motion or validate
physical safety.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from recovery.models import (
    RecoveryAction, RecoveryContext, RecoveryStrategy, RecoverySuggestion,
    VerifierResult,
)
from recovery.recovery_manager import RecoveryManager

POLICIES = (
    "single_verifier",
    "confidence",
    "equal_consensus",
    "confidence_weighted_consensus",
    "reliability_weighted_probabilistic",
)
SCENARIOS = (
    "reliable_majority",
    "misleading_majority",
    "one_unreliable_verifier",
    "unequal_reliability",
    "verifier_abstention",
    "confidence_manipulation",
    "split_decision",
    "multiple_safe_alternatives",
)

# Reliability values are controlled experimental ground-truth priors, not
# learned estimates. The experiment tests whether supplying them helps.
RELIABILITY = {
    "reliable_majority": {"kinematics": .9, "task_logic": .85, "object_state": .8},
    "misleading_majority": {"kinematics": .9, "task_logic": .2, "object_state": .2},
    "one_unreliable_verifier": {"kinematics": .9, "task_logic": .9, "object_state": .15},
    "unequal_reliability": {"kinematics": .95, "task_logic": .65, "object_state": .3},
    "verifier_abstention": {"kinematics": .9, "task_logic": .8, "object_state": .7},
    "confidence_manipulation": {"kinematics": .95, "task_logic": .2, "object_state": .85},
    "split_decision": {"kinematics": .9, "task_logic": .85, "object_state": .8},
    "multiple_safe_alternatives": {"kinematics": .9, "task_logic": .85, "object_state": .8},
}


def make_action(direction: str, risk: float = .15) -> RecoveryAction:
    return RecoveryAction(
        name="move_robot", parameters={"direction": direction},
        description=f"Controlled synthetic recovery: {direction}",
        prerequisites=[], estimated_risk=risk,
    )


def make_result(verifier_id: str, confidence: float, direction: str | None,
                risk: float = .15, decision: bool | None = False) -> VerifierResult:
    suggestion = None
    if direction is not None:
        suggestion = RecoverySuggestion(
            verifier_id=verifier_id, strategy=RecoveryStrategy.GOB,
            actions=[make_action(direction, risk)], confidence=confidence,
            explanation=f"Synthetic recommendation from {verifier_id}",
            modalities=["synthetic_test"], failure_type="reliability_experiment",
            metadata={"synthetic": True}, decision=decision, evidence=[],
            visual_evidence=[],
        )
    return VerifierResult(
        verifier_id=verifier_id, decision=decision, confidence=confidence,
        explanation=f"Synthetic verifier output: {verifier_id}",
        suggestion=suggestion, evidence=[], visual_evidence=[],
        explanation_metadata={"synthetic": True},
    )


def make_context(seed: int) -> RecoveryContext:
    return RecoveryContext(
        f"reliability-{seed}", "ensemble_disagreement", ["robot", "object"],
        {"available": True}, {"safe": True, "status": "stopped"},
        {"simulation": "synthetic"}, [], ["finish_task"],
        {"safe": True}, 0, 3, {"seed": seed},
    )


def scenario_results(name: str, seed: int) -> tuple[list[VerifierResult], set[str], dict[str, float]]:
    rng = random.Random(seed)
    rel = RELIABILITY[name]
    # "right" is the ground-truth preferred action in all single-answer cases.
    if name == "reliable_majority":
        rows = [("kinematics", .91, "right"), ("task_logic", .86, "right"), ("object_state", .82, "left")]
        expected = {"right"}
    elif name == "misleading_majority":
        rows = [("kinematics", .90, "right"), ("task_logic", .95, "left"), ("object_state", .92, "left")]
        expected = {"right"}
    elif name == "one_unreliable_verifier":
        rows = [("kinematics", .88, "right"), ("task_logic", .84, "right"), ("object_state", .99, "left")]
        expected = {"right"}
    elif name == "unequal_reliability":
        rows = [("kinematics", .75, "right"), ("task_logic", .96, "left"), ("object_state", .98, "left")]
        expected = {"right"}
    elif name == "verifier_abstention":
        rows = [("kinematics", .88, "right"), ("task_logic", .60, None), ("object_state", .80, "right")]
        expected = {"right"}
    elif name == "confidence_manipulation":
        rows = [("kinematics", .72, "right"), ("task_logic", .999, "left"), ("object_state", .78, "right")]
        expected = {"right"}
    elif name == "split_decision":
        rows = [("kinematics", .89, "right"), ("task_logic", .87, "left"), ("object_state", .55, None)]
        expected = {"right", "left"}  # either recommendation is acceptable; escalation is also safe
    elif name == "multiple_safe_alternatives":
        # Both alternatives are explicitly safe. The third verifier abstains.
        rows = [("kinematics", .88, "right"), ("task_logic", .86, "left"), ("object_state", .55, None)]
        expected = {"right", "left"}
    else:
        raise ValueError(name)
    results = [make_result(v, c, d) for v, c, d in rows]
    return results, expected, rel


def signature(result: VerifierResult) -> str | None:
    if not result.suggestion:
        return None
    actions = result.suggestion.actions
    return json.dumps([
        {"name": a.name, "parameters": a.parameters or {}, "risk": a.estimated_risk}
        for a in actions
    ], sort_keys=True, separators=(",", ":"))


def filter_by_weight(results: list[VerifierResult], weights: dict[str, float],
                     confidence_weighted: bool = False) -> tuple[list[VerifierResult], dict[str, Any]]:
    """Retain recommendations with >50% of total available verifier weight."""
    total_weight = sum(weights.get(r.verifier_id, 0.0) for r in results)
    support: Counter[str] = Counter()
    for r in results:
        sig = signature(r)
        if sig is None:
            continue
        weight = weights.get(r.verifier_id, 0.0)
        if confidence_weighted:
            weight *= max(0.0, min(1.0, float(r.confidence)))
        support[sig] += weight
    denominator = sum(
        weights.get(r.verifier_id, 0.0) *
        (max(0.0, min(1.0, float(r.confidence))) if confidence_weighted else 1.0)
        for r in results
    )
    eligible = {sig for sig, value in support.items() if value > denominator / 2.0}
    filtered = copy.deepcopy(results)
    retained = []
    for r in filtered:
        sig = signature(r)
        if sig is not None and sig not in eligible:
            r.suggestion = None
        elif sig is not None:
            retained.append(r.verifier_id)
    return filtered, {
        "weighting": "reliability_times_confidence" if confidence_weighted else "reliability_only",
        "total_weight": round(denominator, 6),
        "support": {k: round(v, 6) for k, v in support.items()},
        "eligible_signatures": sorted(eligible),
        "retained_verifiers": retained,
        "no_weighted_majority": not bool(eligible),
    }


def evaluate(name: str, policy: str, seed: int) -> dict[str, Any]:
    results, expected, reliability = scenario_results(name, seed)
    context = make_context(seed)
    meta: dict[str, Any] = {}
    if policy == "single_verifier":
        inputs = [max(results, key=lambda r: r.confidence)]
        manager_policy, candidate_count = "confidence", 1
    elif policy == "confidence":
        inputs, manager_policy, candidate_count = results, "confidence", 1
    elif policy == "equal_consensus":
        # Equal vote: strict majority of all verifier slots.
        inputs, meta = filter_by_weight(results, {k: 1.0 for k in reliability})
        manager_policy, candidate_count = "confidence", 1
    elif policy == "confidence_weighted_consensus":
        inputs, meta = filter_by_weight(results, reliability, confidence_weighted=True)
        manager_policy, candidate_count = "confidence", 1
    else:
        # Proposed policy: reliability-weighted strict majority, then sample
        # only among candidates generated from retained recommendations.
        inputs, meta = filter_by_weight(results, reliability, confidence_weighted=False)
        manager_policy, candidate_count = "confidence", 5

    manager = RecoveryManager(
        max_allowed_risk=.7,
        ess_action=RecoveryAction(name="stop_robot", parameters={},
            description="Controlled ESS stop", prerequisites=[], estimated_risk=0.0),
        selection_policy=manager_policy, candidate_count=candidate_count,
        sampling_temperature=.15, random_seed=seed,
    )
    outcome = manager.process(inputs, context)
    status = outcome.status.value if hasattr(outcome.status, "value") else str(outcome.status)
    plan = outcome.plan
    actions = []
    if plan:
        actions = [{"name": a.name, "parameters": a.parameters or {},
                    "risk": a.estimated_risk} for a in plan.actions]
    directions = {a["parameters"].get("direction") for a in actions if "direction" in a["parameters"]}
    risks = [a["risk"] for a in actions if a["risk"] is not None]
    # Human intervention is treated as a safe abstention, not a correct plan.
    safe_status = status in {"VALIDATED", "HUMAN_INTERVENTION"}
    if name in {"split_decision", "multiple_safe_alternatives"}:
        correct = safe_status and (status == "HUMAN_INTERVENTION" or directions.issubset(expected))
    else:
        correct = status == "HUMAN_INTERVENTION" or (
            status == "VALIDATED" and directions == expected
        )
    safe = safe_status and all(r <= .7 for r in risks)
    return {
        "scenario": name, "policy": policy, "seed": seed, "status": status,
        "correct_or_safe_abstention": bool(correct), "safety_invariant": bool(safe),
        "actions": actions, "selected_directions": sorted(directions),
        "expected_directions": sorted(expected), "reliability_weights": reliability,
        "weighting_metadata": meta,
        "selection_probabilities": getattr(outcome, "selection_probabilities", {}),
        "message": getattr(outcome, "message", ""),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_policy = {}
    for policy in POLICIES:
        group = [r for r in rows if r["policy"] == policy]
        n = len(group)
        correct = sum(r["correct_or_safe_abstention"] for r in group)
        safe = sum(r["safety_invariant"] for r in group)
        by_policy[policy] = {
            "trials": n, "correct_or_safe_abstention": correct,
            "decision_rate": round(correct / n, 6) if n else 0.0,
            "safety_invariant_passes": safe,
            "safety_pass_rate": round(safe / n, 6) if n else 0.0,
            "human_intervention_count": sum(r["status"] == "HUMAN_INTERVENTION" for r in group),
        }
    by_scenario = {}
    for scenario in SCENARIOS:
        by_scenario[scenario] = {}
        for policy in POLICIES:
            group = [r for r in rows if r["scenario"] == scenario and r["policy"] == policy]
            n = len(group)
            by_scenario[scenario][policy] = {
                "trials": n,
                "correct_or_safe_abstention": sum(r["correct_or_safe_abstention"] for r in group),
                "safety_invariant_passes": sum(r["safety_invariant"] for r in group),
                "human_intervention_count": sum(r["status"] == "HUMAN_INTERVENTION" for r in group),
            }
    return {
        "trials": len(rows), "by_policy": by_policy, "by_scenario_and_policy": by_scenario,
        "scope": "Synthetic model-level verifier reliability experiment. Reliability weights are supplied ground-truth experimental inputs, not learned estimates. No live robot motion, physical sensing, or hardware safety validation.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=30,
                        help="repetitions per scenario and policy (default: 30)")
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/verifier_reliability_experiment"))
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be >= 1")
    out = args.output_dir if args.output_dir.is_absolute() else Path.cwd() / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    total = len(SCENARIOS) * len(POLICIES) * args.trials
    index = 0
    for si, scenario in enumerate(SCENARIOS):
        for pi, policy in enumerate(POLICIES):
            for trial in range(args.trials):
                index += 1
                seed = args.seed + si * 100000 + pi * 10000 + trial
                try:
                    row = evaluate(scenario, policy, seed)
                except Exception as exc:
                    row = {
                        "scenario": scenario, "policy": policy, "seed": seed,
                        "status": "EXCEPTION", "correct_or_safe_abstention": False,
                        "safety_invariant": False, "actions": [], "selected_directions": [],
                        "expected_directions": [], "reliability_weights": RELIABILITY[scenario],
                        "weighting_metadata": {}, "selection_probabilities": {},
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                rows.append(row)
                print(f"[{index}/{total}] {scenario}/{policy} #{trial+1}: "
                      f"{'PASS' if row['correct_or_safe_abstention'] and row['safety_invariant'] else 'FAIL'} "
                      f"status={row['status']} selected={row.get('selected_directions', [])}")
    summary = summarize(rows)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = out / f"verifier_reliability_trials_{stamp}.csv"
    json_path = out / f"verifier_reliability_summary_{stamp}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, default=str) if isinstance(value, (dict, list, set)) else value
                for key, value in row.items()
            })
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_seed": args.seed, "repetitions_per_scenario_policy": args.trials,
        "summary": summary, "trials": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("\n=== VERIFIER RELIABILITY EXPERIMENT SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nCSV: {csv_path}\nJSON: {json_path}")
    return 0 if all(r["safety_invariant"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
