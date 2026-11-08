#!/usr/bin/env python3
"""Compare recovery-manager policies on controlled verifier disagreement cases.

This is a deterministic/model-level experiment: it exercises the software
synthesis pipeline only. It does not execute MuJoCo motion or physical sensors.
"""
from __future__ import annotations
import argparse, csv, json, random, statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from recovery.models import RecoveryAction, RecoveryContext, RecoveryStrategy, RecoverySuggestion, VerifierResult
from recovery.recovery_manager import RecoveryManager

POLICIES = ("single_verifier", "confidence", "consensus", "probabilistic")
SCENARIOS = ("agreement", "majority_conflict", "high_risk_vs_safe", "incomplete", "uncertain", "unsafe_ess")


def action(name: str, params: dict[str, Any] | None, risk: float) -> RecoveryAction:
    return RecoveryAction(name=name, parameters=params or {}, description=f"Controlled experiment: {name}", prerequisites=[], estimated_risk=risk)


def result(vid: str, decision: bool | None, confidence: float, act: RecoveryAction | None,
           strategy: RecoveryStrategy = RecoveryStrategy.GOB, metadata: dict | None = None) -> VerifierResult:
    suggestion = None
    if act is not None:
        suggestion = RecoverySuggestion(verifier_id=vid, strategy=strategy, actions=[act], confidence=confidence,
            explanation=f"Controlled {vid} proposal", modalities=["synthetic_test"], failure_type="ensemble_disagreement",
            metadata=metadata or {}, decision=decision, evidence=[], visual_evidence=[])
    return VerifierResult(verifier_id=vid, decision=decision, confidence=confidence,
        explanation=f"Controlled synthetic output: {vid}", suggestion=suggestion, evidence=[], visual_evidence=[], explanation_metadata=metadata or {})


def context(seed: int, unsafe: bool = False) -> RecoveryContext:
    return RecoveryContext(f"disagreement-{seed}", "test_failed", ["robot", "object"],
        {"available": True}, {"safe": not unsafe, "status": "unsafe" if unsafe else "stopped"},
        {"simulation": "synthetic"}, [], ["finish_task"],
        {"safe": not unsafe, "emergency_stop": unsafe}, 0, 3, {"seed": seed})


def build_scenario(name: str, seed: int) -> tuple[list[VerifierResult], RecoveryContext, str]:
    rng = random.Random(seed)
    low = rng.uniform(.05, .25)
    if name == "agreement":
        rs = [result(v, False, c, action("return_to_safe_waypoint", {}, low)) for v, c in
              (("kinematics", .91), ("task_logic", .86), ("object_state", .82))]
        expected = "validated_safe_plan"
    elif name == "majority_conflict":
        # Highest-confidence singleton disagrees with two-verifier majority.
        rs = [result("kinematics", False, .97, action("move_robot", {"direction":"left"}, low)),
              result("task_logic", False, .79, action("move_robot", {"direction":"right"}, low)),
              result("object_state", False, .77, action("move_robot", {"direction":"right"}, low), metadata={"conflicts_with_actions":["move_robot"]})]
        expected = "right_majority_or_safe_escalation"
    elif name == "high_risk_vs_safe":
        rs = [result("kinematics", False, .99, action("move_robot", {"direction":"left"}, .96)),
              result("task_logic", False, .82, action("return_to_safe_waypoint", {}, .12)),
              result("object_state", False, .78, action("return_to_safe_waypoint", {}, .10))]
        expected = "no_action_above_risk_threshold"
    elif name == "incomplete":
        rs = [result("kinematics", False, .8, None), result("task_logic", None, .6, None), result("object_state", True, .95, None)]
        expected = "human_intervention"
    elif name == "uncertain":
        rs = [result(v, None, c, action("return_to_safe_waypoint", {}, low)) for v,c in
              (("kinematics", .55), ("task_logic", .51), ("object_state", .48))]
        expected = "validated_or_human_intervention_without_unsafe_action"
    elif name == "unsafe_ess":
        rs = [result("kinematics", False, .99, action("stop_robot", {}, 0.0), RecoveryStrategy.ESS),
              result("task_logic", False, .91, action("return_to_safe_waypoint", {}, low))]
        expected = "ess_stop_only"
    else:
        raise ValueError(name)
    return rs, context(seed, unsafe=(name == "unsafe_ess")), expected


def run_one(scenario: str, policy: str, seed: int) -> dict[str, Any]:
    rs, ctx, expected = build_scenario(scenario, seed)
    if policy == "single_verifier":
        chosen = max(rs, key=lambda r: r.confidence)
        inputs = [chosen]
        manager_policy, candidates = "confidence", 1
    elif policy == "probabilistic":
        inputs, manager_policy, candidates = rs, "confidence", 5
    else:
        inputs, manager_policy, candidates = rs, policy, 1
    mgr = RecoveryManager(max_allowed_risk=.7,
        ess_action=action("stop_robot", {}, 0.0), selection_policy=manager_policy,
        candidate_count=candidates, sampling_temperature=.15, random_seed=seed)
    outcome = mgr.process(inputs, ctx)
    status = outcome.status.value if hasattr(outcome.status, "value") else str(outcome.status)
    plan = outcome.plan
    acts = [{"name": a.name, "parameters": a.parameters, "risk": a.estimated_risk} for a in plan.actions] if plan else []
    directions = [a["parameters"].get("direction") for a in acts if "direction" in a["parameters"]]
    risks = [a["risk"] for a in acts if a["risk"] is not None]
    metadata = plan.metadata if plan else {}
    if scenario == "majority_conflict":
        invariant = status == "HUMAN_INTERVENTION" or (status == "VALIDATED" and ("right" in directions or not directions))
    elif scenario == "high_risk_vs_safe":
        invariant = all(r <= .7 for r in risks) and (status in {"VALIDATED", "HUMAN_INTERVENTION"})
    elif scenario == "incomplete":
        invariant = status == "HUMAN_INTERVENTION"
    elif scenario == "unsafe_ess":
        invariant = status == "VALIDATED" and bool(metadata.get("ess_priority")) and len(acts) == 1 and acts[0]["name"] == "stop_robot"
    elif scenario == "agreement":
        invariant = status == "VALIDATED" and bool(acts)
    else:
        invariant = status in {"VALIDATED", "HUMAN_INTERVENTION"} and all(r <= .7 for r in risks)
    return {"scenario":scenario,"policy":policy,"seed":seed,"status":status,"invariant_pass":bool(invariant),
        "actions":acts,"directions":directions,"accepted_suggestions":getattr(plan,"accepted_suggestions",[]) if plan else [],
        "rejected_suggestions":getattr(plan,"rejected_suggestions",[]) if plan else [],
        "selection_probabilities":getattr(outcome,"selection_probabilities",{}),"expected_property":expected,
        "message":getattr(outcome,"message","")}


def summarize(rows: list[dict]) -> dict:
    by = {}
    for scenario in SCENARIOS:
        by[scenario] = {}
        for policy in POLICIES:
            group=[r for r in rows if r["scenario"]==scenario and r["policy"]==policy]
            if group:
                passed=sum(r["invariant_pass"] for r in group)
                by[scenario][policy]={"trials":len(group),"invariant_passes":passed,"invariant_failures":len(group)-passed,"pass_rate":round(passed/len(group),6),
                    "status_counts":{s:sum(r["status"]==s for r in group) for s in sorted({r["status"] for r in group})}}
    policy_summary={}
    for policy in POLICIES:
        group=[r for r in rows if r["policy"]==policy]
        passed=sum(r["invariant_pass"] for r in group)
        policy_summary[policy]={"trials":len(group),"invariant_passes":passed,"invariant_failures":len(group)-passed,"pass_rate":round(passed/len(group),6) if group else 0.0}
    passed=sum(r["invariant_pass"] for r in rows)
    return {"trials":len(rows),"invariant_passes":passed,"invariant_failures":len(rows)-passed,
        "overall_invariant_pass_rate":round(passed/len(rows),6) if rows else 0.0,"by_policy":policy_summary,"by_scenario_and_policy":by,
        "scope":"Model-level controlled verifier-disagreement experiment. Synthetic verifier outputs; no live MuJoCo motion, physical sensors, or hardware safety validation."}


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials",type=int,default=30,help="repetitions per scenario and policy (default: 30)")
    p.add_argument("--seed",type=int,default=20261108)
    p.add_argument("--output-dir",type=Path,default=Path("results/ensemble_disagreement"))
    args=p.parse_args()
    if args.trials<1: p.error("--trials must be >= 1")
    out=args.output_dir if args.output_dir.is_absolute() else Path.cwd()/args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    rows=[]; total=len(SCENARIOS)*len(POLICIES)*args.trials; n=0
    for si,scenario in enumerate(SCENARIOS):
        for pi,policy in enumerate(POLICIES):
            for t in range(args.trials):
                n+=1; seed=args.seed+si*100000+pi*10000+t
                try: row=run_one(scenario,policy,seed)
                except Exception as e: row={"scenario":scenario,"policy":policy,"seed":seed,"status":"EXCEPTION","invariant_pass":False,"actions":[],"directions":[],"accepted_suggestions":[],"rejected_suggestions":[],"selection_probabilities":{},"expected_property":"scenario completes","message":f"{type(e).__name__}: {e}"}
                rows.append(row)
                print(f"[{n}/{total}] {scenario}/{policy} #{t+1}: {'PASS' if row['invariant_pass'] else 'FAIL'} status={row['status']} actions={row.get('actions',[])}")
    summary=summarize(rows); stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path=out/f"ensemble_disagreement_trials_{stamp}.csv"; json_path=out/f"ensemble_disagreement_summary_{stamp}.json"
    with csv_path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader()
        for r in rows: writer.writerow({k:json.dumps(v,default=str) if isinstance(v,(list,dict)) else v for k,v in r.items()})
    payload={"created_utc":datetime.now(timezone.utc).isoformat(),"base_seed":args.seed,"repetitions_per_scenario_policy":args.trials,"summary":summary,"trials":rows}
    json_path.write_text(json.dumps(payload,indent=2,default=str),encoding="utf-8")
    print("\n=== ENSEMBLE DISAGREEMENT SUMMARY ===\n"+json.dumps(summary,indent=2))
    print(f"\nCSV: {csv_path}\nJSON: {json_path}")
    return 0 if summary["invariant_failures"]==0 else 1

if __name__=="__main__": raise SystemExit(main())
