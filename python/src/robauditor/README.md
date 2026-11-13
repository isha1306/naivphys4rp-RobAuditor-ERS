# RobAuditor Python Source

This directory contains the Python implementation of the RobAuditor components used for task execution verification, NEEM-based execution context, audit-trail generation, and (in this branch) ensemble-based failure-recovery planning.

The `robauditor-ensemble-recovery` branch adds a recovery synthesis and execution layer to the existing TraceBot/RobAuditor code. Its central goal is to combine recommendations from independent verifiers into a coherent, risk-screened recovery plan while retaining the verifier decisions, confidence values, explanations, and evidence that led to that plan.

> **Scope and safety:** The recovery planning core is simulator-independent. The included integration and regression examples target MuJoCo simulation. The real-robot backend is intentionally fail-closed and sends no hardware commands. A plan being generated or validated does not, by itself, mean that it has been executed or that the original task has succeeded.

## Contents

```text
src/
├── canister_pick_and_place.py
├── functional_verification/
│   ├── ActionParser.py
│   ├── ModalityParser.py
│   ├── ModalityParserProxy.py
│   ├── ObjectParser.py
│   └── VerifierParser.py
├── neem_interface/
│   └── neem_interface.py
├── recovery/
│   ├── models.py
│   ├── suggestion_normalizer.py
│   ├── conflict_resolver.py
│   ├── risk_assessor.py
│   ├── plan_generator.py
│   ├── plan_ranker.py
│   ├── plan_validator.py
│   ├── recovery_manager.py
│   ├── execution_backend.py
│   ├── execution_coordinator.py
│   ├── mujoco_recovery_backend.py
│   ├── simulation_recovery_adapter.py
│   ├── failure_injector.py
│   ├── *experiment*.py
│   └── test_*.py
└── tracebot_audit_trail_generator/
    └── tracebot_audit_trail_generator.py
```

## 1. RobAuditor components already present in this source tree

### Functional verification

The `functional_verification` package contains parsers for the task entities and resources involved in verification:

- `ActionParser.py` parses action descriptions.
- `ObjectParser.py` parses object descriptions.
- `VerifierParser.py` parses verifier specifications.
- `ModalityParser.py` and `ModalityParserProxy.py` handle information modalities and their associated data sources.

These modules form part of the existing TraceBot/RobAuditor integration. They should not be confused with the new, simulator-independent recovery models described below.

### NEEM interface

`neem_interface/neem_interface.py` implements the `NeemInterface` used by the TraceBot integration. Its responsibilities include episode and skill recording, participant/role handling, event and action verification state, feasibility and success verification, explanations and confidence values, data-source access, and audit-trail generation.

The interface connects execution information and verification results to the episode-level representation used by RobAuditor.

### Audit-trail generation

`tracebot_audit_trail_generator/tracebot_audit_trail_generator.py` contains the PDF audit-trail formatter. It assembles episode-level information, action status, confidence and verification information into a human-readable report. The NEEM interface also exposes the audit-trail workflow.

The audit trail is intended to make the course of a process inspectable: what happened, which actions were involved, how verification went, and what explanations were recorded.

## 2. Ensemble-based failure recovery

The main work in this branch is under `recovery/`. It implements a pipeline that accepts results from multiple verifiers and synthesizes a recovery plan without coupling the planning logic to a particular simulator.

### Design goals

A robot action can fail in different ways, and different verifiers may recommend different responses. For example, a kinematics-focused verifier may recommend returning to a safe waypoint, while a task-logic verifier may recommend replanning an action. The recovery component is designed to:

1. preserve each verifier's decision, confidence, explanation and evidence;
2. normalize recovery suggestions into a shared representation;
3. identify duplicate, compatible and conflicting suggestions;
4. apply risk and safety constraints before selection;
5. combine compatible actions and choose between conflicting proposals;
6. generate and validate one or more candidate plans; and
7. keep plan synthesis separate from execution and postcondition verification.

### Data model

`models.py` defines the shared data structures:

| Model | Purpose |
|---|---|
| `VerifierResult` | One verifier's decision, confidence, explanation, recovery suggestion and evidence. |
| `RecoverySuggestion` | A verifier's proposed strategy and ordered recovery actions. |
| `RecoveryAction` | An action name, parameters, prerequisites, description and estimated risk. |
| `RecoveryContext` | Snapshot of the failed action, participants, robot/object/world/safety state and retry counters. |
| `RecoveryPlan` | Synthesized actions, strategies, status, accepted/rejected suggestions, constraints and metadata. |
| `EvidenceItem` | Typed evidence reference; payloads can remain external and be linked by URI. |

Verifier decisions may be `True`, `False` or `None`, represented as `PASS`, `FAIL` or `UNCERTAIN`. `VerifierResult.to_quadruplet()` serializes the RobAuditor-style output:

- **Ds** — verification decision;
- **Cf** — confidence;
- **Ed** — decision explanation and supporting evidence;
- **Er** — recovery strategy and proposed actions, when supplied.

Original verifier outputs are retained by the recovery manager, including PASS results and results that do not contain a recovery suggestion. This preserves provenance instead of reducing the ensemble to only the selected recommendation.

### Recovery strategies

The implementation uses the RobAuditor recovery vocabulary, with an additional explicit safe-state strategy:

| Code | Meaning |
|---|---|
| `ESS` | Enter a safe state (safety-priority path). |
| `HIC` | Human intervention call. |
| `GBB` | Go back before the failed action. |
| `GOB` | Go back to a waypoint/state or retry the failed action. |
| `INA` | Introduce a new action or replan. |
| `INO` | Introduce a new object, such as a replacement object. |

### Synthesis pipeline

The main entry point is `RecoveryManager.process(verifier_results, context)`.

1. **Preserve verifier outputs.** Every input result is serialized into its Ds/Cf/Ed/Er form for the result and plan metadata.
2. **Handle unsafe states first.** Explicit unsafe indicators (for example, `safe=False`, collision, emergency stop or human-intervention flags) route processing to the dedicated ESS path. If no configured safe-state action can be validated, the manager escalates to human intervention.
3. **Normalize suggestions.** `suggestion_normalizer.py` converts suggestions from FAIL and UNCERTAIN results into a common model. PASS results do not request recovery.
4. **Assess risk.** `risk_assessor.py` checks the attempt limit, robot/safety state, forbidden actions and action risk estimates. Unknown risk is not treated as zero risk. A suggestion above the configured risk threshold is rejected; a condition requiring human oversight is escalated.
5. **Resolve conflicts and synthesize.** `conflict_resolver.py` compares suggestions pairwise. It recognizes exact duplicate sequences, explicit verifier/action conflicts, and contradictory values for shared parameters of the same action. Strategy labels alone are not assumed to conflict. `plan_generator.py` retains compatible suggestions, excludes incompatible lower-priority proposals, and deduplicates identical action-plus-parameter combinations.
6. **Rank and validate.** `plan_ranker.py` scores candidates using confidence, risk and action-count cost. `plan_validator.py` checks that a plan is executable, non-empty, within retry limits, consistent with the current safety state, compliant with forbidden-action and risk constraints, and satisfies configured action prerequisites/parameter requirements and incompatibility rules.
7. **Select a validated candidate.** The manager returns a validated plan, or an escalation result when no acceptable candidate remains.

#### Selection policies and candidate sampling

`RecoveryManager` supports two selection policies:

- `confidence`: confidence-first prioritization after risk filtering (the default and backward-compatible path);
- `consensus`: prioritizes agreement on shared action signatures, then confidence, lower risk and fewer conflicts.

The manager can also generate a bounded set of alternatives by giving each verifier a priority turn (`candidate_count > 1`). Only candidates that pass validation are eligible for selection. When multiple validated candidates remain, the manager uses softmax sampling based on confidence, risk and verifier support. The sampling temperature controls how sharply probabilities favor higher-utility candidates; `random_seed` makes the selection repeatable.

This is a heuristic synthesis/selection mechanism, not a learned probabilistic model of verifier correctness. Verifier confidence is not treated as proof of correctness and cannot override hard safety checks.

### Key recovery modules

| File | Responsibility |
|---|---|
| `models.py` | Typed data models and verifier quadruplet serialization. |
| `suggestion_normalizer.py` | Converts verifier outputs into normalized suggestions while retaining evidence and provenance. |
| `conflict_resolver.py` | Pairwise duplicate, compatibility and conflict classification. |
| `risk_assessor.py` | Safety flags, forbidden-action checks, risk thresholds and human escalation. |
| `plan_generator.py` | Risk-filtered synthesis, compatible-action merging and bounded candidate generation. |
| `plan_generator_consensus.py` | Alternative/earlier consensus generator implementation retained in the package. |
| `plan_ranker.py` | Confidence/risk/action-count scoring and candidate ranking. |
| `plan_validator.py` | Preconditions, action parameters, incompatibilities, attempt limits and ESS-specific validation. |
| `recovery_manager.py` | End-to-end orchestration of normalization, risk checks, synthesis, validation and selection. |
| `execution_backend.py` | Backend-neutral interface, simulation callback backend and fail-closed real-robot placeholder. |
| `execution_coordinator.py` | Guarded execution, safety checks, postcondition checking and explicitly authorized retries. |
| `mujoco_recovery_backend.py` | Maps normalized recovery action names to caller-supplied MuJoCo recovery callbacks. |
| `simulation_recovery_adapter.py` | Builds synthetic or MuJoCo-state-based verifier results for the canister task. |
| `failure_injector.py` | Deterministic one-shot failure injection for simulation experiments. |

## 3. MuJoCo canister pick-and-place demonstration

`canister_pick_and_place.py` is the integration scenario. It contains the MuJoCo pick-and-place workflow and recovery functions for actions such as returning to a pre-grasp pose, returning to a safe transfer waypoint, replanning insertion, stopping the robot, and introducing a replacement canister.

The recovery adapter supports three verifier roles:

- **Kinematics verifier** — examines joint-state validity and measured joint speed.
- **Task-logic verifier** — checks whether the expected task-stage transition completed.
- **Object-state verifier** — checks canister availability, pose validity and, during grasp, gripper-pad contact.

When live MuJoCo model/data are supplied, the adapter derives role-specific results from a MuJoCo state snapshot. Otherwise, it can produce deterministic synthetic results for exercising the recovery pipeline. The synthetic fixture is explicitly not a physical sensor-backed verifier.

The simulation includes failure-injection points for:

`perception`, `pre_grasp`, `grasp`, `lift`, `transfer`, `insertion`, `placement`, and `object_unavailable`.

## 4. Planning versus execution

Planning and execution are separate on purpose.

### Execution backend

`execution_backend.py` defines the abstract `RecoveryExecutionBackend` and a `SimulationRecoveryBackend` that dispatches actions to explicitly registered callbacks. A plan must be marked `VALIDATED`, contain actions and have a recovery identifier before it can be dispatched.

`RealRobotRecoveryBackend` is a fail-closed placeholder. It returns that real-robot execution is not configured; it does not issue ROS or hardware commands.

### MuJoCo adapter

`mujoco_recovery_backend.py` binds the generic action interface to recovery functions supplied by the simulation integration. The currently recognized action names are:

- `return_to_pre_grasp`
- `replan_pre_grasp`
- `return_to_last_safe_waypoint`
- `return_to_placement_waypoint`
- `replan_insertion`
- `stop_robot`
- `introduce_replacement_canister`

The callbacks are only registered when their required runtime state/arguments are available. Unsupported actions are refused by the backend rather than silently ignored.

### Execution coordinator

`RecoveryExecutionCoordinator` requires a validated plan and can run a safety checker before execution. A supplied postcondition checker must confirm success before the coordinator reports verified recovery. If execution partially completes and then fails, the coordinator does not replay the sequence because the robot state may already have changed. A retry is considered only when no action ran, the retry limit allows it, and a caller-provided retry authorizer explicitly approves it.

## 5. Experiments and evaluation scripts

The recovery package includes unit tests, controlled planning experiments, stress tests and a headless MuJoCo regression runner.

| Script | Purpose |
|---|---|
| `ensemble_synthesis_experiment.py` | Deterministic comparison of ensemble synthesis behavior using synthetic verifier results. |
| `ensemble_planning_experiment.py` | Scenario-based comparison of a single-verifier baseline, naive risk-screened merging and the full ensemble pipeline. |
| `consensus_policy_experiment.py` | Compares confidence-first and consensus-aware selection. |
| `ensemble_disagreement_experiment.py` | Repeated agreement, conflict, high-risk, incomplete, uncertain and unsafe-ESS scenarios. |
| `consensus_constrained_probabilistic_experiment.py` | Controlled comparison of confidence, consensus, probabilistic and strict-majority-constrained variants. |
| `ensemble_stress_evaluation.py`, `ensemble_stress_test_v3.py` | Randomized/adversarial model-level stress cases. |
| `recovery_evaluation_experiment.py`, `verifier_reliability_experiment.py` | Additional recovery and verifier-reliability experiments. |
| `automated_recovery_regression.py` | Launches baseline plus injected-failure MuJoCo runs in headless mode and writes logs with CSV/JSON summaries. |

The model-level experiments use synthetic verifier fixtures. They test synthesis invariants and algorithmic behavior; they do **not** establish physical safety, live-sensor verifier performance or real-robot reliability. The automated regression runner exercises the MuJoCo simulation scenario and records expected recovery action/postcondition markers.

### Running the automated MuJoCo regression

From the repository root:

```bash
PYTHONPATH=python/src/robauditor/src \
  python python/src/robauditor/src/recovery/automated_recovery_regression.py
```

By default, the runner starts a baseline and the configured injected-failure scenarios in separate headless simulator processes. It writes per-scenario logs and `summary.csv` / `summary.json` under:

```text
results/recovery_experiments/automated_regression/
```

Optional runner arguments:

- `--project-root PATH` — repository root (defaults to the current directory);
- `--output-dir PATH` — output directory;
- `--timeout SECONDS` — per-process timeout (default: 240);
- `--python EXECUTABLE` — Python interpreter for child runs;
- `--skip-baseline` — omit the no-failure baseline run.

### Running model-level ensemble stress evaluation

From the repository root:

```bash
PYTHONPATH=python/src/robauditor/src \
  python python/src/robauditor/src/recovery/ensemble_stress_evaluation.py \
  --trials 20 --seed 20261108
```

The stress runner accepts `--cases` and `--output-dir` in addition to `--trials` and `--seed`. Its default cases cover randomized consensus, conflicting recommendations, high-risk recommendations, incomplete ensembles, uncertain decisions, attempt limits and unsafe states. It writes timestamped CSV and JSON results.

### Running the planning comparison

```bash
PYTHONPATH=python/src/robauditor/src \
  python python/src/robauditor/src/recovery/ensemble_planning_experiment.py \
  --repeats 100
```

This script compares three conditions (single-verifier, naive merge and the full ensemble manager) over its defined synthetic scenarios, and writes CSV/JSON summaries to `results/recovery_experiments/ensemble_planning_v2/` by default.

## 6. Running the canister simulation directly

The scenario script exposes `--mode sim|real` (default: `sim`), failure injection, selection policy and candidate-sampling options. A typical simulation invocation is:

```bash
PYTHONPATH=python/src/robauditor/src \
  python python/src/robauditor/src/canister_pick_and_place.py \
  --mode sim --no-viewer
```

To inject a deterministic failure:

```bash
PYTHONPATH=python/src/robauditor/src \
  python python/src/robauditor/src/canister_pick_and_place.py \
  --mode sim --no-viewer --inject-failure grasp
```

To use consensus selection and generate multiple candidate plans:

```bash
PYTHONPATH=python/src/robauditor/src \
  python python/src/robauditor/src/canister_pick_and_place.py \
  --mode sim --no-viewer \
  --recovery-selection-policy consensus \
  --recovery-candidate-count 5 \
  --recovery-sampling-temperature 0.15 \
  --recovery-random-seed 42
```

The script also exposes bounded initial-state perturbation parameters (`--canister-xy-perturb-m`, `--canister-yaw-perturb-deg`, `--joint-position-perturb-deg` and `--joint-velocity-perturb-rad-s`) for repeatable simulation trials. The script validates configured perturbation bounds before running.

Run `--help` on the script for the complete CLI argument list. MuJoCo model assets and runtime dependencies must be available in the repository/environment for the simulation commands to run.

## 7. Tests

The recovery package contains focused tests for:

- shared models and verifier quadruplet serialization;
- suggestion normalization and preservation of evidence;
- duplicate, compatible and conflicting suggestions;
- risk filtering, forbidden actions and safety-state handling;
- candidate generation, confidence/consensus policies and reproducible sampling;
- plan prerequisites, parameters, incompatibilities and attempt limits;
- ESS safe-state handling;
- execution preflight, safety checks, postcondition confirmation and retry authorization;
- MuJoCo callback registration and action dispatch.

Run the recovery tests from the repository root with pytest installed:

```bash
PYTHONPATH=python/src/robauditor/src \
  python -m pytest python/src/robauditor/src/recovery/test_*.py
```

Some integration tests and experiments require the simulator/runtime dependencies and assets in addition to the Python test dependencies.

## 8. Important implementation boundaries

- The recovery manager can synthesize and validate plans from provided verifier results; it does not create reliable verifier outputs from arbitrary real sensors on its own.
- The MuJoCo adapter includes a role-specific live-state path for the canister scenario, alongside a synthetic test-fixture path.
- The model-level experiments use synthetic data and should be interpreted as controlled software evaluations, not empirical robot-safety evidence.
- The real-robot backend is a placeholder and intentionally sends no hardware commands.
- A backend reporting that actions ran is not equivalent to task recovery. When a postcondition checker is supplied, the coordinator requires it to pass before reporting verified success.
- Full ontology/NEEM-driven recovery-context extraction is not implemented inside the new recovery package; the context is currently supplied through the `RecoveryContext` interface and the surrounding RobAuditor/NEEM integration.

## References

The design is aligned with the RobAuditor methodology for context-adaptive task-execution verification, failure recovery and audit-trail generation. In that framework, verifiers return a decision, confidence, explanation and optional recovery recommendation; multiple verifier outputs are combined in a meta-reasoning step. The recovery module in this branch develops this ensemble-recommendation and guarded-execution aspect for the canister simulation.

- Kenghagho Kenfack et al., **“RobAuditor — A Methodology for Scalable and Context-Adaptive Task Execution Verification in Safety-Critical Robotic Processes,”** IEEE, 2025.
- Kenghagho Kenfack, **“Execution Verification and Recovery in NaivPhys4RP: RobAuditor: A Safety-Centric & Flexible Wrapper,”** Chapter 10, 2025.
