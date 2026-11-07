# naivphys4rp

Perception as [inner realistic world construction](https://github.com/NaivPhys4RP/belief_state/) that [anticipates](https://github.com/NaivPhys4RP/forward_simulation_rendering/) and [explains](https://github.com/NaivPhys4RP/inverse_simulation/) the world state as well as observations in an explainable manner. NaivPhys4RP is a white-box and causal generative model of perception.

## Failure Recovery Plan Synthesis

This branch develops a recovery-plan synthesis component for RobAuditor. The task is to reconcile conflicting or incomplete recovery suggestions from independent verifiers (for example, kinematics, task logic, and safety) into one coherent plan using execution context and risk.

### Planned pipeline

1. Monitor robot actions and capture the failed action and execution context.
2. Collect verifier results and their recovery suggestions.
3. Normalize suggestions into a shared recovery-action model.
4. Detect duplicates, dependencies, compatible actions, and conflicts.
5. Apply hard safety constraints and assess risk.
6. Generate, score, and validate candidate recovery plans.
7. Execute an approved plan, verify its outcome, and re-synthesize within a bounded retry policy if needed.

The initial demonstration uses the existing MuJoCo canister pick-and-place task and three verifier roles: kinematics, task logic, and safety. The recovery engine will remain independent of MuJoCo so it can first be tested with synthetic verifier outputs.

### Recovery strategy vocabulary

The implementation starts with the RobAuditor strategy categories:

- **GOB** — Go back to the failed action.
- **GBB** — Go back a few steps before the failed action.
- **HIC** — Human intervention call.
- **INO** — Introduction of new objects.
- **INA** — Introduction of new actions.

### Implementation milestones

1. Define shared recovery data models.
2. Implement verifier interfaces and suggestion normalization.
3. Implement conflict/dependency analysis, safety filtering, candidate generation, and ranking.
4. Test agreement, disagreement, missing information, unsafe suggestions, and bounded retries.
5. Connect the recovery executor to the existing pick-and-place simulation.
6. Add execution feedback, then integrate ontology/NEEM context where useful.

**Safety principle:** verifier confidence scores must never override hard safety constraints. Planning, validation, and robot execution remain separate stages. A recovery action is not considered successful until the original task postcondition is verified.
