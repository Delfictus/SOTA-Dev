# TRACK B CHRONOLOGICAL CONTROL DOSSIER

Generated: 2026-05-26T20:42:03.249316+00:00

## Scope
Computational calibration only. This dossier makes no biological efficacy claim.

## Calibration Adequacy
Verdict: `CALIBRATION_MANIFOLD_ADEQUATE`

## Candidate Audit
Candidate count: `100`
Dot SMILES count: `0`

## Provenance Classes
- Signal-grid and runtime telemetry layers: L4_RUNTIME_TELEMETRY where emitted by executed runtime artifacts.
- NMA / thermodynamic continuity maps: L3_DERIVED.
- Hydration continuity: L0_MISSING/BLOCKED_WITH_HARD_EVIDENCE unless direct hydration artifact is supplied.
- Candidate generated state: PROJECTED/derived computational calibration, not observed biology.

## Falsification Experiments
- Re-run continuity oracle with missing maps and require fail-closed behavior.
- Perturb TE-Hub subpanel membership and require adequacy/coverage deltas.
- Recompute transition tensor from BOCPD and kinetic strain artifacts and compare row count/event types.

## Wet-Lab Validation Plan
- Prioritize computationally lock-positive, continuity-admissible candidates.
- Validate receptor chronology-control hypotheses with orthogonal kinetic assays.
- Treat species selectivity as L2 structural inference until experimentally tested.

## Production Deployment Runbook
- Instantiate runtime with `scripts/instantiate_track_b_runtime.py`.
- Validate runtime with `scripts/validate_track_b_runtime.py` before any cloud sync.
- Use cloud sync dry-run first; execute only with credentials and post-upload hash verification.

## Continuity Manifest
Maps: `['hydration', 'nma', 'thermodynamic']`
