# Directive 04 Bug Hunter Report

Commit audited: `15e13aad`
Branch: `debt-resolution-clean`
Blocking result: no CRITICAL/HIGH bugs found

Validation run by bug hunter:
- `PYTHONPATH=src python3 -m pytest tests/test_multistep_rollout.py tests/test_oracle_contract.py -q`: `29 passed`.
- D04 smoke completed with Rust subprocess telemetry, `trajectory_length_mean > 1`, `trajectories_ge2=8/8`, and increasing `assembled_state_node_count`.
- Working tree restored clean after generated training artifacts.

Findings:

BUG_1:
- severity: MEDIUM
- location: `scripts/train_gflownet_policy.py:1890`
- description: Product identity is RDKit-canonicalized from feature-derived atom numbers and Rust-returned bonds, but not yet from full reaction/synthon topology.
- reproduction: D04 smoke emits chemically weak long product identities despite real Rust coordinate/bond assembly.
- impact: Live scoring uses real product coordinates, but diversity/identity semantics remain approximate until D08 canonical SMILES lands.
- disposition: Recorded for D08, whose directive explicitly owns canonical RDKit SMILES persistence.

BUG_2:
- severity: LOW
- location: `scripts/train_gflownet_policy.py:1644`
- description: Kinematic assembly launches one subprocess per growth action and has no explicit timeout.
- reproduction: Large `batch_size` and `max_trajectory_steps` scales subprocess count with growth actions.
- impact: Performance and hang risk, not a correctness failure.
- disposition: Non-blocking optimization; batching can be addressed after correctness gates remain green.

BUG_3:
- severity: LOW
- location: `crates/prism-forge/src/bin/kinematic_assemble.rs:97`
- description: Rust validates empty/out-of-range request structure, but finite coordinate validation is enforced mainly on the Python caller side.
- reproduction: Direct CLI use with pathological finite-but-huge coordinates can reach Z-matrix assembly.
- impact: Trainer path is protected by Python finite checks, but standalone CLI is less defensive.
- disposition: Non-blocking hardening item; no trainer correctness failure.

Prior blockers rechecked:
- Real Rust subprocess assembly: PASS.
- Non-finite action coordinates blocked before graph tensors: PASS.
- Reordered duplicate live rewards rejected by `trajectory_id`: PASS.
- Product state from step `t` feeds step `t+1`: PASS.
- Rust `product_bonds` drive edge tensors: PASS.
- Missing binary preflight exists and builds `kinematic_assemble` if absent: PASS by code inspection.
