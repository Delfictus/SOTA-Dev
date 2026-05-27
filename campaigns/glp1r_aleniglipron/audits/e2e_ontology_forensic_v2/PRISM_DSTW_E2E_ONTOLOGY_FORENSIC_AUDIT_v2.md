# PRISM-DSTW E2E Ontology Forensic Audit v2

## 1. Executive verdict

- global_status: E2E_BLOCKED
- generated_at_utc: 2026-05-27T03:52:06Z
- artifact_count: 4973
- symbol_count: 25053
- blocker_count: 12

## 2. Canonical spec discovery

```json
{
  "candidate_count": 0,
  "candidates": [],
  "canonical_status": "BLOCKED_NO_CANONICAL_SPEC",
  "formal_sections_detected": [],
  "git_status": null,
  "modified_time": null,
  "sha256": null,
  "spec_path": null
}
```

## 3. Canonical pipeline map

| chain_id | source | target | status |
| --- | --- | --- | --- |
| LINEAGE-01 | spike events | active sets | LINEAGE_PARTIAL |
| LINEAGE-02 | active sets | transition counts | LINEAGE_COMPLETE |
| LINEAGE-03 | transition counts | W_dir/W_spec | LINEAGE_PARTIAL |
| LINEAGE-04 | W_spec | C4 | LINEAGE_PARTIAL |
| LINEAGE-05 | W_spec | C1 chi | LINEAGE_PARTIAL |
| LINEAGE-06 | W_dir + chi | DTSG | LINEAGE_NON_CANONICAL |
| LINEAGE-07 | W_spec + basin mapping | C6 restricted operator | LINEAGE_PARTIAL |
| LINEAGE-08 | C6 operator | Dirichlet eigensystem | LINEAGE_NON_CANONICAL |
| LINEAGE-09 | Dirichlet eigensystem | C6 reward | LINEAGE_NON_CANONICAL |
| LINEAGE-10 | C6 reward | spectral reward manager | LINEAGE_NON_CANONICAL |
| LINEAGE-11 | captured graph tile | BSR update | LINEAGE_COMPLETE |
| LINEAGE-12 | BSR update | C6 reward solve/cache | LINEAGE_NON_CANONICAL |
| LINEAGE-13 | C6 reward | Log-SubTB | LINEAGE_NON_CANONICAL |
| LINEAGE-14 | trajectory | tile credit | LINEAGE_PARTIAL |
| LINEAGE-15 | tile credit | motif index | LINEAGE_BROKEN |
| LINEAGE-16 | chemical tile | molecule candidate | LINEAGE_COMPLETE |
| LINEAGE-17 | variant operator | variant durability | LINEAGE_COMPLETE |
| LINEAGE-18 | active acquisition | next generation/training batch | LINEAGE_COMPLETE |
| LINEAGE-19 | training run | report/release/manifest | LINEAGE_COMPLETE |


## 4. File artifact inventory

| domain | artifact_count |
| --- | --- |
| raw_observatory_data | 3100 |
| reports/dossiers/manifests | 430 |
| molecule_design_surface | 397 |
| tests/gates/subagent_reports | 371 |
| Log_SubTB_training | 242 |
| variant_durability | 128 |
| release_packages/hashes/tags | 86 |
| C6_dirichlet_reward | 46 |
| active_sets | 37 |
| BSR_operator_runtime | 26 |
| spectral_transfer_operator | 22 |
| markov_validity_C4 | 20 |
| captured_graph_tiles | 19 |
| active_learning_feedback | 12 |
| IP_secret_exposure | 12 |
| metastable_states_C1 | 11 |
| chemical_tile_registry | 5 |
| genealogical_receptor_panel | 5 |
| spectral_reward_manager | 2 |
| coarse_DTSG | 2 |


## 5. Code symbol index summary

| symbol_type | count |
| --- | --- |
| function | 17701 |
| constant | 2675 |
| struct | 2432 |
| module | 926 |
| class | 499 |
| CLI entrypoint | 406 |
| enum | 250 |
| dataclass | 148 |
| trait | 16 |


Risk flags:

| risk_flag | count |
| --- | --- |
| DUPLICATE_IMPLEMENTATION | 4832 |
| HIDDEN_HEURISTIC_LOGIC | 2385 |
| NONCANONICAL_REWARD | 699 |
| UNTRACKED_RUNTIME_PATH | 446 |
| PLACEHOLDER_STUB | 345 |
| NONCANONICAL_OPERATOR | 171 |
| CPU_FALLBACK_PATH | 118 |


## 6. Dependency graph summary

- dependency_edge_count: 527070
- ontology_node_count: 30285
- ontology_edge_count: 47483

## 7. Duplicate/conflict audit

| duplicate_group_id | classification | artifact_count | sha256_count |
| --- | --- | --- | --- |
| DUP-SHA-00001 | BACKUP | 16 | 1 |
| DUP-SHA-00002 | BACKUP | 2 | 1 |
| DUP-SHA-00003 | BACKUP | 2 | 1 |
| DUP-SHA-00004 | BACKUP | 2 | 1 |
| DUP-SHA-00005 | BACKUP | 2 | 1 |
| DUP-SHA-00006 | BACKUP | 2 | 1 |
| DUP-SHA-00007 | BACKUP | 8 | 1 |
| DUP-SHA-00008 | BACKUP | 8 | 1 |
| DUP-SHA-00009 | BACKUP | 2 | 1 |
| DUP-SHA-00010 | BACKUP | 2 | 1 |
| DUP-SHA-00011 | BACKUP | 2 | 1 |
| DUP-SHA-00012 | BACKUP | 2 | 1 |
| DUP-SHA-00013 | BACKUP | 3 | 1 |
| DUP-SHA-00014 | BACKUP | 2 | 1 |
| DUP-SHA-00015 | BACKUP | 2 | 1 |
| DUP-SHA-00016 | BACKUP | 2 | 1 |
| DUP-SHA-00017 | BACKUP | 2 | 1 |
| DUP-SHA-00018 | BACKUP | 2 | 1 |
| DUP-SHA-00019 | BACKUP | 2 | 1 |
| DUP-SHA-00020 | BACKUP | 2 | 1 |
| DUP-SHA-00021 | BACKUP | 2 | 1 |
| DUP-SHA-00022 | BACKUP | 2 | 1 |
| DUP-SHA-00023 | BACKUP | 2 | 1 |
| DUP-SHA-00024 | BACKUP | 2 | 1 |
| DUP-SHA-00025 | BACKUP | 2 | 1 |
| DUP-SHA-00026 | BACKUP | 2 | 1 |
| DUP-SHA-00027 | BACKUP | 2 | 1 |
| DUP-SHA-00028 | BACKUP | 2 | 1 |
| DUP-SHA-00029 | BACKUP | 2 | 1 |
| DUP-SHA-00030 | BACKUP | 2 | 1 |

Truncated to first 30 rows.


## 8. Formal spec conformance

| formal_section | status | symbol_evidence_count |
| --- | --- | --- |
| Section I non-overclaim boundary | BLOCKED_BY_DATA | 18 |
| Section II provenance tiers | BLOCKED_BY_DATA | 237 |
| Axiom 8.5 estimator | BLOCKED_BY_DATA | 2 |
| C0 transfer operator | BLOCKED_BY_DATA | 57 |
| C1 metastable state extraction | BLOCKED_BY_DATA | 263 |
| C2 chronology/eigenvalue decay | BLOCKED_BY_DATA | 77 |
| C3 bisimulation/lumpability | BLOCKED_BY_DATA | 0 |
| C4 timescale convergence | BLOCKED_BY_DATA | 30 |
| C5 calibration/memory kernel | BLOCKED_BY_DATA | 299 |
| C6 restricted Dirichlet survival | BLOCKED_BY_DATA | 221 |
| C6.7 Perron robustness | BLOCKED_BY_DATA | 11 |
| retained federated operators | BLOCKED_BY_DATA | 2 |
| identity caveat | BLOCKED_BY_DATA | 41 |
| open physical blockers | BLOCKED_BY_DATA | 12 |


## 9. Provenance tier audit

| assigned_tier | count |
| --- | --- |
| UNKNOWN_TIER | 3441 |
| L3_DERIVED | 829 |
| L5_OBSERVED | 764 |


## 10. Lineage chains

| chain_id | source | target | status |
| --- | --- | --- | --- |
| LINEAGE-01 | spike events | active sets | LINEAGE_PARTIAL |
| LINEAGE-02 | active sets | transition counts | LINEAGE_COMPLETE |
| LINEAGE-03 | transition counts | W_dir/W_spec | LINEAGE_PARTIAL |
| LINEAGE-04 | W_spec | C4 | LINEAGE_PARTIAL |
| LINEAGE-05 | W_spec | C1 chi | LINEAGE_PARTIAL |
| LINEAGE-06 | W_dir + chi | DTSG | LINEAGE_NON_CANONICAL |
| LINEAGE-07 | W_spec + basin mapping | C6 restricted operator | LINEAGE_PARTIAL |
| LINEAGE-08 | C6 operator | Dirichlet eigensystem | LINEAGE_NON_CANONICAL |
| LINEAGE-09 | Dirichlet eigensystem | C6 reward | LINEAGE_NON_CANONICAL |
| LINEAGE-10 | C6 reward | spectral reward manager | LINEAGE_NON_CANONICAL |
| LINEAGE-11 | captured graph tile | BSR update | LINEAGE_COMPLETE |
| LINEAGE-12 | BSR update | C6 reward solve/cache | LINEAGE_NON_CANONICAL |
| LINEAGE-13 | C6 reward | Log-SubTB | LINEAGE_NON_CANONICAL |
| LINEAGE-14 | trajectory | tile credit | LINEAGE_PARTIAL |
| LINEAGE-15 | tile credit | motif index | LINEAGE_BROKEN |
| LINEAGE-16 | chemical tile | molecule candidate | LINEAGE_COMPLETE |
| LINEAGE-17 | variant operator | variant durability | LINEAGE_COMPLETE |
| LINEAGE-18 | active acquisition | next generation/training batch | LINEAGE_COMPLETE |
| LINEAGE-19 | training run | report/release/manifest | LINEAGE_COMPLETE |


## 11. Active learning loop audit

- status: ACTIVE_LEARNING_SCORING_ONLY

| check | status |
| --- | --- |
| acquisition score exists | PASS |
| uncertainty signal exists | PASS |
| selected candidates logged | PASS |
| selected candidates modify next batch | PASS |
| next training/generation run consumes selected candidates | PASS |
| feedback manifest exists | PASS |
| rejection/negative evidence preserved | PASS |


## 12. Runtime consistency audit

| check | status | risk_flags |
| --- | --- | --- |
| required manifests exist | PASS | [] |
| hashes exist | PASS | [] |
| statuses valid | PASS | [] |
| no CPU fallback in production captured-tile run | PASS | [] |
| no uncaptured tile fallback | MISSING | [] |
| reward source canonical | PASS | [] |
| C6.7 verdict distribution present | MISSING | [] |
| tile credit exists | MISSING | [] |
| cache diagnostics present | PASS | [] |
| operator owner consistent | PASS | [] |
| exact command provenance exists | PASS | [] |


## 13. Captured graph tile audit

- status: CAPTURED_TILE_ARTIFACTS_PRESENT_NOT_CANONICAL

## 14. Log-SubTB audit

- status: LOG_SUBTB_ARTIFACTS_PRESENT_NOT_CANONICAL

## 15. Molecule design readiness

- status: BLOCKED_NO_VALIDITY_FILTERS

| check | status |
| --- | --- |
| chemical tile registry | PASS |
| tile_id -> SMILES/molecular graph | PASS |
| attachment atoms | PASS |
| valency rules | PASS |
| stereochemistry rules | MISSING |
| RDKit sanitization | PASS |
| canonical SMILES | PASS |
| duplicate detection | PASS |
| fragment provenance | PASS |
| PAINS/basic medchem filters | PASS |
| synthetic feasibility status | PASS |
| tile-to-operator-delta mapping | PASS |
| tile-to-C6 reward effect | PASS |
| emitted molecule candidates | PASS |


## 16. Variant/genealogical durability

- status: VARIANT_DURABILITY_PARTIAL

| check | status |
| --- | --- |
| variant panel | PASS |
| genealogical grouping | PASS |
| topology-region grouping | PASS |
| perturbation-family grouping | PASS |
| variant-specific operator context | PASS |
| variant-specific C6 reward | PASS |
| WT-vs-variant comparison | PASS |
| variant uncertainty | PASS |
| acquisition uses variant uncertainty | PASS |


## 17. Claim-provenance audit

| claim_risk_flag | count |
| --- | --- |
| VARIANT_DURABILITY_WITHOUT_VARIANT_OPERATOR_CONTEXT | 505 |
| T1_WITHOUT_L5_EDGES | 159 |
| MOLECULE_DESIGN_WITHOUT_CHEMICAL_TILE_REGISTRY | 71 |
| CLINICAL_EFFECT_CLAIM | 61 |
| C6_WITHOUT_C4_PASS | 8 |


## 18. IP/secret exposure audit

| risk_flag | count |
| --- | --- |
| CREDENTIAL_EXPOSED | 17 |
| PATENT_RISK_DISCLOSURE | 12 |
| TRADE_SECRET_LEAK | 4 |
| CLIENT_SENSITIVE_DATA | 1 |


Secret/IP values are intentionally omitted; see redacted fingerprints in `ip_secret_exposure_audit.*`.

## 19. Test/gate audit

| gate | status |
| --- | --- |
| pytest | STALE |
| mypy | STALE |
| rust tests | STALE |
| clippy | STALE |
| regression gates | STALE |
| subagent reports | STALE |


## 20. Blocker index

| blocker_id | code | severity | notes |
| --- | --- | --- | --- |
| BLOCKER-0001 | BLOCKED_NO_CANONICAL_SPEC | CRITICAL | No exact DSTW v1 canonical spec file or content phrase found |
| BLOCKER-0002 | MISSING_W_HAT | CRITICAL | No W_hat/W_spec evidence found in requested scope |
| BLOCKER-0003 | C4_NOT_PROVEN | CRITICAL | C4 status=BLOCKED_BY_DATA |
| BLOCKER-0004 | C1_NOT_PROVEN | HIGH | C1 status=BLOCKED_BY_DATA |
| BLOCKER-0005 | C6_NOT_CANONICAL | CRITICAL | C6 status=BLOCKED_BY_DATA |
| BLOCKER-0006 | CLAIM_OVERREACH | HIGH | 748 claim risk flags |
| BLOCKER-0007 | PROVENANCE_BREAK | HIGH | 13 incomplete lineage chains |
| BLOCKER-0008 | SCHEMA_MISSING | MEDIUM | 177 schema gates missing required columns |
| BLOCKER-0009 | RUNTIME_NONCANONICAL | HIGH | 3 runtime checks failed or missing |
| BLOCKER-0010 | SECRET_EXPOSED | CRITICAL | 17 credential/secret findings |
| BLOCKER-0011 | IP_OVERDISCLOSURE | HIGH | 17 IP-sensitive findings |
| BLOCKER-0012 | ACTIVE_LEARNING_NO_FEEDBACK | MEDIUM | ACTIVE_LEARNING_SCORING_ONLY |


## 21. Recommended next actions

1. Recover or authoritatively place DSTW_FORMAL_SPECIFICATION_v1.md and rerun this audit.
2. Resolve MISSING_W_HAT/C4/C6 blockers before any operational DSTW claim.
3. Triage credential/IP findings by path and line using redacted fingerprints only.
4. Add chemical tile registry plus validity and tile-to-operator/C6 mappings before molecule-ready claims.
5. Add variant-specific operator and C6 contexts before variant durability operational claims.
