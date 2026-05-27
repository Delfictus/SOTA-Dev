# Blocker Index

row_count: 12

| blocker_id | code | severity | evidence | notes |
| --- | --- | --- | --- | --- |
| BLOCKER-0001 | BLOCKED_NO_CANONICAL_SPEC | CRITICAL | [] | No exact DSTW v1 canonical spec file or content phrase found |
| BLOCKER-0002 | MISSING_W_HAT | CRITICAL | [] | No W_hat/W_spec evidence found in requested scope |
| BLOCKER-0003 | C4_NOT_PROVEN | CRITICAL | ["campaigns/glp1r_aleniglipron/candidate_dossiers_population/cand_041_00fc448d.md", "campaigns/glp1r_aleniglipron/candidate_dossiers_population/cand_012_b0bd0c44.json", "campaigns/glp1r_aleniglipron/candidate_dossiers... | C4 status=BLOCKED_BY_DATA |
| BLOCKER-0004 | C1_NOT_PROVEN | HIGH | ["campaigns/glp1r_aleniglipron/phase_2c_metastable_atlas_triggers.json", "campaigns/glp1r_aleniglipron/candidate_dossiers_population/cand_034_6e97d1c1.json", "campaigns/glp1r_aleniglipron/candidate_dossiers_population... | C1 status=BLOCKED_BY_DATA |
| BLOCKER-0005 | C6_NOT_CANONICAL | CRITICAL | ["campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/bocpd_survival_regimes.propagation.jsonl", "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/bocpd_survival_regimes.parquet", "c... | C6 status=BLOCKED_BY_DATA |
| BLOCKER-0006 | CLAIM_OVERREACH | HIGH | ["campaigns/glp1r_aleniglipron/Phase3_PGx_Exclusion_Manifest.json:28", "campaigns/glp1r_aleniglipron/Phase3_PGx_Exclusion_Manifest.json:36", "campaigns/glp1r_aleniglipron/Phase3_PGx_Exclusion_Manifest.json:41", "campa... | 748 claim risk flags |
| BLOCKER-0007 | PROVENANCE_BREAK | HIGH | ["LINEAGE-01", "LINEAGE-03", "LINEAGE-04", "LINEAGE-05", "LINEAGE-06", "LINEAGE-07", "LINEAGE-08", "LINEAGE-09", "LINEAGE-10", "LINEAGE-12", "LINEAGE-13", "LINEAGE-14", "LINEAGE-15"] | 13 incomplete lineage chains |
| BLOCKER-0008 | SCHEMA_MISSING | MEDIUM | ["campaigns/glp1r_aleniglipron/phase_2d_variant_grid_manifest.json", "campaigns/glp1r_aleniglipron/phase_2d_variant_grid_manifest.propagation.jsonl", "campaigns/glp1r_aleniglipron/track_0_manual_emulation/pubchem_cid_... | 177 schema gates missing required columns |
| BLOCKER-0009 | RUNTIME_NONCANONICAL | HIGH | ["no uncaptured tile fallback", "C6.7 verdict distribution present", "tile credit exists"] | 3 runtime checks failed or missing |
| BLOCKER-0010 | SECRET_EXPOSED | CRITICAL | ["campaigns/glp1r_aleniglipron/visualizer_app/assets/index-CqvL8LbP.js:37", "scripts/vectorize_active_learning.py:167", "scripts/managed-agents/setup_agents.py:462", "scripts/managed-agents/setup_agents.py:468", "scri... | 17 credential/secret findings |
| BLOCKER-0011 | IP_OVERDISCLOSURE | HIGH | ["scripts/audit_e2e_ontology_pipeline.py:204", "scripts/audit_e2e_ontology_pipeline.py:205", "scripts/audit_e2e_ontology_pipeline.py:206", "scripts/quarantine/rebuild_patent_docx.py:22", "scripts/quarantine/rebuild_pa... | 17 IP-sensitive findings |
| BLOCKER-0012 | ACTIVE_LEARNING_NO_FEEDBACK | MEDIUM | [] | ACTIVE_LEARNING_SCORING_ONLY |
