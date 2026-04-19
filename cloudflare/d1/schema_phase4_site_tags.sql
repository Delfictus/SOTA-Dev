-- PRISM-4D Feature Pipeline — Phase 4 schema migration
-- v4 feature-service hardening contract.
--
-- Applies to D1 database `prism-features`.
-- Binding:  schema_version = "v4.1"
--
-- This migration is IDEMPOTENT (IF NOT EXISTS / ADD COLUMN IF NOT EXISTS via
-- sqlite fallback).  D1 / SQLite lacks `ADD COLUMN IF NOT EXISTS`, so every
-- ALTER is wrapped in a defensive `ALTER ... ADD COLUMN`.  Running this file
-- twice must be safe; the second run returns `duplicate column name` which
-- the deploy script filters out.
--
-- Contracts this migration pins:
--   scripts/training/v4_feature_contract.yaml   (single source of truth)
--   docs/contracts/event_schema_v1.yaml         (event-level schema)
--   docs/contracts/site_tags_json_v1.schema.json (blob schema)
--   docs/contracts/persistence_contract.md      (BLOCKED state)
--
-- Writer ownership (W1=queue consumer, W2=/dcc, W3=/temporal, W3b=/event-aggregates,
-- W4=/runtime, W5=/persistence — reserved until BLOCKED lifts).
--
-- No INSERT OR REPLACE on site_features anywhere in this schema's usage.
-- All writers follow: INSERT OR IGNORE(pk) + UPDATE(column-scoped).

-- ──────────────────────────────────────────────────────────────
--  site_features — promoted scalar columns
-- ──────────────────────────────────────────────────────────────

-- From §2 Layer A (already-promoted scalars)
ALTER TABLE site_features ADD COLUMN persistence_source TEXT;
ALTER TABLE site_features ADD COLUMN volume REAL;
ALTER TABLE site_features ADD COLUMN engine_burial_score REAL;

-- Ranker priors
ALTER TABLE site_features ADD COLUMN druggability REAL;
ALTER TABLE site_features ADD COLUMN aromatic_score REAL;
ALTER TABLE site_features ADD COLUMN n_lining_residues INTEGER;
ALTER TABLE site_features ADD COLUMN quality_score REAL;
ALTER TABLE site_features ADD COLUMN rank_score REAL;
ALTER TABLE site_features ADD COLUMN engine_geo REAL;
ALTER TABLE site_features ADD COLUMN engine_chem REAL;
ALTER TABLE site_features ADD COLUMN engine_phys REAL;
ALTER TABLE site_features ADD COLUMN engine_vcs REAL;
ALTER TABLE site_features ADD COLUMN tokenized_score REAL;
ALTER TABLE site_features ADD COLUMN cryptic_score REAL;
ALTER TABLE site_features ADD COLUMN gtck_rank INTEGER;
ALTER TABLE site_features ADD COLUMN rank INTEGER;

-- Per-channel lexicographic rank SCORES (float, not integer; see §3 dtype correction)
ALTER TABLE site_features ADD COLUMN rank_C REAL;
ALTER TABLE site_features ADD COLUMN rank_G REAL;
ALTER TABLE site_features ADD COLUMN rank_K REAL;
ALTER TABLE site_features ADD COLUMN rank_L REAL;
ALTER TABLE site_features ADD COLUMN rank_T REAL;

-- Classification / filter scalars
ALTER TABLE site_features ADD COLUMN classification TEXT;
ALTER TABLE site_features ADD COLUMN therm_class TEXT;
ALTER TABLE site_features ADD COLUMN is_druggable INTEGER;     -- 0/1, bool encoded at W1
ALTER TABLE site_features ADD COLUMN is_cryptic INTEGER;       -- 0/1, derived at W1
ALTER TABLE site_features ADD COLUMN catalytic_residue_count INTEGER;

-- Thermodynamic scalars (non-decomposable)
ALTER TABLE site_features ADD COLUMN ccns_tau REAL;
ALTER TABLE site_features ADD COLUMN hysteresis_asymmetry REAL;
ALTER TABLE site_features ADD COLUMN relative_asymmetry REAL;
ALTER TABLE site_features ADD COLUMN onset_score REAL;
ALTER TABLE site_features ADD COLUMN breathing_score REAL;
ALTER TABLE site_features ADD COLUMN kinetic_accessibility REAL;
ALTER TABLE site_features ADD COLUMN effective_delta_g_kcal_mol REAL;

-- Thermodynamic decomposition (5 components)
ALTER TABLE site_features ADD COLUMN delta_g_aromatic_kcal_mol REAL;
ALTER TABLE site_features ADD COLUMN delta_g_cooperative_kcal_mol REAL;
ALTER TABLE site_features ADD COLUMN delta_g_dewetting_kcal_mol REAL;
ALTER TABLE site_features ADD COLUMN delta_g_electrostatic_kcal_mol REAL;
ALTER TABLE site_features ADD COLUMN delta_g_sti_kcal_mol REAL;

-- KCC / causality scalars
ALTER TABLE site_features ADD COLUMN kcc_active_causal_steps INTEGER;
ALTER TABLE site_features ADD COLUMN kcc_total_steps INTEGER;
ALTER TABLE site_features ADD COLUMN kcc_best_candidate_index INTEGER;
ALTER TABLE site_features ADD COLUMN kcc_driver_residue_id INTEGER;
ALTER TABLE site_features ADD COLUMN kcc_burst_motion REAL;
ALTER TABLE site_features ADD COLUMN kcc_direction_score REAL;
ALTER TABLE site_features ADD COLUMN kcc_confidence REAL;
ALTER TABLE site_features ADD COLUMN kcc_lag_corr_peak REAL;
ALTER TABLE site_features ADD COLUMN kcc_local_cov REAL;
ALTER TABLE site_features ADD COLUMN kcc_motion_efficiency REAL;
ALTER TABLE site_features ADD COLUMN kcc_temporal_corr REAL;
ALTER TABLE site_features ADD COLUMN kcc_site_burst_motion REAL;
ALTER TABLE site_features ADD COLUMN kcc_site_causal_lag REAL;
ALTER TABLE site_features ADD COLUMN kcc_site_direction_score REAL;
ALTER TABLE site_features ADD COLUMN kcc_site_lag_corr_peak REAL;
ALTER TABLE site_features ADD COLUMN kcc_site_local_cov REAL;
ALTER TABLE site_features ADD COLUMN kcc_site_motion_efficiency REAL;

-- Cold-phase-fraction subfields (flattened; parent was a dict)
ALTER TABLE site_features ADD COLUMN cold_phase_cold_fraction REAL;
ALTER TABLE site_features ADD COLUMN cold_phase_hot_fraction REAL;
ALTER TABLE site_features ADD COLUMN cold_phase_delta REAL;
ALTER TABLE site_features ADD COLUMN cold_phase_heating_spike_count INTEGER;  -- int64 per contract
ALTER TABLE site_features ADD COLUMN cold_phase_heating_spike_rate REAL;
ALTER TABLE site_features ADD COLUMN cold_phase_cooling_spike_count INTEGER;  -- int64 per contract
ALTER TABLE site_features ADD COLUMN cold_phase_cooling_spike_rate REAL;

-- Signal-preservation subfields (flattened; parent was a dict; NO derived scalar)
ALTER TABLE site_features ADD COLUMN signal_preservation_causality_density REAL;
ALTER TABLE site_features ADD COLUMN signal_preservation_coupled_voxels INTEGER;
ALTER TABLE site_features ADD COLUMN signal_preservation_max_recurrence INTEGER;
ALTER TABLE site_features ADD COLUMN signal_preservation_mean_recurrence REAL;
ALTER TABLE site_features ADD COLUMN signal_preservation_n_voxels INTEGER;
ALTER TABLE site_features ADD COLUMN signal_preservation_primary_residue_count INTEGER;   -- int64
ALTER TABLE site_features ADD COLUMN signal_preservation_primary_residue_id INTEGER;
ALTER TABLE site_features ADD COLUMN signal_preservation_residue_concentration REAL;
ALTER TABLE site_features ADD COLUMN signal_preservation_total_coupling INTEGER;           -- int64
ALTER TABLE site_features ADD COLUMN signal_preservation_total_recurrence INTEGER;         -- int64

-- Promoted from blob in §6 re-review
ALTER TABLE site_features ADD COLUMN frustrated_solvent_score REAL;
ALTER TABLE site_features ADD COLUMN ray_escape_ratio REAL;
ALTER TABLE site_features ADD COLUMN sphericity REAL;
ALTER TABLE site_features ADD COLUMN localization_score_raw REAL;

-- KCC family scalars (non-subfield)
ALTER TABLE site_features ADD COLUMN tide_coupling_score REAL;
ALTER TABLE site_features ADD COLUMN source_diversity REAL;
ALTER TABLE site_features ADD COLUMN uv_enrichment_score REAL;
ALTER TABLE site_features ADD COLUMN wd_coherence REAL;

-- Bounded decorative blob (≤2KB, schema-validated at W1)
ALTER TABLE site_features ADD COLUMN site_tags_json TEXT;

-- Temporal ratios (W3 owned; v4 FEATURE_COLS members 13 and 14)
ALTER TABLE site_features ADD COLUMN phase_transition_ratio REAL;
ALTER TABLE site_features ADD COLUMN warm_hold_spike_fraction REAL;

-- DCC provenance (W2 owned)
ALTER TABLE site_features ADD COLUMN dcc_metric_source TEXT;

-- Provenance stamps
ALTER TABLE site_features ADD COLUMN source_version TEXT;
ALTER TABLE site_features ADD COLUMN created_at TEXT;
-- updated_at removed in v4.1 narrow patch: D1 SQLITE_ERROR 7500 "too many columns
-- on sqlite_altertab_site_features" — 100-column cap reached. created_at-only
-- semantics retained; no audit-table redesign this phase.

-- Indexes for filter-critical columns (S1 set)
CREATE INDEX IF NOT EXISTS idx_site_features_therm_class ON site_features(therm_class);
CREATE INDEX IF NOT EXISTS idx_site_features_is_druggable ON site_features(is_druggable);
CREATE INDEX IF NOT EXISTS idx_site_features_is_cryptic ON site_features(is_cryptic);
CREATE INDEX IF NOT EXISTS idx_site_features_classification ON site_features(classification);

-- ──────────────────────────────────────────────────────────────
--  site_lining_residues — normalized residue-level lining
-- ──────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS site_lining_residues (
    target TEXT NOT NULL,
    site_name TEXT NOT NULL,
    residue_id INTEGER NOT NULL,
    residue_name TEXT,
    chain TEXT,
    min_distance REAL,
    n_atoms INTEGER,
    is_catalytic INTEGER,                 -- 0/1
    spike_attribution_count INTEGER,      -- nullable; legacy format omits this
    PRIMARY KEY (target, site_name, residue_id),
    FOREIGN KEY (target, site_name) REFERENCES site_features(target, site_name)
);
CREATE INDEX IF NOT EXISTS idx_slr_target ON site_lining_residues(target);
CREATE INDEX IF NOT EXISTS idx_slr_catalytic ON site_lining_residues(is_catalytic);

-- ──────────────────────────────────────────────────────────────
--  site_kcc_candidates — normalized per-candidate KCC rows
-- ──────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS site_kcc_candidates (
    target TEXT NOT NULL,
    site_name TEXT NOT NULL,
    candidate_rank INTEGER NOT NULL,         -- 0,1,2,...
    candidate_residue_id INTEGER,
    candidate_causal_weight REAL,
    candidate_residue_support REAL,
    candidate_burst_motion REAL,
    candidate_causal_lag REAL,
    candidate_confidence REAL,
    candidate_direction_score REAL,
    candidate_local_cov REAL,
    PRIMARY KEY (target, site_name, candidate_rank),
    FOREIGN KEY (target, site_name) REFERENCES site_features(target, site_name)
);
CREATE INDEX IF NOT EXISTS idx_kcc_cand_target ON site_kcc_candidates(target);
CREATE INDEX IF NOT EXISTS idx_kcc_cand_driver ON site_kcc_candidates(candidate_residue_id);

-- ──────────────────────────────────────────────────────────────
--  site_event_aggregates — W3b owned; event-contract-pinned
-- ──────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS site_event_aggregates (
    target TEXT NOT NULL,
    site_name TEXT NOT NULL,
    event_contract_version TEXT NOT NULL,     -- e.g. "event_schema_v1"
    n_events INTEGER,
    -- ccns_phase enum counts (+ unknown bucket per §4 policy)
    count_phase_cold_hold INTEGER,
    count_phase_warm_hold INTEGER,
    count_phase_heating INTEGER,
    count_phase_cooling INTEGER,
    count_phase_cold_return INTEGER,
    count_phase_unknown INTEGER,
    -- spike_source enum counts (+ other bucket)
    count_source_uv INTEGER,
    count_source_lif INTEGER,
    count_source_efp INTEGER,
    count_source_other INTEGER,
    -- type enum counts (+ other bucket)
    count_type_bnz INTEGER,
    count_type_unk INTEGER,
    count_type_anion INTEGER,
    count_type_cation INTEGER,
    count_type_other INTEGER,
    -- numeric aggregates
    mean_intensity REAL,
    std_intensity REAL,
    mean_vibrational_energy REAL,
    mean_water_density REAL,
    mean_n_nearby_excited REAL,
    nonzero_wavelength_count INTEGER,
    aromatic_attribution_count INTEGER,
    source_entropy_nat REAL,
    -- derived site-level ratios (mirrored to site_features by W3a)
    phase_transition_ratio REAL,
    warm_hold_spike_fraction REAL,
    computed_at TEXT NOT NULL,
    PRIMARY KEY (target, site_name),
    FOREIGN KEY (target, site_name) REFERENCES site_features(target, site_name)
);
CREATE INDEX IF NOT EXISTS idx_sea_target ON site_event_aggregates(target);

-- Quarantine table for W3b rows that exceed unknown-enum thresholds.
CREATE TABLE IF NOT EXISTS quarantined_event_aggregates (
    target TEXT NOT NULL,
    site_name TEXT NOT NULL,
    event_contract_version TEXT NOT NULL,
    n_events INTEGER,
    count_phase_unknown INTEGER,
    count_source_other INTEGER,
    count_type_other INTEGER,
    quarantine_reason TEXT NOT NULL,
    quarantine_detail_json TEXT,
    computed_at TEXT NOT NULL,
    PRIMARY KEY (target, site_name),
    FOREIGN KEY (target, site_name) REFERENCES site_features(target, site_name)
);
CREATE INDEX IF NOT EXISTS idx_qea_target ON quarantined_event_aggregates(target);
CREATE INDEX IF NOT EXISTS idx_qea_reason ON quarantined_event_aggregates(quarantine_reason);

-- ──────────────────────────────────────────────────────────────
--  corrected_dcc — ground-truth metadata promotion
-- ──────────────────────────────────────────────────────────────

ALTER TABLE corrected_dcc ADD COLUMN ligand_centroid_x REAL;
ALTER TABLE corrected_dcc ADD COLUMN ligand_centroid_y REAL;
ALTER TABLE corrected_dcc ADD COLUMN ligand_centroid_z REAL;
ALTER TABLE corrected_dcc ADD COLUMN holo_source TEXT;
ALTER TABLE corrected_dcc ADD COLUMN is_pandda_fragment INTEGER;
ALTER TABLE corrected_dcc ADD COLUMN is_templated_complex INTEGER;
ALTER TABLE corrected_dcc ADD COLUMN nucleic_chains TEXT;
ALTER TABLE corrected_dcc ADD COLUMN skip_reason TEXT;
ALTER TABLE corrected_dcc ADD COLUMN valid_for_dcc_validation INTEGER;
ALTER TABLE corrected_dcc ADD COLUMN dcc_metric_used TEXT;

-- ──────────────────────────────────────────────────────────────
--  targets — run-level provenance
-- ──────────────────────────────────────────────────────────────

ALTER TABLE targets ADD COLUMN engine_n_streams INTEGER;
ALTER TABLE targets ADD COLUMN engine_commit TEXT;
ALTER TABLE targets ADD COLUMN engine_mode TEXT;
ALTER TABLE targets ADD COLUMN engine_simulation_time_sec REAL;
ALTER TABLE targets ADD COLUMN engine_total_steps_per_stream INTEGER;
ALTER TABLE targets ADD COLUMN lining_residue_cutoff_angstroms REAL;
ALTER TABLE targets ADD COLUMN feature_contract_version TEXT;
ALTER TABLE targets ADD COLUMN event_contract_version TEXT;
ALTER TABLE targets ADD COLUMN binding_sites_json_sha256 TEXT;
ALTER TABLE targets ADD COLUMN ground_truth_json_sha256 TEXT;
