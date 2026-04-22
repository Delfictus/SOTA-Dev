-- PRISM-4D Feature Store schema (D1)
-- Per directive: Phase 1.2

-- Target metadata
CREATE TABLE IF NOT EXISTS targets (
    target TEXT PRIMARY KEY,
    pdb_id TEXT NOT NULL,
    chain TEXT NOT NULL,
    atom_count INTEGER,
    residue_count INTEGER,
    ligand_code TEXT,
    ligand_heavy_atoms INTEGER,
    run_date TEXT,
    engine_flags TEXT,
    spike_percentile INTEGER DEFAULT 95,
    engine_time_seconds REAL,
    n_sites_detected INTEGER,
    status TEXT DEFAULT 'completed'
);

CREATE INDEX IF NOT EXISTS idx_targets_pdb_id ON targets(pdb_id);
CREATE INDEX IF NOT EXISTS idx_targets_status ON targets(status);
CREATE INDEX IF NOT EXISTS idx_targets_spike_percentile ON targets(spike_percentile);

-- Per-site features (tokenized ranker + EGNN training)
CREATE TABLE IF NOT EXISTS site_features (
    target TEXT NOT NULL,
    site_name TEXT NOT NULL,
    spike_count INTEGER,
    n_streams INTEGER,
    persistence REAL,
    unsat_frac REAL,
    spread REAL,
    burial REAL,
    spike_density REAL,
    min_dist_to_ligand REAL,
    graded_score REAL,
    source TEXT DEFAULT 'parquet',
    PRIMARY KEY (target, site_name),
    FOREIGN KEY (target) REFERENCES targets(target)
);

CREATE INDEX IF NOT EXISTS idx_site_features_target ON site_features(target);

-- Ground truth (corrected DCC)
CREATE TABLE IF NOT EXISTS corrected_dcc (
    target TEXT PRIMARY KEY,
    centroid_dcc REAL,
    spike_dcc REAL,
    spike_site TEXT,
    n_parquet_sites INTEGER,
    dcc_grade TEXT,
    FOREIGN KEY (target) REFERENCES targets(target)
);

CREATE INDEX IF NOT EXISTS idx_corrected_dcc_grade ON corrected_dcc(dcc_grade);

-- Per-residue features (EGNN/student training)
CREATE TABLE IF NOT EXISTS residue_features (
    target TEXT NOT NULL,
    residue_id INTEGER NOT NULL,
    -- 25 structural features
    sasa REAL, secondary_structure INTEGER, phi REAL, psi REAL,
    b_factor REAL, depth REAL, half_sphere_exposure REAL,
    -- 26 NMA features (6 modes * 4 + 2 global)
    nma_mode1_displacement REAL, nma_mode1_dir_x REAL, nma_mode1_dir_y REAL, nma_mode1_dir_z REAL,
    nma_mode2_displacement REAL, nma_mode2_dir_x REAL, nma_mode2_dir_y REAL, nma_mode2_dir_z REAL,
    nma_mode3_displacement REAL, nma_mode3_dir_x REAL, nma_mode3_dir_y REAL, nma_mode3_dir_z REAL,
    nma_mode4_displacement REAL, nma_mode4_dir_x REAL, nma_mode4_dir_y REAL, nma_mode4_dir_z REAL,
    nma_mode5_displacement REAL, nma_mode5_dir_x REAL, nma_mode5_dir_y REAL, nma_mode5_dir_z REAL,
    nma_mode6_displacement REAL, nma_mode6_dir_x REAL, nma_mode6_dir_y REAL, nma_mode6_dir_z REAL,
    nma_global_mobility REAL,
    nma_global_anisotropy REAL,
    -- 5 perturbed NMA features
    nma_perturbed_amplitude_ratio REAL,
    nma_perturbed_alignment REAL,
    nma_perturbed_centroid_shift REAL,
    nma_perturbed_ligand_cosine REAL,
    nma_perturbed_variance_ratio REAL,
    -- Labels
    is_binding_residue INTEGER DEFAULT 0,
    binding_distance REAL,
    PRIMARY KEY (target, residue_id),
    FOREIGN KEY (target) REFERENCES targets(target)
);

CREATE INDEX IF NOT EXISTS idx_residue_features_target ON residue_features(target);
CREATE INDEX IF NOT EXISTS idx_residue_features_binding ON residue_features(is_binding_residue);

-- ESM-2 embeddings (1280-dim float32 as blob = 5120 bytes/row)
CREATE TABLE IF NOT EXISTS esm2_embeddings (
    target TEXT NOT NULL,
    residue_id INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    PRIMARY KEY (target, residue_id),
    FOREIGN KEY (target) REFERENCES targets(target)
);

-- Physics features (per-residue spike statistics, 216 dims)
-- Stored as separate table because it's computed from parquet data post-hoc
CREATE TABLE IF NOT EXISTS physics_features (
    target TEXT NOT NULL,
    residue_id INTEGER NOT NULL,
    -- Per-residue spike aggregates within 5A sphere
    spike_count_near INTEGER,
    mean_intensity_near REAL,
    std_intensity_near REAL,
    temporal_persistence REAL,
    source_diversity REAL,      -- Shannon entropy
    spatial_density REAL,
    cross_stream_consensus REAL,
    burst_frequency REAL,
    isi_mean REAL,              -- inter-spike interval mean
    isi_std REAL,
    isi_cv REAL,                -- coefficient of variation
    rate_warm_hold REAL,
    rate_production REAL,
    rate_cool REAL,
    PRIMARY KEY (target, residue_id),
    FOREIGN KEY (target) REFERENCES targets(target)
);
