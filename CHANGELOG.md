# Changelog

## [Unreleased] — 2026-02-23

### Added

#### Section 1: Spike Thermodynamic Integration (STI)
- New module `crates/prism-nhs/src/spike_thermodynamic_integration.rs` (~800 lines)
- Five analysis layers:
  1. Jarzynski free energy per voxel (forward-process spikes: phases 1+2+3)
  2. Crooks Fluctuation Theorem intersection (requires `--hysteresis`)
  3. Bennett Acceptance Ratio (BAR) estimator
  4. Channel decomposition: UV/aromatic, LIF/dewetting, EFP/electrostatic + cooperative term
  5. Arrhenius barrier estimation per UV wavelength (280/274/258/254/211 nm)
- New struct `GpuSpikeEvent.ramp_phase: i32` (repurposed from `_padding`) — tags phase 1-5
- New methods on `CryoUvProtocol`: `ramp_phase_for_step()`, `tag_ramp_phases()`, `phase_boundaries()`
- New `StiConfig` struct for protocol-aware STI configuration
- Channel inference fallback `infer_spike_source()`: classifies spikes when GPU sets `spike_source=0`
- UV work formula: `E_photon × UV_QUANTUM_YIELD × relative_extinction` (produces ~2 kcal/mol at intensity=1)
- New JSON fields in `binding_sites.json` per site:
  - `delta_g_sti_kcal_mol`, `delta_g_aromatic_kcal_mol`, `delta_g_dewetting_kcal_mol`
  - `delta_g_electrostatic_kcal_mol`, `delta_g_cooperative_kcal_mol`
  - `delta_g_crooks_kcal_mol`, `delta_g_bar_kcal_mol`, `delta_g_branching_kcal_mol`
  - `activation_energy_by_wavelength`, `activation_energy_mean_kcal_mol`
  - `effective_delta_g_kcal_mol`, `kinetic_accessibility`

#### Section 2: Temporal-Synchrony Spike Encoding (N-2 + N-5)
- Added `first_spike_timestep: Option<u32>` and `threshold_adaptation: f32` to `DewettingNeuron`
- New types: `SynchronyEvent`, `SynchronyParams`, `ConfidenceClass` enum
- New functions in `neuromorphic.rs`: `detect_synchrony()`, `compute_ttfs_scores()`, `classify_confidence()`
- New CLI flags: `--single-stream-synchrony`, `--synchrony-k`, `--synchrony-radius`, `--synchrony-delta-t`, `--adaptation-rate`
- New JSON field per site: `confidence_class` (HIGH_CONFIDENCE/MODERATE_CONFIDENCE/LOW_CONFIDENCE/NOISE)

#### Section 3: Adaptive Threshold + Criticality Discrimination
- Added LZW complexity metric to `ccns_analyze.rs`: `compute_lzw_complexity()`
- New fields on `SiteCcnsResult`: `lzw_complexity: f32`, `lzw_classification: String`
  - `>0.6` → complex (genuine pocket), `<0.4` → repetitive (noise)
- Added `PerturbationResponse` struct and `analyze_perturbation_response()` to STI module
- CLI flag `--perturbation-test` (analysis functions ready; MD injection scaffolded)

#### Section 4: Spike Pharmacophore Synthesis
- New `PharmacophoreFeature` struct in STI module
- `generate_pharmacophore_features()`: groups spikes by (inferred_source, wavelength_bucket)
- `generate_pgmg_pharmacophore()`: PGMG-compatible JSON (`Aromatic`/`Hydrophobic`/`HBondDonor`/`HBondAcceptor`)
- `channel_to_feature_type()`: maps UV wavelength → pharmacophore type
- New JSON field per site: `pharmacophore_features` array (6 items typical, all 3 channels represented)

#### Section 5: Thermodynamic Resonance Scanning
- New module `crates/prism-nhs/src/resonance_scan.rs`
- `fit_lorentzian()`: grid search Lorentzian fit, returns (f0, Γ, A0, R²)
- `analyze_resonance_data()`: per-pocket resonance from amplitude sweep
- Structs: `VoxelResonance`, `PocketResonance`, `ResonanceMap`
- CLI flag `--resonance-scan` (protocol scaffolded; MD frequency sweep not yet injected)

### Changed
- `GpuSpikeEvent._padding` → `ramp_phase` (no size change, preserves GPU layout)
- Multi-stream pipeline (`run_multi_stream_pipeline`) now includes full STI + synchrony + pharmacophore integration
- Phase filter for Jarzynski: forward process = phases 1 (cold_hold) + 2 (heating) + 3 (warm_hold)
- UV extinction coefficients now used as relative weights, not absolute cross-sections

### Fixed
- Overlapping range endpoints in `spike_to_work` (`252..=256` → `252..=255`)
- `serde_json::json!` recursion limit when embedding synchrony/pharmacophore computation — extracted to variables
- Pre-existing test compile errors in `solvate.rs`, `rt_targets.rs`, `input.rs` (missing `residues` field)
- Ambiguous float type in `resonance_scan.rs` test

### Validated (canonical command: `--fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering`)

**1btl (TEM-1 β-lactamase), 13 sites:**
- All STI fields present and non-zero in multi-stream output
- `delta_g_aromatic` ~0.37, `delta_g_dewetting` ~2.7, `delta_g_electrostatic` ~1.3 kcal/mol
- `confidence_class: HIGH_CONFIDENCE` for all top sites
- `pharmacophore_features`: 6 items/site, all 3 channel types present

**2gl7 (β-catenin), 26 sites:**
- `delta_g_sti` in [0.49, 0.63] kcal/mol range across top sites
- All channel decomposition values non-zero
- LZW complexity: 0.475–0.559 (moderate; consistent with complex β-catenin PPI surface)

### Unit Tests
- 15/15 pass: `spike_thermodynamic_integration`, `resonance_scan`, `neuromorphic` modules
- 152 library tests pass (3 ignored, 1 pre-existing unrelated failure)

### Known Limitations
- `--resonance-scan` CLI flag parses but frequency sweep not yet injected into MD loop
- `--perturbation-test` CLI flag parses but perturbation not yet injected into MD loop
- `--single-stream-synchrony` reduces stream count flag parses but not yet wired to stream selection
- STDP threshold adaptation rule not yet applied in `DewettingNeuron::step()` (struct fields present)
- PGMG file output (`pharmacophore_<site_id>.json`) not yet written to disk (JSON value computed in memory)
