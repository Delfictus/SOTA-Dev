# PRISM-4D Interface Contracts (WT-0)

Stable data contracts for inter-worktree communication.  These dataclasses
are **frozen after merge** — any change requires a hotfix branch and rebase
of all dependent worktrees.

## Module Index

| Module | Classes | Consumers |
|--------|---------|-----------|
| `spike_pharmacophore.py` | `PharmacophoreFeature`, `ExclusionSphere`, `SpikePharmacophore` | WT-1, WT-2, WT-3, WT-4 |
| `generated_molecule.py` | `GeneratedMolecule` | WT-3, WT-2, WT-4 |
| `filtered_candidate.py` | `FilteredCandidate` | WT-2, WT-4 |
| `docking_result.py` | `DockingPose`, `DockingResult` | WT-2, WT-3, WT-7, WT-4 |
| `fep_result.py` | `FEPResult` | WT-4, WT-8 |
| `pipeline_config.py` | `DockingConfig`, `FilterConfig`, `FEPConfig`, `PipelineConfig` | WT-4 (all WTs read) |
| `residue_mapping.py` | `ResidueEntry`, `ResidueMapping` | All WTs |

---

## Field Reference

### `PharmacophoreFeature`

| Field | Type | Description |
|-------|------|-------------|
| `feature_type` | `str` | Pharmacophore type code: AR, PI, NI, HBD, HBA, HY |
| `x` | `float` | X coordinate (Angstrom, model frame) |
| `y` | `float` | Y coordinate (Angstrom) |
| `z` | `float` | Z coordinate (Angstrom) |
| `intensity` | `float` | PRISM spike intensity, normalised 0–1 |
| `source_spike_type` | `str` | Raw spike type (BNZ, TYR, CATION, ANION, etc.) |
| `source_residue_id` | `int` | Topology residue ID of the source residue |
| `source_residue_name` | `str` | Human-readable label, e.g. "TYR142" |
| `wavelength_nm` | `float` | UV excitation wavelength (nm) |
| `water_density` | `float` | Local solvent accessibility at feature site |

### `ExclusionSphere`

| Field | Type | Description |
|-------|------|-------------|
| `x` | `float` | Centre X (Angstrom) |
| `y` | `float` | Centre Y (Angstrom) |
| `z` | `float` | Centre Z (Angstrom) |
| `radius` | `float` | Exclusion radius (Angstrom) |
| `source_atom` | `str` | Atom label, e.g. "CA:ALA145" |

### `SpikePharmacophore`

| Field | Type | Description |
|-------|------|-------------|
| `target_name` | `str` | Target identifier, e.g. "KRAS_G12C" |
| `pdb_id` | `str` | PDB accession of input structure |
| `pocket_id` | `int` | Zero-based pocket index from PRISM |
| `features` | `List[PharmacophoreFeature]` | Pharmacophore feature points |
| `exclusion_spheres` | `List[ExclusionSphere]` | Steric exclusion volumes |
| `pocket_centroid` | `Tuple[float, float, float]` | Pocket centroid (Angstrom) |
| `pocket_lining_residues` | `List[int]` | Topology residue IDs lining the pocket |
| `prism_run_hash` | `str` | SHA-256 hex digest of PRISM binary + input |
| `creation_timestamp` | `str` | ISO-8601 UTC timestamp |

**Methods:**
- `to_phoregen_json() -> dict` — PhoreGen input format
- `to_pgmg_posp() -> str` — PGMG `.posp` pharmacophore spec
- `to_docking_box(padding=4.0) -> dict` — Docking box (matches gpu_dock format)

### `GeneratedMolecule`

| Field | Type | Description |
|-------|------|-------------|
| `smiles` | `str` | Canonical SMILES string |
| `mol_block` | `str` | 3-D SDF mol-block (V2000/V3000) |
| `source` | `str` | Generator identifier: "phoregen" or "pgmg" |
| `pharmacophore_match_score` | `float` | Fraction of features matched (0–1) |
| `matched_features` | `List[str]` | Matched feature type codes |
| `generation_batch_id` | `str` | UUID/tag for the generation batch |
| `generation_timestamp` | `str` | ISO-8601 UTC timestamp |

### `FilteredCandidate`

| Field | Type | Description |
|-------|------|-------------|
| `molecule` | `GeneratedMolecule` | The underlying generated molecule |
| `qed_score` | `float` | Quantitative Estimate of Drug-likeness (0–1) |
| `sa_score` | `float` | Synthetic Accessibility (1 easy – 10 hard) |
| `lipinski_violations` | `int` | Lipinski Rule-of-Five violations (0–4) |
| `pains_alerts` | `List[str]` | PAINS filter hit names (empty if clean) |
| `tanimoto_to_nearest_known` | `float` | Max Tanimoto (ECFP4) to reference set |
| `nearest_known_cid` | `str` | PubChem CID of nearest known compound |
| `cluster_id` | `int` | Butina/Taylor cluster assignment index |
| `passed_all_filters` | `bool` | True if all filters passed |
| `rejection_reason` | `Optional[str]` | Rejection reason, or None if passed |

### `DockingPose`

| Field | Type | Description |
|-------|------|-------------|
| `pose_rank` | `int` | 1-based rank by primary scoring function |
| `mol_block` | `str` | SDF mol-block of the docked 3-D pose |
| `vina_score` | `float` | AutoDock Vina score (kcal/mol) |
| `cnn_score` | `float` | GNINA CNN pose-quality score (0–1) |
| `cnn_affinity` | `float` | GNINA CNN predicted affinity (kcal/mol) |
| `rmsd_lb` | `float` | RMSD lower-bound to best mode (Angstrom) |
| `rmsd_ub` | `float` | RMSD upper-bound to best mode (Angstrom) |

### `DockingResult`

| Field | Type | Description |
|-------|------|-------------|
| `compound_id` | `str` | Unique compound identifier |
| `smiles` | `str` | Canonical SMILES |
| `site_id` | `int` | PRISM pocket index |
| `receptor_pdb` | `str` | Path to receptor PDB |
| `poses` | `List[DockingPose]` | Ranked docked poses |
| `best_vina_score` | `float` | Best Vina score across poses |
| `best_cnn_affinity` | `float` | Best CNN affinity across poses |
| `docking_engine` | `str` | Engine ("unidock", "gnina", "unidock+gnina") |
| `box_center` | `Tuple[float, float, float]` | Docking box centre (Angstrom) |
| `box_size` | `Tuple[float, float, float]` | Docking box dimensions (Angstrom) |
| `exhaustiveness` | `int` | Search exhaustiveness (default 32) |
| `docking_timestamp` | `str` | ISO-8601 UTC timestamp |

### `FEPResult`

| Field | Type | Description |
|-------|------|-------------|
| `compound_id` | `str` | Unique compound identifier |
| `delta_g_bind` | `float` | Predicted binding free energy (kcal/mol) |
| `delta_g_error` | `float` | Statistical uncertainty ± (kcal/mol) |
| `method` | `str` | FEP method: "ABFE" or "RBFE" |
| `n_repeats` | `int` | Number of independent repeats |
| `convergence_passed` | `bool` | BAR/MBAR convergence criteria met |
| `hysteresis_kcal` | `float` | Forward–reverse hysteresis (kcal/mol) |
| `overlap_minimum` | `float` | Min lambda-window overlap (0–1) |
| `max_protein_rmsd` | `float` | Max backbone RMSD during FEP (Angstrom) |
| `restraint_correction` | `float` | Boresch restraint correction (kcal/mol) |
| `charge_correction` | `float` | Finite-size charge correction (kcal/mol) |
| `vina_score_deprecated` | `Optional[float]` | Legacy Vina score |
| `spike_pharmacophore_match` | `str` | Match summary, e.g. "4/5 features within 2.0A" |
| `classification` | `str` | NOVEL_HIT, RECAPITULATED, WEAK_BINDER, or FAILED_QC |
| `raw_data_path` | `str` | Path to OpenFE output directory |
| `fep_timestamp` | `str` | ISO-8601 UTC timestamp |

**Properties:**
- `corrected_delta_g` — ΔG with restraint + charge corrections applied
- `passed_qc` — True if all QC gates pass (convergence, hysteresis < 1.0, overlap >= 0.03, RMSD < 3.0)

### `PipelineConfig`

| Field | Type | Description |
|-------|------|-------------|
| `project_name` | `str` | Human-readable project name |
| `target_name` | `str` | Target identifier |
| `pdb_id` | `str` | Input PDB accession |
| `receptor_pdb` | `str` | Path to receptor PDB |
| `topology_json` | `str` | Path to PRISM topology JSON |
| `binding_sites_json` | `str` | Path to binding_sites.json |
| `output_dir` | `str` | Root output directory |
| `conda_env_prefix` | `str` | Conda env prefix for subprocess activation |
| `docking` | `DockingConfig` | Docking-stage config |
| `filtering` | `FilterConfig` | Filtering-stage config |
| `fep` | `FEPConfig` | FEP-stage config |
| `stages_enabled` | `List[str]` | Active pipeline stages |
| `random_seed` | `int` | Global random seed (default 42) |

### `ResidueEntry`

| Field | Type | Description |
|-------|------|-------------|
| `topology_id` | `int` | PRISM topology residue index (0-based) |
| `pdb_resid` | `int` | PDB residue sequence number |
| `pdb_chain` | `str` | PDB chain identifier |
| `pdb_insertion_code` | `str` | PDB insertion code (e.g. "A"), or "" |
| `uniprot_position` | `Optional[int]` | 1-based UniProt position, or None |
| `residue_name` | `str` | Three-letter amino acid code |

### `ResidueMapping`

| Field | Type | Description |
|-------|------|-------------|
| `pdb_id` | `str` | PDB accession |
| `uniprot_id` | `str` | UniProt accession |
| `chains` | `List[str]` | Chain identifiers present |
| `entries` | `List[ResidueEntry]` | Complete residue mapping records |
| `topology_residue_count` | `int` | Total residues in PRISM topology |
| `pdb_residue_count` | `int` | Total residues in PDB ATOM records |
| `mapping_source` | `str` | How mapping was generated (default "prism-prep") |

**Lookup methods:**
- `topology_to_pdb(topology_id) -> ResidueEntry`
- `pdb_to_topology(pdb_resid, chain, insertion_code) -> ResidueEntry`
- `uniprot_to_entries(uniprot_pos) -> List[ResidueEntry]`
- `topology_ids_to_pdb_labels(topo_ids) -> List[str]`

---

## Serialization

All dataclasses support:
- `to_dict() -> dict` / `from_dict(d) -> cls` — plain dict
- `to_json() -> str` / `from_json(s) -> cls` — JSON string
- `to_pickle() -> bytes` / `from_pickle(b) -> cls` — pickle bytes

`PipelineConfig` also supports:
- `from_json_file(path)` / `save_json(path)` — file-based I/O

---

## Running Tests

```bash
PYTHONPATH=. python -m pytest tests/test_interfaces/ -v
```
