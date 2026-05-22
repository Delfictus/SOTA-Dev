# LAST-10-TARGET METHOD LOCK
**Locked:** 2026-05-13 UTC  
**Repo HEAD:** f8f368f6b83e118126e691626a823866e49906f5  
**Source:** `/home/diddy/Desktop/Prism4D-v1.1-frozen/corpus-apr12/runs/publication_prep_20260512/`

This file locks the exact methodology used for the 10-target PRISM4D publication run. All blind validation execution must reproduce this method or explicitly document deviations.

---

## 1. Target list

| # | Target | Source PDB | Chain | Residues | multi_stream | Notes |
|---|--------|-----------|-------|----------|-------------|-------|
| 1 | KRAS_G12C | 4OBE | A | 169 | 8 | KRAS G12C apo |
| 2 | Kv31 (PRIMARY) | 7PHH | A | 396 | 8 | Kv3.1 KCNC1, primary benchmark |
| 3 | p53_Y220C | 2J1X | A | 195 | 8 | |
| 4 | AKT1 | 3O96 | A | 367 | 8 | |
| 5 | TEAD3 | 8A0V | A | 208 | 8 | TEAD3 proxy for TEAD1 |
| 6 | TRPV1 | 7LQY | A | 659 | 20 | >400 res rule |
| 7 | GLP1R | 7LCI | A | 244 | 8 | Hard negative / limitation case |
| 8 | MCL1 | 6OQC | A | 152 | 8 | |
| 9 | STING | 6NT5 | A | 292 | 8 | |
| 10 | M4R | 5DSG | A | 392 | 8 | 8DSQ→5DSG substitution (404); may not be in preprint |

---

## 2. Locked run command family

Engine directory used: `/home/diddy/Desktop/Prism4D-v1.1-frozen/corpus-apr12/source-sandbox`  
All runs invoke: `./scripts/prism-validate-and-run.sh`

**Standard targets (≤400 residues, with HMR):**
```bash
RUST_LOG=info ./scripts/prism-validate-and-run.sh \
    -t <topology.json> \
    -o <outdir> \
    --fast \
    --hysteresis \
    --prism-therm \
    --multi-stream 8 \
    --multi-scale \
    --spike-percentile 50 \
    --hmr \
    --adaptive-dt \
    --fused-steps 6 \
    --nma-perturb <nma_modes.json> \
    --nma-amplification 3.0 \
    -v 2>&1 | tee "$OUT/run.log"
```

**TRPV1 (>400 residues):** same flags, `--multi-stream 20`

**Kv3.1 exception:** `--multi-stream 8` (≤400 residues threshold applies), NO --hmr flag, implied 2fs timestep (Kv3.1-specific, no HMR)

**MCL1 exception:** `--spike-percentile 70` (all others use 50 — see METHOD_DRIFT report)

---

## 3. CryoUV protocol parameters (locked from fused_engine.rs source)

| Parameter | Locked value | Source |
|-----------|-------------|--------|
| cold_hold_steps | 14,000 | fused_engine.rs fast_35k() |
| ramp_up_steps | 6,000 | fused_engine.rs |
| warm_hold_steps | 15,000 | fused_engine.rs |
| ramp_down_steps | 6,000 | fused_engine.rs (hysteresis) |
| cold_return_steps | 4,000 | fused_engine.rs (hysteresis) |
| **TOTAL per stream** | **45,000 steps** | source-verified |
| uv_burst_energy | 42.0 kcal/mol | fused_engine.rs |
| uv_burst_interval | 250 steps | fused_engine.rs |
| Temperature cold | 50 K | fused_engine.rs |
| Temperature warm | 300 K | fused_engine.rs |

**NOTE:** docs/PRISM4D_Complete_Technical_Reference.md incorrectly states 55,000 steps. Source code is authoritative. See METHOD_DRIFT report.

---

## 4. Detector parameters (locked)

| Parameter | Locked value | Source |
|-----------|-------------|--------|
| spike_percentile | 50 (9/10 targets); 70 (MCL1) | pub run scripts |
| cluster_threshold | 5.0 Å | nhs_rt_full.rs:195-196 |
| lining_cutoff | 8.0 Å | nhs_rt_full.rs:203-204 |
| fused_steps | 6 | run scripts |
| nma_amplification | 3.0 | run scripts |
| nma_modes (count) | 10 | pub run manifest |

---

## 5. Spike extraction method

Device-native, fused into simulation kernel (no post-run extraction step). Outputs:
- `<target>.topology.spike_events.arrow` — full topology Arrow format
- `<target>.site<ID>.spike_events.parquet` — per-site Parquet

---

## 6. Post-processing scripts

| Script | Purpose | Status |
|--------|---------|--------|
| scripts/quarantine/prism_pub_baseline_validator.py | Shell scoring, LORO, family, baselines | CONFIRMED USED (via /tmp/run_pub_baseline_validation.sh) |
| scripts/prism_canonical.py | Single-run canonical gating | CONFIRMED in pipeline |
| scripts/prism_replicate.py | N-replicate consensus | Available |
| scripts/consensus.py | Cross-run consensus clustering | Available |

---

## 7. Ranking method

Lexicographic: persistence → pass_fraction → stability → quality  
GTCKL+RS gating stack (scripts/gating_stack.py)  
14-feature assertion enforced by scripts/feature_registry.py  
XGBoost ranker: NOT present in pub run flags (--use-xgb-ranker absent from pub scripts)

---

## 8. Candidate-shell mapping method

Pre-freeze: kcc_visualization.json `candidate_residue_ids` (1-indexed topology IDs), 8 Å from site centroid  
Post-freeze: `prism_pub_baseline_validator.py` multi-shell (4/6/8 Å) from holo ligand heavy atoms after Kabsch alignment  
Alignment gate: RMSD ≤ 5.0 Å, seqid ≥ 0.50

---

## 9. Causal driver method

`kcc_visualization.json` → `sites[i]["kcc"]["driver_residue_id"]` (1-indexed topology ID)  
Translation to PDB author numbering: offset = first_pdb_resnum - 1

---

## 10. Thermodynamic descriptor method

`--prism-therm` flag → `<target>.topology.prism_therm.json`  
Key fields: `therm_class` (CRYPTIC/ORTHOSTERIC/ALLOSTERIC), `hysteresis_asymmetry`  
Also present in binding_sites.json per site

---

## 11. LORO method

Embedded in `prism_pub_baseline_validator.py` (lines 771+)  
Algorithm: for each unique holo PDB in TARGET_HOLOS[target], withhold that PDB, score the PRISM site against remaining holos, test if site is rediscovered in withheld holo  
No standalone LORO script exists

---

## 12. Family collapse method

Embedded in `prism_pub_baseline_validator.py` (lines 839+)  
Algorithm: union-find on co-validated ligand instance overlap — sites that co-validate the same holo ligand instances are collapsed into a family  
Output: `prism_family_vs_ligand_shells.csv`

---

## 13. Null controls

**STATUS: NOT EXECUTED in publication run**  
Pair-breaking null: MISSING (no script found)  
Decoy surface patches: MISSING (no script found)  
Permutation null: MISSING  
This is a gap in the publication validation. Must be implemented for blind validation statistical rigor.

---

## 14. Baseline method

**fpocket v4.2.3** (snap): `fpocket -f <clean.pdb>` → parse `<pdb>_out/<pdb>_info.txt`  
**P2Rank v2.4.2** (`/opt/p2rank/prank`): `prank predict -f <clean.pdb> -o <outdir>` → parse `<pdb>_predictions.csv`  
Both invoked within `prism_pub_baseline_validator.py` via subprocess  
Both scored under identical ligand-shell framework as PRISM4D

---

## 15. Visualization method

Engine-native PyMOL/ChimeraX session files written per run  
Publication PyMOL renderer: `render_prism4d_panels.py` with `prism4d_targets_config_filled_pdbs.json`  
Structural overlays (holo-aligned): written by pub baseline validator post-freeze → `pymol_overlay.pml`

---

## 16. Random seeds

**NOT SET in pub run scripts.** No `--replica-seed` flag present in any of the 10 pub run scripts.  
CLAUDE.md canonical mentions `--replica-seed 42` but this was NOT in the actual pub runs.  
Consequence: exact numerical reproducibility is not guaranteed; statistical reproducibility is expected within run-to-run variation.  
**Blind validation WILL set explicit seed** (--replica-seed 42) and document it.

---

## 17. Environment and software versions

| Component | Version | Source |
|-----------|---------|--------|
| PRISM4D engine commit | f8f368f6 (current); pub runs used Prism4D-v1.1-frozen build | git log |
| fpocket | 4.2.3 (snap, installed 2026-03-12) | snap info fpocket |
| P2Rank | v2.4.2 | /opt/p2rank/prank |
| Force field | AMBER ff14SB | simulation_runner.rs:388,421 |
| Implicit solvent | Distance-dependent dielectric ε=0.25r | ultimate_engine.rs:141 |
| Non-bonded cutoff | 12.0 Å | ultimate_engine.rs:136 |
| Timestep (HMR) | 4 fs | canonical |
| Timestep (Kv3.1, no HMR) | 2 fs | Kv3.1 run notes |
| GPU | RTX 5080 Blackwell, SM120 | system |
| Engine build | CUDA sm_120 | compile config |

---

## 18. Output file fingerprint (expected per target)

```
<target>.binding_sites.json        — primary site detections
<target>.binding_sites.{pml,cxc,md,pdb}  — visualization outputs
<target>.ensemble_trajectory.json  — ensemble trajectory metadata
<target>.kcc_session.pml           — KCC PyMOL session
<target>.kcc_validation.json       — KCC validation summary
<target>.kcc_visualization.json    — KCC residue/centroid/driver data
<target>.site<ID>.spike_events.parquet   — per-site spike data (one per site)
<target>.topology.druggability.pdb
<target>.topology.prism_therm.json — thermodynamic descriptors
<target>.topology.spike_events.arrow — full topology spike events
<target>_stream00..N.ensemble_trajectory.pdb — per-stream trajectory PDBs
run.log                            — stdout/stderr
```

---

## 19. Deviations between targets

| Target | Deviation | Value |
|--------|-----------|-------|
| MCL1 | spike_percentile | 70 (all others: 50) |
| Kv3.1 | HMR | NO (all others: YES) |
| Kv3.1 | timestep | 2 fs (all others: 4 fs) |
| TRPV1 | multi_stream | 20 (all others: 8) |
| M4R | source PDB | 5DSG (substituted for 8DSQ/404) |

---

## 20. Uncertainties

1. spike_percentile=50 vs 70: root cause of MCL1 anomaly unknown — operator may have edited manually
2. Exact --nma-perturb flags not in CLAUDE.md canonical — unclear if NMA perturb was used in ALL runs or only some
3. No --replica-seed in pub runs — exact numerical reproducibility is not guaranteed
4. M4R run not confirmed in output directory — may not be in preprint panel
5. 7ATA (p53 holo reference) returned 404 — needs author confirmation
6. Whether --multi-differential and related steering flags were available in the v1.1-frozen build is unconfirmed
