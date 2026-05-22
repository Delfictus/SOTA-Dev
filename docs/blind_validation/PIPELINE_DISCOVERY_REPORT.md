# PIPELINE DISCOVERY REPORT
**Generated:** 2026-05-13 UTC  
**Repo HEAD:** f8f368f6b83e118126e691626a823866e49906f5  
**Host:** Prism4D  
**Scope:** Discovery of the exact pipeline used for the 10-target PRISM4D publication run (20260512)

---

## Directive interpretation and conflict resolution

Priority order applied throughout this report:

1. **No validation-coordinate access before freeze** — any script that reads holo PDB/mmCIF coordinates is post-freeze-only regardless of other considerations.
2. **Reuse prior last-10-target methods** — scripts found in the pub run scripts and prism_pub_baseline_validator.py are authoritative; no parallel workflow invented.
3. **Conservative scientific claims** — where conflicts exist between documentation and source code, source code wins.
4. **Reproducibility and auditability** — all commands recorded verbatim from discovered run scripts; seed omissions noted explicitly.
5. **Record uncertainty** — missing artifacts listed as MISSING, not silently substituted.

**Decisions made:**
- Step-count conflict (45k vs 55k): source code (fused_engine.rs) = 45,000 steps is authoritative; Technical Reference doc is wrong. Flagged in METHOD_DRIFT.
- Spike-percentile conflict (50 vs 70): 9/10 pub run scripts used 50; locked at 50 for blind validation, MCL1 anomaly documented.
- NMA flags absent from CLAUDE.md but present in pub runs: pub run scripts are authoritative for what actually ran; locked to include --nma-perturb + --multi-scale.
- No pair-breaking null script found: recorded as MISSING; will require implementation before blind validation achieves full statistical rigor.

---

## Discovery summary table

| Component | Path | Confidence | Pre-freeze safe | Holo access | Wrapper needed |
|-----------|------|-----------|----------------|-------------|----------------|
| Execution entry point | scripts/prism-validate-and-run.sh | HIGH | YES | NO | NO |
| Structure clean | scripts/prism-clean.py | HIGH | YES | NO | NO |
| Topology generation | scripts/prism-prep (binary) | HIGH | YES | NO | NO |
| NMA mode generation | scripts/prism-prep --nma flag | MEDIUM | YES | NO | NO |
| Post-processing (LORO/family/shell) | scripts/quarantine/prism_pub_baseline_validator.py | HIGH | NO (holo access) | YES | NO |
| Canonical pipeline | scripts/prism_canonical.py | HIGH | YES (pre-holo) | NO | NO |
| Replicate consensus | scripts/prism_replicate.py | HIGH | YES (pre-holo) | NO | NO |
| Consensus clustering | scripts/consensus.py | HIGH | YES | NO | NO |
| Baseline: fpocket | snap fpocket (v4.2.3) | HIGH | YES | NO | NO |
| Baseline: P2Rank | /opt/p2rank/prank (v2.4.2) | HIGH | YES | NO | NO |
| Baseline orchestrator | scripts/run_baselines.py | MEDIUM | YES | NO | Needs target list |
| Shell scoring | embedded in prism_pub_baseline_validator.py | HIGH | NO | YES | NO |
| LORO | embedded in prism_pub_baseline_validator.py | HIGH | NO | YES | NO |
| Family collapse | embedded in prism_pub_baseline_validator.py | HIGH | NO | YES | NO |
| Causal driver extraction | kcc_visualization.json driver_residue_id field | HIGH | YES | NO | NO |
| Thermo descriptors | topology.prism_therm.json (--prism-therm flag) | HIGH | YES | NO | NO |
| Pair-breaking null | **MISSING** | MISSING | — | — | Must implement |
| Decoy surface patches | **MISSING** | MISSING | — | — | Must implement |
| PyMOL visualization | scripts/generate_pymol_viz.py + render_prism4d_panels.py | HIGH | YES | NO | NO |
| Publication orchestrator | /tmp/run_pub_baseline_validation.sh | HIGH | NO | YES | NO |

---

## 1. Execution entry point

**Path:** `scripts/prism-validate-and-run.sh`  
**Invocation evidence:** All 10 pub run scripts invoke this wrapper. Direct invocation of `nhs_rt_full` binary returns exit 2 (enforced by engine, see nhs_rt_full.rs:771-779).  
**Expected inputs:** topology JSON, output dir, run flags  
**Expected outputs:** All per-target output files (see §17)  
**Used in last-10-target run:** YES  
**Holo access:** NO  
**Pre-freeze safe:** YES  

Locked pub run command (representative, from run_01_KRAS_G12C_chainA.sh):
```bash
RUST_LOG=info ./scripts/prism-validate-and-run.sh \
    -t <topology.json> \
    -o <outdir> \
    --fast \
    --hysteresis \
    --prism-therm \
    --multi-stream [8|20] \
    --multi-scale \
    --spike-percentile 50 \
    --hmr \
    --adaptive-dt \
    --fused-steps 6 \
    --nma-perturb <nma_modes.json> \
    --nma-amplification 3.0 \
    -v 2>&1 | tee "$OUT/run.log"
```

Sizing rule: ≤400 residues → --multi-stream 8; >400 → --multi-stream 20 (TRPV1 used 20).

---

## 2. Structure preparation

**Clean script:** `scripts/prism-clean.py`  
- Strips altconfs, keeps single chain, validates residue diversity ≥15 types  
- Invocation: `python3 scripts/prism-clean.py <raw.pdb> <clean.pdb> A`  
- Pre-freeze safe: YES  
- Holo access: NO  

**Topology + NMA:**  
- Binary: `scripts/prism-prep`  
- Invocation: `scripts/prism-prep <clean.pdb> <clean.topology.json>` (also generates `<target>_nma_modes.json` in pub runs)  
- Pre-freeze safe: YES  
- Holo access: NO  

Clean PDB source location (pub runs): `/home/diddy/Desktop/Prism4D-v1.1-frozen/corpus-apr12/runs/publication_prep_20260512/clean_pdbs/`  
Topology location: `.../topologies/`  
NMA modes location: `.../topologies/<target>_nma_modes.json`

---

## 3. Raw spike extraction

**Method:** Engine-native — spike events are detected and written during simulation via device-resident sensing kernel.  
**Output:** `<target>.topology.spike_events.arrow` (full topology Arrow file) and `<target>.site<ID>.spike_events.parquet` (per-site Parquet files)  
**Extraction flag:** `--fast` (activates fast_35k protocol), `--prism-therm` (thermo output), `--multi-scale` (multi-timescale detector banks)  
**No standalone spike extraction script** — extraction is fused into the simulation kernel, not a separate post-run step.  
**Pre-freeze safe:** YES (output is prediction artifact, contains no holo coordinates)  

Arrow schema: per-timestep spike events with residue IDs, event amplitudes, stream index, phase flags.

---

## 4. Post-processing scripts

**PRIMARY SCRIPT:** `scripts/quarantine/prism_pub_baseline_validator.py`  
**Purpose:** Publication-grade target-agnostic baseline comparison and LORO validator  
**Invocation evidence:** `/tmp/run_pub_baseline_validation.sh` calls this script for all 9 preprint targets  
**Inputs:** --run-dir (PRISM output dir), --query-pdb (clean apo PDB), --target (target name), --outdir  
**Outputs:** prism_vs_holo.csv, baseline_vs_holo.csv, loro_results.csv, family_results.csv, validation_report.md, pymol_overlay.pml  
**Holo access:** YES — downloads validation holo PDBs, aligns, extracts shells → **POST-FREEZE ONLY**  
**Pre-freeze safe:** NO  

Embedded holo reference table (TARGET_HOLOS dict):
- MCL1: 4HW4, 5FDR, 5W62, 6OQC, 6W8I
- KRAS_G12C: 6OIM, 6P8Y, 7T8I, 5V9L
- p53_Y220C: 6GGE, 6SI3, 7ATA (404 in prior run), 7O70, 6TP6
- TEAD3: 6CDY, 5OAQ, 6GE0, 5GQM
- GLP1R: 7LCI, 6VCB, 5NX2
- STING: 4KSY, 4LOH, 5CFQ, 7BIQ
- AKT1: 5KCV, 4EKK, 3CQW, 4GV1
- Kv31: 7PHH, 6Y7Y
- TRPV1: 5IS0, 5IRZ

Alignment gates (locked from script): ALIGN_RMSD_GATE=5.0 Å, ALIGN_SEQID_GATE=0.50  
Shell cutoffs: 4.0, 6.0, 8.0 Å  
MIN_LIGAND_HEAVY=6 atoms  
KCC_HIGH_SCORE_THRESH=0.55  

**SUPPORTING SCRIPTS:**  
- `scripts/prism_canonical.py` — single-run canonical gating stack (GTCKL+RS, 14-feature assertion), pre-freeze safe  
- `scripts/prism_replicate.py` — N-replicate consensus, pre-freeze safe  
- `scripts/consensus.py` — cross-run metastable pocket consensus, pre-freeze safe  

---

## 5. Candidate table / KCC extraction

**Method:** Engine-native, from `kcc_visualization.json` output  
**Key fields:**
- `d["sites"]`: list of site objects with `id`, `centroid` [x,y,z], `kcc` subobject containing `candidate_residue_ids` and `driver_residue_id`
- `d["residues"]`: list with `residue_id` (1-indexed sequential), `residue_name`, `ca_position` [x,y,z]
- `d["binding_sites"]`: integer count (not a list)

**Residue ID offset:** topology_resnum (1-indexed) → PDB author resnum via per-target offset (first ATOM residue - 1)  
**Pre-freeze safe:** YES  
**Holo access:** NO  

---

## 6. Residue-shell mapping

**Method:** Embedded in `prism_pub_baseline_validator.py` — all-atom distances from holo ligand heavy atoms to protein residues at 4/6/8 Å cutoffs after Kabsch alignment.  
**Pre-freeze:** The manifold residue list from kcc_visualization.json (8 Å from centroid) is pre-freeze safe. The ligand-derived shell (from holo) is **post-freeze only**.  
**Supporting script:** `scripts/quarantine/prism_manifold_shell_validator.py` — earlier version of shell validation logic  

---

## 7. Causal driver extraction

**Method:** `kcc_visualization.json` → `d["sites"][i]["kcc"]["driver_residue_id"]` (integer, 1-indexed topology ID)  
**Translation to PDB:** `prism-lookup-residue.py` or offset formula  
**Pre-freeze safe:** YES  

---

## 8. Thermodynamic descriptor extraction

**Method:** `--prism-therm` flag → writes `<target>.topology.prism_therm.json`  
**Key fields:** `therm_class` per site (CRYPTIC/ORTHOSTERIC/etc.), `hysteresis_asymmetry`  
**Also in binding_sites.json:** each site has `therm_class` and `hysteresis_asymmetry` fields  
**Pre-freeze safe:** YES  

---

## 9. Family recurrence / LORO

**Method:** Both embedded in `scripts/quarantine/prism_pub_baseline_validator.py`  
- **LORO** (lines 771+): withholds one holo reference, builds support from remaining, tests recovery of withheld  
- **Family collapse** (lines 839+): union-find on co-validated ligand instance overlap → produces family summary dicts sorted by coverage  
**Pre-freeze safe:** NO (requires holo access for shell overlap)  
**Standalone script:** NONE — must use prism_pub_baseline_validator.py  

---

## 10. Ligand-shell scoring

**Method:** Embedded in `prism_pub_baseline_validator.py` — Jaccard@4/6/8Å, min_dist, KCC driver in shell, verdict  
**Scoring script for alternate use:** `scripts/quarantine/site_vs_holo_strict.py` (stricter rules, hardcoded targets; not used for pub run)  
**Pre-freeze safe:** NO  

---

## 11. Strict null controls

**STATUS: MISSING**  
No pair-breaking null script found in scripts/, benchmarks/, or any other directory.  
No decoy surface patch script found.  
The pub baseline validator does NOT implement pair-breaking null permutation.  
**Required for full statistical rigor:** implement before blind validation execution.  
Minimum acceptable: 1,000 iterations (document as reduced power); preferred: 10,000 iterations.  

---

## 12. Baseline wrappers

**fpocket:**  
- Binary: `fpocket` (snap package v4.2.3, installed 2026-03-12)  
- Invocation in pub baseline validator: `subprocess.run(["fpocket", "-f", local_pdb], ...)` with output parsed from `<pdb>_out/<pdb>_info.txt`  
- Pre-freeze safe: YES (runs on apo input only)  

**P2Rank:**  
- Binary: `/opt/p2rank/prank` (v2.4.2)  
- Invocation: `prank predict -f <apo.pdb> -o <outdir>`  
- Output: `<pdb>_predictions.csv`  
- Pre-freeze safe: YES  

**Orchestrator:** `scripts/run_baselines.py` — written for bench30 dataset, needs wrapper to accept pub target list. Thin compatibility wrapper required.

---

## 13. Visualization scripts

- `scripts/generate_pymol_viz.py` — general PyMOL viz from binding_sites.json  
- `scripts/generate_chimerax_viz.py` — ChimeraX viz  
- Engine-native: `<target>.kcc_session.pml`, `<target>.binding_sites.pml`  
- Publication renderer: `render_prism4d_panels.py` (Downloads/PRISM4D_figures_and_pymol_package/pymol/)  
- Post-freeze structural overlays: `prism_pub_baseline_validator.py` writes `pymol_overlay.pml` using holo-aligned coordinates → **POST-FREEZE ONLY**  

---

## 14. Prior target directories discovered

**Publication run outputs** (`/mnt/storage/prism-outputs/runs/`):
| Directory | Status |
|-----------|--------|
| AKT1_chainA_20260512_203906 | COMPLETE |
| KRAS_G12C_chainA_20260512_194818 | COMPLETE |
| STING_chainA_20260512_202612 | COMPLETE |
| MCL1_chainA_20260512_194006 | COMPLETE |
| p53_Y220C_chainA_20260512_195447 | COMPLETE |
| TEAD3_chainA_20260512_200421 | COMPLETE |
| GLP1R_chainA_20260512_201334 | COMPLETE |
| Kv31_chainA_primary_20260512_210836 | COMPLETE |
| TRPV1_chainA_20260512_212518 | COMPLETE |
| 10k_campaign | OTHER |
| blind_validation_100 | OTHER |
| cryptobench199 | OTHER |

Note: M4R run directory not found in /mnt/storage/prism-outputs/runs/ — may be at alternate path or run was not completed.

---

## 15. Expected output files per target

(verified from AKT1_chainA_20260512_203906/):
```
<target>.binding_sites.json
<target>.binding_sites.cxc
<target>.binding_sites.md
<target>.binding_sites.pdb
<target>.binding_sites.pml
<target>.ensemble_trajectory.json
<target>.kcc_session.pml
<target>.kcc_validation.json
<target>.kcc_visualization.json
<target>.site<ID>.spike_events.parquet   (one per detected site)
<target>.topology.druggability.pdb
<target>.topology.prism_therm.json
<target>.topology.spike_events.arrow
<target>_stream00..N.ensemble_trajectory.pdb  (one per stream)
run.log
```

---

## 16. Holo coordinate file scan

Scanned for holo/shell files in current repo: no forbidden pre-freeze files found.  
No `*_holo*.pdb`, `*_holo*.cif`, `ligand_shell*.csv`, `*_shell_overlap*.csv` detected in working tree.  
Current workspace is **CLEAN** for pre-freeze state.

---

## 17. Missing and ambiguous artifacts

| Item | Status | Impact | Resolution |
|------|--------|--------|-----------|
| Pair-breaking null script | MISSING | Affects statistical rigor | Implement thin wrapper using shuffled-rank permutation |
| Decoy surface patch script | MISSING | Affects enrichment analysis | Implement residue-count-matched random surface sampling |
| --replica-seed in pub runs | NOT SET | Affects exact reproducibility | Document as unknown; set explicit seed in blind runs |
| M4R pub run output | NOT FOUND at expected path | Unclear if run completed | Verify alternate path |
| 7ATA PDB ID (p53 holo) | HTTP 404 | One p53 holo reference missing | Author to confirm or substitute |
| Standalone LORO script | MISSING (embedded) | Requires full validator run | Use prism_pub_baseline_validator.py |
| SMILES/3D conformer for decoy ligands | MISSING | Post-freeze only | Defer to post-freeze phase |
