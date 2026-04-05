# PRISM4D — Canonical System Rules

## SYSTEM ARCHITECTURE
Rust + CUDA engine (`nhs_rt_full`) → Python canonical pipeline (`prism_canonical.py`) → Consensus (`prism_replicate.py`)

Single runs are for debugging.  **Production = replicate consensus.**

## CANONICAL ENTRY POINT
```bash
python3 scripts/prism_replicate.py \
    --topology <topo>.json \
    --target-name <name> \
    --n-replicates 5 \
    --output-dir <dir>
```

## IMMUTABLE RULES
1. **No composite scores.** No weighted sums. No DPS. No implicit blending.
2. **Ranking is lexicographic only.** Persistence → pass_fraction → stability → quality.
3. **Every validated feature MUST execute.** `feature_registry.py` asserts 14/14 features.  If a feature is implemented, tested, committed — it is NOT optional.
4. **Gating stack order is fixed:** Therm → Coherence(soft) → Localization → Contact Reorg → Response Selectivity.
5. **Coherence never blocks alone.** It is advisory only.
6. **Consensus operates on site objects, not ranks.** Clustering by centroid + lining residue overlap.
7. **Do not touch the GTCKL ranking formula** in Rust without BENCH60 validation.
8. **Never use "oracle"** in benchmark reporting.
9. **OptiX is removed.** Never mention or reference it.

## PIPELINE FLOW
```
nhs_rt_full (Rust/CUDA)
  ↓ binding_sites.json + kcc_visualization.json + spike_events + trajectory
prism_canonical.py (Python)
  ↓ Load all → Merge KCC → Gating stack → Design layers → DesignBrief
  ↓ feature_registry asserts 14/14
prism_replicate.py (N replicates)
  ↓ N × (engine + canonical) → consensus clustering
  ↓ consensus_sites.json + consensus_design_briefs/ + consensus_gate_summary.json
```

## FEATURE REGISTRY (all must execute)
```
binding_sites_loaded, spike_events_loaded, kcc_loaded, trajectory_loaded,
gating_therm, gating_coherence, gating_localization,
gating_contact_reorg, gating_response_selectivity,
anchor_points, growth_vectors, pocket_profiles,
site_ranking, design_briefs
```

## KEY FILES
- `scripts/prism_replicate.py` — production entry point (replicate + consensus)
- `scripts/prism_canonical.py` — single-run canonical pipeline
- `scripts/gating_stack.py` — GTCKL+RS gating
- `scripts/consensus.py` — cross-run metastable pocket consensus
- `scripts/feature_registry.py` — 14-feature runtime assertions
- `scripts/contact_reorg_gate.py` — contact reorganization gate
- `scripts/response_selectivity.py` — response selectivity gate
- `scripts/anchor_point_map.py` — anchor point extraction (KCC-boosted)
- `scripts/growth_vector_map.py` — growth vector + subpocket segmentation
- `scripts/pocket_profile_builder.py` — pocket chemistry profile
- `scripts/site_ranker.py` — lexicographic ranker
- `scripts/design_brief_builder.py` — JSON + PyMOL + HTML output

## VALIDATION HOOKS
- `cargo check` required after any Rust edit
- `python3 -m pytest` required after any Python edit
- Hooks in `.claude/hooks/` enforce this at session end

## TESTS
```bash
python3 -m pytest tests/test_gating/ tests/test_design/ tests/test_interfaces/ -v
```

## PRISM-4D ENGINE RUN PROTOCOL — MANDATORY

### Single entry point
**`scripts/prism-validate-and-run.sh`** is the ONLY permitted way to invoke the engine.
Direct invocation of `nhs_rt_full` is PROHIBITED in all scripts, Makefiles, and pipelines.

### Canonical run command
```bash
scripts/prism-validate-and-run.sh \
    -t <topology.json> \
    -o <output_dir> \
    --fast --hysteresis --multi-stream 8 \
    --spike-percentile 95 --prism-therm \
    --fused-steps 4 --hmr --adaptive-dt \
    --replica-seed 42 -v
```

### Full prep pipeline for any new target
```bash
# 1. Download
curl -s "https://files.rcsb.org/download/XXXX.pdb" -o xxxx_raw.pdb

# 2. Clean (strips altconfs, keeps chain A, validates residue diversity)
python3 scripts/prism-clean.py xxxx_raw.pdb xxxx_clean.pdb A

# 3. Prep topology (ONLY valid prep tool)
scripts/prism-prep xxxx_clean.pdb xxxx_clean.topology.json

# 4. Run with validation gates
scripts/prism-validate-and-run.sh \
    -t xxxx_clean.topology.json \
    -o output/xxxx \
    --fast --hysteresis --multi-stream 8 \
    --spike-percentile 95 --prism-therm \
    --fused-steps 4 --hmr --adaptive-dt \
    --replica-seed 42 -v

# 5. (Optional) P2Rank reranking
prank predict -f xxxx_clean.pdb -o output/xxxx/p2rank -threads 4
python3 scripts/p2rank_rerank.py \
    --prism-sites output/xxxx/xxxx_clean.binding_sites.json \
    --prism-viz output/xxxx/xxxx_clean.kcc_visualization.json \
    --p2rank-pred output/xxxx/p2rank/xxxx_clean.pdb_predictions.csv \
    --input-pdb xxxx_clean.pdb \
    --output output/xxxx/xxxx_clean.reranked.json
```

### Known failure modes (what the validation gates prevent)

| Check | What it catches | Example |
|-------|----------------|---------|
| prism-clean: residue type diversity < 15 | Corrupted input, altconf pollution | 2iyt: altconfs collapsed 18→7 types |
| preflight: CYS registry | Catalytic cysteine mutated by AMBER | 1bzj: Cys215→Ser during prep |
| preflight: residue diversity < 15 | Topology corruption from bad PDB | Any target with stripped side chains |
| preflight: HIS → HID/HIE/HIP | AMBER protonation not assigned | Missing histidine tautomers |
| postflight: sites array empty | Engine detection failure | Large proteins with no consensus |
| postflight: missing reranked.json | Ranking pipeline incomplete | P2Rank not run or failed |
| postflight: top 5 identical residues | Single pocket repeated | Low-diversity detection |

### CYS requirements registry
```
1bzj: CYS REQUIRED (PTP1B catalytic Cys215)
1r3m: CYS REQUIRED (RNase disulfides)
2iyt: CYS NOT REQUIRED (Shikimate Kinase)
3uyi: CYS REQUIRED
1nna: CYS NOT REQUIRED (Neuraminidase)
1jwp: CYS NOT REQUIRED (TEM-1)
1p38: CYS REQUIRED (p38 MAPK)
2hnp: CYS REQUIRED (EphB2)
```

### Multichain procedure

Before cleaning any multichain PDB, answer:
1. Is the cryptic/binding site at a chain interface? → merge chains
2. Is each chain an independent target? → run separately, one chain per run
3. Is one chain a ligand/peptide/cofactor? → keep protein chain only

Full multichain pipeline:
```bash
# 1. Download
curl -s "https://files.rcsb.org/download/XXXX.pdb" -o xxxx_raw.pdb

# 2. Check what chains exist
grep "^ATOM" xxxx_raw.pdb | awk '{print $5}' | sort -u

# 3. Clean each chain separately
python3 scripts/prism-clean.py xxxx_raw.pdb xxxx_chainA.pdb A
python3 scripts/prism-clean.py xxxx_raw.pdb xxxx_chainB.pdb B

# 4. Merge with chain map
python3 scripts/prism-merge-chains.py xxxx_chainA.pdb xxxx_chainB.pdb \
    -o xxxx_merged.pdb --chain-map xxxx_merged.chain_map.json

# 5. Prep merged structure
scripts/prism-prep xxxx_merged.pdb xxxx_merged.topology.json

# 6. Run with chain map passed through
scripts/prism-validate-and-run.sh \
    -t xxxx_merged.topology.json \
    -o output/xxxx \
    --chain-map xxxx_merged.chain_map.json \
    --fast --hysteresis --multi-stream 8 \
    --spike-percentile 95 --prism-therm \
    --fused-steps 4 --hmr --adaptive-dt \
    --replica-seed 42 -v
```

### Known multichain targets in hard4 benchmark
- **1r3m**: bovine seminal RNase, obligate dimer (chains A+B).
  If chain-A-only run does not detect His12/Lys41/His119 active site,
  rerun with merged AB. Interface pocket requires both chains.

### Residue ID translation
All engine output residue IDs for merged topologies are sequential merged numbers.
Always run `prism-lookup-residue.py` before cross-referencing with literature.
Never assume merged residue ID == PDB author residue number for multichain targets.

## TOPOLOGY RESIDUE OFFSETS (CRITICAL)
Topology residue IDs ≠ PDB residue IDs. The topology renumbers from 1.
Formula: `topology_resnum = pdb_author_resnum - (pdb_first_resnum - 1)`
Always verify offset before cross-referencing detected residues against literature.
- **1btl**: offset -26
- **4obe**: KRAS G12C, check topology for CYS/GLY at position 12
- **1bzj**: topology = PDB - 1 (PDB starts at 2)

### Cascade flag behavior — critical
--cascade eliminates low-persistence sites including thermodynamically-identified
cryptic pockets. Use cases:

Active site / orthosteric detection:
  --cascade --boltzmann-rank
  Reduces ~25 sites to ~14 high-confidence candidates
  Eliminates transient/cryptic sites

Cryptic site detection:
  --boltzmann-rank (no --cascade)
  Keep all sites, filter post-hoc by therm_class=CRYPTIC
  Use prism_therm output as primary signal

For hard4 benchmark (mixed active + cryptic):
  Run both configurations and compare
  reranked output does not exist in current engine — use kcc_visualization.json
  CRYPTIC therm pockets: filter topology.prism_therm.json for therm_class=CRYPTIC

## SCRIPT EXECUTION POLICY (MANDATORY)

Full text: `docs/PRISM4D_DEV_OPS_FRAMEWORK.md` §1.

**Rule — NO SCRIPT EXECUTION WITHOUT PRODUCTION PATH TAG**

- `scripts/production/` — executable freely.
- **All other scripts require explicit permission before execution.**
  This includes `scripts/` (non-production), `benchmarks/`,
  `prism-ai-inference/scripts/`, `scripts/quarantine/`, `/tmp`, and
  any untracked location.
- **Inline `python3` heredocs that WRITE anything require permission.**
  Writing = file creation, file modification, DB mutation, network
  POST/PUT/DELETE, or any side effect outside stdout/stderr.
- **Inline `python3` that only READS is allowed only if tagged
  `[DIAGNOSTIC]`** in the invocation.
- **Multi-line `python3` heredocs are not allowed inline.** Write to
  `scripts/quarantine/` first, then ask for permission to run.
- **No production script may reference `/tmp`.** Enforcement:
  `grep -r "/tmp/" scripts/production/` must return zero results.

Also see `docs/PRISM4D_DEV_OPS_FRAMEWORK.md` §2 for the SOP rule
(documentation is written as part of the procedure, not after).
