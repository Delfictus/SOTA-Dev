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

## TOPOLOGY RESIDUE OFFSETS (CRITICAL)
Topology residue IDs ≠ PDB residue IDs.  Always verify with `find_residue_by_resid_name()`.
- **1btl**: offset -26
- **4obe**: KRAS G12C, check topology for CYS/GLY at position 12
