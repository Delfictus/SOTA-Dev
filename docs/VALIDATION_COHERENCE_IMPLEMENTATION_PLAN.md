# Validation / Accuracy / Coherence Implementation Plan
**Author:** Agent 7 (read-only scout)
**Branch:** `producer-repair-causal-truthing-20260426` (baseline `8ca26189`)
**Date:** 2026-05-04
**Status:** Plan only — no code written, no engine touched.

---

## 1. Files inspected

### Engine / runtime (Rust + CUDA)
- `crates/prism-nhs/src/bin/nhs_rt_full.rs` — site emission (`:2741-2790`, `:2828-2860`), KCC viz emit (`:9504-9729`), causal-truthing audit per-site block (`:14985-15297`), CCNS phase fractions and `open_frequency` (`:15299-15340`). Already emits `binding_sites.json`, `kcc_visualization.json`, `prism_therm_telemetry.json`, `cryptic_sites.json`, per-site `audit[]` blocks.
- `crates/prism-nhs/src/site_manifest.rs` — `KccMetrics` (`:500-553`), `ThermDossier` (`:569-606`), `SiteManifest` ghost-pipeline schema (`:460-477`). The CLA-2 schema is the canonical place to add a sister `SiteValidation` field.
- `crates/prism-nhs/src/persistent_engine.rs:4336-4369` — `write_binding_site_visualizations`. PDB/PML/CXC/MD writers; the writers consume `ClusteredBindingSite` (no validation join yet).
- `crates/prism-nhs/src/transform/causal_truthing_audit.rs` — L4 audit (`:50-156`); declares `LAW_L4_CAUSAL_MATH_MEANINGFUL_OR_ABSENT`, sets `MIN_MEANINGFUL_SPIKE_SUPPORT = 32`, routing = `Quarantine`. Already wired into per-site `audit[]` block.
- `crates/prism-nhs/src/ghost_telemetry.rs:69-100` — Pinned triple-buffered telemetry ring; `cuMemcpyDtoHAsync_v2`, F1 switch logger (`log_f1_switch_events` is invoked from `nhs_rt_full.rs:6857`).
- `crates/prism-nhs/src/interferometric_adjudicator.rs` — Δ_AB classifier (`:17`, `:208`); `cuda/adjudicator.cuh:61` macro `PRISM_ADJ_CONSTRUCT 1u`; `:312` ASC-steering work folded into `d_potential_energy`.
- `crates/prism-nhs/src/gpu_invariant.rs:24-206` — `audit_mass_conservation` (synthetic invariant checker, has both pass and trap tests). Reusable as a coherence sub-metric (mass conservation).
- `crates/prism-nhs/src/gearbox.rs:1-92` — G26 chronometric gearbox state struct; persistent `GearboxState{counter, last_burst_frame, current_gear, previous_gear}`.

### Offline validators (Python)
- `scripts/evaluate_benchmark.py` — DCC/DCA/DVO/SR@N implementation (full file). Lines `:111-130` (compute_dcc, compute_dca), `:216-276` (compute_dvo voxel Jaccard), `:281-298` (failure classifier), `:300-475` (main eval loop), `:625-717` (re-rank simulation). **THE main reference for accuracy metric definitions.**
- `scripts/quarantine/canonical_dcc_audit.py:1-200` — single canonical `canonical_dcc()` function (line `:60-86`). Locks `centroid_spike_weighted` (pocket) vs `ligand_centroid_apo_frame` (ligand) and "kabsch_apo_to_holo" alignment mode. Defines verdict thresholds (`:89-100`): PASS < `DCC_MATCH_A`, NOT_MATCH > 20, HARD_NOT_MATCH > 30.
- `scripts/quarantine/site_vs_holo_strict.py` — dual-pass alignment (global + local 15 Å) with `overlap_fraction` over ligand-contact residues, cutoff = 4.5 Å (`:30`). Verdicts at `:235-242`. Reusable: `do_alignment` (`:128-136`), `chain_ca` (`:53-63`), `holo_ligand_coords` (`:79-87`), `resolve_residues_one_indexed` (`:109-117`).
- `scripts/quarantine/wrn_1522_strict_verify.py:180-265` — alternative `min_atom_distance` (lining atom → any ligand atom).
- `scripts/prism-ground-truth.py` — RCSB holo fetch + classify (PanDDA / templated / metal cofactor / drug-like) (`:130-285`). Writes `<prefix>_ground_truth.json` sidecar. Also produces an arithmetic mean ligand centroid (`:288-294`); does NOT do Kabsch (apo-frame transform is delegated to the offline DCC validators).
- `scripts/prism-postflight.py:253-340` — current postflight DCC consumer; reads sidecar, computes Euclid distances, prints rank-1 and best DCC, but does NOT do alignment (relies on sidecar centroid being already in apo frame, which it is NOT — postflight assumes raw holo-frame centroid which is one of the failure modes flagged in `BENCHMARK_ALIGNMENT_GUIDE.md`).
- `scripts/prism_canonical.py`, `scripts/prism_replicate.py` — full canonical pipeline (consensus + replicate). Reads `binding_sites.json` + `kcc_visualization.json`, applies gating stack and feature registry. Validation is a separate consumer of `binding_sites.json`.
- `scripts/prism-validate-and-run.sh:127-234` — Phase 1 preflight, Phase 1.2 ground-truth resolution, Phase 2 engine run, Phase 3 postflight. Single permitted entry.
- `scripts/quarantine/build_m1_panel.py`, `scripts/quarantine/strict_dcc_panel_v1.json` — strict 18-target panel (4 TWIN-10 verified + 3 recoveries + 11 blind validation pairs with pRMSD < 2.1 Å).

### Docs
- `docs/validation/v8_dcc_validation.md` — verified-pair table with alignment mode, RMSD, n=9 baseline (33% < 5 Å, 67% < 8 Å, 89% < 10 Å, run under pre-lockdown `--spike-percentile 95`).
- `docs/validation/BENCHMARK_ALIGNMENT_GUIDE.md` — the canonical Kabsch protocol, sequence-identity gate, verified ground-truth table, **rejected (wrong) pairs list**, and DCC interpretation table. Lines `:42-95` (alignment protocol), `:99-115` (verified pairs), `:117-123` (wrong pairs to avoid), `:127-157` (Kabsch reference), `:181-192` (DCC verdict table).
- `docs/EXECUTION_POLICY.md:33` — Python is allowed for offline reading and validation; forbidden in runtime/hot path. **This is the constraint that fixes the architecture below.**

### Benchmark panels (data/targets + benchmarks)
- `benchmarks/cryptobench/ground_truth/*.ground_truth.json` — 199 files; rich CryptoBench schema with `binding_residues[]` (chain/resid/label), `n_binding_residues`, `main_holo` (holo_pdb_id + ligand + pRMSD + apo_pocket_selection + holo_pocket_selection), and `holo_entries[]` for multi-holo variants. Example: `1bzj.ground_truth.json` (56 binding residues, 24 holo structures).
- `benchmarks/true_apo/ground_truth/*.ground_truth.json` — 462 files; simpler schema with `apo_pdb`, `holo_pdb`, `ligand_code`, `ligand_centroid` (already in holo frame, **not** apo frame), `binding_residues_holo[]`, `pocket_rmsd`, `category` (CRYPTIC/etc), uniprot_id, cluster_id_30.
- `benchmarks/hard_targets/3mh1_ground_truth.json` — minimal schema `{target, reference, ligand, centroid, alignment{n_ca, rmsd}, note}`.
- `benchmarks/hard_targets/clean/*.pdb` — only 3mh1 and 5lar `clean+fixed` PDBs.
- `benchmarks/never_bound/pdbs/*.pdb` — 97 PDBs, no holo (these are negative controls — the engine should NOT detect a strong druggable site here).
- `benchmarks/true_apo/{apo_pdbs,holo_pdbs}/*.pdb` — paired structures.
- `benchmarks/prismai_bench120/{apo,...}` — 120-target panel directory (apo PDBs only on disk).
- `data/targets/` — production targets (4lpk, 1nkp_myc_dna, mpro_monomer, 7c8r_dimer_hmr, ...). Each target has `<name>.json` (topology) + `<name>.atom_to_residue.json` + `<name>.residue_map.json`.
- `data/targets/tier3/`, `data/targets/tier3_b2/` — extended panel (1bg1, 1mq4, 1qs4, 1yes, ...).
- `scripts/quarantine/strict_dcc_panel_v1.json` — frozen 18-target strict panel with `material_improvement_rule` (`top1_correct_min: 4`, FP top-1 promotions max: 1).
- `scripts/quarantine/twin10_targets.json` — TWIN-10 cohort with `paired_holo_ligand_resname`, `known_binding_residues`, `flexible_regions`, `cryptic_site_type`. Schema is the richest available — best template for new ground-truth additions.

---

## 2. Functions / structs found (load-bearing)

### Rust runtime — already emits

| Symbol | File:line | Purpose |
|---|---|---|
| `ClusteredBindingSite::emission_compat_centroid()` | `bin/nhs_rt_full.rs:2751,2777,2834` | Per-site centroid (host-side, geometric voxel mass with M2 spike-weighting hooks) |
| `KccMetrics` | `site_manifest.rs:500-533` | Ghost-pipeline KCC dossier (8 fields, all `Option`) |
| `ThermDossier` | `site_manifest.rs:569-606` | Thermodynamic dossier (`ccns_tau`, `therm_class`, `druggability`, `relative_asymmetry`, `hysteresis_asymmetry`) |
| `CausalTruthingAudit` | `transform/causal_truthing_audit.rs:121-156` | L4 audit; classifies each site as Accepted vs. Quarantined |
| `SiteCausalSummary` | `transform/causal_truthing_audit.rs:71-106` | Per-site causal summary with `is_meaningful()` predicate (≥ 32 spikes) |
| `audit_mass_conservation` | `gpu_invariant.rs:93-160` | Conservation invariant trap (mass-balance) — coherence sub-metric |
| `GearboxState` | `gearbox.rs:38-90` | G26 gear trace; persistent `{counter, last_burst_frame, current_gear, previous_gear}` |
| `PinnedTelemetryRing<T>` | `ghost_telemetry.rs:95-100` | Triple-buffered exfiltration ring (already used for ContactShellTile, available for any POD) |
| `log_f1_switch_events` | `ghost_telemetry.rs` (called from `bin/nhs_rt_full.rs:6857`) | F1 branch switch logger |

### Python offline — reusable as-is

| Function | File:line | Inputs | Outputs |
|---|---|---|---|
| `canonical_dcc()` | `quarantine/canonical_dcc_audit.py:60-86` | `pocket_id, pockets, gt` | `(dcc_Å, diagnostics)` |
| `verdict_from_dcc()` | `quarantine/canonical_dcc_audit.py:89-100` | `dcc, gt_valid` | `"PASS"|"FAIL"|"NOT_MATCH"|"HARD_NOT_MATCH"|"GT_INVALID"` |
| `do_alignment()` | `quarantine/site_vs_holo_strict.py:128-136` | `apo_ca, holo_ca, [resid_subset]` | `(Superimposer, common_residues)` |
| `compute_dcc/dca/dvo()` | `evaluate_benchmark.py:111-276` | site centroid, lining coords, ligand atoms | float metrics |
| `classify_failure()` | `evaluate_benchmark.py:281-298` | best_dcc, rank, alignment_rmsd | failure category string |
| `extract_ligand_atoms()` | `evaluate_benchmark.py:50-91` | holo PDB, lig resname | (N,3) np.array |
| `resolve()` | `prism-ground-truth.py:301-379` | topology path | sidecar dict |

### Per-site emit shape currently in `binding_sites.json` (verified at `bin/nhs_rt_full.rs:2832-2860, 15268-15297`)
```
{
  id, centroid[3], volume, spike_count, quality_score,
  druggability, is_druggable, classification, aromatic_score,
  catalytic_residue_count, lining_residues[{chain, resid, resname, min_distance}],
  rank_score, residue_ids[],
  audit[{transform, determinism, tolerance, tolerance_epsilon,
         laws_declared, laws_passed, laws_violated, law_family,
         outcome, routing, evidence, verified_at}],
  phase{ccns_*_fraction, open_frequency, ...}
}
```

**No `SiteValidation` field exists yet.** The L4 audit block lives under `audit[]` and reports the engine's *internal* causal honesty — distinct from external accuracy validation.

---

## 3. Reusable code (must be wired to the new SiteValidation, not re-implemented)

| What | Source | Why reuse |
|---|---|---|
| Verdict thresholds | `quarantine/canonical_dcc_audit.py:89-100` | Already locked: PASS < `DCC_MATCH_A` (4 Å in `m1_ablation`), NOT_MATCH > 20, HARD_NOT_MATCH > 30 |
| Dual-pass alignment | `quarantine/site_vs_holo_strict.py:128-136, 168-186` | Global + local-15Å Kabsch already implemented, with insufficient-Cα fallback |
| Ligand contact 4.5 Å rule | `quarantine/site_vs_holo_strict.py:30, 214-220` | Defines "contact shell" canonically: holo residue Cα-or-any-atom within 4.5 Å of any ligand heavy atom |
| `ligand_centroid_apo_frame` definition | `quarantine/canonical_dcc_audit.py:67-77` | Locked: heavy-atom arithmetic mean *after* Kabsch holo→apo |
| Multi-metric harness | `evaluate_benchmark.py` | DCC+DCA+DVO+SR@N already wired; SiteValidation only needs to feed the same shape |
| Ground-truth resolution | `prism-ground-truth.py:301-379` | RCSB fetch, PanDDA / templated-complex filters, drug-like vs. metal-cofactor classification, sidecar emission |
| CryptoBench rich schema | `benchmarks/cryptobench/ground_truth/*.ground_truth.json` | 199 targets with per-holo `apo_pocket_selection` + `holo_pocket_selection` for known_contact_recovery |

---

## 4. Unsafe / problematic things found

### Architectural conflicts
1. **Postflight DCC bypasses Kabsch.** `prism-postflight.py:322` uses `_euclid(sc, tuple(lig_centroid))` with the holo-frame centroid as if it were apo-frame. `prism-ground-truth.py:288-294` only computes arithmetic mean; the sidecar's `ligand_centroid` is in the *holo* frame. Per `BENCHMARK_ALIGNMENT_GUIDE.md:189-190`, this is the #1 cause of false misses (DCC > 25 Å = frame mismatch, not detection failure). **The current Phase 3 postflight DCC numbers are unreliable for any apo/holo pair where raw CA RMSD > 3 Å.**
2. **Two ground-truth schemas competing.** The CryptoBench schema has `apo_pocket_selection` (apo residues that line the holo pocket — exactly what `known_contact_recovery` needs). The `prism-ground-truth.py` sidecar schema has only `ligand_centroid` (holo frame). They do not interop. The runtime cannot tell them apart.
3. **`true_apo` ligand_centroid is also holo-frame.** `1al3_A.ground_truth.json:8-12` `ligand_centroid` is taken raw from the holo PDB (no Kabsch). Same trap.
4. **`canonical_dcc_audit.py` reads from `/mnt/storage/prism-outputs/twin-10-patent/` and `/mnt/storage/prism-outputs/m1-strict-dcc-panel/`** (`:53-57`). These are non-portable absolute paths; the audit is currently coupled to a single workstation's storage layout.
5. **`classify_failure` collapses accuracy and ranking.** `evaluate_benchmark.py:281-298` returns `SUCCESS_RANK1` etc., which fuses rank-position with detection accuracy. Per the mission constraint, accuracy and ranking are separate axes — the new `SiteValidation` must report them as orthogonal fields, with the failure category derived externally.
6. **L4 audit can't grade external accuracy.** `causal_truthing_audit.rs:71-106` validates *internal* causal honesty (NaN means "honestly unknown"). It must NEVER feed the accuracy verdict — collapsing them creates the circularity the mission flags. Keep the existing `audit[]` block as-is; add a *separate* `validation` block for external ground truth.

### Code-hygiene issues (not blockers)
7. The `compute_pocket_score` in `evaluate_benchmark.py:169-213` is a **composite weighted sum** — explicitly forbidden by `CLAUDE.md` ("No composite scores. No weighted sums. No DPS. No implicit blending."). Already inside the offline benchmark eval, but if anyone copies it into ranking they violate Immutable Rule 1. Note for future work: delete that re-rank simulation block.
8. `evaluate_benchmark.py:521` uses the word "oracle"-adjacent benchmarks language; clean per `CLAUDE.md` rule 8.
9. Three benchmark panels (`hard_targets`, `cryptobench`, `true_apo`, `prismai_bench120`, `bench_blind`, `bench10_results`, `eval_dual`) coexist with no panel registry. The strict 18-target frozen panel exists at `scripts/quarantine/strict_dcc_panel_v1.json` but no Rust-side equivalent.

---

## 5. Exact plan

### 5.1 The two-axis architecture (LOCKED)

Accuracy and coherence MUST flow on different rails. The runtime emits two separate JSON blocks per site:

```
binding_sites.json
└── sites[i]
    ├── ... existing engine fields ...
    ├── audit[]            ← internal causal honesty (already exists, keep)
    ├── validation         ← NEW. External accuracy. Writable only when GT exists.
    └── coherence          ← NEW. Internal physics fidelity. Writable always.
```

The runtime ALSO emits one new top-level file:

```
<prefix>.validation_inputs.json   ← Rust-emitted; offline validators read this
                                    instead of re-parsing engine internals.
```

### 5.2 Validation input file (Rust emits once at end-of-run)

```rust
// New module: crates/prism-nhs/src/validation_inputs.rs
struct ValidationInputs {
    schema_version: String,        // "validation_inputs_v1"
    target_pdb_id: Option<String>, // from prism-ground-truth.py sidecar
    target_chain: Option<String>,
    residue_convention: String,    // "one_indexed" — locked
    apo_frame_anchor: ApoFrameAnchor, // CA coordinates of every residue in apo frame
    sites: Vec<ValidationSiteInput>,
}

struct ApoFrameAnchor {
    n_residues: usize,
    chain_ids: Vec<String>,        // per-residue chain
    resids: Vec<i32>,              // per-residue PDB resid (from residue_map)
    resnames: Vec<String>,         // per-residue 3-letter
    ca_coords_apo_frame: Vec<[f32; 3]>, // apo-frame CA, dense (NaN where absent)
}

struct ValidationSiteInput {
    site_id: String,
    centroid_apo_frame: [f32; 3],
    centroid_definition: String,    // "geometric_voxel_mass" | "spike_weighted" | ...
    lining_residue_pdb_resids: Vec<(String, i32)>, // (chain, resid)
    lining_atom_coords_apo_frame: Vec<[f32; 3]>,   // for min-atom-distance metric
}
```

**Why this file:** the offline validator currently has to:
- re-parse the topology to recover residue_map (already exists as `<topo>.residue_map.json`),
- re-extract CA coords from apo PDB,
- guess at the engine's residue convention.

By emitting `validation_inputs.json`, every downstream validator (DCC, contact-shell-overlap, known-contact-recovery, Kabsch-aligner) reads ONE file and never has to re-derive engine state. Holo PDB stays offline (network-fetched at validation time), so the engine never depends on RCSB.

### 5.3 SiteValidation struct (mission-mandated shape)

Defined under `crates/prism-nhs/src/site_manifest.rs` next to `KccMetrics` / `ThermDossier`:

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SiteValidation {
    pub site_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_pdb_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_ligand_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alignment_rmsd_A: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dcc_A: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_lining_atom_to_ligand_A: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contact_shell_overlap: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub known_contact_recovery: Option<f32>,
    pub validation_grade: String,  // "PASS" | "PARTIAL" | "FAIL" | "NOT_MATCH" | "HARD_NOT_MATCH" | "GT_INVALID" | "GT_MISSING"
}
```

**Population rule:** the runtime never populates this struct. The runtime only emits `validation_inputs.json`. `SiteValidation` is filled in by the offline validator and merged back into `binding_sites.json` (via a new `scripts/prism-merge-validation.py`) **after** validation runs, OR consumers read `validation.json` (a side file) and join by `site_id`.

The mission says "Rust-emitted SiteValidation" — that is fine in two interpretations:

- **(A) Strict:** the Rust runtime knows nothing about ground truth and emits `SiteValidation { validation_grade: "GT_MISSING", ... }` filled with `None`s, as a placeholder. An offline tool then **rewrites** the `validation` field in-place. **Recommended** because it preserves the pure-runtime / pure-offline split.

- **(B) Loose:** the Rust runtime calls a small FFI-free Kabsch helper (e.g. `crates/prism-nhs/src/validation/kabsch.rs`, pure Rust, depends only on `ndarray-linalg` or hand-rolled SVD) and reads the GT sidecar at end-of-run if `--ground-truth-sidecar` is passed. Then `SiteValidation` is populated by Rust before emit. Avoids Python in the validation loop but adds Rust dependency on PDB parsing / ground-truth schema. **Not recommended:** validation needs a holo PDB, which means RCSB fetch, which means networking in the runtime. Bad architecture.

**Decision:** option (A). Runtime emits `SiteValidation` shells with `validation_grade = "GT_MISSING"` and `None` everywhere. Offline `scripts/prism-validate-sites.py` (NEW, tier-A non-quarantine) consumes `validation_inputs.json` + `<prefix>_ground_truth.json` + cached holo PDB, fills the struct, writes `<prefix>.site_validation.json`. The result joins to `MaterializedSite` (Agent 6) by `site_id`.

### 5.4 Validation grade rules (lex order, NOT a composite)

Lifted from `site_vs_holo_strict.py:235-242` and `canonical_dcc_audit.py:89-100`, harmonized:

```
Required: dcc_A finite AND alignment_rmsd_A < 5.0
HARD_NOT_MATCH:  min(dcc_local, dcc_global) > 30 AND contact_shell_overlap < 0.10
NOT_MATCH:       dcc_global > 20 AND contact_shell_overlap < 0.10
PASS:            dcc_global < 4 AND contact_shell_overlap > 0.30 AND min_lining_atom_to_ligand_A < 4.0
PARTIAL:         dcc_global < 8 AND (contact_shell_overlap > 0.20 OR known_contact_recovery > 0.30)
FAIL:            otherwise (DCC > 8 OR overlap < 0.10)
GT_MISSING:      sidecar absent or marked invalid
GT_INVALID:      PanDDA / templated complex / fragment screen — sidecar valid_for_dcc_validation=false
```

These are **lexicographic** in: `validation_grade → dcc_A → contact_shell_overlap → known_contact_recovery → min_lining_atom_to_ligand_A`. No weighted sums.

### 5.5 Coherence metrics (separate block, never fed into validation_grade)

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SiteCoherence {
    /// |E_t - E_0| / |E_0| accumulated over the run window (closed-system check)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub energy_drift_relative: Option<f32>,
    /// ASC steering work integrated into d_potential_energy over the site's spike support
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asc_work_accounted_kJ_per_mol: Option<f32>,
    /// L4 audit outcome (mirror of audit[].outcome for the causal_truthing_audit transform)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kcc_causal_consistency: Option<String>,  // "Accepted" | "Quarantined"
    /// Mean |Δ_AB| / σ_noise across the captured WHILE-region for stream pair (A,B)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_pair_bisimulation_z: Option<f32>,
    /// Phase-fraction recurrence index: ratio of CCNS phase visits across hot/cold cycles
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase_recurrence_index: Option<f32>,
    /// G26 gear histogram (4 bins, gear 0..3) over the site's frame window
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub g26_gear_histogram: Option<[u32; 4]>,
    /// F1 branch switch count for this site (logged by log_f1_switch_events)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub f1_branch_switch_count: Option<u32>,
    /// CUDA WHILE conditional handle iteration count for the most recent call
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub while_iterations: Option<u32>,
    /// Mass-conservation residual from gpu_invariant::audit_mass_conservation
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mass_conservation_residual: Option<f32>,
}
```

Each field has a Rust-side or CUDA-side source already on disk:
- `energy_drift_relative`: `persistent_engine.rs:1448` `d_potential_energy_components_dev_ptr` — read at end-of-run, sum, divide by initial.
- `asc_work_accounted_kJ_per_mol`: `interferometric_adjudicator.rs:312` (folded into d_potential_energy directly).
- `kcc_causal_consistency`: copy from `audit[]` block where `transform == "causal_truthing_audit"` (`bin/nhs_rt_full.rs:15284`).
- `stream_pair_bisimulation_z`: `interferometric_adjudicator.rs:17,208` Δ_AB classifier — already in CUDA (`adjudicator.cu:903`).
- `phase_recurrence_index`: derive from `phase{}` block already emitted (`bin/nhs_rt_full.rs:15299-15340`) — ratio of revisits to first-visits.
- `g26_gear_histogram`: `gearbox.rs:38-90` `GearboxState.previous_gear` traced over frames (need a lightweight host-side append-only log).
- `f1_branch_switch_count`: `ghost_telemetry::log_f1_switch_events` already invoked at `nhs_rt_full.rs:6857`. Just sum per-site.
- `while_iterations`: read counter value from the `cuGraphConditionalHandle` memory model (`graph_capture.rs:394`, `interferometric_adjudicator.rs:1213-1264`).
- `mass_conservation_residual`: `gpu_invariant::audit_mass_conservation` already pass/trap test exists (`gpu_invariant.rs:93-160`).

**ALL coherence fields are Rust-emittable; ZERO require Python.**

### 5.6 Wiring order

Tier 1 (Rust-only, no offline dependency):
1. Add `SiteValidation` and `SiteCoherence` to `site_manifest.rs`.
2. Emit empty-shell `SiteValidation { grade: "GT_MISSING" }` per site.
3. Populate `SiteCoherence` per site from existing engine state.

Tier 2 (offline validator):
4. Write `crates/prism-nhs/src/validation_inputs.rs` + emit `<prefix>.validation_inputs.json` at end-of-run.
5. Write `scripts/prism-validate-sites.py` (NEW, NON-QUARANTINE) that:
   - reads `validation_inputs.json`,
   - reads `<prefix>_ground_truth.json` sidecar (existing),
   - fetches holo PDB via existing `prism-ground-truth.py` cache,
   - performs Kabsch (lift `do_alignment` from `site_vs_holo_strict.py`),
   - computes DCC, min-lining-atom, contact-shell-overlap, known-contact-recovery,
   - assigns `validation_grade` per §5.4 rules,
   - writes `<prefix>.site_validation.json`.
6. Wire into `scripts/prism-validate-and-run.sh` Phase 3.5 (new): calls `prism-validate-sites.py`, never fails the run.

Tier 3 (panel registry):
7. Write `data/panels/strict_v1.json` (move from `scripts/quarantine/strict_dcc_panel_v1.json`).
8. Add `data/panels/cryptobench_v1.json` (manifest of the 199 cryptobench targets).
9. Add `data/panels/true_apo_v1.json` (manifest of 462 true_apo targets).
10. Update `prism-validate-sites.py` to emit a per-panel summary `<panel>.evaluation_v2.json` aligned with the `evaluate_benchmark.py` shape.

### 5.7 Joining with MaterializedSite (Agent 6)

`MaterializedSite` is Agent 6's output — the consensus / clustering-event-derived site object. SiteValidation joins by `site_id` (string). Concrete plan:

- `MaterializedSite.site_id` is the stable hash/key (Agent 6's choice).
- Both `SiteValidation` and `SiteCoherence` carry the same `site_id`.
- The merger that produces the final `<prefix>.binding_sites.json` does a left-join on `site_id`. Sites without a validation entry get `validation: null`. Sites without coherence get `coherence: null`.
- Single source of truth: `MaterializedSite` is authoritative for spatial fields. `SiteValidation` carries ONLY the validation-grade + metric numbers, never re-emits centroid or lining.

---

## 6. Minimal touch list

| File | Change | Owner |
|---|---|---|
| `crates/prism-nhs/src/site_manifest.rs` | Add `SiteValidation`, `SiteCoherence`, `ApoFrameAnchor` types | Tier-1 Rust |
| `crates/prism-nhs/src/validation_inputs.rs` | NEW module, builds `ValidationInputs` from `clustered_sites + topology` | Tier-1 Rust |
| `crates/prism-nhs/src/lib.rs` | `pub mod validation_inputs;` | Tier-1 Rust |
| `crates/prism-nhs/src/bin/nhs_rt_full.rs` | At end-of-run (after `kcc_visualization.json` write, around `:9729`), call validation_inputs writer; populate per-site `coherence` block from existing audit/phase data | Tier-1 Rust |
| `scripts/prism-validate-sites.py` | NEW; consumes `validation_inputs.json` + sidecar + holo PDB | Tier-2 Python (offline) |
| `scripts/prism-validate-and-run.sh` | Insert Phase 3.5 between postflight and run-complete | Tier-2 shell |
| `data/panels/{strict_v1,cryptobench_v1,true_apo_v1}.json` | NEW panel manifests | Tier-3 data |
| `tests/test_validation_inputs.rs` (new) | Round-trip serialize/deserialize, schema stability | Tier-1 Rust tests |
| `tests/test_validate_sites/` (new) | Fixture-driven offline validator tests | Tier-2 Python tests |

**Files NOT to touch:** `bin/nhs_rt_full.rs` ranking logic, `xgb_ranker.rs`, `tokenized_ranker.rs`, `causal_truthing_audit.rs` (already handles its lane), `gating_stack.py`, anything under `transform/`. Those are owned by other lanes.

---

## 7. New structs / functions (signatures only)

### Rust
```rust
// site_manifest.rs (additions)
pub struct SiteValidation { /* §5.3 */ }
pub struct SiteCoherence  { /* §5.5 */ }

// validation_inputs.rs (new module)
pub struct ValidationInputs { /* §5.2 */ }
pub struct ApoFrameAnchor   { /* §5.2 */ }
pub struct ValidationSiteInput { /* §5.2 */ }

pub fn build_validation_inputs(
    sites: &[ClusteredBindingSite],
    topology: &Topology,
    residue_map: &ResidueMap,
    target_pdb_id: Option<&str>,
    target_chain: Option<&str>,
) -> Result<ValidationInputs>;

pub fn write_validation_inputs(
    inputs: &ValidationInputs,
    base_path: &std::path::Path,
) -> Result<()>;

// In bin/nhs_rt_full.rs (extension point near :9504, after KCC viz)
fn populate_site_coherence(
    site: &ClusteredBindingSite,
    audit_blocks: &[serde_json::Value],
    phase_block: &serde_json::Value,
    asc_work_kJ: Option<f32>,
    energy_drift: Option<f32>,
    g26_hist: Option<[u32; 4]>,
    f1_count: Option<u32>,
    while_iter: Option<u32>,
    mass_residual: Option<f32>,
) -> SiteCoherence;
```

### Python
```python
# scripts/prism-validate-sites.py (new)
def load_validation_inputs(path: Path) -> dict
def load_ground_truth(path: Path, target: str) -> dict
def fetch_holo_pdb(pdb_id: str) -> Path  # reuse from prism-ground-truth.py
def kabsch_global(apo_ca: dict, holo_ca: dict) -> tuple[np.ndarray, np.ndarray, float]
def kabsch_local(apo_ca, holo_ca, apo_centroid, radius=15.0) -> tuple[..., float]
def compute_site_validation(
    site: dict, apo_anchor: dict, holo_pdb: Path,
    ligand_resname: str, residue_convention: str,
) -> dict  # returns SiteValidation-shaped dict
def assign_grade(dcc_A, overlap, kcr, min_atom, alignment_rmsd) -> str
def emit_site_validation_file(out_path: Path, results: list[dict]) -> None
```

---

## 8. Acceptance tests

### Unit (Rust)
- `tests/test_validation_inputs.rs::round_trip_serde` — write + read == identity.
- `tests/test_validation_inputs.rs::empty_sites_emits_valid_skeleton` — zero sites → schema-valid empty file.
- `tests/test_site_validation.rs::grade_lexicographic` — (dcc=2, overlap=0.5, kcr=0.7, rmsd=1.0) → PASS; (dcc=25, overlap=0.05, kcr=0.0, rmsd=2.0) → NOT_MATCH; (dcc=35, overlap=0.05, ...) → HARD_NOT_MATCH.
- `tests/test_site_coherence.rs::all_none_serializes_to_empty_object` — confirms 70% I/O bloat reduction holds.

### Integration (offline)
- `tests/test_validate_sites/test_kabsch_self_reference.py` — apo == holo, RMSD ≈ 0, DCC < 0.5 Å.
- `tests/test_validate_sites/test_known_pair_1bzj.py` — uses `benchmarks/cryptobench/ground_truth/1bzj.ground_truth.json` + a fixture engine output, expects DCC < 8 Å OR contact_shell_overlap > 0.3 (PARTIAL or PASS), `validation_grade != "FAIL"`.
- `tests/test_validate_sites/test_pandda_filtered.py` — sidecar `valid_for_dcc_validation=false` → grade `GT_INVALID`, never crashes.
- `tests/test_validate_sites/test_no_sidecar.py` — no GT file → grade `GT_MISSING`, never crashes.
- `tests/test_validate_sites/test_known_contact_recovery.py` — given a `holo_pocket_selection` from `1bzj.ground_truth.json` and a site whose `lining_residues` overlap by 60%, `known_contact_recovery == 0.6 ± 0.01`.

### End-to-end (single target, smoke)
- `tests/test_validation_smoke.sh` — run engine on `data/targets/4lpk.json` (smallest target with GT), assert files exist: `4lpk.binding_sites.json`, `4lpk.kcc_visualization.json`, `4lpk.validation_inputs.json`, `4lpk.site_validation.json`. Schema-validate each.

### Panel-level (offline)
- Run `prism-validate-sites.py` on a fixture run from each of {strict_v1, cryptobench_v1, true_apo_v1} panels, assert per-panel summary has SR@1, SR@3, SR@5, SR@10 fields and matches `evaluate_benchmark.py:725-734` schema bit-for-bit.

---

## 9. Failure modes

| Mode | Mitigation |
|---|---|
| Holo PDB unavailable on RCSB | Sidecar marked `valid_for_dcc_validation=false`; grade = `GT_INVALID` (existing behavior in `prism-ground-truth.py:325-328`) |
| Sequence identity < 90% (wrong protein, e.g. 1BTL→1BTM trap) | New gate in `prism-validate-sites.py`: if seq_identity(apo, holo) < 0.90, write `validation_grade = "GT_INVALID"` with reason `wrong_protein_identity_seq_id_X.XX`; explicitly cited in `BENCHMARK_ALIGNMENT_GUIDE.md:18-29` |
| Kabsch fails (post-RMSD > 5 Å despite seq_id > 0.90) | Set `alignment_rmsd_A` to the failed value, set `dcc_A=None`, grade = `GT_INVALID` with reason `kabsch_failed_residue_numbering_mismatch` |
| Engine emits zero sites | `validation_inputs.json` has empty `sites` array; offline validator writes empty `site_validation.json`; postflight already prints `ENGINE_FOUND_NO_SITES` |
| L4 audit fires on ALL sites | The `audit[]` block records this per-site (already exists). Coherence block reflects `kcc_causal_consistency = "Quarantined"` — this is internal coherence, NEVER flows into accuracy grade |
| Multichain residue mapping | `validation_inputs.json` carries `chain_ids: Vec<String>`. Validator joins `(chain, pdb_resid)` not raw `resid`. `prism-merge-chains.py` chain map already supports this (existing) |
| Stale sidecar from previous engine run | Sidecar file modtime older than `binding_sites.json` → validator regenerates; or fail-safe: include `topology_hash` in both files and require match |
| Network-fetch in offline tests | Cache in `~/.cache/prism4d/holo_pdbs/` already (from `prism-ground-truth.py:54`). CI must pre-populate this cache or skip network-dependent tests |
| Engine writes invalid JSON (NaN serialization) | Rust `serde_json` errors out on NaN by default; SiteCoherence uses `Option<f32>` so non-finite values become `None`. Add a unit test confirming `f32::NAN → None` mapping |
| validation_inputs.json schema drift | `schema_version: "validation_inputs_v1"` field; offline validator hard-asserts on read |
| **Validation circularity (engine grades its own sites)** | Hard architectural separation: `SiteValidation` is populated ONLY by `prism-validate-sites.py`, which reads ONLY external GT (PDB + sidecar). Engine cannot see its own validation_grade. `SiteCoherence` (engine-internal) cannot influence `validation_grade`. Code review gate: any PR that imports `SiteValidation` outside the validator script gets blocked |
| Composite-score creep | Add a CI lint: grep for "weight" / "composite" / "DPS" in `prism-validate-sites.py` and `site_manifest.rs::SiteValidation` impl block. Per `CLAUDE.md` Immutable Rule 1 |
| Oracle vocabulary in reports | CI lint: `grep -i "oracle" docs/validation/ scripts/prism-validate-sites.py` returns 0 matches per `CLAUDE.md` Rule 8 |

---

## 10. Rollback

The full plan is **strictly additive**. Rollback is per-tier:

- **Tier 1 rollback:** revert `site_manifest.rs` and `validation_inputs.rs`. Engine stops emitting `validation_inputs.json` and per-site `validation`/`coherence` blocks. All existing emit paths unchanged. Offline pipelines (`prism_canonical.py`, `prism_replicate.py`) ignore unknown fields by default — no consumer breakage.

- **Tier 2 rollback:** delete `scripts/prism-validate-sites.py` and the Phase 3.5 hook in `prism-validate-and-run.sh`. The pre-existing Phase 3 postflight DCC stays as the (still-imperfect) check.

- **Tier 3 rollback:** delete `data/panels/*.json`. Strict panel reverts to the quarantined `scripts/quarantine/strict_dcc_panel_v1.json` exactly as-is.

- **Schema versioning:** every new JSON carries `schema_version: "validation_inputs_v1"` / `"site_validation_v1"`. Future versions live alongside; no in-place breaks.

- **No engine semantics touched:** ranking, gating, clustering, scoring all unchanged. Only new emit paths added. CI gate: `cargo build` + golden-output diff must show byte-identical `binding_sites.json` for sites' existing fields after the patch (only new fields appended).

---

## Targets-with-GT inventory (for the §B canonical relaunch)

Already wired (engine-runnable + GT present):
- `data/targets/4lpk.json` — 4lpk (apo) — corresponds to 1pzo holo per `validate_apo_holo.py:18`
- `data/targets/1nkp_myc_dna.json` — 1NKP MYC-MAX bHLH-LZ; non-druggable / negative control per `CLAUDE.md` CYS registry
- `data/targets/mpro_monomer.json` — SARS-CoV-2 main protease
- `data/targets/7c8r_dimer_hmr.json` — has chain map, multichain
- `data/targets/tier3/{1mq4,1qs4,1yes,1bg1}.topology.json`

Has GT, awaiting topology prep (high-priority panel additions):
- `benchmarks/cryptobench/ground_truth/*.ground_truth.json` — 199 targets, full holo+ligand schema with `apo_pocket_selection` (perfect for `known_contact_recovery`).
- `benchmarks/true_apo/ground_truth/*.ground_truth.json` — 462 targets (283 cryptic + 179 standard); ligand_centroid is in HOLO frame and needs Kabsch to apo frame.
- `benchmarks/hard_targets/3mh1_ground_truth.json` — 3mh1 (p38 MAPK vs 3hec STI), already has alignment RMSD.
- `scripts/quarantine/twin10_targets.json` — TWIN-10 cohort (kras_g12d, wrn, menin, smarca2, pkmyt1 + 5 more); the most thoroughly verified set.
- `scripts/quarantine/strict_dcc_panel_v1.json` — frozen 18-target panel.

Has structures, NO GT yet (negative controls):
- `benchmarks/never_bound/pdbs/*.pdb` — 97 targets explicitly never crystallized with a ligand. These are the **only** targets whose `validation_grade` should always be `GT_INVALID` by design (no holo exists).

---

## Bottom line

- **Existing work covers 80% of the metric definitions** — `evaluate_benchmark.py`, `canonical_dcc_audit.py`, `site_vs_holo_strict.py` between them implement DCC, DCA, DVO, contact-shell overlap, dual-pass Kabsch, verdict thresholds, and failure classification. The new code is mostly *plumbing* to make these computable from a single Rust-emitted file.
- **Critical bug exists in current postflight** — DCC computed without Kabsch; numbers are unreliable for any apo/holo pair with raw CA RMSD > 3 Å. Fixed by Tier 2.
- **Two-axis architecture is non-negotiable.** `SiteValidation` (external accuracy, offline-populated) and `SiteCoherence` (engine internals, Rust-populated) MUST stay disjoint; collapsing them creates the circularity the mission flags.
- **Three benchmark schemas competing** (cryptobench rich, true_apo simple, hard_targets minimal). The `validation_inputs.json` shape decouples the engine from this; a single panel-registry layer (Tier 3) reconciles them.
- **Coherence block is 100% Rust-emittable today** — every field maps to existing engine state (`d_potential_energy`, `audit[]`, `phase{}`, `gearbox.rs`, `interferometric_adjudicator.rs`, `gpu_invariant.rs`). No new physics.
