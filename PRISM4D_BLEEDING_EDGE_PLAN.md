# PRISM-4D — BLEEDING EDGE EXECUTION PLAN
# =========================================
#
# 4 PHASES. Each phase has hard gates. Do NOT skip phases.
# Each phase produces artifacts the next phase depends on.
#
# PHASE 1: Spike Entropy Ranker (pure analysis, no engine changes)
# PHASE 2: Pharma Targets + GT Validation (benchmark expansion)
# PHASE 3: LADD Gate 2-7 (K=10 neuromorphic, CO-FIRE, departure fingerprinting)
# PHASE 4: Model B Retrain + Structure-Only Predictor (ML)
#
# CURRENT STATE:
#   18 targets benchmarked, 36 CRYPTIC (94% genuine multi-channel)
#   Gate 1 LADD merged (density infrastructure live, neurons not)
#   Ranking broken: mean CRYPTIC rank = 19
#   152M spikes on disk with full channel composition
#
# RULES:
#   1. One phase at a time
#   2. Show results before proceeding
#   3. No engine source changes in Phases 1-2
#   4. All output to /mnt/storage/prism-outputs/


# ═══════════════════════════════════════════════════════════════════════
#
#                    PHASE 1: SPIKE ENTROPY RANKER
#
# ═══════════════════════════════════════════════════════════════════════
#
# Fix the ranking problem using existing data. No engine changes.
# The 152M spikes already contain the signal — extract it.


# ── 1A: INSTALL STREAMING DEPENDENCIES ──
#
#   pip install ijson polars pyarrow --break-system-packages


# ── 1B: BUILD PER-SITE SPIKE FEATURE EXTRACTOR ──
#
#   For EACH site on EACH of the 18 targets, compute these features
#   by streaming the spike_events.json with ijson (never load full file):
#
#   CHANNEL FEATURES:
#     source_entropy:      Shannon entropy of spike_source distribution
#                          H = -Σ p(s) log2 p(s) for s in {UV, LIF, EFP}
#                          Range: 0.0 (single source) to 1.585 (equal 3-way)
#
#     intensity_entropy:   Shannon entropy of intensity histogram (100 bins)
#                          Range: 0.0 (constant value) to ~6.6 (uniform)
#
#     n_unique_sources:    Count of distinct spike_source values (1, 2, or 3)
#
#     n_unique_intensities: Count of distinct intensity values (rounded to 0.01)
#
#     efp_fraction:        Fraction of spikes with source=EFP
#                          EFP presence is strongest multi-channel indicator
#
#     lif_fraction:        Fraction of spikes with source=LIF
#
#   PHASE FEATURES:
#     phase_coverage:      Fraction of 5 phases with >1% of spikes
#                          Range: 0.2 (single phase) to 1.0 (all 5)
#
#     cold_hot_ratio:      cold_hold_spikes / warm_hold_spikes
#                          High ratio = preferentially fires cold (UV echo pattern)
#                          Low ratio = fires warm too (real dynamics)
#
#     phase_entropy:       Shannon entropy of phase distribution
#                          Range: 0.0 (single phase) to 2.322 (uniform 5-way)
#
#     heating_cooling_ratio: heating_spikes / cooling_spikes
#                          Captures hysteresis directionality
#
#   TEMPORAL FEATURES (from spike timestamps):
#     burst_index:         Fraction of spikes in the top 10% densest timestep bins
#                          High = bursty (real conformational event)
#                          Low = uniform (background noise)
#
#     inter_spike_cv:      Coefficient of variation of inter-spike intervals
#                          High CV = irregular (real dynamics)
#                          Low CV = regular (oscillator artifact)
#
#   EXISTING FEATURES (from binding_sites.json, no spike parsing needed):
#     hysteresis_asymmetry
#     ccns_tau
#     quality_score
#     spike_count
#     estimated_volume
#     druggability_score
#
#   Implementation:
#     - Stream each spike file with ijson, sample up to 50,000 spikes
#       (first 10K + last 10K + 30K uniformly sampled from middle)
#     - Compute all channel/phase/temporal features
#     - Merge with binding_sites.json features by site_id
#     - Output: per-site feature vector (20+ features)
#
#   Save to:
#     /mnt/storage/prism-outputs/runs/v1.1-physics/site_features.csv
#     /mnt/storage/prism-outputs/runs/v1.1-physics/site_features.json
#
#   The CSV should have columns:
#     target, site_id, rank_original, therm_class,
#     source_entropy, intensity_entropy, n_unique_sources,
#     n_unique_intensities, efp_fraction, lif_fraction,
#     phase_coverage, cold_hot_ratio, phase_entropy,
#     heating_cooling_ratio, burst_index, inter_spike_cv,
#     hysteresis_asymmetry, ccns_tau, quality_score,
#     spike_count, estimated_volume, druggability_score


# ── 1C: COMPOSITE ENTROPY RANKING ──
#
#   Rank sites by a composite score that captures spike information richness.
#
#   DO NOT hardcode weights. Instead, derive them from the data:
#
#   Method 1 — Supervised (if >10 CRYPTIC sites with GT labels):
#     Use the therm_class=CRYPTIC label as positive class.
#     Train a logistic regression on the new features.
#     The learned coefficients ARE the optimal weights.
#     Use LOTO CV (leave-one-target-out) to avoid overfitting.
#
#   Method 2 — Unsupervised (no GT needed):
#     Compute the Fisher discriminant ratio for each feature:
#       F = (μ_cryptic - μ_non_cryptic)² / (σ²_cryptic + σ²_non_cryptic)
#     Weight each feature by its Fisher ratio.
#     This maximizes separation between CRYPTIC and non-CRYPTIC sites
#     without training — pure statistical separation.
#
#   Method 3 — Information-theoretic:
#     Mutual information I(feature; therm_class) for each feature.
#     Weight by MI. This captures nonlinear relationships.
#
#   Implement ALL THREE methods. Compare re-ranking results.
#   Report which method produces the best CRYPTIC promotion:
#     Mean rank improvement = old_mean_rank - new_mean_rank
#     SR@1 = fraction of targets where CRYPTIC is rank 1
#     SR@3 = fraction of targets where CRYPTIC is in top 3
#     SR@5 = fraction of targets where CRYPTIC is in top 5


# ── 1D: RE-RANKING REPORT ──
#
#   For EACH of the 18 targets, show:
#     Target | CRYPTIC site | Old Rank | New Rank (Method 1) | New Rank (Method 2) | New Rank (Method 3)
#
#   Summary statistics:
#     | Method | Mean CRYPTIC Rank | SR@1 | SR@3 | SR@5 | AUROC |
#
#   Save to:
#     /mnt/storage/prism-outputs/runs/v1.1-physics/ENTROPY_RANKING.md
#     /mnt/storage/prism-outputs/runs/v1.1-physics/ENTROPY_RANKING.json
#
#   Also save the site_features.csv — this is the training data for Phase 4.


# ── 1E: FEATURE IMPORTANCE ANALYSIS ──
#
#   For the best-performing method, report:
#     - Top 5 features by weight/importance
#     - Feature correlation matrix (which features are redundant?)
#     - Per-feature box plots: CRYPTIC vs non-CRYPTIC distributions
#     - Save plots as PNG files
#
#   Save to:
#     /mnt/storage/prism-outputs/runs/v1.1-physics/feature_analysis/


# ═══════════════════════════════════════════════════════════════════
# PHASE 1 GATE: Do NOT proceed until:
#   1. site_features.csv exists with all 644 sites × 20+ features
#   2. ENTROPY_RANKING.md shows CRYPTIC rank improvement
#   3. User reviews and approves
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
#
#                    PHASE 2: PHARMA TARGETS + GT VALIDATION
#
# ═══════════════════════════════════════════════════════════════════════
#
# Expand benchmark to 33 targets. Validate against experimental ground truth.


# ── 2A: DOWNLOAD + PREP + RUN 15 PHARMA TARGETS ──
#
#   Use the pharma target list from run_v1.1_benchmark.sh.
#   For each: download apo PDB, download holo PDB (GT), prep topology,
#   validate prep, run engine with canonical flags (20-stream + multi-temp).
#
#   The pharma targets are:
#     kras_g12c (4OBE/4LYJ), il2 (1M47/1M48), tem1 (1JWP/1PZO),
#     p38 (2NPQ/2ZB1), bclxl (1MAZ/2YXJ), ptp1b (2HNP/1T49),
#     kif11 (3HQD/1Q0B), mcl1 (1WSX/6QFI), parp1 (7AAD/7AAB),
#     btk (3GEN/3PJ3), abl1 (2FO0/5MO4), idh1 (4KZO/4UMX),
#     shp2 (2SHP/6MDB), keap1 (1U6D/4IQK), sirt6 (3K35/5X16)
#
#   Run with: --multi-stream 20 --multi-temp [plus standard COMMON flags]
#   Output to: /mnt/storage/prism-outputs/runs/v1.1-physics/<target_name>/


# ── 2B: GROUND TRUTH VALIDATION ──
#
#   For EACH target (all 33), compute distance-to-closest-known-site (DCC):
#
#   1. Parse the holo PDB structure
#   2. Identify all ligand atoms (HETATM, not water/buffer)
#   3. Compute ligand centroid = known binding site location
#   4. For each PRISM-4D detected site, compute distance to ligand centroid
#   5. DCC = minimum distance across all detected sites
#   6. Success = DCC < 5.0 Å
#
#   For the 18 original targets, download holo structures if not already present:
#     1mq4 holo: 1MQ6 (or look up in CryptoBench/CryptoSite dataset)
#     Use CryptoBench apo-holo pairs where available.
#
#   For pharma targets, holo PDBs are downloaded in Step 2A.
#
#   Report:
#     | Target | Apo PDB | Holo PDB | Known Site Centroid | Best PRISM Site | DCC (Å) | Success |
#
#   Summary:
#     SR@1(5Å): fraction where rank-1 site is within 5Å of known site
#     SR@3(5Å): fraction where any top-3 site is within 5Å
#     SR@5(5Å): fraction where any top-5 site is within 5Å
#     Overall DCC success rate: fraction with ANY site within 5Å
#
#   CRITICALLY: compute SR@N with BOTH the original quality_score ranking
#   AND the new entropy ranking from Phase 1. Show the improvement.


# ── 2C: EXTRACT SPIKE FEATURES FOR PHARMA TARGETS ──
#
#   Run the same feature extractor from Phase 1B on all 15 pharma targets.
#   Append to site_features.csv.
#   Re-compute entropy ranking on the expanded 33-target dataset.


# ── 2D: PHARMA BENCHMARK REPORT ──
#
#   /mnt/storage/prism-outputs/runs/v1.1-physics/PHARMA_REPORT.md
#
#   Per-target results (same format as BENCHMARK_REPORT.md).
#   GT validation table.
#   Multi-channel audit of all pharma CRYPTIC sites.
#   Entropy ranking vs quality_score ranking comparison on GT.


# ═══════════════════════════════════════════════════════════════════
# PHASE 2 GATE: Do NOT proceed until:
#   1. All 15 pharma targets complete
#   2. GT validation computed for all 33 targets
#   3. Entropy ranking SR@N computed with and without GT
#   4. User reviews and approves
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
#
#          PHASE 3: LADD GATE 2-7 (NEUROMORPHIC 4TH CHANNEL)
#
# ═══════════════════════════════════════════════════════════════════════
#
# The completed Phase 1+2 benchmark is the regression reference.
# LADD development happens on a branch. Every change validated
# against the 33-target baseline.


# ── 3A: BRANCH FROM BENCHMARKED COMMIT ──
#
#   BENCHMARK_COMMIT=$(git rev-parse HEAD)
#   git checkout -b ladd-gate2-$(date +%Y%m%d)


# ── 3B: THREAD MAPPING DECISION ──
#
#   Read /mnt/storage/prism-outputs/ladd-dev/ARCHITECTURE_NEURON.md
#
#   Run occupancy analysis for BOTH options on SM120 (RTX 5080):
#     Option A: K=10 inline (160 threads/block)
#     Option SEPARATE: K=8 frozen + K=2 LADD observation pass
#
#   For Option A, compile with K=10 and check:
#     nvcc --ptx --resource-usage nhs_amber_fused.cu
#     Compare register count, shared memory, occupancy vs K=8 baseline
#
#   For Option SEPARATE:
#     Estimate overhead of second kernel launch per step (55K launches)
#     vs inline overhead of 2 extra oscillator iterations per voxel per step
#
#   Present BOTH analyses with numbers. Wait for user decision.


# ── 3C: IMPLEMENT CHOSEN OPTION ──
#
#   Follow PRISM4D_LADD_NEURON.md:
#     - K_NEURONS extension (or separate kernel)
#     - Custom LADD timescales: tau(8)=2.0, tau(9)=32.0
#     - Input routing: if (k < 8) water_signal else ladd_input
#     - Ballot mask: lif_fired (bits 0-7), ladd_fired (bits 8-9)
#     - Source assignment: source=4 (LADD), source=5 (CO-FIRE)
#     - Forward cascade: k=7 → k=8 → k=9
#     - Shared memory LUT extension
#
#   Show every diff. Wait for approval.


# ── 3D: BACKWARD COMPAT (ladd_enabled=0) ──
#
#   Run 1MQ4, 2OV5, 1Z1M WITHOUT --ladd.
#   Compare against Phase 2 benchmark results.
#   PASS: ±3 sites, ±0.10 top asymmetry, zero CRYPTIC lost.
#   FAIL → switch to Option SEPARATE and retry.


# ── 3E: LADD FORWARD VALIDATION ──
#
#   Run 1MQ4, 4MNE, 1BG1 WITH --ladd.
#   For each target, audit:
#     - LADD spike count (source=4)
#     - CO-FIRE spike count (source=5)
#     - Phase distribution of LADD/CO-FIRE spikes
#     - Spatial clustering vs known CRYPTIC centroids
#     - UV-echo site suppression (LADD should NOT fire there)
#     - Source distribution across all sites


# ── 3F: DEPARTURE FINGERPRINTING (Gate 3 from LADD_OPTION_B.md) ──
#
#   Implement:
#     - Global departure buffer (768 KB)
#     - LaddMetadata struct (84 bytes, parallel to SpikeEvent)
#     - 8-category atom classification
#     - Displacement coherence computation
#     - Adjacent depletion count
#
#   Validate:
#     - Fingerprints decode correctly on Rust side
#     - CRYPTIC sites show coherence > 0.5
#     - UV-echo sites show zero departures
#     - Atom categories are biologically plausible


# ── 3G: RUST INTEGRATION + JSON OUTPUT ──
#
#   --ladd CLI flag
#   GPU array allocation (conditional)
#   Kernel parameter passing
#   Per-channel filtering for source=4,5
#   JSON output with full LaddMetadata


# ── 3H: PERFORMANCE VALIDATION ──
#
#   3 runs with/without --ladd on 1MQ4.
#   Target: <10% wall time overhead.
#   Memory: document additional VRAM.
#   8-stream stability test.


# ── 3I: 5-TARGET LADD VALIDATION ──
#
#   Run 5 targets with --ladd: 1MQ4, 4MNE, 1BG1, 2P4E, 1UNL
#   (all have strong multi-channel CRYPTIC sites from Phase 1-2)
#
#   For each, produce:
#     - 5-source distribution (UV/LIF/EFP/LADD/CO-FIRE)
#     - CO-FIRE spatial correlation with known CRYPTIC sites
#     - Departure fingerprint at CRYPTIC sites
#     - Displacement coherence distribution
#     - Temporal cascade: LADD→LIF→EFP ordering analysis
#
#   KEY QUESTION: Does CO-FIRE (source=5) correlate with CRYPTIC
#   classification better than multi-channel alone?
#   Compute: what fraction of CO-FIRE sites are CRYPTIC?
#   If >80%, CO-FIRE is a stronger signal than ThermClass.


# ── 3J: COMMIT ON BRANCH ──
#
#   Tag: v1.2-ladd-experimental
#   Do NOT merge to main without user approval + full re-benchmark.


# ═══════════════════════════════════════════════════════════════════
# PHASE 3 GATE: Do NOT proceed until:
#   1. LADD backward compat passes on 3 targets
#   2. LADD forward test shows LADD+CO-FIRE spikes at CRYPTIC sites
#   3. Departure fingerprints decode and are biologically plausible
#   4. Performance overhead < 10%
#   5. 5-target validation report complete
#   6. User reviews and approves
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
#
#        PHASE 4: MODEL B RETRAIN + STRUCTURE-ONLY PREDICTOR
#
# ═══════════════════════════════════════════════════════════════════════
#
# Two ML models. Model B is the improved ranker. The structure-only
# predictor is the platform's ultimate form — no MD at inference time.


# ── 4A: MODEL B v2 — RETRAIN WITH ENTROPY + LADD FEATURES ──
#
#   Training data: site_features.csv from Phase 1 (644+ sites × 20+ features)
#   If LADD Phase 3 complete: add LADD features (co-fire fraction,
#   displacement coherence, departure composition) — up to 30+ features
#
#   Labels: therm_class (CRYPTIC vs non-CRYPTIC, binary)
#   CV: Leave-One-Target-Out (LOTO) — train on 17 targets, test on 1
#
#   Models to train:
#     - Logistic regression (baseline, interpretable)
#     - Random forest (captures nonlinear feature interactions)
#     - Gradient boosted trees (XGBoost/LightGBM — SOTA tabular)
#     - SVM with RBF kernel (for comparison with CryptoSite)
#
#   For each model, report:
#     - LOTO AUROC
#     - LOTO AUPRC (class-imbalanced — CRYPTIC is minority)
#     - Feature importance / SHAP values
#     - Calibration curve
#
#   Compare against:
#     - Model B v1 (struct-only, AUROC 0.850 on buggy physics)
#     - CryptoSite (reported AUROC 0.83)
#     - PocketMiner (reported AUROC 0.87)
#
#   Save:
#     /mnt/storage/prism-outputs/ml/model_b_v2/
#       models/ (serialized)
#       results/ (LOTO predictions per target)
#       MODELB_V2_REPORT.md
#       MODELB_V2_REPORT.json


# ── 4B: STRUCTURE-ONLY PREDICTOR (PRISM-AI Foundation) ──
#
#   This is the long-term strategic model. It predicts cryptic sites
#   from a SINGLE apo PDB structure without running the PRISM-4D engine.
#   The engine becomes the training oracle, not the runtime dependency.
#
#   TRAINING DATA CONSTRUCTION:
#
#   For each of the 33+ targets:
#     Input: apo PDB structure → per-residue structural features
#     Labels: per-residue KCC scores + lining_residue membership +
#             therm_class of the site each residue belongs to +
#             LADD departure participation (if available from Phase 3)
#
#   PER-RESIDUE STRUCTURAL FEATURES (computable from PDB alone):
#     - Secondary structure (DSSP: H/E/C)
#     - Solvent accessible surface area (SASA)
#     - Residue depth (distance to nearest surface point)
#     - B-factor (crystallographic flexibility)
#     - Contact number (within 8Å)
#     - Packing density (atoms within 5Å)
#     - Hydrophobicity (Kyte-Doolittle)
#     - Conservation (from MSA if available, else skip)
#     - Local secondary structure content (helix/sheet/coil in ±5 residues)
#     - Backbone dihedral angles (phi, psi, omega)
#     - Side chain volume
#     - Charge state at pH 7
#     - Aromatic ring presence
#     - Disulfide participation
#     - Distance to nearest aromatic (UV channel relevance)
#     - Distance to nearest charged residue (EFP channel relevance)
#     - Local backbone flexibility (dihedral angle variance in neighbors)
#
#   PER-RESIDUE LABELS FROM PRISM-4D:
#     - KCC score (continuous, 0 to ~0.65)
#     - Binary: is this residue in a CRYPTIC site's lining?
#     - Transfer entropy from PRISM-Therm
#     - Causal ΔG from PRISM-Therm
#     - Fisher information from PRISM-Therm
#     - LADD departure count at this residue (if Phase 3 complete)
#     - LADD displacement coherence at this residue
#
#   MODEL ARCHITECTURE OPTIONS:
#     - Per-residue features → GBM classifier → cryptic probability
#       (simplest, strong baseline, interpretable)
#     - Graph neural network on protein contact graph
#       (captures spatial relationships between residues)
#     - Protein language model fine-tune (ESM-2 embeddings + linear head)
#       (captures sequence-structure relationships, SOTA if enough data)
#
#   START WITH GBM. It's the most robust with limited training data
#   (33 targets × ~300 residues = ~10K training examples).
#   GNN and PLM are stretch goals if GBM proves the concept.
#
#   CV: LOTO (leave-one-target-out)
#   Metric: per-residue AUROC for cryptic lining prediction
#   Also compute: per-target SR@1(5Å) using predicted residue scores
#   to rank candidate sites
#
#   Save:
#     /mnt/storage/prism-outputs/ml/struct_predictor/
#       features/ (per-target feature matrices)
#       models/ (serialized GBM)
#       predictions/ (per-target residue-level predictions)
#       STRUCT_PREDICTOR_REPORT.md


# ── 4C: ENSEMBLE PDB FEATURE EXTRACTION ──
#
#   The 440 ensemble PDB conformations per target (55 models × 8 streams)
#   contain conformational diversity that a single apo structure cannot.
#
#   Extract per-residue DYNAMIC features from the ensemble:
#     - RMSF across 440 conformations
#     - Contact frequency matrix (which contacts form/break)
#     - SASA variance
#     - Secondary structure transition frequency
#     - Backbone dihedral angle variance
#     - Per-residue conformational entropy
#
#   These features bridge static structure and dynamics.
#   Add them to the structure-only predictor as "optional dynamic features."
#   At inference: if the user provides only 1 PDB, use static features.
#   If they run PRISM-4D and provide ensemble PDBs, use static + dynamic.
#
#   This creates a TIERED prediction system:
#     Tier 1: Single PDB → static features → fast prediction (~seconds)
#     Tier 2: Single PDB + PRISM-4D ensemble → static + dynamic → better prediction
#     Tier 3: Full PRISM-4D run → all spike/KCC/LADD features → best prediction


# ═══════════════════════════════════════════════════════════════════
# PHASE 4 GATE: Do NOT proceed until:
#   1. Model B v2 LOTO AUROC computed and compared to v1 + competitors
#   2. Structure-only predictor GBM trained and LOTO evaluated
#   3. Feature importance analysis complete
#   4. All reports saved
#   5. User reviews
# ═══════════════════════════════════════════════════════════════════
